# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 adapter for the NIXL connector and the frontend that names its requests.

Hooks sit at control points that already run once per request or transfer and
hand scalars to the native recorder, whose first-seen identities go to storage
reserved at startup. A hook may build a few small transient integers (a length, a
sum, a unit conversion) that are freed on return; it retains nothing, grows
nothing with the request count, and never formats, performs I/O or blocks. A
hook error disables the profiler and is counted; a full or closed ring is loss,
not an error.
"""

import os
import threading
import weakref

from .compute import TimelineMixin, timeline_context
from .config import NetworkProfilerConfig, profiler_config
from .core import open_recorder
from .peers import PeerClocks
from .schema import EventKind, Failure, IdSource, Subject, Transport

# Plain ints: the recorder accepts any non-bool integer and a plain int is the
# cheapest object to pass.
_REQUEST_BIND = int(EventKind.REQUEST_BIND)
_HANDOFF = int(EventKind.HANDOFF)
_HANDSHAKE_BEGIN = int(EventKind.HANDSHAKE_BEGIN)
_HANDSHAKE_END = int(EventKind.HANDSHAKE_END)
_TRANSFER_BIND = int(EventKind.TRANSFER_BIND)
_POST_RETURN = int(EventKind.POST_RETURN)
_COMPLETE_OBSERVED = int(EventKind.COMPLETE_OBSERVED)
_NOTIFY_SEND_RETURN = int(EventKind.NOTIFY_SEND_RETURN)
_NOTIFY_OBSERVED = int(EventKind.NOTIFY_OBSERVED)
_SENDERS_DONE = int(EventKind.SENDERS_DONE)
_RELEASE_REQUESTED = int(EventKind.RELEASE_REQUESTED)
_FAILED = int(EventKind.FAILED)
_TRANSFER_RETIRE = int(EventKind.TRANSFER_RETIRE)

_MASK64 = (1 << 64) - 1
_OFFSET_SAMPLE = int(Subject.OFFSET_SAMPLE)
_WORKER = int(Subject.WORKER)
_SCHEDULER = int(Subject.SCHEDULER)
_REQUEST_SCOPE = int(Subject.REQUEST)
_TRANSFER_SCOPE = int(Subject.TRANSFER)
_TABLE_FULL = int(Failure.CLOCK_PEER_TABLE_FULL)
_HEADER = int(IdSource.HEADER)
_BODY = int(IdSource.BODY)
_GENERATED = int(IdSource.GENERATED)
_HEADER_TO_BASE = int(Subject.HEADER_TO_BASE)
_BASE_TO_PARENT = int(Subject.BASE_TO_PARENT)
_PARENT_TO_ITEM = int(Subject.PARENT_TO_ITEM)
_EXTERNAL_TO_ENGINE = int(Subject.EXTERNAL_TO_ENGINE)
_CHILD_TO_PARENT = int(Subject.CHILD_TO_PARENT_ENGINE)

# The connector's own failure_type constants, mapped without parsing messages.
FAILURE_CODES = {
    "handshake_failed": int(Failure.HANDSHAKE_FAILED),
    "handshake_setup_failed": int(Failure.HANDSHAKE_SETUP_FAILED),
    "transfer_setup_failed": int(Failure.TRANSFER_SETUP_FAILED),
    "transfer_failed": int(Failure.TRANSFER_FAILED),
    "transfer_exception": int(Failure.TRANSFER_EXCEPTION),
    "transfer_release_failed": int(Failure.TRANSFER_RELEASE_FAILED),
    "notification_failed": int(Failure.NOTIFICATION_FAILED),
}
_UNKNOWN_FAILURE = int(Failure.UNKNOWN)
_LEASE_EXPIRED = int(Failure.LEASE_EXPIRED)
_LATE_NOTIFICATION = int(Failure.LATE_NOTIFICATION)

_live: "weakref.WeakSet[_Profiler]" = weakref.WeakSet()


def _disable_inherited() -> None:
    """A forked child must not close or record into its parent's ring."""
    for profiler in list(_live):
        profiler._detach()


os.register_at_fork(after_in_child=_disable_inherited)


def _discard(*args):
    return None


class _Profiler:
    """Recorder ownership, error accounting and the shutdown order."""

    def __init__(self, config: NetworkProfilerConfig, recorder):
        self.config = config
        self.recorder = recorder
        self.output_dir = recorder.output_dir
        self._record = recorder.record
        # derived per backend in the context, never resolved at record time
        self._transport = int(Transport.UNKNOWN)
        self.hook_errors = 0
        self.first_error: str | None = None
        self.shutdown_failures: list[str] = []
        self.closed = False
        self.result = None
        self._detached = False
        self._lock = threading.Lock()
        _live.add(self)

    def _error(self, exc: BaseException) -> None:
        """Bounded accounting and disabling only: no logging, I/O or locks.

        Logging takes a handler lock and can block or raise, so the diagnostic is
        kept for ``close``, which stores it in ``profiler.json`` and
        ``shutdown_failures`` without calling any log handler.
        """
        self.hook_errors += 1
        self._record = _discard
        if self.first_error is None:
            self.first_error = type(exc).__name__

    def _detach(self) -> None:
        self._detached = True
        self.closed = True
        self._record = _discard
        self._lock = threading.Lock()

    def _before_close(self) -> None:
        """Runs under the close lock while hooks may still record (default: nothing)."""

    def _extra_summary(self) -> dict:
        """Adapter fields of ``profiler.json`` (default: none)."""
        return {}

    def _close_helpers(self) -> list:
        """Stop threads and servers; every helper is attempted, failures returned."""
        return []

    def close(self):
        """Stop hooks, then helpers, then the recorder; idempotent and bounded."""
        with self._lock:
            if self.closed:
                return self.result
            self._before_close()
            self.closed = True
            self._record = _discard
            failures = self._close_helpers()
            try:
                self.result = self.recorder.close()
            except Exception as exc:
                failures.append(("recorder", exc))
            self.shutdown_failures = [
                f"{name}: {type(exc).__name__}" for name, exc in failures
            ]
            try:
                self._write_summary()
            except Exception:
                pass
            return self.result

    def _write_summary(self) -> None:
        import contextlib

        from .sink import atomic_json

        with contextlib.suppress(OSError):
            atomic_json(
                self.output_dir / "profiler.json",
                {
                    "hook_errors": self.hook_errors,
                    "first_error": self.first_error,
                    "shutdown_failures": self.shutdown_failures,
                    "capabilities": self.config.context.get("capabilities"),
                    "transport": self.config.context.get("transport"),
                    "backend": self.config.context.get("backend"),
                    "saved": None if self.result is None else self.result.saved,
                    "complete": None if self.result is None else self.result.complete,
                    **self._extra_summary(),
                },
            )


class NixlProfiler(TimelineMixin, _Profiler):
    """One per NIXL scheduler or worker process."""

    def __init__(self, config, recorder, *, rank: int, engine_id: str):
        super().__init__(config, recorder)
        self._timeline_init(config.step_window_s)
        self.rank = rank
        self.engine_id = engine_id
        self.peers: PeerClocks | None = None
        self.server = None
        self.server_report: dict | None = None  # the producer's clock server
        # Request context of the READ being posted (worker main thread).
        self._request = self._remote_request = self._peer = None
        self._source_rank = 0
        # The pass (one ``start_load_kv`` call) the reads being posted belong to.
        self._pass = 0

    # ------------------------------------------------------------------ worker
    def handshake_begin(self, engine_id):
        try:
            self._record(
                _HANDSHAKE_BEGIN, self.rank, None, None, engine_id, None,
                None, None, None, None, None, None, None,
                self._transport, 0, 0,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def handshake_peer_done(self, engine_id, remote_rank, pp_rank, agent_name):
        try:
            self._record(
                _HANDSHAKE_END, self.rank, None, agent_name, engine_id, None,
                None, None, None, None, None, pp_rank, None,
                self._transport, 0, remote_rank,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def handshake_offset(self, engine_id, offset_s):
        """NIXL's own min-RTT clock offset to the peer, as signed nanoseconds.

        Recorded as a cross-check only (``aux0``, two's complement in 64 bits;
        decoded to signed by the exporter). The clock channel stays authoritative.
        """
        try:
            self._record(
                _HANDSHAKE_END, self.rank, None, None, engine_id, None,
                None, None, None, None, None, int(offset_s * 1e9) & _MASK64, None,
                self._transport, 0, _OFFSET_SAMPLE,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def handshake_waiter(self, request_id, engine_id, ok):
        try:
            self._record(
                _HANDSHAKE_END, self.rank, request_id, None, engine_id, None,
                None, None, None, None, None, None, None,
                self._transport, 0 if ok else 1, 0,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def load_pass(self):
        """One ``start_load_kv`` call begins: the reads it posts form one post group.

        Records nothing.
        """
        self._pass += 1

    def read_context(self, request_id, remote_request_id, engine_id, remote_rank):
        self._request = request_id
        self._remote_request = remote_request_id
        self._peer = engine_id
        self._source_rank = remote_rank

    def transfer_prepared(self, handle, leg, notif, selected):
        """``selected``: the descriptors vLLM selected for this READ or leg (the
        length of the existing array), before NIXL merges any. The first value word
        carries the post pass (never bytes: the decoder names it ``post_pass``)."""
        try:
            self._record(
                _TRANSFER_BIND, self.rank, self._request, self._remote_request,
                self._peer, handle._handle, self._pass, selected, None, None, None,
                leg, notif, self._transport, 0, self._source_rank,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def transfer_posted(self, handle):
        try:
            self._record(
                _POST_RETURN, self.rank, None, None, None, handle._handle,
                None, None, None, None, None, None, None,
                self._transport, 0, 0,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def transfer_done(self, handle, res):
        """``res`` is the telemetry the worker just recorded in its stats, or None.

        NIXL's own integer microseconds go to ``aux0`` (post) and ``aux1`` (transfer)
        unconverted; they are the transport's durations, distinct from the interval
        between this record's timestamp and the BIND/POST_RETURN ones. With no
        telemetry every metric is unknown.
        """
        try:
            if res is None:
                self._record(
                    _COMPLETE_OBSERVED, self.rank, None, None, None, handle._handle,
                    None, None, None, None, None, None, None,
                    self._transport, 0, _WORKER,
                )  # fmt: skip
            else:
                self._record(
                    _COMPLETE_OBSERVED, self.rank, None, None, None, handle._handle,
                    res.totalBytes, None, res.descCount, None, None,
                    res.postDuration, res.xferDuration,
                    self._transport, 0, _WORKER,
                )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def transfer_unposted(self, handle):
        """A prepared READ (a mixed-memory leg) was released, never posted."""
        try:
            self._record(
                _FAILED, self.rank, None, None, None, handle._handle,
                None, None, None, None, None, None, None,
                self._transport, int(Failure.TRANSFER_NOT_POSTED), _TRANSFER_SCOPE,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def transfer_retired(self, handle):
        try:
            self._record(
                _TRANSFER_RETIRE, self.rank, None, None, None, handle._handle,
                None, None, None, None, None, None, None,
                self._transport, 0, 0,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def transfer_failed(self, handle):
        try:
            self._record(
                _FAILED, self.rank, None, None, None, handle._handle,
                None, None, None, None, None, None, None,
                self._transport, FAILURE_CODES["transfer_failed"], _TRANSFER_SCOPE,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def failure(self, failure_type, request_id, extra):
        """Request-scoped cause from ``_log_failure``; never tied to a source."""
        try:
            self._record(
                _FAILED, self.rank, request_id, None, None, None,
                None, None, None, None, None, None, None,
                self._transport, FAILURE_CODES.get(failure_type, _UNKNOWN_FAILURE),
                _REQUEST_SCOPE,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def notify_sent(self, request_id, agent_name):
        try:
            self._record(
                _NOTIFY_SEND_RETURN, self.rank, request_id, None, agent_name, None,
                None, None, None, None, None, None, None,
                self._transport, 0, 0,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def notify_observed(self, request_id, count, expected):
        try:
            self._record(
                _NOTIFY_OBSERVED, self.rank, request_id, None, None, None,
                None, None, None, None, None, count, expected,
                self._transport, 0, 0,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def senders_done(self, request_id):
        try:
            self._record(
                _SENDERS_DONE, self.rank, request_id, None, None, None,
                None, None, None, None, None, None, None,
                self._transport, 0, _WORKER,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def late_notification(self, request_id):
        try:
            self._record(
                _FAILED, self.rank, request_id, None, None, None,
                None, None, None, None, None, None, None,
                self._transport, _LATE_NOTIFICATION, _REQUEST_SCOPE,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def lease_expired(self, request_id, count):
        try:
            self._record(
                _FAILED, self.rank, request_id, None, None, None,
                None, None, None, None, None, count, None,
                self._transport, _LEASE_EXPIRED, _REQUEST_SCOPE,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    # --------------------------------------------------------------- scheduler
    def request_bound(self, request_id):
        try:
            self._record(
                _REQUEST_BIND, self.rank, request_id, None, None, None,
                None, None, None, None, None, None, None,
                self._transport, 0, 0,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def handoff_in(self, request_id, remote_request_id, engine_id, host, port):
        try:
            self._record(
                _HANDOFF, self.rank, request_id, remote_request_id, engine_id, None,
                None, None, None, None, None, None, None,
                self._transport, 0, 0,
            )  # fmt: skip
            if self.peers is not None and self.peers.admit(engine_id, host, port) < 0:
                self._record(
                    _FAILED, self.rank, None, None, engine_id, None,
                    None, None, None, None, None, None, None,
                    self._transport, _TABLE_FULL, _REQUEST_SCOPE,
                )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def handoff_out(self, request_id, generation, block_ids):
        """``block_ids``: the clipped per-group block lists the handoff carries."""
        try:
            count = 0
            for group in block_ids:
                count += len(group)
            self._record(
                _HANDOFF, self.rank, request_id, None, self.engine_id, None,
                None, None, None, None, None, generation, count,
                self._transport, 0, 1,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def handshake_served(self, identity, pp_rank, tp_rank):
        try:
            self._record(
                _HANDSHAKE_END, self.rank, None, None, identity, None,
                None, None, None, None, None, pp_rank, None,
                self._transport, 0, tp_rank,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def recv_finished(self, request_id, failed=False):
        """The scheduler's aggregation of finished receives: not proof of success, since
        a failed request also reports finished (``failed``: it is in the connector
        output's failed-receive set, status 1 as Mooncake's)."""
        try:
            self._record(
                _COMPLETE_OBSERVED, self.rank, request_id, None, None, None,
                None, None, None, None, None, None, None,
                self._transport, 1 if failed else 0, _SCHEDULER,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def releases_requested(self, request_ids):
        """All workers finished sending these requests: the scheduler may now free
        their blocks. This is the permission, not the block-pool return."""
        try:
            if request_ids:
                for request_id in request_ids:
                    self._record(
                        _RELEASE_REQUESTED, self.rank, request_id, None, None, None,
                        None, None, None, None, None, None, None,
                        self._transport, 0, _SCHEDULER,
                    )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def _before_close(self) -> None:
        self._timeline_close()

    def _extra_summary(self) -> dict:
        return {**self._timeline_summary(), "clock": {"server": self.server_report}}

    def _close_helpers(self) -> list:
        failures = []
        if self.peers is not None:
            try:
                self.peers.close(self.output_dir, self.config.shutdown_timeout_s)
            except Exception as exc:
                failures.append(("clock peers", exc))
        if self.server is not None:
            try:
                self.server.close()
            except Exception as exc:
                failures.append(("clock server", exc))
        return failures


class FrontendProfiler(TimelineMixin, _Profiler):
    """Binds the exact header, API ids and engine ids where they coexist, and records
    when the first token output and the final output reached the frontend."""

    def __init__(self, config, recorder):
        super().__init__(config, recorder)
        self.rank = 0
        self._timeline_init(config.step_window_s)

    def _before_close(self) -> None:
        self._timeline_close()

    def _extra_summary(self) -> dict:
        return self._timeline_summary()

    def api_request(self, base, header, parent, body_given):
        """Header, base and parent API id of one request.

        ``header`` is the ``X-Request-Id`` value the caller read (None when the
        request carries none; an empty header is a header, as in
        ``_base_request_id``), so provenance never depends on object identity.
        ``body_given`` is the request model's own field-presence flag, separating
        a body ``request_id`` from one generated by the model's default factory.
        """
        try:
            if header is not None:
                source = _HEADER
            else:
                source = _BODY if body_given else _GENERATED
            self._record(
                _REQUEST_BIND, 0, base, header, None, None,
                None, None, None, None, None, None, None,
                self._transport, source, _HEADER_TO_BASE,
            )  # fmt: skip
            self._record(
                _REQUEST_BIND, 0, parent, base, None, None,
                None, None, None, None, None, None, None,
                self._transport, 0, _BASE_TO_PARENT,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def parent_item(self, parent, item):
        try:
            self._record(
                _REQUEST_BIND, 0, item, parent, None, None,
                None, None, None, None, None, None, None,
                self._transport, 0, _PARENT_TO_ITEM,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def external_engine(self, external, engine):
        """Status 1 when the engine id is the external object (no randomization)."""
        try:
            self._record(
                _REQUEST_BIND, 0, engine, external, None, None,
                None, None, None, None, None, None, None,
                self._transport, 1 if engine is external else 0,
                _EXTERNAL_TO_ENGINE,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def child_parent(self, child, parent):
        try:
            self._record(
                _REQUEST_BIND, 0, child, parent, None, None,
                None, None, None, None, None, None, None,
                self._transport, 0, _CHILD_TO_PARENT,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)


# ------------------------------------------------------------------ attachment


def _profiler_config(vllm_config, context: dict) -> NetworkProfilerConfig | None:
    kv = vllm_config.kv_transfer_config
    return profiler_config(
        None if kv is None else kv.kv_connector_extra_config, context
    )


def capabilities(role: str) -> dict:
    return {
        "attached": True,
        "socket": False,
        "qp": False,
        "selected_descriptors": True,
        "nixl_post_xfer_durations": True,
        "post_pass": role == "worker",
        "compute_timeline": True,
        "nixl_offset": True,
        "handoff_blocks": True,
        "block_return": False,
        "router_id": role == "frontend",
        "push_mode": False,
    }


def attach_nixl(
    vllm_config,
    role: str,
    *,
    engine_id: str,
    rank: int,
    transfer_mode: str = "pull",
    side_channel: tuple[str, int] | None = None,
    context: dict | None = None,
    transport_snapshot=None,
):
    """Return a profiler for a NIXL ``scheduler`` or ``worker``, or None when off.

    ``transport_snapshot``, a callable reporting what the NIXL agent set up, is
    called only for an enabled worker.

    Raises:
        ValueError: invalid configuration, or the push connector, which is not
            instrumented.

    """
    kv = vllm_config.kv_transfer_config
    context = {
        "role": role,
        "kv_role": kv.kv_role if kv is not None else None,
        "engine_id": engine_id,
        "rank": rank,
        "connector": "nixl",
        **timeline_context(vllm_config, role),
        **(context or {}),
    }
    config = _profiler_config(vllm_config, context)
    if config is None:
        return None
    if transfer_mode != "pull":
        raise ValueError(
            "network_profiler levels l1/l3 require the NIXL pull connector"
        )
    from .snapshot import snapshot

    transport = (
        transport_snapshot() if transport_snapshot and role == "worker" else None
    )
    settings = snapshot(vllm_config, config, transport)
    return _build(config, role, engine_id, rank, side_channel, kv.kv_role, settings)


def _build(
    config, role, engine_id, rank, side_channel, kv_role, settings=None
) -> NixlProfiler:
    import dataclasses

    from .snapshot import tent_settings
    from .transport import derive

    backends = config.context.get("backends")
    context = {
        **config.context,
        "transport": derive(backends, "nixl", tent_settings=tent_settings()),
        "capabilities": capabilities(role),
        "backend": backends,
    }
    config = dataclasses.replace(config, context=context)
    recorder = open_recorder(config, settings)
    profiler = NixlProfiler(config, recorder, rank=rank, engine_id=engine_id)
    try:
        if role == "scheduler" and config.clock and side_channel is not None:
            from .clock import start_server

            generation = recorder.metadata["clock_id"]
            host, port = side_channel
            if kv_role in ("kv_producer", "kv_both"):
                profiler.server, profiler.server_report = start_server(
                    config.run_id,
                    generation,
                    engine_id,
                    host,
                    port + config.clock_port_offset,
                )
            if kv_role in ("kv_consumer", "kv_both"):
                profiler.peers = PeerClocks(config, config.run_id, generation)
    except BaseException:
        profiler.close()
        raise
    return profiler


def _api_server_context(vllm_config) -> dict:
    """Where this frontend runs, in its metadata (the row that survives a lost
    capture): its process name (``MainProcess``, vLLM's ``ApiServer_<i>`` with
    several API servers, ``APIServer_DPRank_<r>`` for a rank's own server).
    Several API servers are child processes that, at ``--shutdown-timeout 0``,
    vLLM force-kills right after SIGTERM, before their recorders close: the
    metadata then says why."""
    import multiprocessing

    name = multiprocessing.current_process().name
    if not name.startswith("ApiServer_"):
        return {"api_server": name}
    timeout = getattr(vllm_config, "shutdown_timeout", None)
    return {
        "api_server": name,
        "shutdown_timeout_s": timeout,
        "close_at_shutdown": "not guaranteed: force-killed at --shutdown-timeout 0"
        if timeout == 0
        else "within --shutdown-timeout",
    }


def attach_frontend(vllm_config) -> FrontendProfiler | None:
    """Return a frontend profiler, owned and closed by the engine that calls this."""
    import dataclasses

    kv = vllm_config.kv_transfer_config
    config = _profiler_config(
        vllm_config,
        {"role": "frontend", "kv_role": kv.kv_role if kv is not None else None},
    )
    if config is None:
        return None
    if kv is not None and kv.kv_connector == "MooncakeConnector":
        from .mooncake import capabilities as mooncake_capabilities

        caps = mooncake_capabilities("frontend", kv.kv_role, config)
        config = dataclasses.replace(
            config, context={**config.context, "connector": "mooncake"}
        )
    else:
        caps = capabilities("frontend")
    context = {
        **config.context,
        "capabilities": caps,
        **_api_server_context(vllm_config),
    }
    config = dataclasses.replace(config, context=context)
    from .snapshot import snapshot

    recorder = open_recorder(config, snapshot(vllm_config, config))
    try:
        return FrontendProfiler(config, recorder)
    except BaseException:
        recorder.close()
        raise


def close_with_owner(owner, profiler) -> None:
    """Close ``profiler`` when ``owner`` is collected, without keeping it alive."""
    weakref.finalize(owner, profiler.close)


__all__ = [
    "FrontendProfiler",
    "NixlProfiler",
    "attach_frontend",
    "attach_nixl",
    "close_with_owner",
]
