# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 adapter for the Mooncake connector.

Mooncake's producer *writes* KV into the consumer: the unit of observation is a
write batch, the requests one ``send_kv_to_decode`` loop pass found ready for one
consumer rank. Each batch owns one generation, in the frame that owns the batch:

    BIND (preparation) -> PREPARED (executor entry, optional) -> POST_BEGIN
    (immediately before the synchronous call) -> POST_RETURN (after it returns) ->
    COMPLETE_OBSERVED | FAILED -> member and failure edges -> RETIRE

A batch that moves nothing (a replica of a replicated producer, or every member
rejected) has no POST_BEGIN/POST_RETURN: its terminal says so explicitly. An
exceptional or cancelled batch has a FAILED, never a fabricated return, and RETIRE
runs in the owning frame's ``finally``.

The adapter keeps a bounded registry of open generations (``inflight_cap``). When
the profiler closes while a generation is open it records FAILED(interrupted) and
RETIRE for it, marks the observation incomplete in ``profiler.json`` even with zero
native drops, and later calls for that generation do nothing: an interrupted
observation never becomes a complete one. Hooks are short and never wait; the only
lock is held for dictionary operations and the native record call (which holds the
GIL and does no I/O).
"""

import threading

from .compute import TimelineMixin, timeline_context
from .config import NetworkProfilerConfig
from .core import open_recorder
from .nixl import _Profiler, close_with_owner
from .peers import PeerClocks
from .schema import EventKind, Failure, Subject

_REQUEST_BIND = int(EventKind.REQUEST_BIND)
_HANDOFF = int(EventKind.HANDOFF)
_HANDSHAKE_BEGIN = int(EventKind.HANDSHAKE_BEGIN)
_HANDSHAKE_END = int(EventKind.HANDSHAKE_END)
_TRANSFER_BIND = int(EventKind.TRANSFER_BIND)
_PREPARED = int(EventKind.PREPARED)
_POST_BEGIN = int(EventKind.POST_BEGIN)
_POST_RETURN = int(EventKind.POST_RETURN)
_COMPLETE_OBSERVED = int(EventKind.COMPLETE_OBSERVED)
_SENDERS_DONE = int(EventKind.SENDERS_DONE)
_RELEASE_REQUESTED = int(EventKind.RELEASE_REQUESTED)
_FAILED = int(EventKind.FAILED)
_TRANSFER_RETIRE = int(EventKind.TRANSFER_RETIRE)

_WORKER = int(Subject.WORKER)
_SCHEDULER = int(Subject.SCHEDULER)
_REQUEST_SCOPE = int(Subject.REQUEST)
_TRANSFER_SCOPE = int(Subject.TRANSFER)
_BATCH_MEMBER = int(Subject.BATCH_MEMBER)
_HANDOFF_IDENTITY = int(Subject.HANDOFF_IDENTITY)

_TRANSFER_FAILED = int(Failure.TRANSFER_FAILED)
_SETUP_FAILED = int(Failure.TRANSFER_SETUP_FAILED)
_LEASE_EXPIRED = int(Failure.LEASE_EXPIRED)
_INTERRUPTED = int(Failure.INTERRUPTED)
_READY_TIMEOUT = int(Failure.READY_TIMEOUT)
_MEMBER_REJECTED = int(Failure.MEMBER_REJECTED)
_RECEIVE_FAILED = int(Failure.RECEIVE_FAILED)
_SEND_EXCEPTION = int(Failure.SEND_EXCEPTION)
_HANDSHAKE_FAILED = int(Failure.HANDSHAKE_FAILED)

# COMPLETE_OBSERVED ``status``.
COMPLETE_WRITTEN = 0
COMPLETE_FAILED_RECEIVE_AGGREGATE = 1
COMPLETE_NO_DATA = 2
# Batch member ``status``.
MEMBER_OK = 0
MEMBER_FAILED = 1

# Phases of an open generation, in order.
_BOUND, _PREPARED_PHASE, _POSTING, _POSTED, _TERMINAL = range(5)

_UNMAPPED_CAP = 16


class MooncakeProfiler(TimelineMixin, _Profiler):
    """One per Mooncake scheduler or worker process."""

    def __init__(self, config, recorder, *, rank: int, engine_id: str, kv_role: str):
        super().__init__(config, recorder)
        self._timeline_init(config.step_window_s)
        self.rank = rank
        self.engine_id = engine_id
        self.kv_role = kv_role
        self.peers: PeerClocks | None = None
        self.server = None
        self.server_report: dict | None = None  # the producer's clock server
        self.advertised: dict | None = None  # its advertisement, once it ends
        self._advertiser: threading.Thread | None = None
        self._stop_advertising = threading.Event()
        self.unaddressed_clock_engines: list[str] = []
        # Open generations: handle -> phase. Guarded by ``_glock``, which also
        # serializes every generation record with ``close`` so that nothing is
        # recorded for a generation after its RETIRE.
        self._glock = threading.Lock()
        self._live: dict[int, int] = {}
        self._members: dict[int, set] = {}
        self._next_handle = 0
        self._cap = config.effective_inflight_cap
        self._closing = False
        self.interrupted_generations = 0
        self.inflight_overflow = 0
        self.late_hooks = 0
        self.unbound_batches = 0

    def _detach(self) -> None:
        """A forked child: the generation lock may have been held by a parent thread
        that does not exist here, and the ring belongs to the parent."""
        super()._detach()
        self._glock = threading.Lock()
        self._live = {}
        self._members = {}

    # ------------------------------------------------------------- generations
    def batch_bind(self, session, remote_tp_rank, members) -> int | None:
        """Open the generation of one write batch; None when none is admitted.

        ``session``: the consumer's ``host:port``; ``members``: the batch's request
        count. The handle is private to the caller and never reaches the output.
        """
        if self._detached:
            return None
        try:
            with self._glock:
                if self._closing:
                    self.late_hooks += 1
                    return None
                if len(self._live) >= self._cap:
                    self.inflight_overflow += 1
                    return None
                self._next_handle += 1
                handle = self._next_handle
                self._live[handle] = _BOUND
                self._record(
                    _TRANSFER_BIND, self.rank, None, None, session, handle,
                    None, None, None, None, None, members, None,
                    self._transport, 0, remote_tp_rank,
                )  # fmt: skip
                return handle
        except Exception as exc:
            self._error(exc)
            return None

    def _step(self, handle, expected, to) -> bool:
        """Advance an open generation's phase (``_glock`` held): False for a late or
        out-of-order call, which records nothing."""
        phase = self._live.get(handle)
        if phase is None or phase not in expected:
            self.late_hooks += 1
            return False
        self._live[handle] = to
        return True

    def batch_prepared(self, handle) -> None:
        """The executor thread picked the batch up (queue time ends)."""
        if self._detached:
            return
        try:
            with self._glock:
                if self._closing or not self._step(handle, (_BOUND,), _PREPARED_PHASE):
                    return
                self._record(
                    _PREPARED, self.rank, None, None, None, handle,
                    None, None, None, None, None, None, None,
                    self._transport, 0, 0,
                )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def write_begin(self, handle, bytes_, descriptors) -> None:
        """Immediately before the synchronous write call."""
        if self._detached:
            return
        try:
            with self._glock:
                if self._closing or not self._step(
                    handle, (_BOUND, _PREPARED_PHASE), _POSTING
                ):
                    return
                self._record(
                    _POST_BEGIN, self.rank, None, None, None, handle,
                    bytes_, descriptors, None, None, None, None, None,
                    self._transport, 0, 0,
                )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def write_returned(self, handle) -> None:
        """Immediately after the synchronous write call returned (any result)."""
        if self._detached:
            return
        try:
            with self._glock:
                if self._closing or not self._step(handle, (_POSTING,), _POSTED):
                    return
                self._record(
                    _POST_RETURN, self.rank, None, None, None, handle,
                    None, None, None, None, None, None, None,
                    self._transport, 0, 0,
                )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def write_complete(self, handle, bytes_, descriptors, duration_ns) -> None:
        """The write returned success. ``duration_ns`` is the connector's own timer
        around the call, an observation interval and not engine telemetry."""
        if self._detached:
            return
        try:
            with self._glock:
                if self._closing or not self._step(handle, (_POSTED,), _TERMINAL):
                    return
                self._record(
                    _COMPLETE_OBSERVED, self.rank, None, None, None, handle,
                    bytes_, descriptors, None, duration_ns, None, None, None,
                    self._transport, COMPLETE_WRITTEN, _WORKER,
                )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def write_failed(self, handle, ret) -> None:
        """The write call returned non-zero ``ret``."""
        if self._detached:
            return
        try:
            with self._glock:
                if self._closing or not self._step(handle, (_POSTED,), _TERMINAL):
                    return
                self._record(
                    _FAILED, self.rank, None, None, None, handle,
                    None, None, None, None, None, ret & ((1 << 64) - 1), None,
                    self._transport, _TRANSFER_FAILED, _TRANSFER_SCOPE,
                )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def batch_no_data(self, handle) -> None:
        """The batch moved nothing: a replica of a replicated producer. Explicit zero
        bytes and descriptors, no POST_BEGIN/POST_RETURN."""
        if self._detached:
            return
        try:
            with self._glock:
                if self._closing or not self._step(handle, (_BOUND,), _TERMINAL):
                    return
                self._record(
                    _COMPLETE_OBSERVED, self.rank, None, None, None, handle,
                    0, 0, None, None, None, None, None,
                    self._transport, COMPLETE_NO_DATA, _WORKER,
                )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def batch_rejected(self, handle) -> None:
        """Every member was rejected before any write: the batch's terminal."""
        if self._detached:
            return
        try:
            with self._glock:
                if self._closing or not self._step(handle, (_BOUND,), _TERMINAL):
                    return
                self._record(
                    _FAILED, self.rank, None, None, None, handle,
                    None, None, None, None, None, None, None,
                    self._transport, _MEMBER_REJECTED, _TRANSFER_SCOPE,
                )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def batch_exception(self, handle, cancelled, members=()) -> None:
        """The owning frame raised (or was cancelled).

        Records the generation's failed terminal if none was recorded, then every
        member not already recorded as failed, with its request-scoped failure:
        their pulls received no answer. ``members``: (decode request, transfer ID) or
        (decode request, transfer ID, share), the share being the member's planned
        (bytes, descriptors) when the build had finished and None when it had not.
        """
        if self._detached:
            return
        try:
            with self._glock:
                phase = self._live.get(handle)
                if self._closing or phase is None:
                    return
                if phase != _TERMINAL:
                    self._live[handle] = _TERMINAL
                    self._record(
                        _FAILED, self.rank, None, None, None, handle,
                        None, None, None, None, None, 1 if cancelled else 0, None,
                        self._transport, _SEND_EXCEPTION, _TRANSFER_SCOPE,
                    )  # fmt: skip
                done = self._members.setdefault(handle, set())
                for member in members:
                    d_request, transfer_id = member[0], member[1]
                    share = member[2] if len(member) > 2 else None
                    if d_request in done:
                        continue
                    done.add(d_request)
                    bytes_, descriptors = share if share else (None, None)
                    self._record(
                        _REQUEST_BIND, self.rank, d_request, transfer_id, None, handle,
                        bytes_, descriptors, None, None, None, None, None,
                        self._transport, MEMBER_FAILED, _BATCH_MEMBER,
                    )  # fmt: skip
                    self._record(
                        _FAILED, self.rank, d_request, transfer_id, None, None,
                        None, None, None, None, None, 1 if cancelled else 0, None,
                        self._transport, _SEND_EXCEPTION, _REQUEST_SCOPE,
                    )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def member_shares(self, spans, lengths) -> dict | None:
        """Each member's share of a built write batch: ``spans`` are the build's (decode
        request, first, end) positions in the flat descriptor lists, ``lengths`` the
        batch's descriptor lengths. Returns {decode request: (bytes, descriptors)}; the
        spans partition the lists, so the shares sum to the batch's totals. None if the
        spans do not describe the lists."""
        try:
            shares = {}
            cursor = 0
            for d_request, first, end in spans:
                if first != cursor or end < first or end > len(lengths):
                    return None
                shares[d_request] = (sum(lengths[first:end]), end - first)
                cursor = end
            return shares if cursor == len(lengths) else None
        except Exception as exc:
            self._error(exc)
            return None

    def batch_member(self, handle, d_request, transfer_id, ok, share=None) -> None:
        """One member's settled outcome, after the batch's terminal. ``share``: the
        member's (bytes, descriptors) of the batch as built (posted if the write
        succeeded, planned if it failed); None when unknown."""
        if self._detached:
            return
        try:
            with self._glock:
                if self._closing or self._live.get(handle) != _TERMINAL:
                    self.late_hooks += 1
                    return
                done = self._members.setdefault(handle, set())
                if d_request in done:
                    return
                done.add(d_request)
                bytes_, descriptors = share if share else (None, None)
                self._record(
                    _REQUEST_BIND, self.rank, d_request, transfer_id, None, handle,
                    bytes_, descriptors, None, None, None, None, None,
                    self._transport, MEMBER_OK if ok else MEMBER_FAILED, _BATCH_MEMBER,
                )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def batch_retire(self, handle) -> None:
        """Exactly once, last, in the frame that owns the generation."""
        if self._detached:
            return
        try:
            with self._glock:
                if handle not in self._live:
                    return
                del self._live[handle]
                self._members.pop(handle, None)
                if self._closing:
                    return
                self._record(
                    _TRANSFER_RETIRE, self.rank, None, None, None, handle,
                    None, None, None, None, None, None, None,
                    self._transport, 0, 0,
                )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    # ---------------------------------------------------------- request edges
    def request_failed(self, request_id, transfer_id, code) -> None:
        """A request-scoped failure (not tied to a generation)."""
        try:
            self._record(
                _FAILED, self.rank, request_id, transfer_id, None, None,
                None, None, None, None, None, None, None,
                self._transport, code, _REQUEST_SCOPE,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def member_failed(self, d_request, transfer_id, rejected) -> None:
        """A batch member's request failed: rejected before any write, or its write
        failed."""
        self.request_failed(
            d_request, transfer_id, _MEMBER_REJECTED if rejected else _TRANSFER_FAILED
        )

    def send_rejected(self, d_requests, code=_SETUP_FAILED) -> None:
        """The producer answered a consumer's pull with an error before any batch:
        every request it named failed (``d_requests``: ``req_blocks``)."""
        try:
            for d_request, (transfer_id, _) in d_requests.items():
                self._record(
                    _FAILED, self.rank, d_request, transfer_id, None, None,
                    None, None, None, None, None, None, None,
                    self._transport, code, _REQUEST_SCOPE,
                )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def ready_timeout(self, pending) -> None:
        """Requests the producer did not have ready before the timeout (``pending``:
        d_request -> SendBlockMeta). Not a transport failure."""
        try:
            for d_request, send_meta in pending.items():
                self._record(
                    _FAILED, self.rank, d_request, send_meta.transfer_id, None, None,
                    None, None, None, None, None, None, None,
                    self._transport, _READY_TIMEOUT, _REQUEST_SCOPE,
                )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def expired_before_pull(self, d_requests) -> None:
        """Requests pulled after their KV lease expired on the producer."""
        self.send_rejected(d_requests, _LEASE_EXPIRED)

    def identity(self, p_request, transfer_id) -> None:
        """The producer worker's pairing of its request with the shared transfer ID:
        not a canonical handoff row."""
        try:
            self._record(
                _REQUEST_BIND, self.rank, p_request, transfer_id, None, None,
                None, None, None, None, None, None, None,
                self._transport, 0, _HANDOFF_IDENTITY,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def senders_done(self, p_request) -> None:
        try:
            self._record(
                _SENDERS_DONE, self.rank, p_request, None, None, None,
                None, None, None, None, None, None, None,
                self._transport, 0, _WORKER,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def lease_expired(self, p_request) -> None:
        try:
            self._record(
                _FAILED, self.rank, p_request, None, None, None,
                None, None, None, None, None, None, None,
                self._transport, _LEASE_EXPIRED, _REQUEST_SCOPE,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    # ------------------------------------------------------------ consumer side
    def bootstrap_begin(self, remote_engine_id) -> None:
        try:
            self._record(
                _HANDSHAKE_BEGIN, self.rank, None, None, remote_engine_id, None,
                None, None, None, None, None, None, None,
                self._transport, 0, 0,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def bootstrap_end(self, remote_engine_id, ok) -> None:
        try:
            self._record(
                _HANDSHAKE_END, self.rank, None, None, remote_engine_id, None,
                None, None, None, None, None, None, None,
                self._transport, 0 if ok else _HANDSHAKE_FAILED, 0,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def recv_complete(self, d_request, transfer_id) -> None:
        """This rank's receive completed: every peer answered, none failed."""
        try:
            self._record(
                _COMPLETE_OBSERVED, self.rank, d_request, transfer_id, None, None,
                None, None, None, None, None, None, None,
                self._transport, COMPLETE_WRITTEN, _WORKER,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def recv_failed(self, d_request, transfer_id) -> None:
        self.request_failed(d_request, transfer_id, _RECEIVE_FAILED)

    # --------------------------------------------------------------- scheduler
    def request_bound(self, request_id) -> None:
        try:
            self._record(
                _REQUEST_BIND, self.rank, request_id, None, None, None,
                None, None, None, None, None, None, None,
                self._transport, 0, 0,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def handoff_in(
        self, request_id, transfer_id, remote_engine_id, remote_bootstrap_addr=None
    ) -> None:
        """``remote_bootstrap_addr`` (the request's own) is where the remote
        engine's clock is advertised; the measuring thread looks it up."""
        try:
            self._record(
                _HANDOFF, self.rank, request_id, transfer_id, remote_engine_id, None,
                None, None, None, None, None, None, None,
                self._transport, 0, 0,
            )  # fmt: skip
            if self.peers is not None:
                if not remote_bootstrap_addr:
                    if (
                        remote_engine_id not in self.unaddressed_clock_engines
                        and len(self.unaddressed_clock_engines) < _UNMAPPED_CAP
                    ):
                        self.unaddressed_clock_engines.append(remote_engine_id)
                elif (
                    self.peers.admit_advertised(remote_engine_id, remote_bootstrap_addr)
                    < 0
                ):
                    self._record(
                        _FAILED, self.rank, None, None, remote_engine_id, None,
                        None, None, None, None, None, None, None,
                        self._transport, int(Failure.CLOCK_PEER_TABLE_FULL),
                        _REQUEST_SCOPE,
                    )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def handoff_out(self, request_id, transfer_id, block_ids) -> None:
        """``block_ids``: the clipped per-group block lists the handoff carries."""
        try:
            count = 0
            for group in block_ids:
                count += len(group)
            self._record(
                _HANDOFF, self.rank, request_id, transfer_id, self.engine_id, None,
                None, None, None, None, None, None, count,
                self._transport, 0, 1,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def recv_finished(self, request_id, failed) -> None:
        """The scheduler's aggregation of finished receives: not proof of success,
        since a failed request also reports finished (``failed``: its failure was
        reported by request)."""
        try:
            self._record(
                _COMPLETE_OBSERVED, self.rank, request_id, None, None, None,
                None, None, None, None, None, None, None,
                self._transport,
                COMPLETE_FAILED_RECEIVE_AGGREGATE if failed else COMPLETE_WRITTEN,
                _SCHEDULER,
            )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def releases_requested(self, request_ids) -> None:
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

    # ---------------------------------------------------------------- shutdown
    def _before_close(self) -> None:
        """Interrupt every open generation while recording is still enabled.

        Under ``_glock`` and before ``closed`` is set: a hook that raced the close
        either recorded first (its generation is then interrupted here, or already
        retired) or finds the profiler closed and records nothing.
        """
        self._timeline_close()
        with self._glock:
            self._closing = True
            live = list(self._live)
            self._live.clear()
            self._members.clear()
            for handle in live:
                try:
                    self._record(
                        _FAILED, self.rank, None, None, None, handle,
                        None, None, None, None, None, None, None,
                        self._transport, _INTERRUPTED, _TRANSFER_SCOPE,
                    )  # fmt: skip
                    self._record(
                        _TRANSFER_RETIRE, self.rank, None, None, None, handle,
                        None, None, None, None, None, None, None,
                        self._transport, 0, 0,
                    )  # fmt: skip
                except Exception as exc:
                    self._error(exc)
            self.interrupted_generations = len(live)

    def _extra_summary(self) -> dict:
        complete = (
            self.result is not None
            and self.result.complete
            and not self.interrupted_generations
            and not self.inflight_overflow
            and not self.unbound_batches
            and not self.hook_errors
        )
        return {
            **self._timeline_summary(),
            "connector": "mooncake",
            "kv_role": self.kv_role,
            "observation_complete": bool(complete),
            "interrupted_generations": self.interrupted_generations,
            "inflight_overflow": self.inflight_overflow,
            "late_hooks": self.late_hooks,
            "unbound_batches": self.unbound_batches,
            "inflight_cap": self._cap,
            "clock": {
                "server": self.server_report,
                "advertised": self.advertised,
                "unaddressed_engines": list(self.unaddressed_clock_engines),
            },
        }

    def advertise(self, base: str, record: dict, deadline_s: float) -> None:
        """Advertise this engine's clock to ``base``'s registry in the
        background: serving never waits, and the outcome goes into
        ``profiler.json``."""
        from .clock_registry import advertise

        def run():
            self.advertised = advertise(
                base, record, deadline_s, self._stop_advertising
            )

        self._advertiser = threading.Thread(
            target=run, name="kvprof-clock-advertise", daemon=True
        )
        self._advertiser.start()

    def _close_helpers(self) -> list:
        failures = []
        if self._advertiser is not None:
            self._stop_advertising.set()
            self._advertiser.join(self.config.shutdown_timeout_s)
            if self.advertised is None:
                self.advertised = {"state": "abandoned"}
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


def capabilities(role: str, kv_role: str | None, config=None) -> dict:
    """What this process's captures can and cannot contain (process and KV role)."""
    producer = kv_role in ("kv_producer", "kv_both")
    consumer = kv_role in ("kv_consumer", "kv_both")
    worker = role == "worker"
    scheduler = role == "scheduler"
    clock = config is not None and config.clock
    return {
        "attached": True,
        "connector": "mooncake",
        "compute_timeline": True,
        "kv_role": kv_role,
        "socket": False,
        "qp": False,
        "selected_descriptors": False,
        "posted_descriptors": worker and producer,
        "write_batches": worker and producer,
        "batch_members": worker and producer,
        "batch_member_shares": worker and producer,
        "mooncake_write_duration": worker and producer,
        "nixl_post_xfer_durations": False,
        "nixl_offset": False,
        "handoff_blocks": scheduler and producer,
        "block_return": False,
        "router_id": role == "frontend",
        "push_mode": False,
        "receive_completion": (worker or scheduler) and consumer,
        "clock_server": bool(worker and producer and clock and _clock_worker(config)),
        "clock_peers": bool(scheduler and consumer and clock),
        "transport_metrics": "collector",
    }


def _clock_worker(config) -> bool:
    """Whether this worker runs its engine's clock server: the TP0/PP0 worker,
    given where it serves KV (its host) and its registry."""
    context = config.context
    return (
        context.get("tp_rank") == 0
        and context.get("pp_rank") == 0
        and bool(context.get("clock_host"))
        and bool(context.get("clock_registry"))
    )


def clock_context(vllm_config, hostname: str, dp_rank: int) -> dict:
    """A worker's profiler context for its engine's clock: the host it serves
    KV from, the registry it registers with (the bootstrap server) and its
    registration DP rank; empty when the profiler is not configured."""
    extra = vllm_config.kv_transfer_config.kv_connector_extra_config
    if "network_profiler" not in extra:
        return {}
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (  # noqa: E501
        get_mooncake_bootstrap_addr,
    )
    from vllm.utils.network_utils import make_zmq_path

    host, port = get_mooncake_bootstrap_addr(vllm_config)
    return {
        "clock_host": hostname,
        "clock_registry": make_zmq_path("http", host, port),
        "registration_dp_rank": dp_rank,
    }


def serve_clock_registry(server, vllm_config) -> None:
    """Add the clock route to an embedded bootstrap server (before it starts),
    when the profiler is enabled with clocks."""
    from .clock_registry import ClockRegistry, add_routes
    from .config import profiler_config

    config = profiler_config(
        vllm_config.kv_transfer_config.kv_connector_extra_config, {}
    )
    if config is None or not config.clock:
        return
    server.clock_registry = ClockRegistry()
    add_routes(server.app, server.clock_registry)


def attach_mooncake(
    vllm_config,
    role: str,
    *,
    engine_id: str,
    rank: int,
    context: dict | None = None,
):
    """Return a profiler for a Mooncake ``scheduler`` or ``worker``, or None.

    None means the shared validated configuration attaches nothing (no namespace,
    or a level that is not enabled): no recorder, helper or clock thread exists.

    Raises:
        ValueError: invalid configuration.

    """
    from .config import profiler_config

    kv = vllm_config.kv_transfer_config
    context = {
        "role": role,
        "kv_role": kv.kv_role,
        "engine_id": engine_id,
        "rank": rank,
        "connector": "mooncake",
        **timeline_context(vllm_config, role),
        **(context or {}),
    }
    config = profiler_config(kv.kv_connector_extra_config, context)
    if config is None:
        return None
    from .snapshot import snapshot

    settings = snapshot(vllm_config, config)
    return _build(config, role, engine_id, rank, kv.kv_role, settings)


def _build(
    config: NetworkProfilerConfig, role, engine_id, rank, kv_role, settings=None
):
    import dataclasses

    context = {
        **config.context,
        "transport": config.transport,
        "backend": ["Mooncake"],
        "capabilities": capabilities(role, kv_role, config),
    }
    config = dataclasses.replace(config, context=context)
    recorder = open_recorder(config, settings)
    profiler = MooncakeProfiler(
        config, recorder, rank=rank, engine_id=engine_id, kv_role=kv_role
    )
    try:
        generation = recorder.metadata["clock_id"]
        producer = kv_role in ("kv_producer", "kv_both")
        if config.clock and role == "worker" and producer and _clock_worker(config):
            # An engine's clock is its TP0/PP0 worker's, on the host it serves
            # KV from and an ephemeral port; calibrating it aligns every process
            # of its clock domain (its scheduler on that host included).
            from .clock import start_server
            from .clock_registry import endpoint_text

            profiler.server, profiler.server_report = start_server(
                config.run_id, generation, engine_id, config.context["clock_host"], 0
            )
            if profiler.server is not None:
                profiler.advertise(
                    config.context["clock_registry"],
                    {
                        "run_id": config.run_id,
                        "engine_id": engine_id,
                        "dp_rank": int(config.context.get("registration_dp_rank") or 0),
                        "endpoint": endpoint_text(*profiler.server.address),
                        "generation": generation,
                    },
                    config.clock_discovery_s,
                )
        if (
            config.clock
            and role == "scheduler"
            and kv_role in ("kv_consumer", "kv_both")
        ):
            profiler.peers = PeerClocks(config, config.run_id, generation)
    except BaseException:
        profiler.close()
        raise
    return profiler


__all__ = [
    "MooncakeProfiler",
    "attach_mooncake",
    "capabilities",
    "clock_context",
    "close_with_owner",
    "serve_clock_registry",
]
