# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Out-of-process collector of the transport engine's own metrics (level L2).

Stdlib only; run it by file path (``python <vllm>/kv_profiling/collector.py``, with
no vLLM or Torch import: ``python -m vllm.kv_profiling.collector`` would import the
whole ``vllm`` package first). It scrapes each
target's Prometheus text endpoint (TENT's ``/metrics``) and writes JSONL records of
two collector-only kinds (the native recorder's ABI and kinds are unchanged):

* ``transport_metrics`` (25): by ``transport`` label, the cumulative bytes written
  and read, write and read request counts, and the registered buffer gauge;
* ``transport_metrics_failures`` (26): the transport attempts and attempt failures
  by (``transport``, ``operation``), and task failures by (``transport``, ``reason``);

and, for a target whose ``source`` is ``nixl`` (NIXL's built-in telemetry through its
multi-process Prometheus exporter, one aggregated endpoint a node), two more:

* ``nixl_telemetry`` (27): per exporting process, the cumulative transmit and receive
  bytes and requests, the transfer and post time sums in microseconds and the memory
  registered and deregistered;
* ``nixl_telemetry_events`` (28): per process the events dropped at the producer-side
  staging queue, and the errors NIXL reported by ``status`` (not every failure: some
  early returns bypass its error telemetry).

The two kinds are different label partitions of the source and are not forced into
one row shape: each record carries its own ``labels`` and typed ``fields``, a field
the page did not serve is ``null`` (unavailable, never zero), and the two kinds of
one scrape pair by sample ID, target and reset generation. A page that fails or parses
incompletely yields missing rows for the affected half only, with the sample marked
partial. A counter that decreases starts a new reset generation and is flagged: it is
a discontinuity of the series, not proof that the engine restarted.

Everything is bounded and configured explicitly: response bytes, series seen, label
sizes, targets, retained output bytes, the counter history kept between scrapes,
absolute HTTP deadlines (connect, headers and body), the startup budget and the shutdown
budget. A bound that is hit makes the observation partial or the target unavailable and
never affects serving: the collector only reads. The capture ends with a terminal record
that says whether every scrape was complete.

Targets come from a manifest (rank -> endpoint, written at configuration time). A
target without an explicit endpoint is resolved from its own recorded configuration
(its ``MC_TENT_CONF`` and environment), never from the collector's: ambiguous or
unreadable ownership makes the target unavailable with a reason.
"""

import argparse
import contextlib
import errno
import http.client
import ipaddress
import json
import math
import os
import re
import select
import signal
import socket
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

SCHEMA = 1
KIND_TRANSPORT_METRICS = 25
KIND_TRANSPORT_METRICS_FAILURES = 26
KIND_NIXL_TELEMETRY = 27
KIND_NIXL_TELEMETRY_EVENTS = 28
NAMES = {
    KIND_TRANSPORT_METRICS: "transport_metrics",
    KIND_TRANSPORT_METRICS_FAILURES: "transport_metrics_failures",
    KIND_NIXL_TELEMETRY: "nixl_telemetry",
    KIND_NIXL_TELEMETRY_EVENTS: "nixl_telemetry_events",
}

# The source's families: (metric, label names, typed field, unit), by kind.
_TRANSPORT_FAMILIES = {
    "tent_write_bytes_total": ("write_bytes", "bytes", ("transport",)),
    "tent_read_bytes_total": ("read_bytes", "bytes", ("transport",)),
    "tent_write_requests_total": ("write_requests", "requests", ("transport",)),
    "tent_read_requests_total": ("read_requests", "requests", ("transport",)),
    "tent_registered_buffer_bytes": (
        "registered_buffer_bytes",
        "bytes",
        ("transport",),
    ),
}
_FAILURE_FAMILIES = {
    "tent_transport_attempts_total": (
        "attempts",
        "attempts",
        ("transport", "operation"),
    ),
    "tent_transport_attempt_failures_total": (
        "attempt_failures",
        "attempts",
        ("transport", "operation"),
    ),
    "tent_task_failures_total": ("task_failures", "tasks", ("transport", "reason")),
}
# NIXL's multi-process Prometheus exporter: one series set per exporting process, every
# series present from the first scrape (registered at start: absent means unavailable).
# Last-operation gauges, histogram buckets and the HTTP server's own families are not
# recorded (the totals and the sums carry the information).
_NIXL_PROCESS = ("hostname", "agent_name", "pid", "agent_instance")
_NIXL_FAMILIES = {
    "agent_tx_bytes_total": ("tx_bytes", "bytes", _NIXL_PROCESS),
    "agent_rx_bytes_total": ("rx_bytes", "bytes", _NIXL_PROCESS),
    "agent_tx_requests_num_total": ("tx_requests", "requests", _NIXL_PROCESS),
    "agent_rx_requests_num_total": ("rx_requests", "requests", _NIXL_PROCESS),
    "agent_xfer_time_total": ("xfer_time_us", "microseconds", _NIXL_PROCESS),
    "agent_xfer_post_time_total": ("xfer_post_time_us", "microseconds", _NIXL_PROCESS),
    "agent_memory_registered_total": (
        "memory_registered_bytes",
        "bytes",
        _NIXL_PROCESS,
    ),
    "agent_memory_deregistered_total": (
        "memory_deregistered_bytes",
        "bytes",
        _NIXL_PROCESS,
    ),
}
_NIXL_EVENT_FAMILIES = {
    "agent_telemetry_events_dropped_total": ("events_dropped", "events", _NIXL_PROCESS),
    "agent_errors_total": ("errors", "errors", (*_NIXL_PROCESS, "status")),
}
# Per source: its record kinds with their families, and the labels a series may carry
# besides the documented ones (``local_rank`` appears only when the engine exports it).
SOURCES = {
    "tent": {
        KIND_TRANSPORT_METRICS: _TRANSPORT_FAMILIES,
        KIND_TRANSPORT_METRICS_FAILURES: _FAILURE_FAMILIES,
    },
    "nixl": {
        KIND_NIXL_TELEMETRY: _NIXL_FAMILIES,
        KIND_NIXL_TELEMETRY_EVENTS: _NIXL_EVENT_FAMILIES,
    },
}
_OPTIONAL_LABELS = {"tent": (), "nixl": ("local_rank",)}
_GAUGES = frozenset({"tent_registered_buffer_bytes"})
DEFAULT_METRICS_PORT = 9100
_MAX_LINE = 4096
_SERIES = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$")
_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"\s*(?:,|$)')


# ------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class Limits:
    """Fixed, validated bounds; every one has a hard ceiling."""

    interval_s: float = 5.0
    scrape_timeout_s: float = 2.0
    startup_timeout_s: float = 10.0
    shutdown_timeout_s: float = 10.0
    max_response_bytes: int = 1 << 20
    max_series: int = 4096
    max_label_bytes: int = 128
    max_labels: int = 8
    max_targets: int = 64
    max_history: int = 65536
    max_output_bytes: int = 64 << 20

    def __post_init__(self):
        for name, low, high in (
            ("interval_s", 0.05, 3600),
            ("scrape_timeout_s", 0.05, 60),
            ("startup_timeout_s", 0.05, 600),
            ("shutdown_timeout_s", 0.05, 600),
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"{name} must be a finite number in {low}..{high}")
        for name, low, high in (
            ("max_response_bytes", 1 << 10, 64 << 20),
            ("max_series", 1, 1 << 20),
            ("max_label_bytes", 8, 4096),
            ("max_labels", 1, 32),
            ("max_targets", 1, 1024),
            ("max_history", 1, 1 << 22),
            ("max_output_bytes", 1 << 16, 8 << 30),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in {low}..{high}")


@dataclass(frozen=True)
class Target:
    """One scrape target as the manifest records it."""

    id: str
    role: str | None = None
    engine_id: str | None = None
    rank: dict = field(default_factory=dict)
    endpoint: str | None = None
    process_id: str | None = None
    pid: int | None = None
    mc_tent_conf: str | None = None
    env: dict = field(default_factory=dict)
    port_offset: int = 0
    source: str = "tent"

    @classmethod
    def from_json(cls, raw):
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("id"), str)
            or not raw["id"]
        ):
            raise ValueError("a target needs a nonempty id")
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"target {raw['id']!r}: unknown keys {unknown}")
        if not isinstance(raw.get("env", {}), dict):
            raise ValueError(f"target {raw['id']!r}: env must be an object")
        if raw.get("source", "tent") not in SOURCES:
            raise ValueError(
                f"target {raw['id']!r}: source must be one of {sorted(SOURCES)}"
            )
        return cls(**raw)


def load_manifest(path, limits: Limits):
    """The manifest's targets; at most ``max_targets``, ids unique."""
    document = json.loads(Path(path).read_text())
    raw = document.get("targets") if isinstance(document, dict) else None
    if not isinstance(raw, list):
        raise ValueError("the manifest needs a targets list")
    if len(raw) > limits.max_targets:
        raise ValueError(f"the manifest has {len(raw)} targets, more than max_targets")
    targets = [Target.from_json(item) for item in raw]
    if len({t.id for t in targets}) != len(targets):
        raise ValueError("target ids must be unique")
    return targets


def _lookup(conf, key, accept):
    """TENT's Config::get: a '/'-separated key as a nested path, else as a flat key."""
    node = conf
    for segment in filter(None, key.split("/")):
        if not isinstance(node, dict) or segment not in node:
            node = None
            break
        node = node[segment]
    if node is not None and accept(node):
        return node
    flat = conf.get(key)
    if flat is not None and accept(flat):
        return flat
    return None


def _port_value(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 1 <= value <= 65535
    )


def _endpoint(text):
    host, _, port = str(text).rpartition(":")
    host = host.strip("[]")
    if not host or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError(f"not a host:port endpoint: {text!r}")
    return host, int(port)


def resolve_endpoint(target: Target):
    """(host, port, source) from the target's own record, or (None, None, reason).

    Precedence: the manifest's explicit endpoint; the target's recorded
    ``MC_TENT_CONF`` (inline JSON or a file path; ``metrics/http_port`` nested, else
    flat); the target's recorded environment; TENT's default. The collector's
    environment is never read. A configuration that exists but cannot be read is
    ambiguous ownership: unavailable, not a guessed default.
    """
    if target.endpoint:
        try:
            host, port = _endpoint(target.endpoint)
        except ValueError as exc:
            return None, None, str(exc)
        return host, port, "manifest"
    if target.source == "nixl":
        return None, None, "a nixl target needs an explicit endpoint"
    # TENT's precedence, for the host as for the port: the configuration file,
    # else the environment, else its default; a wildcard host is scraped on
    # the loopback address only after that choice.
    host, port, source = None, None, None
    conf_text = target.mc_tent_conf or target.env.get("MC_TENT_CONF")
    if conf_text:
        try:
            text = (
                Path(conf_text).read_text() if os.path.exists(conf_text) else conf_text
            )
            conf = json.loads(text)
            if not isinstance(conf, dict):
                raise ValueError("not an object")
        except (OSError, ValueError):
            return None, None, "the target's MC_TENT_CONF is unreadable"
        found = _lookup(conf, "metrics/http_port", _port_value)
        if found is not None:
            port, source = int(found), "target MC_TENT_CONF"
        host = _lookup(conf, "metrics/http_host", lambda v: isinstance(v, str) and v)
    if port is None and target.env.get("TENT_METRICS_HTTP_PORT"):
        value = str(target.env["TENT_METRICS_HTTP_PORT"])
        if not value.isdigit() or not 1 <= int(value) <= 65535:
            return None, None, "the target's TENT_METRICS_HTTP_PORT is invalid"
        port, source = int(value), "target environment"
    if port is None:
        port, source = DEFAULT_METRICS_PORT, "default"
    if host is None and target.env.get("TENT_METRICS_HTTP_HOST"):
        host = str(target.env["TENT_METRICS_HTTP_HOST"])
    if host is None or host == "0.0.0.0":
        host = "127.0.0.1"
    port += target.port_offset
    if not 1 <= port <= 65535:
        return None, None, "the target's metrics port is out of range"
    return host, port, source


def resolve_targets(targets):
    """Resolve every target; two targets that resolve to one endpoint are both
    ambiguous and unavailable."""
    resolved = {}
    for target in targets:
        host, port, source = resolve_endpoint(target)
        if host and _literal(host) is None:
            host, port, source = (
                None,
                None,
                f"endpoint host {host!r} is not an IP literal (names are not resolved)",
            )
        resolved[target.id] = (host, port, source) if host else (None, None, source)
    seen = {}
    for tid, (host, port, _) in resolved.items():
        if host:
            seen.setdefault((host, port), []).append(tid)
    for ids in seen.values():
        if len(ids) > 1:
            for tid in ids:
                resolved[tid] = (
                    None,
                    None,
                    f"ambiguous: shares its endpoint with {sorted(set(ids) - {tid})}",
                )
    return resolved


# ----------------------------------------------------------------------- scraping
class ScrapeError(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


_SLICE = 0.05


class _DeadlineSocket(socket.socket):
    """A socket whose connect, send and receive all honour an absolute deadline.

    A socket timeout alone restarts on each byte, so a peer that trickles one byte at a
    time, or a listener that never completes the handshake, can hold an operation past
    the deadline. ``deadline`` is a callable returning the current absolute monotonic
    deadline (it may tighten while an operation is waiting: when the collector is asked
    to stop); every wait is sliced so the deadline is re-read at least every
    ``_SLICE`` seconds.
    """

    deadline = None

    def _left(self):
        left = self.deadline() - time.monotonic()
        if left <= 0:
            raise TimeoutError("deadline")
        return min(left, _SLICE)

    def connect_by_deadline(self, address):
        self.setblocking(False)
        error = self.connect_ex(address)
        if error not in (errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY, 0):
            raise OSError(error, os.strerror(error))
        waiting = select.poll()
        waiting.register(self.fileno(), select.POLLOUT)
        while error:
            if waiting.poll(self._left() * 1000):
                error = self.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                if error:
                    raise OSError(error, os.strerror(error))
                break

    def sendall(self, data, flags=0):
        view = memoryview(data)
        while view:
            self.settimeout(self._left())
            try:
                sent = super().send(view, flags)
            except TimeoutError:
                continue
            view = view[sent:]

    def recv_into(self, buffer, nbytes=0, flags=0):
        while True:
            self.settimeout(self._left())
            try:
                return super().recv_into(buffer, nbytes, flags)
            except TimeoutError:
                continue


def _literal(host):
    """The ``ipaddress`` object of an IP literal, or None for anything else."""
    try:
        return ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return None


def _addresses(host, port):
    """The (family, address) of an IP literal. Names are never resolved: a resolver
    cannot be interrupted, so a lookup that blocks would leave a thread behind for every
    scrape; endpoints are IP literals, and anything else is refused up front."""
    ip = _literal(host)
    if ip is None:
        raise ScrapeError("hostname_unsupported")
    if ip.version == 4:
        return [(socket.AF_INET, (str(ip), port))]
    return [(socket.AF_INET6, (str(ip), port, 0, 0))]


def _connect(host, port, when):
    """A connected ``_DeadlineSocket`` to an IP literal, inside the deadline."""
    last = None
    for family, address in _addresses(host, port):
        sock = _DeadlineSocket(family, socket.SOCK_STREAM)
        sock.deadline = when
        try:
            sock.connect_by_deadline(address)
            return sock
        except TimeoutError:
            sock.close()
            raise
        except OSError as exc:
            sock.close()
            last = exc
    raise last or OSError("no address")


def fetch(host, port, path, deadline, limits: Limits):
    """The page's bytes, under a total deadline (monotonic) and the byte cap.

    ``deadline`` is an absolute monotonic time or a callable returning one. The
    connection, the request, the status line, the headers and every chunk of the body
    are all inside it: a slow or trickling peer, or a listener that never completes the
    handshake, cannot extend it. ``host`` must be an IP literal
    (``hostname_unsupported`` otherwise): no name is ever resolved, so no resolver
    work can outlive a scrape.
    """
    when = deadline if callable(deadline) else (lambda: deadline)
    if when() - time.monotonic() <= 0:
        raise ScrapeError("deadline_exhausted")
    connection = None
    try:
        sock = _connect(host, port, when)
        connection = http.client.HTTPConnection(host, port)
        connection.sock = sock
        connection.request("GET", path, headers={"Connection": "close"})
        response = connection.getresponse()
        if response.status != 200:
            raise ScrapeError(f"http_{response.status}")
        data = bytearray()
        declared = response.getheader("Content-Length")
        while True:
            chunk = response.read(min(65536, limits.max_response_bytes + 1 - len(data)))
            if not chunk:
                # A body that ends before its declared length is a cut page, never
                # a prefix to parse.
                if (
                    declared is not None
                    and declared.isdigit()
                    and len(data) != int(declared)
                ):
                    raise ScrapeError("truncated_response")
                return bytes(data)
            data.extend(chunk)
            if len(data) > limits.max_response_bytes:
                raise ScrapeError("response_too_large")
    except ScrapeError:
        raise
    except TimeoutError as exc:
        raise ScrapeError("scrape_deadline") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise ScrapeError(f"{type(exc).__name__}") from exc
    finally:
        if connection is not None:
            connection.close()


def parse(text, limits: Limits, source="tent"):
    """Series of the known families from a page: ({(metric, labels): value}, problems).

    ``labels`` is a sorted tuple of pairs. Bounded work: at most ``max_series``
    series lines are examined, lines are capped, labels and label values are capped;
    what a bound drops is counted in the problems, never silently.
    """
    series = {}
    problems = {"series_cap": 0, "line_too_long": 0, "label_cap": 0, "malformed": 0}
    seen = 0
    wanted = {k: v for families in SOURCES[source].values() for k, v in families.items()}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if len(line) > _MAX_LINE:
            problems["line_too_long"] += 1
            continue
        seen += 1
        if seen > limits.max_series:
            problems["series_cap"] += 1
            continue
        match = _SERIES.match(line)
        if match is None:
            problems["malformed"] += 1
            continue
        name, label_text, value = match.groups()
        if name not in wanted:
            continue
        labels = []
        ok = True
        position = 0
        label_text = label_text or ""
        while position < len(label_text) and ok:
            part = _LABEL.match(label_text, position)
            if part is None:
                ok = False
                break
            key, raw = part.group(1), part.group(2)
            if (
                len(labels) >= limits.max_labels
                or len(key) + len(raw) > limits.max_label_bytes
            ):
                problems["label_cap"] += 1
                ok = False
                break
            labels.append((key, raw))
            position = part.end()
        try:
            number = float(value)
        except ValueError:
            ok = False
        if not ok or not math.isfinite(number):
            problems["malformed"] += 1
            continue
        series[(name, tuple(sorted(labels)))] = number
    return series, problems


# --------------------------------------------------------------------- the capture
def _identity():
    """Boot ID, time and network namespaces; each is None when unreadable (a kernel
    without time namespaces shares the host's monotonic clock). The network
    namespace attributes a loopback endpoint's counters to the worker serving it."""
    identity = {"boot_id": None, "time_namespace": None, "net_namespace": None}
    with contextlib.suppress(OSError):
        identity["boot_id"] = (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        )
    with contextlib.suppress(OSError):
        identity["time_namespace"] = os.readlink("/proc/self/ns/time")
    with contextlib.suppress(OSError):
        identity["net_namespace"] = os.readlink("/proc/self/ns/net")
    return identity


class Capture:
    """The collector's JSONL file: metadata, records with unique sequences, a terminal
    record. Retained output is capped; a record past the cap is dropped and counted."""

    def __init__(self, directory: Path, run_id: str, limits: Limits, targets):
        self.limits = limits
        sources = sorted({t.source for t in targets}) or ["tent"]
        self.process_id = str(uuid4())
        self.directory = Path(directory) / run_id / self.process_id
        self.directory.mkdir(parents=True)
        self.path = self.directory / "events.jsonl"
        self.file = open(self.path, "wb", buffering=0)  # noqa: SIM115
        self.bytes = 0
        self.sequence = 0
        self.dropped = 0
        self.reserve = 4096
        self.run_id = run_id
        identity = _identity()
        self.write(
            {
                "type": "metadata",
                "schema": SCHEMA,
                "run_id": run_id,
                "process_id": self.process_id,
                "clock_id": str(uuid4()),
                "clock_source": "CLOCK_MONOTONIC",
                **identity,
                "pid": os.getpid(),
                "byteorder": sys.byteorder,
                "context": {
                    "role": "collector",
                    "connector": "mooncake" if "tent" in sources else "nixl",
                    "sources": sources,
                    "targets": [t.id for t in targets],
                    "limits": limits.__dict__,
                },
                "capabilities": {
                    "attached": False,
                    "transport_metrics": True,
                    "kinds": {
                        str(k): NAMES[k] for src in sources for k in SOURCES[src]
                    },
                    "families": sorted(
                        {k for src in sources for f in SOURCES[src].values() for k in f}
                    ),
                },
            },
            force=True,
        )

    def write(self, value, force=False) -> bool:
        line = (
            json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
        if (
            not force
            and self.bytes + len(line) > self.limits.max_output_bytes - self.reserve
        ):
            self.dropped += 1
            return False
        self.file.write(line)
        self.bytes += len(line)
        return True

    def event(self, kind, time_ns, **fields) -> bool:
        self.sequence += 1
        return self.write(
            {
                "schema": SCHEMA,
                "type": "event",
                "kind": NAMES[kind],
                "kind_id": kind,
                "sequence": self.sequence,
                "time_ns": time_ns,
                **fields,
            }
        )

    def end(self, complete, reasons, counters) -> None:
        stats = {
            "head": self.sequence,
            "tail": self.sequence,
            "drop_output_cap": self.dropped,
            **counters,
        }
        self.write(
            {
                "schema": SCHEMA,
                "type": "end",
                "complete": complete and not self.dropped,
                "reasons": sorted(
                    set(reasons) | ({"output_cap"} if self.dropped else set())
                ),
                "stats": stats,
            },
            force=True,
        )
        os.fsync(self.file.fileno())
        self.file.close()


class Collector:
    def __init__(
        self,
        limits: Limits,
        targets,
        capture: Capture,
        resolved=None,
        stop: threading.Event | None = None,
    ):
        self.limits = limits
        self.stop = stop
        self._stop_seen: float | None = None
        self.targets = targets
        self.capture = capture
        self.resolved = resolved if resolved is not None else resolve_targets(targets)
        self.samples = {t.id: 0 for t in targets}
        # Last value per series, bounded by ``max_history`` (least recently seen out
        # first); a series seen again after any of its target's were evicted has no
        # baseline, which its records say.
        self.last: OrderedDict = OrderedDict()
        self.evictions = {t.id: 0 for t in targets}
        self.evicted_total = 0
        self.generation = {t.id: 0 for t in targets}
        self.status = {t.id: "pending" for t in targets}
        self.reasons: set = set()
        # Series a NIXL exporter serves with labels this collector does not know (another
        # version): unavailable, counted, never an incomplete capture.
        self.unserved_label_series = 0
        self.partial_samples = 0
        self.total_samples = 0
        self.unavailable: dict = {}
        self.last_pass: dict | None = None
        self._sample_ids: dict = {}

    def hard_deadline(self) -> float:
        """No scrape of any kind outlives the shutdown budget that starts when stop
        is first seen; before that there is none."""
        if self.stop is not None and self.stop.is_set():
            if self._stop_seen is None:
                self._stop_seen = time.monotonic()
            return self._stop_seen + self.limits.shutdown_timeout_s
        return math.inf

    # ---- one scrape
    def scrape(self, target: Target, deadline) -> bool:
        """Scrape one target once and record both halves.

        Returns True when the sample is complete."""
        tid = target.id
        host, port, source = self.resolved[tid]
        self.samples[tid] += 1
        sample_id = f"{self.capture.process_id[:8]}-{tid}-{self.samples[tid]}"
        self._sample_ids[tid] = sample_id
        identity = {
            "id": tid,
            "role": target.role,
            "engine_id": target.engine_id,
            "rank": target.rank,
            "endpoint": None if host is None else f"{host}:{port}",
            "endpoint_source": source,
            "process_id": target.process_id,
            "pid": target.pid,
        }
        begin_ns = time.monotonic_ns()
        series, problems, reason, page = {}, {}, None, None
        if host is None:
            reason = "unavailable"
            self.unavailable[tid] = source
        else:
            try:
                page = fetch(
                    host,
                    port,
                    "/metrics",
                    lambda: min(deadline, self.hard_deadline()),
                    self.limits,
                )
                if not page.strip():
                    reason = "empty_page"
                else:
                    series, problems = parse(
                        page.decode("utf-8", "replace"), self.limits, target.source
                    )
            except ScrapeError as exc:
                reason = exc.reason
        end_ns = time.monotonic_ns()
        self.total_samples += 1
        capped = {k: v for k, v in problems.items() if v}
        if capped:
            self.reasons.update(capped)
        complete = reason is None and not capped
        base = {
            "sample_id": sample_id,
            "target": identity,
            "scrape": {"begin_ns": begin_ns, "end_ns": end_ns},
        }
        halves = tuple(
            (
                kind,
                families,
                self._partitions(tid, series, families, target.source)
                if reason is None
                else [],
            )
            for kind, families in SOURCES[target.source].items()
        )
        # One generation for the whole sample: every series of both kinds was
        # inspected first, and a discontinuity in any of them moves it once.
        if any(row[3] for _, _, rows in halves for row in rows):
            self.generation[tid] += 1
        generation = self.generation[tid]
        for kind, families, rows in halves:
            if not rows:
                # The half is missing: the page failed (partial) or it was served whole
                # without any series of this kind (unavailable, not zero).
                self.capture.event(
                    kind,
                    end_ns,
                    **base,
                    reset_generation=generation,
                    labels={},
                    fields={},
                    partial=reason is not None,
                    missing=sorted(v[0] for v in families.values()),
                    missing_reason=reason or "no_series_served",
                    discontinuity=False,
                )
                continue
            for labels, fields, units, discontinuity, expected, baseline in rows:
                self.capture.event(
                    kind,
                    end_ns,
                    **base,
                    reset_generation=generation,
                    labels=dict(labels),
                    fields=fields,
                    units=units,
                    partial=bool(capped),
                    missing=sorted(
                        expected - {k for k, x in fields.items() if x is not None}
                    ),
                    missing_reason=None,
                    discontinuity=discontinuity,
                    baseline=baseline,
                )
        self.status[tid] = "ok" if complete else (reason or "partial")
        if not complete:
            self.partial_samples += 1
        return complete

    def _partitions(self, tid, series, families, source="tent"):
        """Rows by label set for one kind.

        Each row is (labels, fields, units, discontinuity, expected, baseline);
        ``baseline`` is ``established`` (every series had a previous value),
        ``first`` (none had one and none was ever evicted) or ``unavailable`` (a
        previous value may have been evicted: a reset cannot be told apart).

        ``expected`` are the typed fields this label scope can carry: a (transport,
        operation) partition never carries a task failure, and no value is copied from
        one scope into another."""
        grouped = {}
        for (name, labels), value in series.items():
            if name not in families:
                continue
            field_name, unit, expected = families[name]
            names = {k for k, _ in labels}
            if not (set(expected) <= names <= set(expected) | set(_OPTIONAL_LABELS[source])):
                # A series with other labels than the source documents is not this
                # family's partition: never merged into one. For TENT that makes the
                # capture incomplete; a NIXL exporter of another version may name its
                # labels otherwise, which is data this collector cannot read (counted,
                # the fields stay unavailable), not a failure.
                if source == "nixl":
                    self.unserved_label_series += 1
                else:
                    self.reasons.add("unexpected_labels")
                continue
            grouped.setdefault((tuple(sorted(expected)), labels), {})[name] = (
                field_name,
                unit,
                value,
            )
        rows = []
        for (scope, labels), metrics in sorted(grouped.items()):
            fields, units, discontinuity = {}, {}, False
            baseline = "established"
            carried = {v[0] for v in families.values() if tuple(sorted(v[2])) == scope}
            for name, (field_name, unit, value) in metrics.items():
                fields[field_name] = value
                units[field_name] = unit
                key = (tid, name, labels)
                previous = self.last.get(key)
                if previous is None:
                    if baseline == "established":
                        baseline = "unavailable" if self.evictions[tid] else "first"
                elif name not in _GAUGES and value < previous:
                    discontinuity = True
                self.last[key] = value
                self.last.move_to_end(key)
            rows.append((labels, fields, units, discontinuity, carried, baseline))
        self._bound_history()
        return rows

    def _bound_history(self) -> None:
        while len(self.last) > self.limits.max_history:
            (tid, _, _), _ = self.last.popitem(last=False)
            self.evictions[tid] += 1
            self.evicted_total += 1
            self.reasons.add("history_evicted")

    def scrape_all(self, budget_s) -> bool:
        """One pass over every target inside ``budget_s`` seconds in total."""
        deadline = min(time.monotonic() + budget_s, self.hard_deadline())
        complete = True
        begin_ns = time.monotonic_ns()
        samples, target_complete = {}, {}
        for index, target in enumerate(self.targets):
            deadline = min(deadline, self.hard_deadline())
            left = deadline - time.monotonic()
            if left <= 0:
                complete = False
                self.reasons.add("pass_budget_exhausted")
                self.status[target.id] = "not_scraped"
                samples[target.id], target_complete[target.id] = None, False
                continue
            share = left / (len(self.targets) - index)
            ok = self.scrape(
                target, time.monotonic() + min(self.limits.scrape_timeout_s, share)
            )
            complete &= ok
            samples[target.id] = self._sample_ids[target.id]
            target_complete[target.id] = ok
        self.last_pass = {
            "begin_ns": begin_ns,
            "end_ns": time.monotonic_ns(),
            "complete": complete,
            "samples": samples,
            "targets_complete": target_complete,
        }
        return complete

    def answer(self, requests) -> None:
        """Append one boundary marker per request the last pass answers: a request
        noticed before the pass began names that pass's samples (the sample IDs,
        the pass's own begin and end instants and each target's completeness). The
        capture is append-only: a marker is a record of its own, nothing already
        written is changed."""
        last = self.last_pass
        if last is None:
            return
        for request in requests:
            if request["detected_ns"] >= last["begin_ns"]:
                continue
            self.capture.write(
                {
                    "schema": SCHEMA,
                    "type": "boundary",
                    "name": request["name"],
                    "request_id": request["id"],
                    "detected_ns": request["detected_ns"],
                    "pass": {
                        k: last[k]
                        for k in ("begin_ns", "end_ns", "complete", "targets_complete")
                    },
                    "samples": last["samples"],
                },
                force=True,
            )
            request["answered"] = True


_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_REQUEST_NAME = re.compile(r"^[a-z_]{1,32}$")


class Boundaries:
    """Boundary requests an orchestrator drops as ``<id>.json`` (``{"name": ...}``) in a
    directory: the collector notices each once, scrapes a pass that begins after it,
    and answers with a marker record naming that pass's samples. An invalid request
    is ignored and counted; a request is answered once."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.requests: dict[str, dict] = {}
        self.invalid = 0
        self._bad: set = set()

    def poll(self) -> bool:
        """Notice new requests; True when one arrived."""
        found = False
        try:
            names = sorted(os.listdir(self.directory))
        except OSError:
            return False
        for name in names:
            stem, dot, suffix = name.rpartition(".")
            if suffix != "json" or stem in self.requests or name in self._bad:
                continue
            try:
                text = (self.directory / name).read_text()
                value = json.loads(text)
                if (
                    not _REQUEST_ID.match(stem)
                    or not isinstance(value, dict)
                    or not _REQUEST_NAME.match(str(value.get("name", "")))
                ):
                    raise ValueError("invalid request")
            except (OSError, ValueError):
                # A half-written file is retried on the next poll; a bad one counted once.
                if (self.directory / name).exists() and name not in self._bad:
                    self._bad.add(name)
                    self.invalid += 1
                continue
            self.requests[stem] = {
                "id": stem,
                "name": value["name"],
                "detected_ns": time.monotonic_ns(),
                "answered": False,
            }
            found = True
        return found

    def pending(self):
        return [r for r in self.requests.values() if not r["answered"]]


def _wait(stop, interval, boundaries):
    """Wait ``interval`` seconds or until ``stop`` is set or a boundary request arrives;
    True when stop is set."""
    deadline = time.monotonic() + interval
    while True:
        left = deadline - time.monotonic()
        if stop.is_set():
            return True
        if boundaries is not None and boundaries.poll():
            return False
        if left <= 0:
            return False
        stop.wait(min(left, 0.05 if boundaries is not None else left))


def run(
    limits: Limits,
    targets,
    output_dir,
    run_id,
    stop: threading.Event,
    *,
    final_scrape=True,
    boundaries: Boundaries | None = None,
):
    """The collector's whole life. Returns the capture's terminal summary.

    An initial scrape at start (bounded by ``startup_timeout_s``), periodic scrapes,
    when ``stop`` is set, one final scrape inside ``shutdown_timeout_s`` followed by
    the terminal record. A bound that is hit is recorded; nothing here raises into the
    caller for a target's failure.
    """
    UUID(run_id)
    capture = Capture(output_dir, run_id, limits, targets)
    collector = Collector(limits, targets, capture, stop=stop)
    complete = True
    reasons = set()
    try:
        if boundaries is not None:
            boundaries.poll()
        complete &= collector.scrape_all(limits.startup_timeout_s)
        if boundaries is not None:
            collector.answer(boundaries.pending())
        while not _wait(stop, limits.interval_s, boundaries):
            complete &= collector.scrape_all(
                limits.interval_s + limits.scrape_timeout_s
            )
            if boundaries is not None:
                collector.answer(boundaries.pending())
        if final_scrape:
            collector.hard_deadline()
            final = collector.scrape_all(limits.shutdown_timeout_s)
            complete &= final
            if not final:
                reasons.add("final_scrape_incomplete")
        else:
            complete = False
            reasons.add("no_final_scrape")
    except BaseException:
        complete = False
        reasons.add("collector_error")
        raise
    finally:
        reasons |= collector.reasons
        complete = complete and not reasons and not collector.unavailable
        counters = {
            "samples": collector.total_samples,
            "partial_samples": collector.partial_samples,
            "unavailable_targets": len(collector.unavailable),
            "history_series": len(collector.last),
            "history_evicted": collector.evicted_total,
            "unserved_label_series": collector.unserved_label_series,
        }
        capture.end(
            complete,
            reasons | ({"target_unavailable"} if collector.unavailable else set()),
            counters,
        )
        summary = {
            "connector": "mooncake",
            "role": "collector",
            "observation_complete": bool(complete and not capture.dropped),
            "reasons": sorted(reasons),
            "unavailable": collector.unavailable,
            "status": collector.status,
            "limits": limits.__dict__,
            "retained_output_bytes": capture.bytes,
            "dropped_records": capture.dropped,
            **counters,
        }
        with contextlib.suppress(OSError):
            (capture.directory / "profiler.json").write_text(json.dumps(summary))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--stop-file", help="stop (final scrape, then exit) when this file exists"
    )
    parser.add_argument("--no-final-scrape", action="store_true")
    parser.add_argument(
        "--boundary-dir",
        help="answer boundary requests (<id>.json) dropped in this directory",
    )
    for name, field_ in Limits.__dataclass_fields__.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=type(field_.default),
            default=field_.default,
        )
    args = parser.parse_args(argv)
    limits = Limits(
        **{name: getattr(args, name) for name in Limits.__dataclass_fields__}
    )
    targets = load_manifest(args.manifest, limits)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    if args.stop_file:

        def watch():
            while not stop.is_set():
                if os.path.exists(args.stop_file):
                    stop.set()
                    return
                time.sleep(min(0.2, limits.interval_s))

        threading.Thread(target=watch, daemon=True).start()
    summary = run(
        limits,
        targets,
        args.output_dir,
        args.run_id,
        stop,
        final_scrape=not args.no_final_scrape,
        boundaries=Boundaries(args.boundary_dir) if args.boundary_dir else None,
    )
    print(json.dumps(summary))
    return 0 if summary["observation_complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
