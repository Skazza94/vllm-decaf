# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in portable TCP event collector; no import-time privilege or kernel probes."""

import collections
import contextlib
import hashlib
import ipaddress
import json
import os
import re
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

RECORD_BYTES = 2048
EVENTS = {
    "tcp_retransmit_skb",
    "tcp_send_reset",
    "tcp_receive_reset",
    "inet_sock_set_state",
}
FIELDS = {
    "srtt_us",
    "snd_cwnd",
    "snd_ssthresh",
    "packets_out",
    "retrans_out",
    "lost_out",
    "total_retrans",
    "mss_cache",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Settings:
    enabled: bool = False
    privileged_opt_in: bool = False
    mode: str | None = None
    artifacts: str | None = None
    # Optional narrowing; the selection itself is the sockets the endpoint's
    # scheduler and worker processes own (discovery), never an address.
    local_ports: tuple = ()
    peer_ports: tuple = ()

    def __post_init__(self):
        for name in ("enabled", "privileged_opt_in"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"ebpf.{name} must be boolean")
        if self.enabled and self.mode != "tcp_tracepoints":
            raise ValueError("ebpf.mode must explicitly name tcp_tracepoints")
        if self.artifacts is not None and (
            not isinstance(self.artifacts, str)
            or not Path(self.artifacts).is_absolute()
        ):
            raise ValueError("ebpf.artifacts requires an absolute path")
        for name in ("local_ports", "peer_ports"):
            ports = getattr(self, name)
            if (
                not isinstance(ports, (tuple, list))
                or len(ports) > 32
                or any(type(p) is not int or not 0 < p <= 65535 for p in ports)
                or len(set(ports)) != len(ports)
            ):
                raise ValueError(f"ebpf.{name} requires up to 32 distinct ports")


def effective_privilege():
    """Never grants capabilities, changes sysctls or raises resource limits."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("CapEff:"):
            caps = int(line.split()[1], 16)
            return bool(caps & (1 << 21) or (caps & (1 << 38) and caps & (1 << 39)))
    return False


def read_offsets(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        name, seconds, nanos = line.split()
        if name not in ("monotonic", "boottime") or name in result:
            raise ValueError("unknown/duplicate clock offset")
        if not re.fullmatch(r"-?[0-9]+", seconds) or not nanos.isdecimal():
            raise ValueError("invalid clock offset")
        ns = int(nanos)
        if ns >= 1_000_000_000:
            raise ValueError("nanosecond offset out of range")
        value = int(seconds) * 1_000_000_000 + ns
        if not -(1 << 63) <= value < 1 << 63:
            raise ValueError("clock offset overflow")
        result[name] = value
    if set(result) != {"monotonic", "boottime"}:
        raise ValueError("both clock offsets required")
    return result


def convert_time(value, offset):
    if type(value) is not int or type(offset) is not int or value < 0:
        raise ValueError("integer clock domains required")
    converted = value + offset
    if not 0 <= converted < 1 << 63:
        raise ValueError("clock conversion overflow")
    return converted


def decode_row(line):
    if len(line) + 1 > RECORD_BYTES:
        raise ValueError("native record bound")
    row = json.loads(line)
    if not isinstance(row, dict) or row.get("type") not in {
        "tcp_event",
        "loss",
        "ready",
        "end",
        "status",
        "resource",
        "ack",
        "rejected",
    }:
        raise ValueError("unknown native row")
    if ("exact" in row and row["exact"] is not False) or any(
        key in row for key in ("request_ref", "handoff_ref", "transfer", "transfer_id")
    ):
        raise ValueError("native observation claimed a transfer")
    if row["type"] in ("ack", "rejected"):
        for key, value in row.items():
            if key in ("type", "op", "ok", "stale", "harvested", "consistent"):
                continue
            if value is not None and (
                type(value) is not int or not 0 <= value < 1 << 64
            ):
                raise ValueError(f"invalid native {row['type']} field {key}")
        if row["type"] == "ack" and (
            row.get("op") not in ("A", "D", "R")
            or type(row.get("ok")) is not bool
            or any(
                key in row and type(row[key]) is not bool
                for key in ("stale", "harvested", "consistent")
            )
        ):
            raise ValueError("invalid native acknowledgement")
        return row
    if row["type"] == "tcp_event":
        if row.get("schema") != 3 or row.get("event") not in EVENTS:
            raise ValueError("unknown TCP event")
        for key in ("cookie", "admission"):
            if type(row.get(key)) is not int or not 0 < row[key] < 1 << 64:
                raise ValueError(f"invalid TCP {key}")
        for key in ("timestamp_ns", "netns"):
            if type(row.get(key)) is not int or not 0 < row[key] < 1 << 64:
                raise ValueError("invalid TCP integer")
        family = row.get("family")
        if type(family) is not int or family not in (2, 10):
            raise ValueError("invalid TCP address family")
        for key in ("local_ip", "peer_ip"):
            address = ipaddress.ip_address(row[key])
            if address.version != (4 if family == 2 else 6):
                raise ValueError("TCP address family mismatch")
        for key in ("local_port", "peer_port"):
            if type(row.get(key)) is not int or not 0 <= row[key] <= 65535:
                raise ValueError("invalid TCP port")
        for key in ("old_state", "new_state"):
            value = row.get(key)
            if row["event"] == "inet_sock_set_state":
                if type(value) is not int or not 0 <= value <= 255:
                    raise ValueError("missing TCP transition state")
            elif value is not None:
                raise ValueError("unexpected TCP transition state")
        if not isinstance(row.get("snapshot"), dict) or set(row["snapshot"]) != FIELDS:
            raise ValueError("invalid TCP snapshot")
        for field in row["snapshot"].values():
            if not isinstance(field, dict):
                raise ValueError("invalid TCP snapshot field")
            state, value = field.get("state"), field.get("value")
            if state == "available":
                if type(value) is not int or not 0 <= value < 1 << 32:
                    raise ValueError("invalid TCP snapshot value")
            elif state not in ("unavailable", "read_error") or value is not None:
                raise ValueError("invalid TCP snapshot availability")
        if row.get("srtt_encoding") != "usec_shift_3":
            raise ValueError("TCP srtt encoding missing")
    return row


# The built package's schema: the observer and the collector it launches are one
# generation (4: owned sockets by cookie, sequenced commands); any other package
# is rebuilt.
PACKAGE_SCHEMA = 4


def validate_artifacts(directory):
    """Check a package against its own manifest, never against its build host."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {"collector", "observe.bpf.o", "limits.json"}
    if isinstance(manifest, dict) and manifest.get("schema") != PACKAGE_SCHEMA:
        raise ValueError(
            f"eBPF package schema {manifest.get('schema')!r}, not {PACKAGE_SCHEMA}: "
            "rebuild it with tools/kv-profiler-ebpf/build.sh"
        )
    if (
        not isinstance(manifest, dict)
        or manifest.get("mode") != "tcp_tracepoints"
        or set(manifest.get("files", {})) != expected
    ):
        raise ValueError("incomplete TCP artifact manifest")
    for name in expected:
        if digest(root / name) != manifest["files"][name]:
            raise ValueError("artifact hash mismatch: " + name)
    return manifest


# The collector's loss scopes that lose endpoint events (or invalidate the
# counters): unlike filter and global unknown exclusions, they make a capture
# degraded.
LOSS_SCOPES = ("endpoint_namespace", "counter_health")


def event_loss(losses, terminal):
    """Known event loss, by counter: the cumulative counters of ``LOSS_SCOPES``
    (``losses``: (scope, counter) -> highest value) and the collector's terminal
    output drops; zero counts and exclusions are not loss. A malformed row is
    no loss but a collector failure (it exits 3), so a run with one is never
    clean."""
    loss = {
        f"{scope}.{counter}": value
        for (scope, counter), value in sorted(losses.items())
        if value and scope in LOSS_SCOPES
    }
    value = (terminal or {}).get("output_drops")
    if isinstance(value, int) and value:
        loss["terminal.output_drops"] = value
    return loss


class Channel:
    """Bounded pipes for an already-owned child; never enables kernel probes."""

    def __init__(self, child, directory, settings=None, inventory=()):
        self.settings = settings if settings is not None else Settings()
        self.inventory = inventory
        self.directory = Path(directory)
        self.child = child
        self.selector = None
        self.buffers = {}
        self.stderr_bytes = 0
        self.final_usage = None
        self.reaped = False
        self.accounting_error = None
        self.terminal = None
        self.losses = {}  # (scope, counter) -> highest cumulative value
        self.monotonic_offset = None
        self.time_namespace = None
        self.namespaces = None
        self.started = False
        self.input_closed = False
        self.input_error = None
        self.stopped_result = None
        self.failed_reason = None
        self.error = None
        self.cpu_s = 0.0
        self.rss_hwm_kib = 0
        # ownership commands: queued, sent (by sequence), and what the kernel
        # holds as far as acknowledgements establish it
        self.sequence = 0
        self.outbox: collections.deque = collections.deque()
        self.pending: dict[int, dict] = {}
        self.inflight: dict[int, int] = {}  # cookie -> its command's sequence
        self.desired: dict[int, int | None] = {}  # cookie -> decided admission
        self.decided: dict[int, dict] = {}  # admission -> its decided interval
        self.kernel: dict[int, object] = {}  # cookie -> held admission|None|UNKNOWN
        self.windows: dict[int, dict] = {}  # admission -> its kernel interval
        self.unknown: dict[int, list] = {}  # cookie -> [[since, until|None], ...]
        self.attempts: dict[int, int] = {}
        self.unresolved: set[int] = set()
        self.ack_failures: list[dict] = []
        self.late_acknowledgements = 0
        self.expired: set[int] = set()
        self.unacknowledged: dict[int, dict] = {}  # cookie -> its lost admit
        self.rejected: dict[int, dict] = {}  # cookie -> final counter (terminal)
        self.cookie_admissions: dict[int, list] = {}  # cookie -> its windows
        self.cookie_decided: dict[int, list] = {}  # cookie -> its decisions
        self.harvests: dict[int, list] = {}  # cookie -> its counter's harvests
        self.status = {
            "available": False,
            "state": "inactive",
            "exact": False,
            "mode": self.settings.mode,
            "transfer_refs": "unbound",
        }
        if child is not None:
            self.open_pipes()

    def open_pipes(self):
        if (
            self.child is None
            or self.child.pid <= 0
            or any(
                stream is None
                for stream in (self.child.stdin, self.child.stdout, self.child.stderr)
            )
        ):
            raise ValueError("owned child with three pipes required")
        if self.namespaces is None:
            self.namespaces = {"net": os.readlink("/proc/self/ns/net")}
            # Time namespaces are optional kernel metadata, not load eligibility.
            with contextlib.suppress(OSError):
                self.time_namespace = os.readlink("/proc/self/ns/time")
                self.namespaces["time"] = self.time_namespace
            # Without conversion, retain raw kernel time and leave joins unaligned.
            with contextlib.suppress(OSError, ValueError):
                self.monotonic_offset = read_offsets("/proc/self/timens_offsets")[
                    "monotonic"
                ]
        os.set_blocking(self.child.stdin.fileno(), False)
        self.selector = selectors.DefaultSelector()
        for name, stream in (
            ("stdout", self.child.stdout),
            ("stderr", self.child.stderr),
        ):
            os.set_blocking(stream.fileno(), False)
            self.selector.register(stream, selectors.EVENT_READ, name)
            self.buffers[name] = bytearray()
        self.started = True
        self.status["state"] = "starting"

    def running(self):
        """Reap only this child with wait4, retaining its final usage and status."""
        if not self.child or self.reaped:
            return False
        if self.child.returncode is not None:
            self.reaped = True
            self.accounting_error = "owned child reaped outside accounting"
            return False
        try:
            pid, status, usage = os.wait4(self.child.pid, os.WNOHANG)
        except ChildProcessError:
            self.reaped = True
            self.accounting_error = "owned child final accounting unavailable"
            return False
        if pid:
            self.reaped = True
            self.child.returncode = os.waitstatus_to_exitcode(status)
            self.cpu_s = usage.ru_utime + usage.ru_stime
            self.rss_hwm_kib = usage.ru_maxrss
            self.final_usage = {
                "cpu_s": self.cpu_s,
                "rss_hwm_kib": self.rss_hwm_kib,
            }
            return False
        return True

    def wait_until(self, deadline):
        while self.running():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired("owned native child", 2)
            time.sleep(min(0.001, remaining))

    def _resources(self):
        if not self.child or self.reaped:
            return
        try:
            status = (Path("/proc") / str(self.child.pid) / "status").read_text()
            rss = next(
                int(row.split()[1])
                for row in status.splitlines()
                if row.startswith("VmHWM:")
            )
            self.rss_hwm_kib = max(self.rss_hwm_kib, rss)
            raw = (
                (Path("/proc") / str(self.child.pid) / "stat")
                .read_text()
                .rsplit(")", 1)[1]
                .split()
            )
            self.cpu_s = max(
                self.cpu_s, (int(raw[11]) + int(raw[12])) / os.sysconf("SC_CLK_TCK")
            )
        except (OSError, ValueError, StopIteration):
            pass  # End accounting is collected by wait4, not inferred as zero.

    def command(self, value):
        if len(value) > 128:
            raise ValueError("native command bound")
        try:
            count = os.write(self.child.stdin.fileno(), value)
        except BlockingIOError:
            return False
        if count != len(value):
            raise ValueError("partial native command")
        return True

    def send_command(self, value):
        if self.input_closed:
            return False
        try:
            return self.command(value)
        except (BrokenPipeError, ConnectionResetError) as error:
            self.input_closed = True
            self.input_error = str(error)
            return False

    def fail(self, reason):
        if self.failed_reason is None:
            self.failed_reason = reason
        self.status.update(
            outcome="FAIL", reason=self.failed_reason, available=False, state="failed"
        )

    # Ownership. Discovery decides which admission each cookie should be under
    # (``desired``, with each admission's decided interval); the channel
    # reconciles the kernel map to it, one command per cookie in flight, and
    # keeps what acknowledgements establish the kernel holds (``kernel``, with
    # each admission's kernel interval). A cookie whose kernel state is
    # unknown is read back; one that does not converge within a bound of
    # attempts is unresolved.

    def admit(self, cookie: int, admission: int, at: int | None = None) -> None:
        """Discovery's decision, made at ``at`` (its scan's time): ``cookie``
        is owned under ``admission`` (a replacement of any earlier admission
        of it)."""
        now = time.monotonic_ns() if at is None else at
        previous = self.desired.get(cookie)
        if previous == admission:
            return
        if previous is not None:
            self.decided[previous]["until"] = now
        self.decided[admission] = {"cookie": cookie, "since": now, "until": None}
        self.cookie_decided.setdefault(cookie, []).append(admission)
        self.desired[cookie] = admission
        self.attempts[cookie] = 0
        self.reconcile(cookie)

    def retire(self, cookie: int, admission: int, at: int | None = None) -> None:
        """Discovery's decision, made at ``at`` (its scan's time): ``cookie``
        is no longer owned under ``admission`` (a later admission of it is
        unaffected)."""
        if self.desired.get(cookie) != admission:
            return
        self.decided[admission]["until"] = time.monotonic_ns() if at is None else at
        self.desired[cookie] = None
        self.attempts[cookie] = 0
        self.reconcile(cookie)

    def reconcile(self, cookie) -> None:
        """Queue the one command that moves the cookie's kernel entry toward
        the decision, unless one is in flight or the cookie has converged."""
        if cookie in self.inflight or cookie not in self.desired:
            return
        held, wanted = self.kernel.get(cookie), self.desired[cookie]
        if held == wanted:
            self.attempts.pop(cookie, None)
            self.unresolved.discard(cookie)
            return
        self.attempts[cookie] = self.attempts.get(cookie, 0) + 1
        if self.attempts[cookie] > OWNERSHIP_ATTEMPTS:
            if cookie not in self.unresolved:
                self.unresolved.add(cookie)
                self.ack_failures.append(
                    {"cookie": cookie, "reason": "unresolved after retries"}
                )
            return
        if held is UNKNOWN:
            self._queue("R", cookie)
        elif wanted is not None:
            self._queue("A", cookie, wanted)
        else:
            self._queue("D", cookie, held)

    def _queue(self, op, cookie, admission=None) -> None:
        self.sequence += 1
        self.inflight[cookie] = self.sequence
        self.outbox.append({"op": op, "seq": self.sequence, "cookie": cookie,
                            "admission": admission,
                            "queued_ns": time.monotonic_ns()})  # fmt: skip
        self.pump()

    def pump(self) -> None:
        """Send what the pipe takes, in order."""
        while self.outbox and not self.input_closed and self.child is not None:
            item = self.outbox[0]
            text = f"{item['op']} {item['seq']} {item['cookie']}"
            if item["admission"] is not None:
                text += f" {item['admission']}"
            if not self.send_command((text + "\n").encode()):
                break
            self.outbox.popleft()
            self.pending[item["seq"]] = {**item, "sent_ns": time.monotonic_ns()}

    def expire(self, timeout_ns=None) -> None:
        """Commands unacknowledged for ACK_TIMEOUT_NS, once every reply that was
        readable has been read: the cookie's kernel state is unknown from when
        the command was sent until a readback establishes it."""
        now = time.monotonic_ns()
        timeout_ns = ACK_TIMEOUT_NS if timeout_ns is None else timeout_ns
        for seq, sent in list(self.pending.items()):
            if now - sent["sent_ns"] < timeout_ns:
                continue
            del self.pending[seq]
            self.expired.add(seq)
            cookie = sent["cookie"]
            self.ack_failures.append({**sent, "reason": "no acknowledgement"})
            if sent["op"] == "A":
                self.unacknowledged[cookie] = sent  # applied later, perhaps
            if self.inflight.get(cookie) == seq:
                del self.inflight[cookie]
            self._uncertain(cookie, sent["sent_ns"])
            self.reconcile(cookie)

    def _uncertain(self, cookie, since) -> None:
        self.kernel[cookie] = UNKNOWN
        intervals = self.unknown.setdefault(cookie, [])
        if not intervals or intervals[-1][1] is not None:
            intervals.append([since, None])

    def _settle(self, cookie, at) -> None:
        intervals = self.unknown.get(cookie)
        if intervals and intervals[-1][1] is None:
            intervals[-1][1] = at

    def _window(self, cookie, admission, since) -> None:
        self.windows[admission] = {"cookie": cookie, "since": since, "until": None}
        self.cookie_admissions.setdefault(cookie, []).append(admission)

    def _close_others(self, cookie, admission, at) -> None:
        for other in self.cookie_admissions.get(cookie, ()):
            window = self.windows[other]
            if other != admission and window["until"] is None:
                window["until"] = at

    def _harvested(self, row) -> None:
        """The cookie's rejection counter, read and deleted by an admit or a
        delete: each rejection in one harvest, or in the terminal dump."""
        self.harvests.setdefault(row["cookie"], []).append(
            {
                "op": row["op"],
                "count": row.get("rejected", 0),
                "errno": row.get("rejected_errno", 0),
                "before_ns": row["before_ns"],
                "applied_ns": row["applied_ns"],
            }
        )

    def acknowledged(self, row) -> None:
        """Apply one acknowledgement to what the kernel is known to hold, then
        reconcile the cookie. A reply to an expired command is too late: the
        readback it caused establishes the state."""
        seq, cookie, admission = row.get("seq"), row.get("cookie"), row.get("admission")
        sent = self.pending.pop(seq, None)
        if sent is None and seq in self.expired:
            self.late_acknowledgements += 1
            return
        if self.inflight.get(cookie) == seq:
            del self.inflight[cookie]
        if row["op"] == "A":
            if row["ok"]:
                self.kernel[cookie] = admission
                self._window(cookie, admission, row["before_ns"])
                self._close_others(cookie, admission, row["before_ns"])
                self._harvested(row)
            else:  # an update that failed changed nothing: retried
                self.ack_failures.append({**(sent or row), "reason": "admit failed",
                                          "errno": row.get("errno")})  # fmt: skip
        elif row["op"] == "D":
            if row.get("harvested") is True:
                self._harvested(row)  # harvested, even if the delete then failed
            if row["ok"] and not row.get("stale"):
                self.kernel[cookie] = None
                window = self.windows.get(admission)
                if window is not None and window["until"] is None:
                    window["until"] = row["applied_ns"]
            else:
                # a failed delete, or the kernel held something else: unknown
                # from when the delete began until read back
                self.ack_failures.append(
                    {**(sent or row), "errno": row.get("errno"),
                     "reason": "delete failed" if not row["ok"] else "stale delete"}
                )  # fmt: skip
                self._uncertain(cookie, row["before_ns"])
        elif row["op"] == "R":
            if row["ok"]:
                held = row.get("admission")
                self.kernel[cookie] = held
                self._close_others(cookie, held, row["applied_ns"])
                lost = self.unacknowledged.pop(cookie, None)
                if held is not None and held not in self.windows:
                    # a lost admit found applied: in force since it was sent at
                    # the earliest (the interval up to now stays unknown)
                    since = (
                        lost["sent_ns"]
                        if lost is not None and lost["admission"] == held
                        else row["applied_ns"]
                    )
                    self._window(cookie, held, since)
                self._settle(cookie, row["applied_ns"])
            else:
                self.ack_failures.append({**(sent or row), "reason": "readback failed",
                                          "errno": row.get("errno")})  # fmt: skip
        self.reconcile(cookie)

    def unresolved_ownership(self) -> dict:
        """Cookies whose kernel entry is not known to match the decision: a
        command unsent or unacknowledged, unknown state, or given up."""
        diverged = sorted(
            c for c, wanted in self.desired.items()
            if c in self.inflight or self.kernel.get(c) != wanted
            or c in self.unresolved
        )  # fmt: skip
        return {"cookies": len(diverged), "unsent": len(self.outbox),
                "unacknowledged": len(self.pending),
                "given_up": len(self.unresolved)}  # fmt: skip

    def verify(self, row, event_ns) -> dict:
        """Whether an event is its socket's while owned: at the event's time
        (observer monotonic) discovery had decided that admission, the kernel
        held it as acknowledged, no change of the cookie was unresolved, and
        its kernel state was not unknown."""
        cookie, admission = row["cookie"], row["admission"]
        if event_ns is None:
            return {"verified": False, "reason": "event time not convertible"}
        decided = self.decided.get(admission)
        if decided is None or decided["cookie"] != cookie:
            return {"verified": False, "reason": "admission not decided"}
        if event_ns < decided["since"] or (
            decided["until"] is not None and event_ns >= decided["until"]
        ):
            return {"verified": False, "reason": "outside the owned interval"}
        window = self.windows.get(admission)
        if window is None or window["cookie"] != cookie:
            return {"verified": False, "reason": "admission not acknowledged"}
        if event_ns < window["since"] or (
            window["until"] is not None and event_ns > window["until"]
        ):
            return {"verified": False, "reason": "outside the admission's interval"}
        inflight = self.inflight.get(cookie)
        if inflight is not None and any(
            item["seq"] == inflight and item["queued_ns"] <= event_ns
            for item in (*self.outbox, *self.pending.values())
        ):
            return {"verified": False, "reason": "change unresolved"}
        for since, until in self.unknown.get(cookie, ()):
            if since <= event_ns and (until is None or event_ns <= until):
                return {"verified": False, "reason": "kernel state unknown"}
        return {"verified": True}

    def poll(self, writer, timeout=0):
        if not self.child or not self.selector:
            return []
        self.pump()
        self._resources()
        rows: list[dict] = []
        drained = self._read(rows, timeout)
        while drained and len(rows) < 2048 and self.selector.get_map():
            drained = self._read(rows, 0)
        # every reply readable now has been read: only then can one be late (a
        # backlog still unread defers it, up to twice the timeout)
        self.expire(ACK_TIMEOUT_NS if not drained else 2 * ACK_TIMEOUT_NS)
        self.pump()
        return rows

    def _read(self, rows, timeout) -> bool:
        """One read of each ready pipe; whether anything was ready."""
        if any(
            os.readlink("/proc/self/ns/" + name) != value
            for name, value in self.namespaces.items()
        ):
            raise ValueError("observer namespace changed")
        ready = self.selector.select(timeout)
        for key, _ in sorted(ready, key=lambda item: item[0].data == "stdout"):
            data = os.read(key.fileobj.fileno(), 65536)
            if not data:
                self.selector.unregister(key.fileobj)
                if key.data == "stdout" and self.buffers["stdout"]:
                    raise ValueError("truncated native record")
                continue
            if key.data == "stderr":
                self.stderr_bytes += len(data)
                if self.stderr_bytes > 1 << 20:
                    raise ValueError("native stderr limit")
                with (self.directory / "native.stderr").open("ab") as stream:
                    stream.write(data)
                continue
            buffer = self.buffers[key.data]
            buffer.extend(data)
            if len(buffer) > 1 << 20:
                raise ValueError("native output limit")
            while b"\n" in buffer:
                line, _, remainder = buffer.partition(b"\n")
                buffer[:] = remainder
                row = decode_row(line)
                if len(rows) >= 4096:
                    raise ValueError("native queue bound")
                if row.get("type") == "ready":
                    self.status.update(
                        available=True,
                        state="partial",
                        mode=self.settings.mode,
                        kernel_miss_visibility="unverified",
                    )
                elif row.get("type") == "status":
                    self.status.update(
                        row,
                        state="failed"
                        if row.get("outcome") == "FAIL"
                        else "unavailable",
                    )
                elif row.get("type") == "resource":
                    self.status["resource"] = row
                elif row.get("type") == "loss" and isinstance(row.get("value"), int):
                    # cumulative counters: the highest value of each is its total
                    key = (row.get("scope"), row.get("counter"))
                    self.losses[key] = max(self.losses.get(key, 0), row["value"])
                elif row.get("type") == "ack":
                    self.acknowledged(row)
                elif row.get("type") == "rejected":
                    # the terminal counter of a cookie: each rejection once
                    self.rejected[row["cookie"]] = row
                elif row.get("type") == "end":
                    self.terminal = row
                    if row.get("failed") is True:
                        self.fail("native_terminal_failure")
                if row.get("outcome") == "FAIL":
                    self.fail(self.status.get("reason", "native_collector_failure"))
                elif self.failed_reason is not None:
                    self.fail(self.failed_reason)
                rows.append(row)
            if len(buffer) >= RECORD_BYTES:
                raise ValueError("unterminated native record bound")
        return bool(ready)

    def stop(self, writer, consume=None):
        if self.stopped_result is not None:
            return self.stopped_result
        if not self.child:
            return {**self.status, "complete": not self.settings.enabled}
        deadline = time.monotonic() + 2
        stop_deadline = deadline - 0.25

        def drain(timeout):
            try:
                rows = self.poll(writer, timeout)
                for row in rows:
                    if consume:
                        consume(row)
                    else:
                        with (self.directory / "shutdown.jsonl").open("a") as stream:
                            stream.write(json.dumps(row) + "\n")
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self.error = str(error)
                self.fail("native_output_incomplete")

        try:
            # the ownership commands in flight first (bounded), ahead of stop
            settle = time.monotonic() + OWNERSHIP_DRAIN_S
            while (
                (self.outbox or self.pending)
                and self.running()
                and not self.input_closed
                and time.monotonic() < settle
            ):
                drain(min(0.01, max(0, settle - time.monotonic())))
            stop_sent = False
            while self.running() and time.monotonic() < stop_deadline:
                if not stop_sent:
                    stop_sent = self.send_command(b"stop\n")
                drain(min(0.01, max(0, stop_deadline - time.monotonic())))
            if self.running():
                os.kill(self.child.pid, signal.SIGKILL)
                self.error = "native shutdown deadline"
                self.fail("native_shutdown_deadline")
            self.wait_until(deadline)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.error = str(error)
            self.fail("native_shutdown_error")
            if self.running():
                try:
                    os.kill(self.child.pid, signal.SIGKILL)
                    self.wait_until(deadline)
                except (OSError, subprocess.TimeoutExpired):
                    self.error = "native child not reaped within deadline"
        finally:
            # EOF and diagnostics are independent of stdin and child liveness.
            while (
                self.selector
                and self.selector.get_map()
                and time.monotonic() < deadline
            ):
                drain(min(0.01, max(0, deadline - time.monotonic())))
            if (self.selector and self.selector.get_map()) or self.buffers.get(
                "stdout"
            ):
                self.fail("native_output_incomplete")
            if self.child.returncode not in (0, 2, None):
                self.fail("native_collector_failure")
            if (
                self.child.returncode == 2
                and self.terminal is None
                and self.status.get("type") != "status"
            ):
                self.fail("native_status_missing")
            self._resources()
            if self.final_usage is None:
                self.accounting_error = (
                    self.accounting_error or "owned child final accounting unavailable"
                )
                self.fail("native_final_accounting_unavailable")
            if self.selector:
                self.selector.close()
            for stream in (self.child.stdin, self.child.stdout, self.child.stderr):
                stream.close()
        loss = event_loss(self.losses, self.terminal)
        rejections = self.rejection_summary()
        ownership = self.unresolved_ownership()
        self.stopped_result = {
            **self.status,
            "complete": False,
            "native_exit": self.child.returncode,
            "error": self.error,
            "stop_delivery_error": self.input_error,
            "resource_complete": self.final_usage is not None,
            "resource_error": self.accounting_error,
            "final_usage": self.final_usage,
            "terminal": self.terminal,
            "coverage": "partial",
            # Known event loss, apart from the lifecycle (``clean``): endpoint
            # events the ring or the reads lost, counter wraps, and the
            # collector's own output drops (a malformed row fails it). Ordinary
            # namespace, protocol, family and port exclusions, and global
            # unknown exclusions, are not loss.
            "event_loss": loss,
            "degraded": bool(loss),
            # filter rejections are not loss: events of sockets not owned
            "rejections": rejections,
            "ownership_command_failures": len(self.ack_failures),
            "late_acknowledgements": self.late_acknowledgements,
            # cookies whose kernel entry was not known to match the decision
            # when the filter ended: its selection unresolved, not clean
            "ownership_unresolved": ownership,
            # Coverage is never provably complete (rare events, unverified kernel
            # misses); a required source is judged on whether it ran cleanly: it
            # became ready and stayed available, recorded no failure, and ended
            # with its own unfailed terminal record, a normal exit and its final
            # resource accounting.
            "clean": bool(
                self.status.get("available") is True
                and self.status.get("state") == "partial"
                and self.failed_reason is None
                and isinstance(self.terminal, dict)
                and self.terminal.get("failed") is False
                and self.child.returncode in (0, 2)
                and self.error is None
                and self.final_usage is not None
                and not any(ownership.values())
            ),
        }
        return self.stopped_result

    def rejection_summary(self) -> dict:
        """Each rejected event counted at most once: the total is the kernel's
        exact counter; per cookie, the counter harvested (read and deleted) at
        each admit and delete, and its terminal remainder (an LRU's, so lower
        bounds). For a cookie discovery admitted, a harvest holds rejections
        before or between its admissions (a miss can only begin while the
        cookie is out of the map), unless its kernel state was unknown or a
        harvest failed first (unresolved); the remainder is after its
        retirement only if the cookie was retired, by a confirmed delete, and
        every remaining event began after that delete began, else unresolved.
        A cookie never admitted is never owned. The rest (evictions, the
        counter map's capacity, untracked) is unattributed."""
        total = self.losses.get(("filter_exclusion", "excluded_unowned"))
        untracked = self.losses.get(("filter_accounting", "rejected_untracked"), 0)
        admitted = set(self.cookie_admissions) | set(self.cookie_decided)
        before = after = unresolved = never = never_cookies = rejected_owned = 0
        for cookie in admitted:
            harvests = self.harvests.get(cookie, ())
            # from when the cookie's phases are not established
            doubts = [since for since, _ in self.unknown.get(cookie, ())] + [
                h["applied_ns"] for h in harvests if h["errno"]
            ]
            doubtful = min(doubts) if doubts else None
            any_rejected = False
            for harvest in harvests:
                if not harvest["count"]:
                    continue
                any_rejected = True
                if doubtful is None or harvest["applied_ns"] < doubtful:
                    before += harvest["count"]
                else:
                    unresolved += harvest["count"]
            final = self.rejected.get(cookie)
            if final is not None and final["count"]:
                any_rejected = True
                last = harvests[-1] if harvests else None
                first = final["first_ns"]
                began = (
                    None
                    if first & KVP_FIRST_UNKNOWN or self.monotonic_offset is None
                    else _converted(first, self.monotonic_offset)
                )
                # discovery's latest retirement of the cookie: an admission
                # decided after the delete, installed or not, was ownership
                decided = [self.decided[a] for a in self.cookie_decided.get(cookie, ())]
                retired_at = max(
                    (d["until"] for d in decided if d["until"] is not None),
                    default=None,
                )
                if (
                    last is not None
                    and last["op"] == "D"
                    and doubtful is None
                    and self.desired.get(cookie) is None
                    and self.kernel.get(cookie) is None
                    and final.get("consistent") is True
                    and began is not None
                    and began > last["before_ns"]
                    and (retired_at is None or began > retired_at)
                ):
                    after += final["count"]
                else:
                    unresolved += final["count"]
            rejected_owned += any_rejected
        for cookie, final in self.rejected.items():
            if cookie not in admitted:
                never += final["count"]
                never_cookies += 1
        attributed = before + after + unresolved + never
        return {
            "total": total,
            "untracked": untracked,
            "attributed": attributed,
            "before_or_between_admissions": before,
            "after_retirement": after,
            "unresolved": unresolved,
            "never_owned": never,
            "admitted_cookies_rejected": rejected_owned,
            "never_owned_cookies": never_cookies,
            "unattributed": None if total is None else max(0, total - attributed),
            "counters_complete": (self.terminal or {}).get("rejected_complete") is True,
            "reconciled": total is not None and attributed <= total,
        }


def _converted(kernel_ns, offset):
    try:
        return convert_time(kernel_ns, offset)
    except ValueError:
        return None


ACK_TIMEOUT_NS = 1_000_000_000
OWNERSHIP_ATTEMPTS = 4  # commands per cookie toward one decision
OWNERSHIP_DRAIN_S = 0.75  # at stop, for the commands in flight
UNKNOWN = "unknown"
KVP_FIRST_UNKNOWN = 1 << 63  # protocol.h: a counter's earliest time unknown


class Collector(Channel):
    """Explicit TCP-only launcher; no image/kernel pin or alternate attachment."""

    def __init__(self, settings, inventory, directory):
        super().__init__(None, directory, settings, inventory)

    def start(self):
        if not self.settings.enabled:
            return
        self.status.update(state="unavailable", mode=self.settings.mode)
        try:
            if not self.settings.privileged_opt_in or not effective_privilege():
                self.status["reason"] = "missing_explicit_effective_privilege"
                return
            if not self.settings.artifacts:
                raise ValueError("artifact directory required")
            validate_artifacts(self.settings.artifacts)
            from .identity import process_identity

            own = process_identity(os.getpid())
            netns = os.stat("/proc/self/ns/net").st_ino
            self.directory.mkdir(parents=True, exist_ok=False)
            spec = self.directory / "tcp.plan"
            local_ports = " ".join(map(str, self.settings.local_ports))
            peer_ports = " ".join(map(str, self.settings.peer_ports))
            spec.write_text(
                f"P {own['pid']} {own['start_ticks']}\nN {netns}\n"
                f"L {local_ports}\nR {peer_ports}\n"
            )
            spec.chmod(0o600)
            root = Path(self.settings.artifacts)
            self.child = subprocess.Popen(
                [
                    str(root / "collector"),
                    "--arm",
                    str(spec),
                    str(root / "observe.bpf.o"),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                start_new_session=True,
            )
            self.open_pipes()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.status["reason"] = str(error)
