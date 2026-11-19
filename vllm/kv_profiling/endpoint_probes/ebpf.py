# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in portable TCP event collector; no import-time privilege or kernel probes."""

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
    }:
        raise ValueError("unknown native row")
    if ("exact" in row and row["exact"] is not False) or any(
        key in row for key in ("request_ref", "handoff_ref", "transfer", "transfer_id")
    ):
        raise ValueError("native observation claimed a transfer")
    if row["type"] == "tcp_event":
        if row.get("schema") != 2 or row.get("event") not in EVENTS:
            raise ValueError("unknown TCP event")
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
        cookie = row.get("cookie")
        if cookie is not None and (type(cookie) is not int or not 0 < cookie < 1 << 64):
            raise ValueError("invalid TCP cookie")
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


def validate_artifacts(directory):
    """Check a package against its own manifest, never against its build host."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {"collector", "observe.bpf.o", "limits.json"}
    if (
        manifest.get("schema") != 2
        or manifest.get("mode") != "tcp_tracepoints"
        or set(manifest.get("files", {})) != expected
    ):
        raise ValueError("incomplete TCP artifact manifest")
    for name in expected:
        if digest(root / name) != manifest["files"][name]:
            raise ValueError("artifact hash mismatch: " + name)
    return manifest


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

    def poll(self, writer, timeout=0):
        if not self.child or not self.selector:
            return []
        self._resources()
        if any(
            os.readlink("/proc/self/ns/" + name) != value
            for name, value in self.namespaces.items()
        ):
            raise ValueError("observer namespace changed")
        rows = []
        for key, _ in sorted(
            self.selector.select(timeout), key=lambda item: item[0].data == "stdout"
        ):
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
        return rows

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
        }
        return self.stopped_result


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
