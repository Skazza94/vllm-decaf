# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit native collector control; no import-time probing or native imports."""

import hashlib
import json
import os
import re
import selectors
import signal
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

IMAGE = "sha256:ff979851b632e2048a4a8dd141cf71b2a4701f243c0bf96fb275f7286daadc76"
LIMITS_SHA256 = "5f9162c8a1e3a907c5ccaadfa7cb29ba09b418f4dc11326eb12701e147be4929"
NIXL_SHA256 = "82c8d0197eb43a4cdbbe3a8cb71e7b0a722cb1af38cf8ade768a1ea22f8ad20d"
NIXL_OFFSET = 0x48900
SETS = {"native_api", "tcp_events", "provider_wr"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Settings:
    enabled: bool = False
    privileged_opt_in: bool = False
    sets: tuple = ("native_api", "tcp_events")
    artifacts: str | None = None
    kernel_audit: str | None = None

    def __post_init__(self):
        for name in ("enabled", "privileged_opt_in"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"ebpf.{name} must be boolean")
        if not isinstance(self.sets, (tuple, list)) or not self.sets:
            raise ValueError("ebpf.sets requires explicit sources")
        if set(self.sets) - SETS or len(set(self.sets)) != len(self.sets):
            raise ValueError("unknown/duplicate ebpf set")
        for name in ("artifacts", "kernel_audit"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not Path(value).is_absolute()
            ):
                raise ValueError(f"ebpf.{name} requires an absolute path")


def effective_privilege():
    """Availability check only; never changes caps, policy or resource limits."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("CapEff:"):
            caps = int(line.split()[1], 16)
            return bool(caps & (1 << 21) or (caps & (1 << 38) and caps & (1 << 39)))
    return False


def executable_offset(path, offset):
    """Validate an executable file offset, not a symbol VMA or data alias."""
    with Path(path).open("rb") as stream:
        head = stream.read(64)
        if len(head) != 64 or head[:6] != b"\x7fELF\x02\x01":
            raise ValueError("ELF64 little-endian required")
        if struct.unpack_from("<H", head, 18)[0] != 62:
            raise ValueError("x86-64 ABI required")
        table = struct.unpack_from("<Q", head, 32)[0]
        size, count = struct.unpack_from("<HH", head, 54)
        if size != 56 or not 0 < count <= 64:
            raise ValueError("bounded ELF program headers required")
        for index in range(count):
            stream.seek(table + index * size)
            raw = stream.read(size)
            if len(raw) != size:
                raise ValueError("truncated ELF")
            kind, flags, start, _, _, length, _, _ = struct.unpack("<IIQQQQQQ", raw)
            if kind == 1 and flags & 1 and start <= offset < start + length:
                return
    raise ValueError("offset is not executable")


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
    if len(line) + 1 > 256:
        raise ValueError("native record bound")
    row = json.loads(line)
    if not isinstance(row, dict) or row.get("type") not in {
        "api",
        "tcp",
        "tcp_namespace",
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
    if row["type"] == "api":
        for key in (
            "generation",
            "epoch",
            "pid",
            "tid",
            "thread_ns",
            "begin_ns",
            "end_ns",
        ):
            if type(row.get(key)) is not int or not 0 <= row[key] < 1 << 64:
                raise ValueError("invalid native integer")
        if not all(row[k] > 0 for k in ("generation", "epoch", "pid", "tid")):
            raise ValueError("positive native identities required")
        if row["begin_ns"] > row["end_ns"] or type(row.get("status")) is not int:
            raise ValueError("invalid native interval/status")
    if row["type"] == "tcp":
        if row.get("counter") not in {
            "rx_events",
            "skb_len_bytes",
            "states",
            "retrans",
            "reset_tx",
            "reset_rx",
        }:
            raise ValueError("unknown TCP counter")
        for key in ("cookie", "value"):
            if type(row.get(key)) is not int or not 0 <= row[key] < 1 << 64:
                raise ValueError("invalid TCP integer")
        if not row["cookie"]:
            raise ValueError("unknown socket identity is not a socket row")
    return row


def validate_artifacts(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("image") != IMAGE or manifest.get("limits_sha256") != LIMITS_SHA256:
        raise ValueError("unreviewed image/limits")
    expected = {"collector", "observe.bpf.o", "limits.json", "vmlinux.h"}
    if set(manifest.get("files", {})) != expected:
        raise ValueError("incomplete artifact manifest")
    for name in expected:
        if digest(root / name) != manifest["files"][name]:
            raise ValueError("artifact hash mismatch: " + name)
    if digest(root / "limits.json") != LIMITS_SHA256:
        raise ValueError("limits hash mismatch")
    if manifest.get("kernel_release") != os.uname().release:
        raise ValueError("unvalidated kernel release")
    if digest("/sys/kernel/btf/vmlinux") != manifest.get("kernel_btf_sha256"):
        raise ValueError("unvalidated kernel BTF")
    return manifest


def validate_audit(path, manifest):
    audit = json.loads(Path(path).read_text())
    bound = audit.get("map_upper_bound_bytes")
    if type(bound) is not int or not 0 < bound <= 16 << 20:
        raise ValueError("kernel map bound absent/over cap")
    for name in ("kernel_release", "kernel_btf_sha256", "limits_sha256"):
        if audit.get(name) != manifest[name]:
            raise ValueError("kernel accounting audit identity mismatch")
    if audit.get("object_sha256") != manifest["files"]["observe.bpf.o"]:
        raise ValueError("kernel accounting object mismatch")
    if audit.get("method") != "audited_upper_bound_and_fdinfo_memlock":
        raise ValueError("kernel map accounting unavailable")
    if not re.fullmatch(r"[0-9a-f]{64}", audit.get("source_audit_sha256", "")):
        raise ValueError("target-kernel allocation/field audit required")
    launch = audit.get("launch", {})
    binding = audit.get("namespace_bindings", {})
    if (
        launch.get("HostConfig", {}).get("Privileged") is not True
        or launch.get("Image") != IMAGE
    ):
        raise ValueError("raw privileged launch proof required")
    if binding.get("pid_namespace") != os.readlink("/proc/self/ns/pid") or binding.get(
        "net_namespace"
    ) != os.readlink("/proc/self/ns/net"):
        raise ValueError("launch namespace proof mismatch")
    # Controller evidence is not a privilege grant; effective caps are checked too.
    return bound


def targets(inventory, need_native=True):
    """Bind reviewed SDK bytes/offset to actual worker mappings and PID generation."""
    from .identity import process_identity

    result, generations = [], set()
    for item in inventory:
        if item.get("context", {}).get("role") != "worker":
            continue
        pid = item.get("pid")
        if type(pid) is not int or pid <= 0:
            raise ValueError("positive TGID required; no autoattach fallback")
        actual = process_identity(pid)
        if any(actual[key] != item.get(key) for key in actual):
            raise ValueError("worker identity changed")
        if actual["pid_namespace"] != os.readlink("/proc/self/ns/pid") or actual[
            "net_namespace"
        ] != os.readlink("/proc/self/ns/net"):
            raise ValueError("worker namespace mismatch")
        tids = list((Path("/proc") / str(pid) / "task").iterdir())
        if len(tids) > 128:
            raise ValueError("per-process thread limit")
        path = Path(
            f"/proc/{pid}/root/opt/nixl/lib/libnixl.so"
            if need_native
            else f"/proc/{pid}/exe"
        )
        description = path.stat()
        inode = description.st_ino
        device = description.st_dev
        mapping = False
        rows = (Path("/proc") / str(pid) / "maps").read_text().splitlines()
        if len(rows) > 8192:
            raise ValueError("worker mapping inventory limit")
        for row in rows:
            fields = row.split(maxsplit=5)
            if (
                len(fields) == 6
                and int(fields[4]) == inode
                and tuple(int(part, 16) for part in fields[3].split(":"))
                == (os.major(device), os.minor(device))
                and "x" in fields[1]
                and fields[5] == "/opt/nixl/lib/libnixl.so"
            ):
                mapping = True
        if need_native:
            if not mapping or digest(path) != NIXL_SHA256:
                raise ValueError("mapped SDK inode/hash unavailable")
            executable_offset(path, NIXL_OFFSET)
        generation = int.from_bytes(
            hashlib.sha256(item["process_id"].encode()).digest()[:8], "little"
        )
        if not generation or generation in generations:
            raise ValueError("process generation collision")
        generations.add(generation)
        result.append(
            (pid, actual["start_ticks"], generation, device, inode, str(path))
        )
    if not result or len(result) > 256:
        raise ValueError("worker process inventory limit/empty")
    if (
        sum(
            len(list((Path("/proc") / str(row[0]) / "task").iterdir()))
            for row in result
        )
        > 2048
    ):
        raise ValueError("endpoint thread limit")
    return result


def mapping_signature(pid, sdk=False):
    """Observe executable mappings; a periodic check is not a gap-free exec monitor."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("positive TGID required")
    root = Path("/proc") / str(pid)
    executable = (root / "exe").stat()
    name = "/opt/nixl/lib/libnixl.so" if sdk else os.readlink(root / "exe")
    raw = (root / "maps").read_text().splitlines()
    if len(raw) > 8192:
        raise ValueError("mapping inventory bound")
    selected = []
    for line in raw:
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5] == name and "x" in fields[1]:
            selected.append(tuple(fields[:5]))
    if not selected:
        raise ValueError("target executable mapping unavailable")
    return executable.st_dev, executable.st_ino, tuple(selected)


class Channel:
    """Bounded pipes for an already-owned child; never enables kernel probes."""

    def __init__(self, child, directory, settings=None, inventory=()):
        self.settings = settings if settings is not None else Settings()
        self.inventory = inventory
        self.directory = Path(directory)
        self.child = child
        self.selector = None
        self.buffers = {}
        self.pending = []
        self.stderr_bytes = 0
        self.final_usage = None
        self.reaped = False
        self.accounting_error = None
        self.terminal = None
        self.monotonic_offset = None
        self.time_namespace = None
        self.namespaces = None
        self.mapping_generations = {}
        self.next_identity_check = 0
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
            "sets": [],
            "provider_wr": "unavailable",
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
            self.namespaces = {
                name: os.readlink("/proc/self/ns/" + name)
                for name in ("pid", "net", "time")
            }
            self.time_namespace = self.namespaces["time"]
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
        if time.monotonic() >= self.next_identity_check:
            self.next_identity_check = time.monotonic() + 1
            for pid, signature in self.mapping_generations.items():
                if (
                    mapping_signature(pid, "native_api" in self.settings.sets)
                    != signature
                ):
                    raise ValueError("target executable/mapping changed")
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
                        sets=list(self.settings.sets),
                        kernel_miss_visibility="unverified",
                    )
                elif row.get("type") == "status":
                    self.status.update(
                        row,
                        state="failed"
                        if row.get("outcome") == "FAIL"
                        else "unavailable",
                    )
                elif row.get("type") == "resource" and row.get("outcome") == "FAIL":
                    self.status.update(
                        available=False,
                        state="failed",
                        outcome="FAIL",
                        reason="kernel_map_budget_exceeded",
                        resource=row,
                    )
                elif row.get("type") == "end":
                    self.terminal = row
                if row.get("outcome") == "FAIL":
                    self.fail(self.status.get("reason", "native_collector_failure"))
                elif self.failed_reason is not None:
                    self.fail(self.failed_reason)
                rows.append(row)
            if len(buffer) >= 256:
                raise ValueError("unterminated native record bound")
        waiting = []
        for ticket, generation, epoch in self.pending:
            if (
                not self.running()
                or writer.saved_sequence < ticket
                or not self.send_command(f"ack {generation} {epoch}\n".encode())
            ):
                waiting.append((ticket, generation, epoch))
        self.pending = waiting
        return rows

    def acknowledge(self, row, ticket):
        if row.get("type") == "api" and ticket is not None:
            if len(self.pending) >= 4096:
                raise ValueError("native acknowledgement queue bound")
            self.pending.append((ticket, row["generation"], row["epoch"]))

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
                        self.acknowledge(row, consume(row))
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
            "kernel_qualification": "pending_live_checkpoint",
        }
        return self.stopped_result


class Collector(Channel):
    """Opt-in launcher: no child or privilege retry on inactive/unavailable paths."""

    def __init__(self, settings, inventory, directory):
        super().__init__(None, directory, settings, inventory)

    def start(self):
        if not self.settings.enabled:
            return
        self.status.update(state="unavailable", requested=list(self.settings.sets))
        if "provider_wr" in self.settings.sets:
            self.status["reason"] = "provider_abi_unavailable"
            return
        if not self.settings.privileged_opt_in or not effective_privilege():
            self.status["reason"] = "missing_explicit_effective_privilege"
            return
        try:
            if not self.settings.artifacts or not self.settings.kernel_audit:
                raise ValueError("artifact and target-kernel accounting audit required")
            manifest = validate_artifacts(self.settings.artifacts)
            bound = validate_audit(self.settings.kernel_audit, manifest)
            admitted = targets(self.inventory, "native_api" in self.settings.sets)
            self.mapping_generations = {
                row[0]: mapping_signature(row[0], "native_api" in self.settings.sets)
                for row in admitted
            }
            offsets = read_offsets("/proc/self/timens_offsets")
            self.monotonic_offset = offsets["monotonic"]
            self.time_namespace = os.readlink("/proc/self/ns/time")
            self.namespaces = {
                name: os.readlink("/proc/self/ns/" + name)
                for name in ("pid", "net", "time")
            }
            self.directory.mkdir(parents=True, exist_ok=False)
            spec = self.directory / "native.plan"
            from .identity import process_identity

            own = process_identity(os.getpid())
            ns = os.stat("/proc/self/ns/pid")
            lines = [
                f"P {own['pid']} {own['start_ticks']}",
                (
                    f"N {ns.st_dev} {ns.st_ino} {os.stat('/proc/self/ns/net').st_ino} "
                    f"{offsets['monotonic']} {int('tcp_events' in self.settings.sets)} "
                    f"{int('native_api' in self.settings.sets)}"
                ),
                f"B {bound}",
            ]
            lines.extend("T " + " ".join(map(str, row)) for row in admitted)
            spec.write_text("\n".join(lines) + "\n")
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
            self.error = str(error)
