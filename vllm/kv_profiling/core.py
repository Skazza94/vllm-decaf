# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unattached recorder lifecycle; no serving or connector imports."""

import contextlib
import errno
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .config import NetworkProfilerConfig
from .sink import FINISHED, OUTPUT_FAILED, RUNNING, remove_staging


def ring_bytes(config: NetworkProfilerConfig) -> int:
    """The shared-memory bytes one recorder's ring takes (the native layout: a
    4 KiB header, 128-byte records, 32-byte identity and handle slots, doubled,
    and the identity arena, its end aligned to 64 bytes)."""
    identities_end = (
        4096 + config.ring_records * 128 + 2 * config.id_slots * 32 + config.id_bytes
    )
    return ((identities_end + 63) & ~63) + 2 * config.handle_slots * 32


def _native_recorder():
    """The native recorder, or, when it is not installed, an error that says so
    and how to build it into the ``vllm`` package this process imports (an
    editable checkout or site-packages). A present but unloadable extension
    keeps its loader error."""
    import importlib

    try:
        return importlib.import_module("vllm._kv_profiler")
    except ModuleNotFoundError as exc:
        if exc.name != "vllm._kv_profiler":
            raise
        import shlex

        import vllm

        package = Path(vllm.__file__).resolve().parent
        interpreter = Path(sys.executable).absolute()
        build = " ".join(
            shlex.quote(part)
            for part in (
                "cmake", "-S", "csrc/kv_profiler", "-B", "build/kv-profiler",
                f"-DPython_EXECUTABLE={interpreter}",
                f"-DCMAKE_INSTALL_PREFIX={package.parent}",
            )
        )  # fmt: skip
        raise ImportError(
            "KV profiler: the native recorder (vllm._kv_profiler) is not "
            f"installed in {package}. A VLLM_USE_PRECOMPILED=1 install skips it "
            "(the upstream wheel has none); it needs Linux and GIL-enabled "
            f"CPython. Build it from the source tree with: {build} && "
            "cmake --build build/kv-profiler && cmake --install build/kv-profiler",
            name="vllm._kv_profiler",
        ) from exc


def _no_room(config: NetworkProfilerConfig, error: OSError) -> OSError:
    """The ring did not fit in ``shared_dir``: say where, how much and what to do."""
    try:
        stat = os.statvfs(config.shared_dir)
        free = f"{stat.f_bavail * stat.f_frsize / 2**20:.1f} MiB free of "
        free += f"{stat.f_blocks * stat.f_frsize / 2**20:.1f} MiB"
    except OSError:
        free = "unknown free space"
    return OSError(
        errno.ENOSPC,
        f"KV profiler: no room for a recorder's ring in {config.shared_dir} "
        f"({free}); each recorder needs {ring_bytes(config) / 2**20:.2f} MiB "
        f"(ring_records={config.ring_records}, id_slots={config.id_slots}, "
        f"id_bytes={config.id_bytes}, handle_slots={config.handle_slots}), and a "
        "server opens one per scheduler, worker and API server process. Give it "
        "a larger tmpfs (for a container, docker run --shm-size) or lower "
        f"ring_records. ({error.strerror})",
    )


def _time_namespace():
    """This process's time namespace, or None on a kernel without time namespaces
    (before Linux 5.6 or built without them), where every process shares the host's
    monotonic clock."""
    try:
        return os.readlink("/proc/self/ns/time")
    except OSError:
        return None


@dataclass(frozen=True)
class CloseResult:
    saved: bool
    complete: bool
    reason: str
    stats: dict
    output_dir: str


class ProcessRecorder:
    """One fixed-memory producer ring and a bounded-lifetime helper process.

    Numeric IDs/string storage last until close, never reclaimed or reused.
    Transfer slots retire explicitly; their generation IDs never repeat.
    Stop producers before close. No production observer is installed here.
    """

    def __init__(
        self, config: NetworkProfilerConfig, *, start_writer=True, settings=None
    ):
        if not config.enabled:
            raise ValueError("use open_recorder for a disabled configuration")
        _kv_profiler = _native_recorder()

        self.config = config
        self.process_id = str(uuid4())
        try:
            self.ring_dir = Path(
                tempfile.mkdtemp(prefix="kvprof-", dir=config.shared_dir)
            )
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise _no_room(config, exc) from exc
            raise
        self.native = None
        self.writer = None
        self._result = None
        try:
            # Everything after the ring directory exists is inside the cleanup
            # scope: a metadata or snapshot failure leaves nothing behind.
            # absolute from the start: a restarted helper, capture.json and the
            # close result all name the same directory whatever the cwd becomes
            self.output_dir = (
                Path(config.output_dir).resolve() / str(config.run_id) / self.process_id
            )
            self.metadata = {
                "run_id": config.run_id,
                "process_id": self.process_id,
                "recovery_directory": str(self.ring_dir),
                "clock_id": str(uuid4()),
                "clock_source": "CLOCK_MONOTONIC",
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "time_namespace": _time_namespace(),
                "pid": os.getpid(),
                "net_namespace": os.readlink("/proc/self/ns/net"),
                "pid_namespace": os.readlink("/proc/self/ns/pid"),
                "start_ticks": int(
                    Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
                ),
                "context": config.context,
                "byteorder": sys.byteorder,
                "capabilities": config.context.get(
                    "capabilities", {"attached": False, "socket": False, "qp": False}
                ),
            }
            if settings is not None:
                # The configuration snapshot, beside (not inside) the caller's context.
                self.metadata["config"] = settings
            _fit_metadata(self.metadata)
            try:
                self.native = _kv_profiler.create(
                    str(self.ring_dir / "ring"),
                    config.ring_records,
                    config.id_slots,
                    config.id_bytes,
                    config.handle_slots,
                )
                (self.ring_dir / "metadata.json").write_text(json.dumps(self.metadata))
                # where this live capture is written: a replacement resumes it
                (self.ring_dir / "capture.json").write_text(
                    json.dumps({"output_dir": str(self.output_dir)})
                )
            except OSError as exc:
                if exc.errno == errno.ENOSPC:
                    raise _no_room(config, exc) from exc
                raise
            self.record = self.native.record
            if start_writer:
                self.start_writer()
        except BaseException:
            # release an allocated ring now, not when the exception is collected
            if self.native is not None:
                self.native.close()
            shutil.rmtree(self.ring_dir)
            raise

    def start_writer(self):
        """Start/restart a real helper; a live writer is never replaced."""
        if self.writer is not None and self.writer.poll() is None:
            raise RuntimeError("writer is still alive")
        self.native.set_writer_state(0)
        self.writer = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm.kv_profiling.sink",
                str(self.ring_dir),
                str(self.output_dir),
                self.config.flush,
                str(self.config.batch_records),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def wait_ready(self, timeout: float = 5) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.native.stats()["writer_state"]
            if state == RUNNING:
                return True
            if state == OUTPUT_FAILED or (
                self.writer is not None and self.writer.poll() is not None
            ):
                return False
            time.sleep(0.005)
        return False

    def _reader_attached(self) -> bool:
        """Whether a process holds the ring's reader lease (its ``flock``), read
        from ``/proc/locks`` without taking the lock; unknown counts as held."""
        try:
            info = os.stat(self.ring_dir / "ring")
        except FileNotFoundError:
            return False
        device = f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}"
        target = f"{device}:{info.st_ino}"
        try:
            locks = Path("/proc/locks").read_text().splitlines()
        except OSError:
            return True
        return any(
            len(fields) > 5 and fields[1] == "FLOCK" and fields[5] == target
            for fields in (line.split() for line in locks)
        )

    def close(self) -> CloseResult:
        """No output I/O here; each helper join/termination has a deadline."""
        if self._result is not None:
            return self._result
        self.native.finish()
        timeout = self.config.shutdown_timeout_s
        deadline = time.monotonic() + timeout
        timed_out = False
        if self.writer is not None:
            try:
                self.writer.wait(timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                self.writer.terminate()
                try:
                    self.writer.wait(timeout)
                except subprocess.TimeoutExpired:
                    self.writer.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        self.writer.wait(timeout)
        state = self.native.stats()
        # A writer this process did not start (a replacement or recovery helper)
        # still holds the ring's reader lease: its terminal state is awaited
        # within the same deadline.
        # From the moment it holds the lease, whatever state its predecessor
        # left (initial, OUTPUT_FAILED): one still working at the deadline is
        # a timeout.
        while (
            not timed_out
            and state["writer_state"] != FINISHED
            and self._reader_attached()
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            state = self.native.stats()
        # FINISHED precedes the helper's removal of the ring: the lease can be
        # gone with a terminal state this loop has not read yet
        state = self.native.stats()
        if (
            not timed_out
            and state["writer_state"] != FINISHED
            and self._reader_attached()
        ):
            timed_out = True
        saved = state["writer_state"] == FINISHED and not timed_out
        lost = sum(v for k, v in state.items() if k.startswith("drop_"))
        complete = saved and not lost and not state["unknown_handles"]
        reason = (
            "writer_timeout"
            if timed_out
            else "complete"
            if complete
            else "loss_or_unknown"
            if saved
            else "writer_unavailable"
        )
        self._result = CloseResult(
            saved,
            complete,
            reason,
            state,
            str(self.output_dir),
        )
        self.native.close()
        if saved:
            # normally its helper removed it already; repeating it is harmless
            remove_staging(self.ring_dir)
        # On failure retain the fixed-size ring for an explicit recovery tool.
        return self._result


def _fit_metadata(metadata: dict) -> None:
    """The metadata row must stay readable by the writer's recovery (its line
    limit): a snapshot that does not fit becomes an explicit unavailable entry,
    and a row too large without it is refused before anything starts."""
    from .schema import dump_line
    from .sink import MAX_LINE

    def size():
        return len(dump_line({"type": "metadata", "schema": 1, **metadata}))

    if size() > MAX_LINE and "config" in metadata:
        metadata["config"] = {
            "unavailable": f"the metadata row would exceed {MAX_LINE} bytes"
        }
    if size() > MAX_LINE:
        raise ValueError(f"profiler metadata exceeds {MAX_LINE} bytes")


def open_recorder(config: NetworkProfilerConfig, settings=None):
    """The off path imports no native extension and starts no helper.

    ``settings`` is the configuration snapshot recorded in the metadata."""
    return ProcessRecorder(config, settings=settings) if config.enabled else None
