# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unattached recorder lifecycle; no serving or connector imports."""

import contextlib
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
from .sink import FINISHED, OUTPUT_FAILED, RUNNING


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
        from vllm import _kv_profiler

        self.config = config
        self.process_id = str(uuid4())
        self.ring_dir = Path(
            tempfile.mkdtemp(
                prefix="kvprof-",
                dir=config.shared_dir,
            )
        )
        self.writer = None
        self._result = None
        try:
            # Everything after the ring directory exists is inside the cleanup
            # scope: a metadata or snapshot failure leaves nothing behind.
            self.output_dir = (
                Path(config.output_dir) / str(config.run_id) / self.process_id
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
            self.native = _kv_profiler.create(
                str(self.ring_dir / "ring"),
                config.ring_records,
                config.id_slots,
                config.id_bytes,
                config.handle_slots,
            )
            (self.ring_dir / "metadata.json").write_text(json.dumps(self.metadata))
            self.record = self.native.record
            if start_writer:
                self.start_writer()
        except BaseException:
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

    def close(self) -> CloseResult:
        """No output I/O here; each helper join/termination has a deadline."""
        if self._result is not None:
            return self._result
        self.native.finish()
        timeout = self.config.shutdown_timeout_s
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
            shutil.rmtree(self.ring_dir)
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
