# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The observer's launcher as a sidecar of the serving process: a container
of the same pod (shared PID and network namespaces, a shared profile volume),
or any process beside the server.

It starts without waiting for the server, so the server can start. Within one
startup deadline it waits for the current serving generation's recorders,
launches the observer as its child and marks readiness once the observer is
ready. On termination (SIGTERM) or the server's exit it withdraws readiness
at once and, within one shutdown deadline shared with the observer, waits for
the server's frozen processes to exit and their L1 captures to close,
publishes completion, stops the observer, reaps it and records its exit. Each
start is a fresh attempt in its own directory, so markers from an earlier
attempt never apply."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .exit_record import write_exit_record
from .health import core_completion, expected_problems, inventory, recorder_counts

READY = "ready"  # in the observer root: the current attempt, once ready
POLL_S = 0.2


def _atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _state(identity) -> str:
    """``exited`` (gone, a zombie, or its PID another process's), ``alive``, or
    ``unknown`` when its status cannot be read: never taken as an exit."""
    try:
        stat = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(")", 1)[1]
    except FileNotFoundError:
        return "exited"
    except OSError:
        return "unknown"
    fields = stat.split()
    if int(fields[19]) != identity["start_ticks"] or fields[0] in ("Z", "X"):
        return "exited"
    return "alive"


class _Scan(threading.Thread):
    """One full L1 closure verification, off the drain loop, so the drain stays
    on its deadline however long the traces take to parse; an abandoned scan
    ends with the sidecar."""

    def __init__(self, directory, run_id, entries):
        super().__init__(daemon=True)
        self.args, self.result, self.finished_at = (
            (directory, run_id, entries),
            None,
            None,
        )
        self.start()

    def run(self):
        try:
            result = core_completion(*self.args)
        except Exception as error:  # verified as not closed, never raised
            result = {"all_closed": False, "error": repr(error)}
        self.finished_at = time.monotonic()  # before the result is visible
        self.result = result

    def closed_by(self, cutoff) -> bool:
        """Delivered closed no later than ``cutoff``."""
        return (
            self.result is not None
            and self.result.get("all_closed") is True
            and self.finished_at <= cutoff
        )


def _alive(identity) -> bool:
    """Running for all this sidecar can tell (unknown counts as running)."""
    return _state(identity) != "exited"


# Of a shutdown budget, the part kept for stopping and reaping the observer.
FINALIZE_SHARE, FINALIZE_MAX_S = 0.25, 5.0


class Sidecar:
    def __init__(self, settings: dict, *, run_id: str, endpoint: str, launcher: str):
        for key in ("probes", "role", "expected", "profiles_dir", "observer_root",
                    "irq_policy"):  # fmt: skip
            if key not in settings:
                raise ValueError(f"sidecar settings need {key!r}")
        if problems := expected_problems(settings["expected"]):
            raise ValueError(problems[0])
        self.settings, self.run_id, self.endpoint = settings, run_id, endpoint
        self.launcher = launcher
        self.profiles = Path(settings["profiles_dir"])
        self.root = Path(settings["observer_root"])
        self.startup_timeout_s = float(settings.get("startup_timeout_s", 900))
        self.stop_grace_s = float(settings.get("stop_grace_s", 25))
        self.cpus = settings.get("cpus") or sorted(os.sched_getaffinity(0))
        self.terminating = False
        self.observer = None
        self.attempt = None
        self.generation = None

    def on_term(self, *_):
        # never forwarded: the observer is stopped by the stop protocol
        self.terminating = True

    def log(self, line: str) -> None:
        with (self.attempt / "sidecar.log").open("a") as stream:
            stream.write(line + "\n")

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self.on_term)
        signal.signal(signal.SIGINT, self.on_term)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / READY).unlink(missing_ok=True)  # an earlier attempt's
        self.attempt = self.root / f"attempt-{time.time_ns()}-{os.getpid()}"
        self.attempt.mkdir()
        deadline = time.monotonic() + self.startup_timeout_s  # through readiness
        self.generation = self._wait_for_recorders(deadline)
        if self.generation is None:
            return 1
        config = {
            "probes": self.settings["probes"],
            "run_id": self.run_id,
            "endpoint": self.endpoint,
            "role": self.settings["role"],
            "expected": self.settings["expected"],
            "cpus": self.cpus,
            "input_dir": str(self.generation),
            "output_dir": str(self.attempt / "captures"),
            "irq_policy": self.settings["irq_policy"],
            "stop_file": str(self.attempt / "stop"),
            "endpoint_complete_file": str(self.attempt / "children.json"),
            # the observer waits as long as this sidecar's shutdown, never less
            "owner_exit_grace_s": self.stop_grace_s,
        }
        (self.attempt / "config.json").write_text(json.dumps(config))
        self.observer = subprocess.Popen(
            [
                sys.executable,
                self.launcher,
                "--config",
                str(self.attempt / "config.json"),
            ],
            stdout=(self.attempt / "observer.stdout.log").open("wb"),
            stderr=(self.attempt / "observer.stderr.log").open("wb"),
        )
        frozen = self._wait_ready(deadline)
        if frozen is None:
            # a timeout only when the deadline expired; termination cancels
            timed_out = False
            if self.observer.poll() is None:
                if self.terminating:
                    self.log("terminated before the observer was ready")
                else:
                    timed_out = True
                    self.log("timed out waiting for the observer to be ready")
                self.observer.kill()
            self.observer.wait()
            self._record_exit(timed_out)
            return 1
        _atomic(self.root / READY, str(self.attempt))
        owner, entries = frozen
        while not self.terminating and self.observer.poll() is None and _alive(owner):
            time.sleep(POLL_S)
        # draining: no longer ready, whatever follows
        (self.root / READY).unlink(missing_ok=True)
        # the serving processes are the recorders' and their owner (with data
        # parallelism, the API servers' common parent)
        timed_out = self._stop(entries, owner)
        self._record_exit(timed_out)
        return 0 if self.observer.returncode == 0 and not timed_out else 1

    def _generations(self):
        """Where the recorders may be: the profiles directory itself, or one
        directory per serving start beneath it (``gen-*``), newest first."""
        children = [p for p in self.profiles.glob("*/") if (p / self.run_id).is_dir()]
        children.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return (
            [self.profiles, *children]
            if (self.profiles / self.run_id).is_dir()
            else children
        )

    def _wait_for_recorders(self, deadline):
        """The current serving generation: its recorders exactly the expected
        ones, every one's process alive. Earlier generations stay on disk."""
        while time.monotonic() < deadline and not self.terminating:
            for directory in self._generations():
                try:
                    found = inventory(directory, self.run_id)
                except (OSError, ValueError):
                    continue
                if recorder_counts(found) == self.settings["expected"] and all(
                    # confirmed running: an unreadable status is not a server
                    _state(entry) == "alive"
                    for entry in found
                ):
                    return directory
            time.sleep(POLL_S)
        reason = "terminated" if self.terminating else "timed out"
        self.log(
            f"{reason} waiting for the expected recorders "
            f"{self.settings['expected']} of run {self.run_id}"
        )
        return None

    def _manifest(self):
        paths = list(self.attempt.glob(f"captures/{self.run_id}/*/manifest.json"))
        return paths[0] if len(paths) == 1 else None

    def _metadata(self):
        path = self._manifest()
        try:
            return json.loads(path.with_name("events.jsonl").open().readline())
        except (AttributeError, OSError, ValueError):
            return None

    def _wait_ready(self, deadline):
        """(the frozen owner, the frozen inventory) once the observer's capture
        is ready; None if it exits, termination comes or the deadline passes."""
        while (
            not self.terminating
            and self.observer.poll() is None
            and time.monotonic() < deadline
        ):
            path = self._manifest()
            try:
                if path is not None and json.loads(path.read_text()).get("ready"):
                    meta = self._metadata()
                    return meta["owner"], meta["inventory"]
            except (OSError, ValueError, KeyError, TypeError):
                pass
            time.sleep(POLL_S)
        if self.observer.poll() is not None:
            self.log(
                f"the observer exited before it was ready ({self.observer.returncode})"
            )
        return None

    def _stop(self, entries, owner=None) -> bool:
        """Within the one shutdown deadline: every frozen serving process gone
        and its L1 capture closed, completion published, the observer stopped
        and reaped. True if it had to be killed."""
        if self.observer.poll() is not None:
            return False
        deadline = time.monotonic() + self.stop_grace_s
        # the drain may use the budget less what stopping the observer needs
        drain_until = deadline - min(FINALIZE_MAX_S, self.stop_grace_s * FINALIZE_SHARE)
        closed = False
        processes = [*entries, *([owner] if owner is not None else [])]
        scan = None  # the one closure scan in flight, under the deadline
        while (remaining := drain_until - time.monotonic()) > 0:
            if scan is None and all(_state(p) == "exited" for p in processes):
                scan = _Scan(self.generation, self.run_id, entries)
            if scan is None:
                time.sleep(min(POLL_S, remaining))
                continue
            # while a scan runs its completion is the wake-up, up to the cutoff
            scan.join(remaining)
            if scan.closed_by(drain_until):
                closed = True
                break
            if scan.result is not None:  # not closed yet: scan again shortly
                scan = None
                time.sleep(min(POLL_S, max(0.0, drain_until - time.monotonic())))
        # a result delivered before the cutoff counts, however the loop ended
        closed = closed or (scan is not None and scan.closed_by(drain_until))
        _atomic(
            self.attempt / "children.json",
            json.dumps(
                {
                    "run_id": self.run_id,
                    "endpoint": self.endpoint,
                    "children_closed": closed,
                }
            ),  # fmt: skip
        )
        (self.attempt / "stop").touch()
        try:
            self.observer.wait(max(0.0, deadline - time.monotonic()))
            return False
        except subprocess.TimeoutExpired:
            self.observer.kill()
            self.observer.wait()
            return True

    def _record_exit(self, timed_out: bool) -> None:
        process_id = None
        path = self._manifest()
        if path is not None:
            try:
                process_id = json.loads(path.read_text()).get("process_id")
            except (OSError, ValueError):
                process_id = None
        code = self.observer.returncode
        signalled = -code if isinstance(code, int) and code < 0 else None
        write_exit_record(
            self.attempt,
            run_id=self.run_id,
            endpoint=self.endpoint,
            process_id=process_id,
            exit_code=None if signalled else code,
            signal=signalled,
            killed="signal" if signalled else None,
            timed_out=timed_out,
            launcher="sidecar",
        )


def main(path: str, run_id: str, endpoint: str, launcher: str) -> int:
    settings = json.loads(Path(path).read_text())
    return Sidecar(settings, run_id=run_id, endpoint=endpoint, launcher=launcher).run()
