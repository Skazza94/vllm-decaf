# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""First-peer admission and clock calibration, off the request path.

Admission is a preallocated table: three slot lists built at attach, a published
count and a last-hit index, all small cached ints. It takes no lock, creates no
tuple, set, iterator or thread, and never waits. One long-lived thread measures
published peers in order. Only the closer writes files and only the closer
touches the recorder.
"""

import threading
import time
from pathlib import Path

from .clock import ClockMapping, measure_peer
from .sink import atomic_json

PEER_OK = "ok"
PEER_FAILED = "failed"
PEER_PENDING = "pending_at_close"
PEER_ABANDONED = "abandoned"


class PeerClocks:
    def __init__(self, config, run_id: str, local_generation: str):
        self.config = config
        self.run_id = run_id
        self.local_generation = local_generation
        cap = self.cap = config.clock_max_peers
        self.engine = [None] * cap
        self.host = [None] * cap
        self.port = [None] * cap
        self.tail = 0
        self.last = 0
        self.closed = False
        self.results: list[dict] = []
        self.thread = threading.Thread(
            target=self._run, name="kvprof-clock", daemon=True
        )
        self.thread.start()

    def admit(self, engine_id, host, port) -> int:
        """Return 1 for a new stored peer, 0 for a known one, -1 when full.

        Publishing order: every slot field is stored before the count moves, so
        the measuring thread never reads a half-written slot.
        """
        tail = self.tail
        if tail and self.engine[self.last] == engine_id:
            return 0
        i = 0
        while i < tail:
            if self.engine[i] == engine_id:
                self.last = i
                return 0
            i += 1
        if tail == self.cap:
            return -1
        self.engine[tail] = engine_id
        self.host[tail] = host
        self.port[tail] = port
        self.last = tail
        self.tail = tail + 1
        return 1

    def _run(self) -> None:
        done = 0
        while not self.closed:
            if done < self.tail:
                self._measure(done)
                done += 1
            else:
                time.sleep(0.1)

    def _address(self, index: int) -> tuple[str, int]:
        return self.host[index], self.port[index] + self.config.clock_port_offset

    def _measure(self, index: int) -> None:
        entry = {"engine_id": self.engine[index], "address": None}
        try:
            address = self._address(index)
            entry["address"] = list(address)
            samples = measure_peer(
                address,
                self.run_id,
                self.local_generation,
                samples=self.config.clock_samples,
                timeout_s=1,
            )
            entry.update(
                status=PEER_OK,
                remote_generation=samples[0].remote_generation,
                start=samples,
            )
        except Exception as exc:
            entry.update(status=PEER_FAILED, error=type(exc).__name__)
        self.results.append(entry)

    def close(self, output_dir: Path, join_s: float) -> dict:
        """Stop, take the end anchors under one budget and write the manifest.

        Called after every hook has stopped. A thread still alive after the
        bounded join is abandoned: its list is only snapshotted, never awaited,
        and nothing it does afterwards reaches a file or the recorder.
        """
        self.closed = True
        output_dir.mkdir(parents=True, exist_ok=True)
        self.thread.join(join_s)
        abandoned = self.thread.is_alive()
        results = list(self.results)
        deadline = time.monotonic() + min(
            self.config.clock_end_budget_s, self.config.shutdown_timeout_s
        )
        peers = []
        files = []
        for index in range(self.tail):
            entry = results[index] if index < len(results) else None
            record = {
                "engine_id": self.engine[index],
                "address": list(self._address(index)),
            }
            if entry is None:
                record["status"] = PEER_ABANDONED if abandoned else PEER_PENDING
                peers.append(record)
                continue
            record["status"] = entry["status"]
            if entry["status"] != PEER_OK:
                record["error"] = entry["error"]
                peers.append(record)
                continue
            record["remote_generation"] = entry["remote_generation"]
            end = []
            remaining = deadline - time.monotonic()
            if remaining > 0:
                try:
                    end = measure_peer(
                        tuple(record["address"]),
                        self.run_id,
                        self.local_generation,
                        expected_remote=entry["remote_generation"],
                        samples=self.config.clock_samples,
                        timeout_s=min(1, remaining),
                    )
                except Exception as exc:
                    record["end_error"] = type(exc).__name__
            record["end_missing"] = not end
            try:
                mapping = ClockMapping.from_samples(entry["start"], end)
            except ValueError as exc:
                record["status"] = PEER_FAILED
                record["error"] = str(exc)
                peers.append(record)
                continue
            name = f"calibration-{len(files)}.json"
            atomic_json(output_dir / name, mapping.to_json())
            record["calibration_file"] = name
            files.append(name)
            peers.append(record)
        manifest = {
            "run_id": self.run_id,
            "local_generation": self.local_generation,
            "clock_thread": PEER_ABANDONED if abandoned else "joined",
            "peers": peers,
        }
        atomic_json(output_dir / "calibration.json", manifest)
        return manifest
