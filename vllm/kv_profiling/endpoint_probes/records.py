# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe-only schema. The native L1 record and transfer fields are unchanged."""

import json
import os
import queue
import resource
import threading
import time
from pathlib import Path

PROBE_SCHEMA = 1
KINDS = {"endpoint_metrics": 22, "probe_status": 23, "probe_resource": 24}
VALUES = (
    "bytes",
    "selected_descriptors",
    "merged_descriptors",
    "duration0_ns",
    "duration1_ns",
    "aux0",
    "aux1",
)


def event(kind, scope, entity, begin, end, data, transport="unknown"):
    if kind not in KINDS or transport not in ("tcp", "rdma", "unknown"):
        raise ValueError("unknown probe kind/transport")
    return {
        "schema": 1,
        "probe_schema": PROBE_SCHEMA,
        "type": "event",
        "kind": kind,
        "time_ns": end,
        "sample_begin_ns": begin,
        "sample_end_ns": end,
        "scope": scope,
        "entity": entity,
        "context": 0,
        "request_ref": 0,
        "handoff_ref": 0,
        "transfer": 0,
        "peer_ref": 0,
        "subject": 0,
        "status": 0,
        "unbound_handle": False,
        **dict.fromkeys(VALUES),
        "transport": {
            "kind": transport,
            "backend": None,
            "provenance": "kernel_observed"
            if transport != "unknown"
            else "endpoint_observed",
            "tcp": data if transport == "tcp" else None,
            "rdma": data if transport == "rdma" else None,
        },
        "observation": data if transport == "unknown" else None,
    }


def validate(row):
    if row.get("schema") != 1 or row.get("probe_schema") != PROBE_SCHEMA:
        raise ValueError("unknown endpoint probe schema")
    if row.get("type") not in ("metadata", "event", "end"):
        raise ValueError("unknown probe record type")
    if row.get("type") != "event":
        return
    for name in ("time_ns", "sample_begin_ns", "sample_end_ns"):
        if type(row.get(name)) is not int or row[name] < 0:
            raise ValueError("invalid probe sample timestamp")
    if row.get("kind") not in KINDS or any(
        row.get(key) != 0
        for key in ("request_ref", "handoff_ref", "transfer", "peer_ref")
    ):
        raise ValueError("probe rows cannot claim transfer/request bindings")
    if any(row.get(key) is not None for key in VALUES):
        raise ValueError("endpoint counters cannot be transfer metrics")
    transport = row["transport"]
    kind = transport["kind"]
    if kind not in ("tcp", "rdma", "unknown") or any(
        transport.get(other) is not None for other in ("tcp", "rdma") if other != kind
    ):
        raise ValueError("mixed probe transport blocks")
    if kind in ("tcp", "rdma") and not isinstance(transport.get(kind), dict):
        raise ValueError("missing typed probe transport block")
    if row["sample_begin_ns"] > row["sample_end_ns"]:
        raise ValueError("reversed probe sample interval")


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Writer:
    """One bounded asynchronous writer; failure cannot propagate into serving."""

    def __init__(self, directory, metadata, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.queue = queue.Queue(config.queue_rows)
        self.maximum = config.output_bytes
        self.sequence = 0
        self.saved_sequence = 0
        self.dropped = 0
        self.error = None
        self.footer = None
        self.ready = threading.Event()
        self.manifest = {
            "probe_schema": 1,
            "run_id": metadata["run_id"],
            "process_id": metadata["process_id"],
            "ready": False,
            "saved": False,
            "complete": False,
        }
        self.metadata = metadata
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()
        if not self.ready.wait(2) or self.error:
            raise RuntimeError(f"probe writer startup failed: {self.error}")

    def publish_ready(self):
        """Mark the capture ready in its manifest, the signal a controller waits
        for: called by the observer once its whole setup (an eBPF collector's
        first answer included) is done."""
        self.manifest["ready"] = True
        atomic_json(self.directory / "manifest.json", self.manifest)

    def room(self) -> int:
        """Rows the queue can take now (approximately: its thread drains it).
        A failed writer takes no more rows, so it holds nothing back: all of
        its queue, its rows dropped and counted (its capture incomplete)."""
        if self.error or not self.thread.is_alive():
            return self.queue.maxsize
        return max(0, self.queue.maxsize - self.queue.qsize())

    def put(self, row):
        validate(row)
        self.sequence += 1
        row = {**row, "sequence": self.sequence}
        raw = (json.dumps(row, allow_nan=False) + "\n").encode()
        if len(raw) > 16 << 10 or self.error:
            self.dropped += 1
            return
        try:
            self.queue.put_nowait((self.sequence, raw))
            return self.sequence
        except queue.Full:
            self.dropped += 1

    def close(self, complete, reason, stats):
        self.footer = {
            "schema": 1,
            "probe_schema": 1,
            "type": "end",
            "complete": complete and not self.dropped,
            "reason": reason,
            "stats": {
                **stats,
                "head": self.sequence,
                "tail": self.sequence,
                "drop_probe_rows": self.dropped,
            },
        }
        try:
            self.queue.put(None, timeout=1)
        except queue.Full:
            self.error = "writer shutdown queue timeout"
        self.thread.join(1)
        if self.thread.is_alive():
            self.error = "writer shutdown timeout"
        return not self.error and self.footer["complete"]

    def _write(self):
        try:
            atomic_json(self.directory / "manifest.json", self.manifest)
            with (self.directory / "events.jsonl").open("xb") as stream:
                raw = (json.dumps(self.metadata, allow_nan=False) + "\n").encode()
                if len(raw) > self.maximum:
                    raise ValueError("probe output byte limit at metadata")
                stream.write(raw)
                stream.flush()
                # the stream is open; the capture's own readiness is the
                # observer's to publish (``publish_ready``)
                self.ready.set()
                size = len(raw)
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    ticket, raw = item
                    size += len(raw)
                    if size > self.maximum:
                        raise ValueError("probe output byte limit")
                    stream.write(raw)
                    stream.flush()
                    self.saved_sequence = ticket
                if self.error:
                    raise RuntimeError(self.error)
                stream.write((json.dumps(self.footer) + "\n").encode())
                stream.flush()
                os.fsync(stream.fileno())
            atomic_json(
                self.directory / "manifest.json",
                {
                    **self.manifest,
                    "saved": True,
                    "complete": self.footer["complete"],
                    "end": self.footer,
                    "observer_process_cpu_s": time.process_time(),
                    "observer_rss_hwm_kib": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                },
            )
        except Exception as error:
            # never empty: put(), room(), close() and the observer test it
            self.error = str(error) or type(error).__name__
        finally:
            self.ready.set()
