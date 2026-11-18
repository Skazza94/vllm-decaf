# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Out-of-process JSONL sink. The consumer never takes the producer guard."""

import contextlib
import errno
import json
import os
import stat
import time
from pathlib import Path

from .schema import RECORD, decode_record, dump_line

RUNNING, FINISHED, OUTPUT_FAILED, BUSY = 1, 2, 3, 4
MAX_LINE = 8192


def write_all(stream, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        count = stream.write(remaining)
        if count is None or count <= 0:
            raise OSError(errno.EIO, "short profiler output")
        remaining = remaining[count:]


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        write_all(f, dump_line(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


class JsonlSink:
    """Single-reader lease plus peek/write/ack transactions.

    A consumer crash can replay rows. Sequence IDs make identical replays
    deduplicable. Acknowledgement never precedes the output flush.
    """

    def __init__(self, ring_path: str):
        from vllm import _kv_profiler

        self.ring = _kv_profiler.attach(ring_path)
        if not self.ring.acquire_reader():
            self.ring.close()
            raise BlockingIOError(errno.EBUSY, "profiler reader already leased")

    def drain_to(self, stream, batch_records: int = 256, after_sequence: int = 0):
        begin, end, raw = self.ring.peek(batch_records)
        for offset in range(0, len(raw), RECORD.size):
            row = decode_record(raw[offset : offset + RECORD.size])
            if row["sequence"] <= after_sequence:
                continue
            write_all(stream, dump_line(row))
        stream.flush()
        os.fsync(stream.fileno())
        self.ring.ack(begin, end)
        return end

    def close(self):
        self.ring.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def recover_output(stream) -> int:
    """Trim only an incomplete final line; complete corrupt rows are errors."""
    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
        raise OSError(errno.EINVAL, "profiler output must be a regular file")
    stream.seek(0)
    last_sequence = 0
    good = 0
    while line := stream.readline(MAX_LINE + 1):
        if len(line) > MAX_LINE:
            raise OSError(errno.EILSEQ, "oversized profiler JSONL row")
        if not line.endswith(b"\n"):
            stream.truncate(good)
            break
        row = json.loads(line)
        if row.get("schema") != 1:
            raise ValueError("unknown JSONL schema version")
        last_sequence = max(last_sequence, row.get("sequence", 0))
        good = stream.tell()
    stream.seek(0, os.SEEK_END)
    return last_sequence


def writer_main(ring_dir: str, output_dir: str, mode: str, batch: int):
    """Helper entry point. All potentially blocking durable output stays here."""
    ring_dir, output_dir = Path(ring_dir), Path(output_dir)
    sink = None
    metadata = json.loads((ring_dir / "metadata.json").read_text())
    manifest = output_dir / "manifest.json"
    watch_parent = os.getppid() == metadata["pid"]
    producer_gone = not watch_parent
    try:
        sink = JsonlSink(str(ring_dir / "ring"))
        if producer_gone:
            sink.ring.finish()
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest, {**metadata, "complete": False, "reason": "open"})
        sink.ring.set_writer_state(RUNNING)
        path = output_dir / "events.jsonl"
        with path.open("a+b", buffering=0) as stream:
            sequence = recover_output(stream)
            if not stream.tell():
                write_all(
                    stream, dump_line({"type": "metadata", "schema": 1, **metadata})
                )
                os.fsync(stream.fileno())
            while True:
                if watch_parent and os.getppid() != metadata["pid"]:
                    producer_gone = True
                    sink.ring.finish()
                state = sink.ring.stats()
                if mode == "background" or state["closing"]:
                    sequence = sink.drain_to(stream, batch, sequence)
                state = sink.ring.stats()
                if (
                    state["closing"]
                    and (producer_gone or sink.ring.quiescent())
                    and state["head"] == state["tail"]
                ):
                    lost = sum(v for k, v in state.items() if k.startswith("drop_"))
                    complete = (
                        not producer_gone and not lost and not state["unknown_handles"]
                    )
                    write_all(
                        stream,
                        dump_line(
                            {
                                "type": "end",
                                "schema": 1,
                                "stats": state,
                                "complete": complete,
                            }
                        ),
                    )
                    os.fsync(stream.fileno())
                    atomic_json(
                        manifest,
                        {
                            **metadata,
                            "complete": complete,
                            "saved": True,
                            "reason": (
                                "producer_gone"
                                if producer_gone
                                else "complete"
                                if complete
                                else "loss_or_unknown"
                            ),
                            "stats": state,
                            # The helper's lifetime CPU up to this manifest:
                            # startup, background drains and the end flush. A run's
                            # accounting subtracts what it had used when its
                            # workload ended to get the shutdown share.
                            "writer_cpu_s": time.process_time(),
                        },
                    )
                    sink.ring.set_writer_state(FINISHED)
                    return
                if mode == "end" or state["head"] == state["tail"]:
                    time.sleep(0.005)
    except Exception as exc:
        if sink is not None:
            sink.ring.set_writer_state(OUTPUT_FAILED, getattr(exc, "errno", 0) or 0)
        with contextlib.suppress(OSError):
            atomic_json(
                manifest,
                {
                    **metadata,
                    "complete": False,
                    "saved": False,
                    "reason": type(exc).__name__,
                },
            )
    finally:
        if sink is not None:
            sink.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ring_dir")
    parser.add_argument("output_dir")
    parser.add_argument("mode", choices=("background", "end"))
    parser.add_argument("batch", type=int)
    args = parser.parse_args()
    writer_main(args.ring_dir, args.output_dir, args.mode, args.batch)
