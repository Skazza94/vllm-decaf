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


def thread_names(pid: int) -> dict[int, str]:
    """The producer's threads and their kernel names, read from /proc; a thread
    that already exited is absent. Never called on a recording path."""
    names = {}
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return names
    for tid in tids:
        with (
            contextlib.suppress(OSError, ValueError),
            open(f"/proc/{pid}/task/{tid}/comm", "rb") as f,
        ):
            names[int(tid)] = f.read(64).decode("utf-8", "replace").strip()
    return names


class JsonlSink:
    """Single-reader lease plus peek/write/ack transactions.

    A consumer crash can replay rows. Sequence IDs make identical replays
    deduplicable. Acknowledgement never precedes the output flush.
    """

    def __init__(self, ring_path: str):
        from vllm import _kv_profiler

        self.ring = _kv_profiler.attach(ring_path)
        self.tids: set[int] = set()
        if not self.ring.acquire_reader():
            self.ring.close()
            raise BlockingIOError(errno.EBUSY, "profiler reader already leased")

    def drain_to(self, stream, batch_records: int = 256, after_sequence: int = 0):
        begin, end, raw = self.ring.peek(batch_records)
        if begin == end:
            # nothing to write or acknowledge: an idle background drain does
            # no I/O (the final rows are synced on their own)
            return end
        for offset in range(0, len(raw), RECORD.size):
            row = decode_record(raw[offset : offset + RECORD.size])
            if row["sequence"] <= after_sequence:
                continue
            if row.get("tid"):
                self.tids.add(row["tid"])
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


THREAD_ROW_BYTES = 6144
"""A ``threads`` row's serialized bound, under the reader's ``MAX_LINE``."""


def thread_rows(threads: dict[str, str | None]) -> list[dict]:
    """The thread inventory as bounded ``threads`` rows (each under
    ``THREAD_ROW_BYTES``), so any number of threads stays readable."""
    rows, chunk = [], {}
    for tid in sorted(threads, key=int):
        trial = {**chunk, tid: threads[tid]}
        if (
            chunk
            and len(dump_line({"type": "threads", "schema": 1, "threads": trial}))
            > THREAD_ROW_BYTES
        ):
            rows.append({"type": "threads", "schema": 1, "threads": chunk})
            chunk = {tid: threads[tid]}
        else:
            chunk = trial
    if chunk:
        rows.append({"type": "threads", "schema": 1, "threads": chunk})
    return rows


def recover_output(stream, tids: set | None = None, names: dict | None = None) -> int:
    """Trim only an incomplete final line; complete corrupt rows are errors.

    ``tids`` and ``names``, when given, are seeded from the rows already written:
    a writer restarted after a crash keeps the recording threads it had seen.
    """
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
        if tids is not None and row.get("tid"):
            tids.add(row["tid"])
        if names is not None and row.get("type") == "threads":
            for tid, name in row["threads"].items():
                if tids is not None:
                    tids.add(int(tid))
                if name is not None:
                    names[int(tid)] = name
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
        names: dict[int, str] = {}
        with path.open("a+b", buffering=0) as stream:
            sequence = recover_output(stream, sink.tids, names)
            if not stream.tell():
                write_all(
                    stream, dump_line({"type": "metadata", "schema": 1, **metadata})
                )
                os.fsync(stream.fileno())
            names_at = 0.0
            while True:
                if time.monotonic() - names_at >= 1.0:
                    # Thread names, out of band: a thread that exits before the
                    # end keeps the name it had when last seen here.
                    names.update(thread_names(metadata["pid"]))
                    names_at = time.monotonic()
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
                    names.update(thread_names(metadata["pid"]))
                    threads = {str(tid): names.get(tid) for tid in sorted(sink.tids)}
                    # The inventory goes in bounded rows before the end record,
                    # which only counts them; the manifest keeps the whole map.
                    for row in thread_rows(threads):
                        write_all(stream, dump_line(row))
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
                                # Recording threads, kernel ID -> name (None when
                                # it exited before it was ever observed here), are
                                # in the "threads" rows before this record.
                                "thread_count": len(threads),
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
                            "threads": threads,
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
