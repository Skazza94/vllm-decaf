# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic endpoint lifecycle using real recorders, not a serving engine."""

import argparse
import json
import os
import signal
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--directory", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--role", default="decode")
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--overlap-cpu", type=int)
    parser.add_argument("--close-delay", type=float, default=0)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    from vllm.kv_profiling.config import NetworkProfilerConfig
    from vllm.kv_profiling.core import ProcessRecorder
    from vllm.kv_profiling.endpoint_probes.health import inventory

    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    profiles = directory / "profiles"
    stop, stop_thread, ready_thread = False, threading.Event(), threading.Event()

    def signal_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, signal_stop)
    recorders = []
    for role in ("frontend", "scheduler", "worker"):
        recorders.append(
            ProcessRecorder(
                NetworkProfilerConfig(
                    level="l1",
                    run_id=args.run,
                    output_dir=str(profiles),
                    ring_records=64,
                    context={
                        "role": role,
                        "engine_id": args.role,
                        "rank": 0,
                        "kv_role": "kv_producer"
                        if args.role == "prefill"
                        else "kv_consumer",
                    },
                )
            )
        )
        assert recorders[-1].wait_ready()
    if args.overlap_cpu is not None:

        def overlap():
            os.sched_setaffinity(0, {args.overlap_cpu})
            ready_thread.set()
            stop_thread.wait()

        threading.Thread(target=overlap, daemon=True).start()
        assert ready_thread.wait(2)
    deadline = time.monotonic() + 10
    while True:
        try:
            entries = inventory(profiles, args.run)
            if len(entries) == 3:
                break
        except (OSError, ValueError):
            pass
        assert time.monotonic() < deadline
        time.sleep(0.01)
    (directory / "ready.json").write_text(json.dumps(entries))
    closed = False
    try:
        while True:
            exiting = stop or (directory / "exit").exists()
            if not closed and (exiting or (directory / "close").exists()):
                # A deliberate slow endpoint, not synchronization by sleeping.
                if args.close_delay:
                    time.sleep(args.close_delay)
                assert all(recorder.close().complete for recorder in recorders)
                closed = True
                (directory / "closed").touch()
            if exiting:
                break
            time.sleep(0.01)
    finally:
        stop_thread.set()
        for recorder in recorders:
            recorder.close()


if __name__ == "__main__":
    main()
