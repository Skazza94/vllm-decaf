# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real NIXL prefill/decode pairs with the L1 profiler off and on."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest

from vllm.kv_profiling.clock import ClockGroup, ClockMapping
from vllm.kv_profiling.perfetto import _load, export_traces

from tests.v1.worker.simulated.test_kernel_geometry import published_path

pytest.importorskip("aisim_core")
pytest.importorskip("nixl")

TARGET = "Qwen/Qwen3-8B"
PEER = (
    "from tests.distributed.kv_transfer.profiled_peer import main\n"
    "if __name__ == '__main__':\n    main()\n"
)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_pair(tmp_path, prefill, decode, *, profile=True, wait=0.0, tolerate=False):
    """Start the producer, wait for its handoff, run the consumer to its end."""
    model = published_path(TARGET)
    script = tmp_path / "peer.py"
    script.write_text(PEER)
    run_id = str(uuid4())
    common = {"prompt": 256, "block_size": 128, "backend": "FLASHINFER"}

    def command(role, settings):
        options = {**common, **settings}
        if profile:
            options["profile"] = {
                "level": "l1",
                "run_id": run_id,
                "output_dir": str(tmp_path / "profiles"),
                "transport": "tcp",
                "clock_port_offset": 100,
                "shutdown_timeout_s": 5,
                "context": {"case": role},
            }
        env = {
            **os.environ,
            "PYTHONPATH": str(Path.cwd()),
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "VLLM_HOST_IP": "127.0.0.1",
            "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(free_port()),
            "NIXL_MOONCAKE_IP_ADDR": "127.0.0.1",
        }
        return [sys.executable, str(script), role, str(model), TARGET,
                json.dumps(options), str(tmp_path)], env  # fmt: skip

    producer_log = tmp_path / "prefill.log"
    cmd, env = command("prefill", prefill)
    with producer_log.open("w") as f:
        producer = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    try:
        end = time.monotonic() + 300
        while not (tmp_path / "prefill.json").exists():
            assert producer.poll() is None, producer_log.read_text()[-6000:]
            assert time.monotonic() < end, producer_log.read_text()[-6000:]
            time.sleep(0.2)
        time.sleep(wait)
        cmd, env = command("decode", decode)
        with (tmp_path / "decode.log").open("w") as f:
            decoder = subprocess.run(cmd, env=env, timeout=300, stdout=f,
                                     stderr=subprocess.STDOUT, check=False)  # fmt: skip
        if not tolerate:
            log = (tmp_path / "decode.log").read_text()
            assert decoder.returncode == 0, log[-6000:]
    finally:
        (tmp_path / "stop").touch()
        try:
            producer.wait(timeout=120)
        finally:
            if producer.poll() is None:
                producer.kill()
    return run_id


def steps(tmp_path, role, rank=0):
    suffix = "" if rank == 0 else f".tp{rank}"
    path = tmp_path / f"{role}{suffix}.steps"
    return [json.loads(line) for line in path.read_text().splitlines()]


def processes(tmp_path, run_id):
    """Every process directory of the run: (metadata, events with ids, summary)."""
    found = []
    for directory in sorted((tmp_path / "profiles" / run_id).iterdir()):
        meta, events, decoder, rows, _ = _load(directory / "events.jsonl")
        for event in events:
            for key in ("request", "handoff", "peer"):
                event[key] = decoder.resolve(event[f"{key}_ref"])
        summary = json.loads((directory / "profiler.json").read_text())
        found.append({"dir": directory, "meta": meta, "events": events,
                      "end": rows[-1], "summary": summary})  # fmt: skip
    return found


def by_role(found, role, case=None):
    return [
        p for p in found
        if p["meta"]["context"]["role"] == role
        and (case is None or p["meta"]["context"].get("case") == case)
    ]  # fmt: skip


def signed(value):
    return value - (1 << 64) if value >> 63 else value


def events_of(process, kind):
    return [e for e in process["events"] if e["kind"] == kind]


def test_a_profiled_pair_moves_the_same_bytes_and_traces_every_request(tmp_path):
    off = tmp_path / "off"
    on = tmp_path / "on"
    off.mkdir(), on.mkdir()
    run_pair(off, {}, {}, profile=False)
    run_id = run_pair(on, {}, {})
    for directory in (off, on):
        records = steps(directory, "decode")
        (check,) = [r["kv_check"] for r in records if "kv_check" in r]
        assert check["ok"] and check["mismatched_bytes"] == 0, check
    assert json.loads((off / "decode.json").read_text())["tokens"] == json.loads(
        (on / "decode.json").read_text()
    )["tokens"]
    # Profiling changed neither the bytes nor the descriptors NIXL moved.
    def transfers(directory):
        return [
            e for r in steps(directory, "decode")
            for e in r.get("kv_transfers", {}).get("load", {}).get("each", ())
        ]  # fmt: skip
    assert [t[:1] for t in transfers(off)] == [t[:1] for t in transfers(on)]

    found = processes(on, run_id)
    assert len(found) == 6  # frontend, scheduler and worker of each peer
    for p in found:
        assert p["summary"]["hook_errors"] == 0, p["dir"]
        assert p["summary"]["saved"] and p["end"]["complete"], (p["dir"], p["end"])
        backend = p["meta"]["context"].get("backend")
        assert backend == (None if p["meta"]["context"]["role"] == "frontend"
                           else ["Mooncake"])  # fmt: skip
    (d_worker,) = by_role(found, "worker", "decode")
    (p_worker,) = by_role(found, "worker", "prefill")
    (d_sched,) = by_role(found, "scheduler", "decode")
    (p_sched,) = by_role(found, "scheduler", "prefill")
    (d_front,) = by_role(found, "frontend", "decode")
    (p_front,) = by_role(found, "frontend", "prefill")
    (load,) = transfers(on)
    kinds = {k: [e for e in d_worker["events"] if e["kind"] == k]
             for k in {e["kind"] for e in d_worker["events"]}}  # fmt: skip
    (bind,) = kinds["transfer_bind"]
    (done,) = kinds["complete_observed"]
    assert bind["transfer"] == done["transfer"] != 0
    # NIXL's own numbers, as its stats recorded them for this READ.
    assert done["bytes"] == load[0]
    assert done["merged_descriptors"] == load[3]
    assert done["aux1"] == round(load[1] * 1e6) and done["aux0"] == round(load[2] * 1e6)
    assert done["duration0_ns"] is None
    check = [r["kv_check"] for r in steps(on, "decode") if "kv_check" in r][0]
    (read,) = check["coverage"]["reads"]
    assert bind["selected_descriptors"] == read["descriptors"]
    assert bind["selected_descriptors"] > done["merged_descriptors"]
    assert bind["aux0"] == 0 and bind["aux1"] == 1
    handoff = json.loads((on / "prefill.json").read_text())["handoff"]
    assert bind["handoff"] == handoff["remote_request_id"]
    assert bind["peer"] == handoff["remote_engine_id"]
    assert len(kinds["transfer_retire"]) == 1
    (p_handoff,) = [e for e in p_sched["events"] if e["kind"] == "handoff"]
    assert p_handoff["request"] == handoff["remote_request_id"]
    assert p_handoff["aux1"] == sum(len(g) for g in handoff["remote_block_ids"])
    (d_handoff,) = [e for e in d_sched["events"] if e["kind"] == "handoff"]
    assert d_handoff["handoff"] == handoff["remote_request_id"]
    recv = events_of(d_sched, "complete_observed")
    assert [e["subject"] for e in recv] == [2]
    release = events_of(p_sched, "release_requested")
    assert [(e["subject"], e["request"]) for e in release] == [
        (2, handoff["remote_request_id"])]  # fmt: skip
    assert len(events_of(p_sched, "handshake_end")) == 1
    notified = [e for e in p_worker["events"] if e["kind"] == "notify_observed"]
    assert [(e["aux0"], e["aux1"]) for e in notified] == [(1, 1)]
    assert len(events_of(p_worker, "senders_done")) == 1
    # Frontend bindings reach the same engine ids the connectors recorded.
    d_engine = [e for e in d_front["events"] if e["subject"] == 4]
    p_engine = [e for e in p_front["events"] if e["subject"] == 4]
    assert [e["request"] for e in p_engine] == [handoff["remote_request_id"]]
    assert d_handoff["request"] in [e["request"] for e in d_engine]
    # The clock channel calibrated the peers at start and end.
    manifest = json.loads((d_sched["dir"] / "calibration.json").read_text())
    (peer,) = manifest["peers"]
    assert peer["status"] == "ok" and not peer["end_missing"], manifest
    mapping = ClockMapping.from_json(
        json.loads((d_sched["dir"] / peer["calibration_file"]).read_text())
    )
    assert abs(float(mapping.start.offset)) <= mapping.start.raw_rtt_ns
    # NIXL's own estimate, kept as a cross-check, is a signed 64-bit value.
    offsets = [e for e in kinds["handshake_end"] if e["subject"] == 65535]
    assert len(offsets) == 1 and offsets[0]["aux0"] is not None
    nixl_offset = signed(offsets[0]["aux0"])
    print("clock offset ns", float(mapping.start.offset), "rtt ns",
          mapping.start.raw_rtt_ns, "nixl offset ns", nixl_offset)  # fmt: skip
    group = ClockGroup(
        d_sched["meta"]["clock_id"], {peer["remote_generation"]: mapping}
    )
    documents = export_traces(
        [str(p["dir"] / "events.jsonl") for p in found], str(on / "perfetto"), group
    )
    assert list(documents) == [("wall", "aligned")]
    document = json.loads(documents[("wall", "aligned")].read_text())
    assert document["metadata"]["cross_process_flows"] == "exact_recorded_handoff_ids"
    assert document["metadata"]["flows_omitted_ambiguous"] == 0
    # No helper outlived its engine: every writer and clock server exited.
    import psutil

    survivors = [
        p.info["pid"] for p in psutil.process_iter(["pid", "cmdline"])
        if any("vllm.kv_profiling" in part for part in (p.info["cmdline"] or []))
    ]  # fmt: skip
    assert survivors == []


def test_an_unequal_tp_consumer_aggregates_only_after_its_delayed_rank(tmp_path):
    """Each decode rank reads its head slice from the one producer rank; the
    scheduler's receive-finished record must wait for the delayed rank."""
    delay = 3
    run_id = run_pair(
        tmp_path,
        {},
        {"tensor_parallel_size": 2, "delay": {"rank": 1, "seconds": delay}},
    )
    found = processes(tmp_path, run_id)
    workers = sorted(
        by_role(found, "worker", "decode"), key=lambda p: p["meta"]["context"]["rank"]
    )
    assert len(workers) == 2 and len(found) == 7
    for p in found:
        assert p["summary"]["hook_errors"] == 0 and p["end"]["complete"], p["dir"]
    binds = [events_of(w, "transfer_bind") for w in workers]
    dones = [events_of(w, "complete_observed") for w in workers]
    assert [len(b) for b in binds] == [1, 1] and [len(d) for d in dones] == [1, 1]
    # The delayed rank posts at least the delay after its handshake released the
    # request (which rank's handshake finishes first is not deterministic).
    waiter = [
        [e for e in events_of(w, "handshake_end") if e["request"]][0] for w in workers
    ]
    held = [b[0]["time_ns"] - w["time_ns"] for b, w in zip(binds, waiter)]
    assert held[1] >= (delay - 0.5) * 1e9, held
    (d_sched,) = by_role(found, "scheduler", "decode")
    (aggregate,) = [
        e for e in events_of(d_sched, "complete_observed") if e["subject"] == 2
    ]
    # The one aggregate follows both ranks' completions and the delayed post.
    assert aggregate["time_ns"] >= max(d[0]["time_ns"] for d in dones)
    assert aggregate["time_ns"] - waiter[1]["time_ns"] >= (delay - 0.5) * 1e9
    assert aggregate["time_ns"] >= binds[1][0]["time_ns"]
    assert all(d[0]["time_ns"] > b[0]["time_ns"] for b, d in zip(binds, dones))
    (p_worker,) = by_role(found, "worker", "prefill")
    assert [(e["aux0"], e["aux1"]) for e in events_of(p_worker, "notify_observed")] == [
        (1, 2),
        (2, 2),
    ]
    assert len(events_of(p_worker, "senders_done")) == 1
    (p_sched,) = by_role(found, "scheduler", "prefill")
    (release,) = events_of(p_sched, "release_requested")
    assert release["time_ns"] >= aggregate["time_ns"] - 1_000_000_000
    # Two decode ranks read one producer rank's request: an unresolved fan-in
    # gets no arrow, and the export says it left one out.
    group = ClockGroup(d_sched["meta"]["clock_id"], {})
    paths = [str(p["dir"] / "events.jsonl") for p in found]
    documents = export_traces(paths, str(tmp_path / "perfetto"), group)
    (document,) = [json.loads(path.read_text()) for path in documents.values()]
    assert [e for e in document["traceEvents"] if e.get("cat") == "kv_handoff"] == []
    assert document["metadata"]["flows_omitted_ambiguous"] == 1
    # Selected counts and NIXL's own numbers match the step log of each rank.
    for rank, worker in enumerate(workers):
        (check,) = [r["kv_check"] for r in steps(tmp_path, "decode", rank)
                    if "kv_check" in r]  # fmt: skip
        (read,) = check["coverage"]["reads"]
        assert binds[rank][0]["selected_descriptors"] == read["descriptors"]
        assert dones[rank][0]["bytes"] == read["bytes"]


def test_a_prefix_hit_pulls_only_the_missing_blocks(tmp_path):
    """The decode instance already holds the first block of the prompt, so the
    READ selects half of the descriptors and moves half of the bytes."""
    full = tmp_path / "full"
    part = tmp_path / "part"
    full.mkdir(), part.mkdir()
    run_full = run_pair(full, {}, {})
    run_part = run_pair(part, {}, {"warm": True, "pattern": False})
    counts = {}
    for name, directory, run_id in (("full", full, run_full), ("part", part, run_part)):
        found = processes(directory, run_id)
        (d_worker,) = by_role(found, "worker", "decode")
        (bind,) = events_of(d_worker, "transfer_bind")
        (done,) = events_of(d_worker, "complete_observed")
        (load,) = [
            e for r in steps(directory, "decode")
            for e in r.get("kv_transfers", {}).get("load", {}).get("each", ())
        ]  # fmt: skip
        assert bind["selected_descriptors"] == load[6][1]
        assert done["bytes"] == load[0]
        counts[name] = (bind["selected_descriptors"], done["bytes"])
        for p in found:
            assert p["summary"]["hook_errors"] == 0 and p["end"]["complete"]
    assert counts["part"][0] * 2 == counts["full"][0]
    assert counts["part"][1] * 2 == counts["full"][1]


def test_a_rejected_handshake_is_recorded_with_its_cause(tmp_path):
    decode = {"dtype": "float16", "pattern": False, "backend": None}
    run_id = run_pair(tmp_path, {}, decode, tolerate=True)
    found = processes(tmp_path, run_id)
    (d_worker,) = by_role(found, "worker", "decode")
    assert len(events_of(d_worker, "handshake_begin")) == 1
    assert events_of(d_worker, "transfer_bind") == []
    failures = {e["status"] for e in events_of(d_worker, "failed")}
    assert failures & {2, 3}, failures  # handshake_failed / handshake_setup_failed
    waiters = [e for e in events_of(d_worker, "handshake_end") if e["request"]]
    assert [w["status"] for w in waiters] == [1]


def test_an_expired_lease_is_recorded_on_the_producer(tmp_path):
    run_id = run_pair(
        tmp_path, {"lease": 2}, {"pattern": False}, wait=6, tolerate=True
    )
    found = processes(tmp_path, run_id)
    (p_worker,) = by_role(found, "worker", "prefill")
    expired = [e for e in events_of(p_worker, "failed") if e["status"] == 9]
    assert len(expired) == 1 and expired[0]["subject"] == 0
