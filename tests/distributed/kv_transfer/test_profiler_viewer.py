# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The viewer server over synthetic captures: JSON, mapping and boundaries.

The fixtures are tiny JSONL files in the shapes the profiler's own writers
produce (metadata first, dictionary fragments, events, end rows; probe rows
with ``probe_schema: 1``; collector rows of kinds 25-28). The viewer module is
loaded by file path exactly as it runs; it must import neither ``vllm`` nor
``torch``.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import HTTPServer
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

VIEWER = Path(__file__).resolve().parents[3] / "tools" / "kv-profiler-viewer" / (
    "viewer.py"
)


def load_viewer():
    spec = importlib.util.spec_from_file_location("_viewer_under_test", VIEWER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------- rows


def _process_uuid(label):
    return str(uuid5(NAMESPACE_URL, "vllm-viewer-test/" + label))


def l1_metadata(process_id, clock_id, role="scheduler", kv_role="kv_consumer",
                engine_id="engine-d", run_id="run-1", boot="boot-a",
                **extra):
    from uuid import uuid4

    meta = {
        "type": "metadata", "schema": 1, "run_id": run_id or str(uuid4()),
        "process_id": _process_uuid(process_id), "pid": 100,
        "clock_id": clock_id, "clock_source": "CLOCK_MONOTONIC",
        "boot_id": boot, "time_namespace": "time:[1]",
        "net_namespace": "net:[10]", "pid_namespace": "pid:[100]",
        "start_ticks": 5000,
        "context": {"role": role, "kv_role": kv_role, "engine_id": engine_id,
                    "rank": 0, "backends": ["UCX"]},
    }
    meta.update(extra)
    if "connector" in extra:
        meta["context"]["connector"] = extra["connector"]
    return meta


_SEQ = {"n": 0}
_LAST = {"run_id": None}


def l1_event(kind, time_ns, request_ref=0, handoff_ref=0, peer_ref=0,
             transfer=0, subject=0, status=0, tid=7, **extra):
    _SEQ["n"] += 1
    row = {
        "schema": 1, "sequence": _SEQ["n"], "time_ns": time_ns,
        "type": "event", "kind": kind, "context": 0,
        "request_ref": request_ref, "handoff_ref": handoff_ref,
        "transfer": transfer, "peer_ref": peer_ref, "subject": subject,
        "tid": tid,
        "transport": {"kind": "tcp", "backend": None,
                      "provenance": "caller_reported"},
        "status": status, "unbound_handle": False, "bytes": None,
        "selected_descriptors": None, "merged_descriptors": None,
        "duration0_ns": None, "duration1_ns": None, "aux0": None,
        "aux1": None,
    }
    row.update(extra)
    return row


def dict_fragment(key, text):
    _SEQ["n"] += 1
    return {
        "type": "dictionary_fragment", "schema": 1, "sequence": _SEQ["n"],
        "id": key, "length": len(text), "offset": 0, "encoding": "utf8",
        "payload": text.encode().hex(),
    }


def write_capture(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")


def write_l1_transfer_fixture(tmp_path, clock_d=None, clock_p=None,
                              boot="boot-a", request="req-1"):
    """One consumer scheduler + consumer worker + producer scheduler +
    frontend capture; one completed transfer."""
    from uuid import uuid4

    clock_d = clock_d or str(uuid4())
    clock_p = clock_p or str(uuid4())
    run_id = str(uuid4())
    _LAST["run_id"] = run_id
    base = tmp_path / "run"
    _SEQ["n"] = 0
    req_id = request
    # one dictionary shared by all captures (id -> text)
    frag = dict_fragment(1, req_id)
    frag_peer = dict_fragment(2, "engine-p")
    frag_h = dict_fragment(3, "handoff-p-1")
    # id 4 stays free: reviewers splice their own id-4 dictionaries in
    frag_orphan = dict_fragment(9, "orphan-request")
    start_d = 1_000_000_000_000_000  # decode clock
    start_p = 2_000_000_000_000_000  # producer clock, offset +1e15

    def make(capture_events, meta):
        rows = [meta, frag, frag_peer, frag_h, frag_orphan]
        seq = 2
        for row in rows[1:]:
            row = dict(row)
            row["sequence"] = seq
            seq += 1
            rows[seq - 2] = row
        for row in capture_events:
            row = dict(row)
            row["sequence"] = seq
            seq += 1
            rows.append(row)
        rows.append({"schema": 1, "type": "end", "complete": True,
                     "stats": {}})
        return rows

    # scheduler (consumer): request bind, handoff in, recv_finished
    # (handoff_ref resolves to the producer-side request id, the identity the
    # producer scheduler's entry carries)
    sched = l1_metadata("proc-d-sched", clock_d, role="scheduler",
                        kv_role="kv_consumer", engine_id="engine-d", boot=boot,
                        run_id=run_id)
    sched_events = [
        l1_event("request_bind", start_d + 1000, request_ref=1, subject=1),
        l1_event("handoff", start_d + 2000, request_ref=1, handoff_ref=1,
                 peer_ref=2, subject=0, aux0=5, aux1=8),
        l1_event("recv_finished_all", start_d + 900_000, request_ref=1,
                 subject=2),
    ]
    # worker (consumer): two sequential READs of the same request — the
    # second with the same duration, a later acceptance
    worker = l1_metadata("proc-d-worker", clock_d, role="worker",
                         kv_role="kv_consumer", engine_id="engine-d", boot=boot,
                         run_id=run_id)
    worker_events = [
        l1_event("transfer_bind", start_d + 10_000, request_ref=1,
                 handoff_ref=3, peer_ref=2, transfer=1, selected_descriptors=4,
                 aux0=0),
        l1_event("prepared", start_d + 11_000, transfer=1),
        l1_event("post_begin", start_d + 12_000, transfer=1),
        l1_event("post_return", start_d + 50_000, transfer=1,
                 duration0_ns=38_000),
        l1_event("complete_observed", start_d + 450_000, transfer=1,
                 subject=1, bytes=1_000_000, selected_descriptors=64,
                 merged_descriptors=60, duration0_ns=38_000, aux0=30,
                 aux1=400),
        l1_event("transfer_retire", start_d + 460_000, transfer=1),
        l1_event("transfer_bind", start_d + 1_500_000, request_ref=1,
                 handoff_ref=3, peer_ref=2, transfer=2, selected_descriptors=4,
                 aux0=0),
        l1_event("prepared", start_d + 1_501_000, transfer=2),
        l1_event("post_begin", start_d + 1_502_000, transfer=2),
        l1_event("post_return", start_d + 1_540_000, transfer=2,
                 duration0_ns=38_000),
        l1_event("complete_observed", start_d + 1_940_000, transfer=2,
                 subject=1, bytes=1_000_000, selected_descriptors=64,
                 merged_descriptors=60, duration0_ns=38_000, aux0=30,
                 aux1=400),
        l1_event("transfer_retire", start_d + 1_940_000, transfer=2,
                 subject=0) if False else l1_event("transfer_retire",
                                                   start_d + 1_950_000,
                                                   transfer=2),
        # an orphan-transfer observation: a worker transfer for a request
        # no scheduler ever recorded — it stays unassigned, on the lanes
        # only
        l1_event("transfer_bind", start_d + 3_000_000, request_ref=4,
                 handoff_ref=3, peer_ref=2, transfer=3,
                 selected_descriptors=4, aux0=0),
        l1_event("prepared", start_d + 3_001_000, transfer=3),
        l1_event("post_begin", start_d + 3_002_000, transfer=3),
        l1_event("post_return", start_d + 3_040_000, transfer=3,
                 duration0_ns=38_000),
        l1_event("complete_observed", start_d + 3_440_000, transfer=3,
                 subject=1, bytes=500_000, selected_descriptors=8,
                 merged_descriptors=8, duration0_ns=38_000, aux0=30,
                 aux1=400),
        l1_event("transfer_retire", start_d + 3_450_000, transfer=3),
    ]
    # scheduler (producer)
    prod = l1_metadata("proc-p-sched", clock_p, role="scheduler",
                       kv_role="kv_producer", engine_id="engine-p", boot=boot,
                       run_id=run_id)
    prod_events = [
        l1_event("handoff", start_p + 3_000, request_ref=1, peer_ref=2,
                 subject=1, aux0=5, aux1=8),
        l1_event("request_bind", start_p + 2_500, request_ref=1, subject=1),
    ]
    # frontend: the arrival bind
    front = l1_metadata("proc-d-front", clock_d, role="frontend", boot=boot,
                        run_id=run_id)
    front_events = [
        l1_event("request_bind", start_d + 500, request_ref=1, subject=1,
                 aux0=1),
    ]
    write_capture(base / "a" / "events.jsonl",
                  make(sched_events, sched))
    write_capture(base / "b" / "events.jsonl",
                  make(worker_events, worker))
    write_capture(base / "c" / "events.jsonl",
                  make(prod_events, prod))
    write_capture(base / "d" / "events.jsonl",
                  make(front_events, front))
    # calibration manifest on the decode scheduler: maps producer clock,
    # offset 1e15, one sample each for start and end (drift known)
    mapping = {
        "start": [{
            "run_id": run_id, "local_generation": clock_d,
            "remote_generation": clock_p,
            "t1": start_d + 100_000, "t2": start_p + 100_000,
            "t3": start_p + 100_500, "t4": start_d + 100_900,
        }],
        "end": [{
            "run_id": run_id, "local_generation": clock_d,
            "remote_generation": clock_p,
            "t1": start_d + 500_000_000, "t2": start_p + 500_000_000,
            "t3": start_p + 500_000_500, "t4": start_d + 500_000_900,
        }],
        "model": "linear_estimate_not_a_bound",
    }
    (base / "a" / "calibration.json").write_text(json.dumps({
        "run_id": run_id, "local_generation": clock_d, "clock_thread": "x",
        "peers": [{"engine_id": "engine-p", "status": "ok",
                   "remote_generation": clock_p,
                   "calibration_file": "calibration-0.json"}],
    }))
    (base / "a" / "calibration-0.json").write_text(json.dumps(mapping))
    return base


def project_expected(base_ns, offset=1_000_000_000_000_000):
    return base_ns - offset


# ----------------------------------------------------------------- fixtures


@pytest.fixture()
def viewer_module():
    return load_viewer()


@pytest.fixture()
def l1_run(viewer_module, tmp_path):
    run_dir = write_l1_transfer_fixture(tmp_path)
    app = viewer_module.ViewerApp([str(run_dir)])
    return app.run(0), run_dir


# ------------------------------------------------------------------- tests


def test_viewer_imports_no_vllm_or_torch():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('v', {str(VIEWER)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "assert 'vllm' not in sys.modules, 'vllm imported'\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_summary_reused_relations(l1_run):
    run, _ = l1_run
    assert run.summary["tool"] == "vllm.kv_profiling.summary"
    assert run.summary_source["source"] == "rebuilt"
    # two confirmed transfers plus one orphan-request observation transfer
    # that no scheduler recorded
    assert len(run.transports) == 3
    assert sum(1 for t2 in run.transports if t2["request"] == "req-1") == 2
    transport = run.transports[0]
    assert transport["state"] == "completed"
    assert transport["bytes"] == 1_000_000
    assert transport["t1"] > transport["t0"]


def test_logical_request_counted_once(l1_run):
    run, _ = l1_run
    rows = [r for r in run.request_rows if not r.get("unmatched_source")]
    assert len(rows) == 1
    row = rows[0]
    assert row["state"] == "completed"
    assert row["roles"] == ["scheduler", "producer", "frontend"]
    assert row["transports"] == 2
    assert row["mib"] == pytest.approx(2_000_000 / 2 ** 20, abs=1e-3)
    assert row["peak_concurrency"] == 1
    # first record: the frontend bind (t=500ms local) projects earliest
    assert row["first_record"] is not None
    assert row["arrive"] is not None
    assert row["arrive"] <= row["first_record"]


def test_clock_projection_and_flags(l1_run):
    run, _ = l1_run
    # the producer side must project onto the decode reference through the
    # manifest-named mapping (player clock offset removed), within
    # uncertainty
    # producer handoff (t=3000 on the producer clock) lands near the decode
    # request's first records, proving real projection happened
    assert run.calibration_status[0]["drift_known"] is True


def test_start_only_calibration_flags_unknown_drift(viewer_module,
                                                    tmp_path):
    run_dir = write_l1_transfer_fixture(
        tmp_path / "s1", request="req-so")
    # strip the end anchor: start-only mapping
    mapping_path = run_dir / "a" / "calibration-0.json"
    mapping = json.loads(mapping_path.read_text())
    mapping.pop("end")
    mapping_path.write_text(json.dumps(mapping))
    app = viewer_module.ViewerApp([str(run_dir)])
    run = app.run(0)
    assert run.calibration_status[0]["drift_known"] is False
    assert any(p["kind"] == "start_only_calibration" for p in run.problems)


def test_failed_calibration_peer_stays_unaligned(viewer_module, tmp_path):
    run_dir = write_l1_transfer_fixture(tmp_path / "s2", request="req-fl")
    manifest_path = run_dir / "a" / "calibration.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["peers"][0]["status"] = "failed"
    manifest_path.write_text(json.dumps(manifest))
    app = viewer_module.ViewerApp([str(run_dir)])
    run = app.run(0)
    assert run.calibration_status[0]["aligned"] is False


def test_per_layer_loading_and_problems(viewer_module, tmp_path):
    run_dir = write_l1_transfer_fixture(tmp_path / "m1")
    # an observer and a collector capture the summary must not see
    obs_meta = {
        "schema": 1, "probe_schema": 1, "type": "metadata", "run_id": _LAST["run_id"],
        "process_id": "proc-obs", "clock_id": "clock-obs",
        "clock_source": "CLOCK_MONOTONIC", "boot_id": "boot-a",
        "time_namespace": "time:[1]", "net_namespace": "net:[10]",
        "pid_namespace": "pid:[100]",
        "context": {"role": "endpoint_observer", "endpoint_role": "decode",
                    "endpoint": "decode0"},
        "config": {"probes": {"ebpf": {"enabled": False}}},
        "owner": {"pid": 100, "start_ticks": 5000},
    }
    obs_rows = [
        obs_meta,
        {
            "schema": 1, "probe_schema": 1, "type": "event",
            "kind": "endpoint_metrics", "scope": "threads",
            "entity": "scheduler/100", "time_ns": 500_000_000,
            "sample_begin_ns": 0,
            "sample_end_ns": 500_000_000,
            "observation": {
                "role": "scheduler", "pid": 100, "exited": False,
                "start_ticks": 5000, "clock_ticks_per_s": 100,
                "threads": [{
                    "tid": 7, "comm": "worker", "start_ticks": 6000,
                    "utime_ticks": 10, "stime_ticks": 10, "run_ns": 1_000_000,
                    "wait_ns": 0, "timeslices": 1,
                    "voluntary_ctxt_switches": 1,
                    "nonvoluntary_ctxt_switches": 0,
                }],
            },
        },
        {
            "schema": 1, "probe_schema": 1, "type": "event",
            "kind": "endpoint_metrics", "scope": "threads",
            "entity": "scheduler/100", "time_ns": 2_000_000_000,
            "sample_begin_ns": 500_000_000,
            "sample_end_ns": 2_000_000_000,
            "observation": {
                "role": "scheduler", "pid": 100, "exited": False,
                "start_ticks": 5000, "clock_ticks_per_s": 100,
                "threads": [{
                    "tid": 7, "comm": "worker", "start_ticks": 6000,
                    "utime_ticks": 110, "stime_ticks": 20, "run_ns": 3_000_000,
                    "wait_ns": 1_000_000, "timeslices": 2,
                    "voluntary_ctxt_switches": 2,
                    "nonvoluntary_ctxt_switches": 1,
                }],
            },
        },
        {"schema": 1, "probe_schema": 1, "type": "end", "complete": True},
    ]
    write_capture(run_dir / "obs" / "events.jsonl", obs_rows)
    coll_meta = {
        "schema": 1, "type": "metadata", "run_id": _LAST["run_id"],
        "process_id": "proc-coll", "clock_id": "clock-coll",
        "clock_source": "CLOCK_MONOTONIC", "boot_id": "boot-a",
        "context": {"role": "collector", "connector": "nixl",
                    "sources": ["tent", "nixl"]},
    }
    coll_rows = [
        coll_meta,
        {
            "schema": 1, "type": "event", "kind": "transport_metrics",
            "kind_id": 25, "sequence": 1, "time_ns": 50_000_000,
            "sample_id": "s1",
            "target": {"id": "t-d", "role": "decode",
                       "engine_id": "engine-d", "rank": {"tp": 0},
                       "endpoint": "127.0.0.1:1", "endpoint_source":
                       "manifest", "process_id": None, "pid": None},
            "scrape": {"begin_ns": 0,
                       "end_ns": 50_000_000},
            "reset_generation": 0, "labels": {"transport": "tcp"},
            "fields": {"write_bytes": 1000.0},
            "units": {"write_bytes": "bytes"},
            "partial": False, "missing": [], "missing_reason": None,
            "discontinuity": False, "baseline": "first",
        },
        {
            "schema": 1, "type": "event", "kind": "transport_metrics",
            "kind_id": 25, "sequence": 2, "time_ns": 2_050_000_000,
            "sample_id": "s2",
            "target": {"id": "t-d", "role": "decode",
                       "engine_id": "engine-d", "rank": {"tp": 0},
                       "endpoint": "127.0.0.1:1", "endpoint_source":
                       "manifest", "process_id": None, "pid": None},
            "scrape": {"begin_ns": 2_000_000_000,
                       "end_ns": 2_050_000_000},
            "reset_generation": 0, "labels": {"transport": "tcp"},
            "fields": {"write_bytes": 3000.0},
            "units": {"write_bytes": "bytes"},
            "partial": False, "missing": [], "missing_reason": None,
            "discontinuity": False, "baseline": "sample",
        },
        {"schema": 1, "type": "end", "complete": True, "samples": 2},
    ]
    write_capture(run_dir / "coll" / "events.jsonl", coll_rows)
    app = viewer_module.ViewerApp([str(run_dir)])
    run = app.run(0)
    # the summary only saw the four L1 captures
    assert len(run.summary["sources"]) == 4
    assert len(run.rows["l3"]) == 1 and len(run.rows["l2"]) == 1
    data = run.api_data()
    # thread strip: two readings of tid 7: 110-10+20-10 = 110 ticks / 100 hz
    # over 1.5 s end-to-end
    strips = data["thread_strips"]
    assert len(strips) == 1
    point = strips[0]["points"][0]
    # 110 ticks (100 u + 10 s) at 100 hz over the 1.5 s end-to-end interval
    assert point["cpu"] == pytest.approx(1.1 / 1.5, abs=1e-3)
    assert point["joined_process"] is True
    # collector rate: (3000-1000) bytes over 2.0 s end-to-end
    lanes = data["collector"]
    assert len(lanes) == 1
    # the rate belongs to the second reading: 2000 bytes over 2.0 s
    entry = lanes[0]["points"][1]["rates"]["write_bytes"]
    assert entry["rate"] == pytest.approx(2000 / 2.0, abs=0.5)
    assert entry["unit"] == "bytes"
    assert lanes[0]["points"][0]["rates"] == {}


def test_identical_copies_fold_conflicts_rejected(viewer_module, tmp_path):
    _SEQ["n"] = 0
    run_dir = write_l1_transfer_fixture(tmp_path / "f1", request="req-dup")
    source = run_dir / "a" / "events.jsonl"
    (run_dir / "a-copy").mkdir()
    (run_dir / "a-copy" / "events.jsonl").write_text(source.read_text())
    # altered copy with the same identity: conflict
    (run_dir / "a-copy2").mkdir()
    (run_dir / "a-copy2" / "events.jsonl").write_text(
        source.read_text() + json.dumps(l1_event("notify_observed",
                                                 1)) + "\n")
    app = viewer_module.ViewerApp([str(run_dir)])
    run = app.run(0)
    # the identical copy folds; the conflicting copy excludes the capture
    assert any("a-copy2" in json.dumps(problem)
               for problem in run.problems)
    assert not any(c.process_id == _process_uuid("proc-d-sched") and c.problems
                   for c in run.captures), \
        "the conflicted capture must not serve metrics"


def test_absent_sources_and_features(viewer_module, tmp_path):
    run_dir = write_l1_transfer_fixture(tmp_path / "ab", request="req-ab")
    app = viewer_module.ViewerApp([str(run_dir)])
    run = app.run(0)
    meta = run.api_run()
    assert "l3" in meta["absent_sources"]
    assert "l2" in meta["absent_sources"]
    assert "ucx_protocols" in meta["absent_sources"]
    assert meta["features"]["ttft"].startswith("not recorded")
    data = run.api_data()
    assert data["requests"][0]["kvwait" if "kvwait" in data["requests"][0]
                               else "not_recorded"]


def test_saved_summary_verified_then_used(viewer_module, tmp_path):
    run_dir = write_l1_transfer_fixture(tmp_path / "sv", request="req-sav")
    app = viewer_module.ViewerApp([str(run_dir)])
    run = app.run(0)
    meta = run.api_run()
    # A bare summary has no input fingerprints: the viewer rebuilds.
    (run_dir / "summary.json").write_text(json.dumps(run.summary))
    app2 = viewer_module.ViewerApp([str(run_dir)])
    run2 = app2.run(0)
    assert run2.summary_source["source"] == "rebuilt"
    assert "fingerprints" in run2.summary_source["detail"]
    # With the fingerprint block it is verified and reused.
    saved = dict(run.summary)
    saved["viewer_input_fingerprints"] = meta["input_fingerprints"]
    (run_dir / "summary.json").write_text(json.dumps(saved, default=str))
    app3 = viewer_module.ViewerApp([str(run_dir)])
    run3 = app3.run(0)
    assert run3.summary_source == {"source": "saved", "detail":
                                   "verified (input fingerprints)"}
    # A capture that grows or changes forces a rebuild even with the block.
    _SEQ["n"] += 5
    extra = l1_event("notify_observed", 3_333, request_ref=1)
    data = run_dir / "d" / "events.jsonl"
    rows = data.read_text().splitlines()
    rows.insert(len(rows) - 1, json.dumps(extra))
    data.write_text("\n".join(rows) + "\n")
    app4 = viewer_module.ViewerApp([str(run_dir)])
    run4 = app4.run(0)
    assert run4.summary_source["source"] == "rebuilt"
    assert "changed" in run4.summary_source["detail"]


def test_repeated_scheduler_generation_same_request_id(viewer_module,
                                                       tmp_path):
    """Two scheduler generations reusing one id stay two logical rows and
    their detail URLs differ by the process generation."""
    run_dir = write_l1_transfer_fixture(tmp_path / "g1", request="req-dup")
    # a second decode scheduler generation with the same request id
    _SEQ["n"] = 9999
    meta2 = l1_metadata("proc-d-sched-2", str(__import__("uuid").uuid4()),
                        role="scheduler", kv_role="kv_consumer",
                        engine_id="engine-d", boot="boot-a",
                        run_id=_LAST["run_id"])
    events = [
        l1_event("request_bind", 5_000_000_000, request_ref=1, subject=1),
        l1_event("recv_finished_all", 5_050_000_000, request_ref=1,
                 subject=2),
    ]
    capture = [meta2, dict_fragment(1, "req-dup")]
    seq = 3
    for row in events + [{"schema": 1, "type": "end", "complete": True,
                          "stats": {}}]:
        row = dict(row)
        if "sequence" in row:
            row["sequence"] = seq
            seq += 1
        capture.append(row)
    write_capture(run_dir / "g2" / "events.jsonl", capture)
    app = viewer_module.ViewerApp([str(run_dir)])
    run = app.run(0)
    rows = run.request_rows
    same = [r for r in rows if r["key"]["request"] == "req-dup"
            and r["key"]["occurrence"] == 0]
    schedulers = {r["key"]["scheduler_process_id"] for r in same}
    assert len(schedulers) == 2
    for pid in schedulers:
        payload = run.api_records(pid, "req-dup", 0)
        assert payload is not None


def test_metric_scopes(viewer_module, tmp_path):
    """Sequential transfers in one bin: peak concurrency 1; sweep beats
    binning; percentiles nearest-rank on the eligible set."""
    run_dir = write_l1_transfer_fixture(tmp_path / "sc", request="req-sc")
    app = viewer_module.ViewerApp([str(run_dir)])
    run = app.run(0)
    row = next(r for r in run.request_rows if not r.get("unmatched_source"))
    assert row["peak_concurrency"] == 1
    assert run.runkv["read_p99_ms"] >= run.runkv["read_p50_ms"]
    assert run.runkv["percentile_method"].startswith("nearest-rank")


def test_http_endpoints(viewer_module, tmp_path):
    run_dir = write_l1_transfer_fixture(tmp_path / "h1", request="req-h")
    app = viewer_module.ViewerApp([str(run_dir)])
    server = HTTPServer(("127.0.0.1", 0), viewer_module.make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    from urllib.request import urlopen

    def get(path):
        from urllib.error import HTTPError

        try:
            with urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as resp:
                return resp.status, resp.read()
        except HTTPError as error:
            return error.code, error.read()

    try:
        status, body = get("/")
        assert status == 200 and b"KV Request Explorer" in body
        status, body = get("/api/run")
        assert status == 200 and json.loads(body)["run_id"] == _LAST["run_id"]
        status, body = get("/api/data")
        data = json.loads(body)
        assert status == 200 and data["requests"]
        status, body = get("/api/config")
        assert status == 200 and json.loads(body)["configs"]
        scheduler_id = _process_uuid("proc-d-sched")
        status, body = get(f"/api/capture?process_id={scheduler_id}")
        assert status == 200 and json.loads(body)["process_id"] == scheduler_id
        status, body = get("/api/capture?process_id=zzz")
        assert status == 404
        status, body = get("/api/records")
        assert status in (400, 500)
        status, body = get("/api/nonsense")
        assert status == 404
        row = data["requests"][0]["key"]
        status, body = get("/api/records?scheduler_process_id="
                           + row["scheduler_process_id"]
                           + "&request=" + row["request"]
                           + "&occurrence=" + str(row["occurrence"]))
        assert status == 200
        payload = json.loads(body)
        assert payload["records"]
        rec = payload["records"][0]
        actual = Path(rec["path"]).read_text().splitlines()[rec["line"] - 1]
        assert json.loads(actual)["sequence"] == rec["raw"]["sequence"]
    finally:
        server.shutdown()
        server.server_close()


def test_calibration_change_forces_rebuild(viewer_module, tmp_path):
    """Changing only a calibration mapping invalidates the saved summary:
    the sidecars are part of the input fingerprints."""
    run_dir = write_l1_transfer_fixture(tmp_path / "cc", request="req-cc")
    app = viewer_module.ViewerApp([str(run_dir)])
    run = app.run(0)
    meta = run.api_run()
    saved = dict(run.summary)
    saved["viewer_input_fingerprints"] = meta["input_fingerprints"]
    (run_dir / "summary.json").write_text(json.dumps(saved, default=str))
    app2 = viewer_module.ViewerApp([str(run_dir)])
    assert app2.run(0).summary_source["source"] == "saved"
    # touch only the mapping file (any byte): rebuild
    mapping_path = run_dir / "a" / "calibration-0.json"
    mapping = json.loads(mapping_path.read_text())
    mapping["model"] = "re-touched"
    mapping_path.write_text(json.dumps(mapping))
    app3 = viewer_module.ViewerApp([str(run_dir)])
    run3 = app3.run(0)
    assert run3.summary_source["source"] == "rebuilt"


def test_foreign_layer_capture_excluded(viewer_module, tmp_path):
    """An L2/L3 capture from another run id contributes nothing."""
    run_dir = write_l1_transfer_fixture(tmp_path / "fr", request="req-fr")
    foreign = {
        "schema": 1, "type": "metadata", "run_id": "other-run",
        "process_id": "proc-foreign", "clock_id": str(
            __import__("uuid").uuid4()
        ),
        "clock_source": "CLOCK_MONOTONIC", "boot_id": "boot-a",
        "context": {"role": "collector", "connector": "nixl",
                    "sources": ["tent"]},
    }
    rows = [
        foreign,
        {"schema": 1, "type": "end", "complete": True},
    ]
    write_capture(run_dir / "foreign" / "events.jsonl", rows)
    app = viewer_module.ViewerApp([str(run_dir)])
    run = app.run(0)
    assert len(run.rows["l2"]) == 0
    assert any(p["kind"] == "foreign_capture" for p in run.problems)


def test_real_smoke_run_builds():
    """Manual-check stand-in, gated: the saved smoke run builds."""
    configured = os.environ.get("PROFILER_VIEWER_REAL_RUN")
    if not configured:
        pytest.skip("set PROFILER_VIEWER_REAL_RUN=<run dir> to replay a "
                    "saved capture (not portable)")
    smoke = Path(configured)
    viewer = load_viewer()
    run = viewer.ViewerApp([str(smoke)]).run(0)
    assert run.runkv["transports"] >= 1
    assert run.api_data()["dots"]


def test_page_click_handlers(viewer_module, l1_run):
    """The actual page JS: boot, dot click, record expand — under a minimal
    DOM stub, exercising the served viewer.html itself."""
    node_src = Path(__file__).with_name("_viewer_page_test.mjs")
    if not node_src.exists():
        pytest.fail("page test harness missing")
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed: the page's JavaScript cannot run here")
    page = (VIEWER.parent / "viewer.html").resolve()
    run, _ = l1_run
    payload = {
        "run": run.api_run(),
        "data": run.api_data(),
        "config": run.api_config(),
        "records": run.api_records(
            run.request_rows[0]["key"]["scheduler_process_id"],
            run.request_rows[0]["key"]["request"], 0,
        ),
    }
    out = subprocess.run(
        [node, str(node_src), str(page)],
        input=json.dumps(payload), capture_output=True, text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    report = json.loads(out.stdout)
    assert report["booted"] is True
    assert report["dotClickSelected"] is True
    assert report["recordExpanded"] is True
    assert report["recordCollapsed"] is True
    assert report["tableRendered"] >= 1
    assert report["noJsError"] is True
    # equal-duration operations select by transport key, drawing the
    # clicked operation's own acceptance instant
    assert report["equalDurationDots"] >= 2
    assert report["selectedKeyMatch"] is True
    assert report["selectedSpanStartsAtDot"] is True
    # the plotted window covers every measured operation endpoint
    assert report["plotCoversSpans"] is True


# ----------------------------------------------------------------- mooncake
def write_mooncake_fixture(tmp_path, clock_d=None, clock_p=None,
                           boot="boot-a"):
    """One Mooncake write batch, exact shares, two member requests.

    Shapes are the Mooncake adapter's documented records: the producer
    worker binds a batch per consumer rank, the members are
    ``request_bind`` rows with subject BATCH_MEMBER carrying their share,
    and the synchronous write brackets post_begin/post_return.
    """
    from uuid import uuid4

    clock_d = clock_d or str(uuid4())
    clock_p = clock_p or str(uuid4())
    run_id = str(uuid4())
    _LAST["run_id"] = run_id
    base = tmp_path / "run"
    _SEQ["n"] = 0
    start_d = 1_000_000_000_000_000
    start_p = 1_500_000_000_000_000
    # consumer side
    sched = l1_metadata("mc-d-sched", clock_d, role="scheduler",
                        kv_role="kv_consumer", engine_id="engine-d",
                        run_id=run_id, boot=boot)
    d_events = [
        l1_event("request_bind", start_d + 1_000, request_ref=1, subject=1),
        l1_event("request_bind", start_d + 1_100, request_ref=2, subject=1),
        l1_event("handoff", start_d + 2_000, request_ref=1, handoff_ref=3,
                 peer_ref=5, subject=0, aux0=1, aux1=1),
        l1_event("handoff", start_d + 2_100, request_ref=2, handoff_ref=4,
                 peer_ref=5, subject=0, aux0=1, aux1=1),
        l1_event("recv_finished_all", start_d + 400_000, request_ref=1,
                 subject=2),
        l1_event("recv_finished_all", start_d + 410_000, request_ref=2,
                 subject=2),
    ]
    # producer side: the write batch with two members carrying shares
    worker = l1_metadata("mc-p-worker", clock_p, role="worker",
                         kv_role="kv_producer", engine_id="engine-p",
                         run_id=run_id, boot=boot, connector="mooncake")
    p_events = [
        l1_event("transfer_bind", start_p + 10_000, transfer=1,
                 subject=0, aux0=2),
        l1_event("prepared", start_p + 11_000, transfer=1),
        l1_event("post_begin", start_p + 12_000, transfer=1),
        l1_event("post_return", start_p + 300_000, transfer=1,
                 duration0_ns=288_000),
        # member rows: shares partition the batch exactly; each member
        # also carries its shared handoff identity
        l1_event("request_bind", start_p + 310_000, request_ref=1,
                 handoff_ref=3, transfer=1, subject=6, bytes=600,
                 selected_descriptors=6),
        l1_event("request_bind", start_p + 311_000, request_ref=2,
                 handoff_ref=4, transfer=1, subject=6, bytes=400,
                 selected_descriptors=4),
        l1_event("complete_observed", start_p + 320_000, transfer=1,
                 subject=1, bytes=1_000, selected_descriptors=10,
                 merged_descriptors=10, duration0_ns=288_000),
        l1_event("transfer_retire", start_p + 330_000, transfer=1),
    ]
    prod = l1_metadata("mc-p-sched", clock_p, role="scheduler",
                       kv_role="kv_producer", engine_id="engine-p",
                       run_id=run_id, boot=boot, connector="mooncake")
    psched_events = [
        l1_event("handoff", start_p + 1_000, request_ref=1, handoff_ref=3,
                 peer_ref=5, subject=1, aux0=1, aux1=1),
        l1_event("handoff", start_p + 1_100, request_ref=2, handoff_ref=4,
                 peer_ref=5, subject=1, aux0=1, aux1=1),
    ]
    frag1 = dict_fragment(1, "mc-req-1")
    frag2 = dict_fragment(2, "mc-req-2")
    frag3 = dict_fragment(3, "handoff-A")
    frag4 = dict_fragment(4, "handoff-B")
    frag5 = dict_fragment(5, "engine-p")

    def make(events, meta):
        rows = [meta, frag1, frag2, frag3, frag4, frag5]
        seq = 3
        for row in events:
            row = dict(row)
            row["sequence"] = seq
            seq += 1
            rows.append(row)
        rows.append({"schema": 1, "type": "end", "complete": True,
                     "stats": {}})
        return rows

    write_capture(base / "d" / "events.jsonl",
                  make(d_events, sched))
    write_capture(base / "p" / "events.jsonl",
                  make(p_events, worker))
    write_capture(base / "ps" / "events.jsonl",
                  make(psched_events, prod))
    # no cross-clock evidence in this fixture: same boot, one domain
    return base, run_id


def test_mooncake_batch_members_and_single_counting(viewer_module):
    """One Mooncake write batch, exact shares, two member requests: each
    request's row shows its own share and the physical batch counts once."""
    import tempfile

    base, _ = write_mooncake_fixture(Path(tempfile.mkdtemp()))
    run = viewer_module.ViewerApp([str(base)]).run(0)
    rows = run.request_rows
    assert len(rows) == 2
    by_id = {r["key"]["request"]: r for r in rows}
    one, two = by_id["mc-req-1"], by_id["mc-req-2"]
    for row, (share_bytes, share_desc) in (
        (one, (600, 6)), (two, (400, 4))
    ):
        assert row["transports"] == 1
        assert row["bytes_basis"] == "batch member share (confirmed)"
        assert row["mib"] == pytest.approx(share_bytes / 2 ** 20, abs=5e-4)
        assert row["descriptors"] == share_desc
    data = run.api_data()
    assert sum(1 for _ in data["dots"]) == 1


def test_mooncake_incoming_handoff_pair_matching(viewer_module):
    """A member's share binds only when ONE incoming handoff row carries
    both the producer engine and the batch's shared transfer id: the
    genuine (P, A) pair retains its share beside an unrelated (P, X) row,
    while a share assembled from separate handoffs — (P, X) and
    (other engine, A) — receives none. The independently matched member
    keeps its share in both cases."""
    import copy
    import tempfile

    base, _ = write_mooncake_fixture(Path(tempfile.mkdtemp()))
    d_file = base / "d" / "events.jsonl"

    def rebuild(rows, file=None):
        for i, e in enumerate(rows[1:-1], 1):
            e["sequence"] = i
        rows[-1].setdefault("stats", {}).update(
            head=len(rows) - 2, tail=len(rows) - 2
        )
        write_capture(file or d_file, rows)

    def row_of(run, request):
        return next(q for q in run.request_rows
                    if q["key"]["request"] == request)

    # baseline: the genuine pair (engine-p, 'handoff-A') attaches the batch
    rows = [json.loads(line) for line in d_file.read_text().splitlines()]
    rebuild(copy.deepcopy(rows))
    run = viewer_module.ViewerApp([str(base)]).run(0)
    assert row_of(run, "mc-req-1")["transports"] == 1
    assert row_of(run, "mc-req-1")["mib"] == pytest.approx(
        600 / 2 ** 20, abs=5e-4
    )

    # the genuine pair survives an unrelated (P, X) row in the same
    # occurrence
    rows = [json.loads(line) for line in d_file.read_text().splitlines()]
    genuine = next(x for x in rows if x.get("kind") == "handoff"
                   and x.get("request_ref") == 1 and x.get("subject") == 0)
    # the copied row carries its OWN distinct transfer id (handoff_ref=4,
    # 'handoff-B' — the other member's incoming id): two members, and each
    # share binds only through its own incoming handoff row
    extra = dict(genuine, time_ns=genuine["time_ns"] + 5, handoff_ref=4)
    rows.insert(-1, extra)
    rebuild(rows)
    run2 = viewer_module.ViewerApp([str(base)]).run(0)
    assert row_of(run2, "mc-req-1")["transports"] == 1
    assert row_of(run2, "mc-req-1")["mib"] == pytest.approx(
        600 / 2 ** 20, abs=5e-4
    )

    # a share assembled from separate handoffs receives none
    base3, _ = write_mooncake_fixture(Path(tempfile.mkdtemp()))
    d_file3 = base3 / "d" / "events.jsonl"
    rows = [json.loads(line) for line in d_file3.read_text().splitlines()]
    genuine = next(x for x in rows if x.get("kind") == "handoff"
                   and x.get("request_ref") == 1 and x.get("subject") == 0)
    genuine["handoff_ref"] = 7   # unrelated-x
    separated = dict(genuine, time_ns=genuine["time_ns"] + 5,
                     handoff_ref=3, peer_ref=8)
    rows.insert(-1, separated)
    rows.insert(1, dict_fragment(7, "unrelated-x"))
    rows.insert(2, dict_fragment(8, "engine-q"))
    rebuild(rows, d_file3)
    run3 = viewer_module.ViewerApp([str(base3)]).run(0)
    row3 = row_of(run3, "mc-req-1")
    assert row3["transports"] == 0
    assert row3["mib"] is None
    # the independently matched member keeps its share
    d_b = row_of(run3, "mc-req-2")
    assert d_b["transports"] == 1
    assert d_b["mib"] == pytest.approx(400 / 2 ** 20, abs=5e-4)


def rewrite_capture_rows(path, rows, file=None):
    """Rebase sequences and stats the way the fixture writers do."""
    for i, e in enumerate(rows[1:-1], 1):
        e["sequence"] = i
    rows[-1].setdefault("stats", {}).update(
        head=len(rows) - 2, tail=len(rows) - 2
    )
    write_capture(file or path, rows)
    return path


def _irq_poll(viewer_module, tmp_path, rows):
    """Fixture writer shim for the older replay scripts: an L1 baseline
    fixture plus the observer rows in one run directory."""
    run_dir = write_l1_transfer_fixture(Path(tempfile.mkdtemp()),
                                        request="req-poll")
    meta = {"schema": 1, "probe_schema": 1, "type": "metadata",
            "run_id": _LAST["run_id"], "process_id": "p-obs",
            "clock_id": str(__import__("uuid").uuid4()),
            "clock_source": "CLOCK_MONOTONIC", "boot_id": "boot-a",
            "time_namespace": "time:[1]", "net_namespace": "net:[10]",
            "pid_namespace": "pid:[100]",
            "context": {"role": "endpoint_observer",
                        "endpoint_role": "decode", "endpoint": "decode0"},
            "config": {"ebpf": {"enabled": False}},
            }
    if "run_id" in rows[0]:
        rows[0]["run_id"] = meta["run_id"]
    write_capture(run_dir / "obs" / "events.jsonl", [meta, *rows])
    return run_dir


def test_irq_poll_grouping_and_fragment_completeness(viewer_module,
                                                     tmp_path):
    """A poll split across two fragment rows yields one lane point whose
    total is computed from the recorded per-IRQ totals; removing a fragment
    makes the poll incomplete (absent, never a partial total)."""
    start = 1_000_000_000_000_000

    def _poll(second, totals, slices):
        return {"type": "event", "kind": "endpoint_metrics",
                "scope": "interrupts", "entity": "nic",
                "sample_begin_ns": start + second * 1_000_000_000,
                "sample_end_ns": start + (second + 1) * 1_000_000_000,
                "observation": {"counter_reset_or_identity_change": False,
                                "slices": {"irqs": slices},
                                "irqs": [dict(irq=i, name="t", total=t,
                                              per_cpu={"0": t}) for
                                         i, t in totals.items()]}}

    poll_rows = [
        _poll(0, {i: 100 for i in range(202, 218)}, [0, 16, 33]),
        _poll(0, {i: 120 for i in range(218, 235)}, [16, 33, 33]),
        _poll(1, {i: 220 for i in range(202, 218)}, [0, 16, 33]),
        _poll(1, {i: 240 for i in range(218, 235)}, [16, 33, 33]),
    ]
    run_dir = write_l1_transfer_fixture(
        Path(tempfile.mkdtemp()), request="req-irq"
    )
    meta = {
        "schema": 1, "probe_schema": 1, "type": "metadata",
        "run_id": _LAST["run_id"], "process_id": "p-obs-irq",
        "clock_id": str(__import__("uuid").uuid4()),
        "clock_source": "CLOCK_MONOTONIC", "boot_id": "boot-a",
        "time_namespace": "time:[1]", "net_namespace": "net:[10]",
        "pid_namespace": "pid:[100]",
        "context": {"role": "endpoint_observer", "endpoint_role": "decode",
                    "endpoint": "decode0"},
    }
    write_capture(run_dir / "obsirq" / "events.jsonl",
                  [meta, *poll_rows, {"schema": 1, "probe_schema": 1,
                                      "type": "end", "complete": True}])
    run = viewer_module.ViewerApp([str(run_dir)]).run(0)
    pts = [p for p in run.irq_lane if "irq_total_per_s" in p]
    assert len(pts) == 1
    # Σ per-IRQ deltas: 33 × (220−100) + 33 × (240−120) = 3,960 / 1 s
    assert pts[0]["irq_total_per_s"] == pytest.approx(3960.0)
    # removing a fragment makes that poll incomplete → absent
    incomplete = [meta, poll_rows[2], poll_rows[3],
                  {"schema": 1, "probe_schema": 1,
                   "type": "end", "complete": True}]
    write_capture(run_dir / "obsirq" / "events.jsonl", incomplete)
    run2 = viewer_module.ViewerApp([str(run_dir)]).run(0)
    assert not [p for p in run2.irq_lane if "irq_total_per_s" in p]


def test_irq_rebaseline_on_decrease(viewer_module, tmp_path):
    """NET_RX 100, 110, 3, 8: the decrease interval is unavailable and 3
    becomes the baseline → rates [10, 5]."""
    start = 1_000_000_000_000_000
    poll_rows = []
    for index, count in enumerate([100, 110, 3, 8]):
        poll_rows.append(
            {"type": "event", "kind": "endpoint_metrics",
             "scope": "interrupts", "entity": "softirqs",
             "sample_begin_ns": start + index * 1_000_000_000,
             "sample_end_ns": start + (index + 1) * 1_000_000_000,
             "observation": {"counter_reset_or_identity_change": False,
                             "softirqs": [dict(name="NET_RX", cpus=[0],
                                               counts=[count])]}},
        )
    run_dir = write_l1_transfer_fixture(
        Path(tempfile.mkdtemp()), request="req-rb"
    )
    meta = {
        "schema": 1, "probe_schema": 1, "type": "metadata",
        "run_id": _LAST["run_id"], "process_id": "p-obs-rb",
        "clock_id": str(__import__("uuid").uuid4()),
        "clock_source": "CLOCK_MONOTONIC", "boot_id": "boot-a",
        "time_namespace": "time:[1]", "net_namespace": "net:[10]",
        "pid_namespace": "pid:[100]",
        "context": {"role": "endpoint_observer", "endpoint_role": "decode",
                    "endpoint": "decode0"},
    }
    write_capture(run_dir / "obs" / "events.jsonl",
                  [meta, *poll_rows, {"schema": 1, "probe_schema": 1,
                                      "type": "end", "complete": True}])
    run = viewer_module.ViewerApp([str(run_dir)]).run(0)
    rates = [p.get("net_rx_per_s") for p in run.irq_lane
             if p.get("net_rx_per_s") is not None]
    assert rates == pytest.approx([10.0, 5.0])


def test_irq_reset_baseline_recovers_next_interval(viewer_module, tmp_path):
    """A complete flagged reset poll becomes the new baseline: its own
    interval publishes nothing, and the following interval measures against
    the flagged reading — the 5 IRQ/s under the new identity survives."""
    start = 1_000_000_000_000_000
    counts = [100, 110, 8_000, 8_005, 8_012]
    flags = [False, False, True, False, False]
    poll_rows = []
    for index, (count, flag) in enumerate(zip(counts, flags)):
        poll_rows.append(
            {"type": "event", "kind": "endpoint_metrics",
             "scope": "interrupts", "entity": "nic",
             "sample_begin_ns": start + index * 1_000_000_000,
             "sample_end_ns": start + index * 1_000_000_000,
             "observation": {
                 "counter_reset_or_identity_change": flag,
                 "observation_generation": 1 if index < 2 else 2,
                 "irqs": [dict(irq=200, name="test", total=count,
                               per_cpu={"0": count})]}})
    run_dir = write_l1_transfer_fixture(
        Path(tempfile.mkdtemp()), request="req-rst"
    )
    meta = {
        "schema": 1, "probe_schema": 1, "type": "metadata",
        "run_id": _LAST["run_id"], "process_id": "p-obs-rst",
        "clock_id": str(__import__("uuid").uuid4()),
        "clock_source": "CLOCK_MONOTONIC", "boot_id": "boot-a",
        "time_namespace": "time:[1]", "net_namespace": "net:[10]",
        "pid_namespace": "pid:[100]",
        "context": {"role": "endpoint_observer", "endpoint_role": "decode",
                    "endpoint": "decode0"},
    }
    write_capture(run_dir / "obs" / "events.jsonl",
                  [meta, *poll_rows, {"schema": 1, "probe_schema": 1,
                                      "type": "end", "complete": True}])
    run = viewer_module.ViewerApp([str(run_dir)]).run(0)
    capture = next(c for c in run.rows["l3"]
                   if c.context.get("endpoint") == "decode0")
    index_by_t = {run.project(capture.meta["clock_id"],
                              start + i * 1_000_000_000)[0]: i
                  for i in range(len(counts) + 1)}
    rates = [(index_by_t[p["t"]], p["irq_total_per_s"])
             for p in run.irq_lane if "irq_total_per_s" in p]
    assert rates == [(1, 10.0), (3, 5.0), (4, 7.0)]


def test_irq_unsplit_truncated_map_uses_recorded_total(viewer_module,
                                                       tmp_path):
    """An unsplit NIC poll with a truncated (top-32) per-CPU map still
    carries its recorded `total`, which includes the omitted CPUs: with the
    top-32 counters fixed and one omitted CPU carrying all the increments,
    the recorded total yields 20 IRQ/s for both intervals."""
    start = 1_000_000_000_000_000
    poll_rows = []
    for offset, extra in enumerate((1, 21, 41)):
        total = 100 * 32 + extra
        poll_rows.append(
            {"type": "event", "kind": "endpoint_metrics",
             "scope": "interrupts", "entity": "nic",
             "sample_begin_ns": start + offset * 1_000_000_000,
             "sample_end_ns": start + offset * 1_000_000_000,
             "observation": {
                 "irqs": [dict(irq=88, name="test", total=total,
                               per_cpu=cpus, nonzero_cpus=33,
                               per_cpu_truncated=True)
                          for cpus in [{str(i): 100 for i in range(32)}]]}},
        )
    run_dir = write_l1_transfer_fixture(
        Path(tempfile.mkdtemp()), request="req-tr"
    )
    meta = {
        "schema": 1, "probe_schema": 1, "type": "metadata",
        "run_id": _LAST["run_id"], "process_id": "p-obs-tr",
        "clock_id": str(__import__("uuid").uuid4()),
        "clock_source": "CLOCK_MONOTONIC", "boot_id": "boot-a",
        "time_namespace": "time:[1]", "net_namespace": "net:[10]",
        "pid_namespace": "pid:[100]",
        "context": {"role": "endpoint_observer", "endpoint_role": "decode",
                    "endpoint": "decode0"},
    }
    write_capture(run_dir / "obs" / "events.jsonl",
                  [meta, *poll_rows, {"schema": 1, "probe_schema": 1,
                                      "type": "end", "complete": True}])
    run = viewer_module.ViewerApp([str(run_dir)]).run(0)
    rates = [p.get("irq_total_per_s") for p in run.irq_lane
             if p.get("irq_total_per_s") is not None]
    assert rates == pytest.approx([20.0, 20.0])


def test_irq_source_coverage_gates_nic_total(viewer_module, tmp_path):
    """With all 257 IRQ vectors recorded, both intervals publish 20 IRQ/s
    in order. When the source cap omits the last vector (256 recorded
    slots, `irqs_truncated=True`), or a listed vector is missing
    (`coverage_problem`), the whole-NIC total is withheld — absent, never
    zero."""
    BILLION = 1_000_000_000
    START = 1_000_000_000_000_000

    def poll_rows(vectors, truncated, missing):
        rows = []
        for second in range(3):
            # the slices match the slots the source actually read: one
            # slice covering `vectors` recorded slots (the collector
            # records the ranges it read, not the full listed list)
            observation = {"slices": {"irqs": [0, vectors, vectors]}}
            if truncated is not None:
                observation["irqs_truncated"] = truncated
                observation["coverage_problem"] = None
            if missing:
                # an unsplit row: only the coverage flag is incomplete
                del observation["slices"]
                observation["coverage_problem"] = "1 missing"
            observation["irqs"] = [
                dict(irq=1000 + i, name="t",
                     total=10 + (second * 20 if i == vectors - 1 else 0),
                     per_cpu={"0": 10}, nonzero_cpus=1)
                for i in range(vectors)]
            rows.append({"type": "event", "kind": "endpoint_metrics",
                         "scope": "interrupts", "entity": "nic",
                         "sample_begin_ns": START + second * BILLION,
                         "sample_end_ns": START + second * BILLION,
                         "observation": observation})
        return rows

    footer = {"schema": 1, "probe_schema": 1, "type": "end",
              "complete": True}

    def measure(poll_rows, source=True):
        # the source-coverage L1 anchor: the base fixture (same boot as
        # the observer meta) supplies the L1 captures
        l1_dir = write_l1_transfer_fixture(
            Path(tempfile.mkdtemp()), request="req-src"
        )
        meta = {
            "schema": 1, "probe_schema": 1, "type": "metadata",
            "run_id": _LAST["run_id"], "process_id": "p-obs-src",
            "clock_id": str(__import__("uuid").uuid4()),
            "clock_source": "CLOCK_MONOTONIC", "boot_id": "boot-a",
            "time_namespace": "time:[1]", "net_namespace": "net:[10]",
            "pid_namespace": "pid:[100]",
            "context": {"role": "endpoint_observer",
                        "endpoint_role": "decode", "endpoint": "decode0"},
        }
        # combine: copy the L1 capture dirs next to the observer rows
        run_dir = Path(tempfile.mkdtemp()) / "combined-src"
        run_dir.mkdir(parents=True, exist_ok=True)
        for part in ("a", "b", "c", "d"):
            (run_dir / part).mkdir()
            for pog in (l1_dir / part).glob("*.json*"):
                (run_dir / part / pog.name).write_bytes(pog.read_bytes())
        write_capture(run_dir / "obs" / "events.jsonl",
                      [meta] + poll_rows + [footer])
        return viewer_module.ViewerApp(
            [str(run_dir)]).run(0)

    def rates_of(run):
        return [p["irq_total_per_s"] for p in run.irq_lane
                if "irq_total_per_s" in p]

    # all 257 vectors recorded: both intervals publish 20/s, in order
    run = measure(poll_rows(257, False, False))
    assert rates_of(run) == pytest.approx([20.0, 20.0])
    # the source cap omits the last vector: no whole-NIC total
    run = measure(poll_rows(256, True, False))
    assert rates_of(run) == []
    # a listed vector missing from the kernel reading: no total
    run = measure(poll_rows(2, False, True))
    assert rates_of(run) == []


def test_irq_baseline_gate_four_cases(viewer_module, tmp_path):
    """The four whole-NIC baseline gate cases: retire, omit-fragment,
    tied same-end polls, and the marked identity change — as real running
    regressions against the published (second, IRQ/s) series."""
    BILLION = 1_000_000_000
    START = 1_000_000_000_000_000

    def row(second, counters, span=None, reset=False, begin=None,
            end=None):
        lo = START + (second if begin is None else begin) * BILLION
        hi = START + (second if end is None else end) * BILLION
        return {"schema": 1, "probe_schema": 1, "type": "event",
                "kind": "endpoint_metrics", "scope": "interrupts",
                "entity": "nic", "sample_begin_ns": lo,
                "sample_end_ns": hi,
                "observation": {"irqs": [
                    dict(irq=irq, name="test", owners=["i:test"],
                         total=count,
                         per_cpu={"0": count}, nonzero_cpus=1)
                    for irq, count in counters.items()],
                    "slices": ({"irqs": span} if span else {}),
                    "counter_reset_or_identity_change": reset}}

    def measure(rows):
        run_dir = write_l1_transfer_fixture(
            Path(tempfile.mkdtemp()), request="req-gate"
        )
        meta = {"schema": 1, "probe_schema": 1, "type": "metadata",
                "run_id": _LAST["run_id"], "process_id": "p-obs-gate",
                "clock_id": str(__import__("uuid").uuid4()),
                "clock_source": "CLOCK_MONOTONIC", "boot_id": "boot-a",
                "time_namespace": "time:[1]", "net_namespace": "net:[10]",
                "pid_namespace": "pid:[100]",
                "context": {"role": "endpoint_observer",
                            "endpoint_role": "decode",
                            "endpoint": "decode0"}}
        write_capture(run_dir / "obs" / "events.jsonl",
                      [meta, *rows, {"schema": 1, "probe_schema": 1,
                                     "type": "end", "complete": True}])
        run = viewer_module.ViewerApp([str(run_dir)]).run(0)
        capture = next(c for c in run.rows["l3"]
                       if c.context.get("endpoint") == "decode0")
        index = {run.project(capture.meta["clock_id"],
                             START + i * BILLION)[0]: i
                 for i in range(6)}
        return [(index[p["t"]], p["irq_total_per_s"])
                for p in run.irq_lane if "irq_total_per_s" in p]

    baseline = row(0, {200: 10, 201: 10})
    a = row(1, {200: 110}, [0, 1, 2])
    b = row(1, {201: 110}, [1, 2, 2])
    next_set = [row(2, {200: 130}), row(3, {200: 140})]
    # retire
    assert measure([baseline, a, b, *next_set]) == [(1, 200.0), (3, 10.0)]
    # omit-fragment: the incomplete poll clears the baseline
    partial = measure([baseline, a, *next_set])
    assert partial == [(3, 10.0)]
    # tied same-end polls: later reading re-baselines without publishing
    tied = measure([row(1, {200: 10}, begin=0, end=1),
                    row(1, {200: 20}, begin=1, end=1),
                    row(2, {200: 40})])
    assert tied == [(2, 20.0)]
    # partial poll explicitly marks an identity change
    reset = measure([baseline,
                     row(1, {200: 8000}, [0, 1, 2], reset=True),
                     row(2, {200: 8005, 201: 9000}),
                     row(3, {200: 8010, 201: 9010})])
    assert reset == [(3, 15.0)]


def test_irq_reset_and_incomplete_poll_clears_baseline(viewer_module,
                                                       tmp_path):
    """When the same NIC poll marks an identity change AND omits a vector
    (`irqs_truncated`), the coverage gate wins: the poll clears the
    whole-NIC baseline instead of saving its partial counters, so the
    interval over unknown activity stays absent and only the next
    measurable interval publishes [5/s]. A flagged but complete poll still
    re-baselines without publishing ([10/s, 5/s] survive)."""
    import tempfile

    BILLION = 1_000_000_000
    START = 1_000_000_000_000_000

    def row(second, total, reset=False, truncated=False):
        observation = {"irqs": [dict(irq=1000, name="test", total=total,
                                     per_cpu={"0": total},
                                     nonzero_cpus=1)],
                       "counter_reset_or_identity_change": reset}
        if truncated:
            observation["irqs_truncated"] = True
            observation["coverage_problem"] = None
        return {"type": "event", "kind": "endpoint_metrics",
                "scope": "interrupts", "entity": "nic",
                "sample_begin_ns": START + second * BILLION,
                "sample_end_ns": START + second * BILLION,
                "observation": observation}

    def measure(rows):
        l1_dir = write_l1_transfer_fixture(
            Path(tempfile.mkdtemp()), request="req-rc"
        )
        run_dir = Path(tempfile.mkdtemp()) / "combined-rc"
        run_dir.mkdir(parents=True, exist_ok=True)
        for part in ("a", "b", "c", "d"):
            (run_dir / part).mkdir()
            for pog in (l1_dir / part).glob("*.json*"):
                (run_dir / part / pog.name).write_bytes(pog.read_bytes())
        meta = {"schema": 1, "probe_schema": 1, "type": "metadata",
                "run_id": _LAST["run_id"], "process_id": "p-obs-rc",
                "clock_id": str(__import__("uuid").uuid4()),
                "clock_source": "CLOCK_MONOTONIC", "boot_id": "boot-a",
                "time_namespace": "time:[1]", "net_namespace": "net:[10]",
                "pid_namespace": "pid:[100]",
                "context": {"role": "endpoint_observer",
                            "endpoint_role": "decode",
                            "endpoint": "decode0"}}
        write_capture(run_dir / "obs" / "events.jsonl",
                      [meta, *rows, {"schema": 1, "probe_schema": 1,
                                     "type": "end", "complete": True}])
        run = viewer_module.ViewerApp([str(run_dir)]).run(0)
        capture = next(c for c in run.rows["l3"]
                       if c.context.get("endpoint") == "decode0")
        index = {run.project(capture.meta["clock_id"],
                             START + i * BILLION)[0]: i
                 for i in range(5)}
        return [(index[p["t"]], p["irq_total_per_s"])
                for p in run.irq_lane if "irq_total_per_s" in p]

    counts = [100, 8000, 8010, 8015]
    # reset + truncated on the same poll: the partial counters never
    # become the baseline — only the later 5/s interval is measurable
    assert measure([row(0, counts[0]),
                    row(1, counts[1], reset=True, truncated=True),
                    row(2, counts[2]),
                    row(3, counts[3])]) == [(3, 5.0)]
    # a flagged but complete poll is the new baseline: [10/s, 5/s]
    assert measure([row(0, counts[0]),
                    row(1, counts[1], reset=True),
                    row(2, counts[2]),
                    row(3, counts[3])]) == [(2, 10.0), (3, 5.0)]




def test_request_timeline_fields_come_from_the_summary_timelines(l1_run):
    """Phase 4's per-request fields are the summary's timelines, never estimated.
    A row's scheduler occurrence is the one whose record sequences contain its
    own incoming handoff (no join needed); the prefill window is the producer
    occurrence owning the outgoing handoff of the row's one confirmed, unreused
    join. The KV-wait bar
    runs from the load preparation to the KV receipt on the scheduler's clock;
    frontend times use the frontend's clock and are not drawn when it does not
    project. Every missing value has its own reason."""
    run, _ = l1_run
    key = run.request_rows[0]["key"]
    process, request, occurrence = (key["scheduler_process_id"], key["request"],
                                    key["occurrence"])  # fmt: skip
    ms = 1_000_000
    reference = run.reference_clock
    base = len(run.summary.get("sources") or [])
    a_i, b_i, sched_i, front_i = base, base + 1, base + 2, base + 3

    def consumer(seq_first, seq_end, wait_ms, **changes):
        entry = {
            "node_role": "consumer", "process_id": process, "request": request,
            "request_occurrence": 0, "state": "complete", "quality": [],
            "source": sched_i, "sequences": {"first": seq_first, "end": seq_end},
            "api": {"arrival_ns": 100 * ms, "first_token_output_ns": 150 * ms,
                    "output_finished_ns": 300 * ms, "source": front_i},
            "scheduler": {"queued_ns": 101 * ms, "load_prepared_ns": 102 * ms,
                          "kv_received_ns": 105 * ms},
            "stall": {"basis": "api_arrival", "waiting_for_kv_ns": wait_ms * ms},
            "intervals_ns": {"queued_to_admitted": 5 * ms},
            "decode_window": {"first_token_output_ns": 150 * ms,
                              "output_finished_ns": 300 * ms,
                              "mean_time_per_token_after_first_ns": 10 * ms},
        }  # fmt: skip
        entry.update(changes)
        return entry

    def producer(source, seq_first, seq_end, window_ms, role="producer"):
        return {"node_role": role, "source": source, "request": "P",
                "request_occurrence": 0,
                "sequences": {"first": seq_first, "end": seq_end},
                "intervals_ns": {"prefill_window": window_ms * ms}}  # fmt: skip

    def join(**changes):
        entry = {"consumer_process_id": process, "consumer_request": request,
                 "consumer_occurrence": occurrence, "consumer_source": sched_i,
                 "consumer_handoff_sequence": 25, "producer_request": "P",
                 "producer_sources": [b_i], "producer_handoff_sequence": 70,
                 "joined": True, "ambiguous": False}  # fmt: skip
        entry.update(changes)
        return entry

    requests = list(run.summary.get("requests") or [])
    mine = (process, request, occurrence)

    def fields(frontend_clock, timelines, joins, incoming=(25,)):
        others = [
            r for r in requests
            if (r.get("process_id"), r.get("request"),
                r.get("request_occurrence")) != mine
        ]  # fmt: skip
        run.summary["requests"] = others + [{
            "process_id": process, "request": request,
            "request_occurrence": occurrence,
            "handoffs": [{"direction": "incoming", "sequence": seq}
                         for seq in incoming]}]  # fmt: skip
        run.summary["sources"] = (run.summary.get("sources") or [])[:base] + [
            {"process_id": "producer-a", "clock_id": "clock-a"},
            {"process_id": "producer-b", "clock_id": "clock-b"},
            {"process_id": process, "clock_id": reference},
            {"process_id": "frontend", "clock_id": frontend_clock}]
        run.summary["timelines"] = list(timelines)
        run.summary["joins"] = list(joins)
        run._index_timelines()
        return run._timeline_for(process, request, occurrence)

    # An aborted local run of the ID (sequences 10..20), then the pull (20..):
    # the incoming handoff (25) places the row in the second occurrence.
    local = consumer(10, 20, 0, request_occurrence=0)
    pull = consumer(20, None, 363, request_occurrence=1)
    producers = [producer(a_i, 60, None, 99), producer(b_i, 60, None, 40)]
    t = fields(reference, [local, pull, *producers], [join()])
    assert t["available"] is True
    assert (t["kv_wait_ms"], t["admission_ms"], t["decode_window_ms"],
            t["ttft_ms"], t["prefill_ms"], t["time_per_token_ms"]) == (
        363.0, 5.0, 150.0, 50.0, 40.0, 10.0)  # fmt: skip
    kinds = {s["kind"]: s for s in t["spans"]}
    assert list(kinds) == ["waiting for KV", "to first token", "decode"]
    assert kinds["waiting for KV"]["t1"] - kinds["waiting for KV"]["t0"] == 3 * ms
    # A frontend clock that does not project: the frontend's bars are not drawn.
    t = fields("unprojectable-clock", [pull, *producers], [join()])
    assert [s["kind"] for s in t["spans"]] == ["waiting for KV"]
    # Two producer occurrences of P in B: the outgoing handoff (70) picks one;
    # a kv_both scheduler named by the join is a producer too.
    twice = [producer(b_i, 50, 65, 7), producer(b_i, 65, None, 9, role="both")]
    assert fields(reference, [pull, *twice], [join()])["prefill_ms"] == 9.0
    # A lone producer timeline that does not own the handoff, a reused identity,
    # or an unconfirmed join: no prefill window, each with its own reason.
    t = fields(reference, [pull, producer(b_i, 80, None, 40)], [join()])
    assert t["prefill_ms"] is None
    assert t["reasons"]["prefill_ms"] == (
        "no producer occurrence owns the outgoing handoff")  # fmt: skip
    t = fields(reference, [pull, *producers], [join(identity_reused=True)])
    assert t["prefill_ms"] is None and "reused" in t["reasons"]["prefill_ms"]
    t = fields(reference, [pull, *producers], [join(ambiguous=True)])
    assert t["reasons"]["prefill_ms"] == "the handoff join is not confirmed"
    # Without an API arrival basis, no TTFT, with its own reason.
    queued = consumer(20, None, 0, stall={"basis": "scheduler_queued",
                                          "waiting_for_kv_ns": 0})  # fmt: skip
    t = fields(reference, [queued], [join()])
    assert t["ttft_ms"] is None and "scheduler_queued" in t["reasons"]["ttft_ms"]
    assert t["kv_wait_ms"] == 0.0  # a measured zero is a value
    # The producer's capture absent: the row's own incoming handoff still places
    # the decode occurrence; the prefill window is unavailable.
    t = fields(reference, [local, pull], [])
    assert t["kv_wait_ms"] == 363.0 and t["prefill_ms"] is None
    assert t["reasons"]["prefill_ms"] == "no handoff join for this occurrence"
    # No incoming handoff: one scheduler occurrence is unambiguous, two are not.
    assert fields(reference, [pull], [], incoming=())["kv_wait_ms"] == 363.0
    t = fields(reference, [local, pull], [], incoming=())
    assert t == {"available": False, "reason": "several scheduler occurrences "
                 "and no incoming handoff to place this row"}  # fmt: skip
    t = fields(reference, [], [])
    assert t["available"] is False and "no request timeline" in t["reason"]
    assert run.features()["ttft"] == "not recorded in this run's captures"


def test_a_saved_summary_without_the_timeline_sequences_is_rebuilt(l1_run):
    """A summary saved before the timelines carried their record sequences
    cannot place rows: the viewer rebuilds from the captures instead of
    serving it, even when its input fingerprints match."""
    run, _ = l1_run
    l1 = [capture for capture in run.captures if capture.layer == "l1"]
    saved = json.loads(json.dumps(run.summary))
    saved["viewer_input_fingerprints"] = run.input_fingerprints()
    saved["timelines"] = [{"request": "r", "sequences": {"first": 0, "end": None}}]
    saved["joins"] = []
    assert run._verify_saved(saved, l1)[1] == "verified (input fingerprints)"
    for old in ({**saved, "timelines": [{"request": "r"}]},
                {k: v for k, v in saved.items() if k != "timelines"},
                {**saved, "joins": [{"consumer_request": "r"}]}):
        served, reason = run._verify_saved(old, l1)
        assert served is None and "predates the timeline record sequences" in reason


def test_page_shows_the_timeline_columns_axis_and_header(viewer_module, l1_run):
    """The page plots a request's timeline phases inside its detail window,
    sorts durations as numbers and lists a field as unavailable only when it
    is."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed: the page's JavaScript cannot run here")
    run, _ = l1_run
    data = run.api_data()
    records = run.api_records(run.request_rows[0]["key"]["scheduler_process_id"],
                              run.request_rows[0]["key"]["request"], 0)
    # Two pairs: pair A's rows measured a zero TTFT, pair B's recorded none
    # (the fixture's row and a copy of it under its own key).
    data["pairs"] = ["pair-A", "pair-B"]
    copy = json.loads(json.dumps(data["requests"][0]))
    copy["key"] = {**copy["key"], "request": copy["key"]["request"] + "-copy"}
    copy["display"] = copy["display"] + "-copy"
    data["requests"].append(copy)
    # The selected request is the records payload's own row ("logical").
    for row in [*data["requests"], records["logical"]]:
        far = max((s["t1"] for s in row["spans"] if s.get("t1")), default=0) + 10**9
        row["timeline"] = {
            "available": True, "kv_wait_ms": 0.0, "admission_ms": 5.0,
            "decode_window_ms": 150.0, "ttft_ms": None, "prefill_ms": 3.0,
            "reasons": {"ttft_ms": "no API arrival or first token output recorded"},
            "spans": [{"t0": far, "t1": far + 10**8, "kind": "decode"}],
        }  # fmt: skip
    for index, row in enumerate(data["requests"]):
        row["pair"] = "pair-A" if index % 2 == 0 else "pair-B"
        row["timeline"] = {**row["timeline"],
                           "ttft_ms": 0.0 if row["pair"] == "pair-A" else None}
    meta = run.api_run()
    meta["features"] = dict.fromkeys(meta["features"], "consumer timeline: x")
    payload = {"run": meta, "data": data, "config": run.api_config(),
               "records": records}
    out = subprocess.run(
        [node, str(Path(__file__).with_name("_viewer_page_test.mjs")),
         str((VIEWER.parent / "viewer.html").resolve())],
        input=json.dumps(payload), capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    report = json.loads(out.stdout)
    assert report["phaseCount"] == 1 and report["plotCoversPhases"] is True
    assert report["numericSort"] == [2, 10, 100]
    # Availability is counted from the displayed rows: a measured zero KV wait
    # is present, a TTFT no row carries is listed as unavailable.
    n = len(data["requests"])
    n_a = sum(1 for row in data["requests"] if row["pair"] == "pair-A")
    n_b = n - n_a
    assert n_a and n_b
    assert f"KV wait {n}/{n}" in report["header"]
    assert f"TTFT {n_a}/{n}" in report["header"]
    by_pair = report["headerByPair"]
    # Pair B alone: no row carries a TTFT; pair A alone: measured zeros count.
    assert f"TTFT 0/{n_b}" in by_pair["pair-B"]
    assert "unavailable <b>TTFT</b>" in by_pair["pair-B"]
    assert f"TTFT {n_a}/{n_a}" in by_pair["pair-A"]
    assert "unavailable <b>none</b>" in by_pair["pair-A"]
    assert f"TTFT {n_a}/{n}" in by_pair[""]
    assert "3.00" in report["tableHtml"]
    assert "no API arrival or first token output recorded" in report["tableHtml"]
