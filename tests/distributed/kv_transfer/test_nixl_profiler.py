# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The NIXL L1 adapter on real recorders, sockets, helper processes and objects."""

import ast
import gc
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import weakref
from pathlib import Path
from uuid import uuid4

import psutil
import pytest

from vllm.kv_profiling import attach_frontend
from vllm.kv_profiling.clock import ClockGroup, ClockServer, measure_peer
from vllm.kv_profiling.config import NetworkProfilerConfig
from vllm.kv_profiling.core import open_recorder
from vllm.kv_profiling.nixl import (
    FrontendProfiler,
    NixlProfiler,
    attach_nixl,
    capabilities,
    close_with_owner,
)
from vllm.kv_profiling.peers import PeerClocks
from vllm.kv_profiling.perfetto import _load, export_traces
from vllm.kv_profiling.schema import Failure, IdSource, Subject

nixl = pytest.importorskip("nixl")

SHM = "/dev/shm"
import vllm

ROOT = Path(vllm.__file__).resolve().parent.parent


def config(tmp_path, **kw):
    context = {"role": "worker", "backend": ["UCX"], **kw.pop("context", {})}
    context.setdefault("capabilities", capabilities(context["role"]))
    kw.setdefault("transport", "tcp")
    return NetworkProfilerConfig(
        level="l1",
        run_id=kw.pop("run_id", str(uuid4())),
        output_dir=str(tmp_path),
        shared_dir=SHM,
        context=context,
        **kw,
    )


def profiler(tmp_path, **kw):
    cfg = config(tmp_path, **kw)
    return NixlProfiler(cfg, open_recorder(cfg), rank=0, engine_id="engine-A")


def handle(value):
    """A real NIXL handle object; ``_released`` keeps its finalizer from
    releasing a transfer that never existed."""
    from nixl._api import nixl_xfer_handle

    result = nixl_xfer_handle(None, value)
    result._released = True
    return result


LOOPBACK = r"""
import time
import torch
from nixl._api import nixl_agent, nixl_agent_config


def loopback(size=1 << 20, pieces=8):
    config = nixl_agent_config(backends=["UCX"], capture_telemetry=True)
    agent = nixl_agent("loop", config)
    src = torch.randint(0, 255, (size,), dtype=torch.uint8)
    dst = torch.zeros_like(src)
    agent.register_memory(agent.get_reg_descs([src, dst]))
    step = size // pieces
    xfer = lambda t: agent.get_xfer_descs(
        [t[i * step:(i + 1) * step * 1] for i in range(0, pieces, 2)])
    handle = agent.initialize_xfer("WRITE", xfer(src), xfer(dst), "loop")
    agent.transfer(handle)
    end = time.time() + 20
    while agent.check_xfer_state(handle) == "PROC" and time.time() < end:
        time.sleep(0.005)
    assert agent.check_xfer_state(handle) == "DONE"
    return agent, handle, agent.get_xfer_telemetry(handle), (src, dst)
"""
exec(LOOPBACK)


def telemetry():
    """Real telemetry (and handle) of a real NIXL transfer on a loopback agent."""
    agent, transfer, result, buffers = loopback()
    return result, (agent, transfer, buffers)


def read(profiler):
    """Close and decode a profiler's JSONL into (metadata, events)."""
    profiler.close()
    meta, events, decoder, rows, missing = _load(profiler.output_dir / "events.jsonl")
    for event in events:
        for key in ("request", "handoff", "peer"):
            event[key] = decoder.resolve(event[f"{key}_ref"])
    return meta, events, rows


# ------------------------------------------------------------------ config
def test_new_config_fields_validate_and_unknown_keys_name_themselves(tmp_path):
    cfg = config(tmp_path, clock_max_peers=8, clock_port_offset=2000)
    assert (cfg.transport, cfg.clock, cfg.clock_samples) == ("tcp", True, 7)
    for bad in (
        {"transport": "ib"},
        {"clock": 1},
        {"clock_port_offset": 0},
        {"clock_port_offset": 70000},
        {"clock_max_peers": 257},
        {"clock_samples": 0},
        {"clock_end_budget_s": 0},
    ):
        with pytest.raises(ValueError):
            config(tmp_path, **bad)
    with pytest.raises(ValueError, match="bogus"):
        NetworkProfilerConfig.from_extra_config(
            {"network_profiler": {"level": "off", "bogus": 1}}
        )


def vllm_config(extra, role="kv_producer"):
    from vllm.config import KVTransferConfig, VllmConfig

    return VllmConfig(
        kv_transfer_config=KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role=role,
            kv_connector_extra_config=extra,
        )
    )


def test_the_disabled_path_imports_nothing_and_starts_nothing():
    code = """
import sys
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import attach_profiler
from vllm.kv_profiling import attach_frontend
for extra in ({}, {"network_profiler": {"level": "off"}}):
    c = VllmConfig(kv_transfer_config=KVTransferConfig(
        kv_connector="NixlConnector", kv_role="kv_producer",
        kv_connector_extra_config=extra))
    assert attach_profiler(c, "worker", "e", 0, "pull") is None
    assert attach_frontend(c) is None
assert attach_frontend(VllmConfig()) is None
loaded = [m for m in sys.modules if m.startswith("vllm.kv_profiling.")]
assert not loaded, loaded
assert "vllm._kv_profiler" not in sys.modules
import threading
assert threading.active_count() == 1
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr


def test_an_enabled_profiler_refuses_the_push_connector_and_bad_options(tmp_path):
    extra = {
        "network_profiler": {
            "level": "l1",
            "run_id": str(uuid4()),
            "output_dir": str(tmp_path),
        }
    }
    with pytest.raises(ValueError, match="pull connector"):
        attach_nixl(vllm_config(extra), "worker", engine_id="e", rank=0,
                    transfer_mode="push")  # fmt: skip
    extra["network_profiler"]["clock_port_offset"] = 0
    with pytest.raises(ValueError, match="clock_port_offset"):
        attach_nixl(vllm_config(extra), "worker", engine_id="e", rank=0)
    extra["network_profiler"].update(clock_port_offset=1000, nope=1)
    with pytest.raises(ValueError, match="nope"):
        attach_nixl(vllm_config(extra), "worker", engine_id="e", rank=0)


# ------------------------------------------------------------ hooks to JSONL
def test_the_worker_hooks_write_the_expected_records(tmp_path):
    p = profiler(tmp_path)
    p.handshake_begin("engine-P")
    p.handshake_peer_done("engine-P", 1, 0, "agent-uuid-1")
    p.handshake_offset("engine-P", -0.0025)
    p.read_context("d-req", "p-req", "engine-P", 1)
    res, (agent, h, _) = telemetry()
    p.transfer_prepared(h, 0, 1, 12)
    p.transfer_posted(h)
    p.transfer_done(h, res)
    p.transfer_retired(h)
    p.notify_sent("d-req", "agent-uuid-1")
    p.failure("notification_failed", "d-req", {})
    p.failure("something_new", "d-req", {})
    failing = handle(0x7F0000001000)
    p.transfer_prepared(failing, 0, 1, 2)
    p.transfer_failed(failing)
    p.transfer_retired(failing)
    p.handshake_waiter("d-req", "engine-P", True)
    meta, events, _ = read(p)
    assert meta["context"]["backend"] == ["UCX"]
    assert meta["context"]["capabilities"]["selected_descriptors"] is True
    by_kind = {}
    for e in events:
        by_kind.setdefault(e["kind"], []).append(e)
    bind, failed_bind = by_kind["transfer_bind"]
    assert (bind["request"], bind["handoff"], bind["peer"]) == (
        "d-req", "p-req", "engine-P")  # fmt: skip
    assert (bind["subject"], bind["selected_descriptors"]) == (1, 12)
    assert (bind["aux0"], bind["aux1"]) == (0, 1)
    assert bind["transport"]["kind"] == "tcp"
    (done,) = by_kind["complete_observed"]
    assert done["bytes"] == res.totalBytes > 0
    assert done["merged_descriptors"] == res.descCount > 0
    assert (done["aux0"], done["aux1"]) == (res.postDuration, res.xferDuration)
    assert done["duration0_ns"] is None and done["duration1_ns"] is None
    assert bind["transfer"] == done["transfer"] != 0
    ends = by_kind["handshake_end"]
    peer = next(e for e in ends if e["subject"] == 1 and not e["request"])
    assert (peer["peer"], peer["handoff"], peer["aux0"]) == (
        "engine-P", "agent-uuid-1", 0)  # fmt: skip
    offset = next(e for e in ends if e["subject"] == Subject.OFFSET_SAMPLE)
    assert offset["aux0"] - (1 << 64) == -2_500_000
    waiter = next(e for e in ends if e["request"] == "d-req")
    assert waiter["status"] == 0
    failures = by_kind["failed"]
    assert [f["status"] for f in failures] == [
        Failure.NOTIFICATION_FAILED,
        Failure.UNKNOWN,
        Failure.TRANSFER_FAILED,
    ]
    assert [f["subject"] for f in failures] == [0, 0, 1]
    assert failures[2]["transfer"] == failed_bind["transfer"] != bind["transfer"]


def test_a_reused_handle_gets_a_new_generation(tmp_path):
    p = profiler(tmp_path)
    p.read_context("r1", "q1", "engine-P", 0)
    first, second = handle(0x1000), handle(0x1000)
    res, _ = telemetry()
    p.transfer_prepared(first, 0, 1, 4)
    p.transfer_done(first, res)
    p.transfer_retired(first)
    p.read_context("r2", "q2", "engine-P", 0)
    p.transfer_prepared(second, 0, 1, 4)
    p.transfer_done(second, res)
    p.transfer_retired(second)
    _, events, _ = read(p)
    done = [e["transfer"] for e in events if e["kind"] == "complete_observed"]
    assert len(done) == 2 and done[0] != done[1]
    assert not any(e["unbound_handle"] for e in events)


def test_unavailable_telemetry_stays_unknown_and_keeps_the_profiler_on(tmp_path):
    p = profiler(tmp_path)
    h = handle(0x2000)
    p.read_context("r", "q", "engine-P", 0)
    p.transfer_prepared(h, 0, 1, 3)
    p.transfer_done(h, None)
    p.transfer_retired(h)
    assert p.hook_errors == 0
    _, events, _ = read(p)
    done = next(e for e in events if e["kind"] == "complete_observed")
    assert all(
        done[k] is None
        for k in ("bytes", "merged_descriptors", "duration0_ns", "aux0", "aux1")
    )


def test_the_scheduler_hooks_write_handoffs_binds_and_aggregates(tmp_path):
    p = profiler(tmp_path, context={"role": "scheduler"})
    p.request_bound("p-req")
    p.handoff_out("p-req", 3, ([1, 2, 3], [4, 5]))
    p.handoff_in("d-req", "p-req", "engine-P", "127.0.0.1", 5600)
    p.handshake_served(b"zmq-identity", 0, 2)
    p.recv_finished("d-req")
    p.releases_requested(set())
    p.releases_requested({"p-req", "p-two"})
    p.notify_observed("p-req", 1, 2)
    p.senders_done("p-req")
    p.lease_expired("p-old", 1)
    p.late_notification("p-late")
    _, events, _ = read(p)
    by = {}
    for e in events:
        by.setdefault(e["kind"], []).append(e)
    out = next(e for e in by["handoff"] if e["request"] == "p-req")
    assert (out["aux0"], out["aux1"]) == (3, 5)
    handoff_in = next(e for e in by["handoff"] if e["request"] == "d-req")
    assert (handoff_in["handoff"], handoff_in["peer"]) == ("p-req", "engine-P")
    (served,) = by["handshake_end"]
    assert served["peer"] == {"bytes_hex": b"zmq-identity".hex()}
    assert served["subject"] == 2
    (recv,) = by["complete_observed"]
    assert (recv["request"], recv["subject"]) == ("d-req", Subject.SCHEDULER)
    releases = by["release_requested"]
    assert {e["request"] for e in releases} == {"p-req", "p-two"}
    assert {e["subject"] for e in releases} == {Subject.SCHEDULER}
    assert by["senders_done"][0]["subject"] == Subject.WORKER
    seen = by["notify_observed"][0]
    assert (seen["aux0"], seen["aux1"]) == (1, 2)
    assert [e["status"] for e in by["failed"]] == [
        Failure.LEASE_EXPIRED, Failure.LATE_NOTIFICATION]  # fmt: skip


# --------------------------------------------------- loss versus defects
def test_a_full_ring_is_counted_loss_and_recording_resumes_after_a_drain(tmp_path):
    p = profiler(tmp_path, ring_records=8, flush="background", context={})
    p.recorder.wait_ready()
    native = p.recorder.native
    for i in range(64):
        p.request_bound(f"req-{i}")
    state = native.stats()
    assert state["attempts"] == 64 and state["events"] + state["drop_full"] == 64
    assert p.hook_errors == 0
    deadline = time.monotonic() + 10
    while native.stats()["head"] != native.stats()["tail"]:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    before = native.stats()["events"]
    p.request_bound("after-drain")
    assert native.stats()["events"] == before + 1
    _, events, rows = read(p)
    assert events[-1]["request"] == "after-drain"
    assert rows[-1]["stats"]["drop_full"] == state["drop_full"] > 0
    assert rows[-1]["complete"] is False


def test_finished_and_unmapped_recorders_are_separate_states(tmp_path):
    p = profiler(tmp_path)
    native = p.recorder.native
    native.finish()
    p.request_bound("late")
    assert p.hook_errors == 0
    assert native.stats()["drop_closed"] == 1
    result = p.close()
    assert result.stats["drop_closed"] == 1
    # Unmapped: the hook returns at once, and the ABI has no counter to read.
    p.request_bound("after-close")
    with pytest.raises(ValueError, match="closed ring"):
        native.stats()
    assert p.close() is result


def test_an_adapter_error_disables_the_profiler_and_never_reaches_the_caller(
    tmp_path,
):
    p = profiler(tmp_path)
    h = handle(0x3000)
    p.read_context("r", "q", "engine-P", 0)
    p.transfer_prepared(h, "not-an-int", 1, 3)  # native TypeError: a defect
    assert p.hook_errors == 1
    p.transfer_posted(h)
    p.request_bound("ignored")
    assert p.hook_errors == 1
    _, events, _ = read(p)
    assert events == []


# ---------------------------------------------- failure containment
def hold_lock_in_thread(lock):
    import threading

    taken, release = threading.Event(), threading.Event()

    def hold():
        with lock:
            taken.set()
            release.wait(30)

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert taken.wait(10)
    return release, thread


def run_with_deadline(function, seconds=5):
    import threading

    done = threading.Event()
    outcome = {}

    def target():
        try:
            outcome["value"] = function()
        except BaseException as exc:
            outcome["error"] = exc
        done.set()

    threading.Thread(target=target, daemon=True).start()
    assert done.wait(seconds), "the hook blocked"
    return outcome


def test_neither_a_hook_error_nor_the_shutdown_waits_on_a_log_handler(tmp_path):
    """The handler lock is held by another thread for the whole test, through the
    hook error and through ``close``; the profiler calls no log handler at all and
    keeps its diagnostics in ``profiler.json`` and ``shutdown_failures``."""
    import logging

    handler = logging.Handler()
    handler.emitted = []
    handler.emit = lambda record: handler.emitted.append(record.getMessage())
    logger = logging.getLogger("vllm.kv_profiling.nixl")
    root = logging.getLogger()
    logger.addHandler(handler)
    root.addHandler(handler)
    p = profiler(tmp_path)
    release, thread = hold_lock_in_thread(handler.lock)
    try:
        h = handle(0x3100)
        p.read_context("r", "q", "engine-P", 0)
        outcome = run_with_deadline(
            lambda: p.transfer_prepared(h, "not-an-int", 1, 3)
        )
        assert "error" not in outcome and p.hook_errors == 1
        assert p.first_error == "TypeError"
        for _ in range(5):  # repeated calls after disabling stay inert and fast
            assert "error" not in run_with_deadline(lambda: p.request_bound("x"))
        assert p.hook_errors == 1
        closed = run_with_deadline(p.close, seconds=p.config.shutdown_timeout_s + 5)
        assert "error" not in closed and closed["value"] is p.result is not None
        assert p.recorder.writer.poll() is not None  # the helper finished
        assert p.close() is p.result
        assert handler.emitted == []
    finally:
        release.set()
        thread.join(10)
        logger.removeHandler(handler)
        root.removeHandler(handler)
    summary = json.loads((p.output_dir / "profiler.json").read_text())
    assert (summary["hook_errors"], summary["first_error"]) == (1, "TypeError")
    assert summary["shutdown_failures"] == []
    manifest = json.loads((p.output_dir / "manifest.json").read_text())
    assert isinstance(manifest["writer_cpu_s"], float) and manifest["writer_cpu_s"] >= 0


def test_a_throwing_log_handler_cannot_escape_a_hook_or_the_shutdown(tmp_path):
    import logging

    class Throwing(logging.Handler):
        def emit(self, record):
            raise OSError("log sink is gone")

    logger = logging.getLogger("vllm.kv_profiling.nixl")
    handler = Throwing()
    logger.addHandler(handler)
    try:
        p = profiler(tmp_path)
        p.read_context("r", "q", "engine-P", 0)
        p.transfer_prepared(handle(0x3200), "not-an-int", 1, 3)
        p.request_bound("after")
        p.request_bound("after")
        assert p.hook_errors == 1
        assert p.close() is not None
    finally:
        logger.removeHandler(handler)


def test_a_failing_peer_output_still_stops_the_clock_server_and_the_recorder(
    tmp_path,
):
    p = consumer(tmp_path)  # a kv_both scheduler holds peers and a server
    p.server = server_for(p, p.config.run_id)
    helper = psutil.Process(p.server.process.pid)
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the calibration directory should be")
    p.output_dir = blocker / "calibration"  # creating it raises NotADirectoryError
    result = p.close()
    assert result is not None and p.closed
    deadline = time.monotonic() + 5
    while helper.is_running() and helper.status() != "zombie":
        assert time.monotonic() < deadline, "the clock server was left running"
        time.sleep(0.02)
    assert not p.peers.thread.is_alive() or p.peers.closed
    assert p.shutdown_failures == ["clock peers: NotADirectoryError"]
    assert p.close() is result  # a second close repeats nothing and fails nothing


# ------------------------------------------------------------ clock peers
def server_for(profiler_or_gen, run_id):
    generation = (
        profiler_or_gen
        if isinstance(profiler_or_gen, str)
        else profiler_or_gen.recorder.metadata["clock_id"]
    )
    return ClockServer(run_id, generation=generation, host="127.0.0.1", port=0)


def consumer(tmp_path, **kw):
    kw.setdefault("clock_port_offset", 1000)
    p = profiler(tmp_path, context={"role": "scheduler"}, **kw)
    p.peers = PeerClocks(
        p.config, p.config.run_id, p.recorder.metadata["clock_id"]
    )
    return p


def wait_results(p, count, timeout=15):
    deadline = time.monotonic() + timeout
    while len(p.peers.results) < count:
        assert time.monotonic() < deadline, p.peers.results
        time.sleep(0.02)


def test_a_peer_is_calibrated_once_at_start_and_end_and_kept_after_it_drains(tmp_path):
    p = consumer(tmp_path)
    remote = server_for(str(uuid4()), p.config.run_id)
    try:
        port = remote.address[1] - 1000
        other = str(uuid4())
        for engine in ("engine-P", other, "engine-P", other, "engine-P"):
            p.handoff_in("d", "q", engine, "127.0.0.1", port)
        # The second engine shares the server, so both calibrate; alternating
        # requests never grow the table or recalibrate.
        assert p.peers.tail == 2
        wait_results(p, 2)
        for engine in ("engine-P", other):
            p.handoff_in("d", "q", engine, "127.0.0.1", port)
        time.sleep(0.3)
        assert len(p.peers.results) == 2
        p.close()
        manifest = json.loads((p.output_dir / "calibration.json").read_text())
        assert manifest["clock_thread"] == "joined"
        statuses = [x["status"] for x in manifest["peers"]]
        assert statuses == ["ok", "ok"]
        assert all(not x["end_missing"] for x in manifest["peers"])
        from vllm.kv_profiling.clock import ClockMapping

        first = manifest["peers"][0]
        mapping = ClockMapping.from_json(
            json.loads((p.output_dir / first["calibration_file"]).read_text())
        )
        assert mapping.end is not None and len(mapping.raw_start) == 7
        assert mapping.start.remote_generation == first["remote_generation"]
    finally:
        remote.close()


def test_unreachable_and_silent_peers_fail_within_their_deadline(tmp_path):
    p = consumer(tmp_path)
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        dead = closed.getsockname()[1]
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen(4)
    try:
        p.handoff_in("d", "q", "dead-peer", "127.0.0.1", dead - 1000)
        silent_port = silent.getsockname()[1] - 1000
        p.handoff_in("d", "q", "silent-peer", "127.0.0.1", silent_port)
        start = time.monotonic()
        wait_results(p, 2)
        assert time.monotonic() - start < 5
        p.close()
        manifest = json.loads((p.output_dir / "calibration.json").read_text())
        assert [x["status"] for x in manifest["peers"]] == ["failed", "failed"]
        assert "calibration_file" not in manifest["peers"][0]
    finally:
        silent.close()


def test_a_full_peer_table_records_loss_and_never_blocks(tmp_path):
    p = consumer(tmp_path, clock_max_peers=2)
    for engine in ("a", "b", "c", "c", "d"):
        p.handoff_in("d", "q", engine, "127.0.0.1", 1500)
    assert p.peers.tail == 2
    _, events, _ = read(p)
    full = [e for e in events if e["kind"] == "failed"]
    assert [e["status"] for e in full] == [Failure.CLOCK_PEER_TABLE_FULL] * 3
    assert [e["peer"] for e in full] == ["c", "c", "d"]


def test_shutdown_during_calibration_abandons_the_thread_without_late_effects(
    tmp_path,
):
    p = consumer(tmp_path, shutdown_timeout_s=0.2, clock_end_budget_s=0.1)
    stall = socket.socket()
    stall.bind(("127.0.0.1", 0))
    stall.listen(4)
    try:
        p.handoff_in("d", "q", "stalled", "127.0.0.1", stall.getsockname()[1] - 1000)
        time.sleep(0.4)  # the thread is now inside measure_peer
        thread = p.peers.thread
        start = time.monotonic()
        result = p.close()
        assert time.monotonic() - start < 5 and result is not None
        path = p.output_dir / "calibration.json"
        first = path.read_bytes()
        manifest = json.loads(first)
        assert manifest["clock_thread"] == "abandoned"
        assert manifest["peers"][0]["status"] == "abandoned"
        assert thread.daemon
        thread.join(5)
        assert not thread.is_alive()
        assert path.read_bytes() == first
    finally:
        stall.close()


def test_a_producer_clock_server_is_stopped_with_the_profiler(tmp_path):
    p = profiler(tmp_path, context={"role": "scheduler"})
    p.server = server_for(p, p.config.run_id)
    helper = p.server.process.pid
    assert psutil.pid_exists(helper)
    p.close()
    deadline = time.monotonic() + 5
    while psutil.pid_exists(helper) and psutil.Process(helper).status() != "zombie":
        assert time.monotonic() < deadline
        time.sleep(0.02)


# ------------------------------------------------------ owners and lifetimes
def writer_pids(p):
    return {p.recorder.writer.pid}


def test_two_owners_are_independent_and_leave_no_helper_behind(tmp_path):
    one, two = profiler(tmp_path / "a"), profiler(tmp_path / "b")
    assert one.output_dir != two.output_dir
    helpers = writer_pids(one) | writer_pids(two)
    one.request_bound("only-one")
    one.close()
    two.request_bound("only-two")
    two.close()
    assert [e["request"] for e in read_after(one)] == ["only-one"]
    assert [e["request"] for e in read_after(two)] == ["only-two"]
    for pid in helpers:
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == "zombie"


def read_after(p):
    _, events, decoder, _, _ = _load(p.output_dir / "events.jsonl")
    return [{**e, "request": decoder.resolve(e["request_ref"])} for e in events]


def test_a_failed_attachment_leaves_no_ring_or_helper(tmp_path, monkeypatch):
    before = set(os.listdir(SHM))
    cfg = config(tmp_path, context={"role": "frontend"})

    def broken(self, config, recorder):
        raise RuntimeError("constructor failed after the recorder existed")

    monkeypatch.setattr(FrontendProfiler, "__init__", broken)
    extra = {
        "network_profiler": {
            "level": "l1",
            "run_id": cfg.run_id,
            "output_dir": str(tmp_path),
            "shared_dir": SHM,
        }
    }
    with pytest.raises(RuntimeError, match="constructor failed"):
        attach_frontend(vllm_config(extra))
    deadline = time.monotonic() + 10
    while {n for n in set(os.listdir(SHM)) - before if n.startswith("kvprof-")}:
        assert time.monotonic() < deadline, set(os.listdir(SHM)) - before
        time.sleep(0.05)


def test_the_finalizer_closes_without_keeping_the_owner_alive(tmp_path):
    p = profiler(tmp_path)

    class Owner:
        pass

    owner = Owner()
    reference = weakref.ref(owner)
    close_with_owner(owner, p)
    p.request_bound("recorded")
    del owner
    gc.collect()
    assert reference() is None
    assert p.closed and p.result is not None
    assert [e["request"] for e in read_after(p)] == ["recorded"]


def test_a_forked_child_cannot_close_or_record_into_the_parents_ring(tmp_path):
    p = profiler(tmp_path)
    p.request_bound("before-fork")
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            p.request_bound("child")
            code = 0 if p.close() is None and p.hook_errors == 0 else 2
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    p.request_bound("after-fork")
    assert psutil.Process(p.recorder.writer.pid).is_running()
    result = p.close()
    assert result.complete
    assert [e["request"] for e in read_after(p)] == ["before-fork", "after-fork"]


# ---------------------------------------------------------------- frontend
def frontend(tmp_path, **kw):
    cfg = config(tmp_path, context={"role": "frontend"}, **kw)
    return FrontendProfiler(cfg, open_recorder(cfg))


def starlette_request(header):
    from starlette.requests import Request

    headers = [] if header is None else [(b"x-request-id", header.encode())]
    return Request({"type": "http", "headers": headers})


def site_expression(relative, name, index):
    """The text of one argument of the ``name`` hook call as the serving file
    has it, so these tests follow the real callsite rather than a copy."""
    tree = ast.parse((ROOT / relative).read_text())
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == name
    ]  # fmt: skip
    assert len(calls) == 1, (relative, name)
    return ast.unparse(calls[0].args[index])


SERVING = (
    "vllm/entrypoints/openai/completion/serving.py",
    "vllm/entrypoints/openai/chat_completion/serving.py",
)


@pytest.mark.parametrize("serving", SERVING)
@pytest.mark.parametrize(
    "header, body, source",
    [
        ("a", "a", IdSource.HEADER),  # equal one-character values
        ("", "", IdSource.HEADER),  # an empty header is a header
        ("same-id", "same-id", IdSource.HEADER),
        ("hdr", "body", IdSource.HEADER),
        (None, "body", IdSource.BODY),
        (None, None, IdSource.GENERATED),
    ],
)
def test_header_provenance_is_the_recorded_header_not_object_identity(
    tmp_path, serving, header, body, source
):
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.entrypoints.serve.engine.serving import BaseServing

    fields = {"model": "m", "prompt": "hi"}
    if body is not None:
        fields["request_id"] = body
    # JSON decoding, as the server does; the header goes through Latin-1.
    request = CompletionRequest.model_validate_json(json.dumps(fields))
    raw_request = starlette_request(header)
    base = BaseServing._base_request_id(raw_request, request.request_id)
    header_of = eval(
        "lambda raw_request: " + site_expression(serving, "api_request", 1)
    )
    given_of = eval("lambda request: " + site_expression(serving, "api_request", 3))
    f = frontend(tmp_path)
    f.api_request(base, header_of(raw_request), f"cmpl-{base}", given_of(request))
    _, events, _ = read(f)
    edge, parent = events
    assert edge["subject"] == Subject.HEADER_TO_BASE
    assert (edge["request"], edge["handoff"], edge["status"]) == (
        base, header, source)  # fmt: skip
    assert (parent["request"], parent["handoff"]) == (f"cmpl-{base}", base)
    assert f.hook_errors == 0
    assert header_of(None) is None  # no raw request: no header


def test_the_frontend_binds_engine_child_and_repeated_ids_exactly(
    tmp_path, monkeypatch
):
    from vllm import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm.v1.engine.parallel_sampling import ParentRequest

    f = frontend(tmp_path)

    def engine_request(request_id):
        return EngineCoreRequest(
            request_id=request_id, prompt_token_ids=[1], mm_features=None,
            sampling_params=SamplingParams(n=2), pooling_params=None,
            arrival_time=0.0, lora_request=None, cache_salt=None,
            data_parallel_rank=None)  # fmt: skip

    request = engine_request("cmpl-foo-0")
    InputProcessor.assign_request_id(request)
    f.external_engine(request.external_req_id, request.request_id)
    parent = ParentRequest(request)
    child, _ = parent.get_child_info(1)
    f.child_parent(child, parent.request_id)
    f.parent_item("cmpl-foo", "cmpl-foo-0")
    monkeypatch.setenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")
    plain = engine_request("cmpl-same")
    InputProcessor.assign_request_id(plain)
    f.external_engine(plain.external_req_id, plain.request_id)
    f.external_engine(plain.external_req_id, plain.request_id)  # a repeated id
    _, events, _ = read(f)
    engine = [e for e in events if e["subject"] == Subject.EXTERNAL_TO_ENGINE]
    assert engine[0]["handoff"] == "cmpl-foo-0"
    assert engine[0]["request"].startswith("cmpl-foo-0-") and engine[0]["status"] == 0
    assert [e["status"] for e in engine[1:]] == [1, 1]
    assert [e["request"] for e in engine[1:]] == ["cmpl-same", "cmpl-same"]
    (edge,) = [e for e in events if e["subject"] == Subject.CHILD_TO_PARENT_ENGINE]
    assert edge["request"] == f"1_{engine[0]['request']}"
    assert edge["handoff"] == engine[0]["request"]
    (item,) = [e for e in events if e["subject"] == Subject.PARENT_TO_ITEM]
    assert (item["request"], item["handoff"]) == ("cmpl-foo-0", "cmpl-foo")


# ------------------------------------------------------------- exporter
def test_the_exporter_aligns_one_domain_draws_the_handoff_and_tags_the_backend(
    tmp_path,
):
    run = str(uuid4())
    shared = {"run_id": run}
    d = profiler(tmp_path / "d", context={"role": "worker"}, **shared)
    ps = profiler(tmp_path / "p", context={"role": "scheduler"}, **shared)
    pw = profiler(
        tmp_path / "p2", context={"role": "worker", "engine_id": "engine-P"}, **shared
    )
    for q in (d, ps, pw):
        assert q.config.run_id == run
    server = server_for(ps, run)
    try:
        mapping = None
        from vllm.kv_profiling.clock import ClockMapping

        mapping = ClockMapping.from_samples(
            measure_peer(server.address, run, d.recorder.metadata["clock_id"]),
            measure_peer(server.address, run, d.recorder.metadata["clock_id"]),
        )
        ps.handoff_out("p-req", 1, ([1, 2],))
        d.handshake_begin("engine-P")
        d.handshake_peer_done("engine-P", 0, 0, "agent-1")
        d.read_context("d-req", "p-req", "engine-P", 0)
        res, (agent, h, _) = telemetry()
        d.transfer_prepared(h, 0, 1, 6)
        d.transfer_posted(h)
        d.transfer_done(h, res)
        d.transfer_retired(h)
        pw.notify_observed("p-req", 1, 1)
        pw.senders_done("p-req")
        paths = []
        for q in (d, ps, pw):
            q.close()
            paths.append(str(q.output_dir / "events.jsonl"))
        group = ClockGroup(
            d.recorder.metadata["clock_id"],
            {ps.recorder.metadata["clock_id"]: mapping},
        )
        documents = export_traces(paths, str(tmp_path / "out"), group)
    finally:
        server.close()
    assert list(documents) == [("wall", "aligned")]
    document = json.loads(documents[("wall", "aligned")].read_text())
    events = document["traceEvents"]
    names = {e["name"] for e in events}
    assert {"handshake", "prepared to posted", "READ until observed DONE"} <= names
    assert {e["ph"] for e in events if e.get("cat") == "kv_handoff"} == {"s", "f"}
    assert document["metadata"]["cross_process_flows"] == "exact_recorded_handoff_ids"
    bind = next(e for e in events if e["name"] == "transfer_bind")["args"]
    assert bind["selected_descriptors"] == 6 and bind["leg"] == "single"
    assert bind["transport"] == {**bind["transport"], "backend": "UCX",
                                 "provenance": "configured", "kind": "tcp"}  # fmt: skip
    handoff = next(e for e in events if e["name"] == "handoff" and "args" in e)["args"]
    assert handoff["handoff_blocks"] == 2
    aliases = {
        e["args"]["clock_alias"]
        for e in events
        if e.get("args") and e["args"].get("clock_alias")
    }
    # Every process shares this host's clock, so the P worker takes the
    # reference's identity projection rather than a measured offset.
    assert aliases == {d.recorder.metadata["clock_id"]}
    models = {
        e["args"]["clock_model"]
        for e in events
        if e.get("args") and "clock_model" in e["args"]
    }
    assert models == {"linear_estimate_not_a_bound"}


def export_flows(tmp_path, build):
    """Build profilers in one clock domain, let ``build`` record, and export."""
    run = str(uuid4())
    made = {}

    def make(name, **context):
        made[name] = profiler(
            tmp_path / name, run_id=run, context={"role": "worker", **context}
        )
        return made[name]

    build(make)
    paths = []
    for q in made.values():
        q.close()
        paths.append(str(q.output_dir / "events.jsonl"))
    reference = made["d"].recorder.metadata["clock_id"]
    documents = export_traces(paths, str(tmp_path / "out"), ClockGroup(reference, {}))
    assert list(documents) == [("wall", "aligned")]
    document = json.loads(documents[("wall", "aligned")].read_text())
    names = {
        e["pid"]: ast.literal_eval(e["args"]["name"])
        for e in document["traceEvents"]
        if e.get("ph") == "M"
    }
    flows = [e for e in document["traceEvents"] if e.get("cat") == "kv_handoff"]
    return document, names, flows


def bind_read(d, request_id, producer, remote_request, handle_value, rank=0):
    h = handle(handle_value)
    d.read_context(request_id, remote_request, producer, rank)
    d.transfer_prepared(h, 0, 1, 4)
    d.transfer_posted(h)
    d.transfer_retired(h)


def test_a_flow_names_the_producer_the_read_was_bound_to(tmp_path):
    """Producers A and B notified the same request id; the read named B."""

    def build(make):
        a = make("a", engine_id="engine-A")
        b = make("b", engine_id="engine-B")
        d = make("d", engine_id="engine-D")
        a.notify_observed("shared-id", 1, 1)
        b.notify_observed("shared-id", 1, 1)
        bind_read(d, "d-req", "engine-B", "shared-id", 0x5000)

    document, names, flows = export_flows(tmp_path, build)
    assert {e["ph"] for e in flows} == {"s", "f"}
    end = next(e for e in flows if e["ph"] == "f")
    assert names[end["pid"]]["engine_id"] == "engine-B"
    assert document["metadata"]["cross_process_flows"] == "exact_recorded_handoff_ids"
    assert document["metadata"]["flows_omitted_ambiguous"] == 0


@pytest.mark.parametrize("case", ["repeated-read", "repeated-notification", "fan-in"])
def test_ambiguous_handoffs_get_no_arrow_instead_of_a_guess(tmp_path, case):
    def build(make):
        b = make("b", engine_id="engine-B")
        d = make("d", engine_id="engine-D")
        if case == "repeated-read":  # the same ids read twice
            b.notify_observed("r1", 1, 1)
            bind_read(d, "d-1", "engine-B", "r1", 0x6000)
            bind_read(d, "d-2", "engine-B", "r1", 0x6001)
        elif case == "repeated-notification":  # two occurrences of the request
            b.notify_observed("r1", 1, 1)
            b.notify_observed("r1", 1, 1)
            bind_read(d, "d-1", "engine-B", "r1", 0x6000)
        else:  # two decode ranks read one producer rank's request
            b.notify_observed("r1", 1, 2)
            b.notify_observed("r1", 2, 2)
            bind_read(d, "d-1", "engine-B", "r1", 0x6000, rank=0)
            bind_read(d, "d-1", "engine-B", "r1", 0x6001, rank=1)

    document, _, flows = export_flows(tmp_path, build)
    assert flows == []
    assert document["metadata"]["cross_process_flows"] == "none_matched"
    assert document["metadata"]["flows_omitted_ambiguous"] == 1


# ------------------------------------------------------------ allocation
def build_interposer(tmp_path):
    import sysconfig

    sources = Path(__file__).parent
    interposer = tmp_path / "allocations.so"
    extension = tmp_path / (
        "_profiler_allocations" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    subprocess.run(
        [*shlex.split(os.environ.get("CC", sysconfig.get_config_var("CC") or "cc")),
         "-shared", "-fPIC", str(sources / "profiler_allocations.c"),
         "-o", str(interposer)], check=True, timeout=60)  # fmt: skip
    subprocess.run(
        [*shlex.split(os.environ.get("CXX", sysconfig.get_config_var("CXX") or "c++")),
         "-std=c++20", "-shared", "-fPIC", "-I" + sysconfig.get_path("include"),
         str(sources / "profiler_allocations.cpp"), "-ldl", "-o", str(extension)],
        check=True, timeout=60)  # fmt: skip
    return interposer


ACCOUNTING = LOOPBACK + r"""
import json, sys
import _profiler_allocations
from pathlib import Path
from uuid import uuid4
from nixl._api import nixl_xfer_handle
from nixl import _bindings
from vllm.kv_profiling.config import NetworkProfilerConfig
from vllm.kv_profiling.core import open_recorder
from vllm.kv_profiling.nixl import NixlProfiler, FrontendProfiler
from vllm.kv_profiling.peers import PeerClocks

out = Path(sys.argv[1])
N = 200
cfg = NetworkProfilerConfig(
    level="l1", run_id=str(uuid4()), output_dir=str(out), shared_dir="/dev/shm",
    transport="tcp", ring_records=1 << 16, id_slots=1 << 14, id_bytes=1 << 22,
    handle_slots=1024, clock_max_peers=256, context={"role": "worker"})
p = NixlProfiler(cfg, open_recorder(cfg), rank=0, engine_id="engine-A")
p.peers = PeerClocks(cfg, cfg.run_id, p.recorder.metadata["clock_id"])
fcfg = NetworkProfilerConfig(
    level="l1", run_id=cfg.run_id, output_dir=str(out / "f"), shared_dir="/dev/shm",
    ring_records=1 << 16, id_slots=1 << 14, id_bytes=1 << 22,
    context={"role": "frontend"})
f = FrontendProfiler(fcfg, open_recorder(fcfg))

def handle(v):
    h = nixl_xfer_handle(None, v); h._released = True; return h
agent, real_handle, telemetry, buffers = loopback()
assert telemetry.totalBytes > 256 and telemetry.xferDuration > 0
handles = [handle(0x10000000 + i) for i in range(N)]
big = [real_handle] * N
groups = ([1] * 500, [2] * 300, [3] * 300)

def fresh(prefix, i):
    return f"{prefix}-{i}-" + "x" * 40

def fixed(prefix, i):
    return f"{prefix}-fixed"

def batches(kind, first):
    ids = fresh if first else fixed
    r = range(N)
    return {
        "handshake_begin": [(ids("e", i),) for i in r],
        "handshake_peer_done": [(ids("e", i), 3, 0, ids("a", i)) for i in r],
        "handshake_offset": [(ids("e", i), -0.0025) for i in r],
        "handshake_waiter": [(ids("r", i), ids("e", i), True) for i in r],
        "read_context": [(ids("r", i), ids("q", i), ids("e", i), 1) for i in r],
        "transfer_prepared": [(big[i], 0, 1, 70000 + i) for i in r],
        "transfer_posted": [(big[i],) for i in r],
        "transfer_done": [(big[i], telemetry) for i in r],
        "transfer_retired": [(big[i],) for i in r],
        "transfer_failed": [(handles[i],) for i in r],
        "failure": [("transfer_failed", ids("r", i), {}) for i in r],
        "notify_sent": [(ids("r", i), ids("a", i)) for i in r],
        "notify_observed": [(ids("r", i), 1, 2) for i in r],
        "senders_done": [(ids("r", i),) for i in r],
        "late_notification": [(ids("r", i),) for i in r],
        "lease_expired": [(ids("r", i), 1) for i in r],
        "request_bound": [(ids("r", i),) for i in r],
        "handoff_out": [(ids("r", i), 7, groups) for i in r],
        "handoff_in": [
            (ids("r", i), ids("q", i), ids("p", i), "127.0.0.1", 5600) for i in r
        ],
        "handshake_served": [(b"id-" + str(i).encode(), 0, 1) for i in r],
        "recv_finished": [(ids("r", i),) for i in r],
        "releases_requested": [({ids("r", i), ids("s", i)},) for i in r],
    }[kind]

def frontend_batches(kind, first):
    ids = fresh if first else fixed
    r = range(N)
    base = [ids("b", i) for i in r]
    return {
        "api_request": [(base[i], ids("d", i), ids("c", i), True) for i in r],
        "parent_item": [(ids("c", i), ids("i", i)) for i in r],
        "external_engine": [(ids("x", i), ids("y", i)) for i in r],
        "child_parent": [(ids("k", i), ids("m", i)) for i in r],
    }[kind]

report = {}
import gc
gc.collect()
for owner, names, make in (
    (p, [
        "handshake_begin", "handshake_peer_done", "handshake_offset",
        "handshake_waiter", "read_context", "transfer_prepared",
        "transfer_posted", "transfer_done", "transfer_retired", "transfer_failed",
        "failure", "notify_sent",
        "notify_observed", "senders_done", "late_notification", "lease_expired",
        "request_bound", "handoff_out", "handoff_in", "handshake_served",
        "recv_finished", "releases_requested"], batches),
    (f, ["api_request", "parent_item", "external_engine", "child_parent"],
     frontend_batches),
):
    for name in names:
        method = getattr(owner, name)
        entry = {}
        for phase, first in (("first_seen", True), ("cached", False)):
            work = make(name, first)
            before = sys.getallocatedblocks()
            counts = _profiler_allocations.measure(method, work)
            after = sys.getallocatedblocks()
            entry[phase] = {
                "calls": N,
                "pymem_raw_mem_obj_libc": list(counts),
                "objects_per_call": (counts[1] + counts[2]) / N,
                "retained_blocks_delta": after - before,
            }
        report[name] = entry
assert p.hook_errors == 0 and f.hook_errors == 0
(out / "accounting.json").write_text(json.dumps(report, indent=1))

# The enabled callsites: each non-trivial argument or test expression exactly as
# the connector, serving and engine files spell it, on real values.
import ast
import builtins
import numpy as np
from starlette.requests import Request
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.outputs import KVConnectorOutput
from vllm import SamplingParams

class Site:
    pass

site = Site()
site._kvp = p
site.consumer_notification_counts_by_req = {"req-a": 1}
site.expected_consumer_notifications_by_req = {"req-a": 2}
site._registration_generation = 3
site.engine_client = LLMEngine
site.kv_frontend_profiler = f
engine_request = EngineCoreRequest(
    request_id="cmpl-foo-0-abcdef12", prompt_token_ids=[1], mm_features=None,
    sampling_params=SamplingParams(n=2), pooling_params=None, arrival_time=0.0,
    lora_request=None, cache_salt=None, data_parallel_rank=None)
engine_request.external_req_id = "cmpl-foo-0"
serving_request = CompletionRequest.model_validate_json(
    '{"model": "m", "prompt": "hi", "request_id": "a"}')
common = dict(
    self=site, p=p, kvp=f, handle=real_handle, res=telemetry,
    handles=[real_handle], req_id="req-a", handle_unused=None,
    local_handle=object(), local_dram_handle=object(),
    local_block_descs_ids=np.arange(70000), leg_ids=np.arange(35000),
    connector_output=KVConnectorOutput(finished_sending={"a", "b"}),
    delay_free_blocks=True, block_ids=([1] * 500, [2] * 300),
    sub_request_id="cmpl-foo-1", request_id="cmpl-foo",
    parent_request=ParentRequest(engine_request),
    parent_req=ParentRequest(engine_request), final_req=engine_request,
    raw_request=Request({"type": "http",
                         "headers": [(b"x-request-id", b"header-id")]}),
)
variants = [
    {**common, "request": serving_request},
    {**common, "request": engine_request},
]
callsites = {}
for text, places in json.loads(sys.argv[2]).items():
    names = sorted(
        {n.id for n in ast.walk(ast.parse(text, mode="eval"))
         if isinstance(n, ast.Name)} - set(dir(builtins)))
    namespace = {}
    exec(f"def evaluate({', '.join(names)}):\n    return ({text})", namespace)
    function = namespace["evaluate"]
    entry = {"places": places, "status": "missing"}
    for values in variants:
        try:
            args = tuple(values[n] for n in names)
            function(*args)
        except (KeyError, AttributeError, NameError):
            continue
        counts = _profiler_allocations.measure(function, [args] * N)
        entry = {"places": places, "status": "measured",
                 "objects_per_call": (counts[1] + counts[2]) / N,
                 "libc": counts[3]}
        break
    callsites[text] = entry
(out / "callsites.json").write_text(json.dumps(callsites, indent=1))
print("accounted", len(report), len(callsites))
"""


def test_every_enabled_hook_is_accounted_for_and_retains_nothing(tmp_path):
    """Allocation counts are measurements, not zero assertions: the report lists
    each hook's transient objects per call (first-seen and cached ids, real
    telemetry, large descriptor counts, the real handle objects), and the test
    only requires them bounded and nothing retained as calls repeat."""
    interposer = build_interposer(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", ACCOUNTING, str(tmp_path), json.dumps(
            callsite_expressions())],
        env=dict(os.environ, LD_PRELOAD=str(interposer), PYTHONPATH=str(tmp_path)),
        capture_output=True, text=True, timeout=300)  # fmt: skip
    assert "accounted" in result.stdout, result.stderr
    callsites = json.loads((tmp_path / "callsites.json").read_text())
    assert len(callsites) >= 8
    for text, entry in callsites.items():
        assert entry["status"] == "measured", (text, entry)
        assert entry["libc"] == 0 and entry["objects_per_call"] <= 8, (text, entry)
    report = json.loads((tmp_path / "accounting.json").read_text())
    for name, entry in report.items():
        for phase, numbers in entry.items():
            assert numbers["objects_per_call"] <= 12, (name, phase, numbers)
            assert numbers["pymem_raw_mem_obj_libc"][3] == 0, (name, phase, numbers)
            # Identities live in the recorder's reserved storage; the process
            # keeps no per-call object (a few blocks of allocator slack allowed).
            assert numbers["retained_blocks_delta"] <= 400, (name, phase, numbers)


# ------------------------------------------------ hook sites in the source
SITES = [
    "vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py",
    "vllm/entrypoints/openai/completion/serving.py",
    "vllm/entrypoints/openai/chat_completion/serving.py",
    "vllm/v1/engine/async_llm.py",
    "vllm/v1/engine/llm_engine.py",
]
ALLOWED_CALLS = {"len", "getattr"}
BANNED_NODES = (
    ast.JoinedStr, ast.Await, ast.With, ast.Import, ast.ImportFrom, ast.For,
    ast.While, ast.Try, ast.ListComp, ast.DictComp, ast.SetComp,
    ast.GeneratorExp, ast.Lambda,
)  # fmt: skip


def hook_blocks(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        walrus = [n for n in ast.walk(node.test) if isinstance(n, ast.NamedExpr)]
        if any(
            isinstance(n.target, ast.Name) and n.target.id in ("p", "kvp")
            for n in walrus
        ):
            yield node


def callsite_expressions():
    """Every non-trivial test and argument expression of every hook block, as
    ``{source text: [file:line, ...]}``; plain names and constants cost nothing."""
    found = {}
    for relative in SITES:
        tree = ast.parse((ROOT / relative).read_text())
        for block in hook_blocks(tree):
            texts = [ast.unparse(block.test)]
            for node in (n for b in block.body for n in ast.walk(b)):
                if isinstance(node, ast.Call):
                    texts += [
                        ast.unparse(a) for a in node.args
                        if not isinstance(a, (ast.Name, ast.Constant))
                    ]  # fmt: skip
            for text in texts:
                found.setdefault(text, []).append(f"{relative}:{block.lineno}")
    return found


def test_the_hook_callsites_format_nothing_and_do_no_io_or_blocking():
    """Serving hooks may build small transient integers, never format a string,
    open a file or socket, import, wait or iterate a request collection."""
    found = 0
    banned_names = {"open", "print", "sleep", "join", "wait", "acquire", "write"}
    for relative in SITES:
        tree = ast.parse((ROOT / relative).read_text())
        for block in hook_blocks(tree):
            found += 1
            for node in ast.walk(block):
                assert not isinstance(node, BANNED_NODES), (relative, node.lineno)
                if isinstance(node, ast.Call):
                    func = node.func
                    name = func.id if isinstance(func, ast.Name) else func.attr
                    assert name not in banned_names, (relative, node.lineno, name)
                    if isinstance(func, ast.Name):
                        assert name in ALLOWED_CALLS, (relative, node.lineno, name)
    assert found >= 30, found
