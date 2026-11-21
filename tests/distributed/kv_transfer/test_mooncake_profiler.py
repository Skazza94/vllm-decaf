# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Mooncake L1 adapter on the real connector's scheduler and worker."""

import asyncio
import gc
import json
import os
import signal
import threading
import time
from unittest.mock import patch
from uuid import uuid4

import pytest
from mooncake_profiled import (
    AsyncMock,
    _region,
    _xfer_meta,
    events_of,
    make_scheduler,
    patch_worker_dependencies,
    profiler_extra,
    ready_send_meta,
    serve,
    sock,
    worker_of,
)

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    PullReqMeta,
    SendBlockMeta,
)
from vllm.kv_profiling import config as profiler_config_module
from vllm.kv_profiling.clock import ClockGroup
from vllm.kv_profiling.perfetto import _load, export_traces
from vllm.kv_profiling.schema import Failure, Subject

pytest.importorskip("vllm._kv_profiler")

BATCH_KINDS = {
    "transfer_bind",
    "prepared",
    "post_begin",
    "post_return",
    "complete_observed",
    "failed",
    "transfer_retire",
}


def batches(events):
    """Events of each write batch (those with a generation), by generation."""
    found = {}
    for e in events:
        if e["transfer"]:
            found.setdefault(e["transfer"], []).append(e)
    return found


def names(batch):
    return [e["kind"] if e["kind"] != "request_bind" else "member" for e in batch]


# ---------------------------------------------------------------- off and invalid
def test_absent_and_off_attach_nothing_and_import_no_recorder(tmp_path):
    for extra in (None, {"network_profiler": {"level": "off"}}):
        with worker_of("kv_producer", extra) as worker:
            assert worker._kvp is None
        assert make_scheduler("kv_producer", extra)._kvp is None


def _refused_worker(extra):
    """Build a worker whose configuration is refused; nothing of it stays referenced."""
    with pytest.raises(ValueError), worker_of("kv_producer", extra):
        pass


def test_invalid_configuration_is_refused_even_when_it_is_off_or_unknown(tmp_path):
    for bad in (
        {"level": "l3"},
        {"level": "l1"},
        {"level": "off", "clock_bind": "nohost"},
        {
            "level": "l1",
            "run_id": "6e4c5b6b-7a77-4c51-93f2-0d2d0f8b8c11",
            "output_dir": str(tmp_path),
            "clock_bind": ":9",
        },
        {
            "level": "l1",
            "run_id": "6e4c5b6b-7a77-4c51-93f2-0d2d0f8b8c11",
            "output_dir": str(tmp_path),
            "clock_peers": {"e": "h:70000"},
        },
        {
            "level": "l1",
            "run_id": "6e4c5b6b-7a77-4c51-93f2-0d2d0f8b8c11",
            "output_dir": str(tmp_path),
            "inflight_cap": 0,
        },
    ):
        with pytest.raises(ValueError):
            make_scheduler("kv_producer", {"network_profiler": bad})
        _refused_worker({"network_profiler": bad})
        with patch_worker_dependencies():
            gc.collect()  # the half-built worker's __del__, dependencies patched


def test_the_default_registry_bound_never_exceeds_the_handle_slots(tmp_path):
    from vllm.kv_profiling.config import NetworkProfilerConfig

    common = {"level": "l1", "run_id": str(uuid4()), "output_dir": str(tmp_path)}
    assert NetworkProfilerConfig(**common).effective_inflight_cap == 1024
    assert NetworkProfilerConfig(**common, handle_slots=2).effective_inflight_cap == 2
    assert NetworkProfilerConfig(**common, inflight_cap=7).effective_inflight_cap == 7
    with pytest.raises(ValueError, match="inflight_cap"):
        NetworkProfilerConfig(**common, handle_slots=4, inflight_cap=5)


def test_an_enabled_level_attaches_through_the_shared_rule(tmp_path):
    """The connector never compares a literal level: every level in the shared
    configuration's enabled set, L3 included, attaches Mooncake too."""
    assert "l3" in profiler_config_module.ENABLED_LEVELS
    scheduler = make_scheduler("kv_producer", profiler_extra(tmp_path, level="l3"))
    try:
        assert scheduler._kvp is not None and scheduler._kvp.config.level == "l3"
    finally:
        scheduler.close_profiler()


# ------------------------------------------------------------------- write batches
@pytest.mark.asyncio
async def test_a_write_batch_has_one_generation_in_the_documented_order(tmp_path):
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        ready_send_meta(worker, "p-req", "t1")
        response = await serve(worker, {"d-req": ("t1", [[2]])})
        assert response.ok_reqs == ["d-req"]
        profiler = worker._kvp
        events = events_of(profiler)
    (batch,) = batches(events).values()
    assert names(batch) == [
        "transfer_bind",
        "prepared",
        "post_begin",
        "post_return",
        "complete_observed",
        "member",
        "transfer_retire",
    ]
    bind, _, begin, _, complete, member, _ = batch
    assert (
        bind["peer"] == "consumer:1234" and bind["subject"] == 0 and bind["aux0"] == 1
    )
    # The write is posted with the connector's own totals, computed once.
    assert (
        begin["bytes"] == complete["bytes"] == 256
        and begin["selected_descriptors"] == 1
    )
    assert (
        complete["selected_descriptors"] == 1 and complete["duration0_ns"] is not None
    )
    assert complete["status"] == 0 and complete["subject"] == Subject.WORKER
    assert (
        member["request"],
        member["handoff"],
        member["subject"],
        member["status"],
    ) == ("d-req", "t1", Subject.BATCH_MEMBER, 0)
    # The producer's own request is done once every consumer rank pulled.
    assert [e["request"] for e in events if e["kind"] == "senders_done"] == ["p-req"]
    assert not any(e["unbound"] for e in events)
    summary = json.loads((profiler.output_dir / "profiler.json").read_text())
    assert (
        summary["observation_complete"] is True
        and summary["interrupted_generations"] == 0
    )


@pytest.mark.asyncio
async def test_two_staggered_ready_subsets_are_two_generations(tmp_path):
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        first = ready_send_meta(worker, "p1", "t1")
        second = ready_send_meta(worker, "p2", "t2", ready=False)

        async def later():
            await asyncio.sleep(0.2)
            second.ready.set()

        task = asyncio.create_task(later())
        response = await serve(worker, {"d1": ("t1", [[2]]), "d2": ("t2", [[3]])})
        await task
        assert first.sent == 1 and second.sent == 1 and response.ok_reqs == ["d2"]
        events = events_of(worker._kvp)
    found = batches(events)
    assert len(found) == 2
    members = []
    for batch in found.values():
        assert names(batch) == [
            "transfer_bind",
            "prepared",
            "post_begin",
            "post_return",
            "complete_observed",
            "member",
            "transfer_retire",
        ]
        members.append((batch[-2]["request"], batch[-2]["handoff"]))
    assert sorted(members) == [("d1", "t1"), ("d2", "t2")]
    assert len({e["transfer"] for e in events if e["transfer"]}) == 2


@pytest.mark.asyncio
async def test_mixed_members_zero_payload_and_rejected(tmp_path):
    """One batch of a data member, a rejected one (group count mismatch) and one with no
    blocks: the data is written, the rejected member fails by request, the empty one is
    answered. An all-rejected batch and a replica's all-zero batch say so explicitly."""
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        for p, t in (("p1", "t1"), ("p2", "t2"), ("p3", "t3")):
            ready_send_meta(worker, p, t)
        response = await serve(
            worker,
            {
                "data": ("t1", [[2]]),
                "rejected": ("t2", [[2], [3]]),
                "empty": ("t3", [[]]),
            },
        )
        assert sorted(response.ok_reqs) == ["data", "empty"] and response.err_reqs == [
            "rejected"
        ]
        # Only the rejected member: nothing to write and nothing accepted.
        ready_send_meta(worker, "p4", "t4")
        response = await serve(
            worker, {"only-rejected": ("t4", [[2], [3]])}, remote_port=1235
        )
        assert response.err_reqs == ["only-rejected"]
        # Only a member with no blocks: answered, no data moved.
        ready_send_meta(worker, "p5", "t5")
        response = await serve(worker, {"only-empty": ("t5", [[]])}, remote_port=1236)
        assert response.ok_reqs == ["only-empty"]
        events = events_of(worker._kvp)
    mixed, rejected, empty = batches(events).values()
    assert names(mixed)[:5] == [
        "transfer_bind",
        "prepared",
        "post_begin",
        "post_return",
        "complete_observed",
    ]
    statuses = {e["request"]: e["status"] for e in mixed if e["kind"] == "request_bind"}
    assert statuses == {"data": 0, "rejected": 1, "empty": 0}
    failed_requests = [
        (e["request"], e["status"])
        for e in events
        if e["kind"] == "failed" and not e["transfer"]
    ]
    assert ("rejected", int(Failure.MEMBER_REJECTED)) in failed_requests
    assert names(rejected) == ["transfer_bind", "failed", "member", "transfer_retire"]
    assert (rejected[1]["status"], rejected[1]["subject"]) == (
        int(Failure.MEMBER_REJECTED),
        Subject.TRANSFER,
    )
    assert names(empty) == [
        "transfer_bind",
        "complete_observed",
        "member",
        "transfer_retire",
    ]
    zero = empty[1]
    assert (zero["status"], zero["bytes"], zero["selected_descriptors"]) == (2, 0, 0)
    assert not any(e["kind"] in ("post_begin", "post_return") for e in empty + rejected)


def members_of(batch):
    return {
        e["request"]: (e["bytes"], e["selected_descriptors"], e["status"])
        for e in batch
        if e["kind"] == "request_bind"
    }


@pytest.mark.asyncio
async def test_members_carry_their_exact_shares_and_partition_the_batch_totals(
    tmp_path,
):
    """Three data members (one run of three blocks coalesced into one descriptor, one block,
    two separate runs) and a member with no blocks: each member's share is its slice of the
    connector's own lists, and the shares sum to the batch's measured totals exactly."""
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        for p, t, blocks in (
            ("p1", "t1", ((2, 3, 4),)),
            ("p2", "t2", ((7,),)),
            ("p3", "t3", ((20, 22),)),
            ("p4", "t4", ((9,),)),
        ):
            ready_send_meta(worker, p, t, blocks=blocks)
        response = await serve(
            worker,
            {
                "a": ("t1", [[2, 3, 4]]),
                "b": ("t2", [[7]]),
                "c": ("t3", [[20, 22]]),
                "empty": ("t4", [[]]),
            },
        )
        assert sorted(response.ok_reqs) == ["a", "b", "c", "empty"]
        assert worker._sender_spans is None, "consumed by the caller, never left behind"
        events = events_of(worker._kvp)
    (batch,) = batches(events).values()
    bind = batch[0]
    begin = next(e for e in batch if e["kind"] == "post_begin")
    complete = next(e for e in batch if e["kind"] == "complete_observed")
    shares = members_of(batch)
    assert shares == {
        "a": (768, 1, 0),
        "b": (256, 1, 0),
        "c": (512, 2, 0),
        "empty": (0, 0, 0),
    }
    assert bind["aux0"] == len(shares) == 4
    assert (begin["bytes"], begin["selected_descriptors"]) == (1536, 4)
    assert (complete["bytes"], complete["selected_descriptors"]) == (1536, 4)
    assert sum(v[0] for v in shares.values()) == begin["bytes"]
    assert sum(v[1] for v in shares.values()) == begin["selected_descriptors"]


@pytest.mark.asyncio
async def test_a_rejected_member_has_a_zero_share_and_the_others_still_sum_to_the_batch(
    tmp_path,
):
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        for p, t in (("p1", "t1"), ("p2", "t2")):
            ready_send_meta(worker, p, t)
        await serve(worker, {"data": ("t1", [[2]]), "rejected": ("t2", [[2], [3]])})
        events = events_of(worker._kvp)
    (batch,) = batches(events).values()
    shares = members_of(batch)
    assert shares == {"data": (256, 1, 0), "rejected": (0, 0, 1)}
    begin = next(e for e in batch if e["kind"] == "post_begin")
    assert (begin["bytes"], begin["selected_descriptors"]) == (256, 1)


@pytest.mark.asyncio
async def test_a_failed_write_keeps_the_planned_shares_and_an_unfinished_build_leaves_them_unknown(
    tmp_path,
):
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        ready_send_meta(worker, "p1", "t1", blocks=((2, 3),))
        worker.engine.batch_transfer_sync_write = lambda *a: 7
        response = await serve(worker, {"d1": ("t1", [[2, 3]])})
        assert response.err_reqs == ["d1"]
        # The build raises midway: the earlier batch's spans are never reused, the members are unknown.
        worker._sender_spans = [("stale", 0, 1)]
        ready_send_meta(worker, "p2", "t2")
        with (
            patch.object(
                worker,
                "_get_sender_transfer_plan",
                side_effect=RuntimeError("build failed"),
            ),
            pytest.raises(RuntimeError, match="build failed"),
        ):
            await serve(worker, {"d2": ("t2", [[2]])}, remote_port=1235)
        assert worker._sender_spans is None
        events = events_of(worker._kvp)
    failed, unbuilt = batches(events).values()
    assert members_of(failed) == {"d1": (512, 1, 1)}, "planned, never confirmed"
    assert members_of(unbuilt) == {"d2": (None, None, 1)}, "unknown, not zero"


@pytest.mark.asyncio
async def test_a_write_that_raises_after_the_build_keeps_the_known_shares(tmp_path):
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        ready_send_meta(worker, "p1", "t1", blocks=((2, 3),))

        def boom(*args):
            raise RuntimeError("engine exploded")

        worker.engine.batch_transfer_sync_write = boom
        with pytest.raises(RuntimeError):
            await serve(worker, {"d1": ("t1", [[2, 3]])})
        events = events_of(worker._kvp)
    (batch,) = batches(events).values()
    assert members_of(batch) == {"d1": (512, 1, 1)}


@pytest.mark.asyncio
async def test_an_unobserved_batch_builds_no_spans_and_the_build_keeps_its_five_values(
    tmp_path,
):
    with worker_of("kv_producer", None) as worker:
        ready_send_meta(worker, "p1", "t1")
        response = await serve(worker, {"d1": ("t1", [[2]])})
        assert response.ok_reqs == ["d1"]
        assert worker._kvp is None and getattr(worker, "_sender_spans", None) is None
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        ready_send_meta(worker, "p1", "t1")
        meta = _xfer_meta(
            [_region(0x2000, row_offset=-1, layer_name="model.layers.0.self_attn")],
            {"d1": ("t1", [[2]])},
            remote_hostname="consumer",
            remote_port=1234,
            remote_tp_rank=0,
        )
        local, remote, _ = worker._prepare_transfer_regions(meta)
        built = await worker._build_transfer_params(
            [("d1", worker.reqs_need_send["t1"])], meta, local, remote
        )
        assert len(built) == 5
        assert worker._sender_spans == [("d1", 0, len(built[2]))]


@pytest.mark.asyncio
async def test_a_batch_the_profiler_declined_builds_no_spans_even_with_a_profiler_attached(
    tmp_path,
):
    """The observed-batch guard: a full registry (or a closing profiler) refuses the batch's bind, and then the build does
    none of the span work."""
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        worker._kvp._cap = 0
        ready_send_meta(worker, "p1", "t1")
        response = await serve(worker, {"d1": ("t1", [[2]])})
        assert response.ok_reqs == ["d1"], "the batch is still served"
        assert worker._sender_spans is None
        assert worker._kvp.inflight_overflow == 1
        events = events_of(worker._kvp)
    assert not any(
        e["kind"] == "request_bind" and e["subject"] == Subject.BATCH_MEMBER
        for e in events
    )


def test_the_share_helper_refuses_spans_that_do_not_partition_the_lists(tmp_path):
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        profiler = worker._kvp
        lengths = [256, 512, 256]
        assert profiler.member_shares([("a", 0, 2), ("b", 2, 3)], lengths) == {
            "a": (768, 2),
            "b": (256, 1),
        }
        for bad in (
            [("a", 0, 2)],
            [("a", 0, 2), ("b", 3, 3)],
            [("a", 0, 4)],
            [("a", 1, 3)],
            [("a", 2, 1), ("b", 1, 3)],
        ):
            assert profiler.member_shares(bad, lengths) is None, bad
        assert profiler.member_shares([("a", 0, 3)], lengths) == {"a": (1024, 3)}


def test_the_member_share_capability_follows_the_producer_role():
    from vllm.kv_profiling.mooncake import capabilities

    assert capabilities("worker", "kv_producer")["batch_member_shares"] is True
    assert capabilities("worker", "kv_consumer")["batch_member_shares"] is False
    assert capabilities("scheduler", "kv_producer")["batch_member_shares"] is False


@pytest.mark.asyncio
async def test_a_failed_write_and_an_engine_exception_never_fake_a_return(tmp_path):
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        ready_send_meta(worker, "p1", "t1")
        worker.engine.batch_transfer_sync_write = lambda *a: 7
        response = await serve(worker, {"d1": ("t1", [[2]])})
        assert response.err_reqs == ["d1"]

        def boom(*args):
            raise RuntimeError("engine exploded")

        ready_send_meta(worker, "p2", "t2")
        worker.engine.batch_transfer_sync_write = boom
        with pytest.raises(RuntimeError):
            await serve(worker, {"d2": ("t2", [[2]])}, remote_port=1235)
        events = events_of(worker._kvp)
    failed, raised = batches(events).values()
    assert names(failed) == [
        "transfer_bind",
        "prepared",
        "post_begin",
        "post_return",
        "failed",
        "member",
        "transfer_retire",
    ]
    assert (failed[4]["status"], failed[4]["aux0"]) == (int(Failure.TRANSFER_FAILED), 7)
    assert [
        (e["request"], e["status"])
        for e in events
        if e["kind"] == "failed" and not e["transfer"]
    ] == [
        ("d1", int(Failure.TRANSFER_FAILED)),
        ("d2", int(Failure.SEND_EXCEPTION)),
    ]
    # The exception: posted but never returned, a failure, the member the batch
    # carried (failed, not fabricated as sent) and one retire.
    assert names(raised) == [
        "transfer_bind",
        "prepared",
        "post_begin",
        "failed",
        "member",
        "transfer_retire",
    ]
    assert (raised[3]["status"], raised[3]["aux0"]) == (int(Failure.SEND_EXCEPTION), 0)
    member = raised[4]
    assert (member["request"], member["transfer"] == raised[0]["transfer"]) == (
        "d2",
        True,
    )
    assert (member["status"], member["subject"]) == (1, Subject.BATCH_MEMBER)
    assert member["handoff"] == "t2"


@pytest.mark.asyncio
async def test_cancellation_retires_in_the_frame_and_a_late_return_records_nothing(
    tmp_path,
):
    release, entered = threading.Event(), threading.Event()

    def slow(*args):
        entered.set()
        release.wait(10)
        return 0

    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        ready_send_meta(worker, "p1", "t1")
        worker.engine.batch_transfer_sync_write = slow
        task = asyncio.create_task(serve(worker, {"d1": ("t1", [[2]])}))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await asyncio.sleep(0.2)  # the executor thread finishes and calls its hooks
        profiler = worker._kvp
        events = events_of(profiler)
    (batch,) = batches(events).values()
    assert names(batch) == [
        "transfer_bind",
        "prepared",
        "post_begin",
        "failed",
        "member",
        "transfer_retire",
    ]
    assert (batch[3]["status"], batch[3]["aux0"]) == (int(Failure.SEND_EXCEPTION), 1)
    assert (batch[4]["request"], batch[4]["handoff"], batch[4]["status"]) == (
        "d1",
        "t1",
        1,
    )
    assert ("d1", int(Failure.SEND_EXCEPTION), 1) in [
        (e["request"], e["status"], e["aux0"])
        for e in events
        if e["kind"] == "failed" and not e["transfer"]
    ]
    summary = json.loads((profiler.output_dir / "profiler.json").read_text())
    assert summary["late_hooks"] >= 2 and summary["observation_complete"] is True
    assert not any(e["unbound"] for e in events)


@pytest.mark.asyncio
async def test_a_response_that_cannot_be_sent_keeps_each_member_once(tmp_path):
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        ready_send_meta(worker, "p1", "t1")
        connection = sock()
        connection.send_multipart = AsyncMock(side_effect=ConnectionError("gone"))
        with pytest.raises(ConnectionError):
            await serve(worker, {"d1": ("t1", [[2]])}, connection=connection)
        events = events_of(worker._kvp)
    (batch,) = batches(events).values()
    assert names(batch) == [
        "transfer_bind",
        "prepared",
        "post_begin",
        "post_return",
        "complete_observed",
        "member",
        "transfer_retire",
    ]
    assert (batch[5]["request"], batch[5]["status"]) == ("d1", 0)
    assert not [e for e in events if e["kind"] == "failed" and not e["transfer"]]


@pytest.mark.asyncio
async def test_an_exception_before_any_write_still_names_every_member(tmp_path):
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        ready_send_meta(worker, "p1", "t1")
        ready_send_meta(worker, "p2", "t2")

        async def broken(*args, **kwargs):
            raise RuntimeError("cannot build the transfer")

        worker._build_transfer_params = broken
        with pytest.raises(RuntimeError):
            await serve(worker, {"d1": ("t1", [[2]]), "d2": ("t2", [[3]])})
        events = events_of(worker._kvp)
    (batch,) = batches(events).values()
    assert names(batch) == [
        "transfer_bind",
        "failed",
        "member",
        "member",
        "transfer_retire",
    ]
    assert {(e["request"], e["handoff"]) for e in batch if e["request"]} == {
        ("d1", "t1"),
        ("d2", "t2"),
    }
    assert not any(e["kind"] in ("post_begin", "post_return") for e in batch)


# ----------------------------------------------------- disabling and fork isolation
def test_an_injected_recorder_error_disables_every_generation_hook(tmp_path):
    from vllm.kv_profiling.nixl import _discard

    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        profiler = worker._kvp
        handle = profiler.batch_bind("consumer:1", 0, 1)
        assert handle is not None
        calls = []

        def failing(*args):
            calls.append(args[0])
            raise RuntimeError("injected recorder error")

        profiler._record = failing
        profiler.batch_prepared(handle)
        assert len(calls) == 1 and profiler.hook_errors == 1
        assert profiler._record is _discard
        hooks = (
            lambda: profiler.batch_bind("consumer:1", 0, 1),
            lambda: profiler.write_begin(handle, 1, 1),
            lambda: profiler.write_returned(handle),
            lambda: profiler.write_complete(handle, 1, 1, 1),
            lambda: profiler.write_failed(handle, 1),
            lambda: profiler.batch_no_data(handle),
            lambda: profiler.batch_rejected(handle),
            lambda: profiler.batch_exception(handle, False, [("d", "t")]),
            lambda: profiler.batch_member(handle, "d", "t", True),
            lambda: profiler.batch_retire(handle),
        )
        for hook in hooks:
            hook()
        assert len(calls) == 1
        assert profiler.hook_errors == 1
        summary_profiler = profiler
    summary = json.loads((summary_profiler.output_dir / "profiler.json").read_text())
    assert summary["observation_complete"] is False and summary["hook_errors"] == 1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_forked_child_neither_waits_on_the_held_lock_nor_touches_the_parents_ring(
    tmp_path,
):
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        profiler = worker._kvp
        handle = profiler.batch_bind("parent:1", 0, 1)
        held, release = threading.Event(), threading.Event()

        def hold():
            with profiler._glock, profiler._lock:
                held.set()
                release.wait(30)

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        assert held.wait(10)
        reader, writer = os.pipe()
        child = os.fork()
        if child == 0:
            code = 1
            try:
                signal.alarm(20)
                started = time.monotonic()
                bound = profiler.batch_bind("child:1", 0, 1)
                profiler.batch_prepared(handle)
                profiler.write_begin(handle, 1, 1)
                profiler.batch_member(handle, "d", "t", True)
                profiler.batch_exception(handle, False, [("d", "t")])
                profiler.batch_retire(handle)
                profiler.close()
                prompt = time.monotonic() - started < 5
                os.write(writer, b"ok" if bound is None and prompt else b"bad")
                code = 0
            finally:
                os._exit(code)
        os.close(writer)
        release.set()
        holder.join(10)
        _, status = os.waitpid(child, 0)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, status
        assert os.read(reader, 8) == b"ok"
        os.close(reader)
        profiler.write_begin(handle, 1, 1)
        profiler.batch_retire(handle)
        events = events_of(profiler)
    assert [e["kind"] for e in events] == [
        "transfer_bind",
        "post_begin",
        "transfer_retire",
    ]
    assert events[0]["peer"] == "parent:1"


@pytest.mark.asyncio
async def test_the_ready_timeout_records_request_failures_before_the_response(tmp_path):
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        ready_send_meta(worker, "p1", "t1", ready=False)
        order = []
        profiler = worker._kvp
        recorded = profiler.ready_timeout

        def traced(pending):
            order.append("failure recorded")
            recorded(pending)

        profiler.ready_timeout = traced
        connection = sock()
        connection.send_multipart = AsyncMock(
            side_effect=lambda *a: order.append("response sent")
        )
        with patch("vllm.envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", 0.2):
            response = await serve(worker, {"d1": ("t1", [[2]])}, connection=connection)
        assert response.err_reqs == ["d1"] and "Timeout" in response.err_msg
        events = events_of(profiler)
    assert order == ["failure recorded", "response sent"]
    timeout = [e for e in events if e["kind"] == "failed"]
    assert [
        (e["request"], e["handoff"], e["status"], e["subject"]) for e in timeout
    ] == [("d1", "t1", int(Failure.READY_TIMEOUT), Subject.REQUEST)]
    assert not batches(events), (
        "a timeout is not a write batch and never a transport failure"
    )


@pytest.mark.asyncio
async def test_a_response_that_cannot_be_sent_still_retires_the_batch(tmp_path):
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        ready_send_meta(worker, "p1", "t1")
        regions = [
            _region(0x2000, row_offset=-1, layer_name="model.layers.0.self_attn")
        ]
        metadata = _xfer_meta(
            regions, {"d1": ("t1", [[2]])}, remote_hostname="consumer", remote_port=1234
        )
        connection = sock()
        connection.send_multipart = AsyncMock(side_effect=ConnectionError("peer gone"))
        with (
            patch.object(worker, "sender_loop", asyncio.get_running_loop()),
            pytest.raises(ConnectionError),
        ):
            await worker.send_kv_to_decode(b"consumer", connection, metadata)
        profiler = worker._kvp
        events = events_of(profiler)
    (batch,) = batches(events).values()
    assert names(batch) == [
        "transfer_bind",
        "prepared",
        "post_begin",
        "post_return",
        "complete_observed",
        "member",
        "transfer_retire",
    ]
    summary = json.loads((profiler.output_dir / "profiler.json").read_text())
    assert (
        summary["interrupted_generations"] == 0
        and summary["observation_complete"] is True
    )


# ------------------------------------------------------------- interruption and bounds
@pytest.mark.asyncio
async def test_closing_while_a_write_is_live_is_an_incomplete_observation(tmp_path):
    release, entered = threading.Event(), threading.Event()

    def slow(*args):
        entered.set()
        release.wait(10)
        return 0

    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        ready_send_meta(worker, "p1", "t1")
        worker.engine.batch_transfer_sync_write = slow
        task = asyncio.create_task(serve(worker, {"d1": ("t1", [[2]])}))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        profiler = worker._kvp
        started = time.monotonic()
        worker.close_profiler()  # races the executor thread inside the write
        assert time.monotonic() - started < 5, (
            "the close does not wait for the transfer"
        )
        release.set()
        response = (
            await task
        )  # the late return neither fails nor completes the observation
        assert response.ok_reqs == ["d1"]
        await asyncio.sleep(0.1)
        directory = profiler.output_dir
        summary = json.loads((directory / "profiler.json").read_text())
        events = events_of(profiler)
    assert summary["interrupted_generations"] == 1
    assert summary["observation_complete"] is False
    assert summary["complete"] is True and summary["saved"] is True, (
        "no native drop: the record loss counters stay clean"
    )
    (batch,) = batches(events).values()
    assert names(batch) == [
        "transfer_bind",
        "prepared",
        "post_begin",
        "failed",
        "transfer_retire",
    ]
    assert (batch[3]["status"], batch[3]["subject"]) == (
        int(Failure.INTERRUPTED),
        Subject.TRANSFER,
    )
    # The reader distinguishes the stopped observation from a transport failure.
    documents = export_traces([directory / "events.jsonl"], str(tmp_path / "trace"))
    (path,) = documents.values()
    trace = json.loads(path.read_text())
    (capture,) = trace["metadata"]["captures"]
    assert (
        capture["loss"] is True
        and capture["observation"]["observation_complete"] is False
    )
    interrupted = [
        e for e in trace["traceEvents"] if e.get("name") == "observation_interrupted"
    ]
    assert (
        interrupted
        and interrupted[0]["args"]["failure_class"] == "observation_interrupted"
    )
    assert not any(
        e.get("args", {}).get("failure_class") == "connector_reported"
        for e in trace["traceEvents"]
    )


def test_concurrent_closes_share_one_report_and_one_interruption(tmp_path):
    scheduler = make_scheduler("kv_producer", profiler_extra(tmp_path))
    profiler = scheduler._kvp
    handles = [profiler.batch_bind("c:1", 0, 1) for _ in range(3)]
    assert all(handles)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(profiler.close()))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(results) == 4 and all(r is results[0] for r in results)
    summary = json.loads((profiler.output_dir / "profiler.json").read_text())
    assert summary["interrupted_generations"] == 3
    events = events_of(profiler)
    assert sum(e["kind"] == "transfer_retire" for e in events) == 3
    assert (
        sum(
            e["kind"] == "failed" and e["status"] == int(Failure.INTERRUPTED)
            for e in events
        )
        == 3
    )
    # Late hooks after the close: nothing new, nothing unbound.
    assert profiler.batch_bind("c:1", 0, 1) is None
    profiler.write_begin(handles[0], 1, 1)
    profiler.batch_retire(handles[0])


def test_the_registry_is_bounded_and_overflow_is_an_incomplete_observation(tmp_path):
    scheduler = make_scheduler("kv_producer", profiler_extra(tmp_path, inflight_cap=2))
    profiler = scheduler._kvp
    first, second = profiler.batch_bind("c:1", 0, 1), profiler.batch_bind("c:1", 0, 1)
    assert first and second and profiler.batch_bind("c:1", 0, 1) is None
    profiler.batch_retire(first)
    third = profiler.batch_bind("c:1", 0, 1)
    assert third, "a retired slot is reusable"
    for handle in (second, third):
        profiler.batch_retire(handle)
    events = events_of(profiler)
    summary = json.loads((profiler.output_dir / "profiler.json").read_text())
    assert (
        summary["inflight_overflow"] == 1 and summary["observation_complete"] is False
    )
    assert sum(e["kind"] == "transfer_bind" for e in events) == 3
    assert sum(e["kind"] == "transfer_retire" for e in events) == 3


def test_out_of_order_and_repeated_calls_record_nothing_twice(tmp_path):
    scheduler = make_scheduler("kv_producer", profiler_extra(tmp_path))
    profiler = scheduler._kvp
    handle = profiler.batch_bind("c:1", 0, 1)
    profiler.write_returned(handle)  # before POST_BEGIN: refused
    profiler.write_complete(handle, 1, 1, 5)  # before the call returned: refused
    profiler.write_begin(handle, 1, 1)
    profiler.write_begin(handle, 1, 1)
    profiler.write_returned(handle)
    profiler.write_complete(handle, 1, 1, 5)
    profiler.write_failed(handle, 1)  # a second terminal: refused
    profiler.batch_member(handle, "d", "t", True)
    profiler.batch_retire(handle)
    profiler.batch_retire(handle)
    profiler.batch_member(handle, "d", "t", True)  # after the retire: refused
    batch = list(batches(events_of(profiler)).values())[0]
    assert names(batch) == [
        "transfer_bind",
        "post_begin",
        "post_return",
        "complete_observed",
        "member",
        "transfer_retire",
    ]
    assert (
        json.loads((profiler.output_dir / "profiler.json").read_text())["late_hooks"]
        >= 5
    )


def test_peer_identities_are_endpoints_and_ranks_stay_in_the_subject(tmp_path):
    scheduler = make_scheduler("kv_producer", profiler_extra(tmp_path))
    profiler = scheduler._kvp
    handle = profiler.batch_bind("10.0.0.7:5555", 3, 2)
    profiler.batch_retire(handle)
    (bind, _) = batches(events_of(profiler)).popitem()[1]
    assert bind["peer"] == "10.0.0.7:5555" and isinstance(bind["peer"], str)
    assert bind["subject"] == 3 and bind["aux0"] == 2


# ------------------------------------------------------------ scheduler and receiver
def _params(request, **extra):
    request.kv_transfer_params.update(extra)
    return request


def test_the_producer_handoff_is_recorded_after_the_success_checks_only(tmp_path):
    from tests.v1.kv_connector.unit.utils import create_request
    from vllm.v1.request import RequestStatus

    scheduler = make_scheduler("kv_producer", profiler_extra(tmp_path))
    profiler = scheduler._kvp
    done = _params(
        create_request(request_id=1, do_remote_decode=True), transfer_id="tx-1"
    )
    scheduler.on_new_request(done)
    done.status = RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler.request_finished(done, block_ids=([10, 11],))[0] is True
    aborted = _params(
        create_request(request_id=2, do_remote_decode=True), transfer_id="tx-2"
    )
    scheduler.on_new_request(aborted)
    aborted.status = RequestStatus.FINISHED_ABORTED
    assert scheduler.request_finished(aborted, block_ids=([13],))[0] is False
    no_blocks = _params(
        create_request(request_id=3, do_remote_decode=True), transfer_id="tx-3"
    )
    no_blocks.status = RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler.request_finished(no_blocks, block_ids=([],))[0] is False
    events = events_of(profiler)
    handoffs = [e for e in events if e["kind"] == "handoff"]
    assert [
        (e["request"], e["handoff"], e["peer"], e["subject"], e["aux1"])
        for e in handoffs
    ] == [("id-1", "tx-1", "engine-a", 1, 2)]
    assert [e["request"] for e in events if e["kind"] == "request_bind"] == [
        "id-1",
        "id-2",
        "id-3",
    ][:2]


def test_a_decode_request_cancelled_before_its_pull_is_not_a_producer_handoff(tmp_path):
    from tests.v1.kv_connector.unit.utils import create_request

    scheduler = make_scheduler("kv_consumer", profiler_extra(tmp_path))
    request = _params(
        create_request(request_id=4, do_remote_prefill=True),
        transfer_id="tx-4",
        remote_bootstrap_addr="http://127.0.0.1:1",
    )
    scheduler.on_new_request(request)
    assert scheduler.request_finished(request, block_ids=([1],)) == (False, None)
    events = events_of(scheduler._kvp)
    assert [
        (e["kind"], e["request"], e["handoff"], e["peer"], e["subject"]) for e in events
    ] == [("handoff", "id-4", "tx-4", "my-engine-id", 0)]


def test_scheduler_receive_aggregation_is_not_proof_of_success_and_releases_ask(
    tmp_path,
):
    from vllm.v1.outputs import KVConnectorOutput

    scheduler = make_scheduler("kv_both", profiler_extra(tmp_path))
    scheduler.update_connector_output(
        KVConnectorOutput(
            finished_sending={"p1", "p2"},
            finished_recving={"ok", "bad"},
            failed_recving={"bad"},
        )
    )
    events = events_of(scheduler._kvp)
    recv = {
        e["request"]: e["status"] for e in events if e["kind"] == "complete_observed"
    }
    assert recv == {"ok": 0, "bad": 1} and {
        e["subject"] for e in events if e["kind"] == "complete_observed"
    } == {Subject.SCHEDULER}
    assert sorted(e["request"] for e in events if e["kind"] == "release_requested") == [
        "p1",
        "p2",
    ]


@pytest.mark.parametrize(
    "outcomes", [(True, True), (False, True), (True, False), (False, False)]
)
def test_a_receive_completes_once_and_a_failed_peer_keeps_it_failed(tmp_path, outcomes):
    with worker_of("kv_consumer", profiler_extra(tmp_path)) as worker:
        meta = PullReqMeta(
            d_req_id="d-req",
            transfer_id="tx",
            local_block_ids=[[1, 2]],
            remote_engine_id="engine-p",
            remote_bootstrap_addr="127.0.0.1:1",
            pull_tasks_count=2,
        )
        for ok in outcomes:
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.FINISH,
                ok_reqs=["d-req"] if ok else None,
                err_reqs=None if ok else ["d-req"],
                err_msg=None if ok else "peer failed",
            )
            worker.process_pulling_result(response, {"d-req": meta})
        events = events_of(worker._kvp)
    done = [e for e in events if e["kind"] == "complete_observed"]
    failed = [e for e in events if e["kind"] == "failed"]
    if all(outcomes):
        assert [(e["request"], e["handoff"], e["subject"]) for e in done] == [
            ("d-req", "tx", Subject.WORKER)
        ] and not failed
    else:
        assert not done, (
            "another peer's failure keeps the request failed, whatever the last ack"
        )
        assert [(e["request"], e["status"]) for e in failed] == [
            ("d-req", int(Failure.RECEIVE_FAILED))
        ]


@pytest.mark.asyncio
async def test_bootstrap_queries_lease_expiry_and_the_producer_identity_are_recorded(
    tmp_path,
):
    with worker_of("kv_both", profiler_extra(tmp_path)) as worker:
        pull = PullReqMeta(
            d_req_id="d",
            transfer_id="tx",
            local_block_ids=[[1]],
            remote_engine_id="engine-p",
            remote_bootstrap_addr="http://127.0.0.1:1",
        )
        worker._query_bootstrap = AsyncMock()
        worker.receive_kv = lambda *a: None
        worker._remote_agents["engine-p"] = {0: {0: "tcp://x"}}
        await worker.handle_new_engine_id("engine-p", {"d": pull})
        await worker.handle_new_engine_id("engine-missing", {"d": pull})
        metadata = MooncakeConnectorMetadata()
        metadata.reqs_to_send["p-req"] = ("t1", [[1]])
        ready_send_meta(worker, "", "t1", blocks=(), ready=False)
        await worker.record_send_reqs(metadata)
        worker.reqs_need_send["t9"] = SendBlockMeta(
            p_req_id="p9",
            transfer_id="t9",
            local_block_ids=[[1]],
            ready=asyncio.Event(),
            expire_time=0.0,
        )
        expired = await worker.fetch_finished_sending_reqs()
        assert expired == {"p9"}
        events = events_of(worker._kvp)
    handshakes = [
        (e["kind"], e["peer"], e["status"])
        for e in events
        if e["kind"].startswith("handshake")
    ]
    assert handshakes == [
        ("handshake_begin", "engine-p", 0),
        ("handshake_end", "engine-p", 0),
        ("handshake_begin", "engine-missing", 0),
        ("handshake_end", "engine-missing", int(Failure.HANDSHAKE_FAILED)),
    ]
    identity = [e for e in events if e["subject"] == Subject.HANDOFF_IDENTITY]
    assert [(e["kind"], e["request"], e["handoff"]) for e in identity] == [
        ("request_bind", "p-req", "t1")
    ]
    assert [
        (e["request"], e["status"])
        for e in events
        if e["kind"] == "failed" and e["request"] == "p9"
    ] == [("p9", int(Failure.LEASE_EXPIRED))]


# ----------------------------------------------------------------------- clocks
def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_clock_endpoints_are_explicit_and_calibrate_a_real_pair(tmp_path):
    from tests.v1.kv_connector.unit.utils import create_request

    port = _free_port()
    run = profiler_extra(tmp_path)
    producer = make_scheduler(
        "kv_producer",
        {
            "network_profiler": {
                **run["network_profiler"],
                "clock_bind": f"127.0.0.1:{port}",
            }
        },
        engine_id="engine-p",
    )
    assert producer._kvp.server is not None
    consumer = make_scheduler(
        "kv_consumer",
        {
            "network_profiler": {
                **run["network_profiler"],
                "clock_peers": {"engine-p": f"127.0.0.1:{port}"},
            }
        },
        engine_id="engine-d",
    )
    request = _params(
        create_request(request_id=7, do_remote_prefill=True),
        transfer_id="tx-7",
        remote_engine_id="engine-p",
    )
    consumer.on_new_request(request)
    deadline = time.monotonic() + 10
    while not consumer._kvp.peers.results and time.monotonic() < deadline:
        time.sleep(0.05)
    consumer._kvp.close()
    producer._kvp.close()
    calibration = json.loads(
        (consumer._kvp.output_dir / "calibration.json").read_text()
    )
    (peer,) = calibration["peers"]
    assert peer["engine_id"] == "engine-p" and peer["status"] == "ok"
    assert peer["address"] == ["127.0.0.1", port], (
        "the explicit endpoint, no side-channel offset added"
    )
    summary = json.loads((consumer._kvp.output_dir / "profiler.json").read_text())
    assert summary["clock"] == {
        "server": False,
        "peers_configured": 1,
        "unmapped_engines": [],
    }
    assert (
        json.loads((producer._kvp.output_dir / "profiler.json").read_text())["clock"][
            "server"
        ]
        is True
    )


def test_a_peer_without_a_clock_mapping_stays_unaligned_with_a_reason(tmp_path):
    from tests.v1.kv_connector.unit.utils import create_request

    run = profiler_extra(tmp_path)
    consumer = make_scheduler(
        "kv_consumer",
        {
            "network_profiler": {
                **run["network_profiler"],
                "clock_peers": {"other": "127.0.0.1:9"},
            }
        },
    )
    request = _params(
        create_request(request_id=8, do_remote_prefill=True),
        transfer_id="tx-8",
        remote_engine_id="engine-x",
    )
    consumer.on_new_request(request)
    consumer.on_new_request(request)
    consumer._kvp.close()
    summary = json.loads((consumer._kvp.output_dir / "profiler.json").read_text())
    assert summary["clock"]["unmapped_engines"] == ["engine-x"]
    assert (
        json.loads((consumer._kvp.output_dir / "calibration.json").read_text())["peers"]
        == []
    )


def test_capabilities_follow_the_kv_role_and_the_process_role(tmp_path):
    from vllm.kv_profiling.mooncake import capabilities

    p_worker, p_sched = (
        capabilities("worker", "kv_producer"),
        capabilities("scheduler", "kv_producer"),
    )
    d_worker, d_sched = (
        capabilities("worker", "kv_consumer"),
        capabilities("scheduler", "kv_consumer"),
    )
    assert (
        p_worker["write_batches"]
        and p_worker["batch_members"]
        and not p_sched["write_batches"]
    )
    assert p_sched["handoff_blocks"] and not p_worker["handoff_blocks"]
    assert (
        d_worker["receive_completion"]
        and d_sched["receive_completion"]
        and not p_worker["receive_completion"]
    )
    assert not d_worker["write_batches"] and not d_sched["handoff_blocks"]
    assert not any(k.startswith("nixl_") and v for k, v in p_worker.items())
    assert p_worker["transport_metrics"] == "collector"
    scheduler = make_scheduler("kv_consumer", profiler_extra(tmp_path))
    assert scheduler._kvp.config.context["capabilities"]["receive_completion"] is True
    assert scheduler._kvp.config.context["capabilities"]["write_batches"] is False


def test_the_frontend_metadata_follows_the_connector(tmp_path):
    from mooncake_profiled import vllm_config

    from vllm.kv_profiling import attach_frontend

    frontend = attach_frontend(vllm_config("kv_producer", profiler_extra(tmp_path)))
    try:
        caps = frontend.config.context["capabilities"]
        assert frontend.config.context["connector"] == "mooncake"
        assert caps["router_id"] and caps["transport_metrics"] == "collector"
        assert not caps["nixl_post_xfer_durations"] and not caps["write_batches"]
    finally:
        frontend.close()
    assert (
        attach_frontend(
            vllm_config("kv_producer", {"network_profiler": {"level": "off"}})
        )
        is None
    )


# ------------------------------------------------------------------------- reader
@pytest.mark.asyncio
async def test_the_reader_names_the_write_and_pairs_the_handoff_across_processes(
    tmp_path,
):
    from tests.v1.kv_connector.unit.utils import create_request
    from vllm.v1.request import RequestStatus

    run = profiler_extra(tmp_path)
    producer_scheduler = make_scheduler("kv_producer", run, engine_id="engine-p")
    consumer_scheduler = make_scheduler("kv_consumer", run, engine_id="engine-d")
    request = _params(
        create_request(request_id=9, do_remote_decode=True), transfer_id="tx-9"
    )
    producer_scheduler.on_new_request(request)
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    producer_scheduler.request_finished(request, block_ids=([5],))
    decode = _params(
        create_request(request_id=10, do_remote_prefill=True),
        transfer_id="tx-9",
        remote_engine_id="engine-p",
    )
    consumer_scheduler.on_new_request(decode)
    with worker_of("kv_producer", run) as worker:
        ready_send_meta(worker, "id-9", "tx-9")
        await serve(worker, {"id-10": ("tx-9", [[2]])})
        worker_dir = worker._kvp.output_dir
        events_of(worker._kvp)
    events_of(producer_scheduler._kvp)
    events_of(consumer_scheduler._kvp)
    paths = [
        p._kvp.output_dir / "events.jsonl"
        for p in (producer_scheduler, consumer_scheduler)
    ] + [worker_dir / "events.jsonl"]
    reference = _load(paths[0])[0]["clock_id"]
    (document,) = export_traces(
        paths, str(tmp_path / "trace"), ClockGroup(reference, {})
    ).values()
    trace = json.loads(document.read_text())
    out = trace["traceEvents"]
    spans = {e["name"]: e for e in out if e.get("ph") == "X"}
    assert {
        "queue until executor entry",
        "write call",
        "prepared to completed (write batch)",
    } <= set(spans)
    assert "READ until observed DONE" not in spans and not any(
        name.startswith("post call") for name in spans
    )
    complete = next(e for e in out if e.get("name") == "write_complete")
    assert complete["args"]["mooncake_write_duration_ns"] is not None
    assert not any(k.startswith("nixl_") for e in out for k in e.get("args", {}))
    assert (
        spans["write call"]["dur"]
        <= spans["prepared to completed (write batch)"]["dur"]
    )
    flows = [e for e in out if e.get("cat") == "kv_handoff"]
    assert (
        len(flows) == 2
        and trace["metadata"]["cross_process_flows"] == "exact_recorded_handoff_ids"
    )
    assert (
        next(e for e in out if e.get("name") == "batch_member")["args"]["batch_member"]
        is True
    )


def test_repeated_batch_membership_never_splits_a_request_into_lanes_or_flows(tmp_path):
    scheduler = make_scheduler("kv_producer", profiler_extra(tmp_path))
    profiler = scheduler._kvp
    for rank in (0, 1):
        handle = profiler.batch_bind(f"consumer:{1000 + rank}", rank, 1)
        profiler.write_begin(handle, 4, 1)
        profiler.write_returned(handle)
        profiler.write_complete(handle, 4, 1, 10)
        profiler.batch_member(handle, "d-req", "tx", True)
        profiler.batch_retire(handle)
    events_of(profiler)
    (document,) = export_traces(
        [profiler.output_dir / "events.jsonl"], str(tmp_path / "trace")
    ).values()
    trace = json.loads(document.read_text())
    members = [e for e in trace["traceEvents"] if e.get("name") == "batch_member"]
    assert len(members) == 2 and len({(e["pid"], e["tid"]) for e in members}) == 1
    assert not [e for e in trace["traceEvents"] if e.get("cat") == "kv_handoff"]


@pytest.mark.asyncio
async def test_the_exporter_carries_each_members_share_under_member_names(tmp_path):
    with worker_of("kv_producer", profiler_extra(tmp_path)) as worker:
        for p, t, blocks in (("p1", "t1", ((2, 3),)), ("p2", "t2", ((7,),))):
            ready_send_meta(worker, p, t, blocks=blocks)
        await serve(worker, {"a": ("t1", [[2, 3]]), "b": ("t2", [[7]])})
        profiler = worker._kvp
        events_of(profiler)
    (document,) = export_traces(
        [profiler.output_dir / "events.jsonl"], str(tmp_path / "trace")
    ).values()
    trace = json.loads(document.read_text())["traceEvents"]
    members = [e["args"] for e in trace if e.get("name") == "batch_member"]
    assert sorted((m["member_bytes"], m["member_descriptors"]) for m in members) == [
        (256, 1),
        (512, 1),
    ]
    begin = [
        e["args"] for e in trace if e.get("args", {}).get("posted_bytes") is not None
    ]
    assert begin and begin[0]["posted_bytes"] == sum(m["member_bytes"] for m in members)
    assert begin[0]["posted_descriptors"] == sum(
        m["member_descriptors"] for m in members
    )


def test_an_exceptional_and_a_no_data_batch_have_no_successful_return_in_the_reader(
    tmp_path,
):
    scheduler = make_scheduler("kv_producer", profiler_extra(tmp_path))
    profiler = scheduler._kvp
    raised = profiler.batch_bind("consumer:1", 0, 1)
    profiler.write_begin(raised, 4, 1)
    profiler.batch_exception(raised, False)
    profiler.batch_retire(raised)
    empty = profiler.batch_bind("consumer:1", 0, 1)
    profiler.batch_no_data(empty)
    profiler.batch_retire(empty)
    events_of(profiler)
    (document,) = export_traces(
        [profiler.output_dir / "events.jsonl"], str(tmp_path / "trace")
    ).values()
    names_ = [
        e["name"]
        for e in json.loads(document.read_text())["traceEvents"]
        if e.get("ph") == "X"
    ]
    assert "write call" not in names_
    assert (
        "write batch failed" in names_ and "write batch with no data movement" in names_
    )
