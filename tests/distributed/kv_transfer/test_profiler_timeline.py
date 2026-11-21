# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request phases, step spans and the summary's timelines.

The hooks run on real adapter profilers over the native recorder and sink; only the
clock readings the hooks take (and, where a test needs exact intervals, the row
timestamps) are scripted. Nothing here operates a transport.
"""

import json
import shutil
from types import SimpleNamespace
from uuid import uuid4

import pytest

from vllm.kv_profiling import compute
from vllm.kv_profiling.config import NetworkProfilerConfig
from vllm.kv_profiling.core import ProcessRecorder
from vllm.kv_profiling.mooncake import MooncakeProfiler
from vllm.kv_profiling.nixl import FrontendProfiler, NixlProfiler
from vllm.kv_profiling.schema import (
    MARK_EMPTY,
    MARK_PHASE,
    EventKind,
    RequestPhase,
    StepSubject,
    decode_record,
)
from vllm.kv_profiling.summary import summarize

PHASES = {p.name.lower(): int(p) for p in RequestPhase}


class Clock:
    """The monotonic reading the hooks take, set by the test."""

    value = 1_000

    def __call__(self):
        return self.value


@pytest.fixture
def clock(monkeypatch):
    stub = Clock()
    monkeypatch.setattr(compute, "_monotonic_ns", stub)
    return stub


class Rig:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.run_id = str(uuid4())
        self.made = []

    def profiler(
        self,
        role,
        kv_role,
        *,
        cls=NixlProfiler,
        engine="engine-a",
        window=1.0,
        **context,
    ):
        context = {
            "role": role,
            "kv_role": kv_role,
            "engine_id": engine,
            "backend": ["Test"],
            "transport": "tcp",
            "rank": 0,
            "dp_rank": 0,
            **context,
        }
        if cls is MooncakeProfiler:
            context["connector"] = "mooncake"
        config = NetworkProfilerConfig(
            level="l1",
            run_id=self.run_id,
            output_dir=str(self.tmp_path),
            ring_records=1024,
            id_slots=256,
            id_bytes=1 << 16,
            handle_slots=16,
            shutdown_timeout_s=2,
            step_window_s=window,
            context=context,
        )
        recorder = ProcessRecorder(config)
        if cls is FrontendProfiler:
            profiler = FrontendProfiler(config, recorder)
        elif cls is MooncakeProfiler:
            profiler = MooncakeProfiler(
                config,
                recorder,
                rank=context["rank"],
                engine_id=engine,
                kv_role=kv_role,
            )
        else:
            profiler = NixlProfiler(
                config, recorder, rank=context["rank"], engine_id=engine
            )
        assert recorder.wait_ready()
        self.made.append((profiler, recorder))
        return profiler

    def scheduler(self, kv_role="kv_producer", tp=1, pp=1, pcp=1, **kw):
        return self.profiler(
            "scheduler",
            kv_role,
            parallel={"tp": tp, "pp": pp, "pcp": pcp, "workers": tp * pp * pcp},
            **kw,
        )

    def worker(self, kv_role="kv_producer", tp_rank=0, pp_rank=0, pcp_rank=0, **kw):
        return self.profiler(
            "worker",
            kv_role,
            rank=tp_rank,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            pcp_rank=pcp_rank,
            **kw,
        )

    def frontend(self, kv_role="kv_producer"):
        return self.profiler("frontend", kv_role, cls=FrontendProfiler)

    def close(self):
        for profiler, _ in self.made:
            profiler.close()
        return self

    def rows(self, profiler):
        path = profiler.output_dir / "events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def events(self, profiler, kind=None, subject=None):
        return [
            row
            for row in self.rows(profiler)
            if row.get("type") == "event"
            and (kind is None or row["kind"] == kind)
            and (subject is None or row["subject"] == subject)
        ]

    def retime(self, profiler, times, kinds=("request_phase",)):
        """Give the rows of these kinds, in recorded order, these timestamps."""
        path = profiler.output_dir / "events.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        times = iter(times)
        for row in rows:
            if row.get("type") == "event" and row["kind"] in kinds:
                row["time_ns"] = next(times)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def summary(self):
        self.close()
        return summarize([self.tmp_path])


@pytest.fixture
def rig(tmp_path):
    rig = Rig(tmp_path)
    yield rig
    for profiler, recorder in rig.made:
        profiler.close()
        if recorder.ring_dir.exists():
            shutil.rmtree(recorder.ring_dir)


def new(req_id, computed, prefill, prompt=None):
    return SimpleNamespace(
        req_id=req_id,
        num_computed_tokens=computed,
        prefill_token_ids=list(range(prefill)),
        prompt_token_ids=list(range(prefill if prompt is None else prompt)),
    )


def out(new_reqs=(), counts=None, preempted=None, total=None):
    counts = counts if counts is not None else {r.req_id: 1 for r in new_reqs}
    return SimpleNamespace(
        scheduled_new_reqs=list(new_reqs),
        num_scheduled_tokens=counts,
        total_num_scheduled_tokens=sum(counts.values()) if total is None else total,
        preempted_req_ids=preempted,
    )


STOP = SimpleNamespace(name="STOP")


def rs(request_id, tokens=None):
    """A frontend request state: its id and the detokenizer's output count."""
    detokenizer = (
        None if tokens is None else SimpleNamespace(num_output_tokens=lambda: tokens)
    )
    return SimpleNamespace(request_id=request_id, detokenizer=detokenizer)


def req(request_id, prompt_tokens):
    return SimpleNamespace(request_id=request_id, num_prompt_tokens=prompt_tokens)


def request(request_id, status="FINISHED_STOPPED", output=4):
    return SimpleNamespace(
        request_id=request_id,
        status=SimpleNamespace(name=status),
        num_output_tokens=output,
    )


def phases(rig, profiler, request_id=None):
    return [
        (
            next(n for n, v in PHASES.items() if v == row["subject"]),
            {
                k: row[k]
                for k in (
                    "step",
                    "computed_tokens",
                    "scheduled_tokens",
                    "prefill_tokens",
                    "output_tokens",
                    "external_tokens",
                    "prompt_tokens",
                )
                if row.get(k) is not None
            },
            row["status"],
        )
        for row in rig.events(profiler, "request_phase")
    ]


# ------------------------------------------------------------------- the schema


def test_native_kinds_29_and_30_are_accepted_and_the_others_refused(rig):
    profiler = rig.scheduler()
    record = profiler.recorder.record
    values = (None,) * 7
    record(EventKind.REQUEST_PHASE, 0, "r", None, None, None, *values, 1, 0, 1)
    record(EventKind.STEP_SPAN, 0, None, None, None, None, *values, 1, 0, 1)
    for kind in (20, 22, 24, 25, 28, 31):
        with pytest.raises(ValueError):
            record(kind, 0, "r", None, None, None, *values, 1, 0, 0)
    profiler.close()
    rows = rig.events(profiler)
    assert [r["kind"] for r in rows] == ["request_phase", "step_span"]


def test_value_words_are_renamed_per_kind_and_subject():
    import struct

    def raw(kind, subject, values):
        words = [0] * 16
        words[0], words[1] = 5, 1
        flags = 0
        for i, value in enumerate(values):
            if value is not None:
                words[9 + i] = value
                flags |= 1 << i
        words[2] = (2 << 48) | (kind << 32) | (1 << 24) | flags
        words[8] = subject
        return struct.pack("=16Q", *words)

    row = decode_record(raw(29, 3, [7, 1, 2, 3, 4, 5, 6]))
    assert (
        row["step"],
        row["computed_tokens"],
        row["scheduled_tokens"],
        row["prefill_tokens"],
        row["output_tokens"],
        row["external_tokens"],
        row["prompt_tokens"],
    ) == (7, 1, 2, 3, 4, 5, 6)
    assert "bytes" not in row and "aux0" not in row
    row = decode_record(
        raw(30, int(StepSubject.STEP), [9, 8, 100, 200, None, None, None])
    )
    assert (row["step"], row["scope"], row["begin_ns"], row["end_ns"]) == (
        9,
        8,
        100,
        200,
    )
    row = decode_record(raw(30, int(StepSubject.WORKER_WINDOW), [1, 2, 3, 4, 5, 6, 7]))
    assert (
        row["steps"],
        row["marked_steps"],
        row["window_begin_ns"],
        row["window_end_ns"],
        row["busy_ns"],
        row["max_ns"],
        row["empty_steps"],
    ) == (1, 2, 3, 4, 5, 6, 7)
    row = decode_record(
        raw(30, int(StepSubject.SCHEDULER_WINDOW), [1, 2, 3, 4, 5, 6, 7])
    )
    assert (
        row["steps"],
        row["scheduled_tokens"],
        row["max_requests"],
        row["empty_steps"],
        row["marked_steps"],
    ) == (1, 2, 5, 6, 7)


# ---------------------------------------------------------------- the scheduler


def test_a_single_chunk_prefill_marks_its_step_and_decode_steps_record_nothing(
    rig, clock
):
    p = rig.scheduler()
    p.request_queued(req("r1", 8))
    meta = SimpleNamespace()
    p.schedule_step(out([new("r1", 0, 8)], {"r1": 8}), meta)
    assert meta.profiler_step == 1 and meta.profiler_flags == MARK_PHASE
    assert meta.profiler_scope == compute.scope_of(p.recorder.metadata["process_id"])
    for _ in range(5):  # pure decode steps: the request is running, not new
        meta = SimpleNamespace()
        p.schedule_step(out(counts={"r1": 1}), meta)
        assert meta.profiler_flags == 0
    p.request_done(request("r1"))
    p.close()
    seen = phases(rig, p)
    assert [name for name, _, _ in seen] == [
        "queued",
        "admitted",
        "prefill_done",
        "finished",
    ]
    assert seen[1][1] == {
        "step": 1,
        "computed_tokens": 0,
        "scheduled_tokens": 8,
        "prefill_tokens": 8,
        "prompt_tokens": 8,
    }
    assert seen[3][1] == {"output_tokens": 4}


def test_a_chunked_prefill_records_each_chunk_and_the_last_one_ends_the_prefill(
    rig, clock
):
    p = rig.scheduler()
    p.request_queued(req("r1", 10))
    for step, (n, new_req) in enumerate([(4, True), (4, False), (2, False)], 1):
        meta = SimpleNamespace()
        p.schedule_step(out([new("r1", 0, 10)] if new_req else (), {"r1": n}), meta)
        assert meta.profiler_flags & MARK_PHASE and meta.profiler_step == step
    p.close()
    seen = phases(rig, p)
    assert [(n, d.get("step"), d.get("computed_tokens")) for n, d, _ in seen[1:]] == [
        ("admitted", 1, 0),
        ("prefill_chunk", 2, 4),
        ("prefill_done", 3, 8),
    ]


def test_a_remote_kv_resume_is_admitted_after_the_load_and_ends_its_prefill_at_once(
    rig, clock
):
    p = rig.scheduler("kv_consumer")
    p.request_queued(req("d1", 16))
    p.load_prepared("d1", 15)
    p.load_prepared("d1", 15)  # a second call in the same cycle is not another event
    p.load_prepared("d2", 0)  # nothing external: nothing prepared
    p.schedule_step(out([new("d1", 15, 16)], {"d1": 1}), SimpleNamespace())
    p.close()
    names = [n for n, _, _ in phases(rig, p)]
    assert names == ["queued", "load_prepared", "admitted", "prefill_done"]
    assert phases(rig, p)[1][1] == {"external_tokens": 15}


def test_a_resumed_request_needing_several_chunks_uses_the_effective_prefill_length(
    rig, clock
):
    p = rig.scheduler()
    p.request_queued(req("r1", 6))
    p.schedule_step(
        out([new("r1", 0, 6)], {"r1": 6}), SimpleNamespace()
    )  # first token emitted
    p.schedule_step(out(counts={"r1": 1}), SimpleNamespace())
    meta = SimpleNamespace()
    p.schedule_step(out(counts={"r2": 1}, preempted={"r1"}), meta)
    assert meta.profiler_flags & MARK_PHASE
    # resumed: the 6 prompt tokens plus 2 generated ones, 4 already cached
    for n, fresh in ((1, True), (1, False), (2, False)):
        p.schedule_step(
            out([new("r1", 4, 8)] if fresh else (), {"r1": n}), SimpleNamespace()
        )
    p.close()
    seen = phases(rig, p)
    assert [(n, d.get("step"), st) for n, d, st in seen] == [
        ("queued", None, 0),
        ("admitted", 1, 0),
        ("prefill_done", 1, 0),
        ("preempted", 3, 0),
        ("admitted", 4, 1),
        ("prefill_chunk", 5, 0),
        ("prefill_done", 6, 0),
    ]
    assert seen[4][1]["prefill_tokens"] == 8 and seen[4][1]["computed_tokens"] == 4
    assert seen[-1][1]["computed_tokens"] == 6 and seen[-1][1]["scheduled_tokens"] == 2


def test_zero_token_polls_are_flagged_and_excluded_from_the_rollups(rig, clock):
    p = rig.scheduler(window=1.0)
    for value, tokens in ((0, 0), (10, 5), (20, 0), (2_000_000_000, 7)):
        clock.value = value
        meta = SimpleNamespace()
        p.schedule_step(
            out(counts={"r1": tokens} if tokens else {}, total=tokens), meta
        )
        assert bool(meta.profiler_flags & MARK_EMPTY) == (tokens == 0)
    p.close()
    windows = rig.events(p, "step_span", int(StepSubject.SCHEDULER_WINDOW))
    assert len(windows) == 1  # closed by the 4th step, nothing left to flush at close
    window = windows[0]
    assert (
        window["steps"],
        window["empty_steps"],
        window["scheduled_tokens"],
        window["max_requests"],
    ) == (2, 2, 12, 1)
    assert (window["window_begin_ns"], window["window_end_ns"]) == (0, 2_000_000_000)


def test_the_tracker_is_bounded_and_counts_what_it_could_not_follow(
    rig, clock, monkeypatch
):
    p = rig.scheduler()
    p._tl_cap = 2
    for i in range(4):
        p.request_queued(req(f"r{i}", 3))
    p.close()
    assert p.tracked_overflow == 2
    assert len([n for n, _, _ in phases(rig, p) if n == "queued"]) == 4
    doc = json.loads((p.output_dir / "profiler.json").read_text())
    assert doc["timeline_tracked_overflow"] == 2


# ------------------------------------------------------------------- the worker


def test_a_marked_step_records_the_hooks_own_readings_and_an_unmarked_one_only_rolls_up(
    rig, clock
):
    s = rig.scheduler()
    w = rig.worker()
    meta = SimpleNamespace()
    s.schedule_step(out([new("r1", 0, 4)], {"r1": 4}), meta)
    clock.value = 5_000
    w.step_begin(meta)
    clock.value = 5_700
    w.step_end()
    w.step_end()  # a second finish_forward of the same step records nothing
    meta2 = SimpleNamespace()
    s.schedule_step(out(counts={"r1": 1}), meta2)
    clock.value = 6_000
    w.step_begin(meta2)
    clock.value = 6_100
    w.step_end()
    w.step_begin(SimpleNamespace())  # metadata from an unprofiled scheduler
    w.step_end()
    s.close()
    w.close()
    steps = rig.events(w, "step_span", int(StepSubject.STEP))
    assert [(r["step"], r["begin_ns"], r["end_ns"]) for r in steps] == [
        (1, 5_000, 5_700)
    ]
    assert steps[0]["scope"] == meta.profiler_scope
    window = rig.events(w, "step_span", int(StepSubject.WORKER_WINDOW))[0]
    assert (
        window["steps"],
        window["marked_steps"],
        window["busy_ns"],
        window["max_ns"],
    ) == (2, 1, 800, 700)
    assert (window["window_begin_ns"], window["window_end_ns"]) == (5_000, 6_100)
    assert w.steps_without_marker == 1


def test_a_zero_window_disables_the_rollups_and_decode_steps_record_nothing(rig, clock):
    s = rig.scheduler(window=0)
    w = rig.worker(window=0)
    for _ in range(50):
        meta = SimpleNamespace()
        s.schedule_step(out(counts={"r1": 1}), meta)
        clock.value += 10
        w.step_begin(meta)
        clock.value += 10
        w.step_end()
    s.close()
    w.close()
    assert rig.events(w, "step_span") == [] and rig.events(s, "step_span") == []


def test_a_hook_error_disables_the_profiler_and_is_counted(rig, clock):
    p = rig.scheduler()
    p.schedule_step(SimpleNamespace(), SimpleNamespace())  # no scheduler output fields
    p.request_queued(req("r1", 3))
    p.close()
    assert p.hook_errors == 1
    assert [n for n, _, _ in phases(rig, p)] == []
    assert json.loads((p.output_dir / "profiler.json").read_text())["hook_errors"] == 1


# ------------------------------------------------------------------ the frontend


def test_frontend_phases_are_recorded_with_their_status_and_counts(rig):
    f = rig.frontend()
    f.first_token_output(rs("r1"), [0] * 3)
    f.output_finished(rs("r1", 9), STOP)
    f.output_aborted(rs("r2"))
    f.output_failed(rs("r3"))
    f.output_finished(rs("r4", 1), SimpleNamespace(name="ERROR"))
    f.close()
    seen = phases(rig, f)
    assert seen == [
        ("first_token_output", {"output_tokens": 3}, 0),
        ("output_finished", {"output_tokens": 9}, 0),
        ("output_finished", {}, 1),
        ("output_finished", {}, 2),
        ("output_finished", {"output_tokens": 1}, 2),
    ]


# ---------------------------------------------------------------- the timelines


def timeline(summary, request, source_role=None, occurrence=0, source=None):
    found = [
        t
        for t in summary["timelines"]
        if t["request"] == request
        and t["request_occurrence"] == occurrence
        and (source_role is None or t["node_role"] == source_role)
        and (source is None or t["process_id"] == source)
    ]
    assert len(found) == 1, found
    return found[0]


def problem_kinds(summary):
    return [p["kind"] for p in summary["problems"]]


def consumer(
    rig,
    clock,
    *,
    tp=1,
    worker_ranks=(0,),
    parallel=True,
    load=True,
    failed=False,
    first_token=True,
    arrival=True,
    step=(1600, 1900),
    computed=None,
):
    """One consumer request, the times scripted: api 900, queued 1000, load prepared
    1100, received 1400, admitted (and prefill done) 1500, step 1600-1900, first token
    2000."""
    sd = (
        rig.scheduler("kv_consumer", tp=tp, engine="engine-d")
        if parallel
        else rig.profiler("scheduler", "kv_consumer", engine="engine-d")
    )
    fd = rig.frontend("kv_consumer")
    workers = [
        rig.worker("kv_consumer", tp_rank=r, engine="engine-d") for r in worker_ranks
    ]
    sd.request_queued(req("d1", 16))
    if load:
        sd.load_expected("d1", 0)  # nothing cached locally: 15 external tokens
        sd.load_prepared("d1", 15)
        sd.recv_finished("d1", failed)
    meta = SimpleNamespace()
    sd.schedule_step(
        out([new("d1", 15 if computed is None else computed, 16)], {"d1": 1}), meta
    )
    for w in workers:
        clock.value = step[0]
        w.step_begin(meta)
        clock.value = step[1]
        w.step_end()
    sd.request_done(request("d1", output=5))
    if arrival:
        fd.api_request("b1", "h1", "p1", False)
        fd.external_engine("p1", "d1")
    if first_token:
        fd.first_token_output(rs("d1"), [0] * 1)
    fd.output_finished(rs("d1", 5), STOP)
    rig.close()
    sched_kinds = ("request_phase", "complete_observed")
    times = [1000] + ([1100, 1400] if load else []) + [1500, 1500, 9000]
    rig.retime(sd, times, sched_kinds)
    front = [900, 900, 950] if arrival else []
    rig.retime(
        fd,
        front + ([2000] if first_token else []) + [9100],
        ("request_bind", "request_phase"),
    )
    return sd, fd, workers


def test_a_consumer_timeline_has_exact_buckets_and_names_what_it_cannot_split(
    rig, clock
):
    consumer(rig, clock)
    summary = summarize([rig.tmp_path])
    json.dumps(summary)  # the document is plain JSON (the CLI writes it)
    t = timeline(summary, "d1")
    assert t["state"] == "complete" and t["node_role"] == "consumer"
    assert t["api"]["arrival_ns"] == 900 and t["api"]["external_request"] == "p1"
    assert (
        t["scheduler"]["queued_ns"] == 1000 and t["scheduler"]["kv_received_ns"] == 1400
    )
    assert t["tokens"]["prefill_effective"] == 16
    assert t["tokens"]["computed_at_admission"] == 15
    assert t["tokens"]["tokens_remaining_at_admission"] == 1
    iv = t["intervals_ns"]
    assert iv["api_to_queued"] == 100 and iv["queued_to_load_prepared"] == 100
    assert iv["load_prepared_to_received"] == 300 and iv["received_to_admitted"] == 100
    assert iv["prefill_window"] == 300 and iv["first_step_host_span"] == 300
    assert (
        iv["after_step_to_first_token_output"] == 100
        and iv["scheduler_to_step_begin"] == 100
    )
    stall = t["stall"]
    assert stall["basis"] == "api_arrival" and stall["end"] == "first_token_output"
    assert (
        stall["pre_engine_ns"],
        stall["queue_ns"],
        stall["waiting_for_kv_ns"],
        stall["step_host_span_ns"],
        stall["between_steps_ns"],
        stall["after_step_to_first_token_ns"],
        stall["other_ns"],
    ) == (100, 200, 300, 300, 0, 100, 100)
    assert stall["complete"] is True
    assert t["kv"]["outcome"] == "received"
    (step,) = t["steps"]
    assert step["phases"] == ["admitted", "prefill_done"] and step["complete"]
    assert step["host_span_ns"] == 300 and step["expected_ranks"] == 1
    assert t["decode_window"]["tokens_after_first_output"] == 4
    assert t["decode_window"]["mean_time_per_token_after_first_ns"] == (9100 - 2000) / 4
    assert not [k for k in problem_kinds(summary) if k != "adapter_unhealthy"]


def test_a_missing_frontend_arrival_names_the_scheduler_queue_as_the_basis(rig, clock):
    consumer(rig, clock, arrival=False)
    t = timeline(summarize([rig.tmp_path]), "d1")
    assert t["stall"]["basis"] == "scheduler_queued"
    assert "api_arrival" in t["unavailable"]
    assert t["stall"]["pre_engine_ns"] == 0 and t["stall"]["complete"]


def test_a_request_with_no_first_token_has_no_stall_and_says_why(rig, clock):
    consumer(rig, clock, first_token=False)
    t = timeline(summarize([rig.tmp_path]), "d1")
    assert t["stall"]["complete"] is False and "first_token_output" in t["unavailable"]
    assert t["decode_window"]["tokens_after_first_output"] is None
    assert "mean_time_per_token_after_first_ns" in t["unavailable"]


def test_a_missing_tp_rank_withholds_the_full_step_and_everything_built_on_it(
    rig, clock
):
    consumer(rig, clock, tp=2, worker_ranks=(0,))
    summary = summarize([rig.tmp_path])
    t = timeline(summary, "d1")
    (step,) = t["steps"]
    assert (
        step["complete"] is False
        and step["present_ranks"] == 1
        and step["expected_ranks"] == 2
    )
    assert step["host_span_ns"] is None and step["begin_skew_ns"] is None
    assert step["rank_spans_ns"] == {"0:0:0": 300}
    assert any("worker 1:0:0: no capture" in r for r in step["reasons"])
    assert t["stall"]["complete"] is False and t["stall"]["step_host_span_ns"] is None
    assert t["intervals_ns"]["prefill_window"] is None
    assert "rank_missing" in problem_kinds(summary)


def test_both_tp_ranks_give_the_shared_span_and_their_skew(rig, clock):
    sd = rig.scheduler("kv_producer", tp=2)
    w0, w1 = rig.worker(tp_rank=0), rig.worker(tp_rank=1)
    sd.request_queued(req("p1", 4))
    meta = SimpleNamespace()
    sd.schedule_step(out([new("p1", 0, 4)], {"p1": 4}), meta)
    for w, (b, e) in ((w0, (100, 400)), (w1, (130, 450))):
        clock.value = b
        w.step_begin(meta)
        clock.value = e
        w.step_end()
    sd.request_done(request("p1", output=1))
    t = timeline(rig.summary(), "p1")
    (step,) = t["steps"]
    assert step["complete"] and step["host_span_ns"] == 350
    assert (step["begin_skew_ns"], step["end_skew_ns"]) == (30, 50)
    assert step["rank_spans_ns"] == {"0:0:0": 300, "1:0:0": 320}


def test_a_capture_without_the_parallel_layout_cannot_say_which_ranks_are_missing(
    rig, clock
):
    consumer(rig, clock, parallel=False)
    t = timeline(summarize([rig.tmp_path]), "d1")
    (step,) = t["steps"]
    assert step["complete"] is False and step["expected_ranks"] is None
    assert step["rank_spans_ns"] == {"0:0:0": 300}
    assert any("expected worker set is unknown" in r for r in step["reasons"])


def test_a_restarted_scheduler_reusing_step_numbers_never_joins_the_old_generation(
    rig, clock
):
    results = []
    for index, name in enumerate(("old", "new")):
        sd = rig.scheduler(engine="engine-r")
        w = rig.worker(engine="engine-r")
        sd.request_queued(req(f"r-{name}", 4))
        meta = SimpleNamespace()
        sd.schedule_step(
            out([new(f"r-{name}", 0, 4)], {f"r-{name}": 4}), meta
        )  # step 1 again
        (clock.value,) = (1000 * (index + 1),)
        w.step_begin(meta)
        clock.value += 100 * (index + 1)
        w.step_end()
        sd.request_done(request(f"r-{name}"))
        results.append(meta.profiler_scope)
    assert results[0] != results[1]
    summary = rig.summary()
    old, new_ = timeline(summary, "r-old"), timeline(summary, "r-new")
    assert old["steps"][0]["rank_spans_ns"] == {"0:0:0": 100}
    assert new_["steps"][0]["rank_spans_ns"] == {"0:0:0": 200}
    assert old["steps"][0]["step"] == new_["steps"][0]["step"] == 1
    assert not [k for k in problem_kinds(summary) if k != "adapter_unhealthy"]


def test_a_step_of_an_unknown_scheduler_generation_or_engine_is_a_problem_not_a_join(
    rig, clock
):
    sd = rig.scheduler(engine="engine-a")
    w = rig.worker(engine="engine-a")
    other = rig.worker(engine="engine-z")
    sd.request_queued(req("r1", 4))
    meta = SimpleNamespace()
    sd.schedule_step(out([new("r1", 0, 4)], {"r1": 4}), meta)
    clock.value = 10
    w.step_begin(meta)
    clock.value = 20
    w.step_end()
    other.step_begin(meta)  # the same marker seen by a worker of another engine
    clock.value = 30
    other.step_end()
    stray = SimpleNamespace(
        profiler_step=5, profiler_scope=123, profiler_flags=MARK_PHASE
    )
    clock.value = 40
    w.step_begin(stray)
    clock.value = 50
    w.step_end()
    summary = rig.summary()
    kinds = problem_kinds(summary)
    assert "step_engine_mismatch" in kinds and "step_scope_unknown" in kinds
    assert timeline(summary, "r1")["steps"][0]["rank_spans_ns"] == {"0:0:0": 10}


def test_a_worker_row_without_a_scheduler_phase_row_is_reported(rig, clock):
    sd = rig.scheduler()
    w = rig.worker()
    sd.request_queued(req("r1", 4))
    meta = SimpleNamespace()
    sd.schedule_step(out([new("r1", 0, 4)], {"r1": 4}), meta)
    clock.value = 1
    w.step_begin(meta)
    clock.value = 2
    w.step_end()
    sd.close()
    rig.retime(sd, [], ("nothing",))
    path = sd.output_dir / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    path.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in rows
            if not (r.get("kind") == "request_phase" and r["subject"] in (3, 5))
        )
    )
    assert "step_without_marker_row" in problem_kinds(rig.summary())


def test_a_preemption_after_the_first_token_stays_visible_outside_the_first_token_stall(
    rig, clock
):
    sd = rig.scheduler()
    w = rig.worker()
    fd = rig.frontend()
    sd.request_queued(req("r1", 4))
    meta = SimpleNamespace()
    sd.schedule_step(out([new("r1", 0, 4)], {"r1": 4}), meta)
    clock.value = 100
    w.step_begin(meta)
    clock.value = 150
    w.step_end()
    sd.schedule_step(out(counts={"r2": 1}, preempted={"r1"}), SimpleNamespace())
    meta3 = SimpleNamespace()
    sd.schedule_step(out([new("r1", 2, 5)], {"r1": 3}), meta3)  # prompt 4 + 1 generated
    clock.value = 400
    w.step_begin(meta3)
    clock.value = 480
    w.step_end()
    sd.request_done(request("r1", output=2))
    fd.first_token_output(rs("r1"), [0] * 1)
    fd.output_finished(rs("r1", 2), STOP)
    rig.close()
    rig.retime(sd, [10, 20, 20, 300, 350, 350, 600])
    rig.retime(fd, [200, 700], ("request_phase",))
    t = timeline(summarize([rig.tmp_path]), "r1")
    assert [a["resumed"] for a in t["scheduler"]["admissions"]] == [False, True]
    assert [a["resumed"] for a in t["scheduler"]["later_admissions"]] == [True]
    assert t["scheduler"]["preemptions"] == 1
    assert [s["step"] for s in t["steps"]] == [1]
    assert t["intervals_ns"]["queued_to_admitted"] == 10
    assert t["stall"]["end"] == "first_token_output" and t["stall"]["complete"]
    assert t["stall"]["preempted_ns"] == 0 and t["stall"]["step_host_span_ns"] == 50
    assert (
        t["stall"]["queue_ns"] == 10
        and t["stall"]["after_step_to_first_token_ns"] == 50
    )


def test_a_failed_load_is_recomputed_and_no_tokens_are_blamed_on_the_failure(
    rig, clock
):
    consumer(rig, clock, failed=True, computed=4)
    t = timeline(summarize([rig.tmp_path]), "d1")
    assert t["kv"]["outcome"] == "load_failed_recomputed"
    assert t["kv"]["receive_reported_failed"] is True
    assert t["kv"]["tokens_attributed_to_failure"] is None
    assert t["tokens"]["tokens_remaining_at_admission"] == 12
    assert "tokens_attributed_to_failure" in t["unavailable"]


def test_a_local_hit_and_a_cold_request_have_no_load(rig, clock):
    for computed, outcome in ((8, "local_hit"), (0, "none")):
        rig.made.clear()
        shutil.rmtree(rig.tmp_path)
        rig.tmp_path.mkdir()
        consumer(rig, clock, load=False, computed=computed)
        assert timeline(summarize([rig.tmp_path]), "d1")["kv"]["outcome"] == outcome


def test_an_unhealthy_adapter_makes_the_timeline_observed_only(rig, clock):
    sd, *_ = consumer(rig, clock)
    path = sd.output_dir / "profiler.json"
    doc = json.loads(path.read_text())
    doc["hook_errors"] = 1
    path.write_text(json.dumps(doc))
    t = timeline(summarize([rig.tmp_path]), "d1")
    assert (
        t["state"] == "observed_only"
        and t["intervals_ns"] is None
        and t["stall"] is None
    )
    assert t["intervals_ns_observed"]["queued_to_admitted"] == 500
    assert "totals" in t["unavailable"]


def test_decode_cadence_comes_from_windows_entirely_inside_the_decode_window(
    rig, clock
):
    sd = rig.scheduler(window=1.0)
    w = rig.worker(window=1.0)
    fd = rig.frontend()
    sd.request_queued(req("r1", 4))
    meta = SimpleNamespace()
    sd.schedule_step(out([new("r1", 0, 4)], {"r1": 4}), meta)
    clock.value = 1_000
    w.step_begin(meta)
    clock.value = 1_100
    w.step_end()
    base = 10_000_000
    for i in range(5):
        meta = SimpleNamespace()
        sd.schedule_step(out(counts={"r1": 1}), meta)
        clock.value = base + i * 1_000_000_000
        w.step_begin(meta)
        clock.value += 200
        w.step_end()
    sd.request_done(request("r1", output=6))
    fd.first_token_output(rs("r1"), [0] * 1)
    fd.output_finished(rs("r1", 6), STOP)
    rig.close()
    rig.retime(sd, [10, 20, 20, 9_000_000_000], ("request_phase",))
    rig.retime(fd, [2_000, 8_000_000_000], ("request_phase",))
    summary = summarize([rig.tmp_path])
    t = timeline(summary, "r1")
    cadence = t["decode_window"]["node_cadence"]
    assert cadence is not None and cadence["0:0:0"]["windows"] >= 1
    assert (
        cadence["0:0:0"]["mean_step_ns"] == 200
        and cadence["0:0:0"]["max_step_ns"] == 200
    )
    assert cadence["0:0:0"]["empty_steps"] == 0


def test_a_chunked_producer_timeline_separates_step_spans_from_the_gaps_between(
    rig, clock
):
    sd = rig.scheduler("kv_producer")
    w = rig.worker("kv_producer")
    fd = rig.frontend("kv_producer")
    sd.request_queued(req("p1", 8))
    m1, m2 = SimpleNamespace(), SimpleNamespace()
    sd.schedule_step(out([new("p1", 0, 8)], {"p1": 4}), m1)
    clock.value = 210
    w.step_begin(m1)
    clock.value = 250
    w.step_end()
    sd.schedule_step(out(counts={"p1": 4}), m2)
    clock.value = 310
    w.step_begin(m2)
    clock.value = 370
    w.step_end()
    sd.request_done(request("p1", output=1))
    fd.first_token_output(rs("p1"), [0] * 1)
    fd.output_finished(rs("p1", 1), STOP)
    rig.close()
    rig.retime(sd, [100, 200, 300, 400])
    rig.retime(fd, [380, 390])
    t = timeline(summarize([rig.tmp_path]), "p1")
    assert [(s["step"], s["phases"], s["host_span_ns"]) for s in t["steps"]] == [
        (1, ["admitted"], 40),
        (2, ["prefill_done"], 60),
    ]
    iv = t["intervals_ns"]
    assert iv["prefill_window"] == 160 and iv["prefill_steps_host_span_sum"] == 100
    stall = t["stall"]
    assert stall["basis"] == "scheduler_queued"
    assert (
        stall["queue_ns"],
        stall["step_host_span_ns"],
        stall["between_steps_ns"],
        stall["after_step_to_first_token_ns"],
        stall["other_ns"],
    ) == (100, 100, 60, 10, 10)


def test_a_simulator_ordered_step_ends_at_finish_forward_and_the_wait_is_not_compute(
    rig, clock
):
    """The simulator binds, waits out its host time, calls finish_forward, and only then
    waits for the step's device deadline: the span stops at finish_forward and the wait
    lands in the time up to the frontend's first output."""
    sd = rig.scheduler()
    w = rig.worker()
    fd = rig.frontend()
    sd.request_queued(req("r1", 4))
    meta = SimpleNamespace()
    sd.schedule_step(out([new("r1", 0, 4)], {"r1": 4}), meta)
    clock.value = 1_000
    w.step_begin(meta)  # execute_model
    clock.value = 1_500  # sample_tokens: the host-time wait
    w.step_end()  # finish_forward
    clock.value = 9_000  # get_output(): the deadline wait; no hook runs
    sd.request_done(request("r1", output=1))
    fd.first_token_output(rs("r1"), [0] * 1)
    fd.output_finished(rs("r1", 1), STOP)
    rig.close()
    rig.retime(sd, [10, 900, 900, 9_500])
    rig.retime(fd, [9_000, 9_100])
    t = timeline(summarize([rig.tmp_path]), "r1")
    assert t["steps"][0]["host_span_ns"] == 500
    assert t["stall"]["step_host_span_ns"] == 500
    assert t["stall"]["after_step_to_first_token_ns"] == 7_500
    assert t["stall"]["complete"]


def test_a_summary_without_the_new_kinds_has_empty_timelines(rig):
    sd = rig.scheduler()
    sd.close()
    summary = summarize([rig.tmp_path])
    assert summary["timelines"] == [] and summary["kv_writes"] == []
    assert "unclassified_events" not in problem_kinds(summary)


# ----------------------------------------------------------------- the producer join


def mooncake_pair(
    rig,
    *,
    consumer_queued=1000,
    prepared=1100,
    received=6000,
    handoff=5000,
    domains=False,
):
    """A producer and a consumer scheduler plus the producer's worker with one write
    batch of two members, with distinct request and transfer IDs."""
    ps = rig.profiler(
        "scheduler", "kv_producer", cls=MooncakeProfiler, engine="engine-p"
    )
    cs = rig.profiler(
        "scheduler",
        "kv_consumer",
        cls=MooncakeProfiler,
        engine="engine-d",
        parallel={"tp": 1, "pp": 1, "pcp": 1, "workers": 1},
    )
    pw = rig.profiler(
        "worker",
        "kv_producer",
        cls=MooncakeProfiler,
        engine="engine-p",
        tp_rank=0,
        pp_rank=0,
        pcp_rank=0,
    )
    for name, tid in (("1", "xfer-1"), ("2", "xfer-2")):
        ps.request_queued(req(f"p-req{name}", 8))
        ps.handoff_out(f"p-req{name}", tid, [[1, 2]])
        cs.request_queued(req(f"d-req{name}", 8))
        cs.handoff_in(f"d-req{name}", tid, "engine-p")
    cs.load_prepared("d-req1", 8)
    cs.recv_finished("d-req1", False)
    handle = pw.batch_bind("consumer:1", 0, 2)
    pw.batch_prepared(handle)
    pw.write_begin(handle, 300, 3)
    pw.write_returned(handle)
    pw.write_complete(handle, 300, 3, 50)
    pw.batch_member(handle, "d-req1", "xfer-1", True, (100, 1))
    pw.batch_member(handle, "d-req2", "xfer-2", True, (200, 2))
    pw.batch_retire(handle)
    rig.close()
    rig.retime(ps, [100, 5000, 100, 6000], ("request_phase", "handoff"))
    rig.retime(
        cs,
        [consumer_queued, 0, consumer_queued, 0, prepared, received],
        ("request_phase", "handoff", "complete_observed"),
    )
    if domains:
        path = cs.output_dir / "events.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        for row in rows:
            if row.get("type") == "metadata":
                row["boot_id"] = "another-host"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return ps, cs, pw


def test_mooncake_joins_on_the_transfer_id_when_request_and_transfer_ids_differ(rig):
    mooncake_pair(rig)
    summary = summarize([rig.tmp_path])
    joins = sorted(summary["joins"], key=lambda j: j["consumer_request"])
    assert [
        (
            j["consumer_request"],
            j["producer_request"],
            j["transfer_id"],
            j["join_key"],
            j["joined"],
        )
        for j in joins
    ] == [
        ("d-req1", "p-req1", "xfer-1", "transfer_id", True),
        ("d-req2", "p-req2", "xfer-2", "transfer_id", True),
    ]
    assert all(j["producer_engine"] == "engine-p" and j["matches"] == 1 for j in joins)


def test_every_join_places_its_handoffs_in_one_occurrence_by_record_sequence(rig):
    """Each timeline names the scheduler record sequences it owns, and each join
    the record sequences of its two handoffs: the incoming one lies in exactly
    one consumer occurrence (the one the summary joined to the producer), the
    outgoing one in exactly one producer occurrence."""
    mooncake_pair(rig)
    summary = summarize([rig.tmp_path])

    def owners(source, request, sequence):
        return [
            t for t in summary["timelines"]
            if t["source"] == source and t["request"] == request
            and t["sequences"]["first"] <= sequence
            and (t["sequences"]["end"] is None or sequence < t["sequences"]["end"])
        ]  # fmt: skip

    assert summary["joins"]
    for join in summary["joins"]:
        (consumer,) = owners(join["consumer_source"], join["consumer_request"],
                             join["consumer_handoff_sequence"])  # fmt: skip
        assert consumer["cross_node"]["producer_request"] == join["producer_request"]
        assert join["producer_handoff_sequence"] is not None
        (producer,) = owners(join["producer_sources"][0], join["producer_request"],
                             join["producer_handoff_sequence"])  # fmt: skip
        assert producer["node_role"] == "producer"


def test_one_write_batch_with_two_transfer_ids_is_counted_once_and_referenced_by_both(
    rig,
):
    mooncake_pair(rig)
    summary = summarize([rig.tmp_path])
    (write,) = summary["kv_writes"]
    assert (write["bytes"], write["descriptors"], write["partition"]) == (
        300,
        3,
        "exact",
    )
    assert sorted((m["request"], m["transfer_id"]) for m in write["members"]) == [
        ("d-req1", "xfer-1"),
        ("d-req2", "xfer-2"),
    ]
    one = timeline(summary, "d-req1")["kv"]["writes"]
    two = timeline(summary, "d-req2")["kv"]["writes"]
    assert [
        (w["key"], w["transfer_id"], w["share_bytes"], w["share_descriptors"])
        for w in one
    ] == [(write["key"], "xfer-1", 100, 1)]
    assert [
        (w["key"], w["transfer_id"], w["share_bytes"], w["share_descriptors"])
        for w in two
    ] == [(write["key"], "xfer-2", 200, 2)]


def test_a_concurrent_launch_shows_the_consumer_queued_before_the_producer_handoff(rig):
    mooncake_pair(rig)
    t = timeline(summarize([rig.tmp_path]), "d-req1")
    cross = t["cross_node"]
    assert cross["join_key"] == "transfer_id" and cross["producer_request"] == "p-req1"
    assert cross["consumer_queued_minus_producer_handoff_ns"] == 1000 - 5000
    assert (
        cross["ordering"] == "consumer_before_handoff" and cross["uncertainty_ns"] == 0
    )
    assert cross["method"] == "same_clock_domain"
    assert cross["producer_pending_ns"] == 5000 - 1100


def test_across_clock_domains_the_ordering_carries_the_calibration_uncertainty(
    rig, tmp_path
):
    from vllm.kv_profiling.clock import ClockMapping, ClockSample

    ps, cs, _ = mooncake_pair(
        rig, consumer_queued=5300, prepared=5400, received=7000, domains=True
    )

    def clock_of(profiler):
        path = profiler.output_dir / "events.jsonl"
        return json.loads(path.read_text().splitlines()[0])["clock_id"]

    without = timeline(summarize([rig.tmp_path]), "d-req1")
    assert without["cross_node"]["consumer_queued_minus_producer_handoff_ns"] is None
    assert (
        without["cross_node"]["ordering"] is None
        and "cross_node" in without["unavailable"]
    )
    reference = clock_of(ps)
    mapping = ClockMapping.from_samples(
        [
            ClockSample(
                run_id=rig.run_id,
                local_generation=reference,
                remote_generation=clock_of(cs),
                t1=1000,
                t2=1000,
                t3=2000,
                t4=2000,
            )
        ]
    )
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(mapping.to_json()))
    summary = summarize([rig.tmp_path], calibration=[str(path)])
    cross = timeline(summary, "d-req1")["cross_node"]
    assert cross["method"] == "calibrated" and cross["uncertainty_ns"] == 1000
    assert cross["consumer_queued_minus_producer_handoff_ns"] == 300
    assert cross["ordering"] == "indeterminate"


# ---------------------------------------------------------------------- Perfetto


def test_the_exporter_draws_phases_as_instants_and_steps_as_slices(
    rig, clock, tmp_path
):
    from vllm.kv_profiling.clock import ClockGroup
    from vllm.kv_profiling.perfetto import export_traces

    sd = rig.scheduler()
    w = rig.worker()
    sd.request_queued(req("r1", 4))
    meta = SimpleNamespace()
    sd.schedule_step(out([new("r1", 0, 4)], {"r1": 4}), meta)
    clock.value = 5_000
    w.step_begin(meta)
    clock.value = 8_000
    w.step_end()
    sd.request_done(request("r1"))
    rig.close()
    paths = [str(sd.output_dir / "events.jsonl"), str(w.output_dir / "events.jsonl")]
    reference = sd.recorder.metadata["clock_id"]
    documents = export_traces(
        paths, str(tmp_path / "trace-out"), ClockGroup(reference, {})
    )
    document = json.loads(next(iter(documents.values())).read_text())
    events = document["traceEvents"]
    names = [e["name"] for e in events if e.get("ph") == "i"]
    assert "request_phase:queued" in names and "request_phase:prefill_done" in names
    assert "step_span:step" in names
    (slice_,) = [e for e in events if e.get("name") == "step host span (shared batch)"]
    assert slice_["ph"] == "X" and slice_["dur"] == 3.0  # 3000 ns, in microseconds
    counters = [e for e in events if e.get("name") == "step window"]
    assert counters and "steps" in counters[0]["args"]


# ------------------------------------------------------------ connector call sites


def _nixl_classes():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
        NixlBaseConnectorScheduler,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlPullConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
        NixlPullConnectorScheduler,
    )

    return (
        NixlBaseConnectorScheduler,
        NixlPullConnectorScheduler,
        NixlPullConnector,
        NixlConnectorMetadata,
    )


def _mooncake_classes():
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
        mooncake_connector as m,
    )

    return (
        m.MooncakeConnectorScheduler,
        m.MooncakeConnector,
        m.MooncakeConnectorMetadata,
    )


def kv_output(finished=(), failed=()):
    return SimpleNamespace(
        finished_recving=set(finished),
        failed_recving=set(failed),
        finished_sending=None,
    )


def test_the_nixl_scheduler_and_worker_hooks_reach_the_profiler_in_the_runners_order(
    rig, clock
):
    base, pull, connector_cls, metadata_cls = _nixl_classes()
    sp = rig.scheduler("kv_consumer", engine="engine-n")
    wp = rig.worker("kv_consumer", engine="engine-n")
    scheduler = SimpleNamespace(_kvp=sp, _stop_heartbeat=lambda request_id: None)
    request = SimpleNamespace(
        request_id="n1",
        num_prompt_tokens=16,
        kv_transfer_params=None,
        num_output_tokens=3,
        status=SimpleNamespace(name="FINISHED_STOPPED"),
    )
    base.on_new_request(scheduler, request)
    pull.update_state_after_alloc(scheduler, request, None, 15)
    pull.update_state_after_alloc(scheduler, request, None, 0)  # nothing external
    base.update_connector_output(scheduler, kv_output({"n1", "n2"}, {"n2"}))
    meta = metadata_cls()
    assert (meta.profiler_step, meta.profiler_scope, meta.profiler_flags) == (0, 0, 0)
    sp.schedule_step(
        out([new("n1", 15, 16)], {"n1": 1}), meta
    )  # build_connector_meta's tail
    assert meta.profiler_step == 1
    worker_side = object.__new__(connector_cls)
    worker_side.connector_worker = SimpleNamespace(_kvp=wp)
    clock.value = 10
    worker_side.bind_connector_metadata(meta)
    clock.value = 25
    worker_side.finish_forward()
    pull.request_finished(scheduler, request, ([1],))
    rig.close()
    assert [n for n, _, _ in phases(rig, sp)] == [
        "queued",
        "load_prepared",
        "admitted",
        "prefill_done",
        "finished",
    ]
    completions = rig.events(sp, "complete_observed")
    assert sorted(r["status"] for r in completions) == [0, 1]  # n2's receive failed
    (step,) = rig.events(wp, "step_span", int(StepSubject.STEP))
    assert (step["step"], step["begin_ns"], step["end_ns"]) == (1, 10, 25)


def test_the_mooncake_scheduler_and_worker_hooks_reach_the_profiler(rig, clock):
    scheduler_cls, connector_cls, metadata_cls = _mooncake_classes()
    sp = rig.profiler(
        "scheduler",
        "kv_consumer",
        cls=MooncakeProfiler,
        engine="engine-m",
        parallel={"tp": 1, "pp": 1, "pcp": 1, "workers": 1},
    )
    wp = rig.profiler(
        "worker",
        "kv_consumer",
        cls=MooncakeProfiler,
        engine="engine-m",
        tp_rank=0,
        pp_rank=0,
        pcp_rank=0,
    )
    scheduler = SimpleNamespace(
        _kvp=sp,
        _has_mamba=False,
        _bounded_replay=False,
        _kv_lease_duration=None,
        _lease_reqs={},
    )
    request = SimpleNamespace(
        request_id="m1",
        num_prompt_tokens=8,
        kv_transfer_params=None,
        num_output_tokens=2,
        status=SimpleNamespace(name="FINISHED_ABORTED"),
    )
    scheduler_cls.on_new_request(scheduler, request)
    scheduler_cls.update_state_after_alloc(scheduler, request, None, 7)
    scheduler_cls.update_connector_output(scheduler, kv_output({"m1"}, {"m1"}))
    meta = metadata_cls()
    assert (meta.profiler_step, meta.profiler_scope, meta.profiler_flags) == (0, 0, 0)
    sp.schedule_step(out([new("m1", 7, 8)], {"m1": 1}), meta)
    worker_side = object.__new__(connector_cls)
    worker_side.connector_worker = SimpleNamespace(_kvp=wp)
    clock.value = 100
    worker_side.bind_connector_metadata(meta)
    clock.value = 160
    worker_side.finish_forward()
    scheduler_cls.request_finished(scheduler, request, ([1],))
    rig.close()
    seen = phases(rig, sp)
    assert [n for n, _, _ in seen] == [
        "queued",
        "load_prepared",
        "admitted",
        "prefill_done",
        "finished",
    ]
    assert seen[-1][2] == 1  # aborted
    (step,) = rig.events(wp, "step_span", int(StepSubject.STEP))
    assert (step["begin_ns"], step["end_ns"]) == (100, 160)


def test_an_unprofiled_connector_changes_nothing_and_imports_nothing_new(rig):
    base, pull, connector_cls, metadata_cls = _nixl_classes()
    scheduler = SimpleNamespace(_kvp=None, _stop_heartbeat=lambda request_id: None)
    request = SimpleNamespace(
        request_id="u1",
        num_prompt_tokens=4,
        kv_transfer_params=None,
        num_output_tokens=1,
        status=SimpleNamespace(name="FINISHED_STOPPED"),
    )
    base.on_new_request(scheduler, request)
    pull.update_state_after_alloc(scheduler, request, None, 3)
    base.update_connector_output(scheduler, kv_output({"u1"}))
    assert pull.request_finished(scheduler, request, ([1],)) == (False, None)
    worker_side = object.__new__(connector_cls)
    worker_side.connector_worker = SimpleNamespace(_kvp=None)
    meta = metadata_cls()
    worker_side.bind_connector_metadata(meta)
    worker_side.finish_forward()
    assert (meta.profiler_step, meta.profiler_scope, meta.profiler_flags) == (0, 0, 0)


def test_the_capture_advertises_the_timeline_and_the_parallel_layout(rig):
    sp = rig.scheduler(tp=2, pp=2, pcp=1)
    wp = rig.worker(tp_rank=1, pp_rank=1)
    rig.close()
    meta = json.loads((sp.output_dir / "events.jsonl").read_text().splitlines()[0])
    assert meta["context"]["parallel"] == {"tp": 2, "pp": 2, "pcp": 1, "workers": 4}
    wmeta = json.loads((wp.output_dir / "events.jsonl").read_text().splitlines()[0])
    assert (
        wmeta["context"]["tp_rank"],
        wmeta["context"]["pp_rank"],
        wmeta["context"]["pcp_rank"],
    ) == (1, 1, 0)


def test_the_capability_flag_is_advertised_by_both_adapters():
    from vllm.kv_profiling import mooncake, nixl

    assert nixl.capabilities("scheduler")["compute_timeline"] is True
    assert mooncake.capabilities("worker", "kv_consumer")["compute_timeline"] is True


RUN = str(uuid4())


def test_the_step_window_option_is_validated():
    for bad in (-1, float("nan"), 61, "1"):
        with pytest.raises(ValueError, match="step_window_s"):
            NetworkProfilerConfig(
                level="l1", run_id=RUN, output_dir="/x", step_window_s=bad
            )
    assert NetworkProfilerConfig(
        level="l1", run_id=RUN, output_dir="/x", step_window_s=0
    )


# ----------------------------------------- the review of agent-3#62: five corrections


def edit_rows(profiler, change):
    path = profiler.output_dir / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows = change(rows) or rows
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_lossy_worker_capture_keeps_the_observations_and_withholds_the_totals(
    rig, clock
):
    _, _, workers = consumer(rig, clock, tp=2, worker_ranks=(0, 1))

    def incomplete(rows):
        for row in rows:
            if row.get("type") == "end":
                row["complete"] = False

    edit_rows(workers[1], incomplete)
    summary = summarize([rig.tmp_path])
    t = timeline(summary, "d1")
    assert t["state"] == "observed_only"
    assert t["stall"] is None and t["intervals_ns"] is None
    assert t["stall_observed"]["step_host_span_ns"] == 300
    assert [(q["role"], q["source"]) for q in t["quality"] if q["role"] == "worker"]
    (step,) = t["steps"]
    assert step["complete"] is False and step["host_span_ns"] is None
    assert step["host_span_observed_ns"] == 300
    assert set(step["rank_observations"]) == {"0:0:0", "1:0:0"}
    assert step["quality"] and "observed" in " ".join(step["reasons"])


def test_an_unhealthy_frontend_makes_the_stall_observed_only(rig, clock):
    _, fd, _ = consumer(rig, clock)
    path = fd.output_dir / "profiler.json"
    doc = json.loads(path.read_text())
    doc["hook_errors"] = 2
    path.write_text(json.dumps(doc))
    t = timeline(summarize([rig.tmp_path]), "d1")
    assert t["state"] == "observed_only" and t["stall"] is None
    assert [q["role"] for q in t["quality"]] == ["frontend"]
    assert t["stall_observed"]["complete"] is True
    assert t["decode_window"]["quality"][0]["role"] == "frontend"


def test_a_step_row_recorded_twice_by_one_rank_is_never_complete(rig, clock):
    _, _, workers = consumer(rig, clock)

    def duplicate(rows):
        step = next(
            r for r in rows if r.get("kind") == "step_span" and r["subject"] == 1
        )
        copy = dict(
            step,
            sequence=max(r.get("sequence", 0) for r in rows) + 1,
            end_ns=step["end_ns"] + 50,
        )
        end = next(i for i, r in enumerate(rows) if r.get("type") == "end")
        return rows[:end] + [copy] + rows[end:]

    edit_rows(workers[0], duplicate)
    summary = summarize([rig.tmp_path])
    t = timeline(summary, "d1")
    (step,) = t["steps"]
    assert step["complete"] is False and step["host_span_ns"] is None
    assert step["rank_spans_ns"] == {}
    assert [o["span_ns"] for o in step["rank_observations"]["0:0:0"]] == [300, 350]
    assert "duplicate_step_row" in problem_kinds(summary)


def test_a_repeated_request_id_keeps_each_occurrences_own_receive_and_state(rig, clock):
    sd = rig.scheduler("kv_consumer", engine="engine-d")
    fd = rig.frontend("kv_consumer")
    w = rig.worker("kv_consumer", engine="engine-d")
    sd.request_queued(req("d1", 16))
    sd.load_prepared("d1", 15)
    sd.recv_finished("d1", False)
    meta = SimpleNamespace()
    sd.schedule_step(out([new("d1", 15, 16)], {"d1": 1}), meta)
    clock.value = 1600
    w.step_begin(meta)
    clock.value = 1900
    w.step_end()
    sd.request_done(request("d1", output=5))
    # The same ID again: its receive fails and it is aborted before it was admitted.
    sd.request_queued(req("d1", 16))
    sd.load_prepared("d1", 15)
    sd.recv_finished("d1", True)
    sd.request_done(request("d1", status="FINISHED_ABORTED", output=0))
    fd.api_request("b1", "h1", "p1", False)
    fd.external_engine("p1", "d1")
    fd.first_token_output(rs("d1"), [0])
    fd.output_finished(rs("d1", 5), STOP)
    rig.close()
    rig.retime(
        sd,
        [1000, 1100, 1400, 1500, 1500, 9000, 10000, 10100, 10400, 10500],
        ("request_phase", "complete_observed"),
    )
    rig.retime(fd, [900, 900, 950, 2000, 9100], ("request_bind", "request_phase"))
    summary = summarize([rig.tmp_path])
    first, second = (
        timeline(summary, "d1", occurrence=0),
        timeline(summary, "d1", occurrence=1),
    )
    assert first["state"] == "complete" and first["kv"]["outcome"] == "received"
    assert first["api"]["first_token_output_ns"] == 2000 and first["stall"]["complete"]
    assert (
        second["state"] == "aborted"
        and second["scheduler"]["finish_state"] == "aborted"
    )
    assert (
        second["kv"]["outcome"] == "load_failed"
    )  # a failure alone is not a recomputation
    assert second["kv"]["recomputation_shown"] is False
    assert second["kv"]["receive_reported_failed"] is True
    assert second["api"]["first_token_output_ns"] is None and second["steps"] == []
    assert first["kv"]["receive_reported_failed"] is False


def test_a_failed_load_is_recomputed_only_when_an_admission_shows_it(rig, clock):
    consumer(rig, clock, failed=True, computed=4)
    shown = timeline(summarize([rig.tmp_path]), "d1")["kv"]
    assert (
        shown["outcome"] == "load_failed_recomputed"
        and shown["recomputation_shown"] is True
    )
    rig.made.clear()
    shutil.rmtree(rig.tmp_path)
    rig.tmp_path.mkdir()
    consumer(rig, clock, failed=True, computed=15)  # admitted with everything loaded
    kv = timeline(summarize([rig.tmp_path]), "d1")["kv"]
    assert kv["outcome"] == "load_failed" and kv["recomputation_shown"] is False


def test_an_engine_delivered_abort_is_not_a_normal_completion(rig):
    f = rig.frontend()
    f.output_finished(rs("r1", 3), SimpleNamespace(name="ABORT"))
    f.output_finished(rs("r2", 3), SimpleNamespace(name="LENGTH"))
    f.close()
    assert [(n, s) for n, _, s in phases(rig, f)] == [
        ("output_finished", 1),
        ("output_finished", 0),
    ]


def test_a_cancelled_request_gets_no_per_token_timing(rig, clock):
    sd = rig.scheduler()
    fd = rig.frontend()
    sd.request_queued(req("r1", 4))
    sd.schedule_step(out([new("r1", 0, 4)], {"r1": 4}), SimpleNamespace())
    sd.request_done(request("r1", status="FINISHED_ABORTED", output=2))
    fd.first_token_output(rs("r1"), [0])
    fd.output_finished(rs("r1", 6), SimpleNamespace(name="ABORT"))
    rig.close()
    t = timeline(summarize([rig.tmp_path]), "r1")
    assert t["state"] == "aborted" and t["api"]["output_finished_status"] == "aborted"
    assert t["decode_window"]["mean_time_per_token_after_first_ns"] is None
    assert "mean_time_per_token_after_first_ns" in t["unavailable"]


def test_a_remote_load_then_a_preemption_counts_every_interval_once(rig, clock):
    sd = rig.scheduler("kv_consumer")
    w = rig.worker("kv_consumer")
    fd = rig.frontend("kv_consumer")
    sd.request_queued(req("d1", 40))
    sd.load_prepared("d1", 30)
    sd.recv_finished("d1", False)
    sd.schedule_step(
        out([new("d1", 15, 40)], {"d1": 5}), SimpleNamespace()
    )  # admitted, partial
    sd.schedule_step(out(counts={"x": 1}, preempted={"d1"}), SimpleNamespace())
    meta = SimpleNamespace()
    sd.schedule_step(out([new("d1", 0, 40)], {"d1": 40}), meta)  # readmitted, done
    clock.value = 280
    w.step_begin(meta)
    clock.value = 300
    w.step_end()
    sd.request_done(request("d1", output=2))
    fd.first_token_output(rs("d1"), [0])
    fd.output_finished(rs("d1", 2), STOP)
    rig.close()
    rig.retime(
        sd,
        [100, 110, 150, 160, 200, 270, 270, 400],
        ("request_phase", "complete_observed"),
    )
    rig.retime(fd, [320, 410], ("request_phase",))
    t = timeline(summarize([rig.tmp_path]), "d1")
    stall = t["stall"]
    assert stall["basis"] == "scheduler_queued"
    assert (
        stall["queue_ns"],
        stall["waiting_for_kv_ns"],
        stall["earlier_admissions_ns"],
        stall["preempted_ns"],
        stall["step_host_span_ns"],
        stall["between_steps_ns"],
        stall["after_step_to_first_token_ns"],
        stall["other_ns"],
    ) == (20, 40, 40, 70, 20, 0, 20, 10)
    assert stall["complete"] is True
    assert t["steps"][0]["scheduler_to_step_begin_ns"] == 10 == stall["other_ns"]
    assert t["scheduler"]["preemptions"] == 1 and len(t["scheduler"]["admissions"]) == 2


def test_the_first_step_and_the_last_step_of_a_chunked_prefill_have_their_own_spans(
    rig, clock
):
    sd = rig.scheduler("kv_producer")
    w = rig.worker("kv_producer")
    sd.request_queued(req("p1", 8))
    m1, m2 = SimpleNamespace(), SimpleNamespace()
    sd.schedule_step(out([new("p1", 0, 8)], {"p1": 4}), m1)
    clock.value = 210
    w.step_begin(m1)
    clock.value = 250
    w.step_end()
    sd.schedule_step(out(counts={"p1": 4}), m2)
    clock.value = 310
    w.step_begin(m2)
    clock.value = 370
    w.step_end()
    sd.request_done(request("p1", output=1))
    rig.close()
    iv = timeline(summarize([rig.tmp_path]), "p1")["intervals_ns"]
    assert iv["first_step_host_span"] == 40 and iv["prefill_done_step_host_span"] == 60
    assert iv["prefill_steps_host_span_sum"] == 100


def test_every_step_reports_its_own_scheduler_to_worker_delay(rig, clock):
    consumer(rig, clock, tp=2, worker_ranks=(0, 1))
    t = timeline(summarize([rig.tmp_path]), "d1")
    assert [s["scheduler_to_step_begin_ns"] for s in t["steps"]] == [100]


def test_the_mooncake_exporter_draws_steps_and_windows_like_nixls(rig, clock, tmp_path):
    from vllm.kv_profiling.clock import ClockGroup
    from vllm.kv_profiling.perfetto import export_traces

    sp = rig.profiler(
        "scheduler",
        "kv_producer",
        cls=MooncakeProfiler,
        engine="engine-m",
        parallel={"tp": 1, "pp": 1, "pcp": 1, "workers": 1},
    )
    wp = rig.profiler(
        "worker",
        "kv_producer",
        cls=MooncakeProfiler,
        engine="engine-m",
        tp_rank=0,
        pp_rank=0,
        pcp_rank=0,
    )
    sp.request_queued(req("m1", 4))
    meta = SimpleNamespace()
    sp.schedule_step(out([new("m1", 0, 4)], {"m1": 4}), meta)
    clock.value = 5_000
    wp.step_begin(meta)
    clock.value = 8_000
    wp.step_end()
    rig.close()
    paths = [str(sp.output_dir / "events.jsonl"), str(wp.output_dir / "events.jsonl")]
    reference = sp.recorder.metadata["clock_id"]
    document = json.loads(
        next(
            iter(
                export_traces(
                    paths, str(tmp_path / "mc-out"), ClockGroup(reference, {})
                ).values()
            )
        ).read_text()
    )
    events = document["traceEvents"]
    (slice_,) = [e for e in events if e.get("name") == "step host span (shared batch)"]
    assert slice_["dur"] == 3.0
    assert [e for e in events if e.get("name") == "step window"]
    assert "request_phase:queued" in [e["name"] for e in events if e.get("ph") == "i"]


def test_a_calibrated_slice_is_drawn_between_its_two_projected_endpoints(
    rig, clock, tmp_path
):
    from fractions import Fraction

    from vllm.kv_profiling.clock import Projection
    from vllm.kv_profiling.perfetto import export_traces

    sp = rig.scheduler()
    wp = rig.worker()
    sp.request_queued(req("r1", 4))
    meta = SimpleNamespace()
    sp.schedule_step(out([new("r1", 0, 4)], {"r1": 4}), meta)
    clock.value = 5_000
    wp.step_begin(meta)
    clock.value = 8_000
    wp.step_end()
    rig.close()

    class Doubling:
        """A group whose projection runs the clock twice as fast (drift)."""

        def __init__(self, reference):
            self.reference = reference
            self.mappings = {}
            self.run_id = None

        def project(self, clock_id, timestamp):
            return Projection(Fraction(2 * timestamp), 0, True, False)

    reference = sp.recorder.metadata["clock_id"]
    paths = [str(sp.output_dir / "events.jsonl"), str(wp.output_dir / "events.jsonl")]
    document = json.loads(
        next(
            iter(
                export_traces(
                    paths, str(tmp_path / "drift-out"), Doubling(reference)
                ).values()
            )
        ).read_text()
    )
    (slice_,) = [
        e
        for e in document["traceEvents"]
        if e.get("name") == "step host span (shared batch)"
    ]
    assert (
        slice_["dur"] == 6.0
    )  # 3000 ns projected to 6000 ns: the drift is in the duration


def test_the_write_inventory_is_built_from_the_producers_records_alone(rig):
    ps, cs, pw = mooncake_pair(rig)
    alone = summarize([ps.output_dir, pw.output_dir])
    (write,) = alone["kv_writes"]
    assert (
        write["bytes"],
        write["descriptors"],
        write["write"],
        write["partition"],
    ) == (300, 3, "written", "exact")
    times = write["times"]
    assert None not in (
        times["bound_ns"],
        times["write_begin_ns"],
        times["write_returned_ns"],
        times["completed_ns"],
    )
    assert (
        times["bound_ns"]
        <= times["write_begin_ns"]
        <= times["write_returned_ns"]
        <= times["completed_ns"]
    )
    assert write["engine_id"] == "engine-p"
    assert [
        (m["request"], m["transfer_id"], m["share_bytes"], m["consumers"])
        for m in write["members"]
    ] == [("d-req1", "xfer-1", 100, []), ("d-req2", "xfer-2", 200, [])]


def test_a_consumer_occurrence_is_linked_to_the_producers_batch_by_its_members(
    rig,
):
    mooncake_pair(rig)
    summary = summarize([rig.tmp_path])
    (write,) = summary["kv_writes"]
    assert write["times"]["completed_ns"] is not None
    links = {(m["request"], m["transfer_id"]): m["consumers"] for m in write["members"]}
    assert [c["request"] for c in links[("d-req1", "xfer-1")]] == ["d-req1"]
    assert links[("d-req1", "xfer-1")][0]["request_occurrence"] == 0
    assert [c["request"] for c in links[("d-req2", "xfer-2")]] == ["d-req2"]
    one = timeline(summary, "d-req1")["kv"]["writes"]
    assert [(w["key"], w["share_bytes"]) for w in one] == [(write["key"], 100)]


# ------------------------------------------- the review of agent-3#63: four corrections


def test_a_restarted_scheduler_reusing_a_completed_id_inherits_nothing_from_the_old_one(
    rig, clock
):
    old = rig.scheduler("kv_consumer", engine="engine-d")
    fd = rig.frontend("kv_consumer")
    prefill_frontend = rig.frontend("kv_producer")
    old.request_queued(req("d1", 16))
    old.schedule_step(out([new("d1", 15, 16)], {"d1": 1}), SimpleNamespace())
    old.request_done(request("d1", output=5))
    fd.api_request("b1", "h1", "p1", False)
    fd.external_engine("p1", "d1")
    fd.first_token_output(rs("d1"), [0])
    fd.output_finished(rs("d1", 5), STOP)
    # The scheduler restarts and receives the same ID; it is aborted before admission.
    restarted = rig.scheduler("kv_consumer", engine="engine-d")
    restarted.request_queued(req("d1", 16))
    restarted.load_expected("d1", 0)
    restarted.load_prepared("d1", 15)
    restarted.recv_finished("d1", True)
    restarted.request_done(request("d1", status="FINISHED_ABORTED", output=0))
    # A prefill node's frontend with the same ID, inside the restarted request's time.
    prefill_frontend.first_token_output(rs("d1"), [0])
    prefill_frontend.output_finished(rs("d1", 1), STOP)
    rig.close()
    rig.retime(old, [1000, 1500, 1500, 9000], ("request_phase",))
    rig.retime(fd, [900, 900, 950, 2000, 9100], ("request_bind", "request_phase"))
    rig.retime(
        restarted,
        [24_520_000_000, 24_520_000_100, 24_520_000_300, 24_520_000_400],
        ("request_phase", "complete_observed"),
    )
    rig.retime(prefill_frontend, [24_520_000_200, 24_520_000_500], ("request_phase",))
    summary = summarize([rig.tmp_path])
    first = timeline(summary, "d1", source=old.recorder.metadata["process_id"])
    second = timeline(summary, "d1", source=restarted.recorder.metadata["process_id"])
    assert (
        first["state"] == "complete" and first["api"]["first_token_output_ns"] == 2000
    )
    assert second["state"] == "aborted"
    assert second["api"]["first_token_output_ns"] is None
    assert second["api"]["output_finished_ns"] is None
    assert second["api"]["source"] is None and "frontend" in second["unavailable"]
    # Each frontend row has at most one occurrence.
    claimed = [
        t["api"]["first_token_output_ns"]
        for t in summary["timelines"]
        if t["api"]["first_token_output_ns"] is not None
    ]
    assert claimed == [2000]


def test_a_prefill_frontend_is_no_candidate_for_a_decode_request_even_when_alone(
    rig, clock
):
    sd = rig.scheduler("kv_consumer", engine="engine-d")
    prefill_frontend = rig.frontend("kv_producer")
    sd.request_queued(req("d1", 16))
    sd.schedule_step(out([new("d1", 15, 16)], {"d1": 1}), SimpleNamespace())
    prefill_frontend.first_token_output(rs("d1"), [0])
    rig.close()
    t = timeline(summarize([rig.tmp_path]), "d1")
    assert t["api"]["source"] is None and t["api"]["first_token_output_ns"] is None


def test_with_request_id_randomization_off_the_arrival_follows_the_preceding_stage(
    rig, clock
):
    sd = rig.scheduler("kv_consumer", engine="engine-d")
    fd = rig.frontend("kv_consumer")
    sd.request_queued(req("x1", 4))
    sd.schedule_step(out([new("x1", 0, 4)], {"x1": 4}), SimpleNamespace())
    sd.request_done(request("x1"))
    fd.api_request("b1", "h1", "x1", False)  # the parent is the engine's own ID
    external = "x1"
    fd.external_engine(external, external)  # the engine ID is the external object
    fd.first_token_output(rs("x1"), [0])
    rig.close()
    rig.retime(sd, [1000, 1100, 1100, 2000], ("request_phase",))
    rig.retime(fd, [900, 900, 950, 1500], ("request_bind", "request_phase"))
    t = timeline(summarize([rig.tmp_path]), "x1")
    assert t["api"]["arrival_ns"] == 900 and "api_arrival" not in t["unavailable"]
    assert t["api"]["external_request"] == "x1"  # the recorded external ID is kept


def test_an_unhealthy_worker_of_an_earlier_scheduler_generation_does_not_touch_this_one(
    rig, clock
):
    results = []
    for index in (0, 1):
        sd = rig.scheduler("kv_consumer", tp=2, engine="engine-d")
        workers = [
            rig.worker("kv_consumer", tp_rank=r, engine="engine-d") for r in (0, 1)
        ]
        sd.request_queued(req(f"g{index}", 4))
        meta = SimpleNamespace()
        sd.schedule_step(out([new(f"g{index}", 0, 4)], {f"g{index}": 4}), meta)
        for w in workers:
            clock.value = 100 * (index + 1)
            w.step_begin(meta)
            clock.value += 30
            w.step_end()
        sd.request_done(request(f"g{index}"))
        results.append(workers)
    rig.close()
    path = results[0][1].output_dir / "profiler.json"  # the older generation's rank 1
    doc = json.loads(path.read_text())
    doc["hook_errors"] = 1
    path.write_text(json.dumps(doc))
    summary = summarize([rig.tmp_path])
    old, current = timeline(summary, "g0"), timeline(summary, "g1")
    assert current["quality"] == [] and current["steps"][0]["complete"] is True
    assert current["steps"][0]["host_span_ns"] == 30
    assert old["state"] == "observed_only" and old["steps"][0]["complete"] is False


def reused_mooncake(rig, *, occurrences=2):
    ps = rig.profiler(
        "scheduler", "kv_producer", cls=MooncakeProfiler, engine="engine-p"
    )
    cs = rig.profiler(
        "scheduler", "kv_consumer", cls=MooncakeProfiler, engine="engine-d"
    )
    pw = rig.profiler(
        "worker",
        "kv_producer",
        cls=MooncakeProfiler,
        engine="engine-p",
        tp_rank=0,
        pp_rank=0,
        pcp_rank=0,
    )
    ps.request_queued(req("p-req1", 8))
    ps.handoff_out("p-req1", "xfer-1", [[1, 2]])
    for _ in range(occurrences):
        cs.request_queued(req("d-req1", 8))
        cs.handoff_in("d-req1", "xfer-1", "engine-p")
        cs.request_done(request("d-req1", status="FINISHED_ABORTED", output=0))
    handle = pw.batch_bind("consumer:1", 0, 1)
    pw.batch_prepared(handle)
    pw.write_begin(handle, 100, 1)
    pw.write_returned(handle)
    pw.write_complete(handle, 100, 1, 50)
    pw.batch_member(handle, "d-req1", "xfer-1", True, (100, 1))
    pw.batch_retire(handle)
    rig.close()


def test_a_reused_request_and_transfer_id_links_no_consumer_to_the_one_write(rig):
    reused_mooncake(rig)
    summary = summarize([rig.tmp_path])
    (write,) = summary["kv_writes"]
    assert write["members"][0]["consumers"] == []
    assert all(j.get("identity_reused") for j in summary["joins"])
    for index in (0, 1):
        t = timeline(summary, "d-req1", occurrence=index)
        assert t["kv"]["writes"] == [] and "kv_write_link" in t["unavailable"]
        assert t["cross_node"] is None


def test_a_single_consumer_occurrence_still_links_its_write(rig):
    reused_mooncake(rig, occurrences=1)
    summary = summarize([rig.tmp_path])
    (write,) = summary["kv_writes"]
    assert [c["request"] for c in write["members"][0]["consumers"]] == ["d-req1"]
    assert len(timeline(summary, "d-req1")["kv"]["writes"]) == 1


def test_a_local_prefix_hit_is_counted_before_deciding_that_a_failed_load_recomputed(
    rig, clock
):
    """96 tokens: 64 cached locally, 32 loaded remotely (the external count is
    incremental to the local one); the load fails and the next admission schedules the
    remaining 32."""
    for failed, computed, shown, outcome in (
        (True, 64, True, "load_failed_recomputed"),
        (True, 95, False, "load_failed"),
        (False, 95, False, "received"),
    ):
        rig.made.clear()
        shutil.rmtree(rig.tmp_path)
        rig.tmp_path.mkdir()
        sd = rig.scheduler("kv_consumer", engine="engine-d")
        sd.request_queued(req("d1", 96))
        sd.load_expected("d1", 64)
        sd.load_prepared("d1", 32)
        sd.recv_finished("d1", failed)
        sd.schedule_step(
            out([new("d1", computed, 96)], {"d1": 96 - computed}), SimpleNamespace()
        )
        sd.request_done(request("d1"))
        rig.close()
        t = timeline(summarize([rig.tmp_path]), "d1")
        assert t["tokens"]["local_cached_at_load"] == 64
        assert t["kv"]["outcome"] == outcome
        if failed:
            assert t["kv"]["recomputation_shown"] is shown


def test_without_the_local_count_recomputation_is_unknown_not_denied(rig, clock):
    sd = rig.scheduler("kv_consumer", engine="engine-d")
    sd.request_queued(req("d1", 96))
    sd.load_prepared("d1", 32)  # no get_num_new_matched_tokens before (an old capture)
    sd.recv_finished("d1", True)
    sd.schedule_step(out([new("d1", 64, 96)], {"d1": 32}), SimpleNamespace())
    sd.request_done(request("d1"))
    rig.close()
    kv = timeline(summarize([rig.tmp_path]), "d1")["kv"]
    assert kv["outcome"] == "load_failed" and kv["recomputation_shown"] is None


# ---------------------------------- the review of agent-3#64: three corrections


def test_a_restarted_request_with_only_an_abort_output_inherits_no_earlier_arrival(
    rig, clock
):
    old = rig.scheduler("kv_consumer", engine="engine-d")
    fd = rig.frontend("kv_consumer")
    old.request_queued(req("d1", 16))
    old.schedule_step(out([new("d1", 15, 16)], {"d1": 1}), SimpleNamespace())
    old.request_done(request("d1", output=5))
    fd.api_request("b1", "h1", "p1", False)
    fd.external_engine("p1", "d1")
    fd.first_token_output(rs("d1"), [0])
    fd.output_finished(rs("d1", 5), STOP)
    restarted = rig.scheduler("kv_consumer", engine="engine-d")
    restarted.request_queued(req("d1", 16))
    restarted.request_done(request("d1", status="FINISHED_ABORTED", output=0))
    fd.output_aborted(rs("d1"))  # its own abort output, with no new API bind
    rig.close()
    rig.retime(old, [1000, 1500, 1500, 9000], ("request_phase",))
    rig.retime(restarted, [24_700_000_000, 24_700_000_300], ("request_phase",))
    rig.retime(
        fd,
        [900, 900, 950, 2000, 9100, 24_700_000_400],
        ("request_bind", "request_phase"),
    )
    summary = summarize([rig.tmp_path])
    first = timeline(summary, "d1", source=old.recorder.metadata["process_id"])
    second = timeline(summary, "d1", source=restarted.recorder.metadata["process_id"])
    assert (
        first["api"]["arrival_ns"] == 900 and first["api"]["output_finished_ns"] == 9100
    )
    assert (
        second["api"]["arrival_ns"] is None and "api_arrival" in second["unavailable"]
    )
    assert second["api"]["output_finished_ns"] == 24_700_000_400
    assert second["api"]["output_finished_status"] == "aborted"


def test_an_occurrence_window_is_half_open_so_a_boundary_row_has_one_owner(rig, clock):
    sd = rig.scheduler("kv_consumer", engine="engine-d")
    fd = rig.frontend("kv_consumer")
    sd.request_queued(req("d1", 16))
    sd.schedule_step(out([new("d1", 15, 16)], {"d1": 1}), SimpleNamespace())
    sd.request_done(request("d1", output=5))
    sd.request_queued(req("d1", 16))
    sd.request_done(request("d1", status="FINISHED_ABORTED", output=0))
    fd.first_token_output(rs("d1"), [0])
    fd.output_finished(rs("d1", 5), STOP)  # exactly at the second queueing
    rig.close()
    rig.retime(sd, [1000, 1500, 1500, 9000, 10000, 10100], ("request_phase",))
    rig.retime(fd, [2000, 10000], ("request_phase",))
    summary = summarize([rig.tmp_path])
    first, second = timeline(summary, "d1"), timeline(summary, "d1", occurrence=1)
    assert first["api"]["first_token_output_ns"] == 2000
    assert first["api"]["output_finished_ns"] is None
    assert second["api"]["output_finished_ns"] == 10000


def test_a_timestamp_inside_a_calibrations_uncertainty_of_a_boundary_has_no_owner(
    rig, clock, tmp_path
):
    from vllm.kv_profiling.clock import ClockMapping, ClockSample

    sd = rig.scheduler("kv_consumer", engine="engine-d")
    fd = rig.frontend("kv_consumer")
    sd.request_queued(req("d1", 16))
    sd.request_done(request("d1", output=1))
    sd.request_queued(req("d1", 16))
    sd.request_done(request("d1", output=1))
    fd.first_token_output(
        rs("d1"), [0]
    )  # 25 ns after the second queueing: either owner
    fd.output_finished(rs("d1", 1), STOP)  # well inside the second
    rig.close()
    rig.retime(sd, [1000, 5000, 10000, 20000], ("request_phase",))
    rig.retime(fd, [10025, 15000], ("request_phase",))
    path = fd.output_dir / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    for row in rows:
        if row.get("type") == "metadata":
            row["boot_id"] = "another-host"
            clock_id = row["clock_id"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    mapping = ClockMapping.from_samples(
        [
            ClockSample(
                run_id=rig.run_id,
                local_generation=sd.recorder.metadata["clock_id"],
                remote_generation=clock_id,
                t1=1000,
                t2=1000,
                t3=2000,
                t4=2000,
            )
        ]
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(mapping.to_json()))
    summary = summarize([rig.tmp_path], calibration=[str(calibration)])
    first, second = timeline(summary, "d1"), timeline(summary, "d1", occurrence=1)
    assert first["api"]["first_token_output_ns"] is None
    assert second["api"]["first_token_output_ns"] is None  # 10025 is unowned
    assert first["api"]["output_finished_ns"] is None
    assert second["api"]["output_finished_ns"] == 15000  # clear of the boundary


def test_a_late_worker_failure_with_only_an_older_generations_marker_is_not_assigned(
    rig, clock
):
    old = rig.scheduler("kv_consumer", engine="engine-d")
    worker = rig.worker("kv_consumer", engine="engine-d")
    meta = SimpleNamespace()
    old.request_queued(req("d0", 4))
    old.schedule_step(out([new("d0", 0, 4)], {"d0": 4}), meta)
    clock.value = 100
    worker.step_begin(meta)
    clock.value = 150
    worker.step_end()
    old.request_done(request("d0"))
    restarted = rig.scheduler("kv_consumer", engine="engine-d")  # no worker marker yet
    restarted.request_queued(req("d1", 16))
    restarted.load_prepared("d1", 15)
    worker.failure("transfer_failed", "d1", None)
    rig.close()
    rig.retime(old, [1000, 1100, 1100, 1200], ("request_phase",))
    rig.retime(restarted, [5000, 5100], ("request_phase",))
    rig.retime(worker, [5500], ("failed",))
    summary = summarize([rig.tmp_path])
    t = timeline(summary, "d1", source=restarted.recorder.metadata["process_id"])
    assert t["kv"]["outcome"] == "unknown" and t["kv"]["failures"] == []
    assert [f["code"] for f in t["kv"]["failures_unlinked"]] == ["TRANSFER_FAILED"]
    assert "kv_failures_unlinked" in t["unavailable"]


def test_a_partial_load_is_judged_against_the_position_a_success_resumes_at(rig, clock):
    """66 tokens, 64 cached locally, one remote token: a success resumes at 65 (the
    last prompt token stays for the scheduler); a failure resumes at 64."""
    for failed, computed, shown in (
        (False, 65, False),
        (True, 65, False),
        (True, 64, True),
    ):
        rig.made.clear()
        shutil.rmtree(rig.tmp_path)
        rig.tmp_path.mkdir()
        sd = rig.scheduler("kv_consumer", engine="engine-d")
        sd.request_queued(req("d1", 66))
        sd.load_expected("d1", 64)
        sd.load_prepared("d1", 1)
        sd.recv_finished("d1", failed)
        sd.schedule_step(
            out([new("d1", computed, 66)], {"d1": 66 - computed}), SimpleNamespace()
        )
        sd.request_done(request("d1"))
        rig.close()
        kv = timeline(summarize([rig.tmp_path]), "d1")["kv"]
        if failed:
            assert kv["recomputation_shown"] is shown, (computed, kv)
        else:
            assert kv["outcome"] == "received"


def test_a_replay_policy_leaves_recomputation_unknown(rig, clock):
    sd = rig.scheduler("kv_consumer", engine="engine-d", bounded_replay=True)
    sd.request_queued(req("d1", 96))
    sd.load_expected("d1", 64)
    sd.load_prepared("d1", 32)
    sd.recv_finished("d1", True)
    sd.schedule_step(out([new("d1", 64, 96)], {"d1": 32}), SimpleNamespace())
    sd.request_done(request("d1"))
    rig.close()
    t = timeline(summarize([rig.tmp_path]), "d1")
    assert (
        t["kv"]["outcome"] == "load_failed" and t["kv"]["recomputation_shown"] is None
    )
    assert "recomputation" in t["unavailable"]


# ---------------------------------- the review of agent-3#65: two attribution errors


def two_engines(rig, *, second_queued, first_token=2000, finish=9100):
    """Two independent decode engines receive the same request ID: A produces the
    token, B is aborted before admission; one frontend recorded the outputs."""
    a = rig.scheduler("kv_consumer", engine="engine-a")
    b = rig.scheduler("kv_consumer", engine="engine-b")
    fd = rig.frontend("kv_consumer")
    a.request_queued(req("d1", 16))
    a.schedule_step(out([new("d1", 15, 16)], {"d1": 1}), SimpleNamespace())
    a.request_done(request("d1", output=5))
    b.request_queued(req("d1", 16))
    b.request_done(request("d1", status="FINISHED_ABORTED", output=0))
    fd.first_token_output(rs("d1"), [0])
    fd.output_finished(rs("d1", 5), STOP)
    rig.close()
    rig.retime(a, [1000, 1500, 1500, 9000], ("request_phase",))
    rig.retime(b, [second_queued, 9500], ("request_phase",))
    rig.retime(fd, [first_token, finish], ("request_phase",))
    summary = summarize([rig.tmp_path])
    return (
        timeline(summary, "d1", source=a.recorder.metadata["process_id"]),
        timeline(summary, "d1", source=b.recorder.metadata["process_id"]),
    )


def test_concurrent_engines_reusing_an_id_leave_the_frontend_rows_unowned(rig, clock):
    for second_queued in (1100, 1000):  # 100 ns later, and simultaneous
        rig.made.clear()
        shutil.rmtree(rig.tmp_path)
        rig.tmp_path.mkdir()
        first, second = two_engines(rig, second_queued=second_queued)
        for t in (first, second):
            assert t["api"]["first_token_output_ns"] is None, t["state"]
            assert t["api"]["output_finished_ns"] is None
            assert "frontend" in t["unavailable"]


def test_a_row_that_only_the_first_engine_could_own_stays_with_it(rig, clock):
    first, second = two_engines(rig, second_queued=1100, first_token=1050)
    assert first["api"]["first_token_output_ns"] == 1050  # before B queued
    assert second["api"]["first_token_output_ns"] is None
    assert first["api"]["output_finished_ns"] is None  # after both: unowned


def test_sequential_reuse_keeps_each_requests_own_api_entry(rig, clock):
    old = rig.scheduler("kv_consumer", engine="engine-d")
    restarted = rig.scheduler("kv_consumer", engine="engine-d")
    fd = rig.frontend("kv_consumer")
    for sd in (old, restarted):
        sd.request_queued(req("d1", 16))
        sd.schedule_step(out([new("d1", 15, 16)], {"d1": 1}), SimpleNamespace())
        sd.request_done(request("d1", output=5))
    for _ in (0, 1):  # two complete API binds of the same ID
        fd.api_request("b1", "h1", "p1", False)
        fd.external_engine("p1", "d1")
        fd.first_token_output(rs("d1"), [0])
        fd.output_finished(rs("d1", 5), STOP)
    rig.close()
    second = 1_000_000_000
    rig.retime(old, [1000, 1500, 1500, 9000], ("request_phase",))
    rig.retime(
        restarted, [second + t for t in (1000, 1500, 1500, 9000)], ("request_phase",)
    )
    rig.retime(
        fd,
        [900, 900, 950, 2000, 9100] + [second + t for t in (900, 900, 950, 2000, 9100)],
        ("request_bind", "request_phase"),
    )
    summary = summarize([rig.tmp_path])
    first = timeline(summary, "d1", source=old.recorder.metadata["process_id"])
    again = timeline(summary, "d1", source=restarted.recorder.metadata["process_id"])
    assert (first["api"]["arrival_ns"], again["api"]["arrival_ns"]) == (
        900,
        second + 900,
    )
    assert first["stall"]["basis"] == again["stall"]["basis"] == "api_arrival"
    assert first["api"]["first_token_output_ns"] == 2000
    assert again["api"]["first_token_output_ns"] == second + 2000


def failed_load_cycle(rig, first_resume, second_resume):
    sd = rig.scheduler("kv_consumer", engine="engine-d")
    sd.request_queued(req("d1", 100))
    sd.load_expected("d1", 64)
    sd.load_prepared("d1", 32)  # 96 cached tokens expected
    sd.recv_finished("d1", True)
    sd.schedule_step(out([new("d1", first_resume, 100)], {"d1": 16}), SimpleNamespace())
    sd.schedule_step(out(counts={"x": 1}, preempted={"d1"}), SimpleNamespace())
    sd.schedule_step(
        out([new("d1", second_resume, 100)], {"d1": 100 - second_resume}),
        SimpleNamespace(),
    )
    sd.request_done(request("d1"))
    rig.close()
    return timeline(summarize([rig.tmp_path]), "d1")["kv"]


def test_a_failed_load_is_paired_with_its_first_admission_not_the_last(rig, clock):
    kv = failed_load_cycle(rig, first_resume=64, second_resume=96)
    assert (
        kv["recomputation_shown"] is True and kv["outcome"] == "load_failed_recomputed"
    )
    rig.made.clear()
    shutil.rmtree(rig.tmp_path)
    rig.tmp_path.mkdir()
    kv = failed_load_cycle(rig, first_resume=96, second_resume=64)
    assert kv["recomputation_shown"] is False and kv["outcome"] == "load_failed"


# ---------------- the review of agent-3#66: a reused client ID, two engine IDs


def reused_client_id(rig, *, second_entry):
    """Two sequential requests with the same client ID (vLLM's default engine-ID
    randomization gives them different engine IDs)."""
    sd = rig.scheduler("kv_consumer", engine="engine-d")
    fd = rig.frontend("kv_consumer")
    for engine in ("d1", "d2"):
        sd.request_queued(req(engine, 16))
        sd.schedule_step(out([new(engine, 15, 16)], {engine: 1}), SimpleNamespace())
        sd.request_done(request(engine, output=5))
    second = 1_000_000_000
    fd.api_request("b1", "h1", "p1", False)  # the client ID p1, used twice
    fd.external_engine("p1", "d1")
    fd.first_token_output(rs("d1"), [0])
    if second_entry:
        fd.api_request("b1", "h1", "p1", False)
    fd.external_engine("p1", "d2")
    fd.first_token_output(rs("d2"), [0])
    rig.close()
    rig.retime(
        sd,
        [
            1000,
            1500,
            1500,
            9000,
            second + 1000,
            second + 1500,
            second + 1500,
            second + 9000,
        ],
        ("request_phase",),
    )
    kinds = ("request_bind", "request_phase")
    first = [900, 900, 950, 2000]
    later = (
        [second + 900, second + 900, second + 950, second + 2000]
        if second_entry
        else [second + 950, second + 2000]
    )
    rig.retime(fd, first + later, kinds)
    summary = summarize([rig.tmp_path])
    return timeline(summary, "d1"), timeline(summary, "d2")


def test_a_client_id_used_twice_gives_each_request_its_own_api_entry(rig, clock):
    first, second = reused_client_id(rig, second_entry=True)
    assert first["api"]["arrival_ns"] == 900
    assert second["api"]["arrival_ns"] == 1_000_000_900
    assert first["stall"]["basis"] == second["stall"]["basis"] == "api_arrival"
    # Not a second of spurious pre-engine delay: the entry is its own.
    assert second["intervals_ns"]["api_to_queued"] == 100


def test_with_the_second_entry_missing_only_the_first_request_has_one(rig, clock):
    first, second = reused_client_id(rig, second_entry=False)
    assert first["api"]["arrival_ns"] == 900
    assert (
        second["api"]["arrival_ns"] is None and "api_arrival" in second["unavailable"]
    )
    assert second["stall"]["basis"] == "scheduler_queued"


def test_overlapping_entries_before_the_binds_are_not_guessed(rig, clock):
    sd = rig.scheduler("kv_consumer", engine="engine-d")
    fd = rig.frontend("kv_consumer")
    for engine in ("d1", "d2"):
        sd.request_queued(req(engine, 16))
        sd.request_done(request(engine, output=1))
    fd.api_request("b1", "h1", "p1", False)
    fd.api_request("b1", "h1", "p1", False)  # two entries before either bind
    fd.external_engine("p1", "d1")
    fd.external_engine("p1", "d2")
    rig.close()
    rig.retime(sd, [1000, 1100, 1200, 1300], ("request_phase",))
    rig.retime(fd, [100, 100, 200, 200, 300, 400], ("request_bind",))
    summary = summarize([rig.tmp_path])
    for engine in ("d1", "d2"):
        assert timeline(summary, engine)["api"]["arrival_ns"] is None


# ---------- the review of agent-3#67: overlapping entries, a second prompt's item bind


def replay_frontend(rig, actions, engines):
    """Drive the frontend's real hooks in time order. ``actions``: (ms, kind, argument)
    with kinds entry / item / bind; every engine also gets a first token and a finish
    after its bind. ``engines``: tag -> (ms queued, external ID); the scheduler queues
    each, admits it and finishes it."""
    sd = rig.scheduler("kv_consumer", engine="engine-d")
    fd = rig.frontend("kv_consumer")
    internal = {tag: f"{ext}-{tag}" for tag, (_, ext) in engines.items()}
    timed = list(actions)
    for tag in engines:
        when = next(t for t, kind, arg in actions if kind == "bind" and arg == tag)
        timed += [(when + 20, "first", tag), (when + 22, "end", tag)]
    times = []
    for ms, kind, argument in sorted(timed, key=lambda item: item[0]):
        if kind == "entry":
            client = argument or "same"
            fd.api_request(client, client, f"cmpl-{client}", False)
            times += [ms, ms]
        elif kind == "item":
            client, number = (
                argument if isinstance(argument, tuple) else ("same", argument)
            )
            fd.parent_item(f"cmpl-{client}", f"cmpl-{client}-{number}")
            times.append(ms)
        elif kind == "bind":
            fd.external_engine(engines[argument][1], internal[argument])
            times.append(ms)
        elif kind == "first":
            fd.first_token_output(rs(internal[argument]), [0])
            times.append(ms)
        else:
            fd.output_finished(rs(internal[argument], 2), STOP)
            times.append(ms)
    scheduler_times = []
    for tag, (queued, _) in engines.items():
        rid = internal[tag]
        sd.request_queued(req(rid, 16))
        sd.schedule_step(out([new(rid, 15, 16)], {rid: 1}), SimpleNamespace())
        sd.request_done(request(rid, output=2))
        scheduler_times += [queued, queued + 3, queued + 3, queued + 8]
    rig.close()
    ms = 1_000_000
    rig.retime(fd, [t * ms for t in times], ("request_bind", "request_phase"))
    rig.retime(sd, [t * ms for t in scheduler_times], ("request_phase",))
    summary = summarize([rig.tmp_path])
    return {tag: timeline(summary, internal[tag]) for tag in engines}


def test_overlapping_entries_leave_every_pairing_that_could_be_swapped_unassigned(
    rig, clock
):
    """A and B entered before either bound, C entered while B still waited, D used the
    same client ID after every earlier bind."""
    actions = [
        (100, "entry", None), (110, "item", 0), (200, "entry", None), (210, "item", 0),
        (300, "bind", "A"), (400, "entry", None), (410, "item", 0), (500, "bind", "B"),
        (600, "bind", "C"), (2000, "entry", None), (2010, "item", 0),
        (2100, "bind", "D"),
    ]  # fmt: skip
    ext = "cmpl-same-0"
    found = replay_frontend(
        rig,
        actions,
        {"A": (310, ext), "B": (510, ext), "C": (610, ext), "D": (2110, ext)},
    )
    ms = 1_000_000
    for tag in "ABC":
        assert found[tag]["api"]["arrival_ns"] is None, tag
        assert found[tag]["stall"]["basis"] == "scheduler_queued"
    assert found["D"]["api"]["arrival_ns"] == 2000 * ms  # the isolated one recovers
    assert found["D"]["stall"]["basis"] == "api_arrival"


def test_a_second_prompt_inherits_the_arrival_of_its_calls_entry(rig, clock):
    """The same client ID in two sequential calls: one prompt, then two; only the second
    call has a parent-to-item bind for item 1."""
    actions = [
        (100, "entry", None), (110, "item", 0), (150, "bind", "A0"),
        (900, "entry", None), (910, "item", 0), (920, "item", 1),
        (950, "bind", "B0"), (970, "bind", "B1"),
    ]  # fmt: skip
    found = replay_frontend(
        rig,
        actions,
        {
            "A0": (160, "cmpl-same-0"),
            "B0": (960, "cmpl-same-0"),
            "B1": (980, "cmpl-same-1"),
        },
    )
    ms = 1_000_000
    assert [found[t]["api"]["arrival_ns"] for t in ("A0", "B0", "B1")] == [
        100 * ms,
        900 * ms,
        900 * ms,
    ]
    assert found["B1"]["stall"]["basis"] == "api_arrival"


# ---------- the review of agent-3#68: a sibling prompt needs the call to be identified


def test_a_call_with_no_recorded_entry_does_not_take_the_first_calls_via_its_item(
    rig, clock
):
    """The #66 missing-entry control with the parent-to-item stage recorded: two
    sequential calls each bind prompt 0; only the first recorded its API entry."""
    actions = [
        (100, "entry", None), (110, "item", 0), (150, "bind", "A0"),
        (910, "item", 0), (950, "bind", "B0"),
    ]  # fmt: skip
    found = replay_frontend(
        rig, actions, {"A0": (160, "cmpl-same-0"), "B0": (960, "cmpl-same-0")}
    )
    assert found["A0"]["api"]["arrival_ns"] == 100_000_000
    assert found["B0"]["api"]["arrival_ns"] is None
    assert found["B0"]["stall"]["basis"] == "scheduler_queued"


def test_an_overlapping_call_does_not_take_the_older_calls_second_prompt(rig, clock):
    """A entered at 100, B at 200; A's second prompt bound while both calls were open,
    so the recorded parent could be either call's."""
    actions = [
        (100, "entry", None), (110, "item", 0), (200, "entry", None), (210, "item", 0),
        (300, "item", 1), (400, "bind", "A0"), (410, "bind", "B0"), (420, "bind", "A1"),
    ]  # fmt: skip
    found = replay_frontend(
        rig,
        actions,
        {
            "A0": (430, "cmpl-same-0"),
            "B0": (440, "cmpl-same-0"),
            "A1": (450, "cmpl-same-1"),
        },
    )
    assert found["A1"]["api"]["arrival_ns"] is None  # either call's: never B's 200 ms
    ms = 1_000_000
    for tag, correct in (("A0", 100), ("B0", 200)):
        assert found[tag]["api"]["arrival_ns"] in (None, correct * ms), tag


def test_genuine_sequential_siblings_still_share_their_calls_entry(rig, clock):
    """The #67 control beside the two above: the first call's prompt was bound before
    the second call entered, so the second call's prompts are identified."""
    actions = [
        (100, "entry", None), (110, "item", 0), (150, "bind", "A0"),
        (900, "entry", None), (910, "item", 0), (920, "item", 1),
        (950, "bind", "B0"), (970, "bind", "B1"),
    ]  # fmt: skip
    found = replay_frontend(
        rig,
        actions,
        {
            "A0": (160, "cmpl-same-0"),
            "B0": (960, "cmpl-same-0"),
            "B1": (980, "cmpl-same-1"),
        },
    )
    ms = 1_000_000
    assert [found[t]["api"]["arrival_ns"] for t in ("A0", "B0", "B1")] == [
        100 * ms,
        900 * ms,
        900 * ms,
    ]


# ---- the review of agent-3#69: every interleaving of two calls, none misattributed


def interleavings(first, second):
    """Every merge of two event sequences that keeps the order inside each."""
    if not first:
        yield list(second)
        return
    if not second:
        yield list(first)
        return
    for rest in interleavings(first[1:], second):
        yield [first[0], *rest]
    for rest in interleavings(first, second[1:]):
        yield [second[0], *rest]


def test_no_interleaving_of_two_calls_gives_a_prompt_the_other_calls_arrival(
    rig, clock
):
    """A submits two prompts, B reuses the client ID with one; the events of each call
    keep their order (entry, item binds, then the engine binds of its prompts). An
    unavailable arrival is accepted for a prompt the records leave ambiguous; a wrong
    one never is; the two sequential orders must be exact."""
    a_events = [
        ("A", "entry"),
        ("A", "item0"),
        ("A", "item1"),
        ("A", "bind0"),
        ("A", "bind1"),
    ]
    b_events = [("B", "entry"), ("B", "item0"), ("B", "bind0")]
    orders = list(interleavings(a_events, b_events))
    assert len(orders) == 56
    wrong, exact_when_sequential = [], []
    for order in orders:
        rig.made.clear()
        shutil.rmtree(rig.tmp_path)
        rig.tmp_path.mkdir()
        actions, entry_time, bind_time = [], {}, {}
        for position, (call, event) in enumerate(order):
            when = 100 + 10 * position
            if event == "entry":
                actions.append((when, "entry", None))
                entry_time[call] = when
            elif event.startswith("item"):
                actions.append((when, "item", int(event[-1])))
            else:
                tag = f"{call}{event[-1]}"
                actions.append((when, "bind", tag))
                bind_time[tag] = when
        engines = {
            "A0": (bind_time["A0"] + 5, "cmpl-same-0"),
            "A1": (bind_time["A1"] + 5, "cmpl-same-1"),
            "B0": (bind_time["B0"] + 5, "cmpl-same-0"),
        }
        found = replay_frontend(rig, actions, engines)
        correct = {"A0": entry_time["A"], "A1": entry_time["A"], "B0": entry_time["B"]}
        ms = 1_000_000
        for tag, truth in correct.items():
            arrival = found[tag]["api"]["arrival_ns"]
            if arrival not in (None, truth * ms):
                wrong.append((order, tag, arrival // ms, truth))
        names = [call for call, _ in order]
        if names in (["A"] * 5 + ["B"] * 3, ["B"] * 3 + ["A"] * 5):
            exact_when_sequential.append(
                {
                    tag: found[tag]["api"]["arrival_ns"] == truth * ms
                    for tag, truth in correct.items()
                }
            )
    assert not wrong, wrong[:3]
    assert len(exact_when_sequential) == 2
    assert all(all(flags.values()) for flags in exact_when_sequential)


def test_the_call_blocks_never_misattribute_any_interleaving_of_three_calls():
    """The matcher alone, over every interleaving of three calls that reuse one client
    ID (two prompts, one, one); an ambiguous or unmatched bind is accepted, a wrong
    entry is not, and calls that do not overlap in time are exact."""
    from vllm.kv_profiling.timeline import _AMBIGUOUS, Timelines

    calls = {
        "A": [("entry", None), ("item", 0), ("item", 1), ("bind", 0), ("bind", 1)],
        "B": [("entry", None), ("item", 0), ("bind", 0)],
        "C": [("entry", None), ("item", 0), ("bind", 0)],
    }

    def merges(sequences):
        sequences = {k: v for k, v in sequences.items() if v}
        if not sequences:
            yield []
            return
        for name, rest in sequences.items():
            tail = {k: (v[1:] if k == name else v) for k, v in sequences.items()}
            for order in merges(tail):
                yield [(name, rest[0]), *order]

    parent = "cmpl-same"
    checked = exact = 0
    for order in merges({k: list(v) for k, v in calls.items()}):
        index = {
            "entries": {}, "items_of": {}, "engine_binds": {}, "item_rows": {},
            "blocks": {},
        }  # fmt: skip
        truth, spans = {}, {}
        for position, (call, (kind, number)) in enumerate(order):
            time = 100 + position
            spans.setdefault(call, [time, time])[1] = time
            if kind == "entry":
                index["entries"].setdefault(parent, []).append((time, f"base-{call}"))
            elif kind == "item":
                index["items_of"].setdefault(parent, []).append(
                    (time, f"{parent}-{number}")
                )
                index["item_rows"].setdefault(f"{parent}-{number}", []).append(
                    (time, parent)
                )
            else:
                engine = f"{call}{number}"
                index["engine_binds"].setdefault(f"{parent}-{number}", []).append(
                    (time, engine)
                )
                truth[(time, engine)] = (call, time)
        result = Timelines.__new__(Timelines)._call_blocks(index, parent)
        entry_of = {
            call: index["entries"][parent][i]
            for i, call in enumerate([c for c, (k, _) in order if k == "entry"])
        }
        for key, (call, _) in truth.items():
            checked += 1
            verdict = result.get(key)
            if verdict is not None and verdict is not _AMBIGUOUS:
                assert verdict == entry_of[call], (order, key, verdict)
        # Calls whose event spans do not overlap anyone else's are exact.
        for call, (low, high) in spans.items():
            alone = all(
                other == call or high < lo or hi < low
                for other, (lo, hi) in spans.items()
            )
            if alone:
                for key, (owner, _) in truth.items():
                    if owner == call:
                        exact += 1
                        assert result.get(key) == entry_of[call], (order, key)
    assert checked == 9240 * 4  # every order (11!/(5!3!3!)), all four engine binds
    assert exact > 0


# -------- the review of agent-3#70: a client ID that equals another call's prompt ID


def test_a_client_id_equal_to_another_calls_prompt_id_keeps_both_arrivals(rig, clock):
    """Clients ``batch`` and ``batch-0``: the first call's prompt and the second call's
    parent are both ``cmpl-batch-0``. Both calls have an item stage."""
    actions = [
        (100, "entry", "batch"), (110, "item", ("batch", 0)), (300, "bind", "P"),
        (200, "entry", "batch-0"), (210, "item", ("batch-0", 0)), (310, "bind", "Q"),
    ]  # fmt: skip
    found = replay_frontend(
        rig,
        sorted(actions),
        {"P": (305, "cmpl-batch-0"), "Q": (315, "cmpl-batch-0-0")},
    )
    ms = 1_000_000
    assert found["P"]["api"]["arrival_ns"] == 100 * ms
    assert found["Q"]["api"]["arrival_ns"] == 200 * ms
    assert found["Q"]["stall"]["basis"] == "api_arrival"


def test_a_third_call_reusing_the_client_id_keeps_the_second_calls_entry(rig, clock):
    actions = [
        (100, "entry", "batch"), (110, "item", ("batch", 0)), (120, "bind", "P"),
        (200, "entry", "batch-0"), (210, "item", ("batch-0", 0)), (220, "bind", "Q"),
        (400, "entry", "batch-0"), (410, "item", ("batch-0", 0)), (420, "bind", "R"),
    ]  # fmt: skip
    found = replay_frontend(
        rig,
        actions,
        {
            "P": (125, "cmpl-batch-0"),
            "Q": (225, "cmpl-batch-0-0"),
            "R": (425, "cmpl-batch-0-0"),
        },
    )
    ms = 1_000_000
    assert found["P"]["api"]["arrival_ns"] == 100 * ms
    assert found["Q"]["api"]["arrival_ns"] == 200 * ms  # not the third call's 400 ms
    assert found["R"]["api"]["arrival_ns"] == 400 * ms


def test_a_parent_with_no_item_stage_that_collides_with_a_prompt_id_stays_unassigned(
    rig, clock
):
    """The first call has an item stage (prompt ``cmpl-batch-0``); the second call's
    parent is that same ID and binds the engine directly: the two engine binds cannot be
    told apart, so neither gets an arrival."""
    actions = [
        (100, "entry", "batch"), (110, "item", ("batch", 0)), (120, "bind", "P"),
        (200, "entry", "batch-0"), (220, "bind", "Q"),
    ]  # fmt: skip
    found = replay_frontend(
        rig, actions, {"P": (125, "cmpl-batch-0"), "Q": (225, "cmpl-batch-0")}
    )
    for tag in "PQ":
        assert found[tag]["api"]["arrival_ns"] is None, tag
        assert found[tag]["stall"]["basis"] == "scheduler_queued"


# ------ the review of agent-3#71: a direct bind under the ID of another call's prompt


def test_a_direct_bind_under_a_prompt_id_leaves_the_sibling_prompt_unassigned(
    rig, clock
):
    """A submits two prompts (``cmpl-batch-0``, ``cmpl-batch-1``); B, client
    ``batch-0``, has no item stage and binds under ``cmpl-batch-0``; a second ``batch``
    call enters at 400 ms before A's prompts bind."""
    actions = [
        (100, "entry", "batch"), (110, "item", ("batch", 0)),
        (120, "item", ("batch", 1)), (130, "entry", "batch-0"), (200, "bind", "B"),
        (400, "entry", "batch"), (410, "item", ("batch", 0)),
        (500, "bind", "A0"), (510, "bind", "A1"), (520, "bind", "C0"),
    ]  # fmt: skip
    found = replay_frontend(
        rig,
        actions,
        {
            "B": (205, "cmpl-batch-0"),
            "A0": (505, "cmpl-batch-0"),
            "A1": (515, "cmpl-batch-1"),
            "C0": (525, "cmpl-batch-0"),
        },
    )
    ms = 1_000_000
    truth = {"B": 130, "A0": 100, "A1": 100, "C0": 400}
    for tag, correct in truth.items():
        arrival = found[tag]["api"]["arrival_ns"]
        assert arrival in (None, correct * ms), (tag, arrival)
    assert found["A1"]["api"]["arrival_ns"] != 400 * ms


def test_no_order_of_a_prompt_call_a_colliding_direct_call_and_a_third_misattributes():
    """The matcher alone: A (two prompts), B (a direct call under A's first prompt ID),
    C (a second call with the same client ID as A), and, in turn, no other call, a later
    call D that reuses B's client ID with an indexed prompt, and the same call earlier:
    no engine bind is given another call's entry in any interleaving."""
    from vllm.kv_profiling.timeline import _AMBIGUOUS, Timelines

    matcher = Timelines.__new__(Timelines)
    sequences = {
        "A": [("entry", 0), ("item", 0), ("item", 1), ("bind", 0), ("bind", 1)],
        "B": [("entry", 0), ("bind", 0)],
        "C": [("entry", 0), ("item", 0), ("bind", 0)],
    }
    indexed_d = [("D", ("entry", 0)), ("D", ("item", 0)), ("D", ("bind", 0))]

    def merges(rest):
        rest = {k: v for k, v in rest.items() if v}
        if not rest:
            yield []
            return
        for name, events in rest.items():
            tail = {k: (v[1:] if k == name else v) for k, v in rest.items()}
            for order in merges(tail):
                yield [(name, events[0]), *order]

    parent, direct_parent = "cmpl-batch", "cmpl-batch-0"
    orders = 0
    base = [list(order) for order in merges(sequences)]
    cases = [(order, "none") for order in base]
    cases += [(order + indexed_d, "after") for order in base]
    cases += [(indexed_d + order, "before") for order in base]
    for order, _ in cases:
        orders += 1
        index = {
            "entries": {}, "items_of": {}, "engine_binds": {}, "item_rows": {},
            "blocks": {},
        }  # fmt: skip
        owner, entry_of = {}, {}
        for position, (call, (kind, number)) in enumerate(order):
            time = 100 + position
            own = direct_parent if call in "BD" else parent
            if kind == "entry":
                index["entries"].setdefault(own, []).append((time, f"base-{call}"))
                entry_of[call] = (time, f"base-{call}")
            elif kind == "item":
                item = f"{own}-{number}"
                index["items_of"].setdefault(own, []).append((time, item))
                index["item_rows"].setdefault(item, []).append((time, own))
            else:
                direct = call == "B"
                external = own if direct else f"{own}-{number}"
                engine = f"{call}{number}"
                index["engine_binds"].setdefault(external, []).append((time, engine))
                owner[(time, engine)] = call
        for target in (parent, direct_parent):
            result = matcher._call_blocks(index, target)
            for key, verdict in result.items():
                if verdict is not None and verdict is not _AMBIGUOUS:
                    assert verdict == entry_of[owner[key]], (order, key, verdict)
    assert orders == 3 * 2520


def test_the_overlapping_three_call_schedule_with_the_collision_stays_unassigned(
    rig, clock
):
    """The #70 schedule itself: ``batch``, then two calls with client ``batch-0``, all
    three entered and item-bound before any engine bind."""
    actions = [
        (100, "entry", "batch"), (110, "item", ("batch", 0)),
        (200, "entry", "batch-0"), (210, "item", ("batch-0", 0)),
        (400, "entry", "batch-0"), (410, "item", ("batch-0", 0)),
        (500, "bind", "P"), (510, "bind", "Q"), (520, "bind", "R"),
    ]  # fmt: skip
    found = replay_frontend(
        rig,
        actions,
        {
            "P": (505, "cmpl-batch-0"),
            "Q": (515, "cmpl-batch-0-0"),
            "R": (525, "cmpl-batch-0-0"),
        },
    )
    ms = 1_000_000
    assert found["P"]["api"]["arrival_ns"] in (None, 100 * ms)
    for tag in "QR":  # either call could own either prompt: scheduler-queue timing
        assert found[tag]["api"]["arrival_ns"] is None
        assert found[tag]["stall"]["basis"] == "scheduler_queued"


def test_a_later_call_reusing_the_direct_calls_id_does_not_change_an_earlier_request(
    rig, clock
):
    """The #71 case unchanged, then a call at 1,000 ms that reuses ``batch-0`` with an
    indexed prompt: A1, finished long before, keeps the scheduler-queue fallback (it
    does not inherit the 400 ms arrival), because the shared ID is used both ways."""
    actions = [
        (100, "entry", "batch"), (110, "item", ("batch", 0)),
        (120, "item", ("batch", 1)), (130, "entry", "batch-0"), (200, "bind", "B"),
        (400, "entry", "batch"), (410, "item", ("batch", 0)),
        (500, "bind", "A0"), (510, "bind", "A1"), (520, "bind", "C0"),
        (1000, "entry", "batch-0"), (1010, "item", ("batch-0", 0)),
        (1020, "bind", "D"),
    ]  # fmt: skip
    found = replay_frontend(
        rig,
        actions,
        {
            "B": (205, "cmpl-batch-0"),
            "A0": (505, "cmpl-batch-0"),
            "A1": (515, "cmpl-batch-1"),
            "C0": (525, "cmpl-batch-0"),
            "D": (1025, "cmpl-batch-0-0"),
        },
    )
    ms = 1_000_000
    for tag, correct in {"B": 130, "A0": 100, "A1": 100, "C0": 400, "D": 1000}.items():
        assert found[tag]["api"]["arrival_ns"] in (None, correct * ms), tag
    assert found["A1"]["api"]["arrival_ns"] is None
    assert found["A1"]["stall"]["basis"] == "scheduler_queued"


def test_first_call_bound_before_the_third_enters_leaves_the_later_two_unassigned(
    rig, clock
):
    """The #70 schedule with the first call bound at 300 ms, before the third call
    enters at 400 ms: the second and third share ``batch-0`` and overlap."""
    actions = [
        (100, "entry", "batch"), (110, "item", ("batch", 0)),
        (200, "entry", "batch-0"), (210, "item", ("batch-0", 0)),
        (300, "bind", "P"),
        (400, "entry", "batch-0"), (410, "item", ("batch-0", 0)),
        (500, "bind", "Q"), (510, "bind", "R"),
    ]  # fmt: skip
    found = replay_frontend(
        rig,
        actions,
        {
            "P": (305, "cmpl-batch-0"),
            "Q": (505, "cmpl-batch-0-0"),
            "R": (515, "cmpl-batch-0-0"),
        },
    )
    ms = 1_000_000
    assert found["P"]["api"]["arrival_ns"] == 100 * ms
    for tag in "QR":  # either call could own either prompt
        assert found[tag]["api"]["arrival_ns"] is None, tag


# ---- the review of agent-3#73: a fuzz of the arrival lookup, entries going missing


FUZZ_SEEDS = 5  # each seed is 20,000 schedules (a wide run of 40 seeds passes too)


class _FuzzSource:
    """Just enough of a frontend capture for the bind index: its rows and name map."""

    index = 0

    def __init__(self, events):
        self.events = events

    @staticmethod
    def _name(value):
        return value


@pytest.mark.parametrize("seed", range(FUZZ_SEEDS))
def test_a_fuzz_of_reused_and_colliding_ids_with_entries_missing_never_misattributes(
    seed,
):
    """Random schedules of indexed and direct calls whose client IDs collide (a client's
    prompt ID can be another client's parent ID), their events interleaved arbitrarily,
    and entries missing as a capture that started late (or ended early) would lose them:
    the whole span of such a call lies before every other event (or after every other
    event). The lookup either leaves an engine request unassigned or assigns it exactly
    its own call's entry; a call whose entry is missing is never given one.

    Which of two calls an entry belongs to cannot be told when an entry-less call's
    events fall between that entry and its own prompts, so that layout is not generated:
    the lookup follows the order in which calls are recorded, entry first."""
    import random

    from vllm.kv_profiling.timeline import Timelines

    rng = random.Random(seed)
    clients = ["b", "b-0", "b-0-0", "b-0-0-0", "c"]
    checked = assigned = 0
    for case in range(20000):
        calls = []
        for number in range(rng.randint(2, 6)):
            client = rng.choice(clients)
            parent = f"cmpl-{client}"
            direct = rng.random() < 0.4
            prompts = 1 if direct else rng.randint(1, 2)
            events = [("entry", parent, f"base-{client}")]
            if not direct:
                events += [("item", parent, f"{parent}-{i}") for i in range(prompts)]
            engines = [
                (parent if direct else f"{parent}-{i}", f"e{number}-{i}")
                for i in range(prompts)
            ]
            rng.shuffle(engines)
            events += [("bind", external, engine) for external, engine in engines]
            where = rng.random()
            calls.append(
                {
                    "id": number,
                    "events": events,
                    "missing": where < 0.15,  # started before the capture did
                    "late": 0.15 <= where < 0.25,  # lost at the end of the capture
                }
            )

        def merged(group):
            cursor = {call["id"]: 0 for call in group}
            live, out = list(group), []
            while live:
                call = rng.choice(live)
                out.append((call, call["events"][cursor[call["id"]]]))
                cursor[call["id"]] += 1
                if cursor[call["id"]] >= len(call["events"]):
                    live.remove(call)
            return out

        early = [c for c in calls if c["missing"]]
        late = [c for c in calls if c["late"]]
        middle = [c for c in calls if not c["missing"] and not c["late"]]
        order = []
        time_of_entry = {}
        for call, (kind, a, b) in merged(early) + merged(middle) + merged(late):
            if kind == "entry" and (call["missing"] or call["late"]):
                continue  # the capture lacks this call's entry
            order.append((call["id"], kind, a, b))
        if (
            rng.random() < 0.3
        ):  # the capture ended early: the tail of the schedule is lost
            order = order[: rng.randint(len(order) // 2, len(order))]
        present = {b for _, kind, _, b in order if kind == "bind"}
        rows = []
        for time, (call_id, kind, a, b) in enumerate(order, start=100):
            if kind == "entry":
                time_of_entry[call_id] = time
                rows += [(time, 1, b, b), (time, 2, a, b)]
            elif kind == "item":
                rows.append((time, 3, b, a))
            else:
                rows.append((time, 4, b, a))
        events = [
            {
                "kind": "request_bind",
                "subject": subject,
                "request_ref": name,
                "handoff_ref": target,
                "time_ns": time,
            }
            for time, subject, name, target in rows
        ]
        matcher = Timelines.__new__(Timelines)
        matcher._edges = {}
        source = _FuzzSource(events)
        for call in calls:
            truth = time_of_entry.get(call["id"])
            for kind, _external, engine in call["events"]:
                if kind != "bind" or engine not in present:
                    continue
                checked += 1
                found, _ = matcher._arrivals(source, engine)
                if found:
                    assigned += 1
                    assert len(found) == 1 and found[0][1] == truth, (
                        f"case {case} engine {engine} found {found} truth {truth}\n"
                        + "\n".join(
                            f"call {c['id']} missing={c['missing']} late={c['late']}: "
                            f"{c['events']}"
                            for c in calls
                        )
                        + f"\norder {order}"
                    )
    assert checked > 40000 and assigned > 5000  # the fuzz is not vacuous


def test_a_later_indexed_call_whose_entry_was_not_recorded_still_leaves_the_fallback(
    rig, clock
):
    """The append test with the later call's API entry omitted (as if the capture had
    started just after it): the shared ID has three engine binds and only two recorded
    prompt declarations, so its binds cannot all be indexed prompts."""
    actions = [
        (100, "entry", "batch"), (110, "item", ("batch", 0)),
        (120, "item", ("batch", 1)), (130, "entry", "batch-0"), (200, "bind", "B"),
        (400, "entry", "batch"), (410, "item", ("batch", 0)),
        (500, "bind", "A0"), (510, "bind", "A1"), (520, "bind", "C0"),
        (1010, "item", ("batch-0", 0)), (1020, "bind", "D"),
    ]  # fmt: skip
    found = replay_frontend(
        rig,
        actions,
        {
            "B": (205, "cmpl-batch-0"),
            "A0": (505, "cmpl-batch-0"),
            "A1": (515, "cmpl-batch-1"),
            "C0": (525, "cmpl-batch-0"),
            "D": (1025, "cmpl-batch-0-0"),
        },
    )
    ms = 1_000_000
    for tag, correct in {"B": 130, "A0": 100, "A1": 100, "C0": 400}.items():
        assert found[tag]["api"]["arrival_ns"] in (None, correct * ms), tag
    assert found["A1"]["api"]["arrival_ns"] is None
    assert found["A1"]["stall"]["basis"] == "scheduler_queued"
    assert found["D"]["api"]["arrival_ns"] is None  # its entry was never recorded


def test_later_declaration_cannot_balance_an_earlier_unmatched_bind(rig, clock):
    """Two engine binds share an ID before its second prompt declaration."""
    shared = "cmpl-batch-0"
    actions = [
        (10, "item", ("batch-0", 0)),  # recording began after this call's entry
        (30, "bind", "D"),
        (100, "entry", "batch"),
        (105, "item", ("batch", 0)),
        (110, "item", ("batch", 1)),
        (150, "entry", "batch-0"),
        (200, "bind", "A0"),
        (250, "bind", "B"),
        (400, "entry", "batch"),
        (410, "item", ("batch", 0)),
        (500, "bind", "A1"),
        (600, "bind", "C0"),
        (1000, "entry", "batch"),
        (1010, "item", ("batch", 0)),  # capture ends before the engine bind
    ]
    found = replay_frontend(
        rig,
        actions,
        {
            "D": (35, "cmpl-batch-0-0"),
            "A0": (205, shared),
            "B": (255, shared),
            "A1": (505, "cmpl-batch-1"),
            "C0": (605, shared),
        },
    )
    assert found["A1"]["api"]["arrival_ns"] is None
    assert found["A1"]["stall"]["basis"] == "scheduler_queued"
