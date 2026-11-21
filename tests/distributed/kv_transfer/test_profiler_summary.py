# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real recorder and sink into the offline summary; no fabricated rows."""

import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from vllm.kv_profiling.config import NetworkProfilerConfig
from vllm.kv_profiling.core import ProcessRecorder
from vllm.kv_profiling.schema import EventKind, Failure, Transport
from vllm.kv_profiling.summary import (
    _COMPLETE,
    _POST_RETURN,
    _PREPARED,
    _REQUEST_KINDS,
    summarize,
)

C1 = os.environ.get("PROFILER_SUMMARY_C1")
POSITIVE = os.environ.get("PROFILER_SUMMARY_POSITIVE")


def fields(
    kind,
    request=None,
    *,
    context=0,
    handoff=None,
    peer=None,
    handle=None,
    size=None,
    selected=None,
    merged=None,
    aux0=None,
    aux1=None,
    transport=Transport.TCP,
    status=0,
    subject=0,
):
    """The sixteen positional fields of the native record call."""
    return (
        kind,
        context,
        request,
        handoff,
        peer,
        handle,
        size,
        selected,
        merged,
        None,
        None,
        aux0,
        aux1,
        transport,
        status,
        subject,
    )


@pytest.fixture
def recorders(tmp_path):
    resources = []

    def create(**overrides):
        options = dict(
            level="l1",
            run_id=str(uuid4()),
            output_dir=str(tmp_path),
            ring_records=128,
            id_slots=32,
            id_bytes=4096,
            handle_slots=16,
            shutdown_timeout_s=2,
            context={
                "role": "worker",
                "kv_role": "kv_consumer",
                "engine_id": "engine-decode",
                "backend": ["Test"],
                "transport": "tcp",
                "rank": 0,
            },
        )
        options.update(overrides)
        recorder = ProcessRecorder(NetworkProfilerConfig(**options))
        resources.append(recorder)
        return recorder

    yield create
    for recorder in resources:
        recorder.close()
        if recorder.ring_dir.exists():
            shutil.rmtree(recorder.ring_dir)


def rows_of(recorder):
    path = recorder.output_dir / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def generation_of(recorder, selected):
    """The native generation allocated for the bind with these descriptors."""
    return next(
        row["transfer"]
        for row in rows_of(recorder)
        if row.get("kind") == "transfer_bind"
        and row["selected_descriptors"] == selected
    )


def kinds_of(summary):
    """The problem kinds of a summary, apart from the adapter's health: the bare recorders of these tests have no adapter,
    hence no ``profiler.json`` (that case has its own test)."""
    return [p["kind"] for p in summary["problems"] if p["kind"] != "adapter_unhealthy"]


def by_transfer(summary, transfer):
    return next(
        item for item in summary["transfers"] if item["transfer"] == transfer
    )


def test_literal_kind_sets_match_the_schema():
    names = {item.name.lower() for item in EventKind}
    assert _PREPARED | _POST_RETURN | _COMPLETE | _REQUEST_KINDS <= names


def test_known_sequence_is_summarized_exactly(recorders):
    run_id = str(uuid4())
    recorder = recorders(run_id=run_id)
    assert recorder.wait_ready()
    record = recorder.record
    record(*fields(EventKind.HANDSHAKE_BEGIN, peer="prod-engine"))
    record(*fields(EventKind.REQUEST_BIND, request="req-c"))
    record(
        *fields(
            EventKind.TRANSFER_BIND,
            "req-c",
            handoff="req-p",
            peer="prod-engine",
            handle=5,
            selected=10,
            aux0=0,
            subject=0,
        )
    )
    record(
        *fields(
            EventKind.TRANSFER_BIND,
            "req-c",
            handoff="req-p",
            peer="prod-engine",
            handle=6,
            selected=20,
            aux0=0,
            subject=0,
        )
    )
    record(*fields(EventKind.POST_RETURN, handle=5))
    record(*fields(EventKind.POST_RETURN, handle=6))
    record(
        *fields(
            EventKind.COMPLETE_OBSERVED,
            handle=5,
            size=1000,
            merged=10,
            aux0=100,
            aux1=200,
            subject=1,
        )
    )
    record(
        *fields(
            EventKind.FAILED,
            handle=6,
            status=int(Failure.TRANSFER_FAILED),
            subject=1,
        )
    )
    record(*fields(EventKind.HANDSHAKE_END, peer="prod-engine"))
    record(*fields(EventKind.RELEASE_REQUESTED, "req-c", subject=2))
    record(*fields(EventKind.TRANSFER_RETIRE, handle=5))
    result = recorder.close()
    assert result.saved and result.complete
    # An orphan post of an unknown generation is recorded loss-free on its
    # own recorder; the summary reports it, and never summarizes less.
    orphan = recorders(run_id=run_id)
    assert orphan.wait_ready()
    orphan.record(*fields(EventKind.POST_RETURN, handle=99))
    orphan_result = orphan.close()
    assert orphan_result.saved and not orphan_result.complete

    def time(kind, transfer=None):
        return next(
            row["time_ns"]
            for row in rows_of(recorder)
            if row.get("kind") == kind
            and (transfer is None or row["transfer"] == transfer)
        )

    summary = summarize([recorder.output_dir, orphan.output_dir])
    # The orphan post's handle was never bound, so the recorder drops the
    # generation (loss) and the summary reports the event it cannot pair.
    summary_problems = set(kinds_of(summary))
    assert summary_problems == {"capture_loss", "orphan_transfer_event"}
    assert len(summary["transfers"]) == 2
    # The recorder maps local handles to sequential native generations.
    gen_done = generation_of(recorder, 10)
    gen_failed = generation_of(recorder, 20)
    done = by_transfer(summary, gen_done)
    failed = by_transfer(summary, gen_failed)
    assert done["state"] == "completed"
    assert done["bytes"] == 1000
    assert done["selected_descriptors"] == 10
    assert done["merged_descriptors"] == 10
    assert done["nixl_post_us"] == 100
    assert done["nixl_xfer_us"] == 200
    assert done["prepared_to_posted_ns"] == (
        time("post_return", gen_done) - time("transfer_bind", gen_done)
    )
    assert done["posted_to_completed_ns"] == (
        time("complete_observed", gen_done) - time("post_return", gen_done)
    )
    assert done["throughput_bytes_per_s"] == (
        1000 * 1_000_000_000 / done["posted_to_completed_ns"]
    )
    assert done["handshake_duration_ns"] == (
        time("handshake_end") - time("handshake_begin")
    )
    assert done["retired_time_ns"] == time("transfer_retire", gen_done)
    assert failed["state"] == "failed"
    assert failed["failure_code"] == "TRANSFER_FAILED"
    assert failed["posted_to_completed_ns"] is None
    assert failed["throughput_bytes_per_s"] is None
    assert len(summary["requests"]) == 1
    request = summary["requests"][0]
    assert request["request"] == "req-c"
    # One completed READ does not complete a request whose other READ failed.
    assert request["state"] == "failed"
    assert request["post_completion_failures"] == []
    assert request["transfers"] == 2
    assert request["request_occurrence"] == 0
    assert request["bytes"] == 1000 and request["bytes_samples"] == 1
    assert request["duration_sum_ns"] == (
        (time("complete_observed", gen_done)
         - time("transfer_bind", gen_done))
        + (time("failed", gen_failed) - time("transfer_bind", gen_failed))
    )
    assert request["duration_sum_samples"] == 2
    assert request["release_requested_time_ns"] == time("release_requested")
    assert request["failures"] == []
    key = summary["aggregates"]["transfers"]["Test/tcp"]
    assert key["count"] == 2 and key["completed"] == 1 and key["failed"] == 1
    assert key["incomplete"] == 0
    assert key["bytes"] == {"sum": 1000, "samples": 1}
    assert key["posted_to_completed_ns"]["samples"] == 1
    assert key["posted_to_completed_ns"]["p50"] == done[
        "posted_to_completed_ns"
    ]
    handshake = summary["aggregates"]["handshakes"]["Test/tcp"]
    assert handshake["count"] == 1
    assert handshake["duration_ns"]["p50"] == done["handshake_duration_ns"]
    scopes = [item["scope"] for item in summary["not_available"]]
    assert "per transfer" in scopes and "per request" in scopes


def test_multi_rank_handshake_and_request_edges(recorders):
    run_id = str(uuid4())
    consumer = recorders(
        run_id=run_id,
        context={
            "role": "scheduler",
            "kv_role": "kv_consumer",
            "engine_id": "engine-decode",
            "backend": ["Test"],
            "transport": "tcp",
            "rank": 0,
        }
    )
    producer_a = recorders(
        run_id=run_id,
        context={
            "role": "scheduler",
            "kv_role": "kv_producer",
            "engine_id": "prod",
            "backend": ["Test"],
            "transport": "tcp",
            "rank": 0,
        },
    )
    producer_b = recorders(
        run_id=run_id,
        context={
            "role": "scheduler",
            "kv_role": "kv_producer",
            "engine_id": "prod",
            "backend": ["Test"],
            "transport": "tcp",
            "rank": 1,
        }
    )
    for recorder in (consumer, producer_a, producer_b):
        assert recorder.wait_ready()
    record = consumer.record
    record(*fields(EventKind.HANDSHAKE_BEGIN, peer="prod"))
    # Rank 1 answers first with a distinct remote rank; the rank-0 end is a
    # separate rank end of the same attempt, and the attempt counts once.
    record(*fields(EventKind.HANDSHAKE_END, peer="prod", aux0=0, subject=1))
    record(*fields(EventKind.HANDSHAKE_END, peer="prod", aux0=0, subject=0))
    record(*fields(EventKind.REQUEST_BIND, request="req-c"))
    record(*fields(EventKind.HANDOFF, "req-c", handoff="req-p", peer="prod"))
    record(
        *fields(
            EventKind.TRANSFER_BIND,
            "req-c",
            handoff="req-p",
            peer="prod",
            handle=10,
            selected=4,
            subject=0,
        )
    )
    record(*fields(EventKind.POST_RETURN, handle=10))
    record(
        *fields(EventKind.COMPLETE_OBSERVED, handle=10, size=64, subject=1)
    )
    record(
        *fields(
            EventKind.TRANSFER_BIND,
            "req-c",
            handoff="req-p",
            peer="prod",
            handle=11,
            selected=5,
            subject=1,
        )
    )
    record(*fields(EventKind.POST_RETURN, handle=11))
    record(
        *fields(
            EventKind.COMPLETE_OBSERVED, handle=11, size=32, subject=1
        )
    )
    record(
        *fields(
            EventKind.FAILED,
            "req-c",
            status=int(Failure.LATE_NOTIFICATION),
            subject=0,
        )
    )
    # A second READ of this occurrence is still running: the request stays
    # incomplete until every transfer finishes.
    record(*fields(EventKind.REQUEST_BIND, request="req-e"))
    record(
        *fields(
            EventKind.TRANSFER_BIND,
            "req-e",
            handoff="req-p",
            peer="prod",
            handle=12,
            selected=6,
            subject=0,
        )
    )
    record(*fields(EventKind.POST_RETURN, handle=12))
    # Repeated ids stay separate occurrences, each carrying its own READ.
    record(*fields(EventKind.REQUEST_BIND, request="req-d"))
    record(
        *fields(
            EventKind.TRANSFER_BIND,
            "req-d",
            handoff="req-p",
            peer="prod",
            handle=20,
            selected=7,
            subject=0,
        )
    )
    record(*fields(EventKind.POST_RETURN, handle=20))
    record(
        *fields(EventKind.COMPLETE_OBSERVED, handle=20, size=70, subject=1)
    )
    record(*fields(EventKind.REQUEST_BIND, request="req-d"))
    record(
        *fields(
            EventKind.TRANSFER_BIND,
            "req-d",
            handoff="req-p",
            peer="prod",
            handle=21,
            selected=8,
            subject=0,
        )
    )
    record(*fields(EventKind.POST_RETURN, handle=21))
    record(
        *fields(EventKind.COMPLETE_OBSERVED, handle=21, size=80, subject=1)
    )
    # One producer id recorded by two processes of one engine: the join is
    # ambiguous and stays unavailable rather than guessed.
    for producer in (producer_a, producer_b):
        producer.record(*fields(EventKind.REQUEST_BIND, request="req-p"))
        producer.record(
            *fields(EventKind.HANDOFF, "req-p", peer="prod",
                    aux0=1, aux1=2, subject=1)
        )
    assert producer_a.close().saved and producer_b.close().saved
    result = consumer.close()
    assert result.saved
    summary = summarize(
        [producer_a.output_dir, producer_b.output_dir, consumer.output_dir]
    )
    handshakes = summary["aggregates"]["handshakes"]["Test/tcp"]
    assert handshakes["count"] == 1
    (consumer_handshake,) = [
        item
        for item in summary["handshakes"]
        if item["state"] == "completed"
    ]
    assert consumer_handshake["peer"] == "prod"
    ranks = {end["subject"] for end in consumer_handshake["ends"]}
    assert ranks == {0, 1}
    by_rank = {item["source_rank"]: item for item in summary["transfers"]
               if item["request"] == "req-c"}
    assert by_rank[0]["state"] == "completed"
    assert by_rank[0]["handshake_duration_ns"] is not None
    assert by_rank[1]["state"] == "completed"
    assert by_rank[1]["handshake_duration_ns"] is not None
    aggregate = summary["aggregates"]["transfers"]["Test/tcp"]
    assert aggregate["count"] == 5 and aggregate["completed"] == 4
    assert aggregate["incomplete"] == 1
    request = next(
        item
        for item in summary["requests"]
        if item["request"] == "req-c"
    )
    assert request["state"] == "completed"
    assert request["transfers"] == 2
    assert [item["code"] for item in request["failures"]] == [
        "LATE_NOTIFICATION"
    ]
    # The late-notification error is cleanup after all movement completed.
    assert request["post_completion_failures"] == request["failures"]
    incomplete = next(
        item for item in summary["requests"] if item["request"] == "req-e"
    )
    assert incomplete["state"] == "incomplete"
    assert incomplete["transfers"] == 1
    assert incomplete["post_completion_failures"] == []
    (join,) = summary["joins"]
    assert join["matches"] == 2 and join["ambiguous"]
    assert not join["joined"]
    reasons = [item["reason"] for item in summary["not_available"]]
    assert any("ambiguous" in reason for reason in reasons)
    # The two req-d occurrences carry exactly their own READ each: sizes,
    # counts and first posts are never pooled across occurrences.
    occurrences = [
        item for item in summary["requests"] if item["request"] == "req-d"
    ]
    assert [item["request_occurrence"] for item in occurrences] == [0, 1]
    assert [item["transfers"] for item in occurrences] == [1, 1]
    assert [item["bytes"] for item in occurrences] == [70, 80]
    assert all(item["state"] == "completed" for item in occurrences)
    assert len(
        {item["first_posted_time_ns"] for item in occurrences}
    ) == 2
    # Without calibration no cross-process span is invented; this join is
    # ambiguous, so nothing was joinable in the first place.
    assert summary["cross_process_spans"] == []


def test_handshake_attempts_by_role(recorders):
    """Served replies belong to schedulers; worker ends without a begin are
    problems, and repeated begins are separate attempts that each count."""
    run_id = str(uuid4())
    worker = recorders(
        run_id=run_id,
        context={
            "role": "worker",
            "kv_role": "kv_consumer",
            "engine_id": "engine-decode",
            "backend": ["Test"],
            "transport": "tcp",
            "rank": 0,
        }
    )
    scheduler = recorders(
        run_id=run_id,
        context={
            "role": "scheduler",
            "kv_role": "kv_producer",
            "engine_id": "prod",
            "backend": ["Test"],
            "transport": "tcp",
            "rank": 0,
        }
    )
    assert worker.wait_ready() and scheduler.wait_ready()
    # A worker peer end before any begin: no local begin, no duration.
    worker.record(*fields(EventKind.HANDSHAKE_END, peer="p", subject=0))
    worker.record(*fields(EventKind.HANDSHAKE_BEGIN, peer="p"))
    worker.record(*fields(EventKind.HANDSHAKE_END, peer="p", subject=0))
    worker.record(*fields(EventKind.HANDSHAKE_BEGIN, peer="p"))
    worker.record(*fields(EventKind.HANDSHAKE_END, peer="p", subject=0))
    assert worker.close().saved
    # The producer scheduler's served reply has no local begin by design.
    scheduler.record(*fields(EventKind.HANDSHAKE_END, peer="engine-decode"))
    assert scheduler.close().saved
    summary = summarize([worker.output_dir, scheduler.output_dir])
    kinds = set(kinds_of(summary))
    assert kinds == {"missing_handshake_begin"}
    states = [(item["peer"], item["state"]) for item in summary["handshakes"]]
    completed = [
        item for item in summary["handshakes"] if item["state"] == "completed"
    ]
    assert ("engine-decode", "served_reply") in states
    assert ("p", "no_local_begin") not in states
    assert len(completed) == 2
    assert {item["peer"] for item in completed} == {"p"}
    aggregate = summary["aggregates"]["handshakes"]["Test/tcp"]
    # The two attempts are two handshakes; each its own begin/end interval.
    assert aggregate["count"] == 2
    assert aggregate["duration_ns"]["samples"] == 2


def test_setup_failure_and_aggregate_after_failure(recorders):
    """A required setup failure keeps the request failed even when every
    created handle completes, and receive-finished cannot mark a failed
    request successful."""
    run_id = str(uuid4())
    worker = recorders(
        run_id=run_id,
        context={
            "role": "worker",
            "kv_role": "kv_consumer",
            "engine_id": "engine-decode",
            "backend": ["Test"],
            "transport": "tcp",
            "rank": 0,
        },
    )
    scheduler = recorders(
        run_id=run_id,
        context={
            "role": "scheduler",
            "kv_role": "kv_consumer",
            "engine_id": "engine-decode",
            "backend": ["Test"],
            "transport": "tcp",
            "rank": 0,
        },
    )
    assert worker.wait_ready() and scheduler.wait_ready()
    record = worker.record
    record(*fields(EventKind.REQUEST_BIND, request="req-s"))
    record(
        *fields(
            EventKind.TRANSFER_BIND,
            "req-s",
            handoff="req-p",
            peer="prod",
            handle=1,
            selected=4,
            subject=0,
        )
    )
    record(*fields(EventKind.POST_RETURN, handle=1))
    # A sibling READ fails setup before the running READ completes: the
    # required work of this request never fully succeeded.
    record(
        *fields(
            EventKind.FAILED,
            "req-s",
            status=int(Failure.TRANSFER_SETUP_FAILED),
            subject=0,
        )
    )
    record(
        *fields(EventKind.COMPLETE_OBSERVED, handle=1, size=10, subject=1)
    )
    assert worker.close().saved
    # A receive that failed, followed by the terminal aggregate event.
    scheduler.record(*fields(EventKind.REQUEST_BIND, request="req-f"))
    scheduler.record(
        *fields(
            EventKind.FAILED,
            "req-f",
            status=int(Failure.TRANSFER_FAILED),
            subject=0,
        )
    )
    scheduler.record(
        *fields(EventKind.COMPLETE_OBSERVED, request="req-f", subject=2)
    )
    assert scheduler.close().saved
    summary = summarize([worker.output_dir, scheduler.output_dir])
    requests = {
        item["request"]: item for item in summary["requests"]
    }
    assert requests["req-s"]["state"] == "failed"
    assert [
        item["code"] for item in requests["req-s"]["failures"]
    ] == ["TRANSFER_SETUP_FAILED"]
    assert requests["req-s"]["post_completion_failures"] == []
    assert requests["req-s"]["transfers"] == 1
    # The aggregate receive-finished event is kept with its time, but it
    # does not turn the failed request successful.
    assert requests["req-f"]["state"] == "failed"
    assert requests["req-f"]["recv_finished_time_ns"] is not None
    assert requests["req-f"]["post_completion_failures"] == []


def test_span_selects_by_projected_order_and_consumer_identity(
    recorders, tmp_path
):
    """The earliest READ is chosen after projection, and only among READs
    of the consumer's own engine and request."""
    from vllm.kv_profiling.clock import ClockMapping, ClockSample

    run_id = str(uuid4())

    def engine(engine_id, role="worker", kv="kv_consumer", rank=0):
        return recorders(
            run_id=run_id,
            context={
                "role": role,
                "kv_role": kv,
                "engine_id": engine_id,
                "backend": ["Test"],
                "transport": "tcp",
                "rank": rank,
            },
        )

    producer = engine("prod", role="scheduler", kv="kv_producer")
    scheduler = engine("engine-decode", role="scheduler")
    unrelated = engine("other-engine")
    worker_a = engine("engine-decode", rank=0)
    worker_b = engine("engine-decode", rank=1)
    for recorder in (
        producer, scheduler, unrelated, worker_a, worker_b,
    ):
        assert recorder.wait_ready()
    producer.record(*fields(EventKind.REQUEST_BIND, request="req-p"))
    producer.record(
        *fields(EventKind.HANDOFF, "req-p", peer="prod",
                aux0=1, aux1=2, subject=1)
    )
    assert producer.close().saved
    # An unrelated consumer engine records the earliest raw READ of the
    # same producer identity: it must not be selected for the span.
    unrelated.record(*fields(EventKind.REQUEST_BIND, request="req-c"))
    unrelated.record(
        *fields(
            EventKind.TRANSFER_BIND,
            "req-c",
            handoff="req-p",
            peer="prod",
            handle=30,
            selected=9,
            subject=0,
        )
    )
    unrelated.record(*fields(EventKind.POST_RETURN, handle=30))
    unrelated.record(
        *fields(EventKind.COMPLETE_OBSERVED, handle=30, size=5, subject=1)
    )
    assert unrelated.close().saved
    scheduler.record(*fields(EventKind.REQUEST_BIND, request="req-c"))
    scheduler.record(
        *fields(EventKind.HANDOFF, "req-c", handoff="req-p", peer="prod")
    )
    assert scheduler.close().saved
    for recorder, handle, selected, size in (
        (worker_a, 1, 1, 10),
        (worker_b, 2, 2, 20),
    ):
        recorder.record(*fields(EventKind.REQUEST_BIND, request="req-c"))
        recorder.record(
            *fields(
                EventKind.TRANSFER_BIND,
                "req-c",
                handoff="req-p",
                peer="prod",
                handle=handle,
                selected=selected,
                subject=0,
            )
        )
        recorder.record(*fields(EventKind.POST_RETURN, handle=handle))
        recorder.record(
            *fields(
                EventKind.COMPLETE_OBSERVED,
                handle=handle,
                size=size,
                subject=1,
            )
        )
    assert worker_a.close().saved and worker_b.close().saved

    def clock_of(recorder):
        (meta,) = [
            row for row in rows_of(recorder) if row.get("type") == "metadata"
        ]
        return meta["clock_id"]

    reference = clock_of(producer)
    # worker_a projects by identity (raw RTT 1000); worker_b's clock runs
    # far behind, so its small raw values project later than worker_a's.
    samples = {
        worker_a: (1000, 1000, 2000, 2000),
        worker_b: (0, 0, 0, 1_000_000),
    }
    files = []
    for recorder, times in samples.items():
        mapping = ClockMapping.from_samples(
            [
                ClockSample(
                    run_id=run_id,
                    local_generation=reference,
                    remote_generation=clock_of(recorder),
                    t1=times[0],
                    t2=times[1],
                    t3=times[2],
                    t4=times[3],
                )
            ]
        )
        path = tmp_path / f"calibration-{len(files)}.json"
        path.write_text(json.dumps(mapping.to_json()))
        files.append(str(path))
    summary = summarize(
        [
            producer.output_dir,
            scheduler.output_dir,
            unrelated.output_dir,
            worker_a.output_dir,
            worker_b.output_dir,
        ],
        calibration=files,
    )
    (span,) = summary["cross_process_spans"]
    # worker_b has the earliest raw post value, but worker_a is earliest
    # after projection, and the unrelated engine is excluded.
    assert span["post_source"] == 3
    assert span["uncertainty_ns"] == 1000

    def only_time(recorder, kind):
        (event,) = [
            row for row in rows_of(recorder) if row.get("kind") == kind
        ]
        return event["time_ns"]

    assert span["duration_ns"] == (
        only_time(worker_a, "post_return")
        - only_time(producer, "handoff")
    )
    (join,) = summary["joins"]
    assert join["joined"] and not join["ambiguous"]


def test_one_incoming_handoff_with_two_producer_occurrences(recorders):
    """Occurrence ordinals are local, not correlation evidence: with one
    consumer incoming handoff and two producer outgoing handoffs of the
    same recorded identity, the join stays ambiguous and no span is made."""
    run_id = str(uuid4())

    producer = recorders(
        run_id=run_id,
        context={
            "role": "scheduler",
            "kv_role": "kv_producer",
            "engine_id": "prod",
            "backend": ["Test"],
            "transport": "tcp",
            "rank": 0,
        },
    )
    consumer = recorders(
        run_id=run_id,
        context={
            "role": "scheduler",
            "kv_role": "kv_consumer",
            "engine_id": "engine-decode",
            "backend": ["Test"],
            "transport": "tcp",
            "rank": 0,
        },
    )
    assert producer.wait_ready() and consumer.wait_ready()
    producer.record(*fields(EventKind.REQUEST_BIND, request="req-p"))
    producer.record(
        *fields(EventKind.HANDOFF, "req-p", peer="prod",
                aux0=1, aux1=2, subject=1)
    )
    producer.record(
        *fields(EventKind.HANDOFF, "req-p", peer="prod",
                aux0=2, aux1=3, subject=1)
    )
    assert producer.close().saved
    consumer.record(*fields(EventKind.REQUEST_BIND, request="req-c"))
    consumer.record(*fields(EventKind.REQUEST_BIND, request="req-c"))
    consumer.record(
        *fields(EventKind.HANDOFF, "req-c", handoff="req-p", peer="prod")
    )
    assert consumer.close().saved
    summary = summarize([producer.output_dir, consumer.output_dir])
    (join,) = summary["joins"]
    assert join["matches"] == 2 and join["ambiguous"]
    assert not join["joined"]
    assert summary["cross_process_spans"] == []
    reasons = [item["reason"] for item in summary["not_available"]]
    assert any("ambiguous" in reason for reason in reasons)


def test_setup_failure_after_completed_read_stays_failed(recorders):
    """A setup failure recorded after a finished READ is still required
    work that failed: timestamps never rebrand it as cleanup."""
    recorder = recorders()
    assert recorder.wait_ready()
    record = recorder.record
    record(*fields(EventKind.REQUEST_BIND, request="req-s"))
    record(
        *fields(
            EventKind.TRANSFER_BIND,
            "req-s",
            handoff="req-p",
            peer="prod",
            handle=1,
            selected=4,
            subject=0,
        )
    )
    record(*fields(EventKind.POST_RETURN, handle=1))
    record(
        *fields(EventKind.COMPLETE_OBSERVED, handle=1, size=10, subject=1)
    )
    record(
        *fields(
            EventKind.FAILED,
            "req-s",
            status=int(Failure.TRANSFER_SETUP_FAILED),
            subject=0,
        )
    )
    assert recorder.close().saved
    summary = summarize([recorder.output_dir])
    (request,) = summary["requests"]
    assert request["state"] == "failed"
    assert [item["code"] for item in request["failures"]] == [
        "TRANSFER_SETUP_FAILED"
    ]
    assert request["post_completion_failures"] == []
    # A real late-notification error after proven movement stays cleanup.
    recorder2 = recorders()
    assert recorder2.wait_ready()
    recorder2.record(*fields(EventKind.REQUEST_BIND, request="req-l"))
    recorder2.record(
        *fields(
            EventKind.TRANSFER_BIND,
            "req-l",
            handoff="req-p",
            peer="prod",
            handle=1,
            selected=4,
            subject=0,
        )
    )
    recorder2.record(*fields(EventKind.POST_RETURN, handle=1))
    recorder2.record(
        *fields(EventKind.COMPLETE_OBSERVED, handle=1, size=10, subject=1)
    )
    recorder2.record(
        *fields(
            EventKind.FAILED,
            "req-l",
            status=int(Failure.LATE_NOTIFICATION),
            subject=0,
        )
    )
    assert recorder2.close().saved
    summary = summarize([recorder2.output_dir])
    (request,) = summary["requests"]
    assert request["state"] == "completed"
    assert request["post_completion_failures"] == request["failures"]


def test_worker_occurrence_ambiguity_and_missing_engine(recorders, tmp_path):
    """Two worker occurrences of one recorded identity stop the span, and a
    consumer without its engine identity can never wildcard one."""
    from vllm.kv_profiling.clock import ClockMapping, ClockSample

    run_id = str(uuid4())
    producer = recorders(
        run_id=run_id,
        context={
            "role": "scheduler",
            "kv_role": "kv_producer",
            "engine_id": "prod",
            "backend": ["Test"],
            "transport": "tcp",
            "rank": 0,
        },
    )
    scheduler = recorders(
        run_id=run_id,
        context={
            "role": "scheduler",
            "kv_role": "kv_consumer",
            "engine_id": "engine-decode",
            "backend": ["Test"],
            "transport": "tcp",
            "rank": 0,
        },
    )
    engineless = recorders(
        run_id=run_id,
        context={
            "role": "scheduler",
            "kv_role": "kv_consumer",
            "backend": ["Test"],
            "transport": "tcp",
            "rank": 0,
        },
    )
    worker = recorders(
        run_id=run_id,
        context={
            "role": "worker",
            "kv_role": "kv_consumer",
            "engine_id": "engine-decode",
            "backend": ["Test"],
            "transport": "tcp",
            "rank": 0,
        },
    )
    for recorder in (producer, scheduler, engineless, worker):
        assert recorder.wait_ready()
    producer.record(*fields(EventKind.REQUEST_BIND, request="req-p"))
    producer.record(
        *fields(EventKind.HANDOFF, "req-p", peer="prod",
                aux0=1, aux1=2, subject=1)
    )
    assert producer.close().saved
    scheduler.record(*fields(EventKind.REQUEST_BIND, request="req-c"))
    scheduler.record(
        *fields(EventKind.HANDOFF, "req-c", handoff="req-p", peer="prod")
    )
    assert scheduler.close().saved
    engineless.record(*fields(EventKind.REQUEST_BIND, request="req-c"))
    engineless.record(
        *fields(EventKind.HANDOFF, "req-c", handoff="req-p", peer="prod")
    )
    assert engineless.close().saved
    record = worker.record
    handle = 1
    for occurrence in range(2):
        record(*fields(EventKind.REQUEST_BIND, request="req-c"))
        record(
            *fields(
                EventKind.TRANSFER_BIND,
                "req-c",
                handoff="req-p",
                peer="prod",
                handle=handle,
                selected=4 + occurrence,
                subject=0,
            )
        )
        record(*fields(EventKind.POST_RETURN, handle=handle))
        record(
            *fields(
                EventKind.COMPLETE_OBSERVED,
                handle=handle,
                size=10 + occurrence,
                subject=1,
            )
        )
        handle += 1
    assert worker.close().saved

    (meta,) = [
        row for row in rows_of(worker) if row.get("type") == "metadata"
    ]
    (producer_meta,) = [
        row for row in rows_of(producer) if row.get("type") == "metadata"
    ]
    mapping = ClockMapping.from_samples(
        [
            ClockSample(
                run_id=run_id,
                local_generation=producer_meta["clock_id"],
                remote_generation=meta["clock_id"],
                t1=1000,
                t2=1000,
                t3=2000,
                t4=2000,
            )
        ]
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(mapping.to_json()))
    ambiguous = summarize(
        [producer.output_dir, scheduler.output_dir, worker.output_dir],
        calibration=[str(calibration)],
    )
    (join,) = ambiguous["joins"]
    assert join["joined"] and not join["ambiguous"]
    # Both worker occurrences carry the identity; no ordinal can select.
    assert ambiguous["cross_process_spans"] == []
    reasons = [item["reason"] for item in ambiguous["not_available"]]
    assert any(
        "more than one request occurrence" in reason for reason in reasons
    )
    engineless_summary = summarize(
        [producer.output_dir, engineless.output_dir, worker.output_dir],
        calibration=[str(calibration)],
    )
    assert engineless_summary["cross_process_spans"] == []
    reasons = [item["reason"] for item in engineless_summary["not_available"]]
    assert any(
        "consumer engine identity is not recorded" in reason
        for reason in reasons
    )


def test_calibrated_cross_process_span(recorders, tmp_path):
    """Handoff to first READ posted over a saved clock mapping.

    The real topology: the incoming handoff is in the consumer's scheduler,
    the READ post in the consumer's worker; the span must resolve the
    worker's READ by the transfer's own recorded identity and project the
    worker's clock.
    """
    from vllm.kv_profiling.clock import ClockMapping, ClockSample

    run_id = str(uuid4())

    def scaler(role, kv_role, engine, rank):
        return recorders(
            run_id=run_id,
            context={
                "role": role,
                "kv_role": kv_role,
                "engine_id": engine,
                "backend": ["Test"],
                "transport": "tcp",
                "rank": rank,
            },
        )

    producer_scheduler = scaler("scheduler", "kv_producer", "prod", 0)
    consumer_scheduler = scaler(
        "scheduler", "kv_consumer", "engine-decode", 0
    )
    consumer_worker = scaler("worker", "kv_consumer", "engine-decode", 0)
    producer_scheduler.record(*fields(EventKind.REQUEST_BIND, "req-p"))
    producer_scheduler.record(
        *fields(EventKind.HANDOFF, "req-p", peer="prod",
                aux0=1, aux1=2, subject=1)
    )
    assert producer_scheduler.close().saved
    record = consumer_worker.record
    record(*fields(EventKind.REQUEST_BIND, "req-c"))
    record(*fields(EventKind.HANDSHAKE_BEGIN, peer="prod"))
    record(*fields(EventKind.HANDSHAKE_END, peer="prod", aux0=0, subject=0))
    record(
        *fields(
            EventKind.TRANSFER_BIND,
            "req-c",
            handoff="req-p",
            peer="prod",
            handle=3,
            selected=4,
            subject=0,
        )
    )
    record(*fields(EventKind.POST_RETURN, handle=3))
    record(
        *fields(EventKind.COMPLETE_OBSERVED, handle=3, size=64, subject=1)
    )
    assert consumer_worker.close().saved
    consumer_scheduler.record(*fields(EventKind.REQUEST_BIND, "req-c"))
    consumer_scheduler.record(
        *fields(EventKind.HANDOFF, "req-c", handoff="req-p", peer="prod")
    )
    assert consumer_scheduler.close().saved

    def clock_of(recorder):
        (meta,) = [
            row for row in rows_of(recorder) if row.get("type") == "metadata"
        ]
        return meta["clock_id"]

    reference = clock_of(producer_scheduler)
    # Offset-zero mappings of both consumer clocks onto the producer's, in
    # the exact to/from-JSON shape a calibration run saves.
    mappings = []
    for recorder in (consumer_scheduler, consumer_worker):
        mappings.append(
            ClockMapping.from_samples(
                [
                    ClockSample(
                        run_id=run_id,
                        local_generation=reference,
                        remote_generation=clock_of(recorder),
                        t1=1000,
                        t2=1000,
                        t3=2000,
                        t4=2000,
                    )
                ]
            )
        )
    calibration_files = []
    for index, mapping in enumerate(mappings):
        path = tmp_path / f"calibration-{index}.json"
        path.write_text(json.dumps(mapping.to_json()))
        calibration_files.append(str(path))
    summary = summarize(
        [
            producer_scheduler.output_dir,
            consumer_scheduler.output_dir,
            consumer_worker.output_dir,
        ],
        calibration=calibration_files,
    )
    (span,) = summary["cross_process_spans"]
    assert span["kind"] == "handoff_to_first_read_posted"
    # The post source is the worker (input index 2), not the scheduler that
    # recorded the incoming handoff.
    assert span["post_source"] == 2

    def only_time(recorder, kind):
        (event,) = [
            row for row in rows_of(recorder) if row.get("kind") == kind
        ]
        return event["time_ns"]

    assert span["duration_ns"] == (
        only_time(consumer_worker, "post_return")
        - only_time(producer_scheduler, "handoff")
    )
    # Each consumer projection contributes its raw RTT; the producer
    # reference contributes none.
    assert span["uncertainty_ns"] == 1000
    assert span["drift_known"] is False
    assert span["model"] == "linear_estimate_not_a_bound"
    (join,) = summary["joins"]
    assert join["joined"] and not join["ambiguous"]
    # A calibration saved for another run must be refused.
    foreign_run = str(uuid4())
    foreign_files = []
    for index, mapping in enumerate(mappings):
        document = json.loads(json.dumps(mapping.to_json()))
        document["start"][0]["run_id"] = foreign_run
        path = tmp_path / f"foreign-{index}.json"
        path.write_text(json.dumps(document))
        foreign_files.append(str(path))
    with pytest.raises(ValueError, match="calibration belongs to run"):
        summarize(
            [
                producer_scheduler.output_dir,
                consumer_scheduler.output_dir,
                consumer_worker.output_dir,
            ],
            calibration=foreign_files,
        )


def test_duplicate_inputs_fold_and_conflicts_reject(recorders):
    run_id = str(uuid4())
    recorder = recorders(run_id=run_id)
    assert recorder.wait_ready()
    recorder.record(*fields(EventKind.REQUEST_BIND, request="req-c"))
    recorder.close()
    events = recorder.output_dir / "events.jsonl"
    copied = recorder.output_dir / "copied-events.jsonl"
    copied.write_text(events.read_text())
    once = summarize([recorder.output_dir])
    twice = summarize([recorder.output_dir, copied, recorder.output_dir])
    assert len(once["sources"]) == len(twice["sources"]) == 1
    assert len(twice["transfers"]) == 0
    assert twice["duplicate_inputs"]
    edited = recorder.output_dir / "edited-events.jsonl"
    edited.write_text(events.read_text() + events.read_text())
    with pytest.raises(ValueError, match="conflicting duplicate"):
        summarize([copied, edited])


def test_loss_and_run_rules(recorders, tmp_path):
    recorder = recorders()
    assert recorder.wait_ready()
    recorder.record(*fields(EventKind.REQUEST_BIND, request="req-c"))
    recorder.close()
    assert not summarize([recorder.output_dir])["sources"][0]["loss"]
    path = recorder.output_dir / "events.jsonl"
    kept = [
        line for line in path.read_text().splitlines() if '"type":"end"'
        not in line
    ]
    truncated = tmp_path / "truncated"
    truncated.write_text("\n".join(kept) + "\n")
    summary = summarize([truncated])
    assert summary["sources"][0]["loss"]
    assert not summary["sources"][0]["loss_detail"]["end_present"]
    assert kinds_of(summary) == ["capture_loss"]
    other = recorders()
    assert other.wait_ready()
    other.record(*fields(EventKind.REQUEST_BIND, request="req-c"))
    other.close()
    with pytest.raises(ValueError, match="different runs"):
        summarize([recorder.output_dir, other.output_dir])


@pytest.mark.skipif(not C1 or not Path(C1).exists(), reason="capture missing")
def test_incomplete_capture_regression():
    """The stage 3 recorders never closed: report loss, keep the bindings."""
    summary = summarize([C1])
    closed = [s for s in summary["sources"] if s["role"] == "frontend"]
    unclosed = [s for s in summary["sources"] if s["role"] != "frontend"]
    assert len(closed) == 2
    assert len(unclosed) == 4
    assert all(not source["loss"] for source in closed)
    assert all(source["loss"] for source in unclosed)
    assert summary["transfers"] == []
    assert summary["aggregates"]["transfers"] == {}
    # The 48 flushed frontend bindings are retained, never discarded.
    assert len(summary["requests"]) == 48
    assert all(item["bytes"] is None for item in summary["requests"])
    assert {item["kind"] for item in summary["problems"]} == {"capture_loss"}


@pytest.mark.skipif(
    not POSITIVE or not Path(POSITIVE).exists(), reason="capture missing"
)
def test_saved_capture_with_one_completed_read():
    summary = summarize([POSITIVE])
    sources = summary["sources"]
    assert len(sources) == 6
    assert all(not source["loss"] for source in sources)
    assert summary["problems"] == []
    (transfer,) = summary["transfers"]
    assert transfer["state"] == "completed"
    assert transfer["bytes"] == 37748736
    assert transfer["selected_descriptors"] == 144
    assert transfer["merged_descriptors"] == 36
    assert transfer["nixl_post_us"] > 0 and transfer["nixl_xfer_us"] > 0
    assert transfer["posted_to_completed_ns"] > 0
    assert transfer["throughput_bytes_per_s"] > 0
    states = [item["state"] for item in summary["handshakes"]]
    assert states.count("completed") == 1
    assert states.count("served_reply") == 1
    (handshake,) = [
        item for item in summary["handshakes"] if item["state"] == "completed"
    ]
    assert handshake["duration_ns"] == transfer["handshake_duration_ns"]
    (join,) = summary["joins"]
    assert join["joined"] and not join["ambiguous"]
    entries = summary["aggregates"]["transfers"]["Mooncake/tcp"]
    assert entries["count"] == 1 and entries["completed"] == 1
    assert entries["bytes"] == {"sum": 37748736, "samples": 1}


def test_cli_writes_the_document(recorders, tmp_path):
    recorder = recorders()
    assert recorder.wait_ready()
    recorder.record(*fields(EventKind.REQUEST_BIND, request="req-c"))
    recorder.close()
    from vllm.kv_profiling.summary import main as summary_main

    output = tmp_path / "out" / "summary.json"
    summary_main([str(recorder.output_dir), "--output", str(output)])
    document = json.loads(output.read_text())
    assert document["tool"] == "vllm.kv_profiling.summary"
    assert document["requests"][0]["request"] == "req-c"


# ------------------------------------------------------- post groups and write batches
WORKER = {
    "role": "worker",
    "kv_role": "kv_consumer",
    "engine_id": "engine-decode",
    "backend": ["Test"],
    "transport": "tcp",
    "rank": 0,
}


def bind(record, handle, request, post_pass, selected, *, leg=0, peer="prod-engine"):
    """A NIXL READ's bind: its first value word is the post pass (never bytes)."""
    record(
        *fields(
            EventKind.TRANSFER_BIND,
            request,
            handoff=f"{request}-p",
            peer=peer,
            handle=handle,
            size=post_pass,
            selected=selected,
            aux0=leg,
            aux1=1,
            subject=0,
        )
    )


def done(record, handle, size, merged, post_us=10, xfer_us=20):
    record(
        *fields(
            EventKind.COMPLETE_OBSERVED,
            handle=handle,
            size=size,
            merged=merged,
            aux0=post_us,
            aux1=xfer_us,
            subject=1,
        )
    )


def post_groups_capture(recorders, *, capabilities=True, with_pass=True, close=True):
    """Pass 1 posts reads 1 and 2 with another record (a notification sent from inside the
    pass) between them; pass 2 posts read 3 right after, adjacent in the ring with no
    record between: the sequence alone cannot tell the passes apart. Pass 3 holds a failed
    unused leg and a completed one."""
    context = {**WORKER}
    if capabilities:
        context["capabilities"] = {"post_pass": True}
    recorder = recorders(context=context)
    assert recorder.wait_ready()
    record = recorder.record
    pass_of = (lambda n: n) if with_pass else (lambda n: None)
    bind(record, 1, "req-a", pass_of(1), 10)
    record(*fields(EventKind.NOTIFY_SEND_RETURN, "req-z"))
    bind(record, 2, "req-b", pass_of(1), 20)
    record(*fields(EventKind.POST_RETURN, handle=1))
    record(*fields(EventKind.POST_RETURN, handle=2))
    bind(record, 3, "req-c", pass_of(2), 30)
    record(*fields(EventKind.POST_RETURN, handle=3))
    bind(record, 4, "req-d", pass_of(3), 40, leg=2)
    bind(record, 5, "req-d", pass_of(3), 50, leg=1)
    record(
        *fields(
            EventKind.FAILED, handle=4, status=int(Failure.TRANSFER_FAILED), subject=1
        )
    )
    record(*fields(EventKind.POST_RETURN, handle=5))
    done(record, 1, 100, 5)
    done(record, 2, 200, 8)
    done(record, 3, 300, 9)
    done(record, 5, 500, 11)
    for handle in (1, 2, 3, 4, 5):
        record(*fields(EventKind.TRANSFER_RETIRE, handle=handle))
    if close:
        assert recorder.close().complete
        adapter_summary(recorder)
    return recorder


def adapter_summary(recorder, **overrides):
    """The adapter's ``profiler.json`` next to a synthetic capture (the recorders here are bare: no adapter writes one)."""
    document = {
        "hook_errors": 0,
        "first_error": None,
        "shutdown_failures": [],
        "saved": True,
        "complete": True,
        **overrides,
    }
    (recorder.output_dir / "profiler.json").write_text(json.dumps(document))


def edit_events(recorder, keep):
    path = recorder.output_dir / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows if keep(row)))


def group_of(summary, number):
    return next(g for g in summary["post_groups"] if g["post_pass"] == number)


def test_the_sink_names_the_bind_word_the_post_pass_and_leaves_bytes_unset(recorders):
    recorder = post_groups_capture(recorders)
    binds = [r for r in rows_of(recorder) if r.get("kind") == "transfer_bind"]
    assert [r["post_pass"] for r in binds] == [1, 1, 2, 3, 3]
    assert all(r["bytes"] is None for r in binds)
    others = [
        r
        for r in rows_of(recorder)
        if r.get("type") == "event" and r["kind"] != "transfer_bind"
    ]
    assert others and all("post_pass" not in r for r in others)
    # A completion's bytes are still bytes.
    assert {r["bytes"] for r in others if r["kind"] == "complete_observed"} == {
        100,
        200,
        300,
        500,
    }


def test_post_groups_follow_the_recorded_pass_never_ring_adjacency(recorders):
    recorder = post_groups_capture(recorders)
    summary = summarize([recorder.output_dir])
    assert [g["post_pass"] for g in summary["post_groups"]] == [1, 2, 3]
    one, two = group_of(summary, 1), group_of(summary, 2)
    # Pass 1: two reads, split in the ring by another record; pass 2 is adjacent to it.
    assert (one["reads"], one["requests"], one["completeness"], one["reasons"]) == (
        2, ["req-a", "req-b"], "exact", [])  # fmt: skip
    assert (one["bytes"], one["selected_descriptors"], one["merged_descriptors"]) == (
        300,
        30,
        13,
    )
    assert (two["reads"], two["bytes"], two["completeness"]) == (1, 300, "exact")
    assert [
        m["transfer"] == by_transfer(summary, m["transfer"])["transfer"]
        for m in one["members"]
    ]
    # "First accepted post to last observed completion", never a group transport time.
    times = {t["transfer"]: t for t in summary["transfers"]}
    ids = [m["transfer"] for m in one["members"]]
    assert one["first_post_time_ns"] == min(times[i]["posted_time_ns"] for i in ids)
    assert (
        one["completion_span_ns"]
        == max(times[i]["completed_time_ns"] for i in ids) - one["first_post_time_ns"]
    )
    assert one["posting_span_ns"] == max(times[i]["posted_time_ns"] for i in ids) - min(
        times[i]["prepared_time_ns"] for i in ids)  # fmt: skip
    assert one["read_xfer_us_sum"] == 40 and one["read_post_us_sum"] == 20
    # No request is split or doubled by the groups: reads stay first-class transfers.
    assert len(summary["transfers"]) == 5


def test_a_failed_post_and_an_unused_leg_are_members_with_outcomes(recorders):
    recorder = post_groups_capture(recorders)
    three = group_of(summarize([recorder.output_dir]), 3)
    assert (three["reads"], three["requests"]) == (2, ["req-d"])
    assert three["outcomes"] == {
        "completed": 1,
        "failed": 1,
        "unused": 0,
        "submission_failed": 1,
        "incomplete": 0,
        "not_posted": 1,
    }
    assert [(m["leg"], m["state"], m["posted"]) for m in three["members"]] == [
        ("dram", "failed", False), ("device", "completed", True)]  # fmt: skip
    # The failed leg was never accepted: the group's totals are not whole, its observed bytes are.
    assert (
        three["bytes"] is None
        and three["bytes_observed"] == 500
        and three["bytes_observed_reads"] == 1
    )
    assert three["completeness"] == "exact"
    assert three["selected_descriptors"] == 90
    # Spans cover the accepted posts only.
    assert (
        three["posting_span_ns"] is not None and three["completion_span_ns"] is not None
    )


def test_a_lost_member_makes_every_group_observed_only_with_no_whole_pass_total(
    recorders,
):
    recorder = post_groups_capture(recorders)
    lost = next(
        r["transfer"]
        for r in rows_of(recorder)
        if r.get("kind") == "transfer_bind" and r["selected_descriptors"] == 20
    )
    edit_events(recorder, lambda r: r.get("transfer") != lost)
    summary = summarize([recorder.output_dir])
    assert any(p["kind"] == "capture_loss" for p in summary["problems"])
    one = group_of(summary, 1)
    assert one["completeness"] == "observed_only" and one["reads"] == 1
    assert one["bytes"] is None and one["bytes_observed"] == 100
    assert one["posting_span_ns"] is None and one["completion_span_ns"] is None
    assert any("capture loss" in r for r in one["reasons"])
    assert all(g["completeness"] == "observed_only" for g in summary["post_groups"])


def test_a_capture_that_ends_mid_pass_is_observed_only_only_where_it_may_be_unfinished(
    recorders,
):
    recorder = recorders(context={**WORKER, "capabilities": {"post_pass": True}})
    assert recorder.wait_ready()
    record = recorder.record
    bind(record, 1, "req-a", 1, 10)
    record(*fields(EventKind.POST_RETURN, handle=1))
    done(record, 1, 100, 5)
    bind(record, 2, "req-b", 2, 20)
    bind(record, 3, "req-c", 2, 30)
    record(*fields(EventKind.POST_RETURN, handle=2))
    record(*fields(EventKind.POST_RETURN, handle=3))
    # the capture ends after pass 2's posts: nothing recorded later is in the file
    done(record, 2, 200, 8)
    done(record, 3, 300, 9)
    assert recorder.close().complete
    adapter_summary(recorder)
    last_post = generation_of(recorder, 30)
    cut = next(
        r["sequence"]
        for r in rows_of(recorder)
        if r.get("kind") == "post_return" and r["transfer"] == last_post
    )
    edit_events(
        recorder, lambda r: r.get("type") != "end" and r.get("sequence", -1) <= cut
    )
    summary = summarize([recorder.output_dir])
    one, two = group_of(summary, 1), group_of(summary, 2)
    # Pass 1 finished before the end of the capture; pass 2 may not have (a pass counter does not prove it did).
    assert (one["completeness"], one["bytes"], one["reasons"]) == ("exact", 100, [])
    assert two["completeness"] == "observed_only" and two["bytes"] is None
    assert any("no complete end" in r for r in two["reasons"])
    assert any("no observed completion" in r for r in two["reasons"])
    assert two["completion_span_ns"] is None and two["outcomes"]["incomplete"] == 2


def test_a_capture_without_the_pass_or_its_capability_has_no_groups_and_says_why(
    recorders,
):
    for kwargs in ({"capabilities": False}, {"with_pass": False}):
        recorder = post_groups_capture(recorders, **kwargs)
        summary = summarize([recorder.output_dir])
        assert summary["post_groups"] == []
        assert any(
            "post_group unavailable" in n["reason"] for n in summary["not_available"]
        ), kwargs
        assert len(summary["transfers"]) == 5


MOONCAKE = {
    "role": "worker",
    "kv_role": "kv_producer",
    "engine_id": "engine-prefill",
    "backend": ["Mooncake"],
    "transport": "tcp",
    "connector": "mooncake",
    "rank": 0,
    "capabilities": {"batch_member_shares": True},
}


def member(record, handle, request, ok, share):
    size, selected = share if share else (None, None)
    record(
        *fields(
            EventKind.REQUEST_BIND,
            request,
            handoff=f"{request}-t",
            handle=handle,
            size=size,
            selected=selected,
            status=0 if ok else 1,
            subject=6,
        )
    )


def write_batch(
    record,
    handle,
    shares,
    *,
    declared=None,
    posted=None,
    ok=True,
    complete=True,
    no_write=False,
):
    """One Mooncake write batch of members ``{request: (bytes, descriptors)}``."""
    record(
        *fields(
            EventKind.TRANSFER_BIND,
            handle=handle,
            peer="consumer:1",
            aux0=declared if declared is not None else len(shares),
            subject=0,
        )
    )
    total = (
        posted
        if posted is not None
        else (
            sum(s[0] for s in shares.values() if s),
            sum(s[1] for s in shares.values() if s),
        )
    )
    if not no_write:
        record(*fields(EventKind.PREPARED, handle=handle))
        record(
            *fields(
                EventKind.POST_BEGIN, handle=handle, size=total[0], selected=total[1]
            )
        )
        record(*fields(EventKind.POST_RETURN, handle=handle))
        if ok and complete:
            record(
                *fields(
                    EventKind.COMPLETE_OBSERVED,
                    handle=handle,
                    size=total[0],
                    selected=total[1],
                    status=0,
                    subject=1,
                )
            )
        elif not ok:
            record(
                *fields(
                    EventKind.FAILED,
                    handle=handle,
                    status=int(Failure.TRANSFER_FAILED),
                    subject=1,
                )
            )
    for request, share in shares.items():
        member(record, handle, request, ok, share)
    record(*fields(EventKind.TRANSFER_RETIRE, handle=handle))


def mooncake_summary(recorders, build, **overrides):
    recorder = recorders(context={**MOONCAKE, **overrides})
    assert recorder.wait_ready()
    build(recorder.record)
    assert recorder.close().complete
    adapter_summary(recorder)
    return summarize([recorder.output_dir])


def test_write_batch_members_partition_the_batch_exactly(recorders):
    def build(record):
        write_batch(
            record,
            1,
            {"d-a": (768, 1), "d-b": (256, 1), "d-c": (512, 2), "d-empty": (0, 0)},
        )

    summary = mooncake_summary(recorders, build)
    (batch,) = summary["write_batches"]
    assert batch["view"] == "write_batch" and batch["write"] == "written"
    assert (batch["members_declared"], batch["members_recorded"]) == (4, 4)
    assert (batch["bytes"], batch["descriptors"]) == (1536, 4)
    assert batch["partition"] == {
        "status": "exact",
        "members_bytes": 1536,
        "members_descriptors": 4,
        "reasons": [],
    }
    assert [(m["request"], m["bytes"], m["descriptors"]) for m in batch["members"]] == [
        ("d-a", 768, 1), ("d-b", 256, 1), ("d-c", 512, 2), ("d-empty", 0, 0)]  # fmt: skip


def test_a_member_off_by_one_or_a_missing_zero_byte_member_is_detected(recorders):
    off_by_one = mooncake_summary(
        recorders,
        lambda r: write_batch(
            r, 1, {"d-a": (769, 1), "d-b": (256, 1)}, posted=(1024, 2)
        ),
    )
    (batch,) = off_by_one["write_batches"]
    assert (
        batch["partition"]["status"] == "mismatch"
        and "1025 bytes" in batch["partition"]["reasons"][0]
    )
    off_descriptor = mooncake_summary(
        recorders,
        lambda r: write_batch(
            r, 1, {"d-a": (768, 2), "d-b": (256, 1)}, posted=(1024, 2)
        ),
    )
    assert off_descriptor["write_batches"][0]["partition"]["status"] == "mismatch"
    # The sums still match, but a zero-byte member was never recorded: count and identity catch it.
    missing = mooncake_summary(
        recorders,
        lambda r: write_batch(
            r, 1, {"d-a": (768, 1), "d-b": (256, 1)}, declared=3, posted=(1024, 2)
        ),
    )
    partition = missing["write_batches"][0]["partition"]
    assert (
        partition["status"] == "member_mismatch"
        and "3 members declared, 2 recorded" in partition["reasons"][0]
    )
    twice = mooncake_summary(
        recorders,
        lambda r: (
            write_batch(r, 1, {"d-a": (1024, 2)}, declared=2, posted=(1024, 2)) or None
        ),
    )
    assert twice["write_batches"][0]["partition"]["status"] == "member_mismatch"


def test_unknown_shares_unposted_batches_and_failed_writes_are_not_exact_and_never_zero(
    recorders,
):
    def build(record):
        write_batch(
            record, 1, {"d-a": None, "d-b": None}, posted=(512, 2), ok=False
        )  # exception before the build ended
        write_batch(
            record, 2, {"d-r": (0, 0)}, no_write=True, ok=False
        )  # every member rejected
        write_batch(
            record, 3, {"d-f": (512, 2)}, ok=False
        )  # the write failed: planned shares kept

    summary = mooncake_summary(recorders, build)
    unknown, rejected, failed = summary["write_batches"]
    assert unknown["partition"]["status"] == "shares_unavailable"
    assert unknown["partition"]["members_bytes"] is None
    assert [m["bytes"] for m in unknown["members"]] == [None, None]
    assert (rejected["partition"]["status"], rejected["write"]) == (
        "not_posted",
        "incomplete",
    )
    assert failed["write"] == "failed" and failed["partition"]["status"] == "exact"
    assert (failed["bytes"], failed["partition"]["members_bytes"]) == (512, 512)


def test_an_unhealthy_or_unknown_adapter_never_gives_an_exact_partition_or_group(
    recorders,
):
    def build(record):
        write_batch(record, 1, {"d-a": (768, 1), "d-b": (256, 1)})

    def capture(**health):
        recorder = recorders(context={**MOONCAKE})
        assert recorder.wait_ready()
        build(recorder.record)
        assert recorder.close().complete
        adapter_summary(recorder, **health)
        return recorder

    (clean,) = summarize([capture().output_dir])["write_batches"]
    assert clean["partition"]["status"] == "exact"
    for health in (
        {"hook_errors": 1, "first_error": "TypeError"},
        {"observation_complete": False},
        {"interrupted_generations": 1},
        {"inflight_overflow": 2},
        {"late_hooks": 1},
        {"shutdown_failures": ["recorder: OSError"]},
        {"complete": False},
    ):
        recorder = capture(**health)
        summary = summarize([recorder.output_dir])
        (batch,) = summary["write_batches"]
        assert batch["partition"]["status"] == "observed_only", health
        assert any("adapter:" in r for r in batch["partition"]["reasons"]), health
        assert any(p["kind"] == "adapter_unhealthy" for p in summary["problems"])
    # no profiler.json at all: unknown is not healthy
    recorder = capture()
    (recorder.output_dir / "profiler.json").unlink()
    summary = summarize([recorder.output_dir])
    assert summary["write_batches"][0]["partition"]["status"] == "observed_only"
    assert "profiler.json missing" in summary["sources"][0]["adapter_health"][0]


def test_a_submission_failure_and_an_unused_leg_are_told_apart(recorders):
    recorder = recorders(context={**WORKER, "capabilities": {"post_pass": True}})
    assert recorder.wait_ready()
    record = recorder.record
    bind(record, 1, "req-a", 1, 10, leg=2)
    bind(record, 2, "req-a", 1, 20, leg=1)
    # the first leg's post raised, the second was released unused (what the connector records)
    record(
        *fields(
            EventKind.FAILED, handle=1, status=int(Failure.TRANSFER_FAILED), subject=1
        )
    )
    record(
        *fields(
            EventKind.FAILED,
            handle=2,
            status=int(Failure.TRANSFER_NOT_POSTED),
            subject=1,
        )
    )
    record(*fields(EventKind.TRANSFER_RETIRE, handle=1))
    record(*fields(EventKind.TRANSFER_RETIRE, handle=2))
    bind(record, 3, "req-b", 2, 30)
    record(*fields(EventKind.POST_RETURN, handle=3))
    done(record, 3, 300, 9)
    assert recorder.close().complete
    adapter_summary(recorder)
    one = group_of(summarize([recorder.output_dir]), 1)
    assert one["outcomes"] == {
        "completed": 0,
        "failed": 2,
        "unused": 1,
        "submission_failed": 1,
        "incomplete": 0,
        "not_posted": 2,
    }
    assert [(m["leg"], m["failure"]) for m in one["members"]] == [
        ("dram", "TRANSFER_FAILED"),
        ("device", "TRANSFER_NOT_POSTED"),
    ]
    assert one["completeness"] == "exact" and one["bytes"] is None


def test_a_graceful_close_during_an_unfinished_last_pass_is_observed_only(recorders):
    """The recorder ends cleanly with every row drained, but nothing recorded later shows
    that the last pass finished posting: a clean end alone proves nothing."""
    recorder = recorders(context={**WORKER, "capabilities": {"post_pass": True}})
    assert recorder.wait_ready()
    record = recorder.record
    bind(record, 1, "req-a", 1, 10)
    record(*fields(EventKind.POST_RETURN, handle=1))
    done(record, 1, 100, 5)
    bind(record, 2, "req-b", 2, 20)
    record(*fields(EventKind.POST_RETURN, handle=2))
    assert recorder.close().complete and rows_of(recorder)[-1]["complete"] is True
    adapter_summary(recorder)
    summary = summarize([recorder.output_dir])
    one, two = group_of(summary, 1), group_of(summary, 2)
    assert (one["completeness"], one["bytes"]) == ("exact", 100)
    assert two["completeness"] == "observed_only" and two["bytes"] is None
    assert any("posting ended" in r for r in two["reasons"])
    assert two["completion_span_ns"] is None


def test_a_capture_with_loss_never_claims_an_exact_partition(recorders):
    def build(record):
        write_batch(record, 1, {"d-a": (768, 1), "d-b": (256, 1)})

    recorder = recorders(context={**MOONCAKE})
    assert recorder.wait_ready()
    build(recorder.record)
    assert recorder.close().complete
    gone = next(
        r["sequence"] for r in rows_of(recorder) if r.get("kind") == "request_bind"
    )
    edit_events(recorder, lambda r: r.get("sequence") != gone)
    (batch,) = summarize([recorder.output_dir])["write_batches"]
    assert batch["partition"]["status"] == "member_mismatch"


def test_batch_membership_is_an_association_not_a_request_occurrence(recorders):
    def build(record):
        record(*fields(EventKind.REQUEST_BIND, "d-a", handoff="d-a-t", subject=7))
        write_batch(record, 1, {"d-a": (768, 1), "d-b": (256, 1)})
        write_batch(
            record, 2, {"d-a": (512, 2)}, ok=False
        )  # the same request in a later batch whose write failed
        write_batch(record, 3, {"d-a": None}, posted=(0, 0), ok=False, no_write=True)

    summary = mooncake_summary(recorders, build)
    (a,) = [r for r in summary["requests"] if r["request"] == "d-a"]
    assert (a["request_occurrences"], a["repeated_id"]) == (1, False)
    assert len(a["write_batches"]) == 3
    # Posted shares, planned-but-unconfirmed shares and unknown shares are kept apart.
    assert (a["write_bytes_posted"], a["write_descriptors_posted"]) == (768, 1)
    assert (
        a["write_bytes_planned_unconfirmed"],
        a["write_descriptors_planned_unconfirmed"],
    ) == (512, 2)
    assert a["write_shares_unknown"] == 1
    assert [(b["write"], b["basis"]) for b in a["write_batches"]] == [
        ("written", "posted"), ("failed", "planned"), ("incomplete", "unknown")]  # fmt: skip
    (b,) = [r for r in summary["requests"] if r["request"] == "d-b"]
    assert (b["request_occurrences"], b["write_bytes_posted"]) == (1, 256)
