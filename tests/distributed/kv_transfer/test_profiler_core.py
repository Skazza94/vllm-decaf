# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real core objects, helper processes, sockets and files; no connector doubles."""

import ctypes
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest

from vllm import _kv_profiler as native
from vllm.kv_profiling.clock import (
    ClockGroup,
    ClockMapping,
    ClockSample,
    ClockServer,
    measure_peer,
    select_min_rtt,
)
from vllm.kv_profiling.config import NetworkProfilerConfig
from vllm.kv_profiling.core import ProcessRecorder
from vllm.kv_profiling.perfetto import export_traces
from vllm.kv_profiling.schema import (
    RECORD,
    DictionaryDecoder,
    EventKind,
    decode_record,
    dump_line,
)
from vllm.kv_profiling.sink import JsonlSink, recover_output


def fields(
    kind=EventKind.REQUEST_BIND,
    request=None,
    *,
    handle=None,
    context=0,
    handoff=None,
    peer=None,
    aux0=None,
):
    return (
        kind,
        context,
        request,
        handoff,
        peer,
        handle,
        None,
        None,
        None,
        None,
        None,
        aux0,
        None,
        0,
        0,
        0,
    )


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.005)


@pytest.fixture
def rings():
    resources = []
    directory = Path(tempfile.mkdtemp(prefix="kvprof-test-", dir="/dev/shm"))

    def create(capacity=64, ids=32, id_bytes=4096, handles=8):
        path = directory / str(len(resources))
        ring = native.create(str(path), capacity, ids, id_bytes, handles)
        resources.append(ring)
        return ring, path

    yield create
    for ring in resources:
        ring.close()
    shutil.rmtree(directory)


def take(ring):
    assert ring.acquire_reader()
    begin, end, data = ring.peek(4096)
    result = [
        decode_record(data[i : i + RECORD.size])
        for i in range(0, len(data), RECORD.size)
    ]
    ring.ack(begin, end)
    return result


def resolve(rows):
    decoder = DictionaryDecoder()
    for row in rows:
        if row["type"] == "dictionary_fragment":
            decoder.feed(row)
    return decoder


def test_first_seen_ids_publish_atomically_and_roundtrip(rings):
    ring, path = rings()
    request = "π" * 200
    ring.record(*fields(request=request, handoff="handoff", peer=b"peer", aux0=0))
    rows = take(ring)
    event = rows[-1]
    decoder = resolve(rows)
    assert decoder.resolve(event["request_ref"]) == request
    assert decoder.resolve(event["handoff_ref"]) == "handoff"
    assert decoder.resolve(event["peer_ref"]) == {"bytes_hex": b"peer".hex()}
    assert event["bytes"] is None and event["transport"]["kind"] == "unknown"
    assert event["aux0"] == 0
    assert [r["sequence"] for r in rows] == list(range(1, len(rows) + 1))
    assert path.stat().st_size == ring.stats()["memory_bytes"]


def test_full_ring_never_commits_half_a_dictionary(rings):
    ring, _ = rings(capacity=2)
    ring.record(*fields(request="a" * 512))
    state = ring.stats()
    assert state["attempts"] == state["drop_full"] == 1
    assert state["head"] == state["events"] == state["ids_used"] == 0
    assert state["id_bytes_used"] == 0
    assert take(ring) == []


def test_missing_or_conflicting_dictionary_never_resolves(rings):
    ring, _ = rings()
    ring.record(*fields(request="x" * 200))
    rows = take(ring)
    event = rows[-1]
    missing = resolve(rows[1:])
    assert missing.resolve(event["request_ref"]) is None
    decoder = resolve(rows)
    corrupt = dict(rows[0], payload=(b"z" * 72).hex())
    decoder.feed(corrupt)
    assert decoder.resolve(event["request_ref"]) is None


def test_wrap_and_stale_acknowledgement_preserve_stream_order(rings):
    ring, _ = rings(capacity=4)
    all_rows = []
    for _ in range(8):
        ring.record(*fields(request="a"))
        all_rows.extend(take(ring))
    assert ring.stats()["head"] > ring.stats()["capacity"]
    assert [r["sequence"] for r in all_rows] == list(range(1, len(all_rows) + 1))
    assert ring.stats()["events"] == 8
    with pytest.raises(ValueError, match="stale"):
        ring.ack(0, 0)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("id_count", {"ids": 2}),
        ("id_bytes", {"id_bytes": 2}),
    ],
)
def test_id_arena_lasts_until_shutdown_and_exhaustion_does_not_reuse_ids(
    rings,
    name,
    kwargs,
):
    ring, _ = rings(**kwargs)
    ring.record(*fields(request="a"))
    first = take(ring)
    ring.record(*fields(request="b"))
    second = take(ring)
    ring.record(*fields(request="c"))
    assert ring.stats()["drop_ids"] == 1
    assert take(ring) == []
    assert first[-1]["request_ref"] != second[-1]["request_ref"]


def test_oversized_and_surrogate_ids_drop_without_partial_publication(rings):
    ring, _ = rings()
    ring.record(*fields(request="x" * 513))
    ring.record(*fields(request="\ud800"))
    assert ring.stats()["drop_identity"] == 2
    assert ring.stats()["head"] == ring.stats()["ids_used"] == 0
    ring.record(*fields(request="x" * 512))
    assert ring.stats()["events"] == 1
    assert resolve(take(ring)).resolve(1) == "x" * 512


def test_handle_reuse_has_a_new_generation_even_if_retirement_record_is_lost(rings):
    ring, _ = rings(capacity=4)
    ring.record(*fields(EventKind.TRANSFER_BIND, handle=7))
    ring.record(*fields())
    ring.record(*fields())
    ring.record(*fields())
    ring.record(*fields(EventKind.TRANSFER_RETIRE, handle=7))
    assert ring.stats()["drop_full"] == 1
    assert ring.stats()["live_handles"] == 0
    first = take(ring)[0]
    ring.record(*fields(EventKind.TRANSFER_BIND, handle=7))
    second = take(ring)[0]
    assert (first["transfer"], second["transfer"]) == (1, 2)


def test_live_handle_conflict_and_unknown_handle_remain_explicit(rings):
    ring, _ = rings()
    ring.record(*fields(EventKind.TRANSFER_BIND, handle=1))
    ring.record(*fields(EventKind.TRANSFER_BIND, handle=1))
    ring.record(*fields(EventKind.POST_RETURN, handle=2))
    assert ring.stats()["drop_conflict"] == 1
    rows = take(ring)
    assert rows[-1]["transfer"] == 0 and rows[-1]["unbound_handle"]


def test_only_one_reader_lease_and_output_failure_does_not_ack(rings, tmp_path):
    ring, path = rings()
    ring.record(*fields(request="request"))
    with JsonlSink(str(path)) as sink:
        with pytest.raises(BlockingIOError):
            JsonlSink(str(path))
        with (
            open("/dev/full", "wb", buffering=0) as full,
            pytest.raises(OSError),
        ):
            sink.drain_to(full)
        assert ring.stats()["tail"] == 0
    with (
        JsonlSink(str(path)) as recovered,
        (tmp_path / "trace.jsonl").open("w+b", buffering=0) as output,
    ):
        recovered.drain_to(output)
    assert ring.stats()["tail"] == ring.stats()["head"]


@pytest.fixture
def recorders(tmp_path):
    resources = []

    def create(**kwargs):
        options = dict(
            level="l1",
            run_id=str(uuid4()),
            output_dir=str(tmp_path),
            ring_records=64,
            id_slots=32,
            id_bytes=4096,
            handle_slots=8,
            shutdown_timeout_s=2,
        )
        options.update(kwargs)
        config = NetworkProfilerConfig(**options)
        recorder = ProcessRecorder(config)
        resources.append(recorder)
        return recorder

    yield create
    for recorder in resources:
        recorder.close()
        if recorder.ring_dir.exists():
            shutil.rmtree(recorder.ring_dir)


def test_background_drain_wraps_with_zero_losses(recorders):
    recorder = recorders(flush="background", batch_records=32)
    assert recorder.wait_ready()
    for _ in range(20):
        for _ in range(32):
            recorder.record(*fields(request="same"))
        wait_for(
            lambda: recorder.native.stats()["tail"] == recorder.native.stats()["head"]
        )
    result = recorder.close()
    assert result.saved and result.complete
    assert result.stats["events"] == 640
    assert result.stats["head"] > result.stats["capacity"]
    rows = [
        json.loads(s)
        for s in (recorder.output_dir / "events.jsonl").read_text().splitlines()
    ]
    assert sum(r["type"] == "event" for r in rows) == 640


def test_dead_writer_holding_reader_lease_does_not_block_record_or_recovery(
    recorders,
):
    recorder = recorders()
    assert recorder.wait_ready()
    os.kill(recorder.writer.pid, signal.SIGKILL)
    recorder.writer.wait(2)
    for _ in range(100):
        recorder.record(*fields(request="request"))
    assert recorder.native.stats()["drop_full"] > 0
    recorder.start_writer()
    assert recorder.wait_ready()
    result = recorder.close()
    assert result.saved and not result.complete
    assert result.stats["head"] == result.stats["tail"]


def test_close_is_bounded_when_real_writer_is_stopped_with_lease(recorders):
    recorder = recorders(shutdown_timeout_s=0.1)
    assert recorder.wait_ready()
    os.kill(recorder.writer.pid, signal.SIGSTOP)
    # This is a real process stop, not a mocked stalled writer.
    start = time.monotonic()
    result = recorder.close()
    assert time.monotonic() - start < 2
    assert not result.saved and result.reason == "writer_timeout"
    manifest = json.loads((recorder.output_dir / "manifest.json").read_text())
    assert not manifest["complete"]
    assert recorder.writer.poll() is not None


def test_bad_output_leaves_recording_independent(tmp_path):
    bad = tmp_path / "not-a-directory"
    bad.write_text("file")
    config = NetworkProfilerConfig(
        level="l1",
        run_id=str(uuid4()),
        output_dir=str(bad),
        ring_records=32,
        id_slots=8,
        id_bytes=4096,
        handle_slots=8,
    )
    recorder = ProcessRecorder(config)
    try:
        assert not recorder.wait_ready()
        recorder.record(*fields(request="still-records"))
        assert recorder.native.stats()["events"] == 1
        result = recorder.close()
        assert not result.complete and not result.saved
    finally:
        recorder.close()
        shutil.rmtree(recorder.ring_dir)


class CIdentity(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_char_p),
        ("length", ctypes.c_uint32),
        ("encoding", ctypes.c_uint32),
    ]


class CEvent(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint64),
        ("context", ctypes.c_uint64),
        ("identities", CIdentity * 3),
        ("handle", ctypes.c_uint64),
        ("values", ctypes.c_uint64 * 7),
        ("transport", ctypes.c_uint64),
        ("status", ctypes.c_uint64),
        ("subject", ctypes.c_uint64),
        ("valid", ctypes.c_uint32),
        ("has_handle", ctypes.c_bool),
    ]


def test_native_producers_and_sink_concurrently_account_for_every_attempt(
    recorders,
    tmp_path,
):
    recorder = recorders(flush="background", batch_records=32)
    assert recorder.wait_ready()
    assert ctypes.sizeof(CEvent) == native.EVENT_SIZE
    library = ctypes.CDLL(native.__file__)
    publish = library.kvprof_record
    publish.argtypes = [ctypes.c_void_p, ctypes.POINTER(CEvent)]
    publish.restype = ctypes.c_int
    address = recorder.native.address()
    barrier = threading.Barrier(4)

    def produce(index):
        label = f"producer-{index}".encode()
        event = CEvent()
        event.kind, event.context = 2, index
        event.identities[0] = CIdentity(label, len(label), 1)
        barrier.wait()
        for _ in range(2000):
            publish(address, ctypes.byref(event))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(produce, range(4)))
    result = recorder.close()
    assert result.saved
    state = result.stats
    assert state["attempts"] == 8000
    assert (
        state["events"] + sum(v for k, v in state.items() if k.startswith("drop_"))
        == 8000
    )
    rows = [
        json.loads(s)
        for s in (recorder.output_dir / "events.jsonl").read_text().splitlines()
    ]
    decoder = resolve(rows)
    events = [r for r in rows if r["type"] == "event"]
    assert len(events) == state["events"]
    assert all(
        decoder.resolve(r["request_ref"]) == f"producer-{r['context']}" for r in events
    )
    (tmp_path / "concurrency.json").write_text(json.dumps(state))


def test_first_seen_and_cached_record_paths_allocate_nothing(tmp_path):
    import sysconfig

    sources = Path(__file__).parent
    interposer = tmp_path / "allocations.so"
    extension = tmp_path / (
        "_profiler_allocations" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    subprocess.run(
        [
            *shlex.split(os.environ.get("CC", sysconfig.get_config_var("CC") or "cc")),
            "-shared",
            "-fPIC",
            str(sources / "profiler_allocations.c"),
            "-o",
            str(interposer),
        ],
        check=True,
        timeout=30,
    )
    subprocess.run(
        [
            *shlex.split(
                os.environ.get("CXX", sysconfig.get_config_var("CXX") or "c++")
            ),
            "-std=c++20",
            "-shared",
            "-fPIC",
            "-I" + sysconfig.get_path("include"),
            str(sources / "profiler_allocations.cpp"),
            "-ldl",
            "-o",
            str(extension),
        ],
        check=True,
        timeout=30,
    )
    command = r"""
import shutil
import tempfile
from pathlib import Path
import _profiler_allocations
from vllm import _kv_profiler as native
root = Path(tempfile.mkdtemp(prefix="kvprof-alloc-", dir="/dev/shm"))
ring = native.create(str(root / "ring"), 2048, 256, 65536, 8)
try:
    # Python inputs exist before allocation counting starts, including fresh IDs.
    batches = [(2, 0, f"first-seen-{i}", None, None, None,
                None, None, None, None, None, None, None, 0, 0, 0)
               for i in range(128)]
    counts = _profiler_allocations.measure(ring.record, batches)
    assert counts == (0, 0, 0, 0), counts
    assert ring.stats()["ids_used"] == 128
    counts = _profiler_allocations.measure(ring.record, [batches[0]] * 128)
    assert counts == (0, 0, 0, 0), counts
    print("first-seen and cached: RAW=MEM=OBJ=libc=0")
finally:
    ring.close()
    shutil.rmtree(root)
"""
    environment = dict(os.environ, LD_PRELOAD=str(interposer), PYTHONPATH=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-c", command],
        env=environment,
        check=True,
        timeout=30,
        text=True,
        capture_output=True,
    )
    assert "first-seen and cached: RAW=MEM=OBJ=libc=0" in result.stdout
    (tmp_path / "allocations.txt").write_text(result.stdout)


def test_disabled_entry_imports_no_native_extension():
    command = """
import sys
import vllm
baseline = set(sys.modules)
from vllm.kv_profiling.config import NetworkProfilerConfig
from vllm.kv_profiling.core import open_recorder
assert open_recorder(NetworkProfilerConfig()) is None
assert "vllm._kv_profiler" not in sys.modules
assert not any(n.startswith("torch") for n in set(sys.modules) - baseline)
assert "vllm.distributed" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", command], check=True, timeout=30)


RUN, LOCAL, REMOTE = str(uuid4()), str(uuid4()), str(uuid4())


def sample(t=1000, offset=100, forward=10, backward=10, remote=REMOTE):
    return ClockSample(
        RUN,
        LOCAL,
        remote,
        t,
        t + forward + offset,
        t + forward + offset + 2,
        t + forward + 2 + backward,
    )


def test_clock_selection_keeps_raw_samples_and_generations():
    samples = [sample(forward=40), sample(), sample(backward=50)]
    mapping = ClockMapping.from_samples(samples)
    assert mapping.start == samples[1]
    assert len(mapping.to_json()["start"]) == 3
    restored = ClockMapping.from_json(mapping.to_json())
    assert restored == mapping
    assert not restored.project(2000).drift_known
    assert restored.project(2000).extrapolated


def test_asymmetric_paths_keep_rtt_uncertainty_not_false_precision():
    symmetric, asymmetric = sample(), sample(forward=3, backward=17)
    assert symmetric.raw_rtt_ns == asymmetric.raw_rtt_ns
    assert symmetric.offset != asymmetric.offset
    mapped = ClockMapping(asymmetric).project(2000)
    assert mapped.uncertainty_ns == asymmetric.raw_rtt_ns


def test_drift_projection_and_restart_are_explicit():
    start, end = sample(t=1000, offset=100), sample(t=2000, offset=200)
    mapping = ClockMapping(start, end)
    midpoint_local = (start.anchor + end.anchor) / 2
    remote = midpoint_local + 150
    assert mapping.project(remote).time_ns == midpoint_local
    with pytest.raises(ValueError, match="generation"):
        ClockMapping(start, sample(t=2000, remote=str(uuid4())))


def test_real_clock_socket_exchange_and_restart():
    with ClockServer(RUN) as server:
        values = measure_peer(
            server.address,
            RUN,
            LOCAL,
            expected_remote=server.generation,
        )
        assert len(values) == 7
        assert select_min_rtt(values).raw_rtt_ns > 0
        generation = server.generation
    with (
        ClockServer(RUN) as replacement,
        pytest.raises(ValueError, match="generation"),
    ):
        measure_peer(
            replacement.address,
            RUN,
            LOCAL,
            expected_remote=generation,
        )


def test_clock_timeout_is_bounded_with_a_real_stopped_peer():
    with ClockServer(RUN) as server:
        os.kill(server.process.pid, signal.SIGSTOP)
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            measure_peer(server.address, RUN, LOCAL, timeout_s=0.1)
        assert time.monotonic() - start < 1


def make_trace(path, ring, *, process, clock, timestamp, modeled=False, partial=False):
    ring.record(
        *fields(
            EventKind.REQUEST_BIND,
            request="x" * 100,
        )
    )
    rows = take(ring)
    for row in rows:
        row["time_ns"] = timestamp
        if modeled and row["type"] == "event":
            row["kind"] = "model_step"
    if partial:
        rows = rows[1:]
    metadata = {
        "type": "metadata",
        "schema": 1,
        "run_id": RUN,
        "process_id": process,
        "clock_id": clock,
        "context": {"role": "test", "rank": process},
    }
    path.write_bytes(
        b"".join(
            dump_line(r)
            for r in [
                metadata,
                *rows,
                {"type": "end", "schema": 1, "complete": not partial},
            ]
        )
    )


def test_perfetto_common_epoch_preserves_cross_process_delay(rings, tmp_path):
    a, _ = rings()
    b, _ = rings()
    pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    make_trace(pa, a, process="a", clock=LOCAL, timestamp=10_000)
    make_trace(pb, b, process="b", clock=REMOTE, timestamp=15_100)
    group = ClockGroup(LOCAL, {REMOTE: ClockMapping(sample())})
    documents = export_traces([pa, pb], str(tmp_path / "export"), group)
    trace = json.loads(documents[("wall", "aligned")].read_text())
    instants = [r for r in trace["traceEvents"] if r["ph"] == "i"]
    assert [r["ts"] for r in instants] == [0, 5]
    assert all(r["args"]["raw_ns"] for r in instants)
    assert all(r["args"]["bytes"] is None for r in instants)
    assert trace["metadata"]["calibrations"][REMOTE]["start"]


def test_perfetto_missing_ids_and_unaligned_model_time_stay_separate(rings, tmp_path):
    a, _ = rings()
    b, _ = rings()
    pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    make_trace(pa, a, process="a", clock=LOCAL, timestamp=100, partial=True)
    make_trace(pb, b, process="b", clock=REMOTE, timestamp=100, modeled=True)
    paths = export_traces([pa, pb], str(tmp_path / "export"))
    assert len(paths) == 2
    trace = json.loads(paths[("wall", LOCAL)].read_text())
    row = next(r for r in trace["traceEvents"] if r["ph"] == "i")
    assert row["args"]["request"] is None
    assert row["args"]["request_resolution"] == "missing"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ring_records": 3},
        {"id_slots": 0},
        {"id_bytes": 0},
        {"batch_records": True},
        {"shutdown_timeout_s": float("nan")},
        {"level": "l2"},
        {"flush": "other"},
        {"level": "l1"},
    ],
)
def test_configuration_rejects_invalid_resource_and_lifecycle_settings(kwargs):
    with pytest.raises(ValueError):
        NetworkProfilerConfig(**kwargs)


def test_existing_modeled_log_is_imported_without_live_step_recording(tmp_path):
    from vllm.kv_profiling.perfetto import import_modeled_steps

    source, target = tmp_path / "steps.jsonl", tmp_path / "modeled.jsonl"
    source.write_text(
        json.dumps({"timing": {"provider": "aisimulate"}})
        + "\n"
        + json.dumps(
            {"step": {"index": 0}, "cost": {"device_s": 0.002, "host_s": 0.001}}
        )
        + "\n"
    )
    import_modeled_steps(source, target, run_id=RUN)
    traces = export_traces([target], str(tmp_path / "export"))
    document = json.loads(next(iter(traces.values())).read_text())
    span = next(e for e in document["traceEvents"] if e["ph"] == "X")
    assert span["dur"] == 2000
    assert span["args"]["clock_model"] == "modeled_elapsed"
    assert not document["metadata"]["calibrations"]


def _abandon_producer(config, output):
    recorder = ProcessRecorder(config)
    assert recorder.wait_ready()
    recorder.record(*fields(request="before-exit"))
    Path(output).write_text(str(recorder.output_dir))
    os._exit(0)


def test_orphaned_helper_finishes_with_incomplete_manifest(tmp_path):
    import multiprocessing

    config = NetworkProfilerConfig(
        level="l1",
        run_id=str(uuid4()),
        output_dir=str(tmp_path / "out"),
        ring_records=64,
        id_slots=32,
        id_bytes=4096,
        handle_slots=8,
    )
    location = tmp_path / "location"
    producer = multiprocessing.get_context("spawn").Process(
        target=_abandon_producer,
        args=(config, str(location)),
    )
    producer.start()
    producer.join(15)
    assert producer.exitcode == 0
    output = Path(location.read_text())
    manifest = output / "manifest.json"
    wait_for(lambda: json.loads(manifest.read_text()).get("reason") == "producer_gone")
    saved = json.loads(manifest.read_text())
    assert not saved["complete"]
    shutil.rmtree(saved["recovery_directory"])


def test_partial_durable_dictionary_is_trimmed_and_replayed(rings, tmp_path):
    ring, path = rings()
    ring.record(*fields(request="x" * 100))
    assert ring.acquire_reader()
    _, _, raw = ring.peek(64)
    first = dump_line(decode_record(raw[: RECORD.size]))
    # A real on-disk prefix with one complete fragment and a truncated next row.
    output = tmp_path / "partial.jsonl"
    output.write_bytes(first + b'{"schema":1,"type":"dictionary_frag')
    ring.close()  # Release the real reader lease before recovery.
    with JsonlSink(str(path)) as sink, output.open("r+b", buffering=0) as stream:
        sequence = recover_output(stream)
        assert sequence == 1
        assert stream.tell() == len(first)
        sink.drain_to(stream, after_sequence=sequence)
        assert sink.ring.stats()["head"] == sink.ring.stats()["tail"]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["sequence"] for row in rows] == [1, 2, 3]
    decoder = resolve(rows)
    assert decoder.resolve(rows[-1]["request_ref"]) == "x" * 100


def test_live_handle_map_exhaustion_is_bounded_and_retirement_recovers(rings):
    ring, _ = rings(capacity=64, handles=2)
    for handle in (1, 2, 3):
        ring.record(*fields(EventKind.TRANSFER_BIND, handle=handle))
    assert ring.stats()["live_handles"] == 2
    assert ring.stats()["drop_handles"] == 1
    memory = ring.stats()["memory_bytes"]
    ring.record(*fields(EventKind.TRANSFER_RETIRE, handle=1))
    ring.record(*fields(EventKind.TRANSFER_BIND, handle=3))
    assert ring.stats()["live_handles"] == 2
    assert ring.stats()["memory_bytes"] == memory
    assert [row["transfer"] for row in take(ring)] == [1, 2, 1, 3]


def test_perfetto_does_not_trust_complete_footer_after_fragment_loss(rings, tmp_path):
    ring, _ = rings()
    source = tmp_path / "lost.jsonl"
    make_trace(source, ring, process="a", clock=LOCAL, timestamp=100, partial=True)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[-1]["complete"] = True
    source.write_bytes(b"".join(dump_line(row) for row in rows))
    paths = export_traces([source], str(tmp_path / "export"))
    trace = json.loads(paths[("wall", LOCAL)].read_text())
    instant = next(row for row in trace["traceEvents"] if row["ph"] == "i")
    assert instant["args"]["loss"]
    assert instant["args"]["request"] is None


def test_modeled_import_preserves_provider_and_refuses_mixed_generations(tmp_path):
    from vllm.kv_profiling.perfetto import import_modeled_steps

    source, target = tmp_path / "steps.jsonl", tmp_path / "modeled.jsonl"
    header = {"timing": {"provider": "aisimulate", "table": "recorded"}}
    step = {"step": {"step_index": 0}, "cost": {"device_s": 0.002}}
    source.write_text(json.dumps(header) + "\n" + json.dumps(step) + "\n")
    import_modeled_steps(source, target, run_id=RUN)
    metadata = json.loads(target.read_text().splitlines()[0])
    assert metadata["timing"] == header["timing"]
    with source.open("a") as out:
        out.write(json.dumps(header) + "\n")
    with pytest.raises(ValueError, match="restarted"):
        import_modeled_steps(source, target, run_id=RUN)


def test_contended_retirement_invalidates_old_associations_before_handle_reuse(rings):
    ring, _ = rings(capacity=1024)
    library = ctypes.CDLL(native.__file__)
    publish = library.kvprof_record
    publish.argtypes = [ctypes.c_void_p, ctypes.POINTER(CEvent)]
    publish.restype = ctypes.c_int
    address = ring.address()
    stop = threading.Event()
    entered = threading.Event()
    long_id = b"x" * 512

    def contend():
        event = CEvent()
        event.kind = int(EventKind.REQUEST_BIND)
        for i in range(3):
            event.identities[i] = CIdentity(long_id, len(long_id), 0)
        entered.set()
        while not stop.is_set():
            publish(address, ctypes.byref(event))

    def operation(kind):
        event = CEvent()
        event.kind, event.handle, event.has_handle = int(kind), 42, True
        return publish(address, ctypes.byref(event))

    worker = threading.Thread(target=contend)
    worker.start()
    generations = []

    def drain(*, all_rows=False):
        while ring.stats()["head"] != ring.stats()["tail"]:
            generations.extend(
                row["transfer"]
                for row in take(ring)
                if row["type"] == "event" and row["kind"] == "transfer_bind"
            )
            if not all_rows:
                break

    observed = False
    try:
        assert entered.wait(2)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = operation(EventKind.TRANSFER_BIND)
            if result == 2:  # Full: drain rather than changing the race ordering.
                drain()
            if result != 0:
                continue
            if operation(EventKind.TRANSFER_RETIRE) == 3:  # Actual native contention.
                observed = True
                break
    finally:
        stop.set()
        worker.join(2)
    assert not worker.is_alive()
    assert observed, "native producers did not reach contended retirement"
    drain(all_rows=True)
    previous = generations[-1]
    ring.record(*fields(EventKind.COMPLETE_OBSERVED, handle=42))
    unknown = take(ring)[-1]
    assert unknown["transfer"] == 0 and unknown["unbound_handle"]
    ring.record(*fields(EventKind.TRANSFER_BIND, handle=42))
    ring.record(*fields(EventKind.COMPLETE_OBSERVED, handle=42))
    rebound, completed = take(ring)
    assert rebound["transfer"] != previous
    assert completed["transfer"] == rebound["transfer"]
    assert not completed["unbound_handle"]
    assert ring.stats()["drop_contention"] > 0


def test_conflicting_rebind_does_not_leave_the_old_transfer_live(rings):
    ring, _ = rings()
    ring.record(*fields(EventKind.TRANSFER_BIND, handle=42))
    ring.record(*fields(EventKind.TRANSFER_BIND, handle=42))
    ring.record(*fields(EventKind.COMPLETE_OBSERVED, handle=42))
    completed = take(ring)[-1]
    assert completed["transfer"] == 0 and completed["unbound_handle"]
    assert ring.stats()["drop_conflict"] == 1


def test_partial_footer_cannot_publish_a_successful_capture(tmp_path):
    config = NetworkProfilerConfig(
        level="l1",
        run_id=str(uuid4()),
        output_dir=str(tmp_path / "output"),
        ring_records=128,
        id_slots=32,
        id_bytes=4096,
        handle_slots=8,
    )
    recorder = ProcessRecorder(config, start_writer=False)
    try:
        for _ in range(50):
            recorder.record(*fields())
        with JsonlSink(str(recorder.ring_dir / "ring")) as reader:
            _, _, raw = reader.ring.peek(4096)
        rows = [
            decode_record(raw[i : i + RECORD.size])
            for i in range(0, len(raw), RECORD.size)
        ]
        prefix = dump_line({"type": "metadata", "schema": 1, **recorder.metadata})
        limit = len(prefix) + sum(len(dump_line(row)) for row in rows) + 16
        assert limit > 8192  # The separate manifest still fits the real file limit.
        recorder.native.finish()
        command = r"""
import resource
import signal
import sys
from vllm.kv_profiling.sink import writer_main
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
limit = int(sys.argv[3])
resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
writer_main(sys.argv[1], sys.argv[2], "end", 256)
"""
        subprocess.run(
            [
                sys.executable,
                "-c",
                command,
                str(recorder.ring_dir),
                str(recorder.output_dir),
                str(limit),
            ],
            check=True,
            timeout=15,
        )
        state = recorder.native.stats()
        manifest = json.loads((recorder.output_dir / "manifest.json").read_text())
        result = recorder.close()
        assert not result.saved and not result.complete
        assert not manifest["complete"] and not manifest["saved"]
        assert state["writer_state"] == 3 and state["writer_errno"] == 27
        assert (recorder.output_dir / "events.jsonl").stat().st_size == limit
        assert recorder.ring_dir.exists()
        (tmp_path / "short-write.json").write_text(
            json.dumps({"limit": limit, "state": state, "manifest": manifest})
        )
    finally:
        recorder.close()
        shutil.rmtree(recorder.ring_dir, ignore_errors=True)


@pytest.mark.parametrize("accepted_first", [False, True])
def test_export_retains_empty_and_lossy_capture_status(
    recorders, tmp_path, accepted_first
):
    recorder = recorders(ring_records=2)
    assert recorder.wait_ready()
    if accepted_first:
        recorder.record(*fields())
    recorder.record(*fields(request="x" * 512))
    result = recorder.close()
    assert result.saved and not result.complete
    paths = export_traces(
        [recorder.output_dir / "events.jsonl"], str(tmp_path / "export")
    )
    assert len(paths) == 1
    trace = json.loads(next(iter(paths.values())).read_text())
    capture = trace["metadata"]["captures"][0]
    assert capture["process_id"] == recorder.process_id
    assert capture["clock_id"] == recorder.metadata["clock_id"]
    assert capture["end_present"] and capture["loss"]
    assert capture["counters"]["drop_full"] == 1
    assert capture["counters"]["events"] == int(accepted_first)
    assert not capture["end_records"][-1]["complete"]
    assert (
        trace["metadata"]["origin_ns"] is not None
        if accepted_first
        else (trace["metadata"]["origin_ns"] is None)
    )


def test_export_reconciles_final_sequence_against_complete_footer(recorders, tmp_path):
    recorder = recorders()
    assert recorder.wait_ready()
    recorder.record(*fields())
    recorder.record(*fields())
    assert recorder.close().complete
    source = recorder.output_dir / "events.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    source.write_bytes(
        b"".join(dump_line(row) for row in rows if row.get("sequence") != 2)
    )
    paths = export_traces([source], str(tmp_path / "export"))
    trace = json.loads(next(iter(paths.values())).read_text())
    instant = next(row for row in trace["traceEvents"] if row["ph"] == "i")
    assert instant["args"]["loss"]
    capture = trace["metadata"]["captures"][0]
    assert capture["final_sequence"] == 1
    assert capture["counters"]["head"] == 2
    assert capture["end_records"][-1]["complete"]
    assert capture["missing_sequences"] and capture["loss"]


def test_root_build_ext_and_component_install_load_the_extension(tmp_path):
    configured = os.environ.get("KV_PROFILER_SOURCE_ROOT")
    source = Path(configured) if configured else Path(__file__).parents[3]
    # setup.py writes version metadata; keep implementation sources read-only.
    shadow = tmp_path / "source"
    shadow.mkdir()
    for entry in source.iterdir():
        if entry.name not in ("vllm", ".git"):
            (shadow / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    (shadow / "vllm").mkdir()
    for entry in (source / "vllm").iterdir():
        if entry.name != "_version.py":
            (shadow / "vllm" / entry.name).symlink_to(
                entry, target_is_directory=entry.is_dir()
            )
    build = tmp_path / "build"
    installed = tmp_path / "build-ext-install"
    command = r"""
from pathlib import Path
import sys
import setuptools
from distutils.core import run_setup

distribution = run_setup("setup.py", script_args=["build_ext"], stop_after="init")
distribution.ext_modules = [e for e in distribution.ext_modules
                            if e.name == "vllm._kv_profiler"]
assert len(distribution.ext_modules) == 1
builder = distribution.get_command_obj("build_ext")
print("builder MRO:", [c.__name__ for c in type(builder).__mro__])
assert any(c.__name__ == "cmake_build_ext" for c in type(builder).__mro__)
builder.ensure_finalized()
builder.build_temp, builder.build_lib = sys.argv[1:]
builder.build_extensions()
assert (Path(builder.build_lib) / "vllm/_kv_profiler.abi3.so").is_file()
"""
    environment = dict(
        os.environ,
        VLLM_TARGET_DEVICE="cpu",
        VLLM_USE_PRECOMPILED="0",
        VLLM_USE_PRECOMPILED_RUST="0",
        VLLM_REQUIRE_RUST_FRONTEND="0",
        VLLM_VERSION_OVERRIDE="0.0.0",
        VLLM_RS_BUILD_VERSION="0.0.0",
        MAX_JOBS="2",
        CMAKE_ARGS="-DVLLM_TARGET_DEVICE=empty",
        FETCHCONTENT_BASE_DIR=str(tmp_path / "dependencies"),
    )
    with (tmp_path / "build-ext.log").open("w") as log:
        subprocess.run(
            [sys.executable, "-c", command, str(build), str(installed)],
            cwd=shadow,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=120,
        )
    component = tmp_path / "component-install"
    with (tmp_path / "component.log").open("w") as log:
        subprocess.run(
            [
                "cmake",
                "--install",
                str(build),
                "--prefix",
                str(component),
                "--component",
                "_kv_profiler",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=30,
        )
    load = r"""
import importlib.util
import tempfile
from pathlib import Path
import sys
path = Path(sys.argv[1]) / "vllm/_kv_profiler.abi3.so"
spec = importlib.util.spec_from_file_location("_kv_profiler", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert Path(module.__file__).resolve() == path.resolve()
with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
    ring = module.create(str(Path(directory) / "ring"), 2, 2, 64, 2)
    assert ring.stats()["events"] == 0
    ring.close()
print("imported:", module.__file__)
"""
    for destination in (installed, component):
        result = subprocess.run(
            [sys.executable, "-c", load, str(destination)],
            check=True,
            timeout=30,
            text=True,
            capture_output=True,
        )
        assert str(destination) in result.stdout
        (destination / "import.log").write_text(result.stdout)
