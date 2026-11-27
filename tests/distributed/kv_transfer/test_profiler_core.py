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


def test_an_empty_drain_does_no_io_and_a_nonempty_one_syncs_before_ack(
    rings, tmp_path, monkeypatch
):
    """An idle background helper drains an empty ring every few milliseconds:
    that must neither flush, sync nor acknowledge. A batch with records is still
    written and synced before its acknowledgement."""
    from vllm.kv_profiling import sink as sink_module

    ring, path = rings()
    syncs = []
    real_fsync = os.fsync

    def fsync(fd):
        syncs.append(ring.stats()["tail"])
        real_fsync(fd)

    monkeypatch.setattr(sink_module.os, "fsync", fsync)
    output = tmp_path / "trace.jsonl"
    with JsonlSink(str(path)) as sink, output.open("w+b", buffering=0) as stream:
        for _ in range(50):
            assert sink.drain_to(stream) == 0
        assert syncs == [] and output.stat().st_size == 0
        ring.record(*fields(request="request"))
        end = sink.drain_to(stream)
        assert end == ring.stats()["head"] > 0
        # synced once, while the batch was still unacknowledged
        assert syncs == [0] and ring.stats()["tail"] == end
        assert sink.drain_to(stream) == end and syncs == [0]


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
            # ring semantics are tested deterministically: drained at close only
            flush="end",
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
    assert not saved["complete"] and saved["saved"]
    # its owner is gone and every record is saved: the helper frees the ring
    wait_for(lambda: not Path(saved["recovery_directory"]).exists())


@pytest.mark.parametrize(
    "sizes",
    [{}, {"ring_records": 64, "id_slots": 32, "id_bytes": 4097, "handle_slots": 8}],
)
def test_ring_bytes_is_the_native_ring_s_size(rings, sizes):
    """The size the no-room error reports is the one the native ring takes."""
    from vllm.kv_profiling.core import ring_bytes

    config = NetworkProfilerConfig(
        level="l1", run_id=str(uuid4()), output_dir="/unused", **sizes
    )
    ring, path = rings(
        capacity=config.ring_records,
        ids=config.id_slots,
        id_bytes=config.id_bytes,
        handles=config.handle_slots,
    )
    assert ring_bytes(config) == ring.stats()["memory_bytes"] == path.stat().st_size


def test_a_full_shared_memory_says_where_how_much_and_what_to_do(
    tmp_path, monkeypatch
):
    """No room for the ring is a clear profiler error, not a bare ENOSPC, and
    leaves no ring directory behind."""
    import errno

    from vllm import _kv_profiler

    def full(*args):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(_kv_profiler, "create", full)
    shared = tmp_path / "shm"
    shared.mkdir()
    config = NetworkProfilerConfig(
        level="l1",
        run_id=str(uuid4()),
        output_dir=str(tmp_path / "out"),
        shared_dir=str(shared),
    )
    with pytest.raises(OSError) as raised:
        ProcessRecorder(config)
    message = str(raised.value)
    assert raised.value.errno == errno.ENOSPC
    for part in (f"in {shared}", "MiB free of", "each recorder needs 17.25 MiB",
                 "ring_records=65536", "--shm-size", "No space left on device"):
        assert part in message, (part, message)
    assert list(shared.iterdir()) == []


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


def test_a_kernel_without_time_namespaces_records_with_none(recorders, monkeypatch):
    """Before Linux 5.6 (or without CONFIG_TIME_NS) there is no /proc/self/ns/time:
    every process shares the host's monotonic clock, so the recorder still records,
    with time_namespace None."""
    real_readlink = os.readlink

    def readlink(path, *args, **kwargs):
        if str(path) == "/proc/self/ns/time":
            raise FileNotFoundError("time namespaces not configured")
        return real_readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", readlink)
    recorder = recorders()
    assert recorder.metadata["time_namespace"] is None
    assert recorder.wait_ready()
    recorder.record(*fields())
    result = recorder.close()
    assert result.saved and result.complete


def _set_thread_name(name):
    libc = ctypes.CDLL(None, use_errno=True)
    assert libc.prctl(15, name.encode(), 0, 0, 0) == 0  # PR_SET_NAME


def test_records_carry_their_thread_and_the_writer_names_it(recorders):
    """Each event records the kernel ID of the thread that recorded it; the writer
    maps every recording thread still alive (or seen alive) to its kernel name."""
    recorder = recorders()
    assert recorder.wait_ready()
    seen, recorded, release = {}, threading.Barrier(4), threading.Event()

    def produce(index):
        _set_thread_name(f"kvp-read-{index}")
        seen[index] = threading.get_native_id()
        for _ in range(3):
            recorder.record(*fields(context=index, request=f"r{index}"))
        recorded.wait()
        release.wait(30)  # long-lived, like the connector's threads

    workers = [threading.Thread(target=produce, args=(i,)) for i in range(3)]
    for worker in workers:
        worker.start()
    recorded.wait()
    recorder.record(*fields(context=99, request="main"))
    result = recorder.close()
    release.set()
    for worker in workers:
        worker.join()
    assert result.saved and result.complete
    rows = [
        json.loads(s)
        for s in (recorder.output_dir / "events.jsonl").read_text().splitlines()
    ]
    events = [r for r in rows if r["type"] == "event"]
    assert {r["context"]: r["tid"] for r in events} == {
        **seen,
        99: threading.get_native_id(),
    }
    assert all(r["subject"] == 0 for r in events)
    threads = {
        k: v for r in rows if r["type"] == "threads" for k, v in r["threads"].items()
    }
    assert rows[-1]["thread_count"] == len(threads) == 4
    assert {threads[str(seen[i])] for i in range(3)} == {
        f"kvp-read-{i}" for i in range(3)
    }
    manifest = json.loads((recorder.output_dir / "manifest.json").read_text())
    assert manifest["threads"] == threads


def test_a_subject_wider_than_32_bits_is_refused(recorders):
    """Word 8 holds the subject in its low 32 bits; a wider subject would collide
    with the thread ID, so the binding refuses it and the C entry point drops it."""
    recorder = recorders()
    assert recorder.wait_ready()
    with pytest.raises(ValueError, match="subject"):
        recorder.record(*fields()[:15], 1 << 32)
    recorder.record(*fields()[:15], (1 << 32) - 1)
    result = recorder.close()
    assert result.saved
    rows = [
        json.loads(s)
        for s in (recorder.output_dir / "events.jsonl").read_text().splitlines()
    ]
    (event,) = [r for r in rows if r["type"] == "event"]
    assert event["subject"] == (1 << 32) - 1
    assert event["tid"] == threading.get_native_id()


def test_a_forked_child_records_its_own_thread_id(recorders, tmp_path):
    """After a fork the child's records carry the child's thread ID, never the
    parent's, although the forking thread recorded in the parent first."""
    parent = recorders()
    assert parent.wait_ready()
    parent.record(*fields(context=1))
    out = tmp_path / "child.json"
    pid = os.fork()
    if pid == 0:  # child: a recorder of its own, then report what it recorded
        code = 1
        try:
            config = NetworkProfilerConfig(
                level="l1", run_id=str(uuid4()), output_dir=str(tmp_path / "child"),
                ring_records=64, id_slots=32, id_bytes=4096, handle_slots=8,
                shutdown_timeout_s=2,
            )  # fmt: skip
            child = ProcessRecorder(config)
            assert child.wait_ready()
            child.record(*fields(context=2))
            assert child.close().saved
            rows = [
                json.loads(s)
                for s in (child.output_dir / "events.jsonl").read_text().splitlines()
            ]
            (event,) = [r for r in rows if r["type"] == "event"]
            out.write_text(
                json.dumps({"tid": event["tid"], "own": threading.get_native_id()})
            )
            shutil.rmtree(child.ring_dir, ignore_errors=True)
            code = 0
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    reported = json.loads(out.read_text())
    assert reported["tid"] == reported["own"] == pid
    assert reported["tid"] != threading.get_native_id()
    assert parent.close().saved


def test_a_restarted_writer_keeps_the_threads_it_recovered(recorders):
    """A writer restarted after it acknowledged events rebuilds its thread
    inventory from the rows already written: a thread seen only before the
    restart stays listed, with a null name if it exited unseen."""
    recorder = recorders(flush="background")
    assert recorder.wait_ready()
    ids, release, recorded = {}, threading.Event(), threading.Event()

    def short():
        ids["short"] = threading.get_native_id()
        recorder.record(*fields(context=1, request="short"))

    def live():
        _set_thread_name("kvp-live")
        ids["live"] = threading.get_native_id()
        recorder.record(*fields(context=2, request="live"))
        recorded.set()
        release.wait(30)

    worker = threading.Thread(target=short)
    worker.start()
    worker.join()
    alive = threading.Thread(target=live)
    alive.start()
    assert recorded.wait(10)
    wait_for(lambda: (s := recorder.native.stats())["tail"] == s["head"] >= 2)
    recorder.writer.kill()
    recorder.writer.wait(10)
    recorder.start_writer()
    assert recorder.wait_ready()
    result = recorder.close()
    release.set()
    alive.join()
    assert result.saved and result.complete
    rows = [
        json.loads(s)
        for s in (recorder.output_dir / "events.jsonl").read_text().splitlines()
    ]
    threads = {
        k: v for r in rows if r["type"] == "threads" for k, v in r["threads"].items()
    }
    assert threads[str(ids["short"])] is None
    assert threads[str(ids["live"])] == "kvp-live"
    assert rows[-1]["thread_count"] == len(threads) == 2
    manifest = json.loads((recorder.output_dir / "manifest.json").read_text())
    assert manifest["threads"] == threads


def test_any_number_of_threads_stays_readable_by_recovery_and_the_reader(
    recorders,
):
    """The inventory is split into rows under the reader's line limit: 1,024
    threads with maximal 15-character names round-trip through recover_output
    and load_capture."""
    from vllm.kv_profiling.sink import MAX_LINE, thread_rows
    from vllm.kv_profiling.summary import load_capture

    recorder = recorders()
    assert recorder.wait_ready()
    recorder.record(*fields(context=1))
    assert recorder.close().complete
    synthetic = {str(100000 + i): f"thread-{i:08d}" for i in range(1024)}
    extra = [dump_line(row) for row in thread_rows(synthetic)]
    assert len(extra) > 1 and all(len(line) <= MAX_LINE for line in extra)
    path = recorder.output_dir / "events.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1] + extra + lines[-1:]))
    tids, names = set(), {}
    with path.open("r+b") as stream:
        recover_output(stream, tids, names)
    assert {str(t) for t in tids} >= set(synthetic)
    assert {str(t): n for t, n in names.items() if str(t) in synthetic} == synthetic
    capture = load_capture(path)
    assert {k: v for k, v in capture["threads"].items() if k in synthetic} == synthetic


def _small(tmp_path, **extra):
    return NetworkProfilerConfig(
        level="l1",
        run_id=str(uuid4()),
        output_dir=str(tmp_path / "out"),
        ring_records=64,
        id_slots=32,
        id_bytes=4096,
        handle_slots=8,
        **extra,
    )


def _producer_without_helper(config, location):
    recorder = ProcessRecorder(config, start_writer=False)
    recorder.record(*fields(request="before-exit"))
    Path(location).write_text(str(recorder.ring_dir))
    os._exit(0)


def _owner_dies_right_after_its_helper(config, location):
    recorder = ProcessRecorder(config)
    assert recorder.wait_ready()
    recorder.record(*fields(request="before-close"))
    Path(location).write_text(
        json.dumps([str(recorder.ring_dir), str(recorder.output_dir)])
    )
    wait = recorder.writer.wait

    def exit_after_save(*args, **kwargs):
        code = wait(*args, **kwargs)
        os._exit(0)  # between the helper's exit and the owner's own cleanup
        return code

    recorder.writer.wait = exit_after_save
    recorder.close()


def test_a_recovery_helper_never_ends_a_live_producer_s_capture(tmp_path):
    """A helper that is not the owner's child checks the recorded producer's
    identity: while it runs, the helper waits for its close instead of ending
    the ring, and only a saved capture's staging is removed."""
    import threading

    from vllm.kv_profiling.sink import writer_main

    recorder = ProcessRecorder(_small(tmp_path), start_writer=False)
    recorder.record(*fields(request="live"))
    helper = threading.Thread(
        target=writer_main,
        args=(str(recorder.ring_dir), str(recorder.output_dir), "end", 256),
    )
    helper.start()
    time.sleep(1.5)  # past the helper's once-a-second identity check
    assert recorder.native.stats()["closing"] == 0 and helper.is_alive()
    assert recorder.ring_dir.exists()
    # the public close awaits the attached outside writer within its deadline
    result = recorder.close()
    helper.join(10)
    assert not helper.is_alive() and result.saved and result.complete
    manifest = json.loads((recorder.output_dir / "manifest.json").read_text())
    assert manifest["saved"] and manifest["reason"] != "producer_gone"
    assert not recorder.ring_dir.exists()


def test_a_dead_producer_s_recovery_keeps_output_written_inside_its_ring_dir(
    tmp_path,
):
    """Recovering a dead producer's ring into a directory inside the ring's own
    staging directory removes only the ring and its metadata."""
    import multiprocessing

    from vllm.kv_profiling.sink import writer_main

    location = tmp_path / "location"
    producer = multiprocessing.get_context("spawn").Process(
        target=_producer_without_helper, args=(_small(tmp_path), str(location))
    )
    producer.start()
    producer.join(15)
    assert producer.exitcode == 0
    ring_dir = Path(location.read_text())
    output = ring_dir / "recovered"
    writer_main(str(ring_dir), str(output), "end", 256)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["saved"] and manifest["reason"] == "producer_gone"
    assert (output / "events.jsonl").stat().st_size > 0
    assert sorted(p.name for p in ring_dir.iterdir()) == ["recovered"]
    shutil.rmtree(ring_dir)


def test_a_saved_capture_s_ring_is_freed_even_if_its_owner_dies_mid_close(
    tmp_path,
):
    """The helper removes a saved capture's staging itself: an owner killed
    between its helper's exit and its own cleanup leaves no ring behind."""
    import multiprocessing

    location = tmp_path / "location"
    owner = multiprocessing.get_context("spawn").Process(
        target=_owner_dies_right_after_its_helper,
        args=(_small(tmp_path), str(location)),
    )
    owner.start()
    owner.join(20)
    assert owner.exitcode == 0
    ring_dir, output = map(Path, json.loads(location.read_text()))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["saved"] and manifest["complete"]
    assert not ring_dir.exists()


@pytest.mark.parametrize("stage", ["directory", "metadata"])
def test_every_shared_memory_startup_stage_explains_a_full_tmpfs(
    tmp_path, monkeypatch, stage
):
    """No room while creating the ring's directory or writing its metadata is
    the same clear error as for the ring, and an allocated ring is released at
    once, not when the exception is collected."""
    import errno
    import tempfile

    from vllm import _kv_profiler

    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    shared = Path(tempfile.mkdtemp(prefix="kvprof-test-", dir="/dev/shm"))
    created = []
    if stage == "directory":
        monkeypatch.setattr(tempfile, "mkdtemp", full)
    else:
        create = _kv_profiler.create

        def tracked(*args):
            created.append(create(*args))
            return created[-1]

        write_text = Path.write_text

        def write(path, *args, **kwargs):
            if path.name == "metadata.json":
                full()
            return write_text(path, *args, **kwargs)

        monkeypatch.setattr(_kv_profiler, "create", tracked)
        monkeypatch.setattr(Path, "write_text", write)
    try:
        with pytest.raises(OSError) as raised:
            ProcessRecorder(_small(tmp_path, shared_dir=str(shared)))
        assert raised.value.errno == errno.ENOSPC
        assert f"no room for a recorder's ring in {shared}" in str(raised.value)
        assert list(shared.iterdir()) == []
        for ring in created:
            with pytest.raises(ValueError, match="closed ring"):
                ring.stats()
        assert len(created) == (stage == "metadata")
    finally:
        shutil.rmtree(shared)


def test_an_output_failure_at_close_keeps_the_ring_and_publishes_no_success(
    tmp_path, monkeypatch
):
    """Success (the saved manifest, FINISHED, removing the staging) is published
    only after the output stream closed cleanly."""
    import errno
    import io
    import threading

    from vllm.kv_profiling import sink as sink_module
    from vllm.kv_profiling.sink import OUTPUT_FAILED, writer_main

    class FailingClose(io.FileIO):
        failed = False

        def close(self):
            super().close()
            if not FailingClose.failed:
                FailingClose.failed = True
                raise OSError(errno.EIO, "injected close failure")

    real_open = sink_module.Path.open

    def open_events(path, mode="r", buffering=-1, *args, **kwargs):
        if path.name == "events.jsonl":
            return FailingClose(path, "a+")
        return real_open(path, mode, buffering, *args, **kwargs)

    monkeypatch.setattr(sink_module.Path, "open", open_events)
    recorder = ProcessRecorder(_small(tmp_path), start_writer=False)
    recorder.record(*fields(request="close-fails"))
    helper = threading.Thread(
        target=writer_main,
        args=(str(recorder.ring_dir), str(recorder.output_dir), "end", 256),
    )
    helper.start()
    wait_for(lambda: recorder.native.stats()["writer_state"] != 0)  # attached
    result = recorder.close()
    helper.join(10)
    manifest = json.loads((recorder.output_dir / "manifest.json").read_text())
    assert FailingClose.failed and not manifest["saved"]
    assert result.stats["writer_state"] == OUTPUT_FAILED and not result.saved
    assert (recorder.ring_dir / "ring").exists()
    shutil.rmtree(recorder.ring_dir)


def test_the_producer_is_judged_by_identity_and_an_unverifiable_one_is_kept(
    tmp_path, monkeypatch
):
    """Exited covers a zombie and a reused PID; an unreadable status or another
    PID namespace is unverifiable, and a recovery helper then never ends the
    ring: it keeps it and says why."""
    import subprocess
    import sys

    from vllm.kv_profiling import sink as sink_module
    from vllm.kv_profiling.sink import (
        ALIVE,
        EXITED,
        UNVERIFIABLE,
        WriterRefused,
        producer_state,
        writer_main,
    )

    def identity(pid):
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": int(stat[19]),
                "pid_namespace": os.readlink("/proc/self/ns/pid")}  # fmt: skip

    own = identity(os.getpid())
    assert producer_state(own) == ALIVE
    assert producer_state({**own, "start_ticks": own["start_ticks"] + 1}) == EXITED
    assert producer_state({**own, "pid_namespace": "pid:[1]"}) == UNVERIFIABLE
    child = subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE)
    try:
        exited = identity(child.pid)
        child.stdin.close()  # it exits, unreaped: a zombie
        wait_for(lambda: Path(f"/proc/{child.pid}/stat").read_text()
                 .rsplit(")", 1)[1].split()[0] == "Z")  # fmt: skip
        assert producer_state(exited) == EXITED
    finally:
        child.wait(10)
    assert producer_state(exited) == EXITED  # reaped: gone
    real_read = sink_module.Path.read_text

    def denied(path, *args, **kwargs):
        if str(path) == f"/proc/{os.getpid()}/stat":
            raise PermissionError("injected")
        return real_read(path, *args, **kwargs)

    real_readlink = sink_module.os.readlink

    def no_namespace_link(path, *args, **kwargs):
        if str(path) == "/proc/self/ns/pid":
            raise FileNotFoundError(path)
        return real_readlink(path, *args, **kwargs)

    monkeypatch.setattr(sink_module.os, "readlink", no_namespace_link)
    # the helper's own namespace is unknown: a missing PID proves nothing
    assert producer_state(own) == UNVERIFIABLE
    assert producer_state({**own, "pid": 2**22 + 1}) == UNVERIFIABLE
    monkeypatch.setattr(sink_module.os, "readlink", real_readlink)
    monkeypatch.setattr(sink_module.Path, "read_text", denied)
    assert producer_state(own) == UNVERIFIABLE
    recorder = ProcessRecorder(_small(tmp_path), start_writer=False)
    recorder.record(*fields(request="kept"))
    with pytest.raises(WriterRefused, match="producer_unverifiable"):
        writer_main(str(recorder.ring_dir), str(recorder.output_dir), "end", 256)
    # refused before the lease and any write: no manifest, the ring untouched
    assert not (recorder.output_dir / "manifest.json").exists()
    assert recorder.native.stats()["closing"] == 0
    assert (recorder.ring_dir / "ring").exists()
    monkeypatch.undo()
    recorder.close()
    shutil.rmtree(recorder.ring_dir, ignore_errors=True)


def test_a_refused_attempt_never_replaces_the_capture_s_manifest(
    tmp_path, monkeypatch
):
    """Only the lease holder publishes the manifest: an attempt refused for a
    held lease or an unverifiable producer leaves the writer's result alone."""
    from vllm.kv_profiling import sink as sink_module
    from vllm.kv_profiling.sink import UNVERIFIABLE, WriterRefused, writer_main

    recorder = ProcessRecorder(_small(tmp_path))
    assert recorder.wait_ready()
    recorder.record(*fields(request="kept"))
    manifest = recorder.output_dir / "manifest.json"
    wait_for(manifest.exists)
    before = manifest.read_bytes()
    arguments = (str(recorder.ring_dir), str(recorder.output_dir), "end", 256)
    with pytest.raises(WriterRefused, match="reader_lease_held"):
        writer_main(*arguments)
    monkeypatch.setattr(sink_module, "producer_state", lambda metadata: UNVERIFIABLE)
    with pytest.raises(WriterRefused, match="producer_unverifiable"):
        writer_main(*arguments)
    assert manifest.read_bytes() == before
    monkeypatch.undo()
    result = recorder.close()
    saved = json.loads(manifest.read_text())
    assert result.complete and saved["saved"] and saved["complete"]


def test_a_live_capture_is_resumed_only_in_its_own_directory(tmp_path):
    """A replacement writer for a live producer must resume the capture's own
    directory, so the saved result and its files agree."""
    from vllm.kv_profiling.sink import WriterRefused, writer_main

    recorder = ProcessRecorder(_small(tmp_path), start_writer=False)
    elsewhere = tmp_path / "elsewhere"
    with pytest.raises(WriterRefused, match="live_capture_elsewhere"):
        writer_main(str(recorder.ring_dir), str(elsewhere), "end", 256)
    assert not elsewhere.exists()
    recorder.close()
    shutil.rmtree(recorder.ring_dir, ignore_errors=True)


@pytest.mark.parametrize("released", [True, False])
def test_close_waits_from_the_lease_and_times_out_a_writer_still_working(
    tmp_path, monkeypatch, released
):
    """Public close waits for a writer from the moment it holds the lease,
    whatever state its predecessor left; one still working at the deadline is
    a timeout."""
    import threading

    from vllm.kv_profiling import sink as sink_module
    from vllm.kv_profiling.sink import OUTPUT_FAILED, writer_main

    gate = threading.Event()
    real = sink_module.atomic_json

    def slow_start(path, value):
        if value.get("reason") == "open":
            gate.wait(20)  # the writer holds the lease, its output not open yet
        real(path, value)

    monkeypatch.setattr(sink_module, "atomic_json", slow_start)
    recorder = ProcessRecorder(
        _small(tmp_path, shutdown_timeout_s=1), start_writer=False
    )
    recorder.record(*fields(request="slow"))
    recorder.native.set_writer_state(OUTPUT_FAILED)  # left by a predecessor
    helper = threading.Thread(
        target=writer_main,
        args=(str(recorder.ring_dir), str(recorder.output_dir), "end", 256),
    )
    helper.start()
    wait_for(recorder._reader_attached)
    if released:
        threading.Timer(0.3, gate.set).start()
    result = recorder.close()
    gate.set()
    helper.join(10)
    if released:
        assert result.saved and result.complete
    else:
        assert not result.saved and result.reason == "writer_timeout"
    shutil.rmtree(recorder.ring_dir, ignore_errors=True)


def test_a_relative_output_dir_is_anchored_when_the_recorder_opens(
    tmp_path, monkeypatch
):
    """The capture directory is resolved once, at creation: a writer restarted
    after the working directory changed resumes the same directory, so every
    record and the saved result stay together."""
    import dataclasses

    monkeypatch.chdir(tmp_path)
    recorder = ProcessRecorder(
        dataclasses.replace(_small(tmp_path), output_dir="profiles")
    )
    assert recorder.wait_ready()
    assert recorder.output_dir.is_absolute()
    assert str(recorder.output_dir).startswith(str(tmp_path / "profiles"))
    recorder.record(*fields(request="first"))
    os.kill(recorder.writer.pid, signal.SIGKILL)
    recorder.writer.wait(5)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    recorder.start_writer()
    assert recorder.wait_ready()
    recorder.record(*fields(request="second"))
    result = recorder.close()
    assert result.saved and Path(result.output_dir) == recorder.output_dir
    lines = (recorder.output_dir / "events.jsonl").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    assert sum(row.get("type") == "event" for row in events) == 2
    assert not (elsewhere / "profiles").exists()


def test_a_missing_native_recorder_says_why_and_how_to_build_it(
    tmp_path, monkeypatch
):
    """An enabled profiler without the native extension (a precompiled install)
    fails with an error naming the cause and the build into the imported
    package, before anything is created in shared memory."""
    import sys

    import vllm

    monkeypatch.setitem(sys.modules, "vllm._kv_profiler", None)
    monkeypatch.delattr(vllm, "_kv_profiler", raising=False)  # set by earlier imports
    monkeypatch.setattr(sys, "executable", "/opt/my env/bin/python")
    shared = tmp_path / "shm"
    shared.mkdir()
    with pytest.raises(ImportError) as raised:
        ProcessRecorder(_small(tmp_path, shared_dir=str(shared)))
    message = str(raised.value)
    package = Path(vllm.__file__).resolve().parent
    for part in (
        "native recorder (vllm._kv_profiler) is not installed",
        "VLLM_USE_PRECOMPILED=1",
        "cmake --install",
    ):
        assert part in message, (part, message)
    # each argument shell-quoted: a path with a space stays one argument
    import shlex

    command = message.split("Build it from the source tree with: ", 1)[1]
    first = shlex.split(command.split(" && ", 1)[0])
    assert first == [
        "cmake", "-S", "csrc/kv_profiler", "-B", "build/kv-profiler",
        "-DPython_EXECUTABLE=/opt/my env/bin/python",
        f"-DCMAKE_INSTALL_PREFIX={package.parent}",
    ]  # fmt: skip
    assert raised.value.name == "vllm._kv_profiler"
    assert list(shared.iterdir()) == []


def test_an_unloadable_native_recorder_keeps_its_loader_error(tmp_path, monkeypatch):
    """A present extension the loader rejects is not reported as missing: its
    own error comes through, without the installation hint."""
    import sys

    import vllm

    broken = tmp_path / "vllm"
    broken.mkdir()
    (broken / "_kv_profiler.abi3.so").write_bytes(b"not an ELF")
    monkeypatch.delitem(sys.modules, "vllm._kv_profiler", raising=False)
    monkeypatch.delattr(vllm, "_kv_profiler", raising=False)
    monkeypatch.setattr(vllm, "__path__", [str(broken), *vllm.__path__])
    with pytest.raises(ImportError) as raised:
        ProcessRecorder(_small(tmp_path))
    assert not isinstance(raised.value, ModuleNotFoundError)
    assert "not installed" not in str(raised.value)
    assert "_kv_profiler.abi3.so" in str(raised.value)


def test_the_transport_is_not_a_setting():
    """The transport is derived per recorder; the old manual label is refused
    like any unknown key."""
    from vllm.kv_profiling.config import profiler_config

    with pytest.raises(ValueError, match="unknown network_profiler keys"):
        profiler_config({"network_profiler": {"level": "l1", "transport": "tcp"}}, {})


_TENT_TCP = {"settings": {"transports": {
    "rdma": {"enable": False}, "tcp": {"enable": False}, "hp_tcp": {"enable": True}}}}
_TENT_RDMA = {"settings": {"transports": {
    "rdma": {"enable": True}, "tcp": {"enable": False}, "hp_tcp": {"enable": False}}}}


@pytest.mark.parametrize(
    "backends, connector, environment, tent, extra, candidates",
    [
        # UCX: allowed network transports only; UCX still selects at runtime
        (["UCX"], "nixl", {"UCX_TLS": "tcp,self"}, None, None, ["tcp"]),
        (["UCX"], "nixl", {"UCX_TLS": "rc,self"}, None, None, ["rdma"]),
        (["UCX"], "nixl", {"UCX_TLS": "tcp,rc_x,sm,cuda_copy"}, None, None,
         ["rdma", "tcp"]),
        (["UCX"], "nixl", {}, None, None, None),
        (["UCX"], "nixl", {"UCX_TLS": "all"}, None, None, None),
        (["UCX"], "nixl", {"UCX_TLS": "^tcp"}, None, None, None),
        (["UCX"], "nixl", {"UCX_TLS": "tcp,unheard"}, None, None, None),
        # LIBFABRIC: a provider filter
        (["LIBFABRIC"], "nixl", {"FI_PROVIDER": "tcp"}, None, None, ["tcp"]),
        (["LIBFABRIC"], "nixl", {"FI_PROVIDER": "tcp;ofi_rxm"}, None, None,
         ["tcp"]),
        (["LIBFABRIC"], "nixl", {"FI_PROVIDER": "verbs,tcp"}, None, None,
         ["rdma", "tcp"]),
        (["LIBFABRIC"], "nixl", {"FI_PROVIDER": "udp"}, None, None, []),
        (["LIBFABRIC"], "nixl", {"FI_PROVIDER": "psm3"}, None, None, None),
        (["LIBFABRIC"], "nixl", {"FI_PROVIDER": "tcp,psm3"}, None, None, None),
        (["LIBFABRIC"], "nixl", {}, None, None, None),
        # Mooncake over TENT (either variable present, any value): its network
        # transports enabled, by TENT's defaults (tcp, rdma on; hp_tcp off)
        (["Mooncake"], "mooncake", {"MC_USE_TENT": "1"}, _TENT_TCP, None,
         ["tcp"]),
        (["Mooncake"], "mooncake", {"MC_USE_TENT": "1"}, _TENT_RDMA, None,
         ["rdma"]),
        (["Mooncake"], "mooncake", {"MC_USE_TENT": "0"}, _TENT_RDMA, None,
         ["rdma"]),
        (["Mooncake"], "mooncake", {"MC_USE_TEV1": ""}, _TENT_RDMA, None,
         ["rdma"]),
        (["Mooncake"], "mooncake", {"MC_USE_TENT": "1"},
         {"settings": {"transports": {"rdma": {"enable": False}}}}, None,
         ["tcp"]),
        (["Mooncake"], "mooncake", {"MC_USE_TENT": "1"},
         {"settings": {"transports": {"rdma": {"enable": False},
                                      "tcp": {"enable": False}}}}, None, []),
        (["Mooncake"], "mooncake", {"MC_USE_TENT": "1"},
         {"settings": {"transports/rdma/enable": False}}, None, ["tcp"]),
        # tcp and rdma read through Config::get: a mistyped enable is the default
        (["Mooncake"], "mooncake", {"MC_USE_TENT": "1"},
         {"settings": {"transports": {"tcp": {"enable": "no"},
                                      "rdma": {"enable": False}}}}, None,
         ["tcp"]),
        # hp_tcp reads only its object, nested or under the exact flat key
        (["Mooncake"], "mooncake", {"MC_USE_TENT": "1"},
         {"settings": {"transports/tcp/enable": False,
                       "transports/rdma/enable": False,
                       "transports/hp_tcp": {"enable": True}}}, None, ["tcp"]),
        (["Mooncake"], "mooncake", {"MC_USE_TENT": "1"},
         {"settings": {"transports/tcp/enable": False,
                       "transports/rdma/enable": False,
                       "transports/hp_tcp/enable": True}}, None, []),
        (["Mooncake"], "mooncake", {"MC_USE_TENT": "1"},
         {"settings": {"transports": {"hp_tcp": {"enable": "yes"}}}}, None,
         None),
        (["Mooncake"], "mooncake", {"MC_USE_TENT": "1"},
         {"settings": {"transports": {"hp_tcp": True}}}, None, None),
        # a mistyped nested enable falls back to the exact flat key first
        (["Mooncake"], "mooncake", {"MC_USE_TENT": "1"},
         {"settings": {"transports": {"tcp": {"enable": "auto"},
                                      "rdma": {"enable": "auto"}},
                       "transports/tcp/enable": False,
                       "transports/rdma/enable": False}}, None, []),
        (["Mooncake"], "nixl", {"MC_USE_TENT": "1"},
         {"settings": {"transports": {"tcp": {"enable": "auto"}},
                       "transports/tcp/enable": False,
                       "transports/rdma/enable": False}}, None, []),
        # an hp_tcp enable that is present must be a boolean: null is invalid
        (["Mooncake"], "mooncake", {"MC_USE_TENT": "1"},
         {"settings": {"transports": {"hp_tcp": {"enable": None}}}}, None,
         None),
        (["Mooncake"], "mooncake", {"MC_USE_TENT": "1"},
         {"settings": {"transports": {"tcp": {"enable": False},
                                      "rdma": {"enable": False},
                                      "hp_tcp": {"port": 1}}}}, None, []),
        (["Mooncake"], "mooncake", {"MC_USE_TENT": "1"}, None, None, None),
        (["Mooncake"], "mooncake", {"MC_USE_TENT": "1"},
         {"settings": "unreadable"}, None, None),
        (["Mooncake"], "nixl", {"MC_USE_TENT": "1"}, _TENT_TCP, None, ["tcp"]),
        # the classic engine discovers its transport (whatever the protocol)
        # unless MC_FORCE_TCP forces TCP: both connector paths
        (["Mooncake"], "mooncake", {}, None, {"mooncake_protocol": "tcp"},
         None),
        (["Mooncake"], "mooncake", {}, None, {}, None),
        (["Mooncake"], "mooncake", {"MC_FORCE_TCP": "1"}, None, {}, ["tcp"]),
        (["Mooncake"], "nixl", {}, _TENT_TCP, None, None),
        (["Mooncake"], "nixl", {"MC_FORCE_TCP": ""}, None, None, ["tcp"]),
        (["POSIX"], "nixl", {"UCX_TLS": "tcp"}, None, None, None),
    ],
)  # fmt: skip
def test_the_transport_is_derived_per_backend_as_candidates(
    backends, connector, environment, tent, extra, candidates
):
    """Each backend's configuration gives only candidates, with the setting it
    came from and why it is no more."""
    from vllm.kv_profiling.transport import derive

    (entry,) = derive(backends, connector, environment, tent, extra)
    assert entry["backend"] == backends[0]
    assert entry["candidates"] == candidates
    assert entry["reason"] and "source" in entry and "kind" not in entry


def test_several_backends_each_get_their_own_entry():
    from vllm.kv_profiling.transport import combined, derive

    entries = derive(["UCX", "LIBFABRIC"], "nixl",
                     {"UCX_TLS": "tcp", "FI_PROVIDER": "verbs"})  # fmt: skip
    assert [(e["backend"], e["candidates"]) for e in entries] == [
        ("UCX", ["tcp"]), ("LIBFABRIC", ["rdma"])]  # fmt: skip
    assert combined([{"kind": "tcp"}, {"kind": "tcp"}]) == "tcp"
    assert combined([{"kind": "tcp"}, {"kind": "unknown"}]) == "unknown"
    assert combined([{"kind": "tcp"}, {"kind": "rdma"}]) == "mixed"
    assert combined([]) == "unknown"


def test_a_long_or_secret_source_is_bounded_and_redacted():
    """A derived value is recorded as the configuration snapshot records one:
    bounded and redacted, so the recorder's metadata stays openable."""
    from vllm.kv_profiling.transport import MAX_TEXT, derive

    many = ",".join(f"unheard{i}" for i in range(2000))
    (entry,) = derive(["UCX"], "nixl", {"UCX_TLS": many})
    assert entry["candidates"] is None
    assert len(entry["source"]["value"]) < MAX_TEXT + 40
    assert len(entry["reason"]) < MAX_TEXT + 40
    (entry,) = derive(["LIBFABRIC"], "nixl",
                      {"FI_PROVIDER": "https://user:pw@host/x"})  # fmt: skip
    assert "pw" not in json.dumps(entry)


def test_a_recorder_with_a_long_transport_setting_still_opens(tmp_path, monkeypatch):
    from vllm.kv_profiling.nixl import _build

    monkeypatch.setenv("UCX_TLS", ",".join(f"unheard{i}" for i in range(5000)))
    config = NetworkProfilerConfig(
        level="l1", run_id=str(uuid4()), output_dir=str(tmp_path),
        context={"role": "worker", "backends": ["UCX"]},
    )  # fmt: skip
    profiler = _build(config, "worker", "engine-A", 0, None, "kv_consumer")
    profiler.close()
    meta = json.loads((profiler.output_dir / "events.jsonl").read_text()
                      .splitlines()[0])  # fmt: skip
    assert meta["context"]["transport"][0]["candidates"] is None


def test_an_invalid_enable_records_its_type_within_bounds():
    """An invalid hp_tcp enable is recorded by its type, never its value: a
    huge one stays within the recorder's bounds."""
    from vllm.kv_profiling.transport import derive

    tent = {"settings": {"transports": {"hp_tcp": {"enable": [0] * 100000},
                                        "tcp": {"enable": [1] * 100000}}}}
    (entry,) = derive(["Mooncake"], "mooncake", {"MC_USE_TENT": "1"}, tent)
    assert entry["candidates"] is None
    assert entry["source"]["value"]["hp_tcp.enable"] == "invalid: list"
    assert len(json.dumps(entry)) < 2048
