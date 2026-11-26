# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract: stdlib CLI samples real endpoints without affecting their owner.

Failures guarded: missing/partial observations mistaken for zero, aggregate
counters labelled as transfers, PID/clock reuse, and false successful shutdown.
The public CLI and real sockets/files/processes are the cheapest real entry point.
"""

import errno
import json
import os
import shlex
import signal
import socket
import struct
import subprocess
import sys
import sysconfig
import time
from dataclasses import asdict
from pathlib import Path
from unittest import mock
from uuid import uuid4

import pytest

from vllm.kv_profiling import probe
from vllm.kv_profiling.config import NetworkProfilerConfig
from vllm.kv_profiling.core import ProcessRecorder
from vllm.kv_profiling.endpoint_probes import netdev, rdma, tcp
from vllm.kv_profiling.endpoint_probes.config import ProbeConfig
from vllm.kv_profiling.endpoint_probes.identity import process_identity
from vllm.kv_profiling.endpoint_probes.netlink import (
    ObservationError,
    attribute,
    attributes,
    messages,
)
from vllm.kv_profiling.endpoint_probes.records import Writer, event, validate
from vllm.kv_profiling.perfetto import export_traces
from vllm.kv_profiling.probe_join import candidates
from vllm.kv_profiling.schema import EventKind


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.005)


def _me(role="worker", pid=None, **identity):
    """An owner as the observer builds it from a recorder's metadata."""
    pid = pid or os.getpid()
    return {**process_identity(pid), "role": role, **identity}


def _owned_by_inode(scan):
    return {record["inode"]: record for record in scan.owned.values()}


@pytest.mark.parametrize(
    "family,address", [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")]
)
def test_discovery_reports_an_owned_socket_with_its_cookie_and_tcp_info(
    family, address
):
    """The dump that decides ownership carries each socket's cookie and
    TCP_INFO, which the TCP source samples."""
    from vllm.kv_profiling.endpoint_probes.sockets import Discovery

    with socket.socket(family) as listener, socket.socket(family) as client:
        listener.bind((address, 0))
        listener.listen()
        client.connect(listener.getsockname())
        server, _ = listener.accept()
        with server:
            client.sendall(b"payload")
            assert server.recv(7) == b"payload"
            scan = Discovery([_me()]).scan()
            client_inode = os.fstat(client.fileno()).st_ino
            listener_inode = os.fstat(listener.fileno()).st_ino
    assert scan.complete, scan.reason
    owned = _owned_by_inode(scan)
    record = owned[client_inode]
    assert record["cookie"] > 0 and record["loopback"] and not record["listening"]
    assert record["owners"][0]["pid"] == os.getpid()
    row = record["row"]
    assert row["tcp_info_length"] >= 104
    assert row["fields"]["bytes_acked"]["value"] >= 1
    assert row["fields"]["rtt_us"]["unit"] == "us"
    assert owned[listener_inode]["listening"]


_HOLDER = """
import socket, sys, time
s = socket.create_connection(("127.0.0.1", int(sys.argv[1])))
print(__import__("os").fstat(s.fileno()).st_ino, flush=True)
time.sleep(30)
"""


def test_a_frontend_socket_on_the_same_address_pair_is_never_owned():
    """A worker and a frontend each connect to one listener over the same
    addresses: only the worker's socket is owned (frontends own none), and a
    socket the observer's own process holds is a recorded holder only."""
    from vllm.kv_profiling.endpoint_probes.sockets import Discovery

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        port = listener.getsockname()[1]
        worker, frontend = (
            subprocess.Popen(
                [sys.executable, "-c", _HOLDER, str(port)],
                stdout=subprocess.PIPE,
                text=True,
            )  # fmt: skip
            for _ in range(2)
        )
        try:
            inodes = {p: int(p.stdout.readline()) for p in (worker, frontend)}
            accepted = [listener.accept()[0] for _ in range(2)]
            scan = Discovery([_me("worker", worker.pid),
                              _me("frontend", frontend.pid)]).scan()  # fmt: skip
            for sock in accepted:
                sock.close()
        finally:
            for p in (worker, frontend):
                p.kill()
                p.wait()
    assert scan.complete, scan.reason
    owned = _owned_by_inode(scan)
    assert set(owned) == {inodes[worker]}
    record = owned[inodes[worker]]
    assert [o["role"] for o in record["owners"]] == ["worker"]
    assert record["peer_port"] == port and not record["shared"]


def test_a_listener_child_is_owned_only_once_accepted():
    """A connection to a worker's listening port is not the worker's until
    accept() puts it in its fd table; the listener itself is."""
    from vllm.kv_profiling.endpoint_probes.sockets import Discovery

    with socket.socket() as listener, socket.socket() as client:
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        port = listener.getsockname()[1]
        discovery = Discovery([_me()])
        client_inode = os.fstat(client.fileno()).st_ino
        # the client is the test's too: drop it from what counts
        client.connect(listener.getsockname())
        before = discovery.scan()
        server, _ = listener.accept()
        with server:
            after = discovery.scan()
            server_inode = os.fstat(server.fileno()).st_ino
    pending = [
        r
        for r in before.owned.values()
        if r["local_port"] == port and not r["listening"]
    ]
    assert pending == []
    assert server_inode in _owned_by_inode(after)
    assert client_inode in _owned_by_inode(after)


def test_empty_discovery_owns_nothing():
    """An owner without sockets, or no scheduler or worker at all: an empty,
    complete scan (it never falls back to every socket)."""
    from vllm.kv_profiling.endpoint_probes.sockets import Discovery, Tracker

    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        for owners in ([_me("worker", sleeper.pid)], [_me("frontend")], []):
            scan = Discovery(owners).scan()
            assert scan.complete and scan.owned == {}, owners
            tracker = Tracker()
            assert tracker.apply(scan) == [] and tracker.active == {}
    finally:
        sleeper.kill()
        sleeper.wait()


def _scan(owned, complete=True, unreadable=None, reason=None):
    from vllm.kv_profiling.endpoint_probes.sockets import Scan

    scan = Scan(time.monotonic_ns())
    scan.complete, scan.reason = complete, reason
    scan.unreadable = unreadable or {}
    scan.finished_ns = time.monotonic_ns()

    def record(cookie, owners):
        pids, holders = owners if isinstance(owners, tuple) else (owners, [])
        return {"cookie": cookie, "inode": cookie, "owner_pids": sorted(pids),
                "owners": [{"pid": pid, "start_ticks": 1, "role": "worker"}
                           for pid in pids],
                "holders": [{"pid": h, "start_ticks": 1} for h in holders],
                "shared": len(pids) > 1 or bool(holders), "row": {},
                "frontend_colocated": False, "loopback": True}  # fmt: skip

    scan.owned = {cookie: record(cookie, owners) for cookie, owners in owned.items()}
    return scan


def test_the_tracker_admits_removes_suspends_and_keeps_every_admission():
    """Admissions are immutable: a change of owning processes or observed
    holders is a new admission superseding the old, whose record stays as it
    was for late events."""
    from vllm.kv_profiling.endpoint_probes.sockets import Tracker

    tracker = Tracker()

    def kinds(changes):
        return [(c["change"], c["cookie"]) for c in changes]

    assert kinds(tracker.apply(_scan({10: [1], 11: [1, 2]}))) == [
        ("admitted", 10), ("admitted", 11)]  # fmt: skip
    first = dict(tracker.active)
    # owner 1 unreadable: 10 is removed with that reason; 11 is kept, owner 2
    # still holding it, under a new admission
    changes = tracker.apply(_scan({11: [2]}, unreadable={1: "EACCES"}))
    assert kinds(changes) == [("removed", 10), ("reversioned", 11), ("admitted", 11)]
    assert changes[0]["reason"] == "owner unreadable"
    assert changes[2]["supersedes"] == first[11]
    assert tracker.admissions[first[11]]["owner_pids"] == [1, 2]  # as admitted
    second = tracker.active[11]
    assert tracker.admissions[second]["owner_pids"] == [2]
    # a newly observed outside holder: another new admission
    changes = tracker.apply(_scan({11: ([2], [99])}))
    assert kinds(changes) == [("reversioned", 11), ("admitted", 11)]
    assert tracker.admissions[second]["holders"] == []
    # an unchanged scan changes nothing
    assert tracker.apply(_scan({11: ([2], [99])})) == []
    # an interrupted dump suspends every admission at once
    changes = tracker.apply(_scan({}, complete=False, reason="netlink deadline"))
    assert kinds(changes) == [("suspended", 11)] and tracker.active == {}
    # confirmed again: a new admission; the old ones stay resolvable
    changes = tracker.apply(_scan({11: [2]}))
    assert kinds(changes) == [("admitted", 11)]
    assert tracker.active[11] not in (first[11], second)
    assert {first[11], second} <= set(tracker.admissions)


def test_an_unreadable_owner_and_an_interrupted_dump_admit_nothing(monkeypatch):
    from vllm.kv_profiling.endpoint_probes import sockets

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        stale = {**_me(), "start_ticks": _me()["start_ticks"] - 1}
        scan = sockets.Discovery([stale]).scan()
        assert scan.complete and scan.owned == {}
        assert os.getpid() in scan.exited  # another process under its PID

        def interrupted(budget, unavailable):
            raise ObservationError("netlink deadline")

        monkeypatch.setattr(sockets, "_dump", interrupted)
        scan = sockets.Discovery([_me()]).scan()
    assert not scan.complete and "netlink deadline" in scan.reason
    assert scan.owned == {}


def test_discovery_is_bounded_as_a_whole_and_per_process():
    from vllm.kv_profiling.endpoint_probes.sockets import Discovery, Limits

    with socket.socket(), socket.socket():
        for name, limits in (
            ("per-process", Limits(fds_per_process=1)),
            ("total", Limits(total_fds=1)),
            ("deadline", Limits(deadline_s=0)),
        ):
            scan = Discovery([_me()], limits).scan()
            assert not scan.complete and scan.reason, name
            assert scan.owned == {}, name
        # other holders are best effort: their bound truncates them only
        scan = Discovery([_me()], Limits(processes=0)).scan()
        assert scan.complete and scan.holders_truncated == "discovery bound"


@pytest.mark.parametrize(
    "data", [b"x", struct.pack("=HH", 3, 1), struct.pack("=HH", 8, 1)]
)
def test_partial_attributes_are_not_empty_results(data):
    with pytest.raises(ObservationError):
        list(attributes(data))


def test_partial_and_interrupted_dumps_are_not_complete():
    with pytest.raises(ObservationError, match="interrupted"):
        list(messages(struct.pack("=IHHII", 16, 20, 0x10, 1, 0), 1))
    with pytest.raises(ObservationError, match="sequence"):
        list(messages(struct.pack("=IHHII", 16, 20, 0, 2, 0), 1))
    assert list(attributes(attribute(2, b"abc"))) == [(2, b"abc")]


def test_short_tcp_info_and_rate_sentinel_remain_explicit():
    assert tcp.tcp_info(bytes(68))["rtt_us"]["available"] is False
    data = bytearray(112)
    struct.pack_into("=Q", data, 104, (1 << 64) - 1)
    result = tcp.tcp_info(data)
    assert result["pacing_rate_Bps"]["sentinel"] == "unlimited_or_unset"
    assert result["bytes_received"]["value"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval_s": float("nan")},
        {"interval_s": 0},
        {"queue_rows": 513},
        {"tcp": {"local_ips": ["127.0.0.1"]}},
        {"tcp": {"owned_sockets": True, "peer_ips": ["127.0.0.1"]}},
        {"tcp": {"local_ports": [1]}},
        {"tcp": {"owned_sockets": "yes"}},
        {"discovery_interval_s": 0.01},
        {"interfaces": ["../net"]},
        {"required": ["tcp"]},
        {"qp": True},
        {"driver_statistics": True},
    ],
)
def test_probe_config_refuses_unbounded_or_implicit_sources(kwargs):
    with pytest.raises(ValueError):
        ProbeConfig(**kwargs)


def test_rdma_port_preserves_raw_units_and_refresh_configuration(tmp_path):
    device = tmp_path / "test0"
    base = device / "ports/1"
    (base / "counters").mkdir(parents=True)
    (base / "hw_counters").mkdir()
    for name in ("node_guid", "sys_image_guid"):
        (device / name).write_text("0000:0000:0000:0001")
    (base / "counters/port_xmit_data").write_text("123")
    (base / "hw_counters/lifespan").write_text("10")
    (base / "hw_counters/vendor").write_text("5")
    row = rdma.port({"device": "test0", "port": 1}, tmp_path)
    fields = {item["name"]: item for item in row["fields"]}
    assert fields["counters/port_xmit_data"]["converted_bytes"] == 492
    assert fields["counters/port_xmit_data"]["value"] == 123
    assert fields["hw_counters/lifespan"]["meaning"] == "refresh_configuration"
    assert fields["hw_counters/vendor"]["unit"] == "unknown"
    assert (base / "hw_counters/lifespan").read_text() == "10"
    with pytest.raises(FileNotFoundError):
        rdma.port({"device": "missing", "port": 1}, tmp_path)


def test_sysfs_reads_refuse_oversize_and_negative_values(tmp_path):
    path = tmp_path / "counter"
    path.write_text("-1")
    with pytest.raises(ObservationError, match="negative"):
        netdev.number(path)
    path.write_text("9" * 4097)
    with pytest.raises(ObservationError, match="oversize"):
        netdev.number(path)


def test_real_interface_read_does_not_equate_link_bytes_to_transfer_bytes():
    row = netdev.interface("lo")
    assert row["ifindex"] > 0
    assert row["fields"]["rx_bytes"]["meaning"] == "cumulative"
    assert "transfer" not in row


def metadata(run_id=None, **extra):
    return {
        "schema": 1,
        "probe_schema": 1,
        "type": "metadata",
        "run_id": run_id or str(uuid4()),
        "process_id": str(uuid4()),
        "clock_id": str(uuid4()),
        "clock_source": "CLOCK_MONOTONIC",
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "time_namespace": os.readlink("/proc/self/ns/time"),
        "net_namespace": os.readlink("/proc/self/ns/net"),
        "context": {"endpoint": "d", "endpoint_role": "decode"},
        **extra,
    }


def test_probe_writer_and_mixed_export_do_not_turn_counters_into_transfers(tmp_path):
    run_id = str(uuid4())
    config = NetworkProfilerConfig(
        level="l3", run_id=run_id, output_dir=str(tmp_path / "core"), ring_records=64
    )
    with_recorder = ProcessRecorder(config)
    assert with_recorder.wait_ready()
    assert with_recorder.close().complete
    meta = metadata(run_id)
    writer = Writer(tmp_path / "probe", meta, ProbeConfig())
    row = event("endpoint_metrics", "socket", "flow", 1, 2, {"rtt_us": 3}, "tcp")
    writer.put(row)
    assert writer.close(True, "test", {})
    results = export_traces(
        [with_recorder.output_dir / "events.jsonl", tmp_path / "probe/events.jsonl"],
        str(tmp_path / "export"),
    )
    documents = [json.loads(Path(path).read_text()) for path in results.values()]
    events = [
        row
        for doc in documents
        for row in doc["traceEvents"]
        if row.get("name") == "endpoint_metrics"
    ]
    assert len(events) == 1
    assert events[0]["args"]["transport"]["provenance"] == "kernel_observed"
    assert events[0]["args"]["bytes"] is None
    assert events[0]["args"]["endpoint_observation"]["entity"] == "flow"
    with pytest.raises(ValueError, match="transfer metrics"):
        validate({**row, "bytes": 3})
    with pytest.raises(ValueError, match="schema"):
        validate({**row, "probe_schema": 999})


def test_writer_byte_limit_leaves_manifest_incomplete(tmp_path):
    writer = Writer(tmp_path / "out", metadata(), ProbeConfig(output_bytes=4096))
    writer.put(event("probe_status", "source", "test", 1, 2, {"data": "x" * 6000}))
    assert not writer.close(True, "test", {})
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert not manifest["complete"]


def start_owner(cpu):
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
        stdin=subprocess.PIPE,
    )
    os.sched_setaffinity(child.pid, {cpu})
    return child


def start_recorded_endpoint(
    tmp_path, cpu, *, overlap_cpu=None, close_delay=0, run_id=None, connect=None
):
    directory = tmp_path / "endpoint"
    run_id = run_id or str(uuid4())
    command = [
        sys.executable,
        str(Path(__file__).with_name("profiled_endpoint.py")),
        "serve",
        "--directory",
        str(directory),
        "--run",
        run_id,
        "--cpu",
        str(cpu),
        "--close-delay",
        str(close_delay),
    ]
    if overlap_cpu is not None:
        command += ["--overlap-cpu", str(overlap_cpu)]
    if connect is not None:
        command += ["--connect", str(connect)]
    log = (tmp_path / "endpoint.log").open("w")
    process = subprocess.Popen(command, stdout=log, stderr=log)
    try:
        wait_for(
            lambda: (directory / "ready.json").exists() or process.poll() is not None,
            timeout=30,
        )
        assert process.poll() is None, (tmp_path / "endpoint.log").read_text()
        entries = json.loads((directory / "ready.json").read_text())
        return process, entries, directory
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        log.close()


@pytest.mark.parametrize(
    "termination,required",
    [
        ("controller", False),
        ("controller", True),
        ("owner_exit", False),
        ("owner_exit_ack", False),
    ],
)
def test_real_cli_separates_owner_and_child_completion(tmp_path, termination, required):
    allowed = sorted(os.sched_getaffinity(0))
    assert len(allowed) >= 2, "probe tests require separate owner/observer CPUs"
    owner, inventory, endpoint_dir = start_recorded_endpoint(tmp_path, allowed[0])
    run_id = inventory[0]["run_id"]
    core_dir = endpoint_dir / "profiles"
    (endpoint_dir / "close").touch()
    wait_for(lambda: (endpoint_dir / "closed").exists(), timeout=20)
    settings = ProbeConfig(
        interval_s=0.1,
        slow_interval_s=0.1,
        interfaces=["missing-device"] if required else ["lo"],
        driver_statistics=True,
        required=["netdev"],
    )
    config = {
        "probes": asdict(settings),
        "expected": {"frontend": 1, "scheduler": 1, "worker": 1},
        "run_id": run_id,
        "endpoint": "d",
        "role": "decode",
        "cpus": [allowed[-1]],
        "input_dir": str(core_dir),
        "output_dir": str(tmp_path / "output"),
        "irq_policy": "virtual_not_applicable",
        "stop_file": str(tmp_path / "stop"),
        "endpoint_complete_file": str(tmp_path / "children"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    process = subprocess.Popen(
        [sys.executable, str(Path(probe.__file__)), "--config", str(config_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        wait_for(lambda: bool(list((tmp_path / "output").glob("*/*/manifest.json"))))
        if termination == "controller":
            (tmp_path / "children").write_text(
                json.dumps({"run_id": run_id, "endpoint": "d", "children_closed": True})
            )
            (tmp_path / "stop").touch()
        else:
            (endpoint_dir / "exit").touch()
            owner.wait(10)
            if termination == "owner_exit_ack":

                def owner_exit_recorded():
                    paths = list((tmp_path / "output").glob("*/*/events.jsonl"))
                    return (
                        bool(paths) and '"owner_exited": true' in paths[0].read_text()
                    )

                wait_for(owner_exit_recorded, timeout=2)
                (tmp_path / "children").write_text(
                    json.dumps(
                        {"run_id": run_id, "endpoint": "d", "children_closed": True}
                    )
                )
                (tmp_path / "stop").touch()
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == (
            0 if termination in ("controller", "owner_exit_ack") and not required else 2
        ), (stdout, stderr)
        path = next((tmp_path / "output").glob("*/*/manifest.json"))
        manifest = json.loads(path.read_text())
        rows = [
            json.loads(line)
            for line in path.with_name("events.jsonl").read_text().splitlines()
        ]
        assert rows[0]["runtime"] == {"torch_imported": False, "vllm_imported": False}
        assert rows[0]["affinity"] == [allowed[-1]]
        assert manifest["complete"] == (
            termination in ("controller", "owner_exit_ack") and not required
        )
        assert manifest["observer_process_cpu_s"] > 0
        if termination == "owner_exit":
            assert manifest["end"]["reason"] == "owner_exit_timeout"
        if required:
            assert manifest["end"]["stats"]["required_source_failures"] > 0
        if termination == "controller":
            assert owner.poll() is None, "observer must not stop its server owner"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if owner.poll() is None:
            (endpoint_dir / "exit").touch()
            owner.wait(10)


def test_off_configuration_imports_no_observer_and_l3_validates_sources():
    script = "from vllm.kv_profiling.config import NetworkProfilerConfig; import sys; "
    script += (
        "NetworkProfilerConfig(); "
        "assert not any('endpoint_probes' in x for x in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    with pytest.raises(ValueError, match="required sources"):
        NetworkProfilerConfig(
            level="l3",
            run_id=str(uuid4()),
            output_dir="/tmp/out",
            probes={"required": ["tcp"]},
        )


@pytest.mark.parametrize("flow_count,unaligned", [(1, False), (2, False), (1, True)])
@pytest.mark.parametrize("rare_event", [False, True])
def test_synthetic_read_windows_have_only_qualified_flow_candidates(
    tmp_path,
    flow_count,
    unaligned,
    rare_event,
):
    """Real recorder and sockets, not evidence of a native NIXL handle/FD map."""
    from tests.distributed.kv_transfer.test_profiler_core import fields
    from vllm.kv_profiling.endpoint_probes.sockets import Discovery, Tracker

    run_id = str(uuid4())
    core_dir = tmp_path / "core"
    core = ProcessRecorder(
        NetworkProfilerConfig(
            level="l3",
            run_id=run_id,
            output_dir=str(core_dir),
            ring_records=128,
            context={
                "role": "worker",
                "engine_id": "d",
                "rank": 0,
                "kv_role": "kv_consumer",
            },
        )
    )
    assert core.wait_ready()
    writer = Writer(
        tmp_path / "probe",
        metadata(
            run_id, **({"time_namespace": "other-time-domain"} if unaligned else {})
        ),
        ProbeConfig(),
    )
    core.record(*fields(EventKind.TRANSFER_BIND, handle=7, request="request"))
    core.record(*fields(EventKind.POST_RETURN, handle=7))
    tracker = Tracker()
    for _ in range(flow_count):
        with socket.socket() as listener, socket.socket() as client:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            client.connect(listener.getsockname())
            server, _ = listener.accept()
            with server:
                # the real sampler's rows, as the observer records them
                client_inode = os.fstat(client.fileno()).st_ino
                discovery = Discovery([_me(engine_id="d", rank=0)])
                scan = discovery.scan()
                assert scan.complete, scan.reason
                scan.owned = {
                    cookie: record
                    for cookie, record in scan.owned.items()
                    if record["inode"] == client_inode
                }
                _sample_once(writer, scan, tracker)
                row = next(iter(scan.owned.values()))["row"]
                now = scan.dump_finished_ns
                if rare_event:
                    native = {
                        **tcp_event_row(),
                        "schema": 2,  # before owned-socket selection: tuples
                        **{
                            k: row[k]
                            for k in (
                                "family",
                                "local_ip",
                                "local_port",
                                "peer_ip",
                                "peer_port",
                            )
                        },
                        "cookie": row["cookie"][0] | (row["cookie"][1] << 32),
                        "timestamp_ns": now,
                        "time_conversion_qualified": True,
                    }
                    writer.put(
                        event("endpoint_metrics", "ebpf", "d", now, now, native, "tcp")
                    )
                    # Same timestamp but wrong tuple must not become this flow's event.
                    writer.put(
                        event(
                            "endpoint_metrics",
                            "ebpf",
                            "d",
                            now,
                            now,
                            {**native, "peer_ip": "192.0.2.99"},
                            "tcp",
                        )
                    )
    core.record(*fields(EventKind.COMPLETE_OBSERVED, handle=7))
    assert core.close().complete
    assert writer.close(True, "test", {})
    result = candidates(
        [core.output_dir / "events.jsonl"], [tmp_path / "probe/events.jsonl"]
    )
    assert len(result) == 1
    assert result[0]["association"] == (
        "unmatched"
        if unaligned
        else "unique_temporal_candidate"
        if flow_count == 1
        else "ambiguous"
    )
    assert result[0]["exact"] is False
    assert len(result[0]["candidates"]) == (0 if unaligned else flow_count)
    if rare_event and not unaligned:
        assert all(len(c["tcp_events"]) == 1 for c in result[0]["candidates"])
        assert all(
            c["tcp_events"][0]["exact"] is False for c in result[0]["candidates"]
        )
    if unaligned:
        assert "unaligned_clocks" in result[0]["reasons"]
        from vllm.kv_profiling.clock import ClockGroup, ClockMapping, ClockSample

        local = core.metadata["clock_id"]
        remote = writer.metadata["clock_id"]
        anchor = ClockSample(run_id, local, remote, 0, 0, 0, 0)
        projected = candidates(
            [core.output_dir / "events.jsonl"],
            [tmp_path / "probe/events.jsonl"],
            ClockGroup(local, {remote: ClockMapping(anchor)}),
        )
        assert projected[0]["association"] == "unique_temporal_candidate"
        assert projected[0]["exact"] is False


def test_an_owner_pid_generation_is_not_inferred_from_a_reused_pid():
    """An owner frozen with another start time (a reused PID) owns nothing."""
    from vllm.kv_profiling.endpoint_probes.sockets import Discovery

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        scan = Discovery([{**_me(), "start_ticks": _me()["start_ticks"] + 1}]).scan()
    assert scan.complete and scan.owned == {}
    assert "not the scheduler or worker it was frozen as" in scan.exited[os.getpid()]


def test_uapi_layout_matches_real_headers(tmp_path):
    """Compiler/header failure is a test failure, not a hardware skip."""
    source = tmp_path / "abi.c"
    source.write_text(r"""
#include <stddef.h>
#include <linux/inet_diag.h>
#include <linux/tcp.h>
#include <linux/ethtool.h>
#include <rdma/rdma_netlink.h>
_Static_assert(sizeof(struct inet_diag_req_v2) == 56, "request ABI");
_Static_assert(sizeof(struct inet_diag_msg) == 72, "response ABI");
_Static_assert(offsetof(struct tcp_info, tcpi_rtt) == 68, "rtt ABI");
_Static_assert(offsetof(struct tcp_info, tcpi_pacing_rate) == 104, "pacing ABI");
_Static_assert(offsetof(struct tcp_info, tcpi_bytes_sent) == 200, "bytes ABI");
_Static_assert(sizeof(struct ethtool_drvinfo) == 196, "driver ABI");
_Static_assert(offsetof(struct ethtool_drvinfo, n_stats) == 180, "stats ABI");
_Static_assert(RDMA_NL_NLDEV == 5, "RDMA client");
_Static_assert(RDMA_NLDEV_CMD_RES_QP_GET == 10, "RDMA command");
_Static_assert(RDMA_NLDEV_ATTR_RES_QP == 19, "QP table");
_Static_assert(RDMA_NLDEV_ATTR_RES_LQPN == 21, "local QPN");
_Static_assert(RDMA_NLDEV_ATTR_RES_RQPN == 22, "remote QPN");
_Static_assert(RDMA_NLDEV_ATTR_RES_TYPE == 26, "QP type");
_Static_assert(RDMA_NLDEV_ATTR_RES_STATE == 27, "QP state");
_Static_assert(RDMA_NLDEV_ATTR_RES_PID == 28, "QP pid");
int main(void) { return 0; }
""")
    executable = tmp_path / "abi"
    compiler = shlex.split(os.environ.get("CC", sysconfig.get_config_var("CC") or "cc"))
    subprocess.run(
        [*compiler, "-std=c11", str(source), "-o", str(executable)], check=True
    )
    subprocess.run([str(executable)], check=True)


def test_probe_only_kinds_cannot_enter_the_native_l1_binary_abi():
    from vllm.kv_profiling.schema import RECORD, VERSION, decode_record

    words = [0] * 16
    words[2] = (VERSION << 48) | (22 << 32)
    with pytest.raises(ValueError, match="probe JSONL schema"):
        decode_record(RECORD.pack(*words))


def test_real_loopback_driver_query_reports_unsupported_not_a_parser_failure():
    with pytest.raises(OSError) as result:
        netdev.driver("lo")
    assert result.value.errno in (errno.EOPNOTSUPP, errno.ENOTTY, errno.ENODEV)


@pytest.mark.parametrize("fault", ["missing_manifest", "overlap", "unverified"])
def test_netdev_only_cli_rejects_missing_inventory_and_bad_worker_placement(
    tmp_path, fault
):
    from vllm.kv_profiling.endpoint_probes.health import core_completion

    cpus = sorted(os.sched_getaffinity(0))
    process, entries, directory = start_recorded_endpoint(
        tmp_path, cpus[0], overlap_cpu=cpus[-1] if fault == "overlap" else None
    )
    run_id = entries[0]["run_id"]
    (directory / "close").touch()
    wait_for(lambda: (directory / "closed").exists(), timeout=20)
    worker = next(row for row in entries if row["context"]["role"] == "worker")
    if fault == "missing_manifest":
        (
            directory / "profiles" / run_id / worker["process_id"] / "manifest.json"
        ).unlink()
        assert not core_completion(directory / "profiles", run_id, entries)[
            "all_closed"
        ]
    if fault == "unverified":
        # Corrupt input, not a patched process/object: the worker recorder's
        # metadata names a stale generation, which discovery then freezes
        events = directory / "profiles" / run_id / worker["process_id"] / "events.jsonl"
        lines = events.read_text().splitlines(keepends=True)
        meta = json.loads(lines[0])
        meta["start_ticks"] += 1
        events.write_text(json.dumps(meta) + "\n" + "".join(lines[1:]))
    settings = ProbeConfig(interfaces=["lo"], required=["netdev"], interval_s=0.1)
    cfg = {
        "probes": asdict(settings),
        "expected": {"frontend": 1, "scheduler": 1, "worker": 1},
        "run_id": run_id,
        "endpoint": "d",
        "role": "decode",
        "cpus": [cpus[-1]],
        "input_dir": str(directory / "profiles"),
        "output_dir": str(tmp_path / "out"),
        "irq_policy": "virtual_not_applicable",
        "stop_file": str(tmp_path / "stop"),
        "endpoint_complete_file": str(tmp_path / "children"),
    }
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    child = subprocess.Popen(
        [
            sys.executable,
            str(Path(probe.__file__)),
            "--config",
            str(tmp_path / "cfg.json"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        wait_for(lambda: bool(list((tmp_path / "out").glob("*/*/manifest.json"))))
        (tmp_path / "children").write_text(
            json.dumps({"run_id": run_id, "endpoint": "d", "children_closed": True})
        )
        (tmp_path / "stop").touch()
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 2, (stdout, stderr)
        manifest = json.loads(
            next((tmp_path / "out").glob("*/*/manifest.json")).read_text()
        )
        assert manifest["complete"] is False
        if fault in ("overlap", "unverified"):
            assert manifest["end"]["stats"]["isolation_failures"] > 0
        assert process.poll() is None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        (directory / "exit").touch()
        process.wait(10)


def test_core_completion_counts_event_files_without_manifests(tmp_path):
    from vllm.kv_profiling.endpoint_probes.health import core_completion

    run = str(uuid4())
    front = tmp_path / run / "front"
    front.mkdir(parents=True)
    (front / "manifest.json").write_text(json.dumps({"run_id": run, "complete": True}))
    worker = tmp_path / run / "worker"
    worker.mkdir()
    (worker / "events.jsonl").write_text(json.dumps({"run_id": run}) + "\n")
    result = core_completion(tmp_path, run)
    assert result["all_closed"] is False
    assert result["problems"]


def test_required_rdma_family_cannot_be_satisfied_by_lifespan(tmp_path):
    root = tmp_path / "rdma"
    device = root / "fake0"
    (device / "ports/1/hw_counters").mkdir(parents=True)
    for name in ("node_guid", "sys_image_guid"):
        (device / name).write_text("test-guid")
    (device / "ports/1/hw_counters/lifespan").write_text("10")
    data = rdma.port({"device": "fake0", "port": 1}, root)
    assert data["coverage"] == {"counters": False, "hw_counters": False}
    from vllm.kv_profiling.endpoint_probes.runner import Observer

    original = os.sched_getaffinity(0)
    cpus = sorted(original)
    owner, entries, directory = start_recorded_endpoint(tmp_path, cpus[0])
    (directory / "close").touch()
    wait_for(lambda: (directory / "closed").exists(), timeout=20)
    run = entries[0]["run_id"]
    observer = Observer(
        ProbeConfig(rdma_ports=[{"device": "fake0", "port": 1}], required=["rdma"]),
        run_id=run,
        endpoint="d",
        role="decode",
        owner_pid=owner.pid,
        owner_start=process_identity(owner.pid)["start_ticks"],
        inventory=entries,
        cpus=[cpus[-1]],
        input_dir=str(directory / "profiles"),
        output_dir=str(tmp_path / "out"),
        irq_policy="virtual_not_applicable",
    )
    try:
        observer.observe(
            "rdma",
            "fake0/1",
            lambda: rdma.port({"device": "fake0", "port": 1}, root),
            "rdma",
        )
        (tmp_path / "stop").touch()
        (tmp_path / "children").write_text(
            json.dumps({"run_id": run, "endpoint": "d", "children_closed": True})
        )
        assert not observer.run(
            stop_file=tmp_path / "stop", endpoint_complete_file=tmp_path / "children"
        )
        manifest = json.loads(
            next((tmp_path / "out").glob("*/*/manifest.json")).read_text()
        )
        assert manifest["end"]["stats"]["required_source_failures"] > 0
        events = next((tmp_path / "out").glob("*/*/events.jsonl")).read_text()
        assert "required RDMA counter families unavailable" in events
    finally:
        os.sched_setaffinity(0, original)
        (directory / "exit").touch()
        owner.wait(10)


@pytest.mark.parametrize(
    "side,damage",
    [
        ("probe", "missing_end"),
        ("probe", "gap"),
        ("core", "missing_end"),
        ("core", "loss"),
    ],
)
def test_candidate_uniqueness_is_qualified_when_either_stream_is_partial(
    tmp_path, side, damage
):
    from tests.distributed.kv_transfer.test_profiler_core import fields

    run = str(uuid4())
    core = ProcessRecorder(
        NetworkProfilerConfig(
            level="l3",
            run_id=run,
            output_dir=str(tmp_path / "core"),
            ring_records=64,
            context={
                "role": "worker",
                "engine_id": "d",
                "rank": 0,
                "kv_role": "kv_consumer",
            },
        )
    )
    assert core.wait_ready()
    core.record(*fields(EventKind.TRANSFER_BIND, handle=7))
    core.record(*fields(EventKind.POST_RETURN, handle=7))
    now = time.monotonic_ns()
    writer = Writer(tmp_path / "probe", metadata(run), ProbeConfig())
    owner = {**process_identity(os.getpid()), "engine_id": "d", "rank": 0}
    writer.put(
        event(
            "endpoint_metrics",
            "socket",
            "observed-flow",
            now,
            now,
            {"owners": [owner]},
            "tcp",
        )
    )
    core.record(*fields(EventKind.COMPLETE_OBSERVED, handle=7))
    assert core.close().complete
    assert writer.close(True, "test", {})
    paths = {
        "core": core.output_dir / "events.jsonl",
        "probe": tmp_path / "probe/events.jsonl",
    }
    result = candidates([paths["core"]], [paths["probe"]])
    assert result[0]["association"] == "unique_temporal_candidate"
    path = paths[side]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if damage == "missing_end":
        rows = [row for row in rows if row["type"] != "end"]
    else:
        for row in rows:
            if row.get("sequence"):
                row["sequence"] += 1
            if row["type"] == "end":
                row["complete"] = False
                row["stats"]["drop_probe_rows" if side == "probe" else "drop_full"] = 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = candidates([paths["core"]], [paths["probe"]])
    assert result[0]["association"] == "partial_coverage"
    assert result[0]["candidates"]
    assert not result[0]["coverage_complete"]
    assert any(reason.startswith(side + ": ") for reason in result[0]["reasons"])


# eBPF contract: unsupported/inactive cannot become a kernel coverage claim.
# Native helper services are emulated only in the C unit driver, never a verifier.
@pytest.mark.parametrize(
    "value",
    [
        {"enabled": 1},
        {"enabled": True, "sets": []},
        {"enabled": True, "sets": ["provider_wr", "provider_wr"]},
        {"enabled": True, "sets": ["unknown"]},
        {"enabled": False, "artifacts": "/unused"},
        # the selection is the owned sockets: address lists are no option
        {"enabled": True, "mode": "tcp_tracepoints", "local_ips": ["10.0.0.1"]},
        {"enabled": True, "mode": "tcp_tracepoints", "peer_ips": ["10.0.0.1"]},
        {"enabled": True, "mode": "tcp_tracepoints", "local_ports": [0]},
    ],
)
def test_ebpf_invalid_options_are_refused_before_start(value):
    with pytest.raises((ValueError, TypeError)):
        ProbeConfig(ebpf=value)


def test_ebpf_disabled_config_preserves_the_previous_identity_and_imports(tmp_path):
    package = Path(__file__).resolve().parents[3] / "vllm/kv_profiling"
    if not package.is_dir():
        package = Path(os.environ["KV_PROFILER_SOURCE_ROOT"]) / "vllm/kv_profiling"
    code = (
        "import sys,json;sys.path.insert(0,sys.argv[1]);"
        "from endpoint_probes.config import ProbeConfig,settings_dict;"
        "assert 'endpoint_probes.ebpf' not in sys.modules;"
        "a=settings_dict(ProbeConfig());b=settings_dict(ProbeConfig(ebpf={'enabled':False}));"
        "assert a==b and 'ebpf' not in a;"
        "assert 'endpoint_probes.ebpf' not in sys.modules;print('inactive')"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(package)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.strip() == "inactive"


def test_ebpf_unavailable_paths_create_no_child_or_output(tmp_path):
    from vllm.kv_profiling.endpoint_probes.ebpf import Collector, Settings

    for options, reason in [
        ({"enabled": False}, None),
        (
            {"enabled": True, "mode": "tcp_tracepoints"},
            "missing_explicit_effective_privilege",
        ),
    ]:
        root = tmp_path / str(len(options)) / str(reason)
        collector = Collector(Settings(**options), [], root)
        collector.start()
        assert collector.child is None and not root.exists()
        assert collector.status["available"] is False
        assert collector.status.get("reason") == reason
        assert collector.status["exact"] is False


@pytest.mark.parametrize(
    "raw,valid",
    [
        ("monotonic -1 900000000\nboottime 0 0\n", True),
        ("monotonic 0 1000000000\nboottime 0 0\n", False),
        ("monotonic 0 0\nmonotonic 0 0\n", False),
        ("monotonic 999999999999999999 0\nboottime 0 0\n", False),
    ],
)
def test_ebpf_clock_domains_are_explicit_and_integer_bounded(tmp_path, raw, valid):
    from vllm.kv_profiling.endpoint_probes.ebpf import convert_time, read_offsets

    path = tmp_path / "offsets"
    path.write_text(raw)
    if valid:
        offsets = read_offsets(path)
        assert offsets["monotonic"] == -100_000_000
        assert convert_time(200_000_000, offsets["monotonic"]) == 100_000_000
    else:
        with pytest.raises(ValueError):
            read_offsets(path)
    for value, delta in [(0, -1), (1 << 63, 0), (True, 0)]:
        with pytest.raises(ValueError):
            convert_time(value, delta)


def tcp_event_row():
    from vllm.kv_profiling.endpoint_probes.ebpf import FIELDS

    return dict(
        type="tcp_event",
        schema=3,
        event="inet_sock_set_state",
        timestamp_ns=10,
        netns=99,
        family=2,
        local_ip="127.0.0.1",
        peer_ip="127.0.0.2",
        local_port=1234,
        peer_port=4321,
        cookie=42,
        admission=7,
        old_state=1,
        new_state=7,
        snapshot={name: {"state": "available", "value": 0} for name in FIELDS},
        srtt_encoding="usec_shift_3",
        exact=False,
    )


def test_ebpf_wire_records_cannot_invent_attribution_or_byte_semantics():
    from vllm.kv_profiling.endpoint_probes.ebpf import decode_row

    row = tcp_event_row()
    assert decode_row(json.dumps(row).encode()) == row
    for update in [
        {"exact": True},
        {"transfer": 0},
        {"cookie": 0},
        {"cookie": None},
        {"admission": 0},
        {"schema": 2},
        {"timestamp_ns": True},
        {"old_state": None},
        {"family": 10},
        {"event": "tcp_probe"},
    ]:
        with pytest.raises(ValueError):
            decode_row(json.dumps({**row, **update}).encode())
    with pytest.raises(ValueError):
        decode_row(b"x" * 2048)
    for name in row["snapshot"]:
        for state in ("unavailable", "read_error"):
            updated = {
                **row,
                "snapshot": {**row["snapshot"], name: {"state": state, "value": None}},
            }
            assert (
                decode_row(json.dumps(updated).encode())["snapshot"][name]["value"]
                is None
            )


def test_ebpf_writer_acknowledges_only_after_stream_flush(tmp_path):
    writer = Writer(tmp_path / "ack", metadata(), ProbeConfig())
    ticket = writer.put(
        event("probe_status", "ebpf", "test", 1, 1, {"available": False})
    )
    assert type(ticket) is int
    wait_for(lambda: writer.saved_sequence == ticket)
    rows = [
        json.loads(row)
        for row in (tmp_path / "ack/events.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["sequence"] == ticket
    assert writer.close(True, "test", {})


def test_ebpf_native_code_contracts_unprivileged(tmp_path):
    root = Path(
        os.environ.get("KV_PROFILER_SOURCE_ROOT", Path(__file__).resolve().parents[3])
    )
    source = root / "tests/distributed/kv_transfer/endpoint_ebpf_native.c"
    compiler = shlex.split(os.environ.get("CC", "cc"))
    binary = tmp_path / "ebpf-native-contracts"
    build = subprocess.run(
        [
            *compiler,
            "-O2",
            "-std=gnu11",
            "-pthread",
            "-I" + str(root),
            "-I" + str(source.parent),
            "-I" + str(root / "csrc/kv_profiler/ebpf"),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    done = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert (
        "external BPF helpers emulated; parent death uses real OS; no BPF load/attach"
        in done.stdout
    )


def test_ebpf_artifacts_match_their_own_manifest_not_build_environment(tmp_path):
    from vllm.kv_profiling.endpoint_probes.ebpf import digest, validate_artifacts

    names = ("collector", "observe.bpf.o", "limits.json")
    for name in names:
        (tmp_path / name).write_text(name)
    manifest = {
        "schema": 4,
        "mode": "tcp_tracepoints",
        "files": {name: digest(tmp_path / name) for name in names},
        "image": "different-build",
        "kernel_release": "different-kernel",
        "kernel_btf_sha256": "different-BTF",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert validate_artifacts(tmp_path) == manifest
    (tmp_path / "collector").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_artifacts(tmp_path)
    # a previous generation's package (address allow-lists) is rebuilt
    (tmp_path / "manifest.json").write_text(json.dumps({**manifest, "schema": 3}))
    with pytest.raises(ValueError, match="rebuild it"):
        validate_artifacts(tmp_path)


def test_ebpf_real_child_channel_preserves_event_time_and_never_qualifies(tmp_path):
    from vllm.kv_profiling.endpoint_probes.ebpf import Channel

    row = tcp_event_row()
    code = (
        "import json,sys;"
        f"print({json.dumps(json.dumps(row))},flush=True);"
        "assert sys.stdin.readline()=='stop\\n';"
        "print(json.dumps({'type':'end','complete':True}),flush=True)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    channel = Channel(child, tmp_path)
    writer = Writer(tmp_path / "writer", metadata(), ProbeConfig())
    captured = []
    try:
        deadline = time.monotonic() + 5
        while not captured:
            assert time.monotonic() < deadline
            captured.extend(channel.poll(writer, 0.05))
        assert captured == [row]
        result = channel.stop(writer, captured.append)
        assert result["complete"] is False
        assert result["native_exit"] == 0 and result["error"] is None
        assert result["terminal"]["complete"] is True
        assert writer.close(True, "test", {})
    finally:
        if channel.running():
            child.kill()
            channel.stop(writer)


def test_ebpf_channel_preserves_stderr_without_a_newline_and_truncated_stdout(tmp_path):
    from vllm.kv_profiling.endpoint_probes.ebpf import Channel

    code = (
        "import sys;sys.stderr.write('stderr-only');sys.stderr.flush();"
        "sys.stdout.write('{bad');sys.stdout.flush()"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    channel = Channel(child, tmp_path)
    writer = Writer(tmp_path / "writer", metadata(), ProbeConfig())
    deadline = time.monotonic() + 5
    try:
        with pytest.raises(ValueError, match="truncated"):
            while time.monotonic() < deadline:
                channel.poll(writer, 0.05)
        assert (tmp_path / "native.stderr").read_bytes() == b"stderr-only"
    finally:
        channel.stop(writer)
        writer.close(False, "test", {})


def test_ebpf_native_pipe_backpressure_is_bounded(tmp_path):
    from vllm.kv_profiling.endpoint_probes.ebpf import Channel

    child = subprocess.Popen(
        [sys.executable, "-c", "import signal;signal.pause()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    channel = Channel(child, tmp_path)
    writer = Writer(tmp_path / "writer", metadata(), ProbeConfig())
    try:
        count = 0
        while channel.command(b"x" * 127 + b"\n"):
            count += 1
            assert count < 8192
        assert count > 0
    finally:
        child.kill()
        child.wait(timeout=5)
        channel.stop(writer)
        writer.close(False, "test", {})


@pytest.mark.parametrize("unrelated_when", ["before", "during"])
def test_ebpf_final_resources_belong_only_to_the_owned_child(tmp_path, unrelated_when):
    import resource

    from vllm.kv_profiling.endpoint_probes.ebpf import Channel

    # A fork/exec child may retain the heavy pytest parent's pre-exec RSS HWM.
    unrelated_bytes = max(
        64 << 20, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 3072
    )
    unrelated_code = (
        "import resource,time,json,sys; a=bytearray(int(sys.argv[1]));"
        "start=time.process_time();"
        "exec('while time.process_time()-start < 0.25: pass');"
        "u=resource.getrusage(resource.RUSAGE_SELF);"
        "print(json.dumps(dict(cpu=u.ru_utime+u.ru_stime,rss=u.ru_maxrss)))"
    )

    def unrelated():
        return json.loads(
            subprocess.check_output(
                [sys.executable, "-c", unrelated_code, str(unrelated_bytes)]
            )
        )

    if unrelated_when == "before":
        other = unrelated()
    code = (
        "import sys,json,resource;assert sys.stdin.readline()=='stop\\n';"
        "u=resource.getrusage(resource.RUSAGE_SELF);"
        "print(json.dumps(dict(type='end',complete=False,"
        "own_cpu=u.ru_utime+u.ru_stime,own_rss=u.ru_maxrss)),flush=True)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    channel = Channel(child, tmp_path)
    writer = Writer(tmp_path / "writer", metadata(), ProbeConfig())
    try:
        if unrelated_when == "during":
            other = unrelated()
        result = channel.stop(writer)
        assert result["resource_complete"] and result["resource_error"] is None
        assert result["native_exit"] == 0 and result["error"] is None
        assert channel.rss_hwm_kib < other["rss"] // 2
        assert channel.cpu_s < other["cpu"] / 2
        assert channel.rss_hwm_kib >= result["terminal"]["own_rss"]
        assert channel.cpu_s >= result["terminal"]["own_cpu"]
        assert result["final_usage"] == {
            "cpu_s": channel.cpu_s,
            "rss_hwm_kib": channel.rss_hwm_kib,
        }
    finally:
        if channel.running():
            child.kill()
            channel.stop(writer)
        writer.close(False, "test", {})


def test_ebpf_poll_keeps_final_wait4_usage_and_forced_cleanup_is_accounted(tmp_path):
    from vllm.kv_profiling.endpoint_probes.ebpf import Channel

    for forced in (False, True):
        directory = tmp_path / str(forced)
        directory.mkdir()
        code = "import signal; signal.pause()" if forced else "pass"
        child = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        channel = Channel(child, directory)
        writer = Writer(directory / "writer", metadata(), ProbeConfig())
        try:
            if not forced:
                wait_for(lambda channel=channel: not channel.running())
                final = dict(channel.final_usage)
            result = channel.stop(writer)
            assert result["resource_complete"]
            assert result["resource_error"] is None
            assert result["final_usage"]["rss_hwm_kib"] > 0
            if forced:
                assert result["native_exit"] == -9
                assert result["error"] == "native shutdown deadline"
            else:
                assert result["final_usage"] == final
        finally:
            if channel.running():
                child.kill()
                channel.stop(writer)
            writer.close(False, "test", {})


def test_ebpf_final_accounting_is_explicitly_missing_if_another_waiter_reaps(tmp_path):
    from vllm.kv_profiling.endpoint_probes.ebpf import Channel

    child = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    channel = Channel(child, tmp_path)
    child.wait(timeout=5)
    writer = Writer(tmp_path / "writer", metadata(), ProbeConfig())
    result = channel.stop(writer)
    assert result["resource_complete"] is False
    assert result["resource_error"] == "owned child reaped outside accounting"
    assert result["outcome"] == "FAIL"
    assert result["final_usage"] is None
    writer.close(False, "test", {})


@pytest.mark.parametrize("outcome", ["FAIL", "UNAVAILABLE"])
def test_ebpf_explicit_failure_invalidates_even_an_optional_source(tmp_path, outcome):
    from vllm.kv_profiling.endpoint_probes.runner import Observer

    original = os.sched_getaffinity(0)
    cpus = sorted(original)
    owner, entries, directory = start_recorded_endpoint(tmp_path, cpus[0])
    (directory / "close").touch()
    wait_for(lambda: (directory / "closed").exists(), timeout=20)
    run = entries[0]["run_id"]
    observer = Observer(
        ProbeConfig(),
        run_id=run,
        endpoint="d",
        role="decode",
        owner_pid=owner.pid,
        owner_start=process_identity(owner.pid)["start_ticks"],
        inventory=entries,
        cpus=[cpus[-1]],
        input_dir=str(directory / "profiles"),
        output_dir=str(tmp_path / "out"),
        irq_policy="virtual_not_applicable",
    )
    resource = {
        "type": "resource",
        "outcome": outcome,
        "output_drops": 101,
        "queue_capacity": 100,
    }
    try:
        observer.ebpf_row(resource)
        (tmp_path / "stop").touch()
        (tmp_path / "children").write_text(
            json.dumps({"run_id": run, "endpoint": "d", "children_closed": True})
        )
        complete = observer.run(
            stop_file=tmp_path / "stop", endpoint_complete_file=tmp_path / "children"
        )
        manifest = json.loads(
            next((tmp_path / "out").glob("*/*/manifest.json")).read_text()
        )
        assert complete is (outcome != "FAIL")
        assert manifest["end"]["stats"]["ebpf_failures"] == int(outcome == "FAIL")
        assert manifest["complete"] is complete
        text = next((tmp_path / "out").glob("*/*/events.jsonl")).read_text()
        assert '"output_drops":101' in text.replace(" ", "")
    finally:
        os.sched_setaffinity(0, original)
        (directory / "exit").touch()
        owner.wait(10)


@pytest.mark.parametrize("exited", [True, False])
def test_ebpf_stop_drains_unread_fail_rows_and_stderr_after_exit_or_epipe(
    tmp_path, exited
):
    from vllm.kv_profiling.endpoint_probes.ebpf import Channel, Settings

    rows = [
        {
            "type": "resource",
            "outcome": "FAIL",
            "output_drops": 101,
            "queue_capacity": 100,
        },
        {
            "type": "status",
            "available": False,
            "outcome": "FAIL",
            "reason": "event_output_failure",
            "exact": False,
        },
        {"type": "end", "complete": False},
    ]
    ready = tmp_path / "ready"
    code = (
        "import os,sys,json,signal;from pathlib import Path;os.close(0);"
        f"rows=json.loads({json.dumps(json.dumps(rows))});"
        "exec('for row in rows: print(json.dumps(row),flush=True)');"
        "sys.stderr.write('event failure diagnostics');sys.stderr.flush();"
        f"Path({str(ready)!r}).touch();"
        + ("sys.exit(3)" if exited else "signal.pause()")
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    channel = Channel(child, tmp_path, Settings(enabled=True, mode="tcp_tracepoints"))
    writer = Writer(tmp_path / "writer", metadata(), ProbeConfig())
    captured = []
    try:
        wait_for(ready.exists)
        if exited:
            wait_for(lambda: not channel.running())
        result = channel.stop(writer, captured.append)
        assert captured == rows
        assert (tmp_path / "native.stderr").read_bytes() == b"event failure diagnostics"
        assert result["outcome"] == "FAIL" and result["state"] == "failed"
        assert result["resource"]["output_drops"] == 101
        assert result["resource"]["queue_capacity"] == 100
        assert result["terminal"] == rows[-1]
        assert result["resource_complete"]
        if exited:
            assert result["native_exit"] == 3 and result["error"] is None
        else:
            assert result["native_exit"] == -9
            assert result["stop_delivery_error"] is not None
        assert channel.stop(writer, captured.append) is result
        assert captured == rows
    finally:
        if channel.running():
            child.kill()
            channel.stop(writer)
        writer.close(False, "test", {})


@pytest.mark.parametrize(
    "payload", ["rows", "missing", "truncated", "lost-status", "missing-unavailable"]
)
def test_ebpf_optional_observer_cannot_pass_with_unread_or_lost_failure_at_shutdown(
    tmp_path, payload
):
    from vllm.kv_profiling.endpoint_probes.ebpf import Channel, Settings
    from vllm.kv_profiling.endpoint_probes.runner import Observer

    original = os.sched_getaffinity(0)
    cpus = sorted(original)
    owner, entries, directory = start_recorded_endpoint(tmp_path, cpus[0])
    (directory / "close").touch()
    wait_for(lambda: (directory / "closed").exists(), timeout=20)
    run = entries[0]["run_id"]
    observer = Observer(
        ProbeConfig(ebpf={"enabled": True, "mode": "tcp_tracepoints"}),
        run_id=run,
        endpoint="d",
        role="decode",
        owner_pid=owner.pid,
        owner_start=process_identity(owner.pid)["start_ticks"],
        inventory=entries,
        cpus=[cpus[-1]],
        input_dir=str(directory / "profiles"),
        output_dir=str(tmp_path / "out"),
        irq_policy="virtual_not_applicable",
    )
    gate = tmp_path / "shutdown-gate"
    data = (
        (
            json.dumps(
                {
                    "type": "resource",
                    "outcome": "FAIL",
                    "output_drops": 101,
                    "queue_capacity": 100,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "status",
                    "outcome": "FAIL",
                    "available": False,
                    "reason": "event_output_failure",
                    "exact": False,
                }
            )
            + "\n"
        )
        if payload == "rows"
        else ("{bad" if payload == "truncated" else "")
    )
    exit_code = 2 if payload == "missing-unavailable" else 3
    code = (
        "import sys,time;from pathlib import Path;"
        f"gate=Path({str(gate)!r});"
        "exec('while not gate.exists(): time.sleep(0.005)');"
        f"sys.stdout.write({data!r});sys.stdout.flush();"
        "sys.stderr.write('failure at shutdown');sys.stderr.flush();"
        f"sys.exit({exit_code})"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    class ExitAtShutdown(Channel):
        def stop(self, writer, consume=None):
            gate.touch()
            if payload == "lost-status":
                wait_for(lambda: os.waitpid(child.pid, os.WNOHANG)[0] == child.pid)
            else:
                wait_for(lambda: not self.running())
            return super().stop(writer, consume)

    channel_dir = tmp_path / "channel"
    channel_dir.mkdir()
    observer.ebpf = ExitAtShutdown(
        child, channel_dir, Settings(enabled=True, mode="tcp_tracepoints")
    )
    try:
        (tmp_path / "stop").touch()
        (tmp_path / "children").write_text(
            json.dumps({"run_id": run, "endpoint": "d", "children_closed": True})
        )
        assert not observer.run(
            stop_file=tmp_path / "stop", endpoint_complete_file=tmp_path / "children"
        )
        manifest = json.loads(
            next((tmp_path / "out").glob("*/*/manifest.json")).read_text()
        )
        assert manifest["complete"] is False
        assert manifest["end"]["stats"]["required_source_failures"] == 0
        assert manifest["end"]["stats"]["ebpf_failures"] > 0
        assert (channel_dir / "native.stderr").read_text() == "failure at shutdown"
        text = next((tmp_path / "out").glob("*/*/events.jsonl")).read_text()
        expected_exit = "null" if payload == "lost-status" else str(exit_code)
        assert f'"native_exit":{expected_exit}' in text.replace(" ", "")
        if payload == "rows":
            assert '"output_drops":101' in text.replace(" ", "")
        assert observer.ebpf.stopped_result["resource_complete"] is (
            payload != "lost-status"
        )
    finally:
        if observer.ebpf.running():
            child.kill()
            observer.ebpf.stop(observer.writer)
        os.sched_setaffinity(0, original)
        (directory / "exit").touch()
        owner.wait(10)


def test_ebpf_missing_time_namespace_metadata_keeps_raw_events(tmp_path, monkeypatch):
    """Optional proc metadata cannot prevent capture on a BTF kernel."""
    from vllm.kv_profiling.endpoint_probes.ebpf import Channel

    capture_metadata = metadata()
    real_readlink, real_read_text = os.readlink, Path.read_text

    def readlink(path, *args, **kwargs):
        if str(path) == "/proc/self/ns/time":
            raise FileNotFoundError("time namespaces not configured")
        return real_readlink(path, *args, **kwargs)

    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/self/timens_offsets":
            raise FileNotFoundError("time namespaces not configured")
        return real_read_text(path, *args, **kwargs)

    # Only the external proc service is emulated; the channel/child are real.
    monkeypatch.setattr(os, "readlink", readlink)
    monkeypatch.setattr(Path, "read_text", read_text)
    row = tcp_event_row()
    code = (
        f"print({json.dumps(json.dumps(row))},flush=True);"
        'input();print(\'{"type":"end","complete":false}\',flush=True)'
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    channel = Channel(child, tmp_path)
    writer = Writer(tmp_path / "writer", capture_metadata, ProbeConfig())
    captured = []
    try:
        wait_for(lambda: captured.extend(channel.poll(writer, 0.01)) or bool(captured))
        assert captured == [row] and captured[0]["timestamp_ns"] == 10
        assert channel.monotonic_offset is None
        result = channel.stop(writer)
        assert result["native_exit"] == 0 and result["complete"] is False
    finally:
        if channel.running():
            child.kill()
            channel.stop(writer)
        writer.close(False, "test", {})


def test_a_kernel_without_time_namespaces_is_one_clock_domain(tmp_path, monkeypatch):
    """Without /proc/self/ns/time every process shares the host's monotonic clock:
    the observer records None and inventories with None validate; an empty or
    non-string value is still refused."""
    from vllm.kv_profiling.endpoint_probes import runner
    from vllm.kv_profiling.endpoint_probes.health import validate_inventory

    owner, entries, directory = start_recorded_endpoint(
        tmp_path, sorted(os.sched_getaffinity(0))[0]
    )
    try:
        run = entries[0]["run_id"]
        validate_inventory([{**e, "time_namespace": None} for e in entries], run)
        for bad in ("", 7):
            with pytest.raises(ValueError, match="time_namespace"):
                validate_inventory([{**e, "time_namespace": bad} for e in entries], run)
    finally:
        (directory / "exit").touch()
        owner.wait(timeout=30)
    real_readlink = os.readlink

    def readlink(path, *args, **kwargs):
        if str(path) == "/proc/self/ns/time":
            raise FileNotFoundError("time namespaces not configured")
        return real_readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", readlink)
    assert runner._time_namespace() is None


def _fake_stat(pid, comm, start_ticks, cpu=3, utime=7, stime=2):
    # Fields 3.. after the command, in the kernel's order (proc(5)).
    after = ["S"] + ["0"] * 49
    after[11], after[12], after[19], after[36] = (
        str(utime), str(stime), str(start_ticks), str(cpu))  # fmt: skip
    return f"{pid} ({comm}) " + " ".join(after) + "\n"


def test_thread_probe_reports_each_thread_and_unavailable_counters_as_none(tmp_path):
    """Per-thread CPU and scheduling come from the kernel's own counters; a
    kernel without schedstats yields None (unavailable), never zero, and a
    command with spaces and parentheses does not shift the fields."""
    from vllm.kv_profiling.endpoint_probes import tasks

    base = tmp_path / "proc/40"
    (base / "task/40").mkdir(parents=True)
    (base / "task/41").mkdir()
    (base / "task/42").mkdir()  # exits before its stat is read
    (base / "stat").write_text(_fake_stat(40, "vllm", 900))
    (base / "task/40/stat").write_text(_fake_stat(40, "vllm", 900, cpu=1))
    (base / "task/40/schedstat").write_text("5000 1200 9\n")
    (base / "task/40/status").write_text(
        "Name:\tvllm\nvoluntary_ctxt_switches:\t11\nnonvoluntary_ctxt_switches:\t4\n"
    )
    (base / "task/41/stat").write_text(_fake_stat(41, "a) (b c", 905, utime=30))
    data = tasks.process_threads(40, 900, proc=tmp_path / "proc")
    assert data["exited"] is False and data["threads_total"] == 3
    assert data["vanished_between_reads"] == 1
    main, other = data["threads"]
    assert (main["tid"], main["processor"], main["run_ns"], main["wait_ns"]) == (
        40, 1, 5000, 1200)  # fmt: skip
    assert (main["voluntary_ctxt_switches"], main["nonvoluntary_ctxt_switches"]) == (
        11, 4)  # fmt: skip
    assert (other["comm"], other["utime_ticks"], other["start_ticks"]) == (
        "a) (b c", 30, 905)  # fmt: skip
    assert other["run_ns"] is None and other["voluntary_ctxt_switches"] is None
    assert data["coverage_problem"] is None  # an absent schedstat is not an error
    # A reused PID (another generation) is the serving process having exited.
    assert tasks.process_threads(40, 899, proc=tmp_path / "proc")["exited"] is True
    assert tasks.process_threads(77, 1, proc=tmp_path / "proc")["exited"] is True


def test_interrupt_probe_reads_only_the_endpoint_nics_and_bounds_rows(tmp_path):
    """Only the configured NICs' MSI interrupts are reported, busiest CPU first;
    a NIC without MSI interrupts is named unavailable; network softirqs on any
    CPU count stay within the probe row limit."""
    from vllm.kv_profiling.endpoint_probes import tasks
    from vllm.kv_profiling.endpoint_probes.runner import Observer

    proc, sys_root = tmp_path / "proc", tmp_path / "sys"
    proc.mkdir()
    (sys_root / "class/net/nic0/device/msi_irqs/30").mkdir(parents=True)
    (sys_root / "class/infiniband/dev0/device/msi_irqs/31").mkdir(parents=True)
    (sys_root / "class/net/veth0").mkdir(parents=True)
    (proc / "interrupts").write_text(
        "           CPU0       CPU1       CPU2\n"
        "  29:          5          0          0   PCI-MSI other-device\n"
        "  30:          1          0         90   PCI-MSI nic0-rx-0\n"
        "  31:          0          4          0   PCI-MSI dev0-comp\n"
        " NMI:          0          0          0   Non-maskable interrupts\n"
    )
    data = tasks.nic_interrupts(["nic0", "veth0"], ["dev0"], proc=proc, sys=sys_root)
    assert [(i["irq"], i["owners"], i["per_cpu"]) for i in data["irqs"]] == [
        (30, ["interface:nic0"], {"2": 90, "0": 1}),
        (31, ["rdma_device:dev0"], {"1": 4}),
    ]
    assert [u.get("interface") for u in data["unavailable"]] == ["veth0"]
    cpus = 1024
    (proc / "softirqs").write_text(
        "".join(f" CPU{c}" for c in range(cpus)) + "\n"
        + "".join(
            f"{name}:" + f" {10**19}" * cpus + "\n"
            for name in ("HI", "NET_TX", "NET_RX")
        )
    )  # fmt: skip
    softirqs = tasks.net_softirqs(proc=proc)
    assert softirqs["cpu_count"] == cpus
    assert {i["name"] for i in softirqs["softirqs"]} == {"NET_TX", "NET_RX"}
    assert sum(len(i["counts"]) for i in softirqs["softirqs"]) == 2 * cpus

    rows = []

    class Sink:
        def put(self, row):
            rows.append(row)

    observer = Observer.__new__(Observer)
    observer.writer = Sink()
    observer.emit("interrupts", "softirqs", 0, softirqs)
    threads = [{"tid": t, "comm": "x" * 15, **dict.fromkeys(
        ("start_ticks", "processor", "utime_ticks", "stime_ticks", "run_ns",
         "wait_ns", "timeslices", "voluntary_ctxt_switches",
         "nonvoluntary_ctxt_switches"), 10**19)} for t in range(1024)]  # fmt: skip
    observer.emit("threads", "worker/1", 0, {"threads": threads, "units": {}})
    assert max(len(json.dumps(row)) for row in rows) < (16 << 10) - 512
    parts = [r["observation"] for r in rows]
    assert sum(len(p["threads"]) for p in parts if "threads" in p) == 1024
    assert sum(len(p["softirqs"]) for p in parts if "softirqs" in p) == 64


@pytest.mark.parametrize("absent_nic", [False, True])
def test_real_cli_records_the_serving_threads_and_host_network_interrupts(
    tmp_path, absent_nic
):
    """Through the observer CLI: every serving process of the inventory is
    sampled thread by thread from outside it, and the interrupt sources are
    observed (a loopback NIC has no MSI interrupts: unavailable, not a failure;
    a configured NIC that does not exist fails the required source)."""
    allowed = sorted(os.sched_getaffinity(0))
    assert len(allowed) >= 2, "probe tests require separate owner/observer CPUs"
    owner, inventory, endpoint_dir = start_recorded_endpoint(tmp_path, allowed[0])
    run_id = inventory[0]["run_id"]
    settings = ProbeConfig(
        interval_s=0.1,
        interfaces=["lo", "absent-nic0"] if absent_nic else ["lo"],
        threads=True,
        interrupts=True,
        required=["threads", "interrupts"],
    )
    config = {
        "probes": asdict(settings),
        "expected": {"frontend": 1, "scheduler": 1, "worker": 1},
        "run_id": run_id,
        "endpoint": "d",
        "role": "decode",
        "cpus": [allowed[-1]],
        "input_dir": str(endpoint_dir / "profiles"),
        "output_dir": str(tmp_path / "output"),
        "irq_policy": "virtual_not_applicable",
        "stop_file": str(tmp_path / "stop"),
        "endpoint_complete_file": str(tmp_path / "children"),
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    process = subprocess.Popen(
        [sys.executable, str(Path(probe.__file__)), "--config",
         str(tmp_path / "config.json")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )  # fmt: skip
    try:

        def sampled():
            paths = list((tmp_path / "output").glob("*/*/events.jsonl"))
            return bool(paths) and paths[0].read_text().count('"softirqs"') >= 2

        wait_for(sampled, timeout=20)
        (endpoint_dir / "close").touch()
        wait_for(lambda: (endpoint_dir / "closed").exists(), timeout=20)
        (tmp_path / "children").write_text(
            json.dumps({"run_id": run_id, "endpoint": "d", "children_closed": True})
        )
        (tmp_path / "stop").touch()
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == (2 if absent_nic else 0), (stdout, stderr)
        path = next((tmp_path / "output").glob("*/*/manifest.json"))
        manifest = json.loads(path.read_text())
        assert manifest["complete"] is not absent_nic
        rows = [
            json.loads(line)
            for line in path.with_name("events.jsonl").read_text().splitlines()
        ]
        threads = [r for r in rows if r.get("scope") == "threads"]
        pids = {meta["pid"] for meta in inventory} | {owner.pid}
        assert {r["observation"]["pid"] for r in threads} == pids
        live = [r["observation"] for r in threads if not r["observation"]["exited"]]
        assert live and all(d["threads"] for d in live)
        tids = {t["tid"] for d in live for t in d["threads"]}
        assert process.pid not in tids, "the observer samples only serving threads"
        nic = [r["observation"] for r in rows if r.get("entity") == "nic"]
        assert nic and nic[0]["irqs"] == []
        assert nic[0]["unavailable"][0]["interface"] == "lo"
        failures = [r for r in rows if r.get("scope") == "source"
                    and r.get("entity") in ("threads", "interrupts")]  # fmt: skip
        if absent_nic:
            assert nic[0]["absent"] == [{"interface": "absent-nic0"}]
            assert manifest["end"]["stats"]["required_source_failures"] > 0
            assert {r["observation"]["error"] for r in failures} == {
                "1 configured devices absent"
            }
            return
        softirqs = next(r["observation"] for r in rows if r.get("entity") == "softirqs")
        assert {i["name"] for i in softirqs["softirqs"]} == {"NET_RX", "NET_TX"}
        assert not failures
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if owner.poll() is None:
            (endpoint_dir / "exit").touch()
            owner.wait(10)


def _fake_thread_tree(root, pid, start, tids):
    base = root / str(pid)
    (base / "task").mkdir(parents=True)
    (base / "stat").write_text(_fake_stat(pid, "vllm", start))
    for tid, (thread_start, run_ns) in tids.items():
        task = base / "task" / str(tid)
        task.mkdir()
        (task / "stat").write_text(_fake_stat(tid, "vllm", thread_start))
        (task / "schedstat").write_text(f"{run_ns} 1 1\n")
        (task / "status").write_text("voluntary_ctxt_switches:\t1\n")
    return base


def _hook_reads(monkeypatch, after):
    """Run ``after(path, name)`` once a pinned read of ``name`` completes."""
    from vllm.kv_profiling.endpoint_probes import tasks

    real = tasks._read_at

    def read(directory, name, limit=4096):
        value = real(directory, name, limit)
        after(Path(os.readlink(f"/proc/self/fd/{directory}")), name)
        return value

    monkeypatch.setattr(tasks, "_read_at", read)


def test_a_process_replaced_after_its_identity_read_is_not_reported(
    tmp_path, monkeypatch
):
    """The process is pinned by its /proc directory: a replacement under the
    same PID right after the generation check is never read as the serving
    process."""
    from vllm.kv_profiling.endpoint_probes import tasks

    proc = tmp_path / "proc"
    _fake_thread_tree(proc, 40, 900, {40: (900, 5000)})
    swapped = []

    def replace(path, name):
        if path == proc / "40" and name == "stat" and not swapped:
            swapped.append((proc / "40").rename(tmp_path / "old-40"))
            _fake_thread_tree(proc, 40, 950, {40: (950, 1), 41: (951, 1)})

    _hook_reads(monkeypatch, replace)
    data = tasks.process_threads(40, 900, proc=proc)
    assert swapped
    assert [(t["tid"], t["start_ticks"], t["run_ns"]) for t in data["threads"]] == [
        (40, 900, 5000)
    ]


def test_a_reused_thread_id_never_mixes_two_threads_counters(tmp_path, monkeypatch):
    """Each thread is pinned by its own directory: a TID reused between its stat
    and schedstat reads cannot pair one thread's start with another's counters."""
    from vllm.kv_profiling.endpoint_probes import tasks

    proc = tmp_path / "proc"
    base = _fake_thread_tree(proc, 40, 900, {40: (900, 1), 41: (905, 5000)})
    swapped = []

    def reuse(path, name):
        if path.name == "41" and name == "stat" and not swapped:
            swapped.append((base / "task/41").rename(tmp_path / "old-41"))
            (base / "task/41").mkdir()
            (base / "task/41/stat").write_text(_fake_stat(41, "new", 990))
            (base / "task/41/schedstat").write_text("99999 9 9\n")

    _hook_reads(monkeypatch, reuse)
    thread = tasks.process_threads(40, 900, proc=proc)["threads"][1]
    assert swapped
    assert (thread["start_ticks"], thread["run_ns"]) == (905, 5000)


def test_long_or_unreadable_task_files_keep_the_other_measurements(
    tmp_path, monkeypatch
):
    """A long status (many supplementary groups) is read; an optional file that
    cannot be read leaves the thread's other counters; a thread that cannot be
    read for permissions is counted as unreadable, not as exited."""
    from vllm.kv_profiling.endpoint_probes import tasks

    proc = tmp_path / "proc"
    base = _fake_thread_tree(
        proc, 40, 900, {40: (900, 10), 41: (901, 20), 42: (902, 30)}
    )
    groups = " ".join(str(1000000 + g) for g in range(600))
    (base / "task/40/status").write_text(
        f"Groups:\t{groups}\nvoluntary_ctxt_switches:\t7\n"
        "nonvoluntary_ctxt_switches:\t3\n"
    )
    assert (base / "task/40/status").stat().st_size > 4096
    (base / "task/41/status").write_text("x" * (128 << 10))
    real = tasks._read_at

    def read(directory, name, limit=4096):
        path = Path(os.readlink(f"/proc/self/fd/{directory}"))
        if path.name == "42" and name == "stat":
            raise PermissionError(13, "Permission denied")
        return real(directory, name, limit)

    monkeypatch.setattr(tasks, "_read_at", read)
    data = tasks.process_threads(40, 900, proc=proc)
    long_status, oversize = data["threads"]
    assert long_status["voluntary_ctxt_switches"] == 7
    assert long_status["nonvoluntary_ctxt_switches"] == 3
    assert oversize["run_ns"] == 20 and oversize["voluntary_ctxt_switches"] is None
    assert oversize["unavailable"] == [{"file": "status", "error": "ObservationError"}]
    assert data["vanished_between_reads"] == 0
    assert data["unreadable"] == {"PermissionError": 1}
    assert data["unreadable_files"] == {"status:ObservationError": 1}
    assert data["coverage_problem"] == "1 threads unreadable; 1 thread files unreadable"


def test_interrupt_counts_keep_the_cpu_ids_their_header_names(tmp_path):
    """With offline CPUs the columns are not CPU IDs: counts are attributed to
    the CPU each header label names, and a row that does not match it fails."""
    from vllm.kv_profiling.endpoint_probes import tasks

    proc, sys_root = tmp_path / "proc", tmp_path / "sys"
    proc.mkdir()
    (sys_root / "class/net/nic0/device/msi_irqs/30").mkdir(parents=True)
    (proc / "interrupts").write_text(
        "       CPU0  CPU3  CPU7\n  30:     0     2    10   PCI-MSI nic0\n"
    )
    (proc / "softirqs").write_text(
        "       CPU0  CPU3  CPU7\nNET_TX:   1  2  3\nNET_RX:   4  5  6\n"
    )
    irq = tasks.nic_interrupts(["nic0"], proc=proc, sys=sys_root)
    assert irq["cpu_count"] == 3 and irq["irqs"][0]["per_cpu"] == {"7": 10, "3": 2}
    softirqs = tasks.net_softirqs(proc=proc)["softirqs"]
    assert {i["name"]: (i["cpus"], i["counts"]) for i in softirqs} == {
        "NET_TX": ([0, 3, 7], [1, 2, 3]),
        "NET_RX": ([0, 3, 7], [4, 5, 6]),
    }
    (proc / "interrupts").write_text("       CPU0  CPU3  CPU7\n  30:   0   2\n")
    with pytest.raises(ObservationError, match="does not match the CPU header"):
        tasks.nic_interrupts(["nic0"], proc=proc, sys=sys_root)
    (proc / "softirqs").write_text("       CPU0  CPU3  CPU7\nNET_RX:   4  5\n")
    with pytest.raises(ObservationError, match="does not match the CPU header"):
        tasks.net_softirqs(proc=proc)


def test_missing_interrupt_coverage_fails_a_required_source(tmp_path):
    """An interrupt sysfs lists but /proc/interrupts lacks, an unreadable
    /proc/interrupts, or a missing network softirq is a required-source
    failure; a device without (or with an empty) MSI list is only unavailable."""
    from vllm.kv_profiling.endpoint_probes import tasks
    from vllm.kv_profiling.endpoint_probes.runner import Observer

    proc, sys_root = tmp_path / "proc", tmp_path / "sys"
    proc.mkdir()
    (sys_root / "class/net/nic0/device/msi_irqs/30").mkdir(parents=True)
    (sys_root / "class/net/nic1/device/msi_irqs").mkdir(parents=True)
    (sys_root / "class/net/lo").mkdir(parents=True)
    rows = []

    class Sink:
        def put(self, row):
            rows.append(row)

    def observer():
        o = Observer.__new__(Observer)
        o.writer, o.seen_identities, o.failed_required = Sink(), {}, 0
        o.config = ProbeConfig(interfaces=["nic0"], interrupts=True,
                               required=["interrupts"])  # fmt: skip
        return o

    def nic(names):
        return lambda: tasks.nic_interrupts(names, proc=proc, sys=sys_root)

    header = "       CPU0  CPU1\n"
    (proc / "interrupts").write_text(header + "  30:   1   2   PCI-MSI nic0\n")
    (proc / "softirqs").write_text(header + "NET_TX:  1  2\nNET_RX:  3  4\n")
    ok = observer()
    ok.observe("interrupts", "nic", nic(["nic0", "nic1", "lo"]))
    ok.observe("interrupts", "softirqs", lambda: tasks.net_softirqs(proc=proc))
    assert ok.failed_required == 0
    reasons = rows[0]["observation"]["unavailable"]
    assert reasons == [
        {"interface": "nic1", "reason": "empty MSI interrupt list"},
        {"interface": "lo", "reason": "no MSI interrupt list"},
    ]
    rows.clear()
    absent = observer()
    absent.observe("interrupts", "nic", nic(["nic0", "lo", "no-such-nic"]))
    assert absent.failed_required == 1  # a configured device that does not exist
    (observed,) = [r for r in rows if r["kind"] == "endpoint_metrics"]
    assert observed["observation"]["absent"] == [{"interface": "no-such-nic"}]
    for interrupts, softirqs in (
        (header, header + "NET_TX:  1  2\nNET_RX:  3  4\n"),  # IRQ 30 absent
        ("", header + "NET_TX:  1  2\nNET_RX:  3  4\n"),  # no CPU header
        (header + "  30:   1   2   PCI-MSI nic0\n", header + "NET_TX:  1  2\n"),
    ):
        (proc / "interrupts").write_text(interrupts)
        (proc / "softirqs").write_text(softirqs)
        rows.clear()
        failing = observer()
        failing.observe("interrupts", "nic", nic(["nic0"]))
        failing.observe("interrupts", "softirqs", lambda: tasks.net_softirqs(proc=proc))
        assert failing.failed_required == 1, rows
        assert [r["observation"]["available"] for r in rows
                if r["kind"] == "probe_status"] == [False]  # fmt: skip


def _probe_rows(data, source, entity, config):
    """One observation through the observer's emit and failure paths, checked
    against the real writer's validation and row limit."""
    from vllm.kv_profiling.endpoint_probes.runner import Observer

    rows, dropped = [], []

    class Sink:
        def put(self, row):
            validate(row)
            raw = json.dumps({**row, "sequence": 0}, allow_nan=False) + "\n"
            (dropped if len(raw.encode()) > 16 << 10 else rows).append(row)

    observer = Observer.__new__(Observer)
    observer.writer, observer.seen_identities, observer.failed_required = (
        Sink(), {}, 0)  # fmt: skip
    observer.config = config
    observer.observe(source, entity, lambda: data)
    return rows, dropped, observer.failed_required


def test_many_missing_interrupts_keep_the_visible_counters_and_a_bounded_reason(
    tmp_path,
):
    """256 vectors shared by a NIC and its RDMA device with one visible: the
    visible IRQ's counters and a short failure survive the writer, the missing
    vectors are an exact count with a bounded sample, and nothing is dropped."""
    from vllm.kv_profiling.endpoint_probes import tasks

    proc, sys_root = tmp_path / "proc", tmp_path / "sys"
    proc.mkdir()
    for path in ("class/net/nic0/device/msi_irqs",
                 "class/infiniband/dev0/device/msi_irqs"):  # fmt: skip
        for irq in range(1000, 1256):
            (sys_root / path / str(irq)).mkdir(parents=True)
    (proc / "interrupts").write_text(
        "       CPU0  CPU1\n 1000:   5   6   PCI-MSI nic0-comp0\n"
    )
    data = tasks.nic_interrupts(["nic0"], ["dev0"], proc=proc, sys=sys_root)
    assert data["missing_count"] == 255 and data["missing_truncated"]
    assert len(data["missing"]) == tasks.MAX_LISTED
    config = ProbeConfig(interfaces=["nic0"], interrupts=True)  # optional source
    rows, dropped, required_failures = _probe_rows(data, "interrupts", "nic", config)
    assert dropped == [] and required_failures == 0
    (irq,) = [i for r in rows if r["kind"] == "endpoint_metrics"
              for i in r["observation"]["irqs"]]  # fmt: skip
    assert (irq["irq"], irq["total"]) == (1000, 11)
    (status,) = [r for r in rows if r["kind"] == "probe_status"]
    assert status["observation"]["error"] == (
        f"255 listed interrupts absent from {proc / 'interrupts'}"
    )


def test_access_errors_fail_a_required_source_and_keep_what_was_read(
    tmp_path, monkeypatch
):
    """An MSI list or every live thread's stat that cannot be read is a source
    failure; the counters that were read are kept."""
    from vllm.kv_profiling.endpoint_probes import tasks

    proc, sys_root = tmp_path / "proc", tmp_path / "sys"
    proc.mkdir()
    (sys_root / "class/net/nic0/device/msi_irqs/30").mkdir(parents=True)
    (sys_root / "class/net/nic1/device/msi_irqs/31").mkdir(parents=True)
    (proc / "interrupts").write_text("       CPU0\n  30:   4   PCI-MSI nic0\n")
    real_list = tasks._msi_list

    def msi_list(path):
        if "nic1" in str(path):
            raise PermissionError(13, "Permission denied")
        return real_list(path)

    monkeypatch.setattr(tasks, "_msi_list", msi_list)
    data = tasks.nic_interrupts(["nic0", "nic1"], proc=proc, sys=sys_root)
    assert data["unreadable"] == [{"interface": "nic1", "error": "PermissionError"}]
    config = ProbeConfig(interfaces=["nic0"], interrupts=True, threads=True,
                         required=["interrupts", "threads"])  # fmt: skip
    rows, _, failures = _probe_rows(data, "interrupts", "nic", config)
    assert failures == 1
    assert [i["irq"] for r in rows if r["kind"] == "endpoint_metrics"
            for i in r["observation"]["irqs"]] == [30]  # fmt: skip

    _fake_thread_tree(proc, 40, 900, {40: (900, 1), 41: (901, 2)})
    real_read = tasks._read_at

    def read(directory, name, limit=4096):
        path = Path(os.readlink(f"/proc/self/fd/{directory}"))
        if name == "stat" and path.parent.name == "task":
            raise PermissionError(13, "Permission denied")
        return real_read(directory, name, limit)

    monkeypatch.setattr(tasks, "_read_at", read)
    threads = tasks.process_threads(40, 900, proc=proc)
    assert threads["exited"] is False and threads["threads"] == []
    assert threads["unreadable"] == {"PermissionError": 2}
    _, _, failures = _probe_rows(threads, "threads", "worker/40", config)
    assert failures == 1


def test_a_process_that_exits_while_its_tasks_are_listed_is_recorded_as_exited(
    monkeypatch,
):
    """A real child exits after its task directory is opened and before the
    listing: the empty listing is confirmed through the pinned process as an
    exit, not reported as a live process without threads."""
    from vllm.kv_profiling.endpoint_probes import tasks

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    start = process_identity(child.pid)["start_ticks"]
    real_list = tasks._list_tasks

    def list_after_exit(tasks_fd):
        child.kill()
        child.wait()
        return real_list(tasks_fd)

    monkeypatch.setattr(tasks, "_list_tasks", list_after_exit)
    try:
        data = tasks.process_threads(child.pid, start)
        assert child.returncode is not None
        assert data["exited"] is True and data["threads"] == []
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_a_shared_pci_function_splits_into_rows_the_writer_keeps(tmp_path):
    """Eight interfaces and an RDMA device sharing 256 vectors, eight visible on
    32 CPUs: rows over the writer's limit are split on their largest list, each
    part labelled with its range, so every visible counter and the diagnostic
    survive and nothing is dropped."""
    from vllm.kv_profiling.endpoint_probes import tasks

    proc, sys_root = tmp_path / "proc", tmp_path / "sys"
    proc.mkdir()
    names = [f"ens{n}f0np0-long-name" for n in range(8)]
    paths = [f"class/net/{n}/device/msi_irqs" for n in names]
    paths.append("class/infiniband/rdma-device-long-name-0/device/msi_irqs")
    for path in paths:
        for irq in range(1000, 1256):
            (sys_root / path / str(irq)).mkdir(parents=True)
    cpus = 32
    lines = ["".join(f"  CPU{c}" for c in range(cpus))]
    for irq in range(1000, 1008):
        lines.append(f" {irq}:" + f" {10**15 + irq}" * cpus + f"  PCI-MSI {irq}-comp")
    (proc / "interrupts").write_text("\n".join(lines) + "\n")
    data = tasks.nic_interrupts(
        names, ["rdma-device-long-name-0"], proc=proc, sys=sys_root
    )
    assert len(json.dumps(data)) > 16 << 10  # one row could not hold it
    config = ProbeConfig(interfaces=names, interrupts=True)  # optional source
    rows, dropped, _ = _probe_rows(data, "interrupts", "nic", config)
    assert dropped == []
    metrics = [r["observation"] for r in rows if r["kind"] == "endpoint_metrics"]
    assert sorted(i["irq"] for m in metrics for i in m["irqs"]) == list(
        range(1000, 1008)
    )
    assert all(len(i["per_cpu"]) == cpus for m in metrics for i in m["irqs"])
    # Every listed item is in exactly one row: no visible IRQ counted twice.
    missing = [e["irq"] for m in metrics for e in m["missing"]]
    assert len(metrics) > 1 and missing == sorted(set(missing)) and len(missing) == 32
    assert all(m["missing_count"] == 248 for m in metrics)
    (status,) = [r for r in rows if r["kind"] == "probe_status"]
    assert status["observation"]["error"].startswith("248 listed interrupts absent")


def test_single_entries_of_different_lists_are_split_into_separate_rows():
    """Two lists of one entry each, together over the row limit, are put in
    separate rows with their ranges; reassembling every split row by its
    slices gives back each list exactly once, with nothing dropped."""
    config = ProbeConfig(interfaces=["nic0"], interrupts=True)
    big = "x" * 8600
    data = {"irqs": [{"irq": 30, "name": big}],
            "unreadable": [{"interface": "nic1", "error": big}]}  # fmt: skip
    rows, dropped, _ = _probe_rows(data, "interrupts", "nic", config)
    assert dropped == []
    parts = [r["observation"] for r in rows]
    assert [(p["slices"]["irqs"], p["slices"]["unreadable"]) for p in parts] == [
        ([0, 1, 1], [0, 0, 1]),
        ([1, 1, 1], [0, 1, 1]),
    ]
    assert [i["irq"] for p in parts for i in p["irqs"]] == [30]
    assert [u["interface"] for p in parts for u in p["unreadable"]] == ["nic1"]

    mixed = {
        "irqs": [{"irq": n, "name": "y" * 900} for n in range(37)],
        "missing": [{"irq": n, "owners": ["z" * 700]} for n in range(23)],
        "unavailable": [{"interface": "lo", "reason": "w" * 5000}],
        "absent": [],
    }
    rows, dropped, _ = _probe_rows(mixed, "interrupts", "nic", config)
    assert dropped == [] and len(rows) > 2
    for key, values in mixed.items():
        rebuilt = [None] * len(values)
        for part in (r["observation"] for r in rows):
            start, end, total = part["slices"].get(key, (0, len(values), len(values)))
            assert total == len(values) and len(part[key]) == end - start
            for index, item in zip(range(start, end), part[key]):
                assert rebuilt[index] is None, "an item in two rows"
                rebuilt[index] = item
        assert rebuilt == values


def test_a_masked_proc_interrupts_reads_the_kernel_irq_sysfs_counts(tmp_path):
    """Container runtimes may mask /proc/interrupts (an empty file): the NICs'
    interrupts come from /sys/kernel/irq per CPU ID, an IRQ without its sysfs
    directory is missing, and with neither source the failure says why."""
    from vllm.kv_profiling.endpoint_probes import tasks

    proc, sys_root = tmp_path / "proc", tmp_path / "sys"
    proc.mkdir()
    (proc / "interrupts").write_text("")
    (sys_root / "class/net/nic0/device/msi_irqs/30").mkdir(parents=True)
    (sys_root / "class/infiniband/dev0/device/msi_irqs/31").mkdir(parents=True)
    with pytest.raises(ObservationError, match="is empty .masked"):
        tasks.nic_interrupts(["nic0"], proc=proc, sys=sys_root)
    irq = sys_root / "kernel/irq/30"
    irq.mkdir(parents=True)
    (sys_root / "devices/system/cpu").mkdir(parents=True)
    (sys_root / "devices/system/cpu/possible").write_text("0-2\n")
    (irq / "per_cpu_count").write_text("0,2,10\n")
    (irq / "actions").write_text("nic0-comp0\n")
    data = tasks.nic_interrupts(["nic0"], ["dev0"], proc=proc, sys=sys_root)
    assert data["source"] == "sys_kernel_irq" and data["cpu_count"] == 3
    (one,) = data["irqs"]
    assert (one["irq"], one["name"], one["per_cpu"]) == (
        30, "nic0-comp0", {"2": 10, "1": 2})  # fmt: skip
    assert [m["irq"] for m in data["missing"]] == [31]
    assert data["coverage_problem"].startswith("1 listed interrupts absent")
    (proc / "interrupts").write_text("    CPU0\n 30:   7   PCI-MSI nic0\n")
    assert tasks.nic_interrupts(["nic0"], proc=proc, sys=sys_root)["source"] == (
        "proc_interrupts")  # fmt: skip
    (proc / "softirqs").write_text("")
    with pytest.raises(ObservationError, match="is empty .masked"):
        tasks.net_softirqs(proc=proc)


def test_sysfs_interrupt_counts_follow_the_possible_cpus_and_complete_rows(tmp_path):
    """Per-IRQ sysfs counts are one per possible CPU, in that order: with
    possible CPUs 0,3-4 they map as the /proc row does; a row with another
    number of counts, without its newline or at the page limit is incomplete,
    and an unreadable IRQ is reported while the readable ones are kept; a
    /proc header with no owned row still gives its CPU count."""
    from vllm.kv_profiling.endpoint_probes import tasks

    proc, sys_root = tmp_path / "proc", tmp_path / "sys"
    proc.mkdir()
    (proc / "interrupts").write_text("")
    for irq in (30, 31):
        (sys_root / f"class/net/nic0/device/msi_irqs/{irq}").mkdir(parents=True)
        (sys_root / f"kernel/irq/{irq}").mkdir(parents=True)
    (sys_root / "devices/system/cpu").mkdir(parents=True)
    (sys_root / "devices/system/cpu/possible").write_text("0,3-4\n")
    assert tasks.possible_cpus(sys_root) == [0, 3, 4]
    (sys_root / "kernel/irq/30/per_cpu_count").write_text("0,2,10\n")
    page = os.sysconf("SC_PAGE_SIZE")
    for bad in ("0,2\n", "0,2,1", "x,2,1\n", "0,1," + "9" * page):
        (sys_root / "kernel/irq/31/per_cpu_count").write_text(bad)
        data = tasks.nic_interrupts(["nic0"], proc=proc, sys=sys_root)
        assert [(i["irq"], i["per_cpu"]) for i in data["irqs"]] == [
            (30, {"4": 10, "3": 2})
        ], bad
        assert data["unreadable_irqs"] == [
            {"irq": 31, "error": "incomplete or malformed per_cpu_count"}
        ]
        assert data["missing"] == [] and "1 interrupt counters unreadable" in (
            data["coverage_problem"])  # fmt: skip
    (sys_root / "kernel/irq/31/per_cpu_count").unlink()
    (sys_root / "kernel/irq/31/per_cpu_count").mkdir()  # unreadable as a file
    data = tasks.nic_interrupts(["nic0"], proc=proc, sys=sys_root)
    assert [i["irq"] for i in data["irqs"]] == [30]
    assert data["unreadable_irqs"][0]["irq"] == 31
    (proc / "interrupts").write_text("  CPU0  CPU3  CPU4\n  29:  1  1  1  other\n")
    data = tasks.nic_interrupts(["nic0"], proc=proc, sys=sys_root)
    assert data["cpu_count"] == 3 and data["missing_count"] == 2


def test_a_possible_cpu_list_or_counter_that_is_not_exact_never_moves_counts(
    tmp_path,
):
    """A cut, overlapping or unordered possible-CPU list fails instead of
    assigning counts to other CPUs; a non-ASCII digit in one IRQ's counts or
    an undecodable handler name affects that IRQ only."""
    from vllm.kv_profiling.endpoint_probes import tasks

    proc, sys_root = tmp_path / "proc", tmp_path / "sys"
    proc.mkdir()
    (proc / "interrupts").write_text("")
    for irq in (30, 31):
        (sys_root / f"class/net/nic0/device/msi_irqs/{irq}").mkdir(parents=True)
        (sys_root / f"kernel/irq/{irq}").mkdir(parents=True)
    (sys_root / "devices/system/cpu").mkdir(parents=True)
    possible = sys_root / "devices/system/cpu/possible"
    for bad in ("0-1,3", "0,3,3\n", "3,0\n", "0-\n", "0-1,\n", "2-1\n", "0-1,²\n"):
        possible.write_text(bad)
        with pytest.raises(ObservationError, match="possible CPU list"):
            tasks.possible_cpus(sys_root)
    possible.write_text("0-1,31\n")
    assert tasks.possible_cpus(sys_root) == [0, 1, 31]
    (sys_root / "kernel/irq/30/per_cpu_count").write_text("1,0,10\n")
    (sys_root / "kernel/irq/31/per_cpu_count").write_text("1,²,10\n")
    data = tasks.nic_interrupts(["nic0"], proc=proc, sys=sys_root)
    assert [(i["irq"], i["per_cpu"]) for i in data["irqs"]] == [
        (30, {"31": 10, "0": 1})
    ]
    assert [e["irq"] for e in data["unreadable_irqs"]] == [31]
    (sys_root / "kernel/irq/31/per_cpu_count").write_text("0,4,0\n")
    (sys_root / "kernel/irq/31/actions").write_bytes(b"nic0-\xff\n")
    data = tasks.nic_interrupts(["nic0"], proc=proc, sys=sys_root)
    assert [i["irq"] for i in data["irqs"]] == [30, 31]
    assert data["irqs"][1]["name"] == "nic0-�" and data["unreadable_irqs"] == []


def _sampler(writer, required=()):
    """An observer reduced to its TCP sampling, writing to ``writer``."""
    from vllm.kv_profiling.endpoint_probes.runner import Observer
    from vllm.kv_profiling.endpoint_probes.sockets import Tracker

    observer = Observer.__new__(Observer)
    observer.config = ProbeConfig(tcp={"owned_sockets": True}, required=list(required))
    observer.tracker, observer.sampled_scan, observer.previous = Tracker(), None, {}
    observer.failed_required, observer.endpoint, observer.writer = 0, "d", writer
    return observer


def _sample_once(writer, scan, tracker=None):
    """Record ``scan`` as the observer does: its admissions (which name the
    owners), then its TCP sample; one capture's scans share one tracker."""
    observer = _sampler(writer)
    if tracker is not None:
        observer.tracker = tracker
    for change in observer.tracker.apply(scan):
        observer.emit("socket_ownership", str(change["cookie"]), scan.started_ns,
                      change, kind="probe_status")  # fmt: skip
    observer.last_scan = scan
    observer.sample_tcp()
    return observer


def _tcp_required_failures(tmp_path, rows):
    """Sample ``rows`` as discovery would: only full records with an inode
    can be owned; the compact ones never are."""
    from vllm.kv_profiling.endpoint_probes.runner import Observer
    from vllm.kv_profiling.endpoint_probes.sockets import Scan, Tracker, cookie_of

    observer = Observer.__new__(Observer)
    observer.config = ProbeConfig(tcp={"owned_sockets": True}, required=["tcp"])
    scan = Scan(time.monotonic_ns())
    scan.complete, scan.finished_ns = True, time.monotonic_ns()
    for row in rows:
        if row["record"] == "full" and row["inode"]:
            scan.owned[cookie_of(row)] = {
                "cookie": cookie_of(row), "row": row, "shared": False,
                "loopback": True, "owner_pids": [1], "holders": [],
                "frontend_colocated": False,
                "owners": [{"pid": 1, "start_ticks": 1, "role": "worker"}],
            }  # fmt: skip
    observer.tracker = Tracker()
    observer.tracker.apply(scan)
    observer.last_scan, observer.sampled_scan = scan, None
    observer.input_dir, observer.run_id, observer.role = tmp_path, str(uuid4()), "d"
    observer.metadata = {"net_namespace": os.readlink("/proc/self/ns/net"),
                         "affinity": []}  # fmt: skip
    observer.previous, observer.failed_required, observer.endpoint = {}, 0, "d"
    observer.writer = type("Sink", (), {"put": lambda self, row: None})()
    observer.sample_tcp()
    return observer.failed_required


def test_the_kernels_compact_socket_records_are_not_required_failures(tmp_path):
    """The kernel reports request sockets (SYN_RECV, timer 1) and time-wait
    sockets (FIN_WAIT2 or TIME_WAIT, timer 3) as compact records without
    TCP_INFO: real ones are read here, and none fails a required TCP source;
    a full socket (FIN_WAIT2 while open) carries TCP_INFO and still fails
    the source when it lacks it."""
    from vllm.kv_profiling.endpoint_probes.sockets import Budget, Limits, _dump

    loopback = {}

    def dumped(selectors):
        return [
            row
            for row in _dump(Budget(Limits()), [])
            if row["local_ip"] == "127.0.0.1"
            and all(row[key[:-1]] in values for key, values in selectors.items())
        ]

    def only(selectors, state, **expect):
        wait_for(lambda: [(r["state"], r["record"]) for r in dumped(selectors)]
                 == [(state, expect["record"])])  # fmt: skip
        (row,) = dumped(selectors)
        assert {k: row[k] for k in expect} == expect
        return row

    with socket.socket() as listener, socket.socket() as client:
        listener.setsockopt(socket.IPPROTO_TCP, socket.TCP_DEFER_ACCEPT, 5)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        client.connect(listener.getsockname())  # no data: the accept is deferred
        request = only({**loopback, "local_ports": [listener.getsockname()[1]],
                        "peer_ports": [client.getsockname()[1]]}, 3,
                       record="request", timer=1, inode=0, tcp_info_length=0,
                       tcp_info_expected=False)  # fmt: skip
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        client = socket.socket()
        # Orphaned FIN_WAIT2 becomes a time-wait record only when its timeout
        # is at most TIME_WAIT's (60 s); pin it, whatever the host's default.
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_LINGER2, 12)
        client.connect(listener.getsockname())
        server, _ = listener.accept()
        selectors = {**loopback, "local_ports": [client.getsockname()[1]]}
        client.shutdown(socket.SHUT_WR)
        assert server.recv(1) == b""
        full = only(selectors, 5, record="full", tcp_info_expected=True)
        assert full["tcp_info_length"] > 0 and full["inode"] > 0
        client.close()  # orphaned in FIN_WAIT2: a time-wait record
        fin_wait2 = only(selectors, 5, record="time_wait", timer=3, inode=0,
                         tcp_info_length=0, tcp_info_expected=False)  # fmt: skip
        server.close()
        time_wait = only(selectors, 6, record="time_wait", timer=3, inode=0,
                         tcp_info_length=0, tcp_info_expected=False)  # fmt: skip
    assert _tcp_required_failures(tmp_path, [request, fin_wait2, time_wait]) == 0
    assert _tcp_required_failures(tmp_path, [{**full, "tcp_info_length": 0}]) == 1


@pytest.mark.parametrize(
    "ready, failed, exit_code, clean",
    [
        (True, False, 2, True),
        (True, True, 2, False),
        (False, False, 2, False),
        (True, False, 1, False),
    ],
)
def test_ebpf_a_required_source_is_judged_on_a_clean_run_not_on_coverage(
    tmp_path, ready, failed, exit_code, clean
):
    """A required eBPF source is judged on a clean run, never on coverage.

    Its coverage is never provably complete, so it must be ready and available,
    record no failure, and end with its own unfailed terminal record, a normal
    exit and its final accounting."""
    from vllm.kv_profiling.endpoint_probes.ebpf import Channel

    ready_row = {"type": "ready", "available": True, "exact": False,
                 "coverage": "partial"}  # fmt: skip
    end_row = {"type": "end", "complete": False, "coverage": "partial",
               "output_drops": 0, "malformed": 0, "failed": failed}  # fmt: skip
    code = (
        "import json,sys;"
        + (f"print({json.dumps(json.dumps(ready_row))},flush=True);" if ready else "")
        + "assert sys.stdin.readline()=='stop\\n';"
        f"print({json.dumps(json.dumps(end_row))},flush=True);"
        f"sys.exit({exit_code})"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    channel = Channel(child, tmp_path)
    writer = Writer(tmp_path / "writer", metadata(), ProbeConfig())
    try:
        if ready:
            deadline = time.monotonic() + 5
            while not channel.status.get("available"):
                assert time.monotonic() < deadline
                channel.poll(writer, 0.05)
        result = channel.stop(writer)
        assert result["complete"] is False
        assert result["clean"] is clean
    finally:
        if channel.running():
            child.kill()
            channel.stop(writer)


def _fake_collector(monkeypatch, code):
    """Replace the native collector's launch by a Python child running ``code``."""
    from vllm.kv_profiling.endpoint_probes import ebpf

    def start(self):
        self.status.update(state="unavailable", mode=self.settings.mode)
        self.child = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.open_pipes()

    monkeypatch.setattr(ebpf.Collector, "start", start)


def _collector_code(ready_after_s, end_row, marker=None):
    lines = ["import json,sys,time,pathlib"]
    if ready_after_s is not None:
        lines.append(f"time.sleep({ready_after_s})")
        if marker is not None:
            lines.append(f"pathlib.Path({str(marker)!r}).touch()")
        lines.append(
            "print(json.dumps({'type':'ready','available':True,"
            "'exact':False,'coverage':'partial'}),flush=True)"
        )
    lines += [
        "sys.stdin.readline()",
        f"print({json.dumps(json.dumps(end_row))},flush=True)",
        "sys.exit(2)",
    ]
    return ";".join(lines)


@pytest.mark.parametrize("ready_after_s", [0.8, 2.6, None])
def test_ebpf_the_capture_is_ready_only_after_the_collector_answers(
    tmp_path, monkeypatch, ready_after_s
):
    """The manifest's ``ready``, the signal a controller waits for, appears only
    once the collector has answered: attached when it is ready late, or, when it
    does not answer within the observer's bound (2 s here), at that bound, with
    the missed deadline a required-source failure even if it answers later."""
    import contextlib
    import threading

    from vllm.kv_profiling.endpoint_probes import runner
    from vllm.kv_profiling.endpoint_probes.runner import Observer

    end_row = {"type": "end", "complete": False, "coverage": "partial",
               "output_drops": 0, "malformed": 0, "failed": False}  # fmt: skip
    answered = tmp_path / "answered"
    _fake_collector(monkeypatch, _collector_code(ready_after_s, end_row, answered))
    monkeypatch.setattr(runner, "EBPF_ANSWER_TIMEOUT_S", 2.0)
    original = os.sched_getaffinity(0)
    cpus = sorted(original)
    owner, entries, directory = start_recorded_endpoint(tmp_path, cpus[0])
    (directory / "close").touch()
    wait_for(lambda: (directory / "closed").exists(), timeout=20)
    run = entries[0]["run_id"]
    built, seen = [], []  # (ready, collector answered, seconds since start)

    def build():
        built.append(Observer(
            ProbeConfig(ebpf={"enabled": True, "privileged_opt_in": True,
                              "mode": "tcp_tracepoints", "artifacts": str(tmp_path)},
                        required=["ebpf"]),
            run_id=run, endpoint="d", role="decode", owner_pid=owner.pid,
            owner_start=process_identity(owner.pid)["start_ticks"],
            inventory=entries, cpus=[cpus[-1]],
            input_dir=str(directory / "profiles"),
            output_dir=str(tmp_path / "out"),
            irq_policy="virtual_not_applicable",
        ))  # fmt: skip

    try:
        thread = threading.Thread(target=build)
        started = time.monotonic()
        thread.start()
        while thread.is_alive():
            for manifest in (tmp_path / "out").glob("*/*/manifest.json"):
                with contextlib.suppress(ValueError):
                    ready = json.loads(manifest.read_text()).get("ready")
                    seen.append((ready, answered.exists(), time.monotonic() - started))
            time.sleep(0.02)
        thread.join()
        observer = built[0]
        manifest = next((tmp_path / "out").glob("*/*/manifest.json"))
        assert any(not ready for ready, _, _ in seen)
        on_time = ready_after_s is not None and ready_after_s < 2.0
        for ready, was_answered, elapsed in seen:
            if ready and on_time:
                assert was_answered  # never ready before the collector answered
            if ready and not on_time:
                assert elapsed >= 2.0  # only at the observer's bound
        assert json.loads(manifest.read_text())["ready"] is True
        if on_time:
            assert observer.ebpf.status.get("available") is True
        (tmp_path / "stop").touch()
        (tmp_path / "children").write_text(
            json.dumps({"run_id": run, "endpoint": "d", "children_closed": True})
        )
        complete = observer.run(
            stop_file=tmp_path / "stop", endpoint_complete_file=tmp_path / "children"
        )
        stats = json.loads(manifest.read_text())["end"]["stats"]
        assert complete is on_time
        assert (stats["required_source_failures"] > 0) is (not on_time)
    finally:
        os.sched_setaffinity(0, original)
        (directory / "exit").touch()
        owner.wait(10)


@pytest.mark.parametrize(
    "losses, output_drops, degraded",
    [
        ([("endpoint_namespace", "scoped_ring_drop", 2)], 0, True),
        ([], 3, True),
        (
            [
                ("filter_exclusion", "excluded_port", 5),
                ("global_unknown", "global_nonfull", 4),
            ],
            0,
            False,
        ),
    ],
)
def test_ebpf_known_event_loss_marks_a_clean_run_degraded(
    tmp_path, losses, output_drops, degraded
):
    """A clean run that lost endpoint events (ring or reads) or output is
    degraded; ordinary exclusions are not loss."""
    from vllm.kv_profiling.endpoint_probes.ebpf import Channel

    rows = [{"type": "ready", "available": True, "exact": False,
             "coverage": "partial"}]  # fmt: skip
    rows += [{"type": "loss", "scope": scope, "counter": counter, "value": value}
             for scope, counter, value in losses]  # fmt: skip
    end_row = {"type": "end", "complete": False, "coverage": "partial",
               "output_drops": output_drops, "malformed": 0,
               "failed": False}  # fmt: skip
    printed = [f"print({json.dumps(json.dumps(row))},flush=True)" for row in rows]
    code = ";".join(["import json,sys", *printed,
                     "sys.stdin.readline()",
                     f"print({json.dumps(json.dumps(end_row))},flush=True)",
                     "sys.exit(2)"])  # fmt: skip
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    channel = Channel(child, tmp_path)
    writer = Writer(tmp_path / "writer", metadata(), ProbeConfig())
    try:
        deadline = time.monotonic() + 5
        while not channel.status.get("available"):
            assert time.monotonic() < deadline
            channel.poll(writer, 0.05)
        time.sleep(0.2)
        channel.poll(writer, 0.05)
        result = channel.stop(writer)
        assert result["clean"] is True
        assert result["degraded"] is degraded
        if output_drops:
            assert result["event_loss"]["terminal.output_drops"] == output_drops
        assert not any(
            k.startswith(("filter_exclusion", "global_unknown"))
            for k in result["event_loss"]
        )
    finally:
        if channel.running():
            child.kill()
            channel.stop(writer)


def test_ebpf_saved_capture_loss_comes_from_the_recorded_counters(
    tmp_path, monkeypatch
):
    """A capture's eBPF loss is derived from the loss counters and terminal
    counts it recorded, never taken from its footer alone; exclusions are not
    loss."""
    from vllm.kv_profiling.endpoint_probes.health import inspect_trace
    from vllm.kv_profiling.endpoint_probes.runner import Observer

    end_row = {"type": "end", "complete": False, "coverage": "partial",
               "output_drops": 0, "malformed": 0, "failed": False}  # fmt: skip
    losses = [
        {"type": "loss", "scope": "endpoint_namespace",
         "counter": "scoped_ring_drop", "value": 2},
        {"type": "loss", "scope": "filter_exclusion",
         "counter": "excluded_port", "value": 7},
    ]  # fmt: skip
    code = _collector_code(0.0, end_row).replace(
        "sys.stdin.readline()",
        ";".join(f"print({json.dumps(json.dumps(r))},flush=True)" for r in losses)
        + ";sys.stdin.readline()",
    )
    _fake_collector(monkeypatch, code)
    original = os.sched_getaffinity(0)
    cpus = sorted(original)
    owner, entries, directory = start_recorded_endpoint(tmp_path, cpus[0])
    (directory / "close").touch()
    wait_for(lambda: (directory / "closed").exists(), timeout=20)
    run = entries[0]["run_id"]
    try:
        observer = Observer(
            ProbeConfig(ebpf={"enabled": True, "privileged_opt_in": True,
                              "mode": "tcp_tracepoints", "artifacts": str(tmp_path)},
                        required=["ebpf"]),
            run_id=run, endpoint="d", role="decode", owner_pid=owner.pid,
            owner_start=process_identity(owner.pid)["start_ticks"],
            inventory=entries, cpus=[cpus[-1]],
            input_dir=str(directory / "profiles"),
            output_dir=str(tmp_path / "out"),
            irq_policy="virtual_not_applicable",
        )  # fmt: skip
        time.sleep(1.5)  # a sample or two: the loss rows are recorded
        (tmp_path / "stop").touch()
        (tmp_path / "children").write_text(
            json.dumps({"run_id": run, "endpoint": "d", "children_closed": True})
        )
        observer.run(
            stop_file=tmp_path / "stop", endpoint_complete_file=tmp_path / "children"
        )
        events = next((tmp_path / "out").glob("*/*/events.jsonl"))
        derived = inspect_trace(events, probe=True)["ebpf_event_loss"]
        assert derived == {"endpoint_namespace.scoped_ring_drop": 2}
        manifest = json.loads(events.with_name("manifest.json").read_text())
        stats = manifest["end"]["stats"]
        assert stats["ebpf_degraded"] is True and stats["ebpf_event_loss"] == derived
    finally:
        os.sched_setaffinity(0, original)
        (directory / "exit").touch()
        owner.wait(10)


def test_ebpf_the_collector_s_plan_carries_only_its_identity_and_ports(
    tmp_path, monkeypatch
):
    """The plan names the observer, its namespace and the optional port lists;
    no address (the selection is the owned sockets, sent as commands)."""
    from vllm.kv_profiling.endpoint_probes import ebpf

    monkeypatch.setattr(ebpf, "effective_privilege", lambda: True)
    monkeypatch.setattr(ebpf, "validate_artifacts", lambda root: {})
    launched, real_popen = [], subprocess.Popen

    def popen(argv, **kwargs):
        launched.append(Path(argv[2]).read_text())
        return real_popen(
            [sys.executable, "-c", "pass"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    monkeypatch.setattr(ebpf.subprocess, "Popen", popen)
    for name, options in [("scoped", {"peer_ports": [7]}), ("any", {})]:
        collector = ebpf.Collector(
            ebpf.Settings(enabled=True, privileged_opt_in=True,
                          mode="tcp_tracepoints", artifacts=str(tmp_path), **options),
            [], tmp_path / name)  # fmt: skip
        collector.start()
        assert collector.child is not None, collector.status.get("reason")
        collector.child.wait(10)
        collector.child.stdin.close()
    lines = [plan.splitlines()[2:] for plan in launched]
    assert lines[0] == ["L ", "R 7"]
    assert lines[1] == ["L ", "R "]


@pytest.mark.parametrize(
    "change, problem",
    [
        ({}, None),
        ({"exit_code": 3}, "not clean"),
        ({"exit_code": None, "signal": 9, "killed": "signal"}, "not clean"),
        ({"killed": "oom"}, "not clean"),
        ({"timed_out": True}, "not clean"),
        ({"problems": ["the container was not removed"]}, "not clean"),
        ({"process_id": "a-previous-attempt"}, "another run, endpoint or attempt"),
        ({"endpoint": "other"}, "another run, endpoint or attempt"),
    ],
)
def test_the_exit_record_shows_this_observer_attempt_exited_cleanly(
    tmp_path, change, problem
):
    """Any launcher publishes the record after reaping the observer; only a
    clean exit of this run's, endpoint's and attempt's observer passes."""
    from vllm.kv_profiling.endpoint_probes.exit_record import (
        exit_problems,
        write_exit_record,
    )

    fields = {"run_id": "run", "endpoint": "d", "process_id": "attempt",
              "exit_code": 0, "launcher": "test"}  # fmt: skip
    write_exit_record(tmp_path, **{**fields, **change})
    found = exit_problems(tmp_path, "run", "d", "attempt")
    assert (found == []) if problem is None else (problem in found[0])
    assert not list(tmp_path.glob("*.tmp"))  # published atomically


def test_a_missing_or_malformed_exit_record_is_never_a_clean_exit(tmp_path):
    from vllm.kv_profiling.endpoint_probes.exit_record import exit_problems

    assert exit_problems(tmp_path, "run", "d", "attempt") == [
        "observer exit record missing"]  # fmt: skip
    for text in ("{", json.dumps({"schema": 1, "exit_code": "0"}), "[]"):
        (tmp_path / "exit.json").write_text(text)
        assert exit_problems(tmp_path, "run", "d", "attempt")[0] in (
            "observer exit record unreadable",
            "observer exit record malformed",
        )


@pytest.mark.parametrize(
    "change",
    [
        {"schema": True},
        {"schema": 1.0},
        {"launcher": ""},
        {"run_id": ""},
        {"process_id": ""},
        {"exit_code": True},
        {"exit_code": 0.0},
        {"signal": "9"},
        {"killed": "maybe"},
        {"timed_out": 0},
        {"problems": [1]},
        {"evidence": []},
        "drop:run_id",
        "drop:signal",
        "drop:evidence",
    ],
)
def test_an_exit_record_with_a_wrong_type_or_missing_field_is_malformed(
    tmp_path, change
):
    """Every field must be present with its type; a record that is not is
    malformed, never an error and never a clean exit."""
    from vllm.kv_profiling.endpoint_probes.exit_record import (
        exit_problems,
        write_exit_record,
    )

    write_exit_record(tmp_path, run_id="run", endpoint="d", process_id="attempt",
                      exit_code=0, launcher="test")  # fmt: skip
    record = json.loads((tmp_path / "exit.json").read_text())
    if isinstance(change, str):
        del record[change.split(":", 1)[1]]
    else:
        record.update(change)
    (tmp_path / "exit.json").write_text(json.dumps(record))
    assert exit_problems(tmp_path, "run", "d", "attempt") == [
        "observer exit record malformed"]  # fmt: skip


def test_overlapping_exit_publications_never_collide_or_leave_partial_records(
    tmp_path,
):
    """Each publication writes its own temporary file and replaces the record
    atomically: concurrent attempts leave one valid record and no residue."""
    import multiprocessing

    with multiprocessing.get_context("spawn").Pool(6) as pool:
        results = pool.map(_publish_exit, [(str(tmp_path), n) for n in range(24)])
    assert results == [None] * 24
    from vllm.kv_profiling.endpoint_probes.exit_record import exit_problems

    assert exit_problems(tmp_path, "run", "d", "attempt") == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["exit.json"]


def _publish_exit(arguments):
    from vllm.kv_profiling.endpoint_probes.exit_record import write_exit_record

    directory, attempt = arguments
    try:
        write_exit_record(directory, run_id="run", endpoint="d",
                          process_id="attempt", exit_code=0,
                          launcher=f"writer-{attempt}")  # fmt: skip
    except Exception as error:  # reported to the test, not raised in the pool
        return repr(error)
    return None


@pytest.mark.parametrize(
    "fields, accepted",
    [
        ({}, True),  # the guide's example
        (
            {
                "schema": 1,
                "signal": None,
                "killed": None,
                "timed_out": False,
                "problems": [],
                "evidence": {},
            },
            True,
        ),  # the full record  # fmt: skip
        ({"problems": {}}, False),
        ({"evidence": []}, False),
        ({"schema": True}, False),
        ({"schema": 2}, False),
        ({"exit_code": "0"}, False),
        ({"killed": "maybe"}, False),
    ],
)
def test_record_exit_through_the_cli_publishes_only_well_formed_input(
    tmp_path, fields, accepted
):
    """The public ``--record-exit`` refuses mistyped input before writing
    anything, rather than converting it into a clean-looking record."""
    from vllm.kv_profiling.endpoint_probes.exit_record import exit_problems

    root = Path(__file__).resolve().parents[3]
    if not (root / "vllm/kv_profiling/probe.py").exists():
        root = Path(os.environ["KV_PROFILER_SOURCE_ROOT"])
    record = {"run_id": "run", "endpoint": "decode0", "process_id": "attempt",
              "exit_code": 0, "launcher": "my-launcher", **fields}  # fmt: skip
    done = subprocess.run(
        [sys.executable, str(root / "vllm/kv_profiling/probe.py"),
         "--record-exit", str(tmp_path)],
        input=json.dumps(record), capture_output=True, text=True, timeout=30,
    )  # fmt: skip
    assert (done.returncode == 0) is accepted, done.stderr
    if accepted:
        assert exit_problems(tmp_path, "run", "decode0", "attempt") == []
    else:
        assert list(tmp_path.iterdir()) == []  # nothing written


def test_the_observer_owner_is_the_api_servers_common_parent(monkeypatch):
    """One API server is its own owner; several (data parallelism) give their
    one common parent, checked against each server's start ticks, read by the
    observer from its own /proc."""
    from vllm.kv_profiling.endpoint_probes import identity

    stats = {11: (7, 201), 12: (7, 202), 7: (1, 150), 13: (8, 203)}

    def fields(pid):
        parent, start = stats[pid]
        return ["S", str(parent)] + ["0"] * 17 + [str(start)]

    monkeypatch.setattr(identity, "_stat_fields", fields)
    owner, evidence = identity.endpoint_owner(
        [{"pid": 11, "start_ticks": 201}, {"pid": 12, "start_ticks": 202}]
    )
    assert owner == {"pid": 7, "start_ticks": 150}
    # the evidence it froze: each frontend's parent as read
    assert evidence == {
        "frontends": [{"pid": 11, "start_ticks": 201, "parent_pid": 7},
                      {"pid": 12, "start_ticks": 202, "parent_pid": 7}],
        "parent": owner,
    }  # fmt: skip
    one = [{"pid": 11, "start_ticks": 201}]
    assert identity.endpoint_owner(one) == (
        {"pid": 11, "start_ticks": 201}, {"frontends": [], "parent": None})  # fmt: skip
    with pytest.raises(ValueError, match="no single parent"):
        identity.endpoint_owner(
            [{"pid": 11, "start_ticks": 201}, {"pid": 13, "start_ticks": 203}]
        )
    with pytest.raises(ValueError, match="changed"):
        identity.endpoint_owner(
            [{"pid": 11, "start_ticks": 999}, {"pid": 12, "start_ticks": 202}]
        )


def test_a_bare_process_observer_discovers_its_endpoint_and_checks_complete(tmp_path):
    """No container anywhere: a launcher names only the expected recorders; the
    observer, run as a plain process from the standalone entry point, finds and
    freezes the inventory and owner, the launcher records its exit, and the
    capture checks complete."""
    allowed = sorted(os.sched_getaffinity(0))
    assert len(allowed) >= 2, "probe tests require separate owner/observer CPUs"
    owner, inventory, endpoint_dir = start_recorded_endpoint(tmp_path, allowed[0])
    run_id = inventory[0]["run_id"]
    core_dir = endpoint_dir / "profiles"
    observer_dir = tmp_path / "observer"
    observer_dir.mkdir()
    settings = ProbeConfig(interval_s=0.1, slow_interval_s=0.1, interfaces=["lo"],
                           required=["netdev"])  # fmt: skip
    config = {
        "probes": asdict(settings),
        "run_id": run_id,
        "endpoint": "d",
        "role": "decode",
        "expected": {"frontend": 1, "scheduler": 1, "worker": 1},
        "cpus": [allowed[-1]],
        "input_dir": str(core_dir),
        "output_dir": str(observer_dir / "captures"),
        "irq_policy": "virtual_not_applicable",
        "stop_file": str(observer_dir / "stop"),
        "endpoint_complete_file": str(observer_dir / "children.json"),
    }
    (observer_dir / "config.json").write_text(json.dumps(config))
    entry = str(Path(probe.__file__))
    process = subprocess.Popen(
        [sys.executable, entry, "--config", str(observer_dir / "config.json")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )  # fmt: skip
    try:
        manifest_glob = f"captures/{run_id}/*/manifest.json"

        def ready():
            paths = observer_dir.glob(manifest_glob)
            return any(json.loads(p.read_text()).get("ready") for p in paths)

        wait_for(ready, timeout=20)
        (endpoint_dir / "close").touch()
        wait_for(lambda: (endpoint_dir / "closed").exists(), timeout=20)
        (observer_dir / "children.json").write_text(
            json.dumps({"run_id": run_id, "endpoint": "d", "children_closed": True})
        )
        (observer_dir / "stop").touch()
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stderr
        manifest = next(observer_dir.glob(manifest_glob))
        meta = json.loads(manifest.with_name("events.jsonl").open().readline())
        assert meta["inventory"] == inventory
        found = (meta["owner"]["pid"], meta["owner"]["start_ticks"])
        assert found == (owner.pid, process_identity(owner.pid)["start_ticks"])
        record = {
            "run_id": run_id,
            "endpoint": "d",
            "process_id": meta["process_id"],
            "exit_code": process.returncode,
            "launcher": "test-process",
        }
        subprocess.run([sys.executable, entry, "--record-exit", str(observer_dir)],
                       input=json.dumps(record), text=True, check=True)  # fmt: skip
        checked = subprocess.run(
            [sys.executable, entry, "--check-capture", str(observer_dir),
             "--core-dir", str(core_dir)],
            capture_output=True, text=True, timeout=30,
        )  # fmt: skip
        summary = json.loads(checked.stdout)
        assert checked.returncode == 0 and summary["complete"], summary
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        (endpoint_dir / "exit").touch()
        owner.wait(10)


def test_discovery_refuses_recorders_other_than_the_expected_ones(tmp_path):
    from vllm.kv_profiling.endpoint_probes.identity import discover

    allowed = sorted(os.sched_getaffinity(0))
    owner, inventory, endpoint_dir = start_recorded_endpoint(tmp_path, allowed[0])
    try:
        run_id = inventory[0]["run_id"]
        one_each = {"frontend": 1, "scheduler": 1, "worker": 1}
        found, found_owner, _ = discover(
            endpoint_dir / "profiles", run_id, one_each, "decode"
        )
        assert found == inventory and found_owner["pid"] == owner.pid
        with pytest.raises(ValueError, match="are not prefill's"):
            discover(endpoint_dir / "profiles", run_id, one_each, "prefill")
        for expected in ({"frontend": 1, "scheduler": 1, "worker": 2},
                         {"frontend": 0, "scheduler": 1, "worker": 1},
                         {"frontend": 1, "scheduler": 1}):  # fmt: skip
            with pytest.raises(ValueError):
                discover(endpoint_dir / "profiles", run_id, expected, "decode")
    finally:
        (endpoint_dir / "exit").touch()
        owner.wait(10)


def test_unreadable_namespaces_name_the_capability_cause(monkeypatch):
    from vllm.kv_profiling.endpoint_probes import identity

    real = identity.os.readlink

    def denied(path, *args, **kwargs):
        if "/ns/" in str(path) and not str(path).startswith("/proc/self"):
            raise PermissionError(13, "Permission denied", str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(identity.os, "readlink", denied)
    with pytest.raises(identity.NamespacesUnreadable) as raised:
        identity.process_identity(os.getpid())
    message = str(raised.value)
    # the denied path and errno, and both kernel causes, are kept
    for part in (f"/proc/{os.getpid()}/ns/net", "errno 13", "not dumpable",
                 "CAP_SYS_PTRACE"):  # fmt: skip
        assert part in message, message
    assert issubclass(identity.NamespacesUnreadable, ObservationError)


def test_the_code_digest_is_the_same_from_a_zipapp_and_from_the_tree(tmp_path):
    """Captures record what actually ran; a zipapp of the same modules and the
    source tree give the same digest."""
    import zipfile

    from vllm.kv_profiling.endpoint_probes.code import code_digest

    package = Path(probe.__file__).with_name("endpoint_probes")
    archive = tmp_path / "observer.pyz"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(probe.__file__, "__main__.py")
        for source in sorted(package.glob("*.py")):
            bundle.write(source, f"endpoint_probes/{source.name}")
    zipped = subprocess.run([sys.executable, str(archive), "--code-digest"],
                            capture_output=True, text=True, check=True)  # fmt: skip
    assert zipped.stdout.strip() == code_digest()
    # the launcher is part of what ran: a changed one changes the digest
    changed = tmp_path / "changed.pyz"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(changed, "w") as bundle:
        for item in source.namelist():
            data = source.read(item)
            if item == "__main__.py":
                data += b"\n# changed\n"
            bundle.writestr(item, data)
    other = subprocess.run([sys.executable, str(changed), "--code-digest"],
                           capture_output=True, text=True, check=True)  # fmt: skip
    assert other.stdout.strip() != code_digest()


def _observer_config(tmp_path, run_id, core_dir, cpu, **override):
    settings = ProbeConfig(interval_s=0.1, slow_interval_s=0.1, interfaces=["lo"],
                           required=["netdev"])  # fmt: skip
    config = {
        "probes": asdict(settings),
        "run_id": run_id,
        "endpoint": "d",
        "role": "decode",
        "expected": {"frontend": 1, "scheduler": 1, "worker": 1},
        "cpus": [cpu],
        "input_dir": str(core_dir),
        "output_dir": str(tmp_path / "observer" / "captures"),
        "irq_policy": "virtual_not_applicable",
        "stop_file": str(tmp_path / "observer" / "stop"),
        "endpoint_complete_file": str(tmp_path / "observer" / "children.json"),
        **override,
    }
    (tmp_path / "observer").mkdir(exist_ok=True)
    path = tmp_path / "observer" / "config.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize(
    "override, refusal",
    [
        ({"role": "prefill"}, "are not prefill's"),
        ({"expected": None}, "expected recorders"),
        ({"inventory": []}, "expected recorders"),
        ({"owner_pid": 1, "owner_start": 1}, "expected recorders"),
    ],
)
def test_the_cli_launches_only_the_discovery_form_on_its_own_side(
    tmp_path, override, refusal
):
    """One launch form (expected recorders, discovered by the observer), and
    only on the endpoint side it is configured for: refused before readiness."""
    allowed = sorted(os.sched_getaffinity(0))
    owner, inventory, endpoint_dir = start_recorded_endpoint(tmp_path, allowed[0])
    try:
        run_id, core_dir = inventory[0]["run_id"], endpoint_dir / "profiles"
        config = _observer_config(tmp_path, run_id, core_dir, allowed[-1], **override)
        done = subprocess.run(
            [sys.executable, str(Path(probe.__file__)), "--config", str(config)],
            capture_output=True, text=True, timeout=30,
        )  # fmt: skip
        assert done.returncode != 0 and refusal in done.stderr, done.stderr
        assert not list((tmp_path / "observer").glob("captures/*/*/manifest.json"))
    finally:
        (endpoint_dir / "exit").touch()
        owner.wait(10)


def test_the_owner_is_checked_offline_against_its_frozen_evidence():
    """A saved capture's owner must be what discovery showed: the single
    frontend, or the frontends' one recorded parent."""
    from vllm.kv_profiling.endpoint_probes.health import owner_problems

    def entry(role, pid, start):
        return {"pid": pid, "start_ticks": start, "context": {"role": role}}

    frozen = [entry("frontend", 11, 201), entry("frontend", 12, 202),
              entry("scheduler", 13, 203), entry("worker", 14, 204)]  # fmt: skip
    evidence = {"frontends": [{"pid": 11, "start_ticks": 201, "parent_pid": 7},
                              {"pid": 12, "start_ticks": 202, "parent_pid": 7}],
                "parent": {"pid": 7, "start_ticks": 150}}  # fmt: skip
    owner = {"pid": 7, "start_ticks": 150}
    assert owner_problems(owner, frozen, evidence) == []
    for bad in ({}, {"pid": 7}, {"pid": 14, "start_ticks": 204},
                {"pid": 7, "start_ticks": 151}):  # fmt: skip
        assert owner_problems(bad, frozen, evidence)
    assert owner_problems(owner, frozen, {**evidence, "supplied": True})
    assert owner_problems(owner, frozen, None)
    moved = {**evidence, "frontends": [{**evidence["frontends"][0], "start_ticks": 1},
                                       evidence["frontends"][1]]}  # fmt: skip
    assert owner_problems(owner, frozen, moved)
    single = [entry("frontend", 11, 201), entry("worker", 14, 204)]
    alone = {"frontends": [], "parent": None}
    assert owner_problems({"pid": 11, "start_ticks": 201}, single, alone) == []
    assert owner_problems({"pid": 14, "start_ticks": 204}, single, alone)


@pytest.mark.parametrize(
    "change",
    [
        {"owner_pid": 7},
        {"inventory": None},
        {"owner_start": None},
        {"expected": {"frontend": 1, "scheduler": 1, "worker": 2.0}},
        {"expected": {"frontend": 1, "scheduler": 1, "worker": True}},
        {"expected": {"frontend": 0, "scheduler": 1, "worker": 1}},
        {"expected": {"frontend": 1, "scheduler": 1}},
        {"expected": None},
    ],
)
def test_one_launch_config_rule_for_launch_and_check(change):
    """Launch and offline check share one rule: reserved identity keys are
    refused on presence, and role counts are integers."""
    from vllm.kv_profiling.endpoint_probes.health import (
        expected_problems,
        launch_config_problems,
    )

    good = {"expected": {"frontend": 1, "scheduler": 1, "worker": 1}}
    assert launch_config_problems(good) == []
    assert launch_config_problems({**good, **change})
    if "expected" in change:
        assert expected_problems(change["expected"])


def _sidecar(tmp_path, run_id, observer_cpu, **settings):
    """A sidecar process beside an endpoint, as a pod would run it."""
    probes = ProbeConfig(interval_s=0.1, slow_interval_s=0.1, interfaces=["lo"],
                         required=["netdev"])  # fmt: skip
    values = {
        "probes": asdict(probes),
        "role": "decode",
        "expected": {"frontend": 1, "scheduler": 1, "worker": 1},
        "profiles_dir": str(tmp_path / "endpoint" / "profiles"),
        "observer_root": str(tmp_path / "kvp"),
        "irq_policy": "virtual_not_applicable",
        "cpus": [observer_cpu],
        "startup_timeout_s": 30,
        **settings,
    }
    path = tmp_path / "sidecar.json"
    path.write_text(json.dumps(values))
    return subprocess.Popen(
        [sys.executable, str(Path(probe.__file__)), "--sidecar", str(path),
         "--run-id", run_id, "--endpoint", "d"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )  # fmt: skip


def _check(attempt, core_dir):
    done = subprocess.run(
        [sys.executable, str(Path(probe.__file__)), "--check-capture", str(attempt),
         "--core-dir", str(core_dir)],
        capture_output=True, text=True, timeout=30,
    )  # fmt: skip
    return json.loads(done.stdout)


@pytest.mark.parametrize("ending", ["server_exits_then_sigterm", "sigterm_first"])
def test_the_sidecar_follows_the_pod_lifecycle_to_a_complete_capture(tmp_path, ending):
    """Started before the server (which can then start), the sidecar waits for
    the expected recorders, marks readiness once its observer is ready, and on
    termination waits for the L1 captures to close, publishes completion, stops
    and reaps the observer and records its exit: the capture checks complete."""
    allowed = sorted(os.sched_getaffinity(0))
    run_id = str(uuid4())
    sidecar = _sidecar(tmp_path, run_id, allowed[-1])
    try:
        time.sleep(0.5)
        assert sidecar.poll() is None and not (tmp_path / "kvp" / "ready").exists()
        owner, _, endpoint_dir = start_recorded_endpoint(
            tmp_path, allowed[0], run_id=run_id
        )
        wait_for(lambda: (tmp_path / "kvp" / "ready").exists(), timeout=30)
        attempt = Path((tmp_path / "kvp" / "ready").read_text())
        (endpoint_dir / "close").touch()
        wait_for(lambda: (endpoint_dir / "closed").exists(), timeout=20)
        if ending == "server_exits_then_sigterm":
            # Kubernetes stops the main container first, then the sidecar
            (endpoint_dir / "exit").touch()
            owner.wait(10)
            time.sleep(0.3)
            sidecar.send_signal(signal.SIGTERM)
        else:
            # SIGTERM while the server still runs: readiness is withdrawn at
            # once, and completion waits for the processes to exit
            sidecar.send_signal(signal.SIGTERM)
            wait_for(lambda: not (tmp_path / "kvp" / "ready").exists(), timeout=5)
            assert sidecar.poll() is None
            time.sleep(1.5)
            (endpoint_dir / "exit").touch()
            owner.wait(10)
        stdout, stderr = sidecar.communicate(timeout=40)
        assert sidecar.returncode == 0, stderr
        assert not (tmp_path / "kvp" / "ready").exists()
        record = json.loads((attempt / "exit.json").read_text())
        assert (record["exit_code"], record["launcher"], record["timed_out"]) == (
            0, "sidecar", False)  # fmt: skip
        summary = _check(attempt, endpoint_dir / "profiles")
        assert summary["complete"], summary
    finally:
        if sidecar.poll() is None:
            sidecar.kill()
            sidecar.wait()
        if (tmp_path / "endpoint").exists():
            (tmp_path / "endpoint" / "exit").touch()


def test_a_forced_stop_records_an_incomplete_capture_never_a_hang(tmp_path):
    """SIGTERM while the server keeps recording: within its grace the sidecar
    publishes that the children did not close, stops the observer and records
    its exit; the capture is incomplete."""
    allowed = sorted(os.sched_getaffinity(0))
    run_id = str(uuid4())
    owner, _, endpoint_dir = start_recorded_endpoint(
        tmp_path, allowed[0], run_id=run_id
    )
    sidecar = _sidecar(tmp_path, run_id, allowed[-1], stop_grace_s=1)
    try:
        wait_for(lambda: (tmp_path / "kvp" / "ready").exists(), timeout=30)
        attempt = Path((tmp_path / "kvp" / "ready").read_text())
        started = time.monotonic()
        sidecar.send_signal(signal.SIGTERM)
        sidecar.communicate(timeout=30)
        assert time.monotonic() - started < 15
        children = json.loads((attempt / "children.json").read_text())
        assert children["children_closed"] is False
        assert (attempt / "exit.json").exists()
        assert not _check(attempt, endpoint_dir / "profiles")["complete"]
    finally:
        if sidecar.poll() is None:
            sidecar.kill()
            sidecar.wait()
        (endpoint_dir / "exit").touch()
        owner.wait(10)


def test_a_restarted_sidecar_never_uses_an_earlier_attempt_s_markers(tmp_path):
    """Each start is a fresh attempt: an earlier attempt's readiness, stop and
    completion markers neither mark it ready nor stop or certify it."""
    allowed = sorted(os.sched_getaffinity(0))
    run_id = str(uuid4())
    root = tmp_path / "kvp"
    stale = root / "attempt-0-1"
    stale.mkdir(parents=True)
    (stale / "stop").touch()
    (stale / "children.json").write_text(
        json.dumps({"run_id": run_id, "endpoint": "d", "children_closed": True})
    )
    (root / "ready").write_text(str(stale))
    owner, _, endpoint_dir = start_recorded_endpoint(
        tmp_path, allowed[0], run_id=run_id
    )
    sidecar = _sidecar(tmp_path, run_id, allowed[-1])
    try:

        def fresh_ready():
            marker = root / "ready"
            return marker.exists() and marker.read_text() != str(stale)

        wait_for(fresh_ready, timeout=30)
        attempt = Path((root / "ready").read_text())
        assert attempt != stale and not (attempt / "stop").exists()
        time.sleep(0.5)
        assert sidecar.poll() is None  # the stale stop file stopped nothing
        (endpoint_dir / "close").touch()
        wait_for(lambda: (endpoint_dir / "closed").exists(), timeout=20)
        (endpoint_dir / "exit").touch()
        owner.wait(10)
        sidecar.send_signal(signal.SIGTERM)
        sidecar.communicate(timeout=40)
        assert _check(attempt, endpoint_dir / "profiles")["complete"]
    finally:
        if sidecar.poll() is None:
            sidecar.kill()
            sidecar.wait()
        (endpoint_dir / "exit").touch()
        owner.wait(10)


def test_a_sidecar_without_its_recorders_times_out_without_an_observer(tmp_path):
    allowed = sorted(os.sched_getaffinity(0))
    (tmp_path / "endpoint" / "profiles").mkdir(parents=True)
    sidecar = _sidecar(tmp_path, str(uuid4()), allowed[-1], startup_timeout_s=1)
    sidecar.communicate(timeout=20)
    assert sidecar.returncode == 1
    (attempt,) = (tmp_path / "kvp").glob("attempt-*")
    assert "timed out waiting for the expected recorders" in (
        attempt / "sidecar.log").read_text()  # fmt: skip
    assert not (attempt / "config.json").exists()


def test_a_sidecar_s_failed_observer_fails_the_sidecar(tmp_path):
    """An observer that refuses to start (here, the wrong endpoint side) makes
    the sidecar exit nonzero, with the exit recorded."""
    allowed = sorted(os.sched_getaffinity(0))
    run_id = str(uuid4())
    owner, _, endpoint_dir = start_recorded_endpoint(
        tmp_path, allowed[0], run_id=run_id
    )
    try:
        sidecar = _sidecar(tmp_path, run_id, allowed[-1], role="prefill")
        sidecar.communicate(timeout=30)
        assert sidecar.returncode == 1
        (attempt,) = (tmp_path / "kvp").glob("attempt-*")
        record = json.loads((attempt / "exit.json").read_text())
        assert record["exit_code"] not in (0, None)
        assert "exited before it was ready" in (attempt / "sidecar.log").read_text()
    finally:
        (endpoint_dir / "exit").touch()
        owner.wait(10)


def test_a_stalled_observer_is_bounded_by_the_startup_deadline(tmp_path):
    """One startup deadline covers the recorders and the observer's
    readiness: an observer that never becomes ready is killed at it."""
    from vllm.kv_profiling.endpoint_probes.sidecar import Sidecar

    allowed = sorted(os.sched_getaffinity(0))
    run_id = str(uuid4())
    owner, _, endpoint_dir = start_recorded_endpoint(
        tmp_path, allowed[0], run_id=run_id
    )
    stalled = tmp_path / "stalled.py"
    stalled.write_text("import time\ntime.sleep(600)\n")
    probes = ProbeConfig(interfaces=["lo"], required=["netdev"])
    settings = {
        "probes": asdict(probes), "role": "decode",
        "expected": {"frontend": 1, "scheduler": 1, "worker": 1},
        "profiles_dir": str(endpoint_dir / "profiles"),
        "observer_root": str(tmp_path / "kvp"),
        "irq_policy": "virtual_not_applicable", "cpus": [allowed[-1]],
        "startup_timeout_s": 2,
    }  # fmt: skip
    try:
        started = time.monotonic()
        code = Sidecar(settings, run_id=run_id, endpoint="d",
                       launcher=str(stalled)).run()  # fmt: skip
        assert code == 1 and time.monotonic() - started < 10
        (attempt,) = (tmp_path / "kvp").glob("attempt-*")
        record = json.loads((attempt / "exit.json").read_text())
        assert record["timed_out"] is True
        assert not (tmp_path / "kvp" / "ready").exists()
    finally:
        (endpoint_dir / "exit").touch()
        owner.wait(10)


def test_a_restarted_server_s_new_generation_is_observed_the_old_one_kept(tmp_path):
    """Each serving start records into its own generation directory: after a
    restart the sidecar observes the generation whose processes are alive and
    leaves the earlier one untouched."""
    allowed = sorted(os.sched_getaffinity(0))
    run_id = str(uuid4())
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    first, _, first_dir = start_recorded_endpoint(
        tmp_path / "one", allowed[0], run_id=run_id
    )
    (first_dir / "close").touch()
    wait_for(lambda: (first_dir / "closed").exists(), timeout=20)
    (first_dir / "exit").touch()
    first.wait(10)
    second, _, second_dir = start_recorded_endpoint(
        tmp_path / "two", allowed[0], run_id=run_id
    )
    generations = tmp_path / "endpoint" / "profiles"
    generations.mkdir(parents=True)
    (generations / "gen-1").symlink_to(first_dir / "profiles")
    (generations / "gen-2").symlink_to(second_dir / "profiles")
    before = sorted(p.name for p in (first_dir / "profiles" / run_id).iterdir())
    sidecar = _sidecar(tmp_path, run_id, allowed[-1])
    try:
        wait_for(lambda: (tmp_path / "kvp" / "ready").exists(), timeout=30)
        attempt = Path((tmp_path / "kvp" / "ready").read_text())
        config = json.loads((attempt / "config.json").read_text())
        assert config["input_dir"] == str(generations / "gen-2")
        (second_dir / "close").touch()
        wait_for(lambda: (second_dir / "closed").exists(), timeout=20)
        (second_dir / "exit").touch()
        second.wait(10)
        sidecar.communicate(timeout=40)
        assert sidecar.returncode == 0
        assert _check(attempt, generations / "gen-2")["complete"]
        after = sorted(p.name for p in (first_dir / "profiles" / run_id).iterdir())
        assert after == before  # the earlier generation is kept as it was
    finally:
        if sidecar.poll() is None:
            sidecar.kill()
            sidecar.wait()
        for directory, process in ((second_dir, second),):
            if process.poll() is None:
                (directory / "exit").touch()
                process.wait(10)


def _bare_sidecar(tmp_path, launcher, **settings):
    """A Sidecar driven directly, with a stand-in observer process."""
    from vllm.kv_profiling.endpoint_probes.sidecar import Sidecar

    probes = ProbeConfig(interfaces=["lo"], required=["netdev"])
    values = {
        "probes": asdict(probes), "role": "decode",
        "expected": {"frontend": 1, "scheduler": 1, "worker": 1},
        "profiles_dir": str(tmp_path / "profiles"),
        "observer_root": str(tmp_path / "kvp"),
        "irq_policy": "virtual_not_applicable", "cpus": [0], **settings,
    }  # fmt: skip
    sidecar = Sidecar(values, run_id=str(uuid4()), endpoint="d", launcher=launcher)
    sidecar.attempt = tmp_path / "attempt"
    sidecar.attempt.mkdir()
    sidecar.generation = tmp_path / "profiles"
    return sidecar


def _gone_process():
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    identity = process_identity(done.pid)
    done.wait()
    return {"pid": identity["pid"], "start_ticks": identity["start_ticks"]}


@pytest.mark.parametrize("case", ["all_gone", "owner_alive", "status_unreadable"])
def test_draining_needs_every_serving_process_confirmed_gone(
    tmp_path, monkeypatch, case
):
    """The recorders' processes and their owner (with data parallelism, the API
    servers' parent) must all be confirmed gone; one alive, or one whose status
    cannot be read, leaves the children not closed."""
    from vllm.kv_profiling.endpoint_probes import sidecar as sidecar_module

    # the L1 captures closed: only the processes' exit decides here
    monkeypatch.setattr(sidecar_module, "core_completion",
                        lambda *args: {"all_closed": True})  # fmt: skip
    sidecar = _bare_sidecar(tmp_path, sys.executable, stop_grace_s=0.8)
    sidecar.observer = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.3)"])  # fmt: skip
    workers = [_gone_process() for _ in range(3)]
    me = process_identity(os.getpid())
    owner = {"pid": me["pid"], "start_ticks": me["start_ticks"]}
    if case == "all_gone":
        owner = _gone_process()
    if case == "status_unreadable":
        owner = _gone_process()
        real = sidecar_module.Path.read_text
        denied = f"/proc/{owner['pid']}/stat"

        def read_text(path, *args, **kwargs):
            if str(path) == denied:
                raise PermissionError(13, "Permission denied", denied)
            return real(path, *args, **kwargs)

        monkeypatch.setattr(sidecar_module.Path, "read_text", read_text)
    sidecar._stop(workers, owner)
    children = json.loads((sidecar.attempt / "children.json").read_text())
    assert children["children_closed"] is (case == "all_gone")


def test_a_stalled_observer_stays_within_the_shutdown_budget(tmp_path):
    """Stopping a stalled observer fits the one deadline (draining keeps time
    for it) and is a timeout."""
    sidecar = _bare_sidecar(tmp_path, sys.executable, stop_grace_s=0.8)
    sidecar.observer = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"])  # fmt: skip
    started = time.monotonic()
    timed_out = sidecar._stop([_gone_process()])
    assert timed_out is True and time.monotonic() - started < 1.5


def test_sigterm_during_startup_is_a_cancellation_not_a_timeout(tmp_path):
    """Termination while waiting for readiness cancels the attempt: the
    observer is stopped, and the exit record is not a timeout."""
    allowed = sorted(os.sched_getaffinity(0))
    run_id = str(uuid4())
    owner, _, endpoint_dir = start_recorded_endpoint(
        tmp_path, allowed[0], run_id=run_id
    )
    stalled = tmp_path / "stalled.py"
    stalled.write_text("import time\ntime.sleep(600)\n")
    settings = tmp_path / "settings.json"
    probes = ProbeConfig(interfaces=["lo"], required=["netdev"])
    settings.write_text(json.dumps({
        "probes": asdict(probes), "role": "decode",
        "expected": {"frontend": 1, "scheduler": 1, "worker": 1},
        "profiles_dir": str(endpoint_dir / "profiles"),
        "observer_root": str(tmp_path / "kvp"),
        "irq_policy": "virtual_not_applicable", "cpus": [allowed[-1]],
        "startup_timeout_s": 30}))  # fmt: skip
    code = (
        "import sys; from vllm.kv_profiling.endpoint_probes.sidecar import main;"
        f"sys.exit(main({str(settings)!r}, {run_id!r}, 'd', {str(stalled)!r}))"
    )
    sidecar = subprocess.Popen([sys.executable, "-c", code])
    try:
        wait_for(lambda: list((tmp_path / "kvp").glob("attempt-*/config.json")),
                 timeout=30)  # fmt: skip
        time.sleep(0.3)
        started = time.monotonic()
        sidecar.send_signal(signal.SIGTERM)
        sidecar.wait(20)
        assert sidecar.returncode == 1 and time.monotonic() - started < 10
        (attempt,) = (tmp_path / "kvp").glob("attempt-*")
        record = json.loads((attempt / "exit.json").read_text())
        assert record["timed_out"] is False and record["killed"] == "signal"
        assert "terminated before the observer was ready" in (
            attempt / "sidecar.log").read_text()  # fmt: skip
    finally:
        if sidecar.poll() is None:
            sidecar.kill()
            sidecar.wait()
        (endpoint_dir / "exit").touch()
        owner.wait(10)


def test_the_closure_scan_never_overruns_the_shutdown_deadline(tmp_path, monkeypatch):
    """Full L1 verification runs under the drain's deadline: a scan still
    parsing when it expires leaves the children not closed, on time."""
    from vllm.kv_profiling.endpoint_probes import sidecar as sidecar_module

    def slow(*args):
        time.sleep(5)  # a large trace being parsed
        return {"all_closed": True}

    monkeypatch.setattr(sidecar_module, "core_completion", slow)
    sidecar = _bare_sidecar(tmp_path, sys.executable, stop_grace_s=0.8)
    sidecar.observer = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"])  # fmt: skip
    started = time.monotonic()
    sidecar._stop([_gone_process()], _gone_process())
    assert time.monotonic() - started < 1.5
    children = json.loads((sidecar.attempt / "children.json").read_text())
    assert children["children_closed"] is False


def test_readiness_waits_for_every_recorder_confirmed_alive(tmp_path):
    """Discovery needs each recorder's process confirmed running: while its
    status cannot be read the sidecar is not ready, and once it can, it is."""
    allowed = sorted(os.sched_getaffinity(0))
    run_id = str(uuid4())
    owner, _, endpoint_dir = start_recorded_endpoint(
        tmp_path, allowed[0], run_id=run_id
    )
    deny = tmp_path / "deny"
    deny.touch()
    probes = ProbeConfig(interval_s=0.1, slow_interval_s=0.1, interfaces=["lo"],
                         required=["netdev"])  # fmt: skip
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "probes": asdict(probes), "role": "decode",
        "expected": {"frontend": 1, "scheduler": 1, "worker": 1},
        "profiles_dir": str(endpoint_dir / "profiles"),
        "observer_root": str(tmp_path / "kvp"),
        "irq_policy": "virtual_not_applicable", "cpus": [allowed[-1]],
        "startup_timeout_s": 60}))  # fmt: skip
    stat, entry = f"/proc/{owner.pid}/stat", str(Path(probe.__file__))
    code = "\n".join([
        "import sys",
        "from vllm.kv_profiling.endpoint_probes import sidecar",
        "real = sidecar.Path.read_text",
        "def read_text(path, *args, **kwargs):",
        f"    if str(path) == {stat!r} and sidecar.Path({str(deny)!r}).exists():",
        "        raise PermissionError(13, 'Permission denied', str(path))",
        "    return real(path, *args, **kwargs)",
        "sidecar.Path.read_text = read_text",
        f"sys.exit(sidecar.main({str(settings)!r}, {run_id!r}, 'd', {entry!r}))",
    ])  # fmt: skip
    process = subprocess.Popen([sys.executable, "-c", code])
    ready = tmp_path / "kvp" / "ready"
    try:
        time.sleep(2.5)
        assert process.poll() is None and not ready.exists()
        assert not list((tmp_path / "kvp").glob("attempt-*/config.json"))
        deny.unlink()  # access returns
        wait_for(ready.exists, timeout=30)
    finally:
        process.send_signal(signal.SIGTERM)
        (endpoint_dir / "close").touch()
        (endpoint_dir / "exit").touch()
        process.wait(90)
        owner.wait(10)


def test_a_scan_finishing_in_the_last_poll_interval_is_used(tmp_path, monkeypatch):
    """The scan's completion wakes the drain: a closed result delivered just
    before the cutoff (inside what used to be the final sleep) counts."""
    from vllm.kv_profiling.endpoint_probes import sidecar as sidecar_module

    sidecar = _bare_sidecar(tmp_path, sys.executable, stop_grace_s=2.0)
    cutoff_s = 2.0 - min(sidecar_module.FINALIZE_MAX_S,
                         2.0 * sidecar_module.FINALIZE_SHARE)  # fmt: skip

    def scan(*args):
        time.sleep(cutoff_s - 0.08)  # done 80 ms before the cutoff
        return {"all_closed": True}

    monkeypatch.setattr(sidecar_module, "core_completion", scan)
    sidecar.observer = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"])  # fmt: skip
    sidecar._stop([_gone_process()], _gone_process())
    children = json.loads((sidecar.attempt / "children.json").read_text())
    assert children["children_closed"] is True


def test_an_observer_samples_the_endpoint_s_own_sockets_by_identity(tmp_path):
    """TCP sampling with no address anywhere: the endpoint's process holds one
    connection to the test's listener; the observer discovers and admits it by
    cookie, samples it labelled with its admission (whose record names the
    owners), never the listener's side (which only the test holds). As the pod
    sidecar does, the observer is stopped only after the endpoint has exited:
    its owner is retired as exited, and the capture with a required TCP source
    checks complete."""
    allowed = sorted(os.sched_getaffinity(0))
    assert len(allowed) >= 2, "probe tests require separate owner/observer CPUs"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        port = listener.getsockname()[1]
        owner, inventory, endpoint_dir = start_recorded_endpoint(
            tmp_path, allowed[0], connect=port
        )
        accepted, _ = listener.accept()
        run_id = inventory[0]["run_id"]
        core_dir = endpoint_dir / "profiles"
        settings = ProbeConfig(
            interval_s=0.1, slow_interval_s=0.1, interfaces=["lo"],
            tcp={"owned_sockets": True}, discovery_interval_s=0.05,
            required=["netdev", "tcp"],
        )  # fmt: skip
        path = _observer_config(tmp_path, run_id, core_dir, allowed[-1],
                                probes=asdict(settings),
                                owner_exit_grace_s=30)  # fmt: skip
        observer_dir = path.parent
        entry = str(Path(probe.__file__))
        process = subprocess.Popen(
            [sys.executable, entry, "--config", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )  # fmt: skip
        try:
            glob = f"captures/{run_id}/*/manifest.json"

            def ready():
                paths = observer_dir.glob(glob)
                return any(json.loads(p.read_text()).get("ready") for p in paths)

            wait_for(ready, timeout=20)
            time.sleep(0.5)  # a few discovery and sampling ticks
            (endpoint_dir / "close").touch()
            wait_for(lambda: (endpoint_dir / "closed").exists(), timeout=20)
            (endpoint_dir / "exit").touch()
            owner.wait(timeout=20)
            time.sleep(0.3)  # discovery ticks after the exit
            closed = {"run_id": run_id, "endpoint": "d", "children_closed": True}
            (observer_dir / "children.json").write_text(json.dumps(closed))
            (observer_dir / "stop").touch()
            _, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stderr
        finally:
            accepted.close()
            (endpoint_dir / "exit").touch()
            owner.wait(timeout=20)
            if process.poll() is None:
                process.kill()
    manifest = next(observer_dir.glob(glob))
    rows = [json.loads(line) for line in manifest.with_name("events.jsonl").open()]
    admitted = [r for r in rows if r.get("scope") == "socket_ownership"
                and r["observation"]["change"] == "admitted"]  # fmt: skip
    sockets_ = [r for r in rows if r.get("scope") == "socket"]
    assert len(admitted) == 1, admitted
    socket_record = admitted[0]["observation"]["socket"]
    assert socket_record["peer_port"] == port
    assert admitted[0]["observation"]["owners"][0]["pid"] == owner.pid
    removed = [r["observation"] for r in rows if r.get("scope") == "socket_ownership"
               and r["observation"]["change"] == "removed"]  # fmt: skip
    # its socket closes during the exit, or its owner is found gone: either
    # retires it, and neither fails the source
    assert [r["reason"] in ("gone", "owner exited") for r in removed] == [True]
    failures = [r for r in rows if r.get("scope") == "source"]
    assert failures == [], failures
    assert sockets_ and all(
        r["transport"]["tcp"]["selection"]["cookie"] == socket_record["cookie"]
        and r["transport"]["tcp"]["selection"]["admission"]
        == socket_record["admission"]
        and r["transport"]["tcp"]["peer_port"] == port
        for r in sockets_
    )
    record = {"run_id": run_id, "endpoint": "d", "process_id": rows[0]["process_id"],
              "exit_code": 0, "launcher": "test-process"}  # fmt: skip
    subprocess.run([sys.executable, entry, "--record-exit", str(observer_dir)],
                   input=json.dumps(record), text=True, check=True)  # fmt: skip
    checked = subprocess.run(
        [sys.executable, entry, "--check-capture", str(observer_dir),
         "--core-dir", str(core_dir)],
        capture_output=True, text=True, timeout=30,
    )  # fmt: skip
    summary = json.loads(checked.stdout)
    assert checked.returncode == 0 and summary["complete"], summary


class _Rows:
    def __init__(self):
        self.rows = []

    def put(self, row):
        self.rows.append(row)
        return True


def test_each_tcp_observation_is_recorded_once_at_its_acquisition_time():
    """A socket's TCP_INFO is when the kernel dumped it: the row spans the
    dump's own interval, and sampling the same scan again records nothing."""
    from vllm.kv_profiling.endpoint_probes.sockets import Discovery

    with socket.socket() as listener, socket.socket() as client:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        client.connect(listener.getsockname())
        server, _ = listener.accept()
        with server:
            scan = Discovery([_me()]).scan()
    sink = _Rows()
    observer = _sample_once(sink, scan)
    first = [r for r in sink.rows if r["scope"] == "socket"]
    assert first and all(
        (r["sample_begin_ns"], r["sample_end_ns"])
        == (scan.dump_started_ns, scan.dump_finished_ns)
        for r in first
    )
    assert scan.dump_started_ns <= scan.dump_finished_ns <= scan.finished_ns
    count = len(sink.rows)
    observer.sample_tcp()  # the same scan: already recorded
    assert len(sink.rows) == count
    selection = first[0]["transport"]["tcp"]["selection"]
    # the owners are in the admission's record, not repeated in every sample
    assert "owners" not in first[0]["transport"]["tcp"]
    assert selection["by"] == "owned_socket" and selection["admission"] >= 1
    assert selection["owner_count"] == 1


def test_one_process_hosting_several_roles_is_read_once_with_all_of_them():
    """Recorders of several roles in one process (a frontend among them): the
    process is read once per pass, its sockets carry every role's identity,
    and they say a frontend is colocated (its sockets cannot be told apart)."""
    from vllm.kv_profiling.endpoint_probes import sockets

    reads = []
    real = sockets._inodes

    def counted(pid, budget):
        reads.append(pid)
        return real(pid, budget)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        owners = [_me(role, engine_id="d", rank=0)
                  for role in ("frontend", "scheduler", "worker")]  # fmt: skip
        with mock.patch.object(sockets, "_inodes", counted):
            scan = sockets.Discovery(owners, sockets.Limits(processes=0)).scan()
        record = _owned_by_inode(scan)[os.fstat(listener.fileno()).st_ino]
    assert scan.complete and reads.count(os.getpid()) == 2  # both reads, once each
    assert sorted(o["role"] for o in record["owners"]) == [
        "frontend", "scheduler", "worker"]  # fmt: skip
    assert record["frontend_colocated"] and record["owner_pids"] == [os.getpid()]


def test_proc_is_streamed_under_the_process_bound(monkeypatch):
    from vllm.kv_profiling.endpoint_probes import sockets

    consumed = []
    real = os.scandir

    class Counting:
        def __init__(self, entries):
            self.entries = entries

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.entries.close()

        def __iter__(self):
            for entry in self.entries:
                consumed.append(entry.name)
                yield entry

    def scandir(path):
        entries = real(path)
        return Counting(entries) if str(path) == "/proc" else entries

    monkeypatch.setattr(sockets.os, "scandir", scandir)
    scan = sockets.Discovery([_me()], sockets.Limits(processes=1)).scan()
    assert scan.complete and scan.holders_truncated == "discovery bound"
    assert len(consumed) <= 2, len(consumed)


def test_a_scan_whose_matching_outlasts_its_deadline_admits_nothing(monkeypatch):
    from vllm.kv_profiling.endpoint_probes import sockets

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        real = sockets.cookie_of

        def slow(row):
            time.sleep(0.001)
            return real(row)

        rows = sockets._dump(sockets.Budget(sockets.Limits()), [])
        monkeypatch.setattr(sockets, "_dump", lambda budget, unavailable: rows * 100)
        monkeypatch.setattr(sockets, "cookie_of", slow)
        scan = sockets.Discovery([_me()], sockets.Limits(deadline_s=0.05)).scan()
    assert not scan.complete and "deadline" in scan.reason and scan.owned == {}


def test_discovery_periods_are_counted_against_their_scheduled_starts():
    """0.25 s periods, 0.40 s of work per scan: starts slip, and each whole
    period skipped is counted missed."""
    from vllm.kv_profiling.endpoint_probes.sockets import Cadence

    cadence, now, starts, missed = Cadence(0.25, 0.0), 0.0, [], []
    while now < 2.0:
        due, skipped = cadence.due(now)
        if due:
            starts.append(round(now, 2))
            missed.append(skipped)
            now += 0.40  # the scan's work
        else:
            now = cadence.next
    assert starts[:3] == [0.0, 0.4, 0.8] and sum(missed) == cadence.missed > 0


def test_an_owner_in_another_namespace_or_unreadable_affects_only_itself():
    """A frozen namespace that is not the live one makes that owner unreadable;
    one owner whose namespace links are refused leaves a socket it shares with
    a readable owner owned by that one."""
    from vllm.kv_profiling.endpoint_probes import sockets
    from vllm.kv_profiling.endpoint_probes.identity import NamespacesUnreadable

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        moved = {**_me(), "net_namespace": "net:[1]"}
        scan = sockets.Discovery([moved]).scan()
        assert scan.complete and scan.owned == {}
        assert os.getpid() in scan.mismatched and not scan.unreadable
        sharer = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            pass_fds=(listener.fileno(),),
        )
        try:
            real = sockets.process_identity

            def refused(pid):
                if pid == sharer.pid:
                    raise NamespacesUnreadable("namespace links denied")
                return real(pid)

            owners = [_me("worker"), {"pid": sharer.pid, "role": "scheduler",
                                      "start_ticks": 1}]  # fmt: skip
            with mock.patch.object(sockets, "process_identity", refused):
                scan = sockets.Discovery(owners).scan()
            record = _owned_by_inode(scan)[os.fstat(listener.fileno()).st_ino]
        finally:
            sharer.kill()
            sharer.wait()
    assert scan.complete and record["owner_pids"] == [os.getpid()]
    assert "NamespacesUnreadable" in scan.unreadable[sharer.pid]


def test_ipv4_sockets_stay_verified_when_the_kernel_has_no_ipv6():
    from vllm.kv_profiling.endpoint_probes import sockets

    real = sockets.query

    def no_ipv6(protocol, kind, payload, **kwargs):
        if payload[0] == socket.AF_INET6:
            raise OSError(errno.EAFNOSUPPORT, "Address family not supported")
        return real(protocol, kind, payload, **kwargs)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with mock.patch.object(sockets, "query", no_ipv6):
            scan = sockets.Discovery([_me()]).scan()
        inode = os.fstat(listener.fileno()).st_ino
    assert scan.complete and scan.families_unavailable == ["ipv6"]
    assert inode in _owned_by_inode(scan)


def test_identities_under_one_pid_are_verified_each_whatever_their_order():
    """A live frontend identity and a stale worker identity (an earlier
    generation) under one PID: the PID is no owner in either order."""
    from vllm.kv_profiling.endpoint_probes.sockets import Discovery

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        live = _me("frontend")
        stale = {**_me("worker"), "start_ticks": live["start_ticks"] - 1}
        for owners in ([live, stale], [stale, live]):
            scan = Discovery(owners).scan()
            assert scan.complete and scan.owned == {}, owners
            assert os.getpid() in scan.exited


def test_building_the_owned_records_counts_against_the_deadline(monkeypatch):
    from vllm.kv_profiling.endpoint_probes import sockets

    real = sockets._record

    def slow(*args):
        time.sleep(0.06)
        return real(*args)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        monkeypatch.setattr(sockets, "_record", slow)
        scan = sockets.Discovery([_me()], sockets.Limits(deadline_s=0.05)).scan()
    assert not scan.complete and "deadline" in scan.reason and scan.owned == {}


def test_an_owner_record_too_long_for_one_row_is_split_and_recovered():
    """128 recorder identities share one socket: its admission splits over
    rows within the limit, the join recovers all of them by admission, the
    TCP row stays small, and every split row keeps its explicit end time."""
    from vllm.kv_profiling.endpoint_probes.runner import ROW_BUDGET, Observer
    from vllm.kv_profiling.endpoint_probes.sockets import Tracker
    from vllm.kv_profiling.probe_join import _admitted_owners, _owners

    scan = _scan({17: list(range(1000, 1128))})
    for record in scan.owned.values():
        record["owners"] = [
            {**owner, "engine_id": "e" * 40, "rank": index,
             "net_namespace": "net:[4026531840]", "pid_namespace": "pid:[4026531836]"}
            for index, owner in enumerate(record["owners"])
        ]  # fmt: skip
    sink = _Rows()
    observer = Observer.__new__(Observer)
    observer.writer = sink
    tracker = Tracker()
    for change in tracker.apply(scan):
        observer.emit("socket_ownership", "17", 5, change, kind="probe_status", end=9)
    rows = [r for r in sink.rows if r["scope"] == "socket_ownership"]
    assert len(rows) > 1 and all(len(json.dumps(r)) <= ROW_BUDGET for r in rows)
    assert {(r["sample_begin_ns"], r["sample_end_ns"]) for r in rows} == {(5, 9)}
    owners = _admitted_owners(rows)[tracker.active[17]]["owners"]
    assert [o["pid"] for o in owners] == list(range(1000, 1128))
    flow = {"selection": {"cookie": 17, "admission": tracker.active[17]}}
    assert len(_owners(flow, _admitted_owners(rows))) == 128
    # the admission's record is of another cookie: no owners
    assert _owners({"selection": {"cookie": 18, "admission": tracker.active[17]}},
                   _admitted_owners(rows)) == []  # fmt: skip
    # an incomplete record resolves to nothing, never to part of its owners
    assert _admitted_owners(rows[:1]) == {}
    # a long status keeps the dump's interval in every part
    sink.rows.clear()
    gone = {"disappeared": [str(n) * 8 for n in range(1024)]}
    observer.emit("tcp_poll", "d", 100, gone, kind="probe_status", end=120)
    assert len(sink.rows) > 1
    assert {(r["sample_begin_ns"], r["sample_end_ns"]) for r in sink.rows} == {
        (100, 120)
    }


@pytest.mark.parametrize("read", [1, 2])
def test_an_owner_exiting_during_either_fd_read_is_retired_not_failed(read):
    """A real child exits cleanly right before its first or second fd read: the
    read's ENOENT is confirmed as an exit, the owner is retired, and a
    required TCP source records no failure."""
    from vllm.kv_profiling.endpoint_probes import sockets

    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    )
    real, calls = sockets._inodes, []

    def racing(pid, budget):
        if pid == child.pid:
            calls.append(pid)
            if len(calls) == read:
                child.stdin.close()  # a clean exit
                child.wait()
        return real(pid, budget)

    observer = _sampler(_Rows(), required=["tcp"])
    observer.discovery = sockets.Discovery([_me("worker", child.pid)])
    observer.last_scan = None
    try:
        with mock.patch.object(sockets, "_inodes", racing):
            observer.discover()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    scan = observer.last_scan
    assert scan.complete and child.pid in scan.exited, scan.exited
    assert not scan.unreadable and child.pid in observer.discovery.retired
    assert observer.failed_required == 0


def test_an_owner_in_other_namespaces_for_now_is_suspended_not_retired():
    """Same start ticks, another network namespace: it holds nothing (a
    coverage gap, failing a TCP source) but is read again, and owns its
    sockets once its namespaces match."""
    from vllm.kv_profiling.endpoint_probes import sockets

    real = sockets.process_identity

    def moved(pid):
        return {**real(pid), "net_namespace": "net:[1]"}

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        discovery = sockets.Discovery([_me()])
        with mock.patch.object(sockets, "process_identity", moved):
            first = discovery.scan()
        second = discovery.scan()
        inode = os.fstat(listener.fileno()).st_ino
    assert first.complete and first.owned == {} and os.getpid() in first.mismatched
    assert os.getpid() in discovery.identities and not discovery.retired
    assert inode in _owned_by_inode(second)


def test_a_generation_change_needs_one_verified_owner_across_both_reads():
    """Two frozen worker generations under one PID. If the live one changes
    during a scan, no identity is verified in both reads, and nothing is
    admitted (never with an empty owner list). Between two scans, the new
    generation is another owner: a new admission naming its engine."""
    from vllm.kv_profiling.endpoint_probes import sockets

    real = sockets.process_identity
    mine = real(os.getpid())
    old = {**_me("worker", engine_id="old", rank=0), "start_ticks": mine["start_ticks"]}
    new = {
        **_me("worker", engine_id="new", rank=0),
        "start_ticks": mine["start_ticks"] + 1,
    }
    calls = []

    def changing(switch_after):
        def identity(pid):
            calls.append(pid)
            live = real(pid)
            if len(calls) > switch_after:
                live = {**live, "start_ticks": new["start_ticks"]}
            return live

        return identity

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        inode = os.fstat(listener.fileno()).st_ino
        discovery = sockets.Discovery([old, new])
        with mock.patch.object(sockets, "process_identity", changing(2)):
            during = discovery.scan()  # first read: old; second read: new
        tracker = sockets.Tracker()
        calls.clear()
        with mock.patch.object(sockets, "process_identity", changing(10**6)):
            before = discovery.scan()  # old throughout
        tracker.apply(before)
        first_admission = tracker.active[_owned_by_inode(before)[inode]["cookie"]]
        calls.clear()
        with mock.patch.object(sockets, "process_identity", changing(0)):
            after = discovery.scan()  # new throughout
        changes = tracker.apply(after)
    assert during.complete and inode not in _owned_by_inode(during)
    assert os.getpid() in during.mismatched
    assert all(record["owners"] for record in during.owned.values())
    assert [o["engine_id"] for o in _owned_by_inode(before)[inode]["owners"]] == ["old"]
    record = _owned_by_inode(after)[inode]
    assert [o["engine_id"] for o in record["owners"]] == ["new"]
    admitted = [c for c in changes if c["change"] == "admitted"
                and c["cookie"] == record["cookie"]]  # fmt: skip
    assert admitted and admitted[0]["supersedes"] == first_admission
    assert [o["engine_id"] for o in admitted[0]["owners"]] == ["new"]


_FAKE_COLLECTOR = r"""
import json, sys, time
# a collector speaking the real protocol; argv: the acknowledgements to lose
# (sequence numbers, applied anyway) and the deletes to fail (sequence numbers,
# or "all")
lose = {int(x) for x in sys.argv[1].split(",") if x}
fail_all = sys.argv[2] == "all"
fail = set() if fail_all else {int(x) for x in sys.argv[2].split(",") if x}
owned = {}
counters = {42: 3}  # rejections so far, harvested by an admit or a delete
def say(row):
    print(json.dumps(row), flush=True)
say({"type": "ready", "available": True, "exact": False, "coverage": "partial"})
for line in sys.stdin:
    words = line.split()
    if words == ["stop"]:
        say({"type": "loss", "scope": "filter_exclusion",
             "counter": "excluded_unowned", "value": 12})
        now = time.monotonic_ns()
        say({"type": "rejected", "cookie": 42, "count": 5, "first_ns": now,
             "last_ns": now, "consistent": True})
        say({"type": "rejected", "cookie": 77, "count": 3, "first_ns": 1, "last_ns": 2,
             "consistent": True})
        say({"type": "end", "complete": False, "coverage": "partial", "output_drops": 0,
             "malformed": 0, "rejected_complete": True, "failed": False})
        sys.exit(2)
    op, seq, cookie = words[0], int(words[1]), int(words[2])
    before = time.monotonic_ns()
    if op == "A":
        owned[cookie] = int(words[3])
        row = {"type": "ack", "op": "A", "seq": seq, "cookie": cookie,
               "admission": int(words[3]), "ok": True, "errno": 0, "before_ns": before,
               "applied_ns": time.monotonic_ns(), "rejected": counters.pop(cookie, 0),
               "rejected_first_ns": 1, "rejected_errno": 0}
    elif op == "D":
        admission = int(words[3])
        stale = owned.get(cookie) != admission
        failed = fail_all or seq in fail
        if not stale and not failed:
            del owned[cookie]
        row = {"type": "ack", "op": "D", "seq": seq, "cookie": cookie,
               "admission": admission, "ok": not failed, "errno": 5 if failed else 0,
               "stale": stale, "before_ns": before, "applied_ns": time.monotonic_ns(),
               "rejected": 0 if stale else counters.pop(cookie, 0),
               "rejected_first_ns": 0, "rejected_errno": 0, "harvested": not stale}
    else:
        row = {"type": "ack", "op": "R", "seq": seq, "cookie": cookie, "ok": True,
               "errno": 0, "admission": owned.get(cookie),
               "applied_ns": time.monotonic_ns()}
    if seq not in lose:
        say(row)
"""


def _fake_channel(tmp_path, lose=(), fail=()):
    from vllm.kv_profiling.endpoint_probes.ebpf import Channel, Settings

    failing = fail if isinstance(fail, str) else ",".join(map(str, fail))
    child = subprocess.Popen(
        [sys.executable, "-c", _FAKE_COLLECTOR, ",".join(map(str, lose)), failing],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )  # fmt: skip
    channel = Channel(child, tmp_path, Settings(enabled=True, privileged_opt_in=True,
                                                mode="tcp_tracepoints"))  # fmt: skip
    _settle(channel, until=lambda: channel.status.get("available") is True)
    channel.monotonic_offset = 0  # the fake's clock is the observer's
    return channel


def _settle(channel, until=None, timeout=5):
    deadline = time.monotonic() + timeout
    while (channel.outbox or channel.pending or (until and not until())) and (
        time.monotonic() < deadline
    ):
        channel.poll(None, 0.02)


def _event(admission, cookie=42):
    return {"cookie": cookie, "admission": admission}


def test_ebpf_a_delayed_event_stays_verified_and_a_replacement_ends_the_old(
    tmp_path,
):
    """An event of a removed admission, delivered late but timed inside its
    interval, is verified; one after it is not. Replacing a cookie's admission
    ends the old one; the old admission's late removal changes nothing."""
    channel = _fake_channel(tmp_path)
    try:
        channel.admit(42, 1)
        _settle(channel)
        inside = time.monotonic_ns()
        channel.admit(42, 2)  # a reversioned socket: replaced
        _settle(channel)
        between = time.monotonic_ns()
        channel.retire(42, 1)  # no longer the decision: nothing is sent
        assert not channel.outbox and not channel.pending
        assert channel.windows[2]["until"] is None
        assert channel.windows[1]["until"] is not None
        assert channel.verify(_event(1), inside)["verified"] is True
        assert channel.verify(_event(1), between)["reason"] == (
            "outside the owned interval"
        )
        assert channel.verify(_event(2), between)["verified"] is True
        channel.retire(42, 2)
        _settle(channel)
        after = time.monotonic_ns()
        assert channel.verify(_event(2), after)["verified"] is False
        assert channel.verify(_event(2, cookie=43), between)["verified"] is False
    finally:
        assert channel.stop(None)["clean"] is True


def test_ebpf_a_lost_acknowledgement_after_apply_is_unknown_until_read_back(
    tmp_path, monkeypatch
):
    """The kernel applied an admission whose acknowledgement never came: events
    in between are unverified (kernel state unknown); a readback establishes
    the admission, and later events are verified."""
    from vllm.kv_profiling.endpoint_probes import ebpf

    monkeypatch.setattr(ebpf, "ACK_TIMEOUT_NS", 200_000_000)
    channel = _fake_channel(tmp_path, lose=[1])
    try:
        channel.admit(42, 7)
        _settle(channel, until=lambda: 7 in channel.windows)
        settled = channel.unknown[42][-1][1]
        assert settled is not None and channel.ack_failures[0]["reason"] == (
            "no acknowledgement"
        )
        sent = channel.ack_failures[0]["sent_ns"]
        assert channel.windows[7]["since"] == sent
        assert channel.verify(_event(7), sent + 1)["reason"] == "kernel state unknown"
        assert channel.verify(_event(7), settled + 1)["verified"] is True
    finally:
        assert channel.stop(None)["clean"] is True


@pytest.mark.parametrize("lose", [(), (2,)])
def test_ebpf_a_retired_socket_is_never_reported_owned_and_its_entry_is_removed(
    tmp_path, monkeypatch, lose
):
    """The delete of a retired socket fails (its acknowledgement lost, too):
    an event after the retirement is unverified when recorded and stays so;
    the readback finds the stale entry and the delete is retried until the
    kernel matches discovery; an event from before the retirement stays
    verified, and the capture ends clean."""
    from vllm.kv_profiling.endpoint_probes import ebpf

    monkeypatch.setattr(ebpf, "ACK_TIMEOUT_NS", 100_000_000)
    channel = _fake_channel(tmp_path, lose=lose, fail=[2])
    try:
        channel.admit(42, 1)
        _settle(channel)
        owned = time.monotonic_ns()
        channel.retire(42, 1)
        retired = time.monotonic_ns()
        assert channel.verify(_event(1), retired)["verified"] is False
        _settle(channel, until=lambda: channel.kernel.get(42) is None)
        assert channel.kernel.get(42) is None and not channel.inflight
        reasons = [f["reason"] for f in channel.ack_failures]
        assert reasons == (["no acknowledgement"] if lose else ["delete failed"])
        assert channel.verify(_event(1), retired)["verified"] is False
        assert channel.verify(_event(1), time.monotonic_ns())["verified"] is False
        assert channel.verify(_event(1), owned)["verified"] is True
        assert channel.windows[1]["until"] is not None
    finally:
        result = channel.stop(None)
    assert result["clean"] is True and not any(result["ownership_unresolved"].values())


def test_ebpf_a_delete_that_never_succeeds_is_bounded_and_not_clean(tmp_path):
    channel = _fake_channel(tmp_path, fail="all")
    channel.admit(42, 1)
    _settle(channel)
    channel.retire(42, 1, time.monotonic_ns())
    _settle(channel)
    assert 42 in channel.unresolved
    assert channel.ack_failures[-1]["reason"] == "unresolved after retries"
    # bounded by event time: before the retirement it was owned
    assert channel.verify(_event(1), channel.decided[1]["until"] - 1)["verified"]
    sent = sum(1 for f in channel.ack_failures if f["reason"] == "delete failed")
    assert sent == 2  # delete, readback, delete, readback, then given up
    result = channel.stop(None)
    assert result["clean"] is False
    assert result["ownership_unresolved"]["given_up"] == 1


def test_ebpf_a_timely_acknowledgement_read_late_is_not_lost(tmp_path):
    """An acknowledgement readable before its deadline, polled only after it,
    is read before anything expires."""
    import select

    channel = _fake_channel(tmp_path)
    try:
        channel.admit(42, 1)
        assert select.select([channel.child.stdout], [], [], 1)[0]
        time.sleep(1.06)
        _settle(channel)
        assert not channel.ack_failures and not channel.unknown
        assert channel.kernel[42] == 1
    finally:
        assert channel.stop(None)["clean"] is True


def test_ebpf_ownership_commands_unsent_at_shutdown_are_not_clean(tmp_path):
    channel = _fake_channel(tmp_path)
    send = channel.send_command
    with mock.patch.object(
        channel, "send_command", lambda wire: wire == b"stop\n" and send(wire)
    ):
        for cookie in range(1, 17):
            channel.admit(cookie, cookie)
        assert len(channel.outbox) == 16
        result = channel.stop(None)
    assert result["clean"] is False
    assert result["ownership_unresolved"]["unsent"] == 16
    assert result["ownership_unresolved"]["cookies"] == 16


def test_ebpf_each_rejection_counts_once_with_the_rest_unattributed(tmp_path):
    """The admitted cookie's counter harvested at its admit (3) is before its
    admission; its terminal counter (5), all begun after its delete began, is
    after its retirement; another cookie's 3 are never owned; of the kernel's
    total (12), 1 is unattributed."""
    channel = _fake_channel(tmp_path)
    channel.admit(42, 7)
    _settle(channel)
    assert channel.harvests[42][0]["count"] == 3
    channel.retire(42, 7)
    _settle(channel)
    result = channel.stop(None)
    assert result["rejections"] == {
        "total": 12, "untracked": 0, "attributed": 11,
        "before_or_between_admissions": 3, "after_retirement": 5,
        "unresolved": 0, "never_owned": 3, "admitted_cookies_rejected": 1,
        "never_owned_cookies": 1, "unattributed": 1, "counters_complete": True,
        "reconciled": True,
    }  # fmt: skip


def _harvest_channel(tmp_path, harvests, final, retired=True, offset=0):
    """A channel given acknowledgements directly: admit at 10..11, delete at
    20..21 (each harvesting ``harvests[op]``), the terminal counter ``final``
    (count, first_ns)."""
    from vllm.kv_profiling.endpoint_probes.ebpf import Channel

    channel = Channel(None, tmp_path)
    channel.monotonic_offset = offset
    count, errno = harvests.get("A", (0, 0))
    channel.acknowledged({"seq": 1, "op": "A", "cookie": 42, "admission": 1,
                          "ok": True, "before_ns": 10, "applied_ns": 11,
                          "rejected": count, "rejected_first_ns": 0,
                          "rejected_errno": errno})  # fmt: skip
    if retired:
        count, errno = harvests.get("D", (0, 0))
        channel.acknowledged({"seq": 2, "op": "D", "cookie": 42, "admission": 1,
                              "ok": True, "stale": False, "before_ns": 20,
                              "applied_ns": 21, "rejected": count,
                              "rejected_first_ns": 0, "rejected_errno": errno,
                              "harvested": True})  # fmt: skip
    if final:
        consistent = final[2] if len(final) > 2 else True
        channel.rejected[42] = {"cookie": 42, "count": final[0],
                                "first_ns": final[1], "last_ns": 99,
                                "consistent": consistent}  # fmt: skip
    channel.losses["filter_exclusion", "excluded_unowned"] = 10
    return channel.rejection_summary()


@pytest.mark.parametrize(
    "harvests, final, retired, offset, expected",
    [
        # missed before the admit, counted after it: the delete's harvest
        # holds it, before the admission (never after the retirement)
        ({"D": (1, 0)}, None, True, 0, (1, 0, 0)),
        # 3 before the admit; 5 begun after the delete began
        ({"A": (3, 0)}, (5, 25), True, 0, (3, 5, 0)),
        # a remainder with an event begun before the delete: undecidable
        ({"A": (3, 0)}, (2, 15), True, 0, (3, 0, 2)),
        # its earliest time unknown under contention
        ({"A": (3, 0)}, (2, 25 | (1 << 63)), True, 0, (3, 0, 2)),
        # a harvest that failed: what came later is unresolved
        ({"A": (0, 5), "D": (4, 0)}, (2, 25), True, 0, (0, 0, 6)),
        # still admitted at the end
        ({"A": (3, 0)}, (2, 25), False, 0, (3, 0, 2)),
        # kernel time unconvertible
        ({"A": (3, 0)}, (2, 25), True, None, (3, 0, 2)),
        # kernel time ahead of the observer's by the namespace offset
        ({"A": (3, 0)}, (2, 18), True, 5, (3, 2, 0)),
        # read while a hook was still updating it: not consistent
        ({"A": (3, 0)}, (2, 25, False), True, 0, (3, 0, 2)),
    ],
)
def test_ebpf_rejections_are_phased_only_where_harvests_prove_it(
    tmp_path, harvests, final, retired, offset, expected
):
    result = _harvest_channel(tmp_path, harvests, final, retired, offset)
    assert (
        result["before_or_between_admissions"],
        result["after_retirement"],
        result["unresolved"],
    ) == expected
    assert result["never_owned"] == 0 and result["reconciled"]


@pytest.mark.parametrize("began, phase", [(25, "unresolved"), (35, "after")])
def test_ebpf_a_rejection_while_discovery_owned_the_socket_again_is_not_after(
    tmp_path, began, phase
):
    """The cookie is retired (its delete pending), admitted again and retired
    again before the delete is acknowledged; the re-admission is never
    installed. A rejection begun while discovery owned it again (25, within
    22..30) is unresolved; one after its final retirement (35) is after."""
    from vllm.kv_profiling.endpoint_probes.ebpf import Channel

    channel = Channel(None, tmp_path)
    channel.child, channel.send_command = object(), lambda wire: True
    channel.monotonic_offset = 0
    channel.admit(42, 1, at=5)
    channel.acknowledged({"seq": 1, "op": "A", "cookie": 42, "admission": 1,
                          "ok": True, "before_ns": 6, "applied_ns": 7, "rejected": 0,
                          "rejected_first_ns": 0, "rejected_errno": 0})  # fmt: skip
    channel.retire(42, 1, at=15)
    channel.admit(42, 2, at=22)
    channel.retire(42, 2, at=30)
    channel.acknowledged({"seq": 2, "op": "D", "cookie": 42, "admission": 1,
                          "ok": True, "stale": False, "before_ns": 16,
                          "applied_ns": 17, "rejected": 0, "rejected_first_ns": 0,
                          "rejected_errno": 0, "harvested": True})  # fmt: skip
    assert not channel.pending and not channel.outbox  # converged: no install
    channel.rejected[42] = {"cookie": 42, "count": 1, "first_ns": began,
                            "last_ns": began, "consistent": True}  # fmt: skip
    channel.losses["filter_exclusion", "excluded_unowned"] = 1
    result = channel.rejection_summary()
    assert (result["after_retirement"], result["unresolved"]) == (
        (1, 0) if phase == "after" else (0, 1)
    )


def test_ebpf_the_terminal_summary_scales_with_cookies_not_their_square(tmp_path):
    """16,384 cookies admitted and retired, one at a time: the summary and
    the per-cookie closing stay linear."""
    from vllm.kv_profiling.endpoint_probes.ebpf import Channel

    channel = Channel(None, tmp_path)
    began = time.monotonic()
    for cookie in range(1, 16385):
        channel.acknowledged({"seq": 0, "op": "A", "cookie": cookie,
                              "admission": cookie, "ok": True, "before_ns": 1,
                              "applied_ns": 2, "rejected": 0,
                              "rejected_first_ns": 0, "rejected_errno": 0})  # fmt: skip
        channel.acknowledged({"seq": 0, "op": "D", "cookie": cookie,
                              "admission": cookie, "ok": True, "stale": False,
                              "before_ns": 3, "applied_ns": 4, "rejected": 0,
                              "rejected_first_ns": 0, "rejected_errno": 0,
                              "harvested": True})  # fmt: skip
    channel.losses["filter_exclusion", "excluded_unowned"] = 0
    assert channel.rejection_summary()["reconciled"]
    assert time.monotonic() - began < 2


def test_ebpf_ownership_ends_at_the_scan_s_decision_time(tmp_path):
    """A removal (or suspension) found by a scan ends the admission at the
    scan's published time: an event after it, processed before the channel
    heard of the removal, is not verified; one before it is."""
    from types import SimpleNamespace

    from vllm.kv_profiling.endpoint_probes import runner, sockets

    channel = _fake_channel(tmp_path)
    try:
        observer = runner.Observer.__new__(runner.Observer)
        observer.ebpf, observer.endpoint = channel, "d"
        observer.tracker = sockets.Tracker()
        observer.config = SimpleNamespace(tcp=None, ebpf={"enabled": True},
                                          required=["ebpf"])  # fmt: skip
        observer.discovery_failures = observer.failed_required = 0
        rows = []
        observer.writer = SimpleNamespace(put=rows.append)
        me = {**process_identity(os.getpid()), "engine_id": "d", "rank": 0,
              "role": "worker"}  # fmt: skip
        record = {"cookie": 42, "owners": [me], "owner_pids": [me["pid"]],
                  "holders": [], "row": {}}  # fmt: skip

        def scan(owned):
            result = sockets.Scan(time.monotonic_ns())
            result.complete, result.owned = True, owned
            result.finished_ns = time.monotonic_ns()
            return result

        observer.discovery = SimpleNamespace(scan=lambda: scan({42: record}))
        observer.discover()
        _settle(channel)
        admission = next(iter(channel.decided))
        observer.discovery = SimpleNamespace(scan=lambda: scan({}))
        observer.discover()
        changes = [r["observation"] for r in rows if r.get("observation")]
        ended = next(c["ended_ns"] for c in changes if c.get("change") == "removed")
        assert channel.decided[admission]["until"] == ended
        event = _event(admission)
        assert channel.verify(event, ended + 1)["verified"] is False
        assert channel.verify(event, ended - 1)["verified"] is True
        _settle(channel)
    finally:
        assert channel.stop(None)["clean"] is True


@pytest.mark.parametrize(
    "verified, polls, damage",
    [
        (True, True, None),
        (False, True, None),
        (True, False, None),
        # identities that do not agree make no candidate
        (True, False, "no selection"),
        (True, False, "selection names another admission"),
        (True, False, "admission record of another cookie"),
    ],
)
def test_a_transferred_cookie_s_ebpf_event_joins_only_its_verified_admission(
    tmp_path, verified, polls, damage
):
    """One cookie, admitted for an old worker and then (another admission) for
    a new one: a verified event under the new admission is a candidate of the
    new worker's READ only, with or without TCP samples; an unverified event is
    no one's."""
    from tests.distributed.kv_transfer.test_profiler_core import fields

    run = str(uuid4())
    me = process_identity(os.getpid())
    cores = {}
    for engine in ("old", "new"):
        recorder = ProcessRecorder(
            NetworkProfilerConfig(
                level="l3", run_id=run, output_dir=str(tmp_path / engine),
                ring_records=64,
                context={"role": "worker", "engine_id": engine, "rank": 0,
                         "kv_role": "kv_consumer"},
            )
        )  # fmt: skip
        assert recorder.wait_ready()
        recorder.record(*fields(EventKind.TRANSFER_BIND, handle=7))
        recorder.record(*fields(EventKind.POST_RETURN, handle=7))
        cores[engine] = recorder
    writer = Writer(tmp_path / "probe", metadata(run), ProbeConfig())
    owners = {e: {**me, "engine_id": e, "rank": 0, "role": "worker"} for e in cores}
    now = time.monotonic_ns()
    for admission, engine in ((1, "old"), (2, "new")):
        cookie = 44 if damage == "admission record of another cookie" else 17
        admitted = {"change": "admitted", "cookie": cookie if admission == 2 else 17,
                    "admission": admission, "owners": [owners[engine]]}  # fmt: skip
        writer.put(event("probe_status", "socket_ownership", "17", now, now, admitted))
    if polls:  # a sample of the socket under each admission
        for admission, engine in ((1, "old"), (2, "new")):
            writer.put(event("endpoint_metrics", "socket", "17", now, now,
                             {"cookie": 17, "owners": [owners[engine]],
                              "selection": {"by": "owned_socket", "cookie": 17,
                                            "admission": admission}},
                             "tcp"))  # fmt: skip
    native = {
        **tcp_event_row(), "cookie": 17, "admission": 2, "timestamp_ns": now,
        "time_conversion_qualified": True,
        "selection": {"by": "owned_socket", "cookie": 17, "admission": 2,
                      "verified": verified},
    }  # fmt: skip
    if damage == "no selection":
        del native["selection"]
    elif damage == "selection names another admission":
        native["selection"]["admission"] = 1
    writer.put(event("endpoint_metrics", "ebpf", "d", now, now, native, "tcp"))
    for recorder in cores.values():
        recorder.record(*fields(EventKind.COMPLETE_OBSERVED, handle=7))
        assert recorder.close().complete
    assert writer.close(True, "test", {})
    result = candidates(
        [cores[e].output_dir / "events.jsonl" for e in cores],
        [tmp_path / "probe/events.jsonl"],
    )
    events = {
        r["engine_id"]: sum(len(c.get("tcp_events", [])) for c in r["candidates"])
        for r in result
    }
    assert events == {"old": 0, "new": int(verified and not damage)}
    if polls:  # the samples stay each admission's
        assert all(r["association"] == "unique_temporal_candidate" for r in result)
    new = next(r for r in result if r["engine_id"] == "new")
    if verified and not damage:
        basis = [e["ownership_basis"] for c in new["candidates"]
                 for e in c.get("tcp_events", [])]  # fmt: skip
        assert basis == ["verified_admission"]


@pytest.mark.skipif(
    not os.environ.get("KVP_EBPF_ARTIFACTS"),
    reason="needs built eBPF artifacts (KVP_EBPF_ARTIFACTS) and CAP_BPF/CAP_PERFMON",
)
def test_a_live_observer_selects_its_endpoint_s_sockets_in_the_kernel(tmp_path):
    """The real collector, under the observer: the endpoint process's socket is
    admitted in the kernel by cookie, its events are recorded under that
    admission and verified, the test's own end of the connection is rejected
    and counted, and the capture with required TCP and eBPF checks complete."""
    allowed = sorted(os.sched_getaffinity(0))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        port = listener.getsockname()[1]
        owner, inventory, endpoint_dir = start_recorded_endpoint(
            tmp_path, allowed[0], connect=port
        )
        accepted, _ = listener.accept()
        run_id = inventory[0]["run_id"]
        core_dir = endpoint_dir / "profiles"
        ebpf = {"enabled": True, "privileged_opt_in": True, "mode": "tcp_tracepoints",
                "artifacts": os.environ["KVP_EBPF_ARTIFACTS"]}  # fmt: skip
        settings = ProbeConfig(
            interval_s=0.1, slow_interval_s=0.1, interfaces=["lo"],
            tcp={"owned_sockets": True}, discovery_interval_s=0.05, ebpf=ebpf,
            required=["netdev", "tcp", "ebpf"],
        )  # fmt: skip
        path = _observer_config(tmp_path, run_id, core_dir, allowed[-1],
                                probes=asdict(settings),
                                owner_exit_grace_s=30)  # fmt: skip
        observer_dir = path.parent
        entry = str(Path(probe.__file__))
        process = subprocess.Popen(
            [sys.executable, entry, "--config", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )  # fmt: skip
        glob = f"captures/{run_id}/*/manifest.json"
        try:

            def ready():
                paths = observer_dir.glob(glob)
                return any(json.loads(p.read_text()).get("ready") for p in paths)

            wait_for(ready, timeout=30)
            time.sleep(0.3)
            accepted.close()  # the connection closes: state changes on both ends
            time.sleep(0.5)
            (endpoint_dir / "close").touch()
            wait_for(lambda: (endpoint_dir / "closed").exists(), timeout=20)
            (endpoint_dir / "exit").touch()
            owner.wait(timeout=20)
            time.sleep(0.3)
            closed = {"run_id": run_id, "endpoint": "d", "children_closed": True}
            (observer_dir / "children.json").write_text(json.dumps(closed))
            (observer_dir / "stop").touch()
            _, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stderr
        finally:
            if process.poll() is None:
                process.kill()
            if owner.poll() is None:
                (endpoint_dir / "exit").touch()
                owner.wait(timeout=20)
    manifest = next(observer_dir.glob(glob))
    rows = [json.loads(line) for line in manifest.with_name("events.jsonl").open()]
    admitted = [r["observation"] for r in rows if r.get("scope") == "socket_ownership"
                and r["observation"]["change"] == "admitted"]  # fmt: skip
    assert len(admitted) == 1, admitted
    cookie, admission = admitted[0]["cookie"], admitted[0]["admission"]
    events = [r["transport"]["tcp"] for r in rows if r.get("scope") == "ebpf"
              and (r.get("transport") or {}).get("tcp")
              and r["transport"]["tcp"].get("type") == "tcp_event"]  # fmt: skip
    assert events and all(
        e["selection"]["cookie"] == cookie
        and e["selection"]["admission"] == admission
        and e["selection"]["verified"] is True
        for e in events
    ), events
    lifecycle = [
        r["observation"]
        for r in rows
        if r.get("scope") == "ebpf"
        and (r.get("observation") or {}).get("type") == "lifecycle"
    ]
    rejections = lifecycle[-1]["rejections"]
    assert rejections["total"] >= 1 and rejections["reconciled"], rejections
    record = {"run_id": run_id, "endpoint": "d", "process_id": rows[0]["process_id"],
              "exit_code": 0, "launcher": "test-process"}  # fmt: skip
    subprocess.run([sys.executable, entry, "--record-exit", str(observer_dir)],
                   input=json.dumps(record), text=True, check=True)  # fmt: skip
    checked = subprocess.run(
        [sys.executable, entry, "--check-capture", str(observer_dir),
         "--core-dir", str(core_dir)],
        capture_output=True, text=True, timeout=30,
    )  # fmt: skip
    summary = json.loads(checked.stdout)
    assert checked.returncode == 0 and summary["complete"], summary
