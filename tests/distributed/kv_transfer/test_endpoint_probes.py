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
import socket
import struct
import subprocess
import sys
import sysconfig
import time
from dataclasses import asdict
from pathlib import Path
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


@pytest.mark.parametrize(
    "family,address", [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")]
)
def test_real_tcp_dump_is_filtered_and_reports_cookie_and_bytes(family, address):
    with socket.socket(family) as listener, socket.socket(family) as client:
        listener.bind((address, 0))
        listener.listen()
        client.connect(listener.getsockname())
        server, _ = listener.accept()
        with server:
            client.sendall(b"payload")
            assert server.recv(7) == b"payload"
            selectors = {
                "local_ips": [address],
                "peer_ips": [address],
                "local_ports": [client.getsockname()[1]],
                "peer_ports": [listener.getsockname()[1]],
            }
            rows = tcp.sockets(selectors)
            assert len(rows) == 1
            row = rows[0]
            assert row["cookie"] is not None
            assert row["inode"] > 0
            assert row["tcp_info_length"] >= 104
            assert row["fields"]["bytes_acked"]["value"] >= 1
            assert row["fields"]["rtt_us"]["unit"] == "us"
            assert row["send_queue_bytes"] >= 0
            assert (
                tcp.sockets({**selectors, "exclude_ports": selectors["peer_ports"]})
                == []
            )


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


def start_recorded_endpoint(tmp_path, cpu, *, overlap_cpu=None, close_delay=0):
    directory = tmp_path / "endpoint"
    run_id = str(uuid4())
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
        "inventory": inventory,
        "run_id": run_id,
        "endpoint": "d",
        "role": "decode",
        "owner_pid": owner.pid,
        "owner_start": process_identity(owner.pid)["start_ticks"],
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
def test_synthetic_read_windows_have_only_qualified_flow_candidates(
    tmp_path,
    flow_count,
    unaligned,
):
    """Real recorder and sockets, not evidence of a native NIXL handle/FD map."""
    from tests.distributed.kv_transfer.test_profiler_core import fields
    from vllm.kv_profiling.endpoint_probes.identity import ownership

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
    for index in range(flow_count):
        with socket.socket() as listener, socket.socket() as client:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            client.connect(listener.getsockname())
            server, _ = listener.accept()
            with server:
                rows = tcp.sockets(
                    {
                        "local_ips": ["127.0.0.1"],
                        "peer_ips": ["127.0.0.1"],
                        "local_ports": [client.getsockname()[1]],
                    }
                )
                owners, unavailable, _ = ownership(
                    core_dir, run_id, "decode", os.readlink("/proc/self/ns/net")
                )
                assert not unavailable
                assert owners[rows[0]["inode"]][0]["engine_id"] == "d"
                row = {**rows[0], "owners": owners[rows[0]["inode"]]}
                now = time.monotonic_ns()
                writer.put(
                    event(
                        "endpoint_metrics", "socket", str(index), now, now, row, "tcp"
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


def test_worker_pid_generation_is_not_inferred_from_a_reused_pid(tmp_path):
    from vllm.kv_profiling.endpoint_probes.identity import ownership

    run_id = str(uuid4())
    core = ProcessRecorder(
        NetworkProfilerConfig(
            level="l3",
            run_id=run_id,
            output_dir=str(tmp_path),
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
    assert core.close().complete
    path = core.output_dir / "events.jsonl"
    rows = path.read_text().splitlines()
    old = json.loads(rows[0])
    old["start_ticks"] -= 1
    rows[0] = json.dumps(old)
    path.write_text("\n".join(rows) + "\n")
    owners, unavailable, _ = ownership(
        tmp_path, run_id, "decode", os.readlink("/proc/self/ns/net")
    )
    assert not owners
    assert len(unavailable) == 1
    assert "generation/namespace mismatch" in unavailable[0]["error"]


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
    from vllm.kv_profiling.schema import RECORD, decode_record

    words = [0] * 16
    words[2] = (1 << 48) | (22 << 32)
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
        # Corrupt input, not a patched process/object: the expected generation is stale.
        worker["start_ticks"] += 1
    settings = ProbeConfig(interfaces=["lo"], required=["netdev"], interval_s=0.1)
    cfg = {
        "probes": asdict(settings),
        "inventory": entries,
        "run_id": run_id,
        "endpoint": "d",
        "role": "decode",
        "owner_pid": process.pid,
        "owner_start": process_identity(process.pid)["start_ticks"],
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
