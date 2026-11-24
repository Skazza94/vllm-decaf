# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-rate endpoint observations, independent of serving and L1 helper closure."""

import hashlib
import json
import os
import resource
import select
import sys
import threading
import time
from pathlib import Path
from uuid import UUID, uuid4

from . import netdev, rdma, tasks, tcp
from .code import code_digest
from .config import settings_dict
from .exit_record import CAPTURE_FORMAT
from .health import (
    core_completion,
    recorder_counts,
    side_problems,
    validate_inventory,
)
from .identity import discover, ownership, placement, process_identity
from .netlink import ObservationError
from .records import Writer, event

# Lists an observation row may carry, split when a row exceeds the budget: the
# writer's 16 KiB row limit less room for the sequence number it adds.
SPLITTABLE = (
    "fields",
    "qps",
    "disappeared",
    "ownership_unavailable",
    "snapshots",
    "threads",
    "irqs",
    "softirqs",
    "missing",
    "unavailable",
    "unreadable",
    "absent",
    "unreadable_irqs",
)
ROW_BUDGET = (16 << 10) - 64


def _time_namespace():
    """This process's time namespace, or None on a kernel without time namespaces
    (before Linux 5.6 or built without them), where every process shares the host's
    monotonic clock."""
    try:
        return os.readlink("/proc/self/ns/time")
    except OSError:
        return None


# How long an observer waits for its eBPF collector to load and attach (or to
# report itself unavailable) before it starts sampling.
EBPF_ANSWER_TIMEOUT_S = 10.0


class Observer:
    """Observe an explicit owner; never signals it, rewrites sockets or retunes IRQs."""

    def __init__(
        self,
        config,
        *,
        run_id,
        endpoint,
        role,
        cpus,
        input_dir,
        output_dir,
        irq_policy,
        expected=None,
        owner_pid=None,
        owner_start=None,
        inventory=None,
        package_sha256=None,
        owner_exit_grace_s=2.0,
    ):
        """A launcher names the ``expected`` recorders per role; the observer
        finds and freezes them and their owner itself (``identity.discover``).
        An explicit ``inventory`` with its ``owner_pid``/``owner_start`` is for
        callers that already hold them."""
        UUID(run_id)
        if type(owner_exit_grace_s) not in (int, float) or not (
            0 < owner_exit_grace_s <= 600
        ):
            raise ValueError("owner_exit_grace_s is in (0, 600] seconds")
        # how long, after the owner exits, the launcher has to report its
        # children closed and stop the observer (one deadline with the launcher)
        self.owner_exit_grace_s = float(owner_exit_grace_s)
        owner_evidence = {"frontends": [], "parent": None, "supplied": True}
        if inventory is None:
            if expected is None or owner_pid is not None:
                raise ValueError("observer discovery takes expected counts only")
            inventory, found, owner_evidence = discover(
                input_dir, run_id, expected, role
            )
            owner_pid, owner_start = found["pid"], found["start_ticks"]
        validate_inventory(inventory, run_id)
        if wrong := side_problems(inventory, role):
            raise ValueError(wrong[0])
        if expected is None:
            expected = recorder_counts(inventory)
        if recorder_counts(inventory) != expected:
            raise ValueError("the endpoint's recorders are not the expected ones")
        if role not in ("prefill", "decode") or not endpoint:
            raise ValueError("observer requires a prefill/decode endpoint")
        if irq_policy not in ("operator_verified", "virtual_not_applicable"):
            raise ValueError("operator must verify IRQ separation")
        if not isinstance(endpoint, str) or type(owner_pid) is not int or owner_pid < 1:
            raise ValueError("invalid endpoint or owner PID")
        if not isinstance(cpus, list) or any(
            type(cpu) is not int or cpu < 0 for cpu in cpus
        ):
            raise ValueError("cpus must be a non-negative integer list")
        if Path(output_dir).resolve().is_relative_to(Path(input_dir).resolve()):
            raise ValueError("observer output must be separate from core input")
        self.owner = process_identity(owner_pid)
        if self.owner["start_ticks"] != owner_start:
            raise ValueError("owner PID generation mismatch")
        if self.owner["net_namespace"] != os.readlink("/proc/self/ns/net"):
            raise ValueError("observer must share the endpoint network namespace")
        if self.owner["pid_namespace"] != os.readlink("/proc/self/ns/pid"):
            raise ValueError("observer must share the endpoint PID namespace")
        current = os.sched_getaffinity(0)
        if (
            not cpus
            or not set(cpus) <= current
            or set(cpus) & os.sched_getaffinity(owner_pid)
        ):
            raise ValueError(
                "observer CPUs must be allowed and separate from owner CPUs"
            )
        os.sched_setaffinity(0, cpus)
        self.pidfd = os.pidfd_open(owner_pid)
        if process_identity(owner_pid) != self.owner:
            os.close(self.pidfd)
            raise ValueError("owner changed while opening pidfd")
        self.ebpf = None
        self.config, self.run_id, self.role = config, run_id, role
        self.tcp_reader = (
            tcp.SocketReader(config.tcp, config.max_sockets) if config.tcp else None
        )
        self.input_dir, self.endpoint = input_dir, endpoint
        self.process_id = str(uuid4())
        self.stop_event = threading.Event()
        self.failed_required = 0
        self.ebpf_failures = 0
        self.ebpf_event_loss = {}
        self.isolation_failures = 0
        self.verified_processes = set()
        self.inventory = inventory
        # The endpoint's serving processes: every recorder's process in the
        # inventory and the owner, each with the generation it was verified with.
        targets = {}
        for meta in inventory:
            targets.setdefault(meta["pid"], {
                "pid": meta["pid"], "start_ticks": meta["start_ticks"],
                "role": meta["context"]["role"]})  # fmt: skip
        targets.setdefault(owner_pid, {
            "pid": owner_pid, "start_ticks": owner_start, "role": "owner"})  # fmt: skip
        self.thread_targets = list(targets.values())
        self.samples = self.gaps = 0
        self.previous = {}
        self.seen_identities = {}
        self.started_cpu = time.process_time()
        self.started_ns = time.monotonic_ns()
        self.metadata = {
            "schema": 1,
            "probe_schema": 1,
            "capture_format": CAPTURE_FORMAT,
            "type": "metadata",
            "run_id": run_id,
            "process_id": self.process_id,
            "clock_id": str(uuid4()),
            "clock_source": "CLOCK_MONOTONIC",
            "boot_id": netdev.read_small("/proc/sys/kernel/random/boot_id"),
            "time_namespace": _time_namespace(),
            "net_namespace": os.readlink("/proc/self/ns/net"),
            "pid_namespace": os.readlink("/proc/self/ns/pid"),
            "context": {
                "role": "endpoint_observer",
                "endpoint_role": role,
                "endpoint": endpoint,
            },
            "owner": self.owner,
            "config": settings_dict(config),
            "inventory": inventory,
            "expected": expected,
            "owner_evidence": owner_evidence,
            # what actually ran, wherever it came from (zipapp, tree, package)
            "code_sha256": code_digest(),
            "package_sha256": package_sha256,
            "config_sha256": hashlib.sha256(
                json.dumps(settings_dict(config), sort_keys=True).encode()
            ).hexdigest(),
            "affinity": sorted(cpus),
            "irq_policy": irq_policy,
            "runtime": {
                "torch_imported": "torch" in sys.modules,
                "vllm_imported": "vllm" in sys.modules,
            },
            "capabilities": {
                "exact_read_to_flow": False,
                "exact_read_to_qp": False,
                "short_lived_flows": "may_be_unobserved",
                "ownership": "optional_proc_snapshot",
                "eBPF": False,
                "eBPF_requested": bool(config.ebpf.get("enabled", False)),
                "L2": False,
            },
        }
        try:
            self.writer = Writer(
                Path(output_dir) / run_id / self.process_id, self.metadata, config
            )
        except BaseException:
            os.close(self.pidfd)
            raise
        if config.ebpf.get("enabled"):
            from .ebpf import Collector, Settings

            self.ebpf = Collector(
                Settings(**config.ebpf), inventory, self.writer.directory / "ebpf"
            )
            self.ebpf.start()
            self.wait_ebpf_answer()
        # ready for the workload only now: every source started, an eBPF
        # collector attached (or reported unavailable)
        self.writer.publish_ready()

    def emit(
        self, scope, entity, begin, data, transport="unknown", kind="endpoint_metrics"
    ):
        row = event(kind, scope, entity, begin, time.monotonic_ns(), data, transport)
        if len(json.dumps(row, allow_nan=False)) <= ROW_BUDGET:
            return self.writer.put(row)
        # Over the writer's row limit: split the row's list items (all splittable
        # lists, in order, as one sequence) in half, recursively, until every row
        # fits. Each row's "slices" gives, per list, the exact [start, end, total]
        # range it carries, so every item is in exactly one row.
        lists = [
            key for key in SPLITTABLE if isinstance(data.get(key), list) and data[key]
        ]
        count = sum(len(data[key]) for key in lists)
        if count < 2:
            return self.writer.put(row)  # one indivisible item: the writer counts it
        slices = data.get("slices", {})
        result = None
        for low, high in ((0, count // 2), (count // 2, count)):
            part = {**data, "slices": dict(slices)}
            offset = 0
            for key in lists:
                values = data[key]
                first, _, whole = slices.get(key, (0, len(values), len(values)))
                begin_at = min(max(low - offset, 0), len(values))
                end_at = min(max(high - offset, 0), len(values))
                part[key] = values[begin_at:end_at]
                part["slices"][key] = [first + begin_at, first + end_at, whole]
                offset += len(values)
            result = self.emit(scope, entity, begin, part, transport, kind)
        return result

    def failure(self, source, error, begin):
        required = source in self.config.required
        self.failed_required += int(required)
        self.emit(
            "source",
            source,
            begin,
            {"available": False, "required": required, "error": str(error)},
            kind="probe_status",
        )

    def observe(self, source, entity, function, transport="unknown"):
        begin = time.monotonic_ns()
        try:
            data = function()
            if source == "rdma" and source in self.config.required:
                absent = [
                    family
                    for family in self.config.rdma_required_families
                    if not data.get("coverage", {}).get(family)
                ]
                if absent:
                    self.failure(
                        source,
                        "required RDMA counter families unavailable: "
                        + ", ".join(absent),
                        begin,
                    )
            # Missing coverage or access errors fail the source (the required
            # rule applies); the counters that were read are still recorded.
            if source in ("threads", "interrupts") and data.get("coverage_problem"):
                self.failure(source, data["coverage_problem"], begin)
            # Counter continuity applies only to documented cumulative metrics.
            key = source + ":" + entity
            current = (
                {
                    item["name"]: item["value"]
                    for item in data.get("fields", [])
                    if item.get("meaning") == "cumulative"
                }
                if isinstance(data.get("fields"), list)
                else {
                    name: item["value"]
                    for name, item in data.get("fields", {}).items()
                    if item.get("meaning") == "cumulative"
                }
            )
            identity = json.dumps({k: data.get(k) for k in ("ifindex", "identity")})
            previous = self.seen_identities.get(key)
            reset = bool(
                previous
                and (
                    previous[0] != identity
                    or any(
                        name in previous[1] and value < previous[1][name]
                        for name, value in current.items()
                    )
                )
            )
            generation = (previous[2] + int(reset)) if previous else 1
            self.seen_identities[key] = identity, current, generation
            self.emit(
                source,
                entity,
                begin,
                {
                    **data,
                    "observation_generation": generation,
                    "counter_reset_or_identity_change": reset,
                    "reuse_between_polls_excluded": False,
                },
                transport,
            )
        except (OSError, ValueError, ObservationError) as error:
            self.failure(source, error, begin)

    def sample_tcp(self):
        begin = time.monotonic_ns()
        try:
            rows = self.tcp_reader.read()
            owners, unavailable, _ = ownership(
                self.input_dir,
                self.run_id,
                self.role,
                self.metadata["net_namespace"],
                self.metadata["affinity"],
            )
            now = {}
            for row in rows:
                identity = json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "family",
                            "local_ip",
                            "local_port",
                            "peer_ip",
                            "peer_port",
                            "cookie",
                        )
                    },
                    sort_keys=True,
                )
                if row["cookie"] is None:
                    identity += f":unidentified_snapshot:{begin}"
                row["owners"] = owners.get(row["inode"], [])
                row["ownership_basis"] = "proc_inode_snapshot_not_cookie_verified"
                row["ownership_unavailable"] = unavailable
                row["association"] = (
                    "process_owned" if row["owners"] else "configured_filter"
                )
                row["continuity"] = (
                    "observed_cookie_and_tuple"
                    if (row["cookie"] is not None and identity in self.previous)
                    else "new_or_unknown"
                )
                row["reuse_between_polls_excluded"] = False
                values = {
                    name: field["value"]
                    for name, field in row["fields"].items()
                    if field["meaning"] == "cumulative" and field["available"]
                }
                prior = self.previous.get(identity, {})
                row["counter_reset"] = any(
                    name in prior and value < prior[name]
                    for name, value in values.items()
                )
                if row["counter_reset"]:
                    row["continuity"] = "counter_reset"
                now[identity] = values
                self.emit("socket", identity, begin, row, "tcp")
                if (
                    "tcp" in self.config.required
                    and row["tcp_info_expected"]
                    and not row["tcp_info_length"]
                ):
                    self.failure("tcp", "TCP_INFO unavailable", begin)
            self.emit(
                "tcp_poll",
                self.endpoint,
                begin,
                {
                    "matched_sockets": len(rows),
                    "empty_snapshot": not rows,
                    "unobserved_short_lived_flows_possible": True,
                    "disappeared": sorted(set(self.previous) - set(now)),
                    "ownership_unavailable": unavailable,
                },
                kind="probe_status",
            )
            self.previous = now
        except (OSError, ValueError, KeyError, ObservationError) as error:
            self.failure("tcp", error, begin)
            self.previous = {}

    def ebpf_row(self, row):
        if row.get("outcome") == "FAIL":
            self.ebpf_failures += 1
        data = {**row, "exact": False, "transfer_refs": "unbound"}
        begin = end = time.monotonic_ns()
        if row.get("type") == "tcp_event":
            data["clock_source"] = "kernel_monotonic"
            data["time_conversion_qualified"] = False
            if self.ebpf is not None and self.ebpf.monotonic_offset is not None:
                from .ebpf import convert_time

                begin = end = convert_time(
                    row["timestamp_ns"], self.ebpf.monotonic_offset
                )
                data["endpoint_time_ns"] = begin
                data["time_conversion_qualified"] = True
        return self.writer.put(
            event(
                "endpoint_metrics",
                "ebpf",
                self.endpoint,
                begin,
                end,
                data,
                "tcp" if row.get("type") == "tcp_event" else "unknown",
            )
        )

    def wait_ebpf_answer(self, timeout_s=None):
        """Wait, bounded, for the native collector's first answer: ``ready``, or
        a status that it is unavailable. Loading and attaching take a moment, and
        a sample taken meanwhile would count the source as missing; the observer
        reports itself ready only after this, so no workload event precedes the
        attachment. A collector still starting at the deadline stays
        ``not_ready``: the missed deadline is recorded before the workload is
        released, a failure of a required source even if a later poll sees the
        collector ready."""
        if timeout_s is None:
            timeout_s = EBPF_ANSWER_TIMEOUT_S
        deadline = time.monotonic() + timeout_s
        while (
            self.ebpf.status.get("state") == "starting"
            and self.ebpf.running()
            and time.monotonic() < deadline
        ):
            for row in self.ebpf.poll(self.writer, 0.05):
                self.ebpf_row(row)
        if self.ebpf.status.get("state") == "starting":
            self.failure("ebpf", "not_ready_within_deadline", time.monotonic_ns())

    def sample_ebpf(self):
        if self.ebpf is None:
            return
        try:
            for row in self.ebpf.poll(self.writer):
                self.ebpf_row(row)
            if not self.ebpf.status["available"]:
                self.failure(
                    "ebpf",
                    self.ebpf.status.get("reason", "not_ready"),
                    time.monotonic_ns(),
                )
        except (OSError, ValueError, KeyError) as error:
            self.failure("ebpf", error, time.monotonic_ns())

    def sample(self, slow=True):
        begin = time.monotonic_ns()
        isolation = placement(
            self.inventory,
            self.metadata["affinity"],
            self.metadata["net_namespace"],
            self.verified_processes,
        )
        if not isolation["verified"]:
            self.isolation_failures += 1
        self.emit("cpu_isolation", self.endpoint, begin, isolation, kind="probe_status")
        if self.config.tcp:
            self.sample_tcp()
        for name in self.config.interfaces:
            self.observe("netdev", name, lambda name=name: netdev.interface(name))
            if slow and self.config.driver_statistics:
                self.observe("driver", name, lambda name=name: netdev.driver(name))
        if slow:
            for spec in self.config.rdma_ports:
                entity = f"{spec['device']}/{spec['port']}"
                self.observe("rdma", entity, lambda spec=spec: rdma.port(spec), "rdma")
                if self.config.qp:
                    self.observe(
                        "qp", entity, lambda spec=spec: {"qps": rdma.qps(spec)}, "rdma"
                    )
        if self.config.threads:
            for target in self.thread_targets:
                self.observe(
                    "threads",
                    f"{target['role']}/{target['pid']}",
                    lambda t=target: {
                        "role": t["role"],
                        **tasks.process_threads(
                            t["pid"], t["start_ticks"], self.config.max_threads
                        ),
                    },
                )
        if self.config.interrupts:
            self.observe(
                "interrupts",
                "nic",
                lambda: tasks.nic_interrupts(
                    self.config.interfaces,
                    [spec["device"] for spec in self.config.rdma_ports],
                ),
            )
            self.observe("interrupts", "softirqs", tasks.net_softirqs)
        self.sample_ebpf()
        self.samples += 1
        self.emit(
            "observer", self.process_id, begin, self.resources(), kind="probe_resource"
        )

    def resources(self):
        native_cpu = self.ebpf.cpu_s if self.ebpf else 0.0
        native_rss = self.ebpf.rss_hwm_kib if self.ebpf else 0
        return {
            "cpu_s": time.process_time() - self.started_cpu + native_cpu,
            "rss_hwm_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            + native_rss,
            "elapsed_ns": time.monotonic_ns() - self.started_ns,
            "samples": self.samples,
            "observation_gaps": self.gaps,
        }

    def run(self, *, stop_file, endpoint_complete_file):
        next_tick = next_slow = time.monotonic()
        reason, controller = "owner_exit_timeout", False
        owner_exited_at = None
        try:
            while True:
                now = time.monotonic()
                if self.writer.error:
                    reason = "writer_error"
                    break
                if now >= next_tick:
                    self.sample(now >= next_slow)
                    if now >= next_slow:
                        next_slow = now + self.config.slow_interval_s
                    skipped = max(
                        0, int((time.monotonic() - next_tick) / self.config.interval_s)
                    )
                    self.gaps += skipped
                    if skipped:
                        self.emit(
                            "sampling",
                            self.endpoint,
                            time.monotonic_ns(),
                            {"missed_ticks": skipped},
                            kind="probe_status",
                        )
                    next_tick += (skipped + 1) * self.config.interval_s
                if Path(stop_file).exists() or self.stop_event.is_set():
                    reason = (
                        "controller_stop"
                        if not self.stop_event.is_set()
                        else "signal_stop"
                    )
                    break
                if select.select([self.pidfd], [], [], 0)[0]:
                    if owner_exited_at is None:
                        owner_exited_at = time.monotonic()
                        self.emit(
                            "lifecycle",
                            self.endpoint,
                            time.monotonic_ns(),
                            {"owner_exited": True, "children_closed": "unknown"},
                            kind="probe_status",
                        )
                    if time.monotonic() - owner_exited_at >= self.owner_exit_grace_s:
                        break
                self.stop_event.wait(min(0.1, max(0, next_tick - time.monotonic())))
            self.sample(True)
            if Path(endpoint_complete_file).exists():
                notice = json.loads(netdev.read_small(endpoint_complete_file))
                controller = (
                    notice.get("run_id") == self.run_id
                    and notice.get("endpoint") == self.endpoint
                    and notice.get("children_closed") is True
                )
            closure = core_completion(self.input_dir, self.run_id, self.inventory)
            if self.ebpf:
                native = self.ebpf.stop(self.writer, self.ebpf_row)
                self.ebpf_row({"type": "lifecycle", **native})
                if not native.get("clean") and "ebpf" in self.config.required:
                    self.failed_required += 1
                # a clean run with known event loss is degraded: it ran, but its
                # events cannot certify counter agreement
                self.ebpf_event_loss = native.get("event_loss") or {}
            complete = (
                controller
                and closure["all_closed"]
                and not self.failed_required
                and not self.ebpf_failures
                and not self.isolation_failures
                and not self.gaps
                and reason == "controller_stop"
            )
            self.emit(
                "lifecycle",
                self.endpoint,
                time.monotonic_ns(),
                {
                    "stop_reason": reason,
                    "controller_children_closed": controller,
                    "core": closure,
                    "owner_exit_is_not_child_closure": True,
                },
                kind="probe_status",
            )
            return self.writer.close(
                complete,
                reason,
                {
                    **self.resources(),
                    "required_source_failures": self.failed_required,
                    "ebpf_failures": self.ebpf_failures,
                    "ebpf_degraded": bool(self.ebpf_event_loss),
                    "ebpf_event_loss": self.ebpf_event_loss,
                    "isolation_failures": self.isolation_failures,
                    "controller_children_closed": controller,
                    "core": closure,
                },
            )
        finally:
            if self.ebpf and self.ebpf.child and self.ebpf.running():
                self.ebpf.stop(self.writer, self.ebpf_row)
            os.close(self.pidfd)
