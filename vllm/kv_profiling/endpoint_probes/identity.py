# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded worker discovery and optional socket ownership, with PID reuse checks."""

import json
import os
from pathlib import Path

from .health import core_completion as core_completion
from .netdev import read_small
from .netlink import ObservationError


def process_identity(pid):
    base = Path("/proc") / str(pid)
    # comm may contain spaces and parentheses; fields after its final ')' start at 3.
    fields = read_small(base / "stat").rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "start_ticks": int(fields[19]),
        "net_namespace": os.readlink(base / "ns/net"),
        "pid_namespace": os.readlink(base / "ns/pid"),
    }


def ownership(directory, run_id, endpoint_role, namespace, observer_cpus=()):
    """Optional fd lookup; mandatory CPU placement is checked independently."""
    owners, unavailable, identities = {}, [], []
    paths = Path(directory).glob(f"{run_id}/*/events.jsonl")
    for count, path in enumerate(paths):
        if count >= 256:
            raise ObservationError("worker registry limit")
        with path.open() as stream:
            line = stream.readline(16385)
        if len(line) > 16384:
            raise ObservationError("worker metadata limit")
        meta = json.loads(line)
        context = meta.get("context", {})
        if meta.get("run_id") != run_id or context.get("role") != "worker":
            continue
        expected_role = "kv_producer" if endpoint_role == "prefill" else "kv_consumer"
        if context.get("kv_role") not in (expected_role, "kv_both"):
            raise ObservationError("worker endpoint role mismatch")
        if (
            not isinstance(context.get("engine_id"), str)
            or type(context.get("rank")) is not int
        ):
            raise ObservationError("worker engine/rank identity missing")
        try:
            identity = process_identity(meta["pid"])
            if any(identity[key] != meta.get(key) for key in identity):
                raise ObservationError("worker PID generation/namespace mismatch")
            if identity["net_namespace"] != namespace:
                raise ObservationError("worker network namespace mismatch")
            identity.update(engine_id=context["engine_id"], rank=context["rank"])
            identities.append(identity)
            entries = (Path("/proc") / str(meta["pid"]) / "fd").iterdir()
            for index, entry in enumerate(entries):
                if index >= 8192:
                    raise ObservationError("worker fd limit")
                try:
                    target = os.readlink(entry)
                except FileNotFoundError:
                    continue
                if target.startswith("socket:["):
                    owners.setdefault(int(target[8:-1]), []).append(identity)
            if process_identity(meta["pid"]) != {
                key: identity[key]
                for key in ("pid", "start_ticks", "net_namespace", "pid_namespace")
            }:
                raise ObservationError("worker changed during ownership read")
        except (OSError, ValueError, KeyError, ObservationError) as error:
            unavailable.append(
                {"process_id": meta.get("process_id"), "error": str(error)}
            )
            # A failed snapshot must not certify any inode from that process.
            owners = {
                inode: [item for item in items if item["pid"] != meta.get("pid")]
                for inode, items in owners.items()
            }
    return owners, unavailable, identities


def placement(expected, cpus, namespace, verified):
    """Check every live serving thread independently of optional fd visibility."""
    problems, snapshots = [], []
    boot = read_small("/proc/sys/kernel/random/boot_id")
    for meta in expected:
        pid, process = meta["pid"], meta["process_id"]
        try:
            if meta["boot_id"] != boot:
                raise ObservationError("serving boot identity mismatch")
            actual = process_identity(pid)
            if any(actual[key] != meta.get(key) for key in actual):
                raise ObservationError("serving PID generation/namespace mismatch")
            if actual["net_namespace"] != namespace:
                raise ObservationError("serving network namespace mismatch")
            tids = list((Path("/proc") / str(pid) / "task").iterdir())
            if not tids or len(tids) > 8192:
                raise ObservationError("serving thread inventory unavailable/limit")
            for tid in tids:
                try:
                    affinity = os.sched_getaffinity(int(tid.name))
                except ProcessLookupError:
                    if not tid.exists():
                        continue
                    raise
                if set(cpus) & affinity:
                    raise ObservationError(
                        f"serving thread {tid.name} overlaps observer CPUs"
                    )
            if process_identity(pid) != actual:
                raise ObservationError("serving process changed during placement check")
            verified.add(process)
            snapshots.append(
                {"process_id": process, "threads": len(tids), "verified": True}
            )
        except FileNotFoundError as error:
            exited = not (Path("/proc") / str(pid)).exists()
            if not exited:
                try:
                    fields = (
                        read_small(Path("/proc") / str(pid) / "stat")
                        .rsplit(")", 1)[1]
                        .split()
                    )
                    exited = (
                        fields[0] in ("Z", "X")
                        and int(fields[19]) == meta["start_ticks"]
                    )
                except (OSError, ValueError, IndexError):
                    pass
            if process in verified and exited:
                snapshots.append({"process_id": process, "exited": True})
            else:
                problems.append(f"{process}: placement unverified: {error}")
        except (OSError, ValueError, ObservationError) as error:
            problems.append(f"{process}: {error}")
    return {"verified": not problems, "problems": problems, "snapshots": snapshots}
