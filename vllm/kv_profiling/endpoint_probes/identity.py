# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded worker discovery and optional socket ownership, with PID reuse checks."""

import os
from pathlib import Path

from .health import core_completion as core_completion
from .health import expected_problems, recorder_counts, side_problems
from .netdev import read_small
from .netlink import ObservationError


class NamespacesUnreadable(ObservationError):
    """The kernel refused a serving process's namespace links to this observer."""


def _stat_fields(pid):
    # comm may contain spaces and parentheses; fields after its final ')' start at 3.
    return read_small(Path("/proc") / str(pid) / "stat").rsplit(")", 1)[1].split()


def process_identity(pid):
    base = Path("/proc") / str(pid)
    fields = _stat_fields(pid)
    try:
        net, pid_ns = os.readlink(base / "ns/net"), os.readlink(base / "ns/pid")
    except PermissionError as exc:
        # the kernel's ptrace access check: the target's capabilities must be a
        # subset of this observer's (or it holds CAP_SYS_PTRACE), and the target
        # must be dumpable
        raise NamespacesUnreadable(
            f"the kernel refused {exc.filename} (errno {exc.errno}, "
            f"{os.strerror(exc.errno)}): serving PID {pid} holds capabilities the "
            "observer lacks, or is not dumpable; give the observer the same added "
            "capabilities, or CAP_SYS_PTRACE"
        ) from exc
    return {
        "pid": pid,
        "start_ticks": int(fields[19]),
        "net_namespace": net,
        "pid_namespace": pid_ns,
    }


def endpoint_owner(frontends):
    """The process the observer follows, from its own /proc (it shares the
    serving PID namespace): the frontend itself with one API server, else the
    API servers' one common parent, the serving process that runs until the
    end (data parallelism runs an API server per DP rank, each a child).

    Returns the owner and the evidence it was derived from: each frontend's
    parent as read, so a saved capture's owner can be checked offline."""
    if len(frontends) == 1:
        owner = {"pid": frontends[0]["pid"], "start_ticks": frontends[0]["start_ticks"]}
        return owner, {"frontends": [], "parent": None}
    seen = []
    for frontend in frontends:
        fields = _stat_fields(frontend["pid"])
        if int(fields[19]) != frontend["start_ticks"]:
            raise ValueError("an API server's process changed before the observer")
        seen.append(
            {
                "pid": frontend["pid"],
                "start_ticks": frontend["start_ticks"],
                "parent_pid": int(fields[1]),
            }  # fmt: skip
        )
    parents = {entry["parent_pid"] for entry in seen}
    if len(parents) != 1:
        raise ValueError("the API servers have no single parent process")
    (parent,) = parents
    owner = {"pid": parent, "start_ticks": int(_stat_fields(parent)[19])}
    return owner, {"frontends": seen, "parent": owner}


def discover(input_dir, run_id, expected, endpoint_role):
    """The endpoint's recorders and owner, found by the observer itself in the
    shared profiles directory and PID namespace: exactly the ``expected``
    frontend, scheduler and worker recorders, all of this side's KV role,
    frozen before it is ready. Returns (inventory, owner, owner evidence)."""
    from .health import inventory, validate_inventory

    if problems := expected_problems(expected):
        raise ValueError(problems[0])
    entries = inventory(input_dir, run_id)
    validate_inventory(entries, run_id)
    if recorder_counts(entries) != expected:
        raise ValueError(
            f"the endpoint's recorders {recorder_counts(entries)} are not the "
            f"expected {expected}"
        )
    if wrong := side_problems(entries, endpoint_role):
        raise ValueError(wrong[0])
    owner, evidence = endpoint_owner(
        [e for e in entries if e["context"]["role"] == "frontend"]
    )
    return entries, owner, evidence


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
