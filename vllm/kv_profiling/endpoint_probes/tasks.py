# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-thread CPU and scheduling of the endpoint's serving processes, and the
host's network interrupt and softirq load per CPU.

Read-only, from /proc and /sys, outside the serving processes. Every counter is
the kernel's own cumulative value (the reader derives rates); a value the kernel
does not expose (a kernel without schedstats, a virtual NIC without MSI
interrupts) is reported unavailable, never as zero.
"""

import os
from pathlib import Path

from .netlink import ObservationError

TICKS_PER_S = os.sysconf("SC_CLK_TCK")
_PROC_LIMIT = 4 << 20  # /proc/interrupts grows with CPUs and IRQs


def _after_comm(stat: str) -> list[str]:
    """The fields of a stat file after the command (field 3 onward); the command
    may hold spaces and parentheses, so the split is after its final ')'."""
    return stat.rsplit(")", 1)[1].split()


def _comm(stat: str) -> str:
    return stat[stat.index("(") + 1 : stat.rindex(")")]


_STATUS_LIMIT = 64 << 10  # status grows with supplementary groups and CPU lists
_OPEN = os.O_RDONLY | os.O_CLOEXEC
_GONE = (FileNotFoundError, ProcessLookupError)  # exited (ESRCH on a pinned fd)
MAX_LISTED = 32  # missing interrupts kept per observation, beside an exact count


def _read_at(directory: int, name: str, limit: int = 4096) -> str:
    """One file under a pinned /proc directory. A /proc/<pid> (or task/<tid>)
    directory descriptor stays bound to the task it was opened for: once that
    task exits, reads fail with ESRCH instead of reaching a task that reused
    the ID, so every counter read through it belongs to one generation."""
    fd = os.open(name, _OPEN, dir_fd=directory)
    try:
        data = os.read(fd, limit + 1)
    finally:
        os.close(fd)
    if len(data) > limit:
        raise ObservationError(f"oversize {name}")
    return data.decode()


def _ctxt_switches(status: str) -> tuple[int | None, int | None]:
    values = {}
    for line in status.splitlines():
        name, _, value = line.partition(":")
        if name in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"):
            values[name] = int(value)
    return (
        values.get("voluntary_ctxt_switches"),
        values.get("nonvoluntary_ctxt_switches"),
    )


def _optional(directory: int, name: str, limit: int = 4096) -> str:
    """A file a kernel may not provide (schedstat without schedstats). Under a
    pinned directory, a task that exited also fails lookups with ENOENT, so the
    pinned stat is reread: it fails too (exited) or the file is unavailable."""
    try:
        return _read_at(directory, name, limit)
    except FileNotFoundError:
        try:
            _read_at(directory, "stat")
        except _GONE as gone:
            raise ProcessLookupError(f"task exited reading {name}") from gone
        raise


def _thread(tasks_fd: int, tid: int) -> dict:
    """One thread, every counter read through the same pinned directory."""
    fd = os.open(str(tid), _OPEN | os.O_DIRECTORY, dir_fd=tasks_fd)
    try:
        stat = _read_at(fd, "stat")
        f = _after_comm(stat)
        row = {
            "tid": tid,
            "comm": _comm(stat),
            "start_ticks": int(f[19]),
            "processor": int(f[36]),
            "utime_ticks": int(f[11]),
            "stime_ticks": int(f[12]),
            "run_ns": None,
            "wait_ns": None,
            "timeslices": None,
            "voluntary_ctxt_switches": None,
            "nonvoluntary_ctxt_switches": None,
        }
        unavailable = []
        try:
            run, wait, slices = map(int, _optional(fd, "schedstat").split()[:3])
            row.update(run_ns=run, wait_ns=wait, timeslices=slices)
        except ProcessLookupError:
            raise
        except (OSError, ValueError, ObservationError) as error:
            # Absent (a kernel without schedstats) is unavailable; anything
            # else is an error the caller reports.
            unavailable.append({"file": "schedstat", "error": type(error).__name__})
        try:
            voluntary, involuntary = _ctxt_switches(
                _optional(fd, "status", _STATUS_LIMIT)
            )
            row.update(
                voluntary_ctxt_switches=voluntary,
                nonvoluntary_ctxt_switches=involuntary,
            )
        except ProcessLookupError:
            raise
        except (OSError, ValueError, ObservationError) as error:
            unavailable.append({"file": "status", "error": type(error).__name__})
        if unavailable:
            row["unavailable"] = unavailable
        return row
    finally:
        os.close(fd)


def _list_tasks(tasks_fd: int) -> list[int]:
    return sorted(int(t) for t in os.listdir(tasks_fd))


def _msi_list(path: Path) -> list[int]:
    return sorted(int(entry) for entry in os.listdir(path))


def process_threads(pid: int, start_ticks: int, max_threads: int = 256, proc=None):
    """Every thread of one serving process: its name, last CPU, user and system
    CPU ticks, scheduler run and run-queue wait time, timeslices and context
    switches (all cumulative). The process and each thread are pinned by their
    /proc directory and the process generation is checked through the pin, so a
    reused PID or TID is never read as the serving process or thread. A serving
    process that exited is an observation (``exited``), not a failure; a thread
    that exits mid-sample is counted in ``vanished_between_reads``; one that
    cannot be read (permissions) in ``unreadable``."""
    exited = {"pid": pid, "exited": True, "threads": [], "coverage_problem": None}
    try:
        process_fd = os.open(Path(proc or "/proc") / str(pid), _OPEN | os.O_DIRECTORY)
    except _GONE:
        return exited
    try:
        try:
            if int(_after_comm(_read_at(process_fd, "stat"))[19]) != start_ticks:
                return exited
            tasks_fd = os.open("task", _OPEN | os.O_DIRECTORY, dir_fd=process_fd)
        except _GONE:
            return exited
        try:
            try:
                tids = _list_tasks(tasks_fd)
            except _GONE:
                tids = []
            threads, vanished, unreadable = [], 0, {}
            for tid in tids[:max_threads]:
                try:
                    threads.append(_thread(tasks_fd, tid))
                except _GONE:
                    vanished += 1
                except (OSError, ValueError, ObservationError) as error:
                    name = type(error).__name__
                    unreadable[name] = unreadable.get(name, 0) + 1
        finally:
            os.close(tasks_fd)
        if not threads and not unreadable:
            # An empty or vanished listing: the process may have exited after
            # its task directory was opened. Confirm through the pin.
            try:
                _read_at(process_fd, "stat")
            except _GONE:
                return exited
    finally:
        os.close(process_fd)
    file_errors: dict[str, int] = {}
    for row in threads:
        for item in row.get("unavailable", ()):
            if item["error"] != "FileNotFoundError":
                key = f"{item['file']}:{item['error']}"
                file_errors[key] = file_errors.get(key, 0) + 1
    problems = []
    if unreadable:
        problems.append(f"{sum(unreadable.values())} threads unreadable")
    if file_errors:
        problems.append(f"{sum(file_errors.values())} thread files unreadable")
    if not threads and not vanished and not unreadable:
        problems.append("no thread listed for a live process")
    return {
        "pid": pid,
        "exited": False,
        "clock_ticks_per_s": TICKS_PER_S,
        "threads": threads,
        "threads_total": len(tids),
        "truncated": len(tids) > max_threads,
        "vanished_between_reads": vanished,
        "unreadable": unreadable,
        "unreadable_files": file_errors,
        "coverage_problem": "; ".join(problems) or None,
        "units": {
            "utime_ticks": "clock ticks, cumulative",
            "stime_ticks": "clock ticks, cumulative",
            "run_ns": "ns on CPU, cumulative",
            "wait_ns": "ns runnable but waiting for a CPU, cumulative",
            "timeslices": "count, cumulative",
            "voluntary_ctxt_switches": "count, cumulative",
            "nonvoluntary_ctxt_switches": "count, cumulative",
        },
    }


def _read_proc(path: Path) -> str:
    with path.open("rb") as stream:
        data = stream.read(_PROC_LIMIT + 1)
    if len(data) > _PROC_LIMIT:
        raise ObservationError(f"oversize {path}")
    return data.decode()


def _cpu_ids(header: str, path: Path) -> list[int]:
    """The CPU IDs a /proc/interrupts or /proc/softirqs header names, in column
    order (offline CPUs have no column, so a column index is not a CPU ID)."""
    labels = header.split()
    if not labels or any(
        not (label.startswith("CPU") and label[3:].isdigit()) for label in labels
    ):
        raise ObservationError(f"unrecognized CPU header in {path}")
    return [int(label[3:]) for label in labels]


def _counts(values: list[str], cpus: list[int], what: str) -> list[int]:
    if len(values) < len(cpus) or not all(v.isdigit() for v in values[: len(cpus)]):
        raise ObservationError(f"{what} does not match the CPU header")
    return [int(v) for v in values[: len(cpus)]]


def nic_interrupts(
    interfaces=(),
    rdma_devices=(),
    max_irqs: int = 256,
    max_cpus_per_irq: int = 32,
    proc=None,
    sys=None,
):
    """The endpoint NICs' interrupts per CPU ID (the device's MSI interrupts,
    from sysfs), cumulative, busiest CPUs first. A device without an MSI list
    (virtual, not PCI) or with an empty one is named in ``unavailable``; an
    interrupt sysfs lists but /proc/interrupts does not show is in ``missing``
    (missing coverage, a source failure). Host scope: /proc/interrupts is not
    namespaced."""
    proc, sys = Path(proc or "/proc"), Path(sys or "/sys")
    owners: dict[int, list[str]] = {}
    unavailable, unreadable, absent = [], [], []
    for kind, name, device in [
        *(("interface", n, sys / "class/net" / n) for n in interfaces),
        *(("rdma_device", n, sys / "class/infiniband" / n) for n in rdma_devices),
    ]:
        try:
            listed = _msi_list(device / "device/msi_irqs")
        except FileNotFoundError:
            if device.exists():  # a device without MSI (loopback, virtual)
                unavailable.append({kind: name, "reason": "no MSI interrupt list"})
            else:  # the configured device does not exist
                absent.append({kind: name})
            continue
        except (OSError, ValueError) as error:
            # Present but not readable (permissions): an error, not absence.
            unreadable.append({kind: name, "error": type(error).__name__})
            continue
        if not listed:
            unavailable.append({kind: name, "reason": "empty MSI interrupt list"})
        for irq in listed:
            owners.setdefault(irq, []).append(f"{kind}:{name}")
    path = proc / "interrupts"
    lines = _read_proc(path).splitlines()
    cpus = _cpu_ids(lines[0] if lines else "", path) if owners else []
    irqs, seen = [], set()
    for line in lines[1:] if owners else ():
        label, _, rest = line.partition(":")
        if not label.strip().isdigit() or int(label) not in owners:
            continue
        irq = int(label)
        parts = rest.split()
        counts = _counts(parts, cpus, f"interrupt {irq}")
        seen.add(irq)
        nonzero = sorted(
            ((cpu, count) for cpu, count in zip(cpus, counts) if count),
            key=lambda item: -item[1],
        )
        irqs.append(
            {
                "irq": irq,
                "owners": owners[irq],
                "name": " ".join(parts[len(cpus) :]),
                "total": sum(counts),
                "per_cpu": {
                    str(cpu): count for cpu, count in nonzero[:max_cpus_per_irq]
                },
                "nonzero_cpus": len(nonzero),
                "per_cpu_truncated": len(nonzero) > max_cpus_per_irq,
            }
        )
    missing = [
        {"irq": irq, "owners": owners[irq]} for irq in sorted(set(owners) - seen)
    ]
    problems = []
    if missing:
        problems.append(f"{len(missing)} listed interrupts absent from {path}")
    if unreadable:
        problems.append(f"{len(unreadable)} MSI interrupt lists unreadable")
    if absent:
        problems.append(f"{len(absent)} configured devices absent")
    return {
        "cpu_count": len(cpus),
        "irqs": irqs[:max_irqs],
        "irqs_truncated": len(irqs) > max_irqs,
        "unavailable": unavailable,
        "unreadable": unreadable,
        "absent": absent,
        "missing": missing[:MAX_LISTED],
        "missing_count": len(missing),
        "missing_truncated": len(missing) > MAX_LISTED,
        "coverage_problem": "; ".join(problems) or None,
        "host_scope": True,
    }


def net_softirqs(cpus_per_item: int = 32, proc=None):
    """The host's network softirqs per CPU ID (NET_RX, NET_TX), cumulative, in
    items of at most ``cpus_per_item`` CPUs so a row stays bounded on any CPU
    count. A softirq the kernel does not list is in ``missing``."""
    path = Path(proc or "/proc") / "softirqs"
    lines = _read_proc(path).splitlines()
    cpus = _cpu_ids(lines[0] if lines else "", path)
    items, found = [], set()
    for line in lines[1:]:
        name, _, rest = line.partition(":")
        name = name.strip()
        if name not in ("NET_RX", "NET_TX"):
            continue
        values = rest.split()
        if len(values) != len(cpus):
            raise ObservationError(f"softirq {name} does not match the CPU header")
        counts = _counts(values, cpus, f"softirq {name}")
        found.add(name)
        for first in range(0, len(counts), cpus_per_item):
            items.append(
                {
                    "name": name,
                    "cpus": cpus[first : first + cpus_per_item],
                    "counts": counts[first : first + cpus_per_item],
                }
            )
    missing = sorted({"NET_RX", "NET_TX"} - found)
    return {
        "cpu_count": len(cpus),
        "softirqs": items,
        "missing": missing,
        "coverage_problem": f"absent from {path}: {', '.join(missing)}"
        if missing
        else None,
        "host_scope": True,
    }
