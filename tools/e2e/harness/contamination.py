"""A simple, cheap monitor of foreign CPU use on the measured CPUs, and the rule that judges a run with it.

A latency comparison is only worth reading if nothing but the campaign's own containers used the measured CPUs
(0-13) while the benchmark ran. The monitor is deliberately small: one stdlib process on a CPU outside the measured
set (15) that, once a second, reads the per-CPU lines of ``/proc/stat`` and the ``cpu.stat`` of the owned containers'
cgroups, and timestamps both readings. **Nothing here scans ``/proc`` per process** (a richer sampler exceeded its own
read bound under load and blocked a whole campaign).

*Foreign busy* of an interval is the CPU time the measured CPUs spent in user, nice and system mode (``/proc/stat``)
minus the owned containers' user and system time (their cgroups): time spent by something that is not ours. Interrupt
and softirq time is reported, never counted (our own KV traffic produces it and the cgroups do not account it).

The subtraction is of two readings taken at slightly different instants, each with its own resolution, so the judgement
carries its uncertainty and does not hide it. For every interval the estimate gets a bound made of (1) the timing
mismatch of the two readings at both ends times the owned capacity (the owned cgroups cannot have used more CPU in the
mismatch than their CPUs allow) and (2) the tick quantization of ``/proc/stat`` and of the cgroup accounting. A threshold
decision is *exceeded* only if the estimate minus its bound is over the threshold, *clear* only if the estimate plus its
bound is under it, and otherwise **unknown**; an unknown can never make a run clean. Accounting that is inconsistent
beyond its own bound (owned usage above the host's busy time) is unknown, never clamped into evidence of cleanliness.
Missing or reset counters, lost coverage of either end of the client's benchmark, owned cgroups that are not disjoint,
whose identities changed or whose effective CPU sets are not inside 0-13 are unknown as well."""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

MEASURED_CPUS = tuple(range(14))
MONITOR_CPU = 15
INTERVAL_S = 1.0
MEAN_THRESHOLD_CPU = 0.20
SAMPLE_THRESHOLD_CPU = 0.75
GAP_S = 3.0
SLOW_READ_MS = 250.0
SLOW_FRACTION = 0.10
TICK_S = 1.0 / os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 0.01
CGROUP_ROOT = "/sys/fs/cgroup/system.slice"
CONSTANTS = {"measured_cpus": [0, 13], "monitor_cpu": MONITOR_CPU, "interval_s": INTERVAL_S,
             "mean_threshold_cpu": MEAN_THRESHOLD_CPU, "sample_threshold_cpu": SAMPLE_THRESHOLD_CPU,
             "gap_s": GAP_S, "slow_read_ms": SLOW_READ_MS,
             "slow_fraction": SLOW_FRACTION}

# The sampler process: stdlib only, run pinned to MONITOR_CPU. Arguments: the output file, the file listing the owned
# cgroups, the interval. It stops when ``<out>.stop`` exists. Each record holds both readings' instants (begin and end of
# each read, on CLOCK_MONOTONIC) and the raw counters; nothing is interpreted here.
SAMPLER = r"""
import json, os, sys, time

out, owned, interval = sys.argv[1], json.load(open(sys.argv[2])), float(sys.argv[3])
cpus = set(range(14))
parent = int(sys.argv[4]) if len(sys.argv) > 4 else os.getppid()  # the starter's pid, given by the starter: no race with its death
tick = time.monotonic()
index = 0
with open(out, "a", buffering=1) as handle:
    while True:
        if os.getppid() != parent:
            sys.exit(3)  # orphaned: whoever started it is gone; no stop request will come
        stopping = os.path.exists(out + ".stop")  # looked at before the read: the last sample begins after the stop request
        h0 = time.monotonic_ns()
        with open("/proc/stat") as stat:
            lines = stat.read().splitlines()
        h1 = time.monotonic_ns()
        cpu = {}
        for line in lines:
            name = line.split(" ", 1)[0]
            if name.startswith("cpu") and name[3:].isdigit() and int(name[3:]) in cpus:
                cpu[name[3:]] = [int(v) for v in line.split()[1:9]]
        c0 = time.monotonic_ns()
        cg = {}
        for cid, path in owned.items():
            try:
                with open(path + "/cpu.stat") as stat:
                    fields = dict(line.split() for line in stat.read().splitlines())
                cg[cid] = {k: int(fields[k]) for k in ("user_usec", "system_usec", "nr_throttled")}
            except (OSError, KeyError, ValueError):
                cg[cid] = None
        c1 = time.monotonic_ns()
        handle.write(json.dumps({"i": index, "h": [h0, h1], "c": [c0, c1], "cpu": cpu, "cg": cg}) + "\n")
        index += 1
        if stopping:
            break
        tick += interval
        time.sleep(max(0.0, tick - time.monotonic()))
"""


# ------------------------------------------------------------------------------------------------ the judgement
def _num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _cpu_set(text: str) -> set[int]:
    found: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        low, _, high = part.partition("-")
        found.update(range(int(low), int(high or low) + 1))
    return found


def _snapshot_problems(snapshot, label: str) -> list[str]:
    """Why one snapshot of the owned containers cannot be trusted: empty, malformed, nested or shared cgroups, an
    effective CPU set that is unreadable, empty or reaches outside the measured CPUs. Each snapshot is judged on its own."""
    if not isinstance(snapshot, dict) or not snapshot:
        return [f"{label}: no owned container"]
    problems = []
    paths = []
    for cid, entry in sorted(snapshot.items()):
        if not (isinstance(entry, dict) and all(isinstance(entry.get(k), str) for k in ("path", "started", "cpus"))):
            problems.append(f"{label}: container {str(cid)[:12]} has a malformed record")
            continue
        paths.append(entry["path"].rstrip("/") + "/")
        try:
            effective = _cpu_set(entry["cpus"])
        except ValueError:
            problems.append(f"{label}: container {cid[:12]}: unreadable effective CPU set {entry['cpus']!r}")
            continue
        if not effective or effective - set(MEASURED_CPUS):
            problems.append(f"{label}: container {cid[:12]}: effective CPUs {entry['cpus']!r} are empty or not inside 0-13")
    if len(set(paths)) != len(paths):
        problems.append(f"{label}: two owned containers share a cgroup")
    ordered = sorted(set(paths))
    for a, b in zip(ordered, ordered[1:]):
        if b.startswith(a):
            problems.append(f"{label}: owned cgroups are not disjoint ({a} contains {b})")
    return problems


def owned_problems(start: dict, end: dict) -> list[str]:
    """Why the owned containers cannot be trusted over the window: either snapshot invalid on its own, or the two differ
    (a container added or gone, a changed id, start instant, cgroup path or effective CPU assignment). ``start`` and ``end``
    map a container ID to ``{"path", "started", "cpus"}``."""
    problems = _snapshot_problems(start, "at the start") + _snapshot_problems(end, "at the end")
    if problems:
        return problems
    if set(start) != set(end):
        problems.append("the owned containers are not the same at both ends of the run")
    for cid in sorted(set(start) & set(end)):
        a, b = start[cid], end[cid]
        if a["started"] != b["started"]:
            problems.append(f"container {cid[:12]} was restarted during the run")
        if a["path"] != b["path"]:
            problems.append(f"container {cid[:12]} changed cgroup ({a['path']} -> {b['path']})")
        if _cpu_set(a["cpus"]) != _cpu_set(b["cpus"]):
            problems.append(f"container {cid[:12]} changed effective CPUs ({a['cpus']} -> {b['cpus']})")
    return problems


def validate_samples(samples) -> list[str]:
    """Why the recording cannot be admitted: a record that is not well formed, a missing or duplicate sequence entry (the
    sampler numbers its records 0, 1, 2 ... in order, so the file must hold exactly that), invalid read instants or
    instants that go backwards. Nothing is sorted or dropped to repair a capture."""
    problems = []
    if not isinstance(samples, list) or not samples:
        return ["no samples"]
    previous_end = None
    for position, sample in enumerate(samples):
        if not (isinstance(sample, dict) and all(k in sample for k in ("i", "h", "c", "cpu", "cg"))):
            problems.append(f"record {position} is malformed")
            continue
        if type(sample["i"]) is not int or sample["i"] != position:
            problems.append(f"record {position} carries sequence {sample['i']!r} (missing or duplicate entries)")
        ok = True
        for name in ("h", "c"):
            read = sample[name]
            if not (isinstance(read, list) and len(read) == 2 and all(type(v) is int for v in read) and 0 <= read[0] <= read[1]):
                problems.append(f"record {position}: invalid {name} read instants {read!r}")
                ok = False
        if not isinstance(sample["cpu"], dict) or not isinstance(sample["cg"], dict):
            problems.append(f"record {position}: counters are not mappings")
            ok = False
        if ok:
            if sample["h"][1] > sample["c"][0]:
                problems.append(f"record {position}: the cgroup reading began before the host reading ended")
            if previous_end is not None and sample["h"][0] < previous_end:
                problems.append(f"record {position}: read instants go backwards")
            previous_end = sample["c"][1]
        if len(problems) >= 20:
            problems.append("further problems not listed")
            break
    return problems


def lifecycle_problems(lifecycle) -> list[str]:
    """Why the sampler's shutdown is not evidence of a sound recording: no record of it, a failed start, a non-zero exit, a
    forced kill, or a process that was not reaped."""
    if not isinstance(lifecycle, dict):
        return ["no sampler termination evidence"]
    problems = []
    if lifecycle.get("started") is not True:
        problems.append("the sampler never started")
    if lifecycle.get("returncode") != 0:
        problems.append(f"the sampler exited with status {lifecycle.get('returncode')!r}")
    if lifecycle.get("forced_kill"):
        problems.append("the sampler had to be killed")
    if lifecycle.get("reaped") is not True:
        problems.append("the sampler was not confirmed reaped")
    return problems


def _mismatch_s(sample: dict) -> float:
    """The worst-case offset between the instants at which the host and the owned counters of one sample were read."""
    begin = min(sample["h"][0], sample["c"][0])
    end = max(sample["h"][1], sample["c"][1])
    return (end - begin) / 1e9


def _mid(read) -> float:
    return (read[0] + read[1]) / 2e9


CPU_FIELDS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
CG_FIELDS = ("user_usec", "system_usec", "nr_throttled")


def _busy_rows(sample: dict):
    """The measured CPUs' eight counters each, as ``{cpu: [8 ints]}``; None if a CPU is missing or a counter malformed."""
    rows = {}
    for cpu in MEASURED_CPUS:
        row = sample["cpu"].get(str(cpu))
        if not (isinstance(row, list) and len(row) == 8 and all(type(v) is int and v >= 0 for v in row)):
            return None
        rows[cpu] = row
    return rows


def _cg_rows(sample: dict, ids: list[str]):
    """The owned cgroups' counters each, as ``{id: {field: int}}``; None if one is missing or malformed."""
    rows = {}
    for cid in ids:
        row = sample["cg"].get(cid)
        if not (isinstance(row, dict) and all(type(row.get(k)) is int and row[k] >= 0 for k in CG_FIELDS)):
            return None
        rows[cid] = row
    return rows


def _resets(a_cpu, b_cpu, a_cg, b_cg) -> list[str]:
    """Every counter that went backwards between two readings, found one by one BEFORE anything is summed (a fall in one
    counter offset by a rise in another must not pass)."""
    found = []
    for cpu in MEASURED_CPUS:
        for name, before, after in zip(CPU_FIELDS, a_cpu[cpu], b_cpu[cpu]):
            if after < before:
                found.append(f"CPU {cpu} {name} fell from {before} to {after}")
    for cid in a_cg:
        for name in CG_FIELDS:
            if b_cg[cid][name] < a_cg[cid][name]:
                found.append(f"cgroup {cid[:12]} {name} fell from {a_cg[cid][name]} to {b_cg[cid][name]}")
    return found


def _decide(rate: float, bound: float, threshold: float) -> str:
    if rate - bound > threshold:
        return "exceeded"
    if rate + bound < threshold:
        return "clear"
    return "unknown"


def judge(samples: list[dict], window: tuple[int, int], start: dict, end: dict, *, lifecycle=None, read_errors=(),
          mean_threshold: float = MEAN_THRESHOLD_CPU, sample_threshold: float = SAMPLE_THRESHOLD_CPU) -> dict:
    """Judge one window: ``samples`` are the sampler's records, ``window`` the client benchmark's own begin and end
    (CLOCK_MONOTONIC nanoseconds, bracketing the whole benchmark command), ``start``/``end`` the owned containers'
    identities at both ends, ``lifecycle`` the sampler's termination evidence (``Sampler.stop``), ``read_errors`` what the
    reader could not parse. Returns ``{"status": clean | contaminated | unknown, "reasons": [...], ...}``. Evidence that
    cannot be admitted (malformed records, an untrustworthy owned set, an unconfirmed sampler shutdown) is unknown, whatever
    the readings would say."""
    reasons: list[str] = [*read_errors]
    exceeded: list[str] = []
    reasons += validate_samples(samples) if not read_errors else []
    reasons += lifecycle_problems(lifecycle)
    reasons += owned_problems(start, end)
    if reasons:
        # Without admissible evidence the subtraction means nothing: neither clean nor contaminated can be said.
        return {"status": "unknown", "reasons": reasons, "intervals": 0}
    ids = sorted(start)
    capacity = 0.0
    for entry in start.values():
        try:
            effective = _cpu_set(entry["cpus"])
        except ValueError:
            continue
        if effective <= set(MEASURED_CPUS):
            capacity += len(effective)
    capacity = min(float(len(MEASURED_CPUS)), capacity)
    begin_ns, end_ns = window
    ordered = samples
    before = [i for i, s in enumerate(ordered) if max(s["h"][1], s["c"][1]) <= begin_ns]
    after = [i for i, s in enumerate(ordered) if min(s["h"][0], s["c"][0]) >= end_ns]
    if not before or not after:
        reasons.append("the samples do not bracket both ends of the client's benchmark")
        return {"status": "unknown", "reasons": reasons, "intervals": 0}
    low, high = before[-1], after[0]
    span = ordered[low:high + 1]
    if len(span) < 2:
        return {"status": "unknown", "reasons": reasons + ["fewer than two samples cover the window"], "intervals": 0}
    gaps = [_mid(b["h"]) - _mid(a["h"]) for a, b in zip(span, span[1:])]
    if max(gaps) > GAP_S:
        reasons.append(f"a gap of {max(gaps):.2f} s between samples inside the window")
    slow = [s for s in span if (max(s["h"][1], s["c"][1]) - min(s["h"][0], s["c"][0])) / 1e6 > SLOW_READ_MS]
    if len(slow) > SLOW_FRACTION * len(span):
        reasons.append(f"{len(slow)} of {len(span)} samples took more than {SLOW_READ_MS:g} ms to read")
    rows = []
    for a, b in zip(span, span[1:]):
        ha, hb, oa, ob = _busy_rows(a), _busy_rows(b), _cg_rows(a, ids), _cg_rows(b, ids)
        if None in (ha, hb):
            reasons.append("a measured CPU is missing or malformed in the readings")
            continue
        if None in (oa, ob):
            reasons.append("an owned cgroup could not be read")
            continue
        resets = _resets(ha, hb, oa, ob)
        if resets:
            reasons.append(f"a counter went backwards (a reset): {resets[0]}" + (f" and {len(resets) - 1} more" if len(resets) > 1 else ""))
            continue
        ha = (sum(r[0] + r[1] + r[2] for r in ha.values()) * TICK_S, sum(r[4] + r[5] + r[6] + r[7] for r in ha.values()) * TICK_S)
        hb = (sum(r[0] + r[1] + r[2] for r in hb.values()) * TICK_S, sum(r[4] + r[5] + r[6] + r[7] for r in hb.values()) * TICK_S)
        oa = (sum(r["user_usec"] + r["system_usec"] for r in oa.values()) / 1e6, sum(r["nr_throttled"] for r in oa.values()))
        ob = (sum(r["user_usec"] + r["system_usec"] for r in ob.values()) / 1e6, sum(r["nr_throttled"] for r in ob.values()))
        dt = _mid(b["h"]) - _mid(a["h"])
        if dt <= 0:
            reasons.append("samples out of order")
            continue
        foreign = (hb[0] - ha[0]) - (ob[0] - oa[0])
        bound = capacity * (_mismatch_s(a) + _mismatch_s(b)) + 2 * len(MEASURED_CPUS) * TICK_S
        rows.append({"dt": dt, "foreign": foreign, "bound": bound, "other": hb[1] - ha[1],
                     "owned": ob[0] - oa[0], "throttled": ob[1] - oa[1]})
    if not rows:
        return {"status": "unknown", "reasons": reasons + ["no interval could be evaluated"], "intervals": 0}
    verdicts = {"exceeded": 0, "clear": 0, "unknown": 0}
    worst = 0.0
    for row in rows:
        rate, bound = row["foreign"] / row["dt"], row["bound"] / row["dt"]
        row["decision"] = _decide(rate, bound, sample_threshold)
        verdicts[row["decision"]] += 1
        worst = max(worst, rate)
        if rate + bound < 0:
            row["inconsistent"] = True
    if verdicts["exceeded"]:
        exceeded.append(f"{verdicts['exceeded']} one-second sample(s) above {sample_threshold} CPU")
    if verdicts["unknown"]:
        reasons.append(f"{verdicts['unknown']} sample(s) could not be decided against {sample_threshold} CPU")
    if any(row.get("inconsistent") for row in rows):
        reasons.append("owned usage is above the host's busy time beyond the accounting's own bound (inconsistent)")
    total = sum(row["dt"] for row in rows)
    mean = sum(row["foreign"] for row in rows) / total
    mean_bound = (capacity * (_mismatch_s(span[0]) + _mismatch_s(span[-1])) + 2 * len(MEASURED_CPUS) * TICK_S) / total
    mean_decision = _decide(mean, mean_bound, mean_threshold)
    if mean_decision == "exceeded":
        exceeded.append(f"the window's mean foreign busy {mean:.3f} CPU is above {mean_threshold} CPU")
    elif mean_decision == "unknown":
        reasons.append(f"the window's mean foreign busy {mean:.3f} +- {mean_bound:.3f} CPU cannot be decided against {mean_threshold} CPU")
    if mean + mean_bound < 0:
        reasons.append("the mean foreign busy is negative beyond its bound (inconsistent accounting)")
    throttled = sum(row["throttled"] for row in rows)
    if throttled:
        exceeded.append(f"an owned cgroup was throttled ({throttled:.0f} periods)")
    status = "contaminated" if exceeded else "unknown" if reasons else "clean"
    return {"status": status, "reasons": [*exceeded, *reasons], "intervals": len(rows), "decisions": verdicts,
            "mean_foreign_cpu": mean, "mean_bound_cpu": mean_bound, "max_sample_foreign_cpu": worst,
            "other_cpu_mean": sum(row["other"] for row in rows) / total,
            "owned_cpu_mean": sum(row["owned"] for row in rows) / total, "samples": len(span),
            "slow_reads": len(slow), "window_s": (end_ns - begin_ns) / 1e9, "capacity_cpu": capacity}


# ------------------------------------------------------------------------------------------- host-side helpers
def owned_containers(topology: str) -> dict[str, dict]:
    """The running containers of ``topology`` (the harness labels each one), resolved from Docker's own records:
    ``{id: {"path", "started", "cpus"}}``. The llm-d components and every ``docker exec`` run inside these containers'
    cgroups, so they are owned work."""
    from . import shell
    from .topology import LABEL

    ids = shell.run(["docker", "ps", "-aq", "--no-trunc", "--filter", f"label={LABEL}={topology}"]).split()
    found = {}
    for cid in ids:
        info = shell.run(["docker", "inspect", "-f", "{{.State.StartedAt}}|{{.State.Running}}", cid]).strip()
        started, _, running = info.partition("|")
        path = f"{CGROUP_ROOT}/docker-{cid}.scope"
        try:
            cpus = Path(path, "cpuset.cpus.effective").read_text().strip()
        except OSError:
            cpus = ""
        if running == "true":
            found[cid] = {"path": path, "started": started, "cpus": cpus}
    return found


class Sampler:
    """The sampler process, started and stopped around a window, with the evidence of how it ended."""

    def __init__(self, directory: Path, owned: dict[str, dict], interval_s: float = INTERVAL_S):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.out = self.directory / "samples.jsonl"
        (self.directory / "owned-list.json").write_text(json.dumps({cid: e["path"] for cid, e in owned.items()}))
        self.owned, self.interval_s, self.process = owned, interval_s, None

    def start(self) -> None:
        for name in ("samples.jsonl", "samples.jsonl.stop", "sampler.json"):
            (self.directory / name).unlink(missing_ok=True)
        self.process = subprocess.Popen(
            ["taskset", "-c", str(MONITOR_CPU), sys.executable, "-c", SAMPLER, str(self.out),
             str(self.directory / "owned-list.json"), str(self.interval_s), str(os.getpid())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def wait_first(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.out.exists() and self.out.stat().st_size:
                return
            if self.process is not None and self.process.poll() is not None:
                break
            time.sleep(0.05)
        raise RuntimeError("the contamination sampler wrote no sample")

    def stop(self, timeout_s: float = 10.0) -> dict:
        """Stop it (one more sample begins after the stop request, so the window's end is bracketed), reap it, and save how
        it ended (``sampler.json``): started, exit status, whether it had to be killed, whether it was reaped."""
        (self.directory / "samples.jsonl.stop").touch()
        lifecycle = {"started": self.process is not None, "returncode": None, "forced_kill": False, "reaped": False}
        if self.process is not None:
            try:
                lifecycle["returncode"] = self.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                lifecycle["forced_kill"] = True
                self.process.kill()
                try:
                    lifecycle["returncode"] = self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            lifecycle["reaped"] = self.process.poll() is not None
        (self.directory / "sampler.json").write_text(json.dumps(lifecycle) + "\n")
        return lifecycle


def read_samples(path) -> tuple[list[dict], list[str]]:
    """The recording's records IN FILE ORDER and every line that could not be read (a truncated or malformed record is an
    error to report, never skipped silently: dropping it can turn a contaminated window into a clean one)."""
    rows, errors = [], []
    try:
        text = Path(path).read_text()
    except OSError as exc:
        return [], [f"samples unreadable ({type(exc).__name__})"]
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    elif lines:
        errors.append("the recording ends in a line without its newline (truncated)")
    for number, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except ValueError:
            errors.append(f"record {number} is not valid JSON")
    return rows, errors


def reap_samplers(directory) -> dict:
    """Stop and reap every sampler process whose recording lives under ``directory`` (a block's results): the stop request
    first, then SIGTERM, then SIGKILL. For a block's cleanup after an interrupted run; returns what was found."""
    import signal

    root = str(Path(directory).resolve()) + "/"

    def found():
        pids = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                argv = (entry / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            args = [a.decode("utf-8", "replace") for a in argv]
            if "-c" in args and any(a.startswith(root) and a.endswith("samples.jsonl") for a in args[1:]):
                pids.append(int(entry.name))
        return pids

    report = {"found": found(), "killed": []}
    for stopfile in Path(directory).rglob("samples.jsonl"):
        Path(str(stopfile) + ".stop").touch()
    deadline = time.monotonic() + 5
    while found() and time.monotonic() < deadline:
        time.sleep(0.1)
    for sig, name in ((signal.SIGTERM, "term"), (signal.SIGKILL, "kill")):
        left = found()
        for pid in left:
            try:
                os.kill(pid, sig)
                report["killed"].append([pid, name])
            except OSError:
                pass
        end = time.monotonic() + 3
        while found() and time.monotonic() < end:
            time.sleep(0.1)
    report["remaining"] = found()
    return report
