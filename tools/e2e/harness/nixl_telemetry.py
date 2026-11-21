"""NIXL's built-in telemetry as an L2 source: the expected agents, the boundary samples and the checks.

NIXL exports its counters asynchronously (a 100 ms flush), so an HTTP success proves neither that every agent
has appeared nor that every completed transfer has reached its counters. The collector's boundaries are
therefore defined samples: a *baseline* whose agent set equals the inventory the harness recorded (one agent
for each (DP rank, TP rank) of the node's instance, found through the processes' titles, parents and start
times) with every
required family served, and a *final* sample taken only once the counters have been unchanged for a stretch
that began after the workload, still equal to the inventory, and in agreement with the transfer records.
Everything here is pure (no container, no clock) so each rule can be tested on its own; the orchestration
that feeds it is ``harness.collector``.

An agent is identified by its full series labels (``agent_name`` is a UUID in vLLM and encodes no rank) and
bound to a rank only through the process it lives in: its ``pid`` and that process's start time at both
boundaries. Ambiguous joins, unexpected agents and restarts fail the run; nothing is merged into a rank."""

import json
import math
import re

# The error statuses and fields this harness KNOWS (the series of NIXL's multi-process exporter as the collector records
# them). Nothing here is required: an exporter of another version that renames, drops or adds a series is read for what it
# serves, and what it does not serve is reported unavailable, never a failed check, collector or capture.
STATUSES = ("not_posted", "invalid_param", "backend", "not_found", "mismatch", "not_allowed", "repost_active",
            "unknown", "not_supported", "remote_disconnect", "canceled", "no_telemetry")
FIELDS = ("tx_bytes", "rx_bytes", "tx_requests", "rx_requests", "xfer_time_us", "xfer_post_time_us",
          "memory_registered_bytes", "memory_deregistered_bytes")
# What must not move while the final sample waits for the counters to settle.
WATCHED = ("tx_bytes", "rx_bytes", "tx_requests", "rx_requests", "xfer_time_us", "xfer_post_time_us",
           "events_dropped")
STABLE_SAMPLES = 3
STABLE_SPAN_NS = 500_000_000  # five flush intervals of NIXL's default 100 ms
FINAL_TIMEOUT_S = 15.0

# The vLLM processes of a node (their process titles), with their parents and start times in clock ticks.
PROCESS_TABLE = r"""
import json, os, socket

host = socket.gethostname()
rows = []
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        stat = open(f"/proc/{pid}/stat").read().rpartition(")")[2].split()
    except OSError:
        continue
    title = argv[0].decode("utf-8", "replace") if argv else ""
    if title.startswith("VLLM::"):
        rows.append({"pid": int(pid), "ppid": int(stat[1]), "start": int(stat[19]), "title": title,
                     "host": host})
print(json.dumps(sorted(rows, key=lambda r: r["pid"])))
"""

# The harness's own read of the endpoint, with its instants on the clock the collector's samples use.
OWN_SCRAPE = r"""
import errno, json, socket, sys, time, urllib.error, urllib.request

begin = time.monotonic_ns()
body = error = None
status = None
outcome = "answered"
try:
    response = urllib.request.urlopen(sys.argv[1], timeout=float(sys.argv[2]) if len(sys.argv) > 2 else 5)
    status = response.status
    body = response.read().decode("utf-8", "replace")
except urllib.error.HTTPError as exc:
    outcome, status, error = "http_error", exc.code, repr(exc)
except urllib.error.URLError as exc:
    reason = exc.reason
    error = repr(exc)
    if isinstance(reason, ConnectionRefusedError) or getattr(reason, "errno", None) == errno.ECONNREFUSED:
        outcome = "refused"
    elif isinstance(reason, (socket.timeout, TimeoutError)):
        outcome = "timeout"
    else:
        outcome = "other"
except (socket.timeout, TimeoutError) as exc:
    outcome, error = "timeout", repr(exc)
except Exception as exc:
    outcome, error = "other", repr(exc)
print(json.dumps({"begin_ns": begin, "end_ns": time.monotonic_ns(), "body": body, "error": error,
                  "outcome": outcome, "status": status}))
"""


# The exporter's series names for the fields the collector records (kinds 27 and 28).
METRICS = {
    "agent_tx_bytes_total": "tx_bytes", "agent_rx_bytes_total": "rx_bytes",
    "agent_tx_requests_num_total": "tx_requests", "agent_rx_requests_num_total": "rx_requests",
    "agent_xfer_time_total": "xfer_time_us", "agent_xfer_post_time_total": "xfer_post_time_us",
    "agent_memory_registered_total": "memory_registered_bytes",
    "agent_memory_deregistered_total": "memory_deregistered_bytes",
    "agent_telemetry_events_dropped_total": "events_dropped", "agent_errors_total": "errors",
}


class InventoryError(ValueError):
    """The node's processes do not match the instance the run configured."""


def check_table(rows) -> list[dict]:
    """A process table's rows, each with an integer pid, ppid and start and a string title."""
    if not isinstance(rows, list) or any(
            not (isinstance(r, dict) and type(r.get("pid")) is int and type(r.get("ppid")) is int
                 and type(r.get("start")) is int and isinstance(r.get("title"), str)) for r in rows):
        raise InventoryError(f"the process table is malformed: {json.dumps(rows)[:200]}")
    pids = [row["pid"] for row in rows]
    if len(set(pids)) != len(pids):
        raise InventoryError(f"the process table repeats pid(s) {sorted({p for p in pids if pids.count(p) > 1})}")
    return rows


def parse_table(text: str) -> list[dict]:
    try:
        rows = json.loads(text.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise InventoryError(f"the process table is not readable: {error}") from None
    return check_table(rows)


def expected_agents(table: list[dict], tp: int, dp: int = 1) -> dict[int, dict]:
    """The agents an instance of ``dp`` local DP ranks of TP size ``tp`` must show, by pid, each with its rank:
    the node-local index ``dp_rank * tp + tp_rank`` (the host order of the step logs and metrics ports).

    The engine of DP rank d is ``EngineCore_DP<d>`` (``EngineCore`` at DP 1). vLLM runs the workers in the
    engine process only when TP x DP is 1 (its uniprocess executor; the L1 traces show the scheduler and
    the worker sharing a pid there): that engine owns the one agent and has no worker. Otherwise (the
    multiprocess executor) every engine's workers are its children, titled ``Worker[_DP<d>][_TP<r>][...]``,
    one per TP rank (no ``TP`` suffix at TP1), each owning one agent and the engine none. A worker joins its
    DP rank through its parent only. Anything else is an error."""
    pattern = r"^VLLM::EngineCore_DP(\d+)$" if dp > 1 else r"^VLLM::EngineCore(?:_DP0)?$"
    engines: dict[int, list[dict]] = {}
    for row in table:
        if (match := re.match(pattern, row["title"])):
            engines.setdefault(int(match.group(1) or 0) if dp > 1 else 0, []).append(row)
    if sorted(engines) != list(range(dp)) or any(len(v) != 1 for v in engines.values()):
        raise InventoryError(f"DP{dp} expects one engine process for each DP rank, found "
                             f"{[r['title'] for r in table if r['title'].startswith('VLLM::EngineCore')]}")
    parent = {rows[0]["pid"]: d for d, rows in engines.items()}
    children: dict[int, list[dict]] = {d: [] for d in engines}
    for row in table:
        if re.match(r"^VLLM::Worker(?:_|$)", row["title"]):
            if row["ppid"] not in parent:
                raise InventoryError(f"a worker ({row['title']}, pid {row['pid']}) whose parent is no engine process")
            d = parent[row["ppid"]]
            if (named := re.search(r"_DP(\d+)", row["title"])) and int(named.group(1)) != d:
                raise InventoryError(f"{row['title']} is a child of DP rank {d}'s engine")
            children[d].append(row)
    owners: dict[tuple[int, int], dict] = {}
    for d, workers in sorted(children.items()):
        ranks: dict[int, list[dict]] = {}
        for row in workers:
            named = re.search(r"_TP(\d+)", row["title"])
            ranks.setdefault(int(named.group(1)) if named else 0 if tp == 1 else -1, []).append(row)
        if tp == dp == 1:
            if workers:
                raise InventoryError(f"TP1 expects one engine process and no worker, found "
                                     f"{[engines[0][0]['title']]} and {sorted(w['title'] for w in workers)}")
            owners[(d, 0)] = engines[d][0]
        elif sorted(ranks) == list(range(tp)) and all(len(v) == 1 for v in ranks.values()):
            owners.update({(d, r): rows[0] for r, rows in ranks.items()})
        else:
            raise InventoryError(f"TP{tp} expects one worker for each rank under DP rank {d}'s engine, found "
                                 f"{sorted(w['title'] for w in workers)}")
    return {row["pid"]: {"rank": d * tp + r, "dp": d, "tp": r, "label": f"DP{d} TP{r}", "start": row["start"],
                         "title": row["title"], "host": row.get("host")}
            for (d, r), row in owners.items()}


def inventory_problems(saved, tables: dict, tp: int, dp: int) -> list[str]:
    """The saved rank mapping against the one the configured local sizes give from each completed boundary's
    process table (``None``: that boundary never read one, which is not judged here): the same pids, each with
    the same node-local rank, DP rank, TP rank and start ticks. A malformed or missing table, an empty or
    malformed mapping, or one that disagrees with a table (a rank given twice, a worker under another engine
    at the end, a restarted process) is not used to index any counter."""
    problems: list[str] = []
    try:
        if not isinstance(saved, dict) or not saved:
            raise ValueError("there is no saved rank mapping")
        owners = {int(pid): (row["rank"], row["dp"], row["tp"], row["start"]) for pid, row in saved.items()}
        if len(owners) != len(saved) or any(type(v) is not int for owner in owners.values() for v in owner):
            raise ValueError(f"the saved rank mapping is malformed: {owners}")
        ranks = sorted(owner[0] for owner in owners.values())
        if ranks != list(range(dp * tp)):
            raise ValueError(f"the saved rank mapping holds ranks {ranks}, not each of the DP{dp} TP{tp} "
                             f"instance's {dp * tp} once")
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        return [str(error)]
    for name, table in tables.items():
        if table is None:
            continue
        try:
            if not isinstance(table, list) or not table:
                raise InventoryError("it is missing or empty")
            derived = expected_agents(check_table(table), tp, dp)
        except InventoryError as error:
            problems.append(f"the {name} process table is not a DP{dp} TP{tp} instance: {error}")
            continue
        wanted = {pid: (e["rank"], e["dp"], e["tp"], e["start"]) for pid, e in derived.items()}
        if wanted != owners:
            problems.append(f"the saved rank mapping (pid: rank, DP, TP, start) {owners} is not the {name} "
                            f"process table's {wanted}")
    return problems


def identity(labels: dict) -> str:
    """The canonical identity of an agent: the whole label set (hostname, agent_name, pid, agent_instance and local_rank when
    present), sorted, excluding only the error-status dimension. Nothing short of it identifies an agent."""
    return "|".join(f"{name}={labels[name]}" for name in sorted(labels) if name != "status")


def rows_agents(rows: list[dict]) -> dict[str, dict]:
    """The agents of the rows of one sample (kind 27 and 28): ``{key: {labels, fields, errors, partial}}``, the key
    made of the whole label set (``pid/instance/name``)."""
    agents: dict[str, dict] = {}
    for row in rows:
        labels = dict(row.get("labels") or {})
        status = labels.pop("status", None)
        if not labels:
            continue  # a half served as missing: nothing of any agent
        key = identity(labels)
        agent = agents.setdefault(key, {"labels": labels, "fields": {}, "errors": {}, "partial": False})
        agent["partial"] = agent["partial"] or bool(row.get("partial"))
        for name, value in (row.get("fields") or {}).items():
            if name == "errors":
                agent["errors"][status] = value
            else:
                agent["fields"][name] = value
    return agents


def _rows(records: list[dict], target: str):
    for row in records:
        if row.get("type") == "event" and row.get("kind_id") in (27, 28) and row["target"]["id"] == target:
            yield row


def sample_agents(records: list[dict], target: str, sample: str) -> dict[str, dict]:
    """The agents one sample of the NIXL target served, from the capture's kind 27 and 28 rows."""
    return rows_agents([r for r in _rows(records, target) if r["sample_id"] == sample])


def sample_evidence_problems(records: list[dict], target: str, sample: str | None) -> list[str]:
    """Whether the selected sample's own records are in the capture, whole: ``sample_agents`` gives an empty dictionary both for a
    sample that observed no agent and for records that are absent, so absence is judged here, from the rows: some must exist, both
    kinds (27 and 28) must be there and none may be partial."""
    rows = [r for r in _rows(records, target) if r["sample_id"] == sample] if sample else []
    if not rows:
        return [f"the sample's records are absent from the capture ({sample})"]
    problems = []
    if {r["kind_id"] for r in rows} != {27, 28}:
        problems.append(f"the sample {sample} does not have both kinds of record")
    if any(r.get("partial") for r in rows):
        problems.append(f"the sample {sample} is partial")
    return problems


def samples(records: list[dict], target: str) -> list[dict]:
    """The NIXL target's samples in order: ``{id, begin_ns, end_ns, complete, agents}``. A sample is complete
    when both kinds were written, nothing is partial and at least one agent was served."""
    found: dict[str, dict] = {}
    for row in _rows(records, target):
        sample = found.setdefault(row["sample_id"], {
            "id": row["sample_id"], "sequence": row["sequence"], "begin_ns": row["scrape"]["begin_ns"],
            "end_ns": row["scrape"]["end_ns"], "kinds": set(), "partial": False, "rows": []})
        sample["kinds"].add(row["kind_id"])
        sample["partial"] = sample["partial"] or bool(row.get("partial"))
        sample["rows"].append(row)
    out = []
    for sample in sorted(found.values(), key=lambda item: item["sequence"]):
        agents = rows_agents(sample.pop("rows"))
        out.append({"id": sample["id"], "begin_ns": sample["begin_ns"], "end_ns": sample["end_ns"],
                    "complete": sample["kinds"] == {27, 28} and not sample["partial"] and bool(agents),
                    "agents": agents})
    return out


def label_rule(native):
    """The collector's own rule for which series of a family it reads, as a function ``accepts(metric, label_names)``: the labels the
    source documents, plus the optional ones. The harness's independent scrape applies the SAME rule, so the two readers interpret a
    page alike (a series whose labels neither knows is unavailable to both; nothing is stripped to force a join)."""
    families = {k: v for kind in native.SOURCES["nixl"].values() for k, v in kind.items()}
    optional = set(native._OPTIONAL_LABELS["nixl"])

    def accepts(metric: str, names: set) -> bool:
        expected = set(families[metric][2])
        return expected <= names <= expected | optional

    return accepts


def series_agents(series: dict, accepts=None, skipped: list | None = None) -> dict[str, dict]:
    """The agents of a scraped page, in the shape ``sample_agents`` gives a capture's sample: from the parsed series
    (``{(metric, sorted label pairs): value}``). With ``accepts`` (``label_rule``), a series with labels the collector does not read is
    left out and listed in ``skipped``, as the collector leaves it out of its own sample."""
    agents: dict[str, dict] = {}
    for (name, label_pairs), value in series.items():
        if name not in METRICS:
            continue
        if accepts is not None and not accepts(name, {k for k, _ in label_pairs}):
            if skipped is not None:
                skipped.append(name)
            continue
        labels = dict(label_pairs)
        status = labels.pop("status", None)
        key = identity(labels)
        agent = agents.setdefault(key, {"labels": labels, "fields": {}, "errors": {}, "partial": False})
        if METRICS[name] == "errors":
            agent["errors"][status] = value
        else:
            agent["fields"][METRICS[name]] = value
    return agents


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def agent_problems(expected: dict[int, dict], agents: dict[str, dict], kv_bytes: dict[int, int] | None = None
                   ) -> list[str]:
    """Why a sample is not a usable boundary: the agent set against the inventory (unexpected pid, ambiguous pid,
    missing pid), every required field served and finite for every agent, an error series for every status,
    and a positive registration (at least the rank's allocated KV bytes when given)."""
    problems: list[str] = []
    by_pid: dict[str, list[dict]] = {}
    for agent in agents.values():
        by_pid.setdefault(str(agent["labels"].get("pid")), []).append(agent)
    wanted = {str(pid) for pid in expected}
    if not agents:
        return ["no agent is served"]
    if unexpected := sorted(set(by_pid) - wanted):
        problems.append(f"unexpected agent(s) in pid(s) {unexpected}")
    if missing := sorted(wanted - set(by_pid)):
        problems.append(f"expected agent(s) missing in pid(s) {missing}")
    for pid, found in sorted(by_pid.items()):
        if pid in wanted and len(found) != 1:
            problems.append(f"pid {pid} owns {len(found)} agents, the inventory expects one (ambiguous join)")
    hosts = {agent["labels"].get("hostname") for agent in agents.values()}
    if len(hosts) > 1:
        problems.append(f"the agents carry different hostnames {sorted(map(str, hosts))}")
    for pid, entry in expected.items():
        if entry.get("host") is not None and any(
                a["labels"].get("hostname") != entry["host"] for a in agents.values() if str(a["labels"].get("pid")) == str(pid)):
            problems.append(f"pid {pid}: the agent's hostname is not the node's {entry['host']!r}")
    for agent in agents.values():
        pid = agent["labels"].get("pid")
        if agent["partial"]:
            problems.append(f"pid {pid}: a partial sample")
        if (str(pid) in wanted and "agent_instance" in agent["labels"]
                and agent["labels"].get("agent_instance") != "0"):
            problems.append(f"pid {pid}: agent_instance {agent['labels'].get('agent_instance')!r}, expected '0'")
        # A series the exporter serves must be a finite number; one it does not serve is unavailable (see ``unavailable``).
        malformed = sorted(n for n, v in agent["fields"].items() if not _finite(v))
        malformed += sorted(f"errors[{s}]" for s, v in agent["errors"].items() if not _finite(v))
        if malformed:
            problems.append(f"pid {pid}: served but not a finite number: {malformed}")
        registered = agent["fields"].get("memory_registered_bytes")
        if _finite(registered) and registered <= 0:
            problems.append(f"pid {pid}: nothing registered yet")
        rank = expected.get(int(pid), {}).get("rank") if str(pid).isdigit() else None
        if (kv_bytes and rank in kv_bytes and _finite(registered) and registered < kv_bytes[rank]):
            problems.append(f"pid {pid}: registered {registered:.0f} B, less than rank {rank}'s {kv_bytes[rank]} B of KV")
    return problems


def unavailable(agent: dict) -> list[str]:
    """What this agent's exporter did not serve of the names the harness knows: fields, ``errors`` when no status series is served at
    all. Reported, never a problem."""
    missing = [n for n in (*FIELDS, "events_dropped") if not _finite(agent["fields"].get(n))]
    if not agent["errors"]:
        missing.append("errors")
    return missing


def error_coverage(first: dict, last: dict) -> dict:
    """Per-status coverage between two boundaries: the statuses served at both (compared), those that appeared or disappeared in between
    (their change is UNAVAILABLE, never read as a change from or to zero; the value served at the boundary that has it is reported
    as it is), and the statuses with a non-finite value (malformed, still a problem for the sample check)."""
    a, b = first.get("errors") or {}, last.get("errors") or {}
    return {"compared": sorted(set(a) & set(b)),
            "appeared": {s: b[s] for s in sorted(set(b) - set(a))},
            "disappeared": {s: a[s] for s in sorted(set(a) - set(b))}}


def values(agents: dict[str, dict]) -> dict:
    """Everything a sample says, for equality between the collector's and the harness's reads."""
    return {key: {"labels": dict(a["labels"]), "fields": dict(a["fields"]), "errors": dict(a["errors"])}
            for key, a in sorted(agents.items())}


def by_rank(expected: dict[int, dict], agents: dict[str, dict]) -> dict[int, dict]:
    """The agent of each rank, joined through the pid only."""
    found = {}
    for agent in agents.values():
        pid = agent["labels"].get("pid")
        if str(pid).isdigit() and int(pid) in expected:
            found[expected[int(pid)]["rank"]] = agent
    return found


def watched(sample: dict) -> dict:
    """What must not move while the final sample waits: every agent's watched counters, by identity."""
    return {k: {n: a["fields"].get(n) for n in WATCHED} for k, a in sample["agents"].items()}


def settled(samples: list[dict], after_ns: int) -> tuple[bool, list[str]]:
    """Whether the counters have been unchanged for a stretch that began after ``after_ns``: the trailing run of
    consecutive complete samples, each begun after ``after_ns``, shows the same watched values for every agent
    (at least ``STABLE_SAMPLES`` of them, spanning at least ``STABLE_SPAN_NS``). An incomplete sample ends the
    run. A stable but stale reading cannot pass: the stretch must start after the workload, and the caller still
    requires the inventory and the agreement with the transfers."""
    run: list[dict] = []
    for sample in reversed(samples):
        if sample["begin_ns"] < after_ns or not sample["complete"]:
            break
        if run and watched(sample) != watched(run[-1]):
            break
        run.append(sample)
    run.reverse()
    ok = len(run) >= STABLE_SAMPLES and run[-1]["end_ns"] - run[0]["end_ns"] >= STABLE_SPAN_NS
    return ok, [s["id"] for s in run]


def boundary_problems(name: str, stage: dict, records: list[dict], target: str) -> list[str]:
    """The ordering of an accepted boundary re-derived from the capture, not trusted from what the harness stored: the request was
    noticed before the pass that answers it began; the selected sample's own scrape lies inside that pass; the harness's own read
    began after the pass ended (it reads the marker first) and ends no earlier than it began."""
    accepted = stage.get("accepted") or {}
    marker = next((r for r in records if r.get("type") == "boundary" and r.get("request_id") == accepted.get("request")), None)
    sample = next((s for s in samples(records, target) if s["id"] == accepted.get("sample")), None)
    if marker is None or sample is None:
        return [f"the {name} marker or its sample is not in the capture"]
    window, harness = marker["pass"], accepted.get("harness") or {}
    problems = []
    if not marker["detected_ns"] < window["begin_ns"] <= sample["begin_ns"] <= sample["end_ns"] <= window["end_ns"]:
        problems.append(f"the {name} request, pass and sample are not in order (detected {marker['detected_ns']}, pass "
                        f"{window['begin_ns']}..{window['end_ns']}, sample {sample['begin_ns']}..{sample['end_ns']})")
    if not (isinstance(harness.get("begin_ns"), int) and window["end_ns"] <= harness["begin_ns"] <= harness.get("end_ns", -1)):
        problems.append(f"the harness's own {name} scrape did not run after the pass ({harness.get('begin_ns')}, "
                        f"{harness.get('end_ns')}, pass ended {window['end_ns']})")
    return problems


def final_interval_problems(saved: dict, records: list[dict], target: str, workload_end_s: float | None) -> list[str]:
    """The claimed stable interval of the final sample reconstructed from the raw samples: the stretch the harness names is
    made of real, consecutive, complete samples of the capture, at least ``STABLE_SAMPLES`` of them spanning at least
    ``STABLE_SPAN_NS``, every one begun after the workload ended (``after_ns`` is later than the baseline and the stretch
    starts after the workload's own last step when the step logs give it), with the same watched values for every agent and
    the same as the selected final sample's; and the final request was noticed after the stretch ended."""
    final, baseline = saved.get("final") or {}, saved.get("baseline") or {}
    ordered = samples(records, target)
    position = {s["id"]: i for i, s in enumerate(ordered)}
    ids = final.get("stretch") or []
    problems = []
    if not ids or any(i not in position for i in ids):
        return [f"the final stretch names samples that are not in the capture ({[i for i in ids if i not in position][:2]})"]
    at = [position[i] for i in ids]
    stretch = [ordered[i] for i in at]
    if at != list(range(at[0], at[0] + len(at))):
        problems.append("the final stretch is not made of consecutive samples")
    if len(stretch) < STABLE_SAMPLES:
        problems.append(f"the final stretch has {len(stretch)} samples, fewer than {STABLE_SAMPLES}")
    if not all(s["complete"] for s in stretch):
        problems.append("the final stretch contains an incomplete sample")
    if stretch[-1]["end_ns"] - stretch[0]["end_ns"] < STABLE_SPAN_NS:
        problems.append(f"the final stretch spans {(stretch[-1]['end_ns'] - stretch[0]['end_ns']) / 1e9:.2f} s, less than "
                        f"{STABLE_SPAN_NS / 1e9:g} s")
    if any(watched(s) != watched(stretch[0]) for s in stretch):
        problems.append("the counters moved inside the final stretch")
    after = final.get("after_ns")
    if not isinstance(after, int) or any(s["begin_ns"] < after for s in stretch):
        problems.append("the final stretch did not begin after the recorded end of the workload")
    base_end = ((baseline.get("accepted") or {}).get("harness") or {}).get("end_ns")
    if isinstance(after, int) and isinstance(base_end, int) and after <= base_end:
        problems.append("the workload is recorded as ending before the baseline was complete")
    if workload_end_s is not None and stretch[0]["begin_ns"] < workload_end_s * 1e9:
        problems.append("the final stretch began before the workload's own last step ended (the step logs)")
    marker = next((r for r in records if r.get("type") == "boundary"
                   and r.get("request_id") == (final.get("accepted") or {}).get("request")), None)
    if marker is not None and marker["detected_ns"] < stretch[-1]["end_ns"]:
        problems.append("the final request was noticed before the stable stretch ended")
    chosen = (final.get("accepted") or {}).get("sample")
    if chosen in position and position[chosen] <= at[-1]:
        problems.append("the selected final sample is not after the stable stretch")
    if chosen in position and watched(ordered[position[chosen]]) != watched(stretch[-1]):
        problems.append("the selected final sample differs from the stable stretch")
    return problems


def layout_bytes(text: str) -> int | None:
    """A rank's allocated KV bytes, from its step log's layout record."""
    for line in text.splitlines():
        if '"kv_layout"' in line:
            layout = json.loads(line).get("kv_layout")
            if layout is not None:
                return layout.get("bytes")
    return None


def step_loads(text: str) -> dict:
    """What a decode rank's step log says it read: the transfers, their bytes, and their transfer and post times
    each converted back to integer microseconds (the log stores NIXL's integer microseconds as seconds) before
    summing."""
    found = {"requests": 0, "bytes": 0, "xfer_us": 0, "post_us": 0}
    for line in text.splitlines():
        for entry in json.loads(line).get("kv_transfers", {}).get("load", {}).get("each", ()):
            found["requests"] += 1
            found["bytes"] += int(entry[0])
            found["xfer_us"] += round(entry[1] * 1e6)
            found["post_us"] += round(entry[2] * 1e6)
    return found


def workload_problems(role: str, expected: dict[int, dict], baseline: dict[str, dict], final: dict[str, dict],
                      loads: dict[int, dict]) -> list[str]:
    """The workload's change (final minus baseline, the two boundary samples) per agent against the transfers the
    step logs recorded: a decode rank's receive counters equal them exactly (zero tolerance, frozen before any
    cell), transmit is zero, a prefill rank moves nothing, nothing is dropped, no served error moved and the
    registration does not change."""
    problems: list[str] = []
    first, last = by_rank(expected, baseline), by_rank(expected, final)
    for rank in sorted(expected.values(), key=lambda e: e["rank"]):
        r = rank["rank"]
        who = rank.get("label", f"rank {r}")
        if r not in first or r not in last:
            problems.append(f"{who}: no agent at both boundaries")
            continue
        a, b = first[r], last[r]
        if identity(a["labels"]) != identity(b["labels"]):
            problems.append(f"{who}: the agent's identity changed between the boundaries "
                            f"({identity(a['labels'])} then {identity(b['labels'])})")
            continue
        # Only what the exporter served at BOTH boundaries is compared; the rest is unavailable (see ``unavailable``).
        change = {n: b["fields"][n] - a["fields"][n] for n in (*FIELDS, "events_dropped")
                  if _finite(a["fields"].get(n)) and _finite(b["fields"].get(n))}
        errors = {s: b["errors"][s] - a["errors"][s] for s in sorted(set(a["errors"]) & set(b["errors"]))
                  if _finite(a["errors"][s]) and _finite(b["errors"][s])}
        if change.get("events_dropped"):
            problems.append(f"{who}: {change['events_dropped']:.0f} telemetry event(s) dropped")
        if any(errors.values()):
            problems.append(f"{who}: errors reported by NIXL {({s: v for s, v in errors.items() if v})}")
        if change.get("memory_registered_bytes") or change.get("memory_deregistered_bytes"):
            problems.append(f"{who}: registration changed over the workload "
                            f"({change.get('memory_registered_bytes', 0):.0f} / "
                            f"{change.get('memory_deregistered_bytes', 0):.0f} B)")
        if role == "decode":
            seen = loads.get(r)
            if seen is None:
                problems.append(f"{who}: no step log to compare with")
                continue
            for name, key in (("rx_requests", "requests"), ("rx_bytes", "bytes"), ("xfer_time_us", "xfer_us"),
                              ("xfer_post_time_us", "post_us")):
                if name in change and change[name] != seen[key]:
                    problems.append(f"{who}: {name} changed by {change[name]:.0f}, the step log's transfers "
                                    f"by {seen[key]}")
            if change.get("tx_requests") or change.get("tx_bytes"):
                problems.append(f"{who}: a decode rank transmitted ({change.get('tx_requests', 0):.0f} requests)")
        elif any(change.get(n) for n in ("rx_requests", "rx_bytes", "tx_requests", "tx_bytes")):
            problems.append(f"{who}: a prefill rank moved bytes by NIXL's account "
                            f"({ {n: change[n] for n in ('tx_bytes', 'rx_bytes', 'tx_requests', 'rx_requests') if n in change} })")
    return problems


# ---------------------------------------------------------------------------------------- arms without an exporter
FAILURE_SIGNS = ("transfer_exception", "transfer_failed", "NIXL_ERR_NO_TELEMETRY",
                 "cannot return values when telemetry is not enabled")


def arm_problems(output, run) -> tuple[list[str], dict]:
    """The correctness of a latency A/B arm that has no exporter (``default``, ``emissions_disabled``), judged from
    what a run saved, before any of its measurements is interpreted: the servers were started with exactly the
    arm's telemetry environment (what it sets, everything else of NIXL's telemetry settings cleared), no endpoint
    answers (nothing exports), the per-transfer statistics the connector reads are intact (every decode rank's
    step log has its transfers with positive integer microsecond durations), and no server log shows a telemetry
    failure of the transfer path. The dropped-event count is *unavailable* in these arms (the collect-only sink
    discards that record too) and is reported so: never a measured zero."""
    from pathlib import Path

    from . import pd

    output = Path(output)
    arm = pd.nixl_telemetry_arm(run)
    report = {"arm": arm, "dropped_events": ("not emitted (a property of the configuration)" if arm == "emissions_disabled"
                                             else "unavailable (the collect-only sink discards the record)")}
    problems: list[str] = []
    wanted = pd.nixl_telemetry_env(arm)
    cleared = {name for name in pd.NIXL_TELEMETRY_VARS if name not in wanted}
    try:
        commands = json.loads((output / "servers.json").read_text())
    except (OSError, ValueError) as error:
        return [f"NIXL arm {arm}: no servers.json: {error}"], report
    wanted_servers = {role: set(side.nodes) for role, side in (("prefill", run.prefill), ("decode", run.decode))}
    found_servers = {role: set((commands.get(role) or {})) for role in ("prefill", "decode")}
    if found_servers != wanted_servers or set(commands) != set(wanted_servers):
        problems.append(f"NIXL arm {arm}: servers.json lists {({r: sorted(n) for r, n in found_servers.items()})}, not "
                        f"{({r: sorted(n) for r, n in wanted_servers.items()})}")
    for role, nodes in commands.items():
        for node, command in nodes.items():
            if pd.VLLM not in command:
                problems.append(f"NIXL arm {arm}: the {role} command on {node} is not a vllm command")
                continue
            head = command[1:command.index(pd.VLLM)]
            unset = {head[i + 1] for i, part in enumerate(head) if part == "-u"}
            given = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in head if "=" in part and "NIXL_TELEMETRY" in part}
            if given != wanted or unset != cleared:
                problems.append(f"NIXL arm {arm}: the {role} server on {node} ran with telemetry settings {given} and "
                                f"cleared {sorted(unset)}, not {wanted} and {sorted(cleared)}")
    try:
        probe = json.loads((output / "nixl-arm.json").read_text())
    except (OSError, ValueError) as error:
        problems.append(f"NIXL arm {arm}: no endpoint probe: {error}")
        probe = {}
    for node, found in probe.get("nodes", {}).items():
        # Only a positively identified connection refusal shows that nothing listens: an HTTP error, a timeout or an
        # unclassified failure says something is there or that nothing is known.
        if found.get("outcome") != "refused" or found.get("body") is not None:
            problems.append(f"NIXL arm {arm}: the endpoint probe on {node} was {found.get('outcome')!r} "
                            f"(status {found.get('status')}, {found.get('error')}), not a connection refusal")
    expected_nodes = {node for side in (run.prefill, run.decode) for node in side.nodes}
    if set(probe.get("nodes", {})) != expected_nodes:
        problems.append(f"NIXL arm {arm}: the endpoint was probed on {sorted(probe.get('nodes', {}))}, not {sorted(expected_nodes)}")
    transfers = {}
    size = run.decode.tp(run.server)
    for dp, tp in run.decode.ranks(size):
        rank = f"{run.decode.node_of(dp)} DP{dp} TP{tp}"
        path = output / run.decode.step_log("decode", dp, tp)
        if not path.is_file():
            problems.append(f"NIXL arm {arm}: no step log of decode rank {rank}")
            continue
        count, slow, owned = 0, 0, set()
        for line in path.read_text().splitlines():
            record = json.loads(line)
            owned.update(((record.get("kv_events") or {}).get("allocated") or {}))
            for entry in record.get("kv_transfers", {}).get("load", {}).get("each", ()):
                count += 1
                slow += round(entry[1] * 1e6) <= 0
        transfers[rank] = count
        # A DP replica the router gave no request allocated nothing and read nothing: idle, not missing statistics.
        if slow or (count == 0 and owned):
            problems.append(f"NIXL arm {arm}: decode rank {rank} logged {count} transfers for {len(owned)} allocated "
                            f"request(s), {slow} without a positive integer-microsecond duration (the per-transfer "
                            f"statistics must be intact)")
    if transfers and not sum(transfers.values()):
        problems.append(f"NIXL arm {arm}: no decode rank logged a transfer (the per-transfer statistics must be intact)")
    report["decode_transfers"] = transfers
    for role, side in (("prefill", run.prefill), ("decode", run.decode)):
        for node in side.nodes:
            path = output / "logs" / role / node / f"{role}.log"
            try:
                text = path.read_text(errors="replace")
            except OSError as error:
                problems.append(f"NIXL arm {arm}: the {role} log on {node} is not readable: {error}")
                continue
            if not text.strip():
                problems.append(f"NIXL arm {arm}: the {role} log on {node} is empty")
                continue
            seen = [sign for sign in FAILURE_SIGNS if sign in text]
            if seen:
                problems.append(f"NIXL arm {arm}: the {role} log on {node} shows {seen}")
    return problems, report


def probe_endpoint(container: str, port: int) -> dict:
    """One node's endpoint probe: the harness's own scrape attempt (an arm without an exporter must be refused)."""
    from . import pd, topology

    text = topology.exec_in(container, [pd.PYTHON, "-c", OWN_SCRAPE, f"http://127.0.0.1:{port}/metrics"])
    found = json.loads(text.strip().splitlines()[-1])
    return {"outcome": found["outcome"], "status": found["status"], "error": found["error"],
            "body": None if found["body"] is None else found["body"][:200]}
