"""The L2 collector (``vllm/kv_profiling/collector.py``) on a run's server nodes.

One collector a node, started inside the node's container, after every server
answers and before the workload: it scrapes TENT's metrics of the node's ranks
(a node-local address, so it has to run there) from outside the serving
processes. It is stopped after the workload and before the servers are: its
final scrape needs their metrics servers. Its capture is copied to
``collector/<node>/`` and checked: complete, and its counters' change over the
workload equal to the harness's own scrape of the same ranks.

The collector file is the repository's, copied into the container as it is (the
image's copy of the package is not relied on). It is not a service of the run
(``pd.is_service``) and its CPU is counted as its own kind of helper."""

import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import time
import uuid
from pathlib import Path

from . import nixl_telemetry as nixl
from . import pd, shell, topology

SOURCE = pd.REPOSITORY / "vllm" / "kv_profiling" / "collector.py"
BOUNDARY_DIR = f"{pd.COLLECTOR_DIRECTORY}/boundaries"
# The record kinds each source writes (the collector's ``SOURCES``).
KINDS = {"tent": {25, 26}, "nixl": {27, 28}}
MANIFEST = f"{pd.COLLECTOR_DIRECTORY}/targets.json"
STOP_FILE = f"{pd.COLLECTOR_DIRECTORY}/stop"
LOG = f"{pd.WORK}/logs/collector.log"
# The harness's own TENT scrape sums a counter over its labels; so does this check.
COUNTERS = {
    25: {"write_bytes": "tent_write_bytes_total", "read_bytes": "tent_read_bytes_total",
         "write_requests": "tent_write_requests_total", "read_requests": "tent_read_requests_total"},
    26: {"attempts": "tent_transport_attempts_total",
         "attempt_failures": "tent_transport_attempt_failures_total",
         "task_failures": "tent_task_failures_total"},
}
SIGNAL_COLLECTOR = f"""
import json, os, signal, sys, time

path = {pd.COLLECTOR_PATH!r}.encode()
me = os.getpid()


def collectors():
    found = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            argv = open(f"/proc/{{pid}}/cmdline", "rb").read().split(b"\\0")
        except OSError:
            continue
        if int(pid) != me and len(argv) > 1 and argv[1] == path:
            found.append(int(pid))
    return found


signalled = []
for sig, wait in ((signal.SIGTERM, float(sys.argv[1])), (signal.SIGKILL, 5.0)):
    for pid in collectors():
        signalled.append([pid, sig.name])
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + wait
    while collectors() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not collectors():
        break
remaining = collectors()
print(json.dumps({{"signalled": signalled, "remaining": remaining}}))
sys.exit(1 if remaining else 0)
"""


def targets(run: pd.Run, node: str) -> list[dict]:
    """The collector's manifest for a node. TENT (when configured): every rank of it, by the host port the
    harness's own TENT scrape uses, at the node's loopback address. NIXL telemetry (when configured): the node's
    one aggregated endpoint, whatever its number of ranks."""
    found = []
    if run.collector.tent:
        for role, side in (("prefill", run.prefill), ("decode", run.decode)):
            size = side.tp(run.server)
            for dp, tp in side.ranks(size):
                if side.node_of(dp) == node:
                    found.append({"id": f"{role}-dp{dp}-tp{tp}", "role": role, "rank": {"dp": dp, "tp": tp},
                                  "endpoint": f"127.0.0.1:{pd.TENT_METRICS_PORT + side.local_rank(dp, tp, size)}"})
    if run.collector.nixl_telemetry:
        found.append({"id": f"nixl-{node}", "role": "prefill" if node in run.prefill.nodes else "decode",
                      "source": "nixl", "endpoint": f"127.0.0.1:{pd.NIXL_TELEMETRY_PORT}"})
    return found


def arguments(config: pd.Collector, run_id: str) -> list[str]:
    return [pd.PYTHON, pd.COLLECTOR_PATH, "--run-id", run_id, "--output-dir", pd.COLLECTOR_OUTPUT,
            "--manifest", MANIFEST, "--stop-file", STOP_FILE,
            *(["--boundary-dir", BOUNDARY_DIR] if config.nixl_telemetry else []),
            *(item for name in pd.COLLECTOR_BOUNDS
              for item in (f"--{name.replace('_', '-')}", str(getattr(config, name))))]


def clear(container: str) -> None:
    """Make a node free of any collector: signal one a previous run left behind
    (a node's container outlives its runs, and the service filters leave a
    collector alone on purpose), then remove its files. Raises when one
    survives its signals: it would scrape into, or contaminate, the next run."""
    text = topology.exec_in(container, [pd.PYTHON, "-c", SIGNAL_COLLECTOR, "2"], check=False).strip()
    try:
        result = json.loads(text.splitlines()[-1])
    except (IndexError, ValueError):
        raise RuntimeError(f"could not tell whether a collector runs on {container}: {text!r}") from None
    if result["remaining"]:
        raise RuntimeError(f"a collector on {container} survived its signals: {result['remaining']}")
    topology.exec_in(container, ["sh", "-c", f"rm -rf {pd.COLLECTOR_DIRECTORY} {pd.COLLECTOR_OUTPUT}"])


class NodeCollector:
    """The collector of one node while it runs. It is owned from the moment its
    launch is attempted: whatever fails after that, ``stop`` is how it ends."""

    def __init__(self, node: str, container: str, run_id: str, config: pd.Collector, directory: Path,
                 expected: tuple[str, ...] = (), sources: dict[str, str] | None = None, nixl_state: dict | None = None):
        self.node, self.container, self.run_id, self.config = node, container, run_id, config
        self.directory = directory
        self.expected = tuple(expected)
        self.sources = dict(sources or {})
        self.nixl_state = nixl_state
        self.boundaries: dict = {}
        self.stopped = False
        self.errors: list[str] = []

    @classmethod
    def start(cls, run: pd.Run, node: str, container: str, run_id: str, output: Path) -> "NodeCollector":
        """Copy the collector and its manifest in, start it, and wait until every
        target has had its first scrape (the baseline of the workload's change).
        A failure after the launch stops this collector before it propagates."""
        config = run.collector
        directory = output / "collector" / node
        directory.mkdir(parents=True, exist_ok=True)
        wanted = targets(run, node)
        manifest = directory / "targets.json"
        manifest.write_text(json.dumps({"targets": wanted}, indent=2) + "\n")
        clear(container)
        topology.exec_in(container, ["sh", "-c", f"mkdir -p {pd.COLLECTOR_DIRECTORY} {pd.COLLECTOR_OUTPUT} "
                                                 f"{BOUNDARY_DIR}"])
        shell.run(["docker", "cp", str(SOURCE), f"{container}:{pd.COLLECTOR_PATH}"])
        shell.run(["docker", "cp", str(manifest), f"{container}:{MANIFEST}"])
        (directory / "collector.json").write_text(json.dumps({
            "run_id": run_id, "node": node, "container": container, "settings": config.__dict__,
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "command": arguments(config, run_id)}, indent=2) + "\n")
        state = None
        if config.nixl_telemetry:
            side = run.prefill if node in run.prefill.nodes else run.decode
            role = "prefill" if side is run.prefill else "decode"
            size = side.tp(run.server)
            # every (DP rank, TP rank) of this node's instance, by its node-local index
            state = {"target": f"nixl-{node}", "role": role, "tp": size,
                     "dp": 1 if side.instances else side.data_parallel_size,
                     "steps": {side.local_rank(dp, tp, size): side.remote_step_log(dp, tp)
                               for dp, tp in side.ranks(size) if side.node_of(dp) == node}}
        started = cls(node, container, run_id, config, directory, tuple(t["id"] for t in wanted),
                      {t["id"]: t.get("source", "tent") for t in wanted}, state)
        try:
            pd._background(container, arguments(config, run_id), LOG)
            started._wait_for_baseline()
            if state is not None:
                started.nixl_baseline()
        except BaseException:
            started.stop(force=True)
            raise
        return started

    def _capture(self, name: str) -> str:
        return f"{pd.COLLECTOR_OUTPUT}/{self.run_id}/*/{name}"

    def _events(self) -> list[dict]:
        text = topology.exec_in(self.container, ["sh", "-c", f"cat {self._capture('events.jsonl')} 2>/dev/null || true"])
        rows = []
        for line in text.splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass  # a line still being written
        return rows

    def baseline(self) -> tuple[list[str], list[str]]:
        """(waiting, bad): the expected targets whose first scrape has not yet
        written both record kinds, and those whose first scrape is partial or
        unavailable."""
        first: dict[str, dict[int, dict]] = {}
        for row in self._events():
            if row.get("type") == "event":
                first.setdefault(row["target"]["id"], {}).setdefault(row["kind_id"], row)
        waiting, bad = [], []
        for target in self.expected:
            kinds = first.get(target, {})
            source = self.sources.get(target, "tent")
            if set(kinds) != KINDS[source]:
                waiting.append(target)
            elif source == "tent" and any(row["partial"] or row["missing_reason"] not in (None, "no_series_served")
                                          for row in kinds.values()):
                bad.append(target)
            # A NIXL target's first scrape may be empty while the agents appear: the baseline protocol decides.
        return waiting, bad

    def _wait_for_baseline(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_s + 5
        waiting, bad = list(self.expected), []
        while time.monotonic() < deadline:
            waiting, bad = self.baseline()
            if bad or not waiting:
                break
            if self._alive() is False:
                break
            time.sleep(0.25)
        if not waiting and not bad:
            return
        tail = topology.exec_in(self.container, ["tail", "-n", "30", LOG], check=False)
        raise RuntimeError(f"the L2 collector on {self.node} has no complete baseline (no scrape yet: {waiting}, "
                           f"partial or unavailable: {bad}); {LOG} ends with:\n{tail}")

    # ---- NIXL telemetry: the boundary samples
    def _write_boundaries(self) -> None:
        (self.directory / "boundaries.json").write_text(json.dumps(self.boundaries, indent=2, default=list) + "\n")

    def _table(self) -> list[dict]:
        return nixl.parse_table(topology.exec_in(self.container, [pd.PYTHON, "-c", nixl.PROCESS_TABLE]))

    def _container_ns(self) -> int:
        return int(topology.exec_in(self.container, [pd.PYTHON, "-c", "import time; print(time.monotonic_ns())"]).strip())

    def _step_text(self, rank: int) -> str:
        return topology.exec_in(self.container, ["cat", self.nixl_state["steps"][rank]], check=False)

    def _own_scrape(self) -> dict:
        """The harness's own read of the endpoint, with its instants on the collector's clock."""
        text = topology.exec_in(self.container, [pd.PYTHON, "-c", nixl.OWN_SCRAPE,
                                                 f"http://127.0.0.1:{pd.NIXL_TELEMETRY_PORT}/metrics"])
        got = json.loads(text.strip().splitlines()[-1])
        agents, skipped = {}, []
        if got["body"] is not None:
            native = _native()
            agents = nixl.series_agents(native.parse(got["body"], native.Limits(), "nixl")[0], nixl.label_rule(native), skipped)
        return {"begin_ns": got["begin_ns"], "end_ns": got["end_ns"], "error": got["error"], "agents": agents,
                "unserved_label_series": len(skipped)}

    def _request(self, name: str) -> str:
        """Ask the collector for a boundary pass (a file it notices once, written atomically)."""
        request = f"{name}-{uuid.uuid4().hex[:8]}"
        body = json.dumps({"name": name})
        topology.exec_in(self.container, ["sh", "-c", f"printf '%s' '{body}' > {BOUNDARY_DIR}/{request}.tmp && "
                                                      f"mv {BOUNDARY_DIR}/{request}.tmp {BOUNDARY_DIR}/{request}.json"])
        return request

    def _wait_marker(self, request: str) -> tuple[dict | None, list[dict]]:
        deadline = time.monotonic() + 2 * self.config.interval_s + self.config.scrape_timeout_s + 5
        while True:
            records = self._events()
            for row in records:
                if row.get("type") == "boundary" and row.get("request_id") == request:
                    return row, records
            if time.monotonic() >= deadline:
                return None, records
            time.sleep(0.1)

    def _boundary_attempt(self, name: str, expected: dict[int, dict], kv: dict[int, int] | None) -> dict:
        """One request for a boundary pass and what it showed: the marker's sample against the inventory, every
        required field, and the harness's own scrape taken right after (the values must be equal: during quiescence
        that is a common accounting boundary, not simultaneous reads; both reads' instants are kept)."""
        target = self.nixl_state["target"]
        request = self._request(name)
        marker, records = self._wait_marker(request)
        attempt: dict = {"request": request, "outcome": "no_marker", "problems": ["the collector answered no marker"]}
        if marker is None:
            return attempt
        sample = marker["samples"].get(target)
        agents = nixl.sample_agents(records, target, sample) if sample else {}
        pass_problems = [] if marker["pass"]["targets_complete"].get(target) else ["the pass did not complete this target"]
        problems = list(pass_problems)
        problems += nixl.agent_problems(expected, agents, kv)
        own = self._own_scrape()
        if (not pass_problems and not own["error"] and not own["agents"] and not agents and own["unserved_label_series"]):
            # The endpoint answers, but every series carries labels this reader does not know (another exporter version): the
            # telemetry cannot be observed or joined to a rank. That is unavailable coverage, not a failed run. (Only when the
            # pass itself completed: a failed pass keeps its error, and the process identity is still checked at the end.)
            attempt.update(marker=marker, sample=sample, agents={}, problems=[], outcome="unavailable_schema",
                           unserved_label_series=own["unserved_label_series"],
                           harness={"begin_ns": own["begin_ns"], "end_ns": own["end_ns"], "error": None, "agents": {}})
            return attempt
        if own["error"]:
            problems.append(f"the harness's own scrape failed: {own['error']}")
        elif nixl.values(own["agents"]) != nixl.values(agents):
            problems.append("the harness's own scrape differs from the collector's sample")
        attempt.update(marker=marker, sample=sample, agents=agents, problems=problems,
                       harness={"begin_ns": own["begin_ns"], "end_ns": own["end_ns"], "error": own["error"],
                                "agents": own["agents"]},
                       outcome="ok" if not problems else "rejected")
        return attempt

    def nixl_baseline(self) -> None:
        """The baseline: the inventory first (the node's vLLM processes, with their start ticks), then boundary
        passes until one shows exactly the expected agents with every family served and a positive registration,
        agreeing with the harness's own scrape; bounded by the startup budget. A failure is recorded and raised: a
        workload must not start without it."""
        state = self.nixl_state
        record = self.boundaries.setdefault("baseline", {"attempts": [], "outcome": None})
        self.boundaries.update(node=self.node, target=state["target"], role=state["role"], tp=state["tp"],
                               dp=state["dp"])
        try:
            table = self._table()
            expected = nixl.expected_agents(table, state["tp"], state["dp"])
        except (nixl.InventoryError, ValueError, KeyError) as error:
            record["outcome"] = "inventory_error"
            record["problems"] = [str(error)]
            self._write_boundaries()
            raise RuntimeError(f"the NIXL telemetry inventory of {self.node}: {error}") from None
        kv = {rank: nixl.layout_bytes(self._step_text(rank)) for rank in state["steps"]}
        kv = {rank: size for rank, size in kv.items() if size}
        self.boundaries.update(table_start=table, expected={str(pid): found for pid, found in expected.items()},
                               kv_bytes={str(rank): size for rank, size in kv.items()})
        deadline = time.monotonic() + self.config.startup_timeout_s
        while True:
            attempt = self._boundary_attempt("baseline", expected, kv)
            record["attempts"].append(attempt)
            if attempt["outcome"] == "ok":
                record["outcome"], record["accepted"] = "ok", attempt
                break
            if attempt["outcome"] == "unavailable_schema":
                record["outcome"], record["accepted"] = "unavailable_schema", attempt
                break
            if time.monotonic() >= deadline:
                record["outcome"] = "baseline_timeout"
                record["problems"] = attempt["problems"]
                self._write_boundaries()
                raise RuntimeError(f"no usable NIXL telemetry baseline on {self.node} within "
                                   f"{self.config.startup_timeout_s} s: {attempt['problems']}")
            time.sleep(0.25)
        self._write_boundaries()

    def nixl_final(self) -> None:
        """The final sample, with the servers and the collector alive: wait (bounded) until the counters have been
        unchanged for a stretch that began after the workload and, with the complete inventory and every field, agree
        with the transfers the step logs recorded; then one boundary pass, which must show the same values and
        agree with the harness's own scrape; the processes must be the same at both ends. Every outcome, including a
        timeout, is recorded (``final_timeout`` fails the validation; nothing is raised here)."""
        state = self.nixl_state
        baseline = self.boundaries.get("baseline", {})
        if state is not None and baseline.get("outcome") == "unavailable_schema":
            self._final_without_values()
            return
        if state is None or baseline.get("outcome") != "ok":
            return
        record = self.boundaries.setdefault("final", {"attempts": [], "outcome": None})
        expected = {int(pid): found for pid, found in self.boundaries["expected"].items()}
        after_ns = self._container_ns()
        record["after_ns"] = after_ns
        base_agents = baseline["accepted"]["agents"]
        deadline = time.monotonic() + nixl.FINAL_TIMEOUT_S
        reasons: list[str] = []
        while True:
            found = nixl.samples(self._events(), state["target"])
            settled, stretch = nixl.settled(found, after_ns)
            if settled:
                agents = next(s["agents"] for s in found if s["id"] == stretch[-1])
                reasons = nixl.agent_problems(expected, agents)
                reasons += nixl.workload_problems(state["role"], expected, base_agents, agents, self._loads())
                if not reasons:
                    break
            else:
                reasons = [f"the counters were not unchanged for {nixl.STABLE_SPAN_NS / 1e9:g} s after the workload "
                           f"({len(stretch)} stable sample(s))"]
            if time.monotonic() >= deadline:
                record.update(outcome="final_timeout", problems=reasons, stretch=stretch)
                self._write_boundaries()
                return
            time.sleep(0.25)
        record["stretch"] = stretch
        attempt = self._boundary_attempt("final", expected, None)
        record["attempts"].append(attempt)
        problems = list(attempt["problems"])
        if attempt["outcome"] != "no_marker":
            if nixl.values(attempt["agents"]) != nixl.values(agents):
                problems.append("the final marker's sample differs from the stable stretch")
            problems += nixl.workload_problems(state["role"], expected, base_agents, attempt["agents"], self._loads())
        try:
            table_end = self._table()
        except (ValueError, KeyError) as error:
            table_end, problems = None, [*problems, f"the process table could not be read again: {error}"]
        record["table_end"] = table_end
        if table_end is not None:  # every table read, an empty one included, is judged
            problems += nixl.inventory_problems(expected, {"start": self.boundaries["table_start"], "end": table_end},
                                                self.nixl_state["tp"], self.nixl_state["dp"])
        record.update(outcome="ok" if not problems else "rejected", problems=problems, accepted=attempt)
        self._write_boundaries()

    def _final_without_values(self) -> None:
        """The end of a run whose telemetry could not be read (every series carried labels this reader does not know): the metric
        comparisons are unavailable, but nothing else is skipped: a final boundary pass is requested and recorded as it is (a failed
        pass keeps its error), the process table is read again and the same processes (pid and start ticks) must stand at both ends."""
        expected = {int(pid): found for pid, found in self.boundaries["expected"].items()}
        record = self.boundaries.setdefault("final", {"attempts": [], "outcome": None})
        record["after_ns"] = self._container_ns()
        attempt = self._boundary_attempt("final", expected, None)
        record["attempts"].append(attempt)
        problems = list(attempt["problems"])
        try:
            table_end = self._table()
        except (ValueError, KeyError) as error:
            table_end, problems = None, [*problems, f"the process table could not be read again: {error}"]
        record["table_end"] = table_end
        if table_end is not None:  # every table read, an empty one included, is judged
            problems += nixl.inventory_problems(expected, {"start": self.boundaries["table_start"], "end": table_end},
                                                self.nixl_state["tp"], self.nixl_state["dp"])
        # A reading that has recovered by the end is not itself an error: its values are kept and checked as any final sample is, and
        # only the change across the boundaries is unavailable (there is no baseline to subtract).
        outcome = ("rejected" if problems or attempt["outcome"] not in ("unavailable_schema", "ok")
                   else "unavailable_schema" if attempt["outcome"] == "unavailable_schema" else "readable_after_unavailable")
        record.update(outcome=outcome, problems=problems, accepted=attempt)
        self._write_boundaries()

    def _loads(self) -> dict[int, dict]:
        if self.nixl_state["role"] != "decode":
            return {}
        return {rank: nixl.step_loads(self._step_text(rank)) for rank in self.nixl_state["steps"]}

    def _alive(self) -> bool | None:
        text = topology.exec_in(self.container, [pd.PYTHON, "-c", ALIVE, pd.COLLECTOR_PATH], check=False).strip()
        return {"yes": True, "no": False}.get(text)

    def _attempt(self, what: str, action):
        """One cleanup step on its own: a failure is kept, never lets the next step be skipped."""
        try:
            return action()
        except Exception as error:
            self.errors.append(f"{what}: {error}")
            return None

    def stop(self, *, force: bool = False) -> dict:
        """End it: ask by its stop file (one final scrape, the terminal record) and
        wait for it to finish, else signal it; then copy its capture and record how
        it ended. Every step is attempted whatever the others do, and the collector
        counts as stopped only once it is known to have exited; until then a later
        call tries again."""
        if self.stopped:
            return {"node": self.node, "repeated": True}
        outcome = "stop_file"
        if force:
            outcome = "forced"
        else:
            self._attempt("stop file", lambda: topology.exec_in(self.container, ["sh", "-c", f"touch {STOP_FILE}"]))
            deadline = time.monotonic() + self.config.shutdown_timeout_s + 5
            while time.monotonic() < deadline:
                done = self._attempt("wait", lambda: topology.exec_in(
                    self.container, ["sh", "-c", f"ls {self._capture('profiler.json')} 2>/dev/null | wc -l"]
                ).strip() == "1" and self._alive() is False)
                if done:
                    break
                time.sleep(0.25)
            else:
                outcome = "forced"
        if self._attempt("alive", self._alive) is not False:
            outcome = "forced"
            self._attempt("signal", lambda: topology.exec_in(self.container, [pd.PYTHON, "-c", SIGNAL_COLLECTOR, "5"],
                                                             check=False))
        exited = self._attempt("alive", self._alive) is False
        capture = self.directory / "capture"
        self._attempt("copy", lambda: (capture.mkdir(parents=True, exist_ok=True), shell.run(
            ["docker", "cp", f"{self.container}:{pd.COLLECTOR_OUTPUT}/.", str(capture)])))
        self._attempt("record", self._record(outcome, exited))
        self.stopped = exited
        return {"node": self.node, "stopped": outcome, "exited": exited, **({"errors": self.errors} if self.errors else {})}

    def _record(self, outcome: str, exited: bool):
        def write():
            path = self.directory / "collector.json"
            record = json.loads(path.read_text()) if path.exists() else {}
            record.update(stopped=outcome, exited=exited, errors=list(self.errors))
            path.write_text(json.dumps(record, indent=2) + "\n")
        return write


ALIVE = """
import os, sys

path = sys.argv[1].encode()
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\\0")
    except OSError:
        continue
    if int(pid) != os.getpid() and len(argv) > 1 and argv[1] == path:
        print("yes")
        break
else:
    print("no")
"""


def _native():
    """The collector module itself, loaded by path (it imports nothing outside the standard library)."""
    if "harness_native_collector" not in sys.modules:
        spec = importlib.util.spec_from_file_location("harness_native_collector", SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["harness_native_collector"]


def finish_all(collectors: list[NodeCollector]) -> None:
    """Every NIXL telemetry final sample, each node independently: a failure on one is recorded and does not stop
    the others or raise (the validation reports it)."""
    for one in collectors:
        if one.nixl_state is None:
            continue
        try:
            one.nixl_final()
        except Exception as error:
            one.boundaries.setdefault("final", {})["outcome"] = "final_error"
            one.boundaries["final"]["problems"] = [f"{type(error).__name__}: {error}"]
            one._write_boundaries()


def start_all(run: pd.Run, container: dict[str, str], output: Path, run_id: str) -> list[NodeCollector]:
    """A collector on every server node; when one fails to start, every one
    already started is stopped (each attempted whatever the others do) and the
    failure is raised."""
    started: list[NodeCollector] = []
    try:
        for side in (run.prefill, run.decode):
            for node in side.nodes:
                started.append(NodeCollector.start(run, node, container[node], run_id, output))
    except BaseException:
        stop_all(started, force=True)
        raise
    return started


def stop_all(collectors: list[NodeCollector], *, force: bool = False) -> list[dict]:
    """Stop every collector (each is stopped even if another's stop fails)."""
    results = []
    for one in collectors:
        try:
            results.append(one.stop(force=force))
        except Exception as error:
            results.append({"node": one.node, "error": str(error)})
    return results


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sample_totals(records: list[dict], kind: int, fields: dict[str, str]) -> list[dict]:
    """A target's samples of one record kind in order: their reset generation,
    completeness and each counter summed over its label sets; a counter no row of
    the sample served is None (unavailable), never zero."""
    samples: dict[str, dict] = {}
    for row in records:
        if row.get("type") != "event" or row.get("kind_id") != kind:
            continue
        sample = samples.setdefault(row["sample_id"], {
            "sample_id": row["sample_id"], "sequence": row["sequence"], "generation": row["reset_generation"],
            "partial": False, "discontinuity": False, "totals": {name: None for name in fields}})
        sample["partial"] |= bool(row["partial"]) or row.get("missing_reason") not in (None, "no_series_served")
        sample["discontinuity"] |= bool(row["discontinuity"])
        for name in fields:
            value = (row.get("fields") or {}).get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                sample["totals"][name] = (sample["totals"][name] or 0.0) + value
    return sorted(samples.values(), key=lambda item: item["sequence"])


# A transfer that moved anything went through the transport's attempts: if the harness's own scrape
# saw requests, ``attempts`` has to be served too (families are created on first use, so absence
# elsewhere is a fact of a quiet rank, not a gap).
ACTIVE = ("tent_write_requests_total", "tent_read_requests_total", "tent_write_bytes_total",
          "tent_read_bytes_total")
REQUIRED_WHEN_ACTIVE = ("tent_transport_attempts_total",)


def _identity(run: pd.Run, node: str, directory: Path, output: Path, problems: list[str]):
    """The capture's identity against what the harness started: its run ID (the
    one recorded at the start, the profiled run's when there is one), its
    process, its settings and its targets (role, rank, endpoint) against the
    manifest recomputed from the run. Returns the events' file and its records."""
    record = json.loads((directory / "collector.json").read_text())
    saved = json.loads((directory / "targets.json").read_text())["targets"]
    if saved != targets(run, node):
        problems.append(f"L2 collector {node}: the saved manifest is not this run's targets for the node")
    if record.get("node") != node or record.get("settings") != run.collector.__dict__:
        problems.append(f"L2 collector {node}: its record does not carry this run's node and settings")
    profile = output / "profiler.json"
    if profile.exists() and json.loads(profile.read_text()).get("run_id") != record.get("run_id"):
        problems.append(f"L2 collector {node}: its run ID is not the profiled run's")
    runs = sorted((directory / "capture").glob("*/*/events.jsonl"))
    if len(runs) != 1:
        raise ValueError(f"{len(runs)} captures, one expected")
    path = runs[0]
    records = _records(path)
    metadata = records[0] if records and records[0].get("type") == "metadata" else {}
    if not (metadata.get("run_id") == path.parent.parent.name == record.get("run_id")):
        problems.append(f"L2 collector {node}: its capture is not run {record.get('run_id')} "
                        f"(directory {path.parent.parent.name}, metadata {metadata.get('run_id')})")
    if metadata.get("process_id") != path.parent.name:
        problems.append(f"L2 collector {node}: its metadata names another process than its directory")
    context = metadata.get("context") or {}
    if context.get("role") != "collector" or sorted(context.get("targets") or []) != sorted(t["id"] for t in saved):
        problems.append(f"L2 collector {node}: its metadata does not name this node's targets")
    by_id = {target["id"]: target for target in saved}
    for row in records:
        if row.get("type") != "event":
            continue
        target, expected = row["target"], by_id.get(row["target"]["id"])
        if expected is None or (target["role"], target["rank"] or {}, target["endpoint"]) != (
                expected["role"], expected.get("rank") or {}, expected["endpoint"]):
            problems.append(f"L2 collector {node}: a record's target {target['id']} is not in the manifest as "
                            f"recorded")
            break
    return path, records


def _check_unavailable(saved: dict, label: str, records: list[dict], target: str, expected: dict[int, dict], report: dict,
                       problems: list[str]) -> None:
    """The saved evidence of a run whose telemetry could not be read (``unavailable_schema``): the metric comparisons are unavailable
    and reported so, but the evidence that supports the coverage is validated as for any run: each boundary's marker must be in the
    capture and name the recorded sample, the pass must have completed this target, the harness's read must be clean and agree with the
    sample (both readers see no agent), a failed pass keeps its error, a final boundary must have been recorded, and the same
    processes (pid and start ticks) stand at both ends."""
    allowed = {"baseline": ("unavailable_schema",), "final": ("unavailable_schema", "readable_after_unavailable")}
    for name in ("baseline", "final"):
        stage = saved.get(name) or {}
        report[name] = stage.get("outcome")
        accepted = stage.get("accepted") or {}
        if stage.get("outcome") not in allowed[name]:
            problems.append(f"{label}: the {name} boundary was not recorded as complete with unavailable metrics "
                            f"({stage.get('outcome')}: {(stage.get('problems') or [])[:3]})")
            continue
        marker = next((r for r in records if r.get("type") == "boundary" and r.get("request_id") == accepted.get("request")), None)
        if marker is None or marker["samples"].get(target) != accepted.get("sample"):
            problems.append(f"{label}: the {name} marker is missing from the capture or names another sample")
            continue
        if not marker["pass"]["targets_complete"].get(target):
            problems.append(f"{label}: the {name} pass did not complete this target")
        # The selected sample's records must exist (an empty sample and absent records both give no agents), whole, in order.
        evidence = nixl.sample_evidence_problems(records, target, accepted.get("sample"))
        problems.extend(f"{label}: {issue}" for issue in evidence)
        if not evidence:
            problems.extend(f"{label}: {issue}" for issue in nixl.boundary_problems(name, stage, records, target))
        sample = nixl.sample_agents(records, target, accepted.get("sample")) if not evidence else {}
        harness = accepted.get("harness") or {}
        if harness.get("error"):
            problems.append(f"{label}: the harness's own {name} read failed ({harness.get('error')})")
        elif nixl.values(harness.get("agents") or {}) != nixl.values(sample):
            problems.append(f"{label}: the harness's own {name} read differs from the collector's sample")
        if stage["outcome"] == "readable_after_unavailable":
            # recovered by the end: judged as any final sample is (inventory, finite values); no change across the boundaries
            if nixl.values(sample) != nixl.values(accepted.get("agents") or {}):
                problems.append(f"{label}: the {name} sample in the capture is not what the harness accepted")
            issues = nixl.agent_problems(expected, sample)
            problems.extend(f"{label}: the {name} sample fails the inventory: {issue}" for issue in issues[:3])
            report["final_unavailable"] = {a["labels"].get("pid"): nixl.unavailable(a) for a in sample.values()}
            report["metric_deltas"] = "unavailable (the baseline could not be read)"
        elif sample:
            problems.append(f"{label}: the {name} sample in the capture has agents the harness recorded as none")
        if accepted.get("problems"):
            problems.append(f"{label}: the {name} attempt carries problems {accepted['problems'][:3]}")
    report["unavailable_schema"] = {"unserved_label_series": ((saved.get("baseline") or {}).get("accepted") or {}).get(
        "unserved_label_series")}


def check_nixl(output: Path, run: pd.Run, node: str, role: str, side: pd.Side, records: list[dict], found: dict,
               problems: list[str]) -> None:
    """The NIXL telemetry of one node, judged again from what was saved: the boundary record, the capture's markers
    and samples, the harness's own reads and the step logs. Both boundaries must have been accepted; each accepted
    sample is re-derived from the capture (the marker names the sample, the sample's agents equal what the harness
    kept, and they satisfy the inventory and every required field); the harness's own scrape equals the collector's
    at both; the same processes (pid and start ticks) stand at both ends; and the workload's change per agent
    agrees with the transfers the step logs recorded (zero tolerance), nothing dropped, no served error moved,
    registration unchanged, a prefill rank's counters still."""
    label = f"NIXL telemetry {node}"
    report = found["nixl"] = {}
    try:
        saved = json.loads((output / "collector" / node / "boundaries.json").read_text())
        target = saved["target"]
        expected = {int(pid): row for pid, row in saved["expected"].items()}
        kv = {int(rank): size for rank, size in saved.get("kv_bytes", {}).items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        problems.append(f"{label}: no boundary record: {error}")
        return
    # The full configured inventory at every completed boundary, before any counter or coverage is published:
    # the baseline's table once it was accepted, the final's once the final ran (a final that timed out keeps
    # its own failure, below). The rank mapping is re-derived from each with the run's local sizes.
    local_dp = 1 if side.instances else side.data_parallel_size
    baseline, final = saved.get("baseline") or {}, saved.get("final") or {}
    completed = {"start": baseline.get("outcome") in ("ok", "unavailable_schema"),
                 "end": final.get("outcome") not in (None, "final_timeout")}
    # a completed boundary without its table is judged (and fails) as a malformed one
    tables = {"start": saved.get("table_start"), "end": final.get("table_end")}
    tables = {name: ([] if table is None else table) if completed[name] else None for name, table in tables.items()}
    issues = nixl.inventory_problems(saved["expected"], tables, side.tp(run.server), local_dp)
    if (saved.get("tp"), saved.get("dp")) != (side.tp(run.server), local_dp):
        issues.append(f"the boundary record is of DP{saved.get('dp')} TP{saved.get('tp')}, the run's {node} is "
                      f"DP{local_dp} TP{side.tp(run.server)}")
    if issues:
        problems.extend(f"{label}: {issue}" for issue in issues)
        for name in ("baseline", "final"):
            stage = saved.get(name) or {}
            if stage.get("outcome") not in ("ok", "unavailable_schema", "readable_after_unavailable"):
                problems.append(f"{label}: the {name} sample was not accepted ({stage.get('outcome')}: "
                                f"{(stage.get('problems') or [])[:3]})")
        return
    agents = {}
    if (saved.get("baseline") or {}).get("outcome") == "unavailable_schema":
        _check_unavailable(saved, label, records, target, expected, report, problems)
        return
    for name in ("baseline", "final"):
        stage = saved.get(name) or {}
        report[name] = stage.get("outcome")
        if stage.get("outcome") != "ok":
            problems.append(f"{label}: the {name} sample was not accepted ({stage.get('outcome')}: "
                            f"{(stage.get('problems') or [])[:3]})")
            continue
        accepted = stage["accepted"]
        marker = next((r for r in records if r.get("type") == "boundary" and r.get("request_id") == accepted["request"]),
                      None)
        if marker is None or marker["samples"].get(target) != accepted["sample"]:
            problems.append(f"{label}: the {name} marker is missing from the capture or names another sample")
            continue
        sample = nixl.sample_agents(records, target, accepted["sample"])
        if nixl.values(sample) != nixl.values(accepted["agents"]):
            problems.append(f"{label}: the {name} sample in the capture is not what the harness accepted")
            continue
        issues = nixl.agent_problems(expected, sample, kv if name == "baseline" else None)
        if issues:
            problems.append(f"{label}: the {name} sample fails the inventory: {issues[:3]}")
            continue
        if accepted["harness"].get("error") or nixl.values(accepted["harness"]["agents"]) != nixl.values(sample):
            problems.append(f"{label}: the harness's own {name} scrape differs from the collector's")
            continue
        agents[name] = sample
        report[f"{name}_sample"] = accepted["sample"]
        order = nixl.boundary_problems(name, stage, records, target)
        problems.extend(f"{label}: {issue}" for issue in order)
    if (saved.get("final") or {}).get("outcome") == "ok":
        steps = []
        for step_role, step_side in (("prefill", run.prefill), ("decode", run.decode)):
            for dp, tp in step_side.ranks(step_side.tp(run.server)):
                path = output / step_side.step_log(step_role, dp, tp)
                if path.is_file():
                    steps.append(path.read_text())
        window = pd.workload_window(steps) if steps else None
        interval = nixl.final_interval_problems(saved, records, target, None if window is None else window[1])
        problems.extend(f"{label}: {issue}" for issue in interval)
    if len(agents) != 2:
        return
    loads = {}
    size = side.tp(run.server)
    for dp, tp in side.ranks(size) if role == "decode" else ():
        if side.node_of(dp) != node:
            continue
        path = output / side.step_log(role, dp, tp)
        loads[side.local_rank(dp, tp, size)] = nixl.step_loads(path.read_text()) if path.is_file() else None
    loads = {rank: seen for rank, seen in loads.items() if seen is not None}
    issues = nixl.workload_problems(role, expected, agents["baseline"], agents["final"], loads)
    problems.extend(f"{label}: {issue}" for issue in issues)
    first, last = nixl.by_rank(expected, agents["baseline"]), nixl.by_rank(expected, agents["final"])
    report["workload"] = {rank: last[rank]["fields"]["rx_requests"] - first[rank]["fields"]["rx_requests"]
                          for rank in sorted({e["rank"] for e in expected.values()})
                          if rank in first and rank in last
                          and "rx_requests" in last[rank]["fields"] and "rx_requests" in first[rank]["fields"]}
    # What the exporter did not serve is reported, never a failure, and a comparison that was skipped is told apart from one that was
    # validated: per rank, the fields compared at both boundaries and the names not served at one of them.
    def names(rank):
        coverage = nixl.error_coverage(first[rank], last[rank])
        return sorted({*nixl.unavailable(first[rank]), *nixl.unavailable(last[rank]),
                       *(f"errors[{s}] (appeared)" for s in coverage["appeared"]),
                       *(f"errors[{s}] (disappeared)" for s in coverage["disappeared"])})

    report["unavailable"] = {rank: names(rank) for rank in sorted(first) if rank in last and names(rank)}
    report["coverage"] = {rank: {"validated": sorted(n for n in (*nixl.FIELDS, "events_dropped")
                                                     if n not in nixl.unavailable(first[rank]) and n not in nixl.unavailable(last[rank])),
                                 "unavailable": sorted(set(nixl.unavailable(first[rank])) | set(nixl.unavailable(last[rank]))),
                                 "errors": nixl.error_coverage(first[rank], last[rank])}
                          for rank in sorted(first) if rank in last}


def check(output: Path, run: pd.Run, tent: dict, verdict: dict) -> None:
    """The collectors' evidence against the run: each node's capture is this
    run's (run ID, process, settings, targets), complete, each of its ranks was
    sampled at the start and the end and the counters did not reset, and the
    counters' change between the first and the last sample equals ``tent`` (the
    harness's own scrape of the same ranks around the workload). A counter one
    side did not serve is compared as unavailable, never as zero: both absent is
    recorded as such (and is a problem for ``attempts`` on a rank that moved
    anything); one side only is a problem; a counter served at any point and
    missing from the last sample is a problem (its change, or a reset, cannot be
    ruled out), whichever sample first served it. Problems go to
    ``verdict['problems']``, the figures to ``verdict['collector']``."""
    if run.collector is None:
        return
    report = verdict.setdefault("collector", {})
    problems = verdict["problems"]
    for role, side in (("prefill", run.prefill), ("decode", run.decode)):
        for node in side.nodes:
            found = report[node] = {}
            directory = output / "collector" / node
            try:
                record = json.loads((directory / "collector.json").read_text())
                path, records = _identity(run, node, directory, output, problems)
                summary = json.loads((path.parent / "profiler.json").read_text())
            except (OSError, ValueError, KeyError, TypeError) as error:
                problems.append(f"L2 collector {node}: no usable capture: {error}")
                continue
            found.update(stopped=record.get("stopped"), samples=summary.get("samples"),
                         complete=summary.get("observation_complete"))
            if record.get("stopped") != "stop_file" or record.get("exited") is False or record.get("errors"):
                problems.append(f"L2 collector {node}: did not finish cleanly on its own "
                                f"({record.get('stopped')}, errors {record.get('errors')})")
            if summary.get("observation_complete") is not True:
                problems.append(f"L2 collector {node}: observation incomplete "
                                f"(reasons {summary.get('reasons')}, unavailable {summary.get('unavailable')})")
            end = [row for row in records if row.get("type") == "end"]
            if len(end) != 1 or end[0].get("complete") is not True:
                problems.append(f"L2 collector {node}: its terminal record is missing or says incomplete")
            sequences = [row["sequence"] for row in records if row.get("type") == "event"]
            if sequences != sorted(set(sequences)):
                problems.append(f"L2 collector {node}: event sequences are not unique and increasing")
            size = side.tp(run.server)
            if run.collector.nixl_telemetry:
                check_nixl(output, run, node, role, side, records, found, problems)
            for index, (dp, tp) in enumerate(side.ranks(size) if run.collector.tent else ()):
                if side.node_of(dp) != node:
                    continue
                target = f"{role}-dp{dp}-tp{tp}"
                own = [row for row in records if row.get("type") == "event" and row["target"]["id"] == target]
                after = tent.get(role, [])
                expected = after[index] if index < len(after) else None
                found[target] = rank = {}
                if expected is None:
                    problems.append(f"L2 collector {target}: the harness has no TENT scrape of this rank to "
                                    f"compare with")
                active = expected is not None and any(expected.get(metric) for metric in ACTIVE)
                for kind, fields in COUNTERS.items():
                    samples = _sample_totals(own, kind, fields)
                    if len(samples) < 2:
                        problems.append(f"L2 collector {target}: {len(samples)} samples of kind {kind}, at least "
                                        f"two (the start and the end) are needed")
                        continue
                    first, last = samples[0], samples[-1]
                    if first["partial"] or last["partial"]:
                        problems.append(f"L2 collector {target}: its first or last sample is partial")
                    if any(sample["discontinuity"] for sample in samples) or first["generation"] != last["generation"]:
                        problems.append(f"L2 collector {target}: a counter went backwards during the run")
                    for name, metric in fields.items():
                        seen, before = last["totals"][name], first["totals"][name]
                        harness = None if expected is None else expected.get(metric)
                        rank[name] = state = {"collector": None if seen is None else seen - (before or 0.0),
                                              "harness": harness,
                                              "baseline": "absent" if before is None else "observed"}
                        if expected is None:
                            continue
                        ever = any(sample["totals"][name] is not None for sample in samples)
                        state["observed_earlier"] = ever
                        if seen is None and ever:
                            problems.append(f"L2 collector {target}: {name} was served earlier in the run but not "
                                            f"in its last sample, so its change cannot be established")
                        elif seen is None and harness is None:
                            state["unavailable"] = "both"
                            if active and metric in REQUIRED_WHEN_ACTIVE:
                                problems.append(f"L2 collector {target}: {name} is served by neither side although "
                                                f"the rank moved data")
                        elif seen is None or harness is None:
                            problems.append(f"L2 collector {target}: {name} is served by "
                                            f"{'the harness only' if seen is None else 'the collector only'}")
                        elif abs(state["collector"] - harness) > 1e-6:
                            problems.append(f"L2 collector {target}: {name} changed by {state['collector']:g}, the "
                                            f"harness's own scrape of {metric} by {harness:g}")
                    rank[f"samples_kind_{kind}"] = len(samples)
