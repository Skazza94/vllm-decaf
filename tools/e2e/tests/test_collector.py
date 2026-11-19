"""The harness's orchestration of the L2 collector: run options, targets, the
node-side filters that must recognise it, and (with the real collector as a
process, against local metrics pages) the capture it is checked on."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from harness import collector, pd

RUN = pd.Run(name="t", topology="t", model=Path("/m"), prefill=pd.Side("p", "data0"),
             decode=pd.Side("d", "data0"), client="c", control_network="control", router=pd.Router("gw"),
             server=pd.Server(connector="mooncake", timing="linear"), workload=pd.Workload(num_prompts=2),
             collector=pd.Collector())


ORIGINAL_TARGETS = collector.targets


def run_with(**changes) -> pd.Run:
    return pd.Run(**{**{f: getattr(RUN, f) for f in RUN.__dataclass_fields__}, **changes})


def raw_run(**changes) -> dict:
    return {"name": "t", "topology": "t", "model": "/m", "client": "c", "control_network": "control",
            "prefill": {"node": "p", "kv_network": "data0"}, "decode": {"node": "d", "kv_network": "data0"},
            "router": {"gateway": "gw"}, "server": {"connector": "mooncake", "timing": "linear"}, **changes}


class RunOptions(unittest.TestCase):
    def parse(self, **changes) -> pd.Run:
        with mock.patch.object(pd.models, "resolve", lambda model, fetch=False: Path(model)):
            return pd.parse(raw_run(**changes), Path("/r.yaml"))

    def test_the_collector_is_an_independent_option_with_the_collectors_own_limits(self):
        self.assertIsNone(self.parse().collector)
        self.assertEqual(self.parse(collector={}).collector, pd.Collector())
        run = self.parse(collector={"interval_s": 0.5, "shutdown_timeout_s": 20})
        self.assertEqual((run.collector.interval_s, run.collector.shutdown_timeout_s), (0.5, 20))
        self.assertIsNone(run.profiler)

    def test_it_needs_tents_metrics_and_sane_bounds(self):
        nixl = lambda backend: {"connector": "nixl", "nixl_backend": backend, "timing": "linear"}  # noqa: E731
        self.assertEqual(self.parse(collector={}, server=nixl("mooncake")).collector, pd.Collector())
        for backend in ("ucx", "libfabric"):
            with self.assertRaisesRegex(ValueError, "collector needs TENT"):
                self.parse(collector={}, server=nixl(backend))
        for bad in ({"interval_s": 0.01}, {"interval_s": float("nan")}, {"scrape_timeout_s": "1"},
                    {"startup_timeout_s": 601}, {"shutdown_timeout_s": True}):
            with self.assertRaisesRegex(ValueError, "collector"):
                self.parse(collector=bad)
        with self.assertRaisesRegex(ValueError, "unexpected keyword"):
            self.parse(collector={"nope": 1})

    def test_a_saved_run_keeps_it(self):
        with tempfile.TemporaryDirectory() as directory:
            run = run_with(collector=pd.Collector(interval_s=0.25))
            Path(directory, "run.json").write_text(json.dumps(
                {**pd.asdict(run), "model": "/m", "addresses": {}}, default=str))
            self.assertEqual(pd.saved_run(directory).collector, pd.Collector(interval_s=0.25))
            Path(directory, "run.json").write_text(json.dumps(
                {**pd.asdict(run_with(collector=None)), "model": "/m"}, default=str))
            self.assertIsNone(pd.saved_run(directory).collector)


class Manifest(unittest.TestCase):
    def test_a_nodes_manifest_lists_its_ranks_at_the_ports_the_harness_scrapes(self):
        run = run_with(server=pd.Server(connector="mooncake", timing="linear", tensor_parallel_size=2),
                       prefill=pd.Side("p", ["data0", "data1"]), decode=pd.Side("d", ["data0", "data1"]))
        self.assertEqual(collector.targets(run, "p"), [
            {"id": "prefill-dp0-tp0", "role": "prefill", "rank": {"dp": 0, "tp": 0},
             "endpoint": f"127.0.0.1:{pd.TENT_METRICS_PORT}"},
            {"id": "prefill-dp0-tp1", "role": "prefill", "rank": {"dp": 0, "tp": 1},
             "endpoint": f"127.0.0.1:{pd.TENT_METRICS_PORT + 1}"}])
        self.assertEqual([t["id"] for t in collector.targets(run, "d")], ["decode-dp0-tp0", "decode-dp0-tp1"])

    def test_a_node_holding_one_of_several_instances_lists_only_its_own(self):
        run = run_with(prefill=pd.Side(["p0", "p1"], "data0"))
        # The instance's rank index is global, its metrics port its host's: 9100 on each node.
        self.assertEqual([(t["id"], t["endpoint"]) for t in collector.targets(run, "p1")],
                         [("prefill-dp1-tp0", f"127.0.0.1:{pd.TENT_METRICS_PORT}")])
        self.assertEqual(collector.targets(run, "d")[0]["role"], "decode")


class NodeFilters(unittest.TestCase):
    COMMAND = b"/opt/venv/bin/python\0" + pd.COLLECTOR_PATH.encode() + b"\0--run-id\0x"

    def test_it_is_not_a_service_of_the_run_and_the_embedded_filters_name_the_same_path(self):
        self.assertFalse(pd.is_service(self.COMMAND, b"python"))
        self.assertTrue(pd.is_service(b"/opt/venv/bin/python\0/opt/venv/bin/vllm\0serve", b"python"))
        for script in (pd.STOP_OWNERS, pd.STOP, pd.CPU_SAMPLER):
            self.assertIn("/e2e/collector/collector.py", script)
        self.assertEqual(pd.COLLECTOR_PATH, "/e2e/collector/collector.py")

    def test_its_cpu_is_a_helpers_of_its_own_kind(self):
        def sample(t, *processes):
            return {"t": t, "processes": [{"pid": p, "start": p, "kind": kind, "cpu_s": cpu, "rss_kb": 0, "hwm_kb": 0}
                                          for p, kind, cpu in processes]}

        samples = [sample(float(t), (1, "serving", float(t)), (2, "collector", 0.05 * t)) for t in range(5)]
        report = pd.cpu_over(samples, 1.0, 3.0)
        self.assertAlmostEqual(report["serving_s"], 2.0)
        self.assertAlmostEqual(report["collector_s"], 0.1)
        self.assertAlmostEqual(report["helpers_s"], 0.1)

    def test_the_sampler_files_it_as_a_collector_and_not_as_serving(self):
        classify = pd.CPU_SAMPLER
        for fragment in ('scraper = len(argv) > 1 and argv[1] == b"/e2e/collector/collector.py"',
                         '"collector" if scraper else "serving"'):
            self.assertIn(fragment, classify)

    def test_the_signal_script_matches_only_the_exact_collector(self):
        for sample in (b"/opt/venv/bin/python\0" + pd.COLLECTOR_PATH.encode() + b"\0",
                       b"/opt/venv/bin/vllm\0serve"):
            argv = sample.split(b"\0")
            self.assertEqual(len(argv) > 1 and argv[1] == collector.pd.COLLECTOR_PATH.encode(),
                             b"collector" in sample)
        self.assertIn(repr(pd.COLLECTOR_PATH), collector.SIGNAL_COLLECTOR)

    def test_the_signal_script_accounts_for_what_it_signalled_and_what_survived(self):
        argv = [sys.executable, "-c", collector.SIGNAL_COLLECTOR, "1"]
        done = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual((done.returncode, json.loads(done.stdout)), (0, {"signalled": [], "remaining": []}))


class Page(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = self.server.page().encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class Counters:
    """A local TENT metrics page whose counters the test moves."""

    def __init__(self):
        self.written = self.requests = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Page)
        self.server.page = self.page
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_address[1]

    def page(self) -> str:
        return "\n".join([
            f'tent_write_bytes_total{{transport="tcp"}} {self.written}',
            'tent_read_bytes_total{transport="tcp"} 0',
            f'tent_write_requests_total{{transport="tcp"}} {self.requests}',
            'tent_read_requests_total{transport="tcp"} 0',
            'tent_registered_buffer_bytes{transport="tcp"} 4096',
            f'tent_transport_attempts_total{{transport="tcp",operation="write"}} {self.requests}',
            'tent_transport_attempt_failures_total{transport="tcp",operation="write"} 0',
            'tent_task_failures_total{transport="tcp",reason="timeout"} 0', ""])

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def tent(written, requests) -> dict:
    return {"tent_write_bytes_total": written, "tent_read_bytes_total": 0.0, "tent_write_requests_total": requests,
            "tent_read_requests_total": 0.0, "tent_registered_buffer_bytes": 4096.0,
            "tent_transport_attempts_total": requests, "tent_transport_attempt_failures_total": 0.0,
            "tent_task_failures_total": 0.0}


def manifest_of(work: Path) -> dict:
    return json.loads((work / "targets.json").read_text())["targets"]


class CaptureFixture(unittest.TestCase):
    """The real collector as a process on the command line the harness builds,
    its capture laid out as the harness copies it, and the harness's check."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "output"
        self.pages = {"p": Counters(), "d": Counters()}
        for page in self.pages.values():
            self.addCleanup(page.close)

    def local_targets(self, run, node):
        return [{**target, "endpoint": f"127.0.0.1:{self.pages[node].port}"} for target in ORIGINAL_TARGETS(run, node)]

    def observe(self, run: pd.Run, moves, *, run_id="2d9d2c75-4f43-4a0e-9a16-0f3a1b7b8a10"):
        """Run one collector a node over local pages; ``moves(pages)`` is the workload
        between the first sample and the stop. Returns when both have finished."""
        self.run_id = run_id
        processes = []
        for node in self.pages:
            work = Path(self.directory.name) / node
            work.mkdir()
            manifest = {"targets": self.local_targets(run, node)}
            (work / "targets.json").write_text(json.dumps(manifest))
            with mock.patch.object(collector, "MANIFEST", str(work / "targets.json")), \
                    mock.patch.object(collector, "STOP_FILE", str(work / "stop")), \
                    mock.patch.object(pd, "COLLECTOR_PATH", str(collector.SOURCE)), \
                    mock.patch.object(pd, "COLLECTOR_OUTPUT", str(work / "out")):
                command = collector.arguments(run.collector, run_id)
            command[0] = sys.executable
            processes.append((node, work, subprocess.Popen(command, stdout=subprocess.PIPE, text=True)))
        for _, work, _ in processes:
            self.wait_for(lambda work=work: any((work / "out").glob("*/*/events.jsonl"))
                          and "transport_metrics" in next((work / "out").glob("*/*/events.jsonl")).read_text())
        moves(self.pages)
        time.sleep(0.4)
        for node, work, process in processes:
            (work / "stop").touch()
            code, summary = process.wait(timeout=30), process.stdout.read()
            process.stdout.close()
            self.assertEqual(code, 0, summary)
            target = self.output / "collector" / node
            (target / "capture").mkdir(parents=True)
            (target / "targets.json").write_text(json.dumps({"targets": manifest_of(work)}))
            (target / "collector.json").write_text(json.dumps({
                "run_id": run_id, "node": node, "settings": run.collector.__dict__, "stopped": "stop_file",
                "exited": True, "errors": []}))
            subprocess.run(["cp", "-r", f"{work}/out/.", str(target / "capture")], check=True)

    @staticmethod
    def wait_for(condition, timeout=30.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.05)
        raise AssertionError("timed out")

    def verdict(self, run, tent_by_role):
        verdict = {"problems": []}
        with mock.patch.object(collector, "targets", self.local_targets):
            collector.check(self.output, run, tent_by_role, verdict)
        return verdict



class Capture(CaptureFixture):
    def test_the_built_command_runs_the_real_collector_and_its_counters_match_the_harness_scrape(self):
        run = run_with(collector=pd.Collector(interval_s=0.1, shutdown_timeout_s=5))

        def workload(pages):
            for page in pages.values():
                page.written, page.requests = 123456, 7

        self.observe(run, workload)
        verdict = self.verdict(run, {"prefill": [tent(123456.0, 7.0)], "decode": [tent(123456.0, 7.0)]})
        self.assertEqual(verdict["problems"], [])
        rank = verdict["collector"]["p"]["prefill-dp0-tp0"]
        self.assertEqual((rank["write_bytes"]["collector"], rank["write_bytes"]["harness"]), (123456.0, 123456.0))
        self.assertGreaterEqual(rank["samples_kind_25"], 3)
        self.assertTrue(verdict["collector"]["d"]["complete"])

    def test_a_change_the_harness_did_not_see_is_a_problem_naming_the_counter(self):
        run = run_with(collector=pd.Collector(interval_s=0.1, shutdown_timeout_s=5))

        def workload(pages):
            for page in pages.values():
                page.written, page.requests = 1000, 1

        self.observe(run, workload)
        verdict = self.verdict(run, {"prefill": [tent(1000.0, 1.0)], "decode": [tent(900.0, 1.0)]})
        self.assertEqual(len(verdict["problems"]), 1, verdict["problems"])
        self.assertIn("decode-dp0-tp0", verdict["problems"][0])
        self.assertIn("write_bytes changed by 1000", verdict["problems"][0])
        self.assertIn("tent_write_bytes_total by 900", verdict["problems"][0])

    def test_a_counter_that_went_backwards_or_a_missing_harness_scrape_is_a_problem(self):
        run = run_with(collector=pd.Collector(interval_s=0.1, shutdown_timeout_s=5))

        def workload(pages):
            pages["p"].written = 5000
            time.sleep(0.4)
            pages["p"].written = 10  # a restart of the engine's counters
            time.sleep(0.4)

        self.observe(run, workload)
        verdict = self.verdict(run, {"prefill": [tent(10.0, 0.0)], "decode": [None]})
        joined = "\n".join(verdict["problems"])
        self.assertIn("prefill-dp0-tp0: a counter went backwards", joined)
        self.assertIn("decode-dp0-tp0: the harness has no TENT scrape", joined)

    def test_a_missing_or_unfinished_capture_is_a_problem_not_a_pass(self):
        run = run_with(collector=pd.Collector())
        verdict = self.verdict(run, {"prefill": [tent(0.0, 0.0)], "decode": [tent(0.0, 0.0)]})
        self.assertEqual(len(verdict["problems"]), 2)
        self.assertTrue(all("no usable capture" in problem for problem in verdict["problems"]))
        # A run without the option has nothing to check.
        untouched = {"problems": []}
        collector.check(self.output, run_with(collector=None), {}, untouched)
        self.assertEqual(untouched, {"problems": []})

    def test_a_collector_that_had_to_be_signalled_is_a_problem_even_with_a_full_capture(self):
        run = run_with(collector=pd.Collector(interval_s=0.1, shutdown_timeout_s=5))
        self.observe(run, lambda pages: None)
        for node in ("p", "d"):
            path = self.output / "collector" / node / "collector.json"
            path.write_text(json.dumps({**json.loads(path.read_text()), "stopped": "forced"}))
        verdict = self.verdict(run, {"prefill": [tent(0.0, 0.0)], "decode": [tent(0.0, 0.0)]})
        self.assertEqual(len([p for p in verdict["problems"] if "did not finish cleanly on its own" in p]), 2)


# ---- the capture's evidence, edited
def edit_captures(output: Path, change) -> None:
    """Apply ``change(rows)`` to every node's saved events (a list of records, edited in place)."""
    for path in (output / "collector").glob("*/capture/*/*/events.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        change(rows)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class Evidence(CaptureFixture):
    """What a matching total must not hide: unavailable metrics and another run's capture."""

    def observed(self):
        run = run_with(collector=pd.Collector(interval_s=0.1, shutdown_timeout_s=5))

        def workload(pages):
            for page in pages.values():
                page.written, page.requests = 1000, 3

        self.observe(run, workload)
        return run

    def test_a_family_neither_side_served_is_unavailable_not_zero_and_a_problem_for_attempts_on_a_busy_rank(self):
        run = self.observed()

        def unserved(rows):
            for row in rows:
                if row.get("kind_id") == 26:
                    row.update(fields={}, labels={}, partial=False, missing_reason="no_series_served",
                               missing=["attempts", "attempt_failures", "task_failures"])

        edit_captures(self.output, unserved)
        counters = {role: [{k: v for k, v in tent(1000.0, 3.0).items() if "attempt" not in k and "task" not in k}]
                    for role in ("prefill", "decode")}
        verdict = self.verdict(run, counters)
        joined = "\n".join(verdict["problems"])
        self.assertIn("prefill-dp0-tp0: attempts is served by neither side although the rank moved data", joined)
        self.assertEqual(verdict["collector"]["p"]["prefill-dp0-tp0"]["attempts"]["unavailable"], "both")
        self.assertIsNone(verdict["collector"]["p"]["prefill-dp0-tp0"]["attempts"]["collector"])
        # attempt failures are absent in a healthy run: recorded as unavailable, no problem of their own
        self.assertNotIn("attempt_failures", joined)

    def test_a_counter_served_earlier_and_missing_from_the_last_sample_is_a_problem(self):
        """Lazy creation and absence throughout stay legitimate; disappearance does not."""
        good = {role: [{k: v for k, v in tent(1000.0, 3.0).items() if k != "tent_transport_attempt_failures_total"}]
                for role in ("prefill", "decode")}
        for when in ("first", "middle"):
            self.setUp()
            run = self.observed()
            # The stub page always serves the failure counter; take it out of the capture: a quiet rank.
            edit_captures(self.output, lambda rows: [row["fields"].pop("attempt_failures", None)
                                                     for row in rows if row.get("kind_id") == 26])
            self.assertEqual(self.verdict(run, good)["problems"], [], "absent throughout is a quiet rank")

            def serve_once(rows, when=when):
                events = [row for row in rows if row.get("kind_id") == 26 and row["target"]["id"] == "prefill-dp0-tp0"
                          and row["fields"]]
                samples = sorted({row["sample_id"]: row["sequence"] for row in events}.items(), key=lambda item: item[1])
                if not samples:
                    return
                chosen = samples[0][0] if when == "first" else samples[len(samples) // 2][0]
                for row in events:
                    if row["sample_id"] == chosen:
                        row["fields"]["attempt_failures"] = 7.0

            edit_captures(self.output, serve_once)
            verdict = self.verdict(run, good)
            self.assertIn("prefill-dp0-tp0: attempt_failures was served earlier in the run but not in its last sample",
                          "\n".join(verdict["problems"]), when)
            self.assertTrue(verdict["collector"]["p"]["prefill-dp0-tp0"]["attempt_failures"]["observed_earlier"])

    def test_a_counter_created_mid_run_and_served_at_the_end_is_legitimate(self):
        run = self.observed()

        def lazily_created(rows):
            """TENT serves a series from its first use: take the byte counter out of the first sample."""
            events = [row for row in rows if row.get("kind_id") == 25 and row["target"]["id"] == "prefill-dp0-tp0"]
            if events:
                first = min(events, key=lambda row: row["sequence"])["sample_id"]
                for row in events:
                    if row["sample_id"] == first:
                        row["fields"].pop("write_bytes", None)

        edit_captures(self.output, lazily_created)
        verdict = self.verdict(run, {"prefill": [tent(1000.0, 3.0)], "decode": [tent(1000.0, 3.0)]})
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["collector"]["p"]["prefill-dp0-tp0"]["write_bytes"]["baseline"], "absent")

    def test_a_family_only_one_side_served_is_a_problem(self):
        run = self.observed()
        edit_captures(self.output, lambda rows: [row.update(fields={}, labels={}, missing_reason="no_series_served")
                                                 for row in rows if row.get("kind_id") == 26])
        verdict = self.verdict(run, {"prefill": [tent(1000.0, 3.0)], "decode": [tent(1000.0, 3.0)]})
        self.assertIn("attempts is served by the harness only", "\n".join(verdict["problems"]))
        # the other way round: the harness lacks what the collector saw
        self.setUp()
        run = self.observed()
        counters = {role: [{k: v for k, v in tent(1000.0, 3.0).items() if k != "tent_write_bytes_total"}]
                    for role in ("prefill", "decode")}
        self.assertIn("write_bytes is served by the collector only", "\n".join(self.verdict(run, counters)["problems"]))

    def test_a_boundary_sample_whose_page_failed_is_a_problem(self):
        run = self.observed()

        def failed_last(rows):
            events = [row for row in rows if row.get("type") == "event" and row["target"]["id"] == "prefill-dp0-tp0"]
            if not events:
                return
            last = max(events, key=lambda row: row["sequence"])["sample_id"]
            for row in events:
                if row["sample_id"] == last:
                    row.update(fields={}, partial=True, missing_reason="timeout")

        edit_captures(self.output, failed_last)
        verdict = self.verdict(run, {"prefill": [tent(1000.0, 3.0)], "decode": [tent(1000.0, 3.0)]})
        self.assertIn("first or last sample is partial", "\n".join(verdict["problems"]))

    def test_another_runs_capture_does_not_validate_by_matching_totals(self):
        run = self.observed()
        good = {"prefill": [tent(1000.0, 3.0)], "decode": [tent(1000.0, 3.0)]}
        self.assertEqual(self.verdict(run, good)["problems"], [])
        edit_captures(self.output, lambda rows: rows[0].update(run_id="00000000-0000-4000-8000-000000000001"))
        problems = "\n".join(self.verdict(run, good)["problems"])
        self.assertIn("its capture is not run", problems)
        self.assertIn("00000000-0000-4000-8000-000000000001", problems)

    def test_a_record_of_another_target_or_endpoint_and_a_foreign_manifest_are_problems(self):
        run = self.observed()
        good = {"prefill": [tent(1000.0, 3.0)], "decode": [tent(1000.0, 3.0)]}
        edit_captures(self.output, lambda rows: [row["target"].update(endpoint="127.0.0.1:1")
                                                 for row in rows if row.get("type") == "event"][:1])
        self.assertIn("a record's target", "\n".join(self.verdict(run, good)["problems"]))
        self.setUp()
        run = self.observed()
        path = self.output / "collector" / "p" / "targets.json"
        saved = json.loads(path.read_text())
        saved["targets"][0]["rank"] = {"dp": 0, "tp": 7}
        path.write_text(json.dumps(saved))
        self.assertIn("saved manifest is not this run's targets", "\n".join(self.verdict(run, good)["problems"]))

    def test_a_captures_process_and_settings_must_be_the_runs(self):
        run = self.observed()
        good = {"prefill": [tent(1000.0, 3.0)], "decode": [tent(1000.0, 3.0)]}
        edit_captures(self.output, lambda rows: rows[0].update(process_id="elsewhere"))
        self.assertIn("another process than its directory", "\n".join(self.verdict(run, good)["problems"]))
        path = self.output / "collector" / "d" / "collector.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), "settings": {"interval_s": 99}}))
        self.assertIn("this run's node and settings", "\n".join(self.verdict(run, good)["problems"]))


# ---- the first sample of every target
def native_collector():
    spec = importlib.util.spec_from_file_location("agent3_native_collector", collector.SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def local_exec(container, command, **kwargs):
    """``docker exec`` as a local command, the node's paths as the test's."""
    command = [sys.executable if part == pd.PYTHON else part for part in command]
    return subprocess.run(command, capture_output=True, text=True, check=kwargs.get("check", True)).stdout


class Baseline(unittest.TestCase):
    """The real collector scraping two ranks, the second's page held back or failing."""

    PAGE = (b'tent_write_bytes_total{transport="tcp"} 0\n'
            b'tent_transport_attempts_total{transport="tcp",operation="write"} 0\n')

    def setUp(self):
        self.native = native_collector()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.entered, self.release, self.stop = threading.Event(), threading.Event(), threading.Event()
        self.mode = "hold"
        self.errors = []
        self.run_id = "2d9d2c75-4f43-4a0e-9a16-0f3a1b7b8a11"
        self.out = Path(self.directory.name) / "out"
        self.targets = [self.native.Target.from_json({
            "id": f"prefill-dp0-tp{r}", "role": "prefill", "rank": {"dp": 0, "tp": r},
            "endpoint": f"127.0.0.1:{10000 + r}"}) for r in range(2)]

    def fetch(self, host, port, path, deadline, limits):
        if port == 10001:
            if self.mode == "fail":
                raise self.native.ScrapeError("timeout")
            self.entered.set()
            assert self.release.wait(10)
        return self.PAGE

    def start_collector(self):
        patcher = mock.patch.object(self.native, "fetch", self.fetch)
        patcher.start()
        self.addCleanup(patcher.stop)

        def go():
            try:
                self.native.run(self.native.Limits(interval_s=1), self.targets, self.out, self.run_id, self.stop)
            except BaseException as error:
                self.errors.append(repr(error))

        thread = threading.Thread(target=go)
        thread.start()

        def finish():
            self.release.set()
            self.stop.set()
            thread.join(10)

        self.addCleanup(finish)

    def node(self, **settings):
        return collector.NodeCollector("p", "local", self.run_id, pd.Collector(**settings), Path(self.directory.name),
                                       tuple(t.id for t in self.targets))

    def test_ready_only_when_every_targets_first_scrape_has_both_record_kinds(self):
        self.start_collector()
        self.assertTrue(self.entered.wait(5))
        one = self.node()
        with mock.patch.object(pd, "COLLECTOR_OUTPUT", str(self.out)), mock.patch.object(collector.topology, "exec_in", local_exec):
            self.assertEqual(one.baseline(), (["prefill-dp0-tp1"], []), "rank 0 is sampled, rank 1 still waits")
            with mock.patch.object(one, "_alive", lambda: True), mock.patch.object(collector.time, "monotonic",
                                                                                    mock.Mock(side_effect=[0.0, 1.0, 1.5, 99.0, 99.0])), \
                    mock.patch.object(collector.time, "sleep", lambda seconds: None):
                with self.assertRaisesRegex(RuntimeError, r"no scrape yet: \['prefill-dp0-tp1'\]"):
                    one._wait_for_baseline()
            self.release.set()
            self.wait_until(lambda: one.baseline() == ([], []))
            one._wait_for_baseline()

    def test_a_first_scrape_that_failed_is_reported_not_waited_for(self):
        self.mode = "fail"
        self.start_collector()
        one = self.node()
        with mock.patch.object(pd, "COLLECTOR_OUTPUT", str(self.out)), mock.patch.object(collector.topology, "exec_in", local_exec):
            self.wait_until(lambda: one.baseline()[1])
            self.assertEqual(one.baseline(), ([], ["prefill-dp0-tp1"]))
            with self.assertRaisesRegex(RuntimeError, r"partial or unavailable: \['prefill-dp0-tp1'\]"):
                one._wait_for_baseline()

    @staticmethod
    def wait_until(condition, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.05)
        raise AssertionError("timed out")


# ---- ownership and cleanup
class Fake:
    """A node's container as a stub: commands answered by a script, every one recorded."""

    def __init__(self, alive=("no",), fail=()):
        self.calls, self.alive, self.fail = [], list(alive), fail
        self.events = ""

    def __call__(self, container, command, **kwargs):
        self.calls.append(command)
        text = " ".join(map(str, command))
        for needle in self.fail:
            if needle in text:
                raise RuntimeError(f"injected failure of {needle}")
        if command[:2] == [pd.PYTHON, "-c"]:
            if command[2] == collector.SIGNAL_COLLECTOR:
                return json.dumps({"signalled": [], "remaining": []})
            if command[2] == collector.ALIVE:
                return self.alive.pop(0) if len(self.alive) > 1 else self.alive[0]
        if "events.jsonl" in text:
            return self.events
        if "profiler.json" in text:
            return "1"
        return ""

    def signalled(self) -> int:
        return sum(1 for call in self.calls if call[:2] == [pd.PYTHON, "-c"] and call[2] == collector.SIGNAL_COLLECTOR)


def first_scrape(*targets) -> str:
    rows = [{"type": "event", "kind_id": kind, "sample_id": f"s-{target}", "partial": False, "missing_reason": None,
             "target": {"id": target}} for target in targets for kind in collector.COUNTERS]
    return "".join(json.dumps(row) + "\n" for row in rows)


class Lifecycle(unittest.TestCase):
    """The orchestration's order and its failure paths, with docker stubbed."""

    def started(self, fake, directory, nodes=("p", "d")):
        with mock.patch.object(collector.topology, "exec_in", fake), \
                mock.patch.object(pd, "_background") as background, \
                mock.patch.object(collector.shell, "run") as docker:
            return collector.start_all(RUN, {"p": "c-p", "d": "c-d"}, Path(directory), "run-id"), background, docker

    def test_start_clears_leftovers_copies_the_source_and_the_manifest_and_waits_for_every_baseline(self):
        fake = Fake()
        fake.events = first_scrape("prefill-dp0-tp0", "decode-dp0-tp0")
        with tempfile.TemporaryDirectory() as directory:
            started, background, docker = self.started(fake, directory)
            record = json.loads(Path(directory, "collector", "p", "collector.json").read_text())
        self.assertEqual([one.node for one in started], ["p", "d"])
        self.assertEqual(fake.calls[0][:2], [pd.PYTHON, "-c"], "a leftover collector is signalled before anything starts")
        self.assertEqual(fake.calls[0][2], collector.SIGNAL_COLLECTOR)
        copies = [call.args[0][2:] for call in docker.call_args_list]
        self.assertIn([str(collector.SOURCE), f"c-p:{pd.COLLECTOR_PATH}"], copies)
        command, log = background.call_args_list[0].args[1:]
        self.assertEqual(command[:2], [pd.PYTHON, pd.COLLECTOR_PATH])
        self.assertEqual(log, collector.LOG)
        self.assertEqual(record["source_sha256"], hashlib.sha256(collector.SOURCE.read_bytes()).hexdigest())

    def test_a_node_whose_targets_have_not_all_been_scraped_is_not_ready(self):
        fake = Fake(alive=("yes",))
        fake.events = first_scrape("prefill-dp0-tp0")  # the decode node's own target never appears there
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(collector.time, "sleep", lambda seconds: None), \
                mock.patch.object(collector.time, "monotonic", mock.Mock(side_effect=itertools.count(0, 4))), \
                mock.patch.object(collector.topology, "exec_in", fake), \
                mock.patch.object(pd, "_background"), mock.patch.object(collector.shell, "run"):
            with self.assertRaisesRegex(RuntimeError, r"L2 collector on d has no complete baseline"):
                collector.start_all(RUN, {"p": "c-p", "d": "c-d"}, Path(directory), "run-id")
        # the node that did start is stopped too, with a signal
        self.assertGreaterEqual(fake.signalled(), 3)

    def test_a_failure_after_the_launch_stops_that_collector_and_the_others(self):
        fake = Fake(alive=("yes", "no"))
        fake.events = ""
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(collector.time, "sleep", lambda seconds: None), \
                mock.patch.object(collector.time, "monotonic", mock.Mock(side_effect=itertools.count(0, 4))), \
                mock.patch.object(collector.topology, "exec_in", fake), \
                mock.patch.object(pd, "_background"), mock.patch.object(collector.shell, "run"):
            with self.assertRaisesRegex(RuntimeError, "p has no complete baseline"):
                collector.start_all(RUN, {"p": "c-p", "d": "c-d"}, Path(directory), "run-id")
            record = json.loads(Path(directory, "collector", "p", "collector.json").read_text())
        self.assertEqual(record["stopped"], "forced")

    def test_cleanup_continues_past_a_failing_collector_and_keeps_the_original_error(self):
        one = collector.NodeCollector("p", "c-p", "r", pd.Collector(), Path("/nonexistent"))
        two = collector.NodeCollector("d", "c-d", "r", pd.Collector(), Path("/nonexistent"))
        calls = []

        def stop(name, boom):
            def inner(*, force=False):
                calls.append(name)
                if boom:
                    raise RuntimeError("cleanup failed")
                return {"node": name}
            return inner

        one.stop, two.stop = stop("p", True), stop("d", False)
        self.assertEqual(collector.stop_all([one, two], force=True), [{"node": "p", "error": "cleanup failed"}, {"node": "d"}])
        self.assertEqual(calls, ["p", "d"])
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(collector.topology, "exec_in", Fake()), \
                mock.patch.object(collector.NodeCollector, "start", side_effect=[one, ValueError("second start failed")]), \
                mock.patch.object(collector, "stop_all") as cleanup:
            with self.assertRaisesRegex(ValueError, "second start failed"):
                collector.start_all(RUN, {"p": "c-p", "d": "c-d"}, Path(directory), "run-id")
        cleanup.assert_called_once_with([one], force=True)

    def test_stop_asks_by_file_waits_for_the_summary_and_is_safe_twice(self):
        fake = Fake(alive=("yes", "no"))
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(collector.topology, "exec_in", fake), mock.patch.object(collector.shell, "run") as docker:
            one = collector.NodeCollector("p", "c-p", "run-id", pd.Collector(), Path(directory))
            Path(directory, "collector.json").write_text("{}")
            first, again = one.stop(), one.stop()
            record = json.loads(Path(directory, "collector.json").read_text())
        self.assertEqual(first, {"node": "p", "stopped": "stop_file", "exited": True})
        self.assertEqual(again, {"node": "p", "repeated": True})
        self.assertEqual((record["stopped"], record["exited"], record["errors"]), ("stop_file", True, []))
        self.assertTrue(any(f"touch {collector.STOP_FILE}" in " ".join(map(str, call)) for call in fake.calls))
        self.assertEqual(docker.call_args.args[0][:2], ["docker", "cp"])

    def test_a_failed_stop_file_does_not_prevent_the_signal_and_the_error_is_kept(self):
        fake = Fake(alive=("yes", "no"), fail=(f"touch {collector.STOP_FILE}",))
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(collector.topology, "exec_in", fake), mock.patch.object(collector.shell, "run"), \
                mock.patch.object(collector.time, "sleep", lambda seconds: None), \
                mock.patch.object(collector.time, "monotonic", mock.Mock(side_effect=itertools.count(0, 8))):
            one = collector.NodeCollector("p", "c-p", "run-id", pd.Collector(shutdown_timeout_s=1), Path(directory))
            Path(directory, "collector.json").write_text("{}")
            result = one.stop()
            record = json.loads(Path(directory, "collector.json").read_text())
        self.assertEqual((result["stopped"], result["exited"]), ("forced", True))
        self.assertIn("stop file: injected failure", result["errors"][0])
        self.assertEqual(fake.signalled(), 1)
        self.assertEqual(record["errors"], result["errors"])

    def test_a_collector_not_known_to_have_exited_is_not_marked_stopped_and_a_second_call_tries_again(self):
        fake = Fake(alive=("yes",))
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(collector.topology, "exec_in", fake), mock.patch.object(collector.shell, "run"), \
                mock.patch.object(collector.time, "sleep", lambda seconds: None), \
                mock.patch.object(collector.time, "monotonic", mock.Mock(side_effect=itertools.count(0, 8))):
            one = collector.NodeCollector("p", "c-p", "run-id", pd.Collector(shutdown_timeout_s=1), Path(directory))
            Path(directory, "collector.json").write_text("{}")
            first = one.stop()
            second = one.stop()
        self.assertFalse(first["exited"])
        self.assertNotIn("repeated", second)
        self.assertEqual(fake.signalled(), 2)

    def test_clear_signals_a_leftover_and_refuses_one_that_survives_or_cannot_be_checked(self):
        fake = Fake()
        collector.NodeCollector  # noqa: B018
        with mock.patch.object(collector.topology, "exec_in", fake):
            collector.clear("c-p")
        self.assertEqual(fake.signalled(), 1)
        self.assertTrue(any("rm -rf" in " ".join(map(str, call)) for call in fake.calls))
        for answer, message in ((json.dumps({"signalled": [[7, "SIGKILL"]], "remaining": [7]}), "survived its signals"),
                                ("", "could not tell")):
            with mock.patch.object(collector.topology, "exec_in", lambda *a, **k: answer):
                with self.assertRaisesRegex(RuntimeError, message):
                    collector.clear("c-p")


if __name__ == "__main__":
    unittest.main()
