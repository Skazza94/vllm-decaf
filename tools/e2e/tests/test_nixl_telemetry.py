"""NIXL's built-in telemetry as an L2 source: the collector's kinds 27 and 28 and boundary markers, the pure rules
(inventory, identity, settling, agreement with the transfers) and the protocol through the harness's actual helpers
(the real collector as a process on local metrics pages, the real ``NodeCollector`` methods and ``check``), with the
failure cases a review named: delayed flush, a missing or late agent, an extra agent, ambiguous and reused
identities, a restart between the boundaries, a stalled exporter, TP1's engine-process agent, and the alignment of
the collector's and the harness's reads."""

from __future__ import annotations

import importlib.util
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

from harness import collector, nixl_telemetry as nixl, pd

DATA = Path(__file__).resolve().parent / "data"
STATUS_NAMES = nixl.STATUSES


def native():
    return collector._native()


# --------------------------------------------------------------------------- a faithful exposition
def agent(name="a", pid=10, instance=0, rank_label=None, **fields) -> dict:
    values = {name_: 0 for name_ in (*nixl.FIELDS, "events_dropped")}
    values["memory_registered_bytes"] = 2048
    values.update(fields)
    return {"name": name, "pid": pid, "instance": instance, "local_rank": rank_label, "fields": values,
            "errors": {status: 0 for status in STATUS_NAMES}}


FAMILIES = (("agent_tx_bytes_total", "tx_bytes"), ("agent_rx_bytes_total", "rx_bytes"),
            ("agent_tx_requests_num_total", "tx_requests"), ("agent_rx_requests_num_total", "rx_requests"),
            ("agent_memory_registered_total", "memory_registered_bytes"),
            ("agent_memory_deregistered_total", "memory_deregistered_bytes"),
            ("agent_xfer_time_total", "xfer_time_us"), ("agent_xfer_post_time_total", "xfer_post_time_us"),
            ("agent_telemetry_events_dropped_total", "events_dropped"))


def exposition(agents: list[dict], host="node", zone=()) -> str:
    """The text ``prometheus_mp`` serves: the HTTP server's own families, then each family with one series for every
    agent (as the saved exposition of the image), with a last-operation gauge and a histogram that the collector
    must ignore."""
    lines = ["# TYPE exposer_scrapes_total counter", "exposer_scrapes_total 3",
             "# TYPE exposer_request_latencies summary", 'exposer_request_latencies{quantile="0.5"} Nan']

    def labels(a, extra="", family=None):
        rank = f',local_rank="{a["local_rank"]}"' if a["local_rank"] is not None else ""
        if family in zone or "all" in zone or f"agent:{a['name']}" in zone:
            extra = f',zone="z"{extra}'  # a label this reader does not know (another exporter version)
        return f'{{hostname="{host}",agent_name="{a["name"]}",pid="{a["pid"]}",agent_instance="{a["instance"]}"{rank}{extra}}}'

    for family, key in FAMILIES:
        lines += [f"# TYPE {family} counter"] + [f"{family}{labels(a, family=family)} {a['fields'][key]}" for a in agents if key in a["fields"]]
    lines += ["# TYPE agent_rx_last_bytes gauge"] + [f"agent_rx_last_bytes{labels(a)} 1" for a in agents]
    lines += ["# TYPE agent_xfer_time_us histogram"]
    for a in agents:
        lines += [f'agent_xfer_time_us_bucket{labels(a, ",le=" + chr(34) + "10" + chr(34))} 0',
                  f"agent_xfer_time_us_count{labels(a)} 0", f"agent_xfer_time_us_sum{labels(a)} 0"]
    lines += ["# TYPE agent_errors_total counter"]
    for a in agents:
        lines += [f'agent_errors_total{labels(a, ",status=" + chr(34) + status + chr(34), "agent_errors_total")} {value}'
                  for status, value in a["errors"].items()]
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = self.server.render().encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class Endpoint:
    """A local stand-in for the node's aggregated endpoint, with agents the test moves."""

    def __init__(self, agents: list[dict] | None = None, port: int = 0):
        self.agents = agents or []
        self.lock = threading.Lock()
        self.empty = False
        self.zone: tuple = ()  # families served with an extra, unknown label
        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.render = self.render
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_address[1]
        self.timers: list[threading.Timer] = []

    def render(self) -> str:
        with self.lock:
            return "" if self.empty else exposition(self.agents, zone=self.zone)

    def change(self, mutate, after: float = 0.0) -> None:
        """Apply ``mutate(agents)`` now, or after ``after`` seconds (a flush that arrives late)."""
        def apply():
            with self.lock:
                mutate(self.agents)

        if after <= 0:
            apply()
            return
        timer = threading.Timer(after, apply)
        timer.daemon = True
        timer.start()
        self.timers.append(timer)

    def close(self):
        for timer in self.timers:
            timer.cancel()
        self.server.shutdown()
        self.server.server_close()


# --------------------------------------------------------------------------- the collector itself
class CollectorKinds(unittest.TestCase):
    def test_the_real_exposition_gives_two_agents_and_ignores_what_it_should(self):
        text = (DATA / "nixl-prometheus-mp-exposition.txt").read_text()
        series, problems = native().parse(text, native().Limits(), "nixl")
        self.assertFalse(any(problems.values()), problems)
        names = {name for name, _ in series}
        self.assertEqual(names, {"agent_tx_bytes_total", "agent_rx_bytes_total", "agent_tx_requests_num_total",
                                 "agent_rx_requests_num_total", "agent_xfer_time_total", "agent_xfer_post_time_total",
                                 "agent_memory_registered_total", "agent_memory_deregistered_total",
                                 "agent_telemetry_events_dropped_total", "agent_errors_total"})
        agents = nixl.series_agents(series)
        self.assertEqual(sorted(a["labels"]["agent_name"] for a in agents.values()), ["initiator", "target"])
        initiator = next(a for a in agents.values() if a["labels"]["agent_name"] == "initiator")
        self.assertEqual((initiator["fields"]["rx_requests"], initiator["fields"]["rx_bytes"],
                          initiator["fields"]["xfer_time_us"]), (8, 3072, 1289))
        self.assertEqual(sorted(initiator["errors"]), sorted(STATUS_NAMES))
        self.assertEqual(nixl.agent_problems({23: {"rank": 0, "start": 1}, 10: {"rank": 1, "start": 1}}, agents), [])

    def test_a_target_source_is_validated_and_a_nixl_target_needs_its_endpoint(self):
        module = native()
        self.assertEqual(module.Target.from_json({"id": "t"}).source, "tent")
        self.assertEqual(module.Target.from_json({"id": "t", "source": "nixl", "endpoint": "127.0.0.1:9"}).source, "nixl")
        with self.assertRaisesRegex(ValueError, "source must be one of"):
            module.Target.from_json({"id": "t", "source": "ucx"})
        host, port, reason = module.resolve_endpoint(module.Target.from_json({"id": "t", "source": "nixl"}))
        self.assertEqual((host, port, reason), (None, None, "a nixl target needs an explicit endpoint"))

    def run_collector(self, endpoint, seconds=0.6, boundaries=None, rows=None):
        module = native()
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, True))
        target = module.Target.from_json({"id": "nixl-n", "role": "decode", "source": "nixl",
                                         "endpoint": f"127.0.0.1:{endpoint.port}"})
        stop = threading.Event()
        run_id = "2d9d2c75-4f43-4a0e-9a16-0f3a1b7b8a12"
        done = []
        thread = threading.Thread(target=lambda: done.append(module.run(
            module.Limits(interval_s=0.1, shutdown_timeout_s=5), [target], directory, run_id, stop,
            boundaries=boundaries)))
        thread.start()
        time.sleep(seconds)
        stop.set()
        thread.join(15)
        path = next(directory.glob("*/*/events.jsonl"))
        return [json.loads(line) for line in path.read_text().splitlines()], done[0]

    def test_kinds_27_and_28_carry_the_agents_labels_fields_and_errors_and_an_optional_local_rank(self):
        endpoint = Endpoint([agent("u1", 10, rx_requests=3, rx_bytes=900, rank_label="0"), agent("u2", 11)])
        self.addCleanup(endpoint.close)
        rows, summary = self.run_collector(endpoint)
        events = [r for r in rows if r.get("type") == "event"]
        self.assertEqual({r["kind_id"] for r in events}, {27, 28})
        self.assertEqual({r["kind"] for r in events}, {"nixl_telemetry", "nixl_telemetry_events"})
        first = events[0]["sample_id"]
        shown = nixl.sample_agents(rows, "nixl-n", first)
        self.assertEqual(sorted(a["labels"]["pid"] for a in shown.values()), ["10", "11"])
        one = next(a for a in shown.values() if a["labels"]["pid"] == "10")
        self.assertEqual((one["fields"]["rx_requests"], one["fields"]["rx_bytes"], one["labels"]["local_rank"]), (3, 900, "0"))
        self.assertEqual(sorted(one["errors"]), sorted(STATUS_NAMES))
        self.assertTrue(summary["observation_complete"], summary)
        meta = rows[0]
        self.assertEqual((meta["context"]["sources"], meta["context"]["connector"]), (["nixl"], "nixl"))
        self.assertEqual(sorted(meta["capabilities"]["kinds"].items()), [("27", "nixl_telemetry"), ("28", "nixl_telemetry_events")])
        # the tent kinds' tables are unchanged
        self.assertEqual({k: v for k, v in native().NAMES.items() if k in (25, 26)},
                         {25: "transport_metrics", 26: "transport_metrics_failures"})

    def test_an_empty_page_is_a_partial_sample_with_no_agents_never_zeros(self):
        endpoint = Endpoint([agent()])
        self.addCleanup(endpoint.close)
        endpoint.empty = True
        rows, summary = self.run_collector(endpoint, seconds=0.4)
        found = nixl.samples(rows, "nixl-n")
        self.assertTrue(found and all(not s["complete"] and not s["agents"] for s in found))
        self.assertFalse(summary["observation_complete"])

    def test_a_label_the_source_does_not_document_is_unavailable_data_never_an_incomplete_capture(self):
        text = exposition([agent()]).replace('agent_instance="0"', 'agent_instance="0",zone="z"', 3)
        series, _ = native().parse(text, native().Limits(), "nixl")
        endpoint = Endpoint([agent()])
        self.addCleanup(endpoint.close)
        endpoint.render = lambda: text  # noqa: E731
        endpoint.server.render = endpoint.render
        rows, summary = self.run_collector(endpoint, seconds=0.4)
        self.assertNotIn("unexpected_labels", summary["reasons"])
        self.assertGreater(summary["unserved_label_series"], 0, "counted, not merged and not silent")

    def test_a_page_that_does_not_serve_some_names_is_a_complete_capture_with_those_names_missing(self):
        """Another exporter version may rename or drop a family: the collector records what is served, lists the rest as missing and its
        capture stays complete; families it does not know are ignored."""
        dropped = ("agent_xfer_post_time_total", "agent_telemetry_events_dropped_total", "agent_errors_total")
        text = "".join(line + "\n" for line in exposition([agent()]).splitlines()
                       if not any(line.startswith(name) or line.startswith(f"# TYPE {name}") for name in dropped))
        text += '# TYPE agent_brand_new_total counter\nagent_brand_new_total{pid="10"} 5\n'
        endpoint = Endpoint([agent()])
        self.addCleanup(endpoint.close)
        endpoint.render = lambda: text  # noqa: E731
        endpoint.server.render = endpoint.render
        rows, summary = self.run_collector(endpoint, seconds=0.5)
        self.assertTrue(summary["observation_complete"], summary)
        self.assertEqual(summary["reasons"], [])
        found = nixl.samples(rows, "nixl-n")
        self.assertTrue(found and all(s["complete"] for s in found))
        (only,) = found[-1]["agents"].values()
        self.assertNotIn("xfer_post_time_us", only["fields"])
        self.assertIn("rx_bytes", only["fields"])
        self.assertEqual(sorted(nixl.unavailable(only)), ["errors", "events_dropped", "xfer_post_time_us"])

    def test_boundary_requests_are_answered_by_a_later_pass_with_an_append_only_marker(self):
        endpoint = Endpoint([agent("u", 10)])
        self.addCleanup(endpoint.close)
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, True))
        boundaries = native().Boundaries(directory)
        (directory / "base.json").write_text(json.dumps({"name": "baseline"}))
        (directory / "bad name.json").write_text(json.dumps({"name": "baseline"}))
        (directory / "worse.json").write_text("{not json")
        rows, _ = self.run_collector(endpoint, seconds=0.8, boundaries=boundaries)
        markers = [r for r in rows if r.get("type") == "boundary"]
        self.assertEqual([(m["request_id"], m["name"]) for m in markers], [("base", "baseline")])
        marker = markers[0]
        self.assertEqual(set(marker["samples"]), {"nixl-n"})
        self.assertTrue(marker["pass"]["complete"] and marker["pass"]["targets_complete"]["nixl-n"])
        self.assertGreater(marker["pass"]["begin_ns"], marker["detected_ns"], "a pass that began after the request")
        self.assertEqual(boundaries.invalid, 2)
        # the marker names samples that exist, and every record is still in sequence order
        self.assertIn(marker["samples"]["nixl-n"], {r["sample_id"] for r in rows if r.get("type") == "event"})
        sequences = [r["sequence"] for r in rows if r.get("type") == "event"]
        self.assertEqual(sequences, sorted(set(sequences)))

    def test_a_request_that_arrives_between_passes_is_never_answered_by_an_earlier_pass(self):
        module = native()
        capture = mock.Mock()
        collector_ = module.Collector.__new__(module.Collector)
        collector_.capture = capture
        collector_.last_pass = {"begin_ns": 100, "end_ns": 200, "complete": True, "targets_complete": {"t": True},
                                "samples": {"t": "s1"}}
        early = {"id": "a", "name": "baseline", "detected_ns": 150, "answered": False}
        late = {"id": "b", "name": "baseline", "detected_ns": 50, "answered": False}
        collector_.answer([early, late])
        self.assertFalse(early["answered"])
        self.assertTrue(late["answered"])
        self.assertEqual(capture.write.call_count, 1)
        self.assertEqual(capture.write.call_args.args[0]["request_id"], "b")


# --------------------------------------------------------------------------- the pure rules
TABLE_TP1 = [{"pid": 51, "start": 100, "title": "VLLM::EngineCore"}]
TABLE_TP2 = [{"pid": 50, "start": 99, "title": "VLLM::EngineCore"}, {"pid": 61, "start": 110, "title": "VLLM::Worker_TP0"},
             {"pid": 62, "start": 111, "title": "VLLM::Worker_TP1"}]


class Rules(unittest.TestCase):
    def agents(self, *items) -> dict:
        return nixl.series_agents(native().parse(exposition(list(items)), native().Limits(), "nixl")[0])

    def test_the_inventory_of_tp1_is_the_engine_process_and_of_tp2_the_workers(self):
        self.assertEqual(nixl.expected_agents(TABLE_TP1, 1),
                         {51: {"rank": 0, "start": 100, "title": "VLLM::EngineCore", "host": None}})
        found = nixl.expected_agents(TABLE_TP2, 2)
        self.assertEqual({pid: e["rank"] for pid, e in found.items()}, {61: 0, 62: 1})
        for table, tp, message in (
                (TABLE_TP2, 1, "TP1 expects one engine process and no worker"),
                (TABLE_TP1, 2, "TP2 expects one worker for each rank"),
                (TABLE_TP2[:2], 2, "TP2 expects one worker for each rank"),
                ([*TABLE_TP2, {"pid": 70, "start": 1, "title": "VLLM::Worker_TP1"}], 2, "one worker for each rank"),
                ([{"pid": 5, "start": 1, "title": "VLLM::Worker_DP1_TP0"}], 1, "data parallelism is not qualified"),
                ([], 1, "TP1 expects")):
            with self.assertRaisesRegex(nixl.InventoryError, message):
                nixl.expected_agents(table, tp)

    def test_every_rule_of_agent_problems(self):
        expected = nixl.expected_agents(TABLE_TP2, 2)
        good = self.agents(agent("u1", 61), agent("u2", 62))
        self.assertEqual(nixl.agent_problems(expected, good, {0: 2048, 1: 2048}), [])
        cases = {
            "unexpected agent": (self.agents(agent("u1", 61), agent("u2", 62), agent("u3", 99)), "unexpected agent(s) in pid(s) ['99']"),
            "missing agent": (self.agents(agent("u1", 61)), "missing in pid(s) ['62']"),
            "ambiguous join": (self.agents(agent("u1", 61), agent("u2", 62), agent("u9", 62, 1)), "owns 2 agents"),
            "second instance": (self.agents(agent("u1", 61, 1), agent("u2", 62)), "agent_instance '1'"),
            "nothing registered": (self.agents(agent("u1", 61, memory_registered_bytes=0), agent("u2", 62)), "nothing registered"),
            "no agents": ({}, "no agent is served"),
        }
        for what, (agents, fragment) in cases.items():
            self.assertTrue(any(fragment in p for p in nixl.agent_problems(expected, agents)), (what, nixl.agent_problems(expected, agents)))
        # A series the exporter does not serve is unavailable, never a problem; one it serves must be a number.
        short = self.agents(agent("u1", 61), agent("u2", 62))
        del next(iter(short.values()))["fields"]["rx_bytes"]
        self.assertEqual(nixl.agent_problems(expected, short), [])
        self.assertEqual(nixl.unavailable(next(iter(short.values()))), ["rx_bytes"])
        fewer = self.agents(agent("u1", 61), agent("u2", 62))
        del next(iter(fewer.values()))["errors"]["backend"]
        self.assertEqual(nixl.agent_problems(expected, fewer), [])
        broken = self.agents(agent("u1", 61), agent("u2", 62))
        next(iter(broken.values()))["fields"]["rx_bytes"] = float("nan")
        self.assertTrue(any("served but not a finite number" in p for p in nixl.agent_problems(expected, broken)))
        small = self.agents(agent("u1", 61, memory_registered_bytes=100), agent("u2", 62))
        self.assertTrue(any("less than rank 0's 2048 B of KV" in p for p in nixl.agent_problems(expected, small, {0: 2048})))
        partial = self.agents(agent("u1", 61), agent("u2", 62))
        next(iter(partial.values()))["partial"] = True
        self.assertTrue(any("a partial sample" in p for p in nixl.agent_problems(expected, partial)))

    def test_the_agent_identity_is_the_whole_label_set_and_distinct_hosts_never_collapse(self):
        expected = nixl.expected_agents([{"pid": 51, "start": 1, "title": "VLLM::EngineCore", "host": "node-a"}], 1)
        a = agent("u", 51)
        b = agent("u", 51, memory_registered_bytes=777)
        text = exposition([a], host="node-a") + exposition([b], host="node-b")
        agents = nixl.series_agents(native().parse(text, native().Limits(), "nixl")[0])
        self.assertEqual(len(agents), 2, "two hostnames are two agents, not one")
        found = nixl.agent_problems(expected, agents)
        self.assertTrue(any("owns 2 agents" in p for p in found), found)
        self.assertTrue(any("different hostnames" in p for p in found), found)
        # the retained labels are part of what a read says
        one = nixl.series_agents(native().parse(exposition([agent("u", 51, rank_label="0")], host="node-a"), native().Limits(), "nixl")[0])
        key = next(iter(one))
        self.assertEqual(key, "agent_instance=0|agent_name=u|hostname=node-a|local_rank=0|pid=51")
        self.assertEqual(nixl.values(one)[key]["labels"]["hostname"], "node-a")
        other = nixl.series_agents(native().parse(exposition([agent("u", 51, rank_label="0")], host="node-b"), native().Limits(), "nixl")[0])
        self.assertNotEqual(nixl.values(one), nixl.values(other))
        # the node's own hostname is part of the inventory
        wrong = nixl.series_agents(native().parse(exposition([agent("u", 51)], host="node-b"), native().Limits(), "nixl")[0])
        self.assertTrue(any("not the node's 'node-a'" in p for p in nixl.agent_problems(expected, wrong)))
        self.assertEqual(nixl.agent_problems(expected, nixl.series_agents(
            native().parse(exposition([agent("u", 51)], host="node-a"), native().Limits(), "nixl")[0])), [])

    def test_an_identity_that_changes_between_the_boundaries_is_a_problem_not_a_merged_rank(self):
        expected = nixl.expected_agents(TABLE_TP1, 1)
        base = nixl.series_agents(native().parse(exposition([agent("u", 51)], host="a"), native().Limits(), "nixl")[0])
        for changed in (exposition([agent("u", 51)], host="b"), exposition([agent("v", 51)], host="a"),
                        exposition([agent("u", 51, rank_label="1")], host="a")):
            final = nixl.series_agents(native().parse(changed, native().Limits(), "nixl")[0])
            found = nixl.workload_problems("prefill", expected, base, final, {})
            self.assertTrue(any("identity changed between the boundaries" in p for p in found), found)

    def sample(self, begin, end, **fields):
        return {"id": f"s{begin}", "begin_ns": begin, "end_ns": end, "complete": True,
                "agents": {"1/0/a": {"fields": {n: 0 for n in nixl.WATCHED} | fields, "errors": {}}}}

    def test_settling_needs_a_stretch_after_the_workload_of_unchanged_counters_long_enough(self):
        second = 1_000_000_000
        steady = [self.sample(i * 200_000_000, i * 200_000_000 + 10, rx_requests=8) for i in range(10)]
        ok, stretch = nixl.settled(steady, 0)
        self.assertTrue(ok and len(stretch) >= 3)
        # stable but stale: the stretch began before the workload ended
        self.assertFalse(nixl.settled(steady, 10 * second)[0])
        self.assertFalse(nixl.settled(steady, 1_700_000_000)[0], "a stretch too short after the cut")
        moving = [self.sample(i * 200_000_000, i * 200_000_000 + 10, rx_requests=i) for i in range(10)]
        self.assertFalse(nixl.settled(moving, 0)[0])
        # it settled, but only the last samples count: an earlier change does not hurt
        late = moving[:5] + [self.sample(i * 200_000_000, i * 200_000_000 + 10, rx_requests=9) for i in range(5, 12)]
        self.assertTrue(nixl.settled(late, 0)[0])
        partial = [dict(s, complete=False) for s in steady]
        self.assertFalse(nixl.settled(partial, 0)[0])

    def test_step_log_durations_are_converted_back_to_integer_microseconds_before_summing(self):
        line = json.dumps({"kv_transfers": {"load": {"each": [[100, 0.000123, 0.000045, 3], [50, 0.001001, 0.000002, 3]]}}})
        found = nixl.step_loads(line + "\n" + json.dumps({"step": 1}))
        self.assertEqual(found, {"requests": 2, "bytes": 150, "xfer_us": 123 + 1001, "post_us": 45 + 2})
        self.assertEqual(nixl.layout_bytes(json.dumps({"kv_layout": {"bytes": 77}})), 77)

    def test_a_version_that_renames_drops_or_adds_names_is_read_for_what_it_serves(self):
        """Portability: nothing is required by name. Dropped fields, a missing ``agent_instance`` label, no error series at all and extra
        statuses are unavailable data; what is served is still checked."""
        expected = nixl.expected_agents(TABLE_TP1, 1)

        def thin(a, drop=("xfer_post_time_us", "events_dropped", "memory_deregistered_bytes"), errors=None, label=False):
            for name in drop:
                del a["fields"][name]
            a["errors"] = {} if errors is None else errors
            return a

        base = self.agents(thin(agent("u", 51)))
        final = self.agents(thin(agent("u", 51, rx_requests=2, rx_bytes=200, xfer_time_us=30)))
        loads = {0: {"requests": 2, "bytes": 200, "xfer_us": 30, "post_us": 7}}
        self.assertEqual(nixl.agent_problems(expected, base), [])
        self.assertEqual(nixl.workload_problems("decode", expected, base, final, loads), [], "post time is unavailable, not a mismatch")
        self.assertEqual(sorted(nixl.unavailable(next(iter(final.values())))),
                         ["errors", "events_dropped", "memory_deregistered_bytes", "xfer_post_time_us"])
        # what IS served is still compared, and a failure is still a failure
        wrong = {0: {**loads[0], "bytes": 201}}
        self.assertTrue(any("rx_bytes changed by 200" in p for p in nixl.workload_problems("decode", expected, base, final, wrong)))
        # an extra status of another version moves a counter: reported; one served on one side only is not compared
        more = self.agents(thin(agent("u", 51, rx_requests=2, rx_bytes=200, xfer_time_us=30), errors={"new_status": 1.0}))
        before = self.agents(thin(agent("u", 51), errors={"new_status": 0.0}))
        found = nixl.workload_problems("decode", expected, before, more, loads)
        self.assertTrue(any("errors reported by NIXL" in p and "new_status" in p for p in found), found)
        one_sided = nixl.workload_problems("decode", expected, base, more, loads)
        self.assertEqual(one_sided, [])
        # an exporter without the agent_instance label is not an inventory failure
        text = exposition([agent("u", 51)]).replace(',agent_instance="0"', "")
        agents = nixl.series_agents(native().parse(text, native().Limits(), "nixl")[0])
        self.assertTrue(agents and not any("agent_instance" in p for p in nixl.agent_problems(expected, agents)))

    def test_workload_problems_each_branch(self):
        expected = nixl.expected_agents(TABLE_TP1, 1)
        base = self.agents(agent("u", 51))
        loads = {0: {"requests": 2, "bytes": 200, "xfer_us": 30, "post_us": 7}}

        def final(**fields):
            return self.agents(agent("u", 51, rx_requests=2, rx_bytes=200, xfer_time_us=30, xfer_post_time_us=7, **fields))

        self.assertEqual(nixl.workload_problems("decode", expected, base, final(), loads), [])
        for fields, fragment in (({"events_dropped": 1}, "telemetry event(s) dropped"),
                                 ({"tx_requests": 1}, "a decode rank transmitted"),
                                 ({"memory_registered_bytes": 4096}, "registration changed")):
            self.assertTrue(any(fragment in p for p in nixl.workload_problems("decode", expected, base, final(**fields), loads)), fields)
        for key, fragment in (("requests", "rx_requests changed by 2"), ("bytes", "rx_bytes changed by 200"),
                              ("xfer_us", "xfer_time_us changed by 30"), ("post_us", "xfer_post_time_us changed by 7")):
            wrong = {0: {**loads[0], key: loads[0][key] + 1}}
            self.assertTrue(any(fragment in p for p in nixl.workload_problems("decode", expected, base, final(), wrong)), key)
        errored = final()
        next(iter(errored.values()))["errors"]["backend"] = 1
        self.assertTrue(any("errors reported by NIXL" in p for p in nixl.workload_problems("decode", expected, base, errored, loads)))
        self.assertTrue(any("no step log" in p for p in nixl.workload_problems("decode", expected, base, final(), {})))
        self.assertTrue(any("a prefill rank moved bytes" in p for p in nixl.workload_problems("prefill", expected, base, final(), {})))
        self.assertEqual(nixl.workload_problems("prefill", expected, base, base, {}), [])
        self.assertTrue(any("no agent at both boundaries" in p for p in nixl.workload_problems("decode", expected, {}, final(), loads)))


# --------------------------------------------------------------------------- the protocol through the real helpers
def local_exec(container, command, **kwargs):
    command = [sys.executable if part == pd.PYTHON else part for part in command]
    done = subprocess.run(command, capture_output=True, text=True, check=kwargs.get("check", True))
    return done.stdout


STEP_HEAD = json.dumps({"kv_layout": {"bytes": 2048, "tp_rank": 0}})


def step_log(requests: int, bytes_each=100, xfer_us=10, post_us=2, ends_at=2.0) -> str:
    """A rank's step log: its layout, and one step that ran the requests (its instants on the monotonic clock, far earlier
    than the collector's samples unless a test moves them)."""
    entries = [[bytes_each, xfer_us / 1e6, post_us / 1e6, 3] for _ in range(requests)]
    step = {"step": {"requests": 1}, "t_start": ends_at - 1.0, "t_deadline": ends_at,
            "kv_transfers": {"load": {"each": entries}}}
    return STEP_HEAD + "\n" + json.dumps(step) + "\n"


class Rig(unittest.TestCase):
    """One decode node: the real collector as a process on a local page, the real ``NodeCollector`` methods, and
    ``collector.check`` over the saved layout. The container's commands run locally; the process table and the step
    logs are the test's."""

    TP = 1
    ROLE = "decode"
    THIN = False  # an exporter that does not serve some names (another version)
    ZONE: tuple = ()  # families an exporter serves with a label this reader does not know

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.table = TABLE_TP1 if self.TP == 1 else TABLE_TP2
        self.steps = {rank: step_log(0) for rank in range(self.TP)}
        agents = [agent("u0", 51)] if self.TP == 1 else [agent("u0", 61), agent("u1", 62)]
        if self.THIN:
            for a in agents:
                for name in ("xfer_post_time_us", "events_dropped", "memory_deregistered_bytes"):
                    del a["fields"][name]
                a["errors"] = {}
        self.endpoint = Endpoint(agents)
        self.endpoint.zone = tuple(self.ZONE)
        self.addCleanup(self.endpoint.close)
        side = pd.Side("n", "data0")
        self.run = pd.Run(name="t", topology="t", model=Path("/m"), prefill=pd.Side("p", "data0"),
                          decode=side, client="c", control_network="control", router=pd.Router("gw"),
                          server=pd.Server(connector="nixl", nixl_backend="ucx", timing="linear", tensor_parallel_size=self.TP),
                          collector=pd.Collector(interval_s=0.1, startup_timeout_s=3, shutdown_timeout_s=5, tent=False,
                                                 nixl_telemetry=True))
        self.patches = [mock.patch.object(collector.topology, "exec_in", local_exec),
                        mock.patch.object(pd, "NIXL_TELEMETRY_PORT", self.endpoint.port),
                        mock.patch.object(collector, "BOUNDARY_DIR", str(self.root / "boundaries")),
                        mock.patch.object(nixl, "FINAL_TIMEOUT_S", 4.0),
                        mock.patch.object(collector.NodeCollector, "_table", lambda one: list(self.table)),
                        mock.patch.object(collector.NodeCollector, "_step_text", lambda one, rank: self.steps[rank])]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)
        (self.root / "boundaries").mkdir()
        self.run_id = "2d9d2c75-4f43-4a0e-9a16-0f3a1b7b8a13"
        self.stopfile = self.root / "stop"
        manifest = {"targets": [{"id": "nixl-n", "role": "decode", "source": "nixl",
                                 "endpoint": f"127.0.0.1:{self.endpoint.port}"}]}
        (self.root / "targets.json").write_text(json.dumps(manifest))
        command = [sys.executable, str(collector.SOURCE), "--run-id", self.run_id, "--output-dir", str(self.root / "out"),
                   "--manifest", str(self.root / "targets.json"), "--stop-file", str(self.stopfile),
                   "--boundary-dir", str(self.root / "boundaries"), "--interval-s", "0.1", "--startup-timeout-s", "5",
                   "--shutdown-timeout-s", "5"]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.addCleanup(self._finish, self.process, self.stopfile)
        self.node = self.make_node()

    def make_node(self):
        state = {"target": "nixl-n", "role": self.ROLE, "tp": self.TP,
                 "steps": {rank: f"/steps{rank}" for rank in range(self.TP)}}
        node = collector.NodeCollector("n", "local", self.run_id, self.run.collector, self.root / "node",
                                       ("nixl-n",), {"nixl-n": "nixl"}, state)
        node.directory.mkdir(parents=True, exist_ok=True)
        node._events = self.events
        return node

    def events(self):
        rows = []
        for path in (self.root / "out").glob("*/*/events.jsonl"):
            for line in path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
        return rows

    def finish(self):
        self._finish(self.process, self.stopfile)

    @staticmethod
    def _finish(process, stopfile):
        stopfile.touch()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        if process.stdout:
            process.stdout.close()

    def workload(self, requests=2, after=0.0):
        """The decode agent's counters as the step log says a workload moved them (the flush may arrive late)."""
        self.steps = {rank: step_log(requests) for rank in range(self.TP)}

        def move(agents):
            for a in agents:
                a["fields"].update(rx_requests=requests, rx_bytes=100 * requests, xfer_time_us=10 * requests,
                                   xfer_post_time_us=2 * requests)

        self.endpoint.change(move, after)

    def saved(self):
        """Lay the node's capture out as the harness copies it, for ``check``."""
        self.finish()
        target = self.root / "result" / "collector" / "n"
        (target / "capture").mkdir(parents=True)
        subprocess.run(["cp", "-r", f"{self.root}/out/.", str(target / "capture")], check=True)
        (target / "boundaries.json").write_text((self.root / "node" / "boundaries.json").read_text())
        (target / "targets.json").write_text(json.dumps({"targets": collector.targets(self.run, "n")}))
        (target / "collector.json").write_text(json.dumps({"run_id": self.run_id, "node": "n", "stopped": "stop_file",
                                                           "exited": True, "errors": [],
                                                           "settings": self.run.collector.__dict__}))
        for rank, text in self.steps.items():
            (self.root / "result" / self.run.decode.step_log("decode", 0, rank)).write_text(text)
        return self.root / "result"

    def verdict(self, output):
        """The check of this one decode node (the run's prefill side has no node here)."""
        run = pd.Run(**{**{f: getattr(self.run, f) for f in self.run.__dataclass_fields__},
                        "prefill": pd.Side([], "data0")})
        verdict = {"problems": []}
        collector.check(output, run, {}, verdict)
        return verdict


class HappyPath(Rig):
    def test_baseline_workload_final_and_the_check_over_what_was_saved(self):
        self.node.nixl_baseline()
        baseline = self.node.boundaries["baseline"]
        self.assertEqual(baseline["outcome"], "ok")
        accepted = baseline["accepted"]
        self.assertLess(accepted["marker"]["pass"]["begin_ns"], accepted["harness"]["end_ns"] + 10 ** 12)
        self.assertEqual(accepted["marker"]["samples"]["nixl-n"], accepted["sample"])
        self.assertEqual(self.node.boundaries["expected"]["51"]["rank"], 0)
        self.workload(2)
        self.node.nixl_final()
        final = self.node.boundaries["final"]
        self.assertEqual((final["outcome"], final["problems"]), ("ok", []), final.get("problems"))
        self.assertGreaterEqual(len(final["stretch"]), 3)
        verdict = self.verdict(self.saved())
        self.assertEqual(verdict["problems"], [])
        report = verdict["collector"]["n"]["nixl"]
        self.assertEqual((report["baseline"], report["final"], report["workload"]), ("ok", "ok", {0: 2}))
        self.assertEqual(report["unavailable"], {})
        self.assertEqual(report["coverage"][0]["unavailable"], [])
        self.assertIn("xfer_post_time_us", report["coverage"][0]["validated"])


class UnknownLabels(Rig):
    """Portability: an exporter whose series carry a label this reader does not know. Both readers (the collector and the harness's own
    scrape) apply the same rule, so the boundary is not rejected for a difference between them; what neither can read is unavailable
    coverage, and the genuine missing-worker and identity checks still apply."""

    ZONE = ("agent_rx_bytes_total", "agent_tx_bytes_total", "agent_xfer_time_total")

    def workload(self, requests=2, after=0.0):
        self.steps = {rank: step_log(requests) for rank in range(self.TP)}

        def move(agents):
            for a in agents:
                a["fields"].update(rx_requests=requests, rx_bytes=100 * requests, xfer_time_us=10 * requests,
                                   xfer_post_time_us=2 * requests)

        self.endpoint.change(move, after)

    def test_three_series_with_an_unknown_label_leave_the_boundaries_accepted_and_their_fields_unavailable(self):
        self.node.nixl_baseline()
        self.assertEqual(self.node.boundaries["baseline"]["outcome"], "ok", self.node.boundaries["baseline"])
        accepted = self.node.boundaries["baseline"]["accepted"]
        self.assertEqual(nixl.values(accepted["harness"]["agents"]), nixl.values(accepted["agents"]), "the two readers agree")
        self.workload(2)
        # the step log says two reads; rx_bytes is unreadable (unknown label), so its comparison is skipped, not failed
        self.node.nixl_final()
        final = self.node.boundaries["final"]
        self.assertEqual((final["outcome"], final["problems"]), ("ok", []), final.get("problems"))
        verdict = self.verdict(self.saved())
        self.assertEqual(verdict["problems"], [], "the saved-evidence validator agrees as well")
        report = verdict["collector"]["n"]["nixl"]
        self.assertEqual(report["unavailable"], {0: ["rx_bytes", "tx_bytes", "xfer_time_us"]})
        coverage = report["coverage"][0]
        self.assertEqual(coverage["unavailable"], ["rx_bytes", "tx_bytes", "xfer_time_us"])
        self.assertIn("rx_requests", coverage["validated"])
        self.assertNotIn("rx_bytes", coverage["validated"])


class UnknownLabelsEverywhere(UnknownLabels):
    ZONE = ("all",)

    def test_three_series_with_an_unknown_label_leave_the_boundaries_accepted_and_their_fields_unavailable(self):
        pass

    def test_when_every_series_has_an_unknown_label_nothing_is_observable_and_that_is_unavailable_coverage_not_a_failed_run(self):
        self.node.nixl_baseline()  # does not raise: the workload may start
        baseline = self.node.boundaries["baseline"]
        self.assertEqual(baseline["outcome"], "unavailable_schema")
        self.assertGreater(baseline["accepted"]["unserved_label_series"], 0)
        self.workload(2)
        self.node.nixl_final()
        verdict = self.verdict(self.saved())
        self.assertEqual(verdict["problems"], [])
        report = verdict["collector"]["n"]["nixl"]
        self.assertEqual((report["baseline"], report["final"]), ("unavailable_schema", "unavailable_schema"))
        self.assertGreater(report["unavailable_schema"]["unserved_label_series"], 0)
        # unavailable metrics do not skip the rest: a final boundary was recorded, the process table read again
        final = self.node.boundaries["final"]
        self.assertEqual(final["outcome"], "unavailable_schema")
        self.assertTrue(final["table_end"] and final["attempts"] and final["after_ns"])


class UnknownLabelsKeepTheRestOfTheValidation(Rig):
    """Review of agent-3#57: unavailable metrics must not bypass what is still observable: process identity, the boundary evidence
    and a failed pass."""

    TP = 2
    ZONE = ("all",)

    def test_a_restart_during_the_run_is_caught_although_the_metrics_were_unreadable(self):
        self.node.nixl_baseline()
        self.assertEqual(self.node.boundaries["baseline"]["outcome"], "unavailable_schema")
        self.workload(2)
        self.table = [dict(row, start=row["start"] + 1) if row["pid"] == 62 else dict(row) for row in self.table]  # worker 62 restarted
        self.node.nixl_final()
        final = self.node.boundaries["final"]
        self.assertEqual(final["outcome"], "rejected")
        self.assertTrue(any("pid 62 is not the same process at both boundaries" in p for p in final["problems"]), final["problems"])
        verdict = self.verdict(self.saved())
        self.assertTrue(any("pid 62" in p for p in verdict["problems"]), verdict["problems"])

    def test_a_missing_baseline_marker_in_the_saved_capture_is_caught(self):
        self.node.nixl_baseline()
        self.workload(2)
        self.node.nixl_final()
        request = self.node.boundaries["baseline"]["accepted"]["request"]
        path = self.saved()
        for events in (path / "collector" / "n" / "capture").glob("*/*/events.jsonl"):
            rows = [json.loads(line) for line in events.read_text().splitlines()]
            events.write_text("".join(json.dumps(r) + "\n" for r in rows
                                      if not (r.get("type") == "boundary" and r.get("request_id") == request)))
        verdict = self.verdict(path)
        self.assertTrue(any("baseline marker is missing from the capture" in p for p in verdict["problems"]), verdict["problems"])

    def test_the_samples_own_records_must_exist_an_empty_sample_is_not_absent_evidence(self):
        """Review of agent-3#58: ``sample_agents`` is empty both for a sample that saw no agent and for records that are gone."""
        self.node.nixl_baseline()
        self.workload(2)
        self.node.nixl_final()
        sample = self.node.boundaries["baseline"]["accepted"]["sample"]
        path = self.saved()
        capture = next((path / "collector" / "n" / "capture").glob("*/*/events.jsonl"))
        rows = [json.loads(line) for line in capture.read_text().splitlines()]
        kept = [r for r in rows if r.get("sample_id") != sample]
        self.assertLess(len(kept), len(rows), "the sample had records to remove")
        capture.write_text("".join(json.dumps(r) + "\n" for r in kept))
        verdict = self.verdict(path)
        self.assertTrue(any("records are absent from the capture" in p for p in verdict["problems"]), verdict["problems"])

    def test_one_kind_of_the_samples_records_missing_is_caught_too(self):
        self.node.nixl_baseline()
        self.workload(2)
        self.node.nixl_final()
        sample = self.node.boundaries["baseline"]["accepted"]["sample"]
        path = self.saved()
        capture = next((path / "collector" / "n" / "capture").glob("*/*/events.jsonl"))
        rows = [json.loads(line) for line in capture.read_text().splitlines()]
        capture.write_text("".join(json.dumps(r) + "\n" for r in rows
                                   if not (r.get("sample_id") == sample and r.get("kind_id") == 28)))
        verdict = self.verdict(path)
        self.assertTrue(any("does not have both kinds" in p for p in verdict["problems"]), verdict["problems"])

    def test_a_reading_that_recovers_by_the_end_is_not_an_error_and_only_the_deltas_are_unavailable(self):
        self.node.nixl_baseline()
        self.assertEqual(self.node.boundaries["baseline"]["outcome"], "unavailable_schema")
        self.endpoint.zone = ()  # the exporter's labels are readable from here on
        self.workload(2)
        self.node.nixl_final()
        final = self.node.boundaries["final"]
        self.assertEqual((final["outcome"], final["problems"]), ("readable_after_unavailable", []), final["problems"])
        self.assertEqual(len(final["accepted"]["agents"]), 2, "the observed final values are kept")
        verdict = self.verdict(self.saved())
        self.assertEqual(verdict["problems"], [], verdict["problems"])
        report = verdict["collector"]["n"]["nixl"]
        self.assertEqual((report["baseline"], report["final"]), ("unavailable_schema", "readable_after_unavailable"))
        self.assertIn("unavailable", report["metric_deltas"])
        self.assertEqual(sorted(report["final_unavailable"]), ["61", "62"])

    def test_a_recovered_reading_still_gets_the_normal_identity_checks(self):
        self.node.nixl_baseline()
        self.endpoint.zone = ()
        self.workload(2)
        self.table = [dict(row, start=row["start"] + 1) if row["pid"] == 62 else dict(row) for row in self.table]
        self.node.nixl_final()
        self.assertEqual(self.node.boundaries["final"]["outcome"], "rejected")
        verdict = self.verdict(self.saved())
        self.assertTrue(any("pid 62" in p for p in verdict["problems"]), verdict["problems"])

    def test_a_pass_that_did_not_complete_keeps_its_error_whatever_the_schema(self):
        original = self.node._wait_marker

        def incomplete(request):
            marker, records = original(request)
            if marker is not None:
                marker["pass"]["targets_complete"][self.node.nixl_state["target"]] = False
            return marker, records

        self.node._wait_marker = incomplete
        with self.assertRaises(RuntimeError):
            self.node.nixl_baseline()
        baseline = self.node.boundaries["baseline"]
        self.assertNotEqual(baseline["outcome"], "unavailable_schema")
        self.assertTrue(any("the pass did not complete this target" in p for p in baseline["problems"]), baseline["problems"])

    def test_the_clean_unreadable_run_is_still_clean(self):
        self.node.nixl_baseline()
        self.workload(2)
        self.node.nixl_final()
        verdict = self.verdict(self.saved())
        self.assertEqual(verdict["problems"], [], verdict["problems"])


class UnknownLabelsOnOneWorker(Rig):
    TP = 2
    ZONE = ("agent:u1",)

    def test_a_worker_whose_every_series_is_unreadable_is_still_a_missing_worker_while_the_other_is_served(self):
        """Unavailable coverage is for a schema nothing can be read from; with one worker served, the other is missing."""
        with self.assertRaises(RuntimeError):
            self.node.nixl_baseline()
        baseline = self.node.boundaries["baseline"]
        self.assertNotEqual(baseline["outcome"], "unavailable_schema")
        self.assertTrue(any("missing in pid(s) ['62']" in p for p in baseline["problems"]), baseline["problems"])


class OneSidedStatuses(Rig):
    """Portability: an exporter that serves other error statuses at the end than at the start: the statuses served at both are
    compared; one that appeared or disappeared has its change reported unavailable, not read as a change from or to zero."""

    def workload(self, requests=2, after=0.0):
        self.steps = {rank: step_log(requests) for rank in range(self.TP)}

        def move(agents):
            for a in agents:
                a["fields"].update(rx_requests=requests, rx_bytes=100 * requests, xfer_time_us=10 * requests,
                                   xfer_post_time_us=2 * requests)
                a["errors"] = {"backend": 0, "new_status": 7.0}

        self.endpoint.change(move, after)

    def test_appearing_and_disappearing_statuses_are_per_status_coverage_and_a_served_change_still_fails(self):
        self.node.nixl_baseline()
        self.workload(2)
        self.node.nixl_final()
        verdict = self.verdict(self.saved())
        self.assertEqual(verdict["problems"], [])
        report = verdict["collector"]["n"]["nixl"]
        errors = report["coverage"][0]["errors"]
        self.assertEqual(errors["compared"], ["backend"])
        self.assertEqual(errors["appeared"], {"new_status": 7.0})
        self.assertIn("not_found", errors["disappeared"])
        self.assertIn("errors[new_status] (appeared)", report["unavailable"][0])
        self.assertIn("errors[not_found] (disappeared)", report["unavailable"][0])
        # a status served at both boundaries that moved is still a failure
        problems = nixl.workload_problems(
            "decode", nixl.expected_agents(TABLE_TP1, 1),
            {"a": {"labels": {"pid": "51", "hostname": "n", "agent_name": "u", "agent_instance": "0"}, "fields": {}, "errors": {"backend": 0.0}, "partial": False}},
            {"a": {"labels": {"pid": "51", "hostname": "n", "agent_name": "u", "agent_instance": "0"}, "fields": {}, "errors": {"backend": 2.0}, "partial": False}},
            {0: {"requests": 0, "bytes": 0, "xfer_us": 0, "post_us": 0}})
        self.assertTrue(any("errors reported by NIXL" in p and "backend" in p for p in problems), problems)


class ThinExporter(Rig):
    """Portability: an exporter that serves fewer names (no post time, no dropped-event counter, no deregistration series, no
    error series) is read for what it serves; the rest is reported unavailable per rank, and a comparison that was skipped is
    told apart from one that was validated."""

    THIN = True

    def workload(self, requests=2, after=0.0):
        self.steps = {rank: step_log(requests) for rank in range(self.TP)}

        def move(agents):
            for a in agents:
                a["fields"].update(rx_requests=requests, rx_bytes=100 * requests, xfer_time_us=10 * requests)

        self.endpoint.change(move, after)

    def test_a_thin_exporter_is_a_clean_run_with_per_rank_unavailable_and_validated_coverage(self):
        self.node.nixl_baseline()
        self.workload(2)
        self.node.nixl_final()
        final = self.node.boundaries["final"]
        self.assertEqual((final["outcome"], final["problems"]), ("ok", []), final.get("problems"))
        verdict = self.verdict(self.saved())
        self.assertEqual(verdict["problems"], [], "unavailable names are not a failure of the check, the collector or the capture")
        report = verdict["collector"]["n"]["nixl"]
        self.assertEqual(report["workload"], {0: 2})
        gone = ["errors", "events_dropped", "memory_deregistered_bytes", "xfer_post_time_us"]
        self.assertEqual(report["unavailable"], {0: gone})
        coverage = report["coverage"][0]
        self.assertEqual(coverage["unavailable"], gone)
        self.assertEqual(coverage["validated"], ["memory_registered_bytes", "rx_bytes", "rx_requests", "tx_bytes", "tx_requests", "xfer_time_us"])

    def test_what_a_thin_exporter_does_serve_is_still_validated(self):
        self.node.nixl_baseline()
        self.workload(2)
        self.steps = {rank: step_log(3) for rank in range(self.TP)}  # the step logs say three reads, the counters two
        self.node.nixl_final()
        verdict = self.verdict(self.saved())
        self.assertTrue(any("rx_requests changed by 2" in p for p in verdict["problems"]), verdict["problems"])

    def test_both_independent_reads_keep_their_instants_and_the_selected_sample(self):
        self.node.nixl_baseline()
        accepted = self.node.boundaries["baseline"]["accepted"]
        self.assertEqual(set(accepted["harness"]), {"begin_ns", "end_ns", "error", "agents"})
        self.assertLessEqual(accepted["harness"]["begin_ns"], accepted["harness"]["end_ns"])
        self.assertTrue(accepted["sample"] and accepted["marker"]["request_id"] == accepted["request"])


class DelayedAndStalled(Rig):
    def test_a_flush_that_arrives_late_is_waited_for_and_a_pre_arrival_stretch_is_not_accepted(self):
        self.node.nixl_baseline()
        self.workload(2, after=1.3)  # the transfers completed; the counters get there 1.3 s later
        started = time.monotonic()
        self.node.nixl_final()
        final = self.node.boundaries["final"]
        self.assertEqual(final["outcome"], "ok", final.get("problems"))
        self.assertGreaterEqual(time.monotonic() - started, 1.2, "it must not accept the stale stretch before the flush")
        self.assertEqual(self.verdict(self.saved())["problems"], [])

    def test_a_stalled_exporter_is_a_final_timeout_with_its_evidence_and_fails_the_validation(self):
        self.node.nixl_baseline()
        self.workload(2, after=3600)  # the counters never arrive, though the step log has the transfers
        with mock.patch.object(nixl, "FINAL_TIMEOUT_S", 2.0):
            self.node.nixl_final()
        final = self.node.boundaries["final"]
        self.assertEqual(final["outcome"], "final_timeout")
        self.assertTrue(any("rx_requests changed by 0, the step log's transfers by 2" in p for p in final["problems"]), final["problems"])
        verdict = self.verdict(self.saved())
        self.assertTrue(any("the final sample was not accepted (final_timeout" in p for p in verdict["problems"]), verdict["problems"])

    def test_a_workload_the_counters_disagree_with_stays_a_problem_even_when_stable(self):
        self.node.nixl_baseline()
        self.steps = {0: step_log(5)}  # five transfers logged; the counters stay at 0, perfectly stable
        with mock.patch.object(nixl, "FINAL_TIMEOUT_S", 1.5):
            self.node.nixl_final()
        self.assertEqual(self.node.boundaries["final"]["outcome"], "final_timeout")


class Inventory(Rig):
    TP = 2

    def test_a_missing_agent_at_the_baseline_fails_the_run_with_the_missing_set(self):
        self.endpoint.change(lambda agents: agents.pop())
        self.run.collector = pd.Collector(interval_s=0.1, startup_timeout_s=1, tent=False, nixl_telemetry=True)
        self.node.config = self.run.collector
        with self.assertRaisesRegex(RuntimeError, "no usable NIXL telemetry baseline"):
            self.node.nixl_baseline()
        record = self.node.boundaries["baseline"]
        self.assertEqual(record["outcome"], "baseline_timeout")
        self.assertTrue(any("missing in pid(s) ['62']" in p for p in record["problems"]), record["problems"])
        self.assertTrue((self.node.directory / "boundaries.json").is_file())

    def test_a_late_agent_is_waited_for_and_the_baseline_waits_for_it(self):
        self.endpoint.change(lambda agents: agents.pop())
        self.endpoint.change(lambda agents: agents.append(agent("u1", 62)), after=1.0)
        self.node.nixl_baseline()
        record = self.node.boundaries["baseline"]
        self.assertEqual(record["outcome"], "ok")
        self.assertGreaterEqual(len(record["attempts"]), 2)
        self.assertEqual(record["attempts"][0]["outcome"], "rejected")
        self.assertEqual(len(record["accepted"]["agents"]), 2)

    def test_an_extra_agent_and_an_ambiguous_pid_are_rejected_not_merged_into_a_rank(self):
        for extra, fragment in ((agent("u9", 99), "unexpected agent(s) in pid(s) ['99']"),
                                (agent("u9", 62, instance=1), "owns 2 agents")):
            self.setUp()
            self.endpoint.change(lambda agents, extra=extra: agents.append(extra))
            self.run.collector = pd.Collector(interval_s=0.1, startup_timeout_s=1, tent=False, nixl_telemetry=True)
            self.node.config = self.run.collector
            with self.assertRaises(RuntimeError):
                self.node.nixl_baseline()
            self.assertTrue(any(fragment in p for p in self.node.boundaries["baseline"]["problems"]), fragment)

    def test_a_restart_between_the_boundaries_a_pid_reused_with_another_start_is_rejected(self):
        self.node.nixl_baseline()
        self.workload(2)
        self.table = [dict(row, start=row["start"] + 5) if row["pid"] == 62 else row for row in TABLE_TP2]
        self.node.nixl_final()
        final = self.node.boundaries["final"]
        self.assertEqual(final["outcome"], "rejected")
        self.assertTrue(any("pid 62 is not the same process at both boundaries" in p for p in final["problems"]), final["problems"])
        verdict = self.verdict(self.saved())
        self.assertTrue(any("the final sample was not accepted (rejected" in p for p in verdict["problems"]), verdict["problems"])

    def test_the_inventory_mismatch_is_an_error_before_any_workload(self):
        self.table = TABLE_TP1  # a TP2 run showing one engine process only
        with self.assertRaisesRegex(RuntimeError, "TP2 expects one worker for each rank"):
            self.node.nixl_baseline()
        self.assertEqual(self.node.boundaries["baseline"]["outcome"], "inventory_error")


class Tp1Engine(Rig):
    def test_tp1s_agent_lives_in_the_engine_process_and_a_worker_process_for_tp1_is_an_error(self):
        self.node.nixl_baseline()
        self.assertEqual(list(self.node.boundaries["expected"]), ["51"])
        self.setUp()
        self.table = [{"pid": 51, "start": 100, "title": "VLLM::EngineCore"}, {"pid": 52, "start": 101, "title": "VLLM::Worker_TP0"}]
        with self.assertRaisesRegex(RuntimeError, "TP1 expects one engine process and no worker"):
            self.node.nixl_baseline()


class Alignment(Rig):
    def test_a_counter_that_moves_between_the_collectors_pass_and_the_harness_scrape_is_rejected_then_accepted(self):
        calls = []
        original = collector.NodeCollector._own_scrape

        def moving(one):
            calls.append(1)
            if len(calls) == 1:
                self.endpoint.change(lambda agents: agents[0]["fields"].update(memory_registered_bytes=4096))
            return original(one)

        self.endpoint.change(lambda agents: None)
        with mock.patch.object(collector.NodeCollector, "_own_scrape", moving):
            # the first own scrape reads after the change the collector's sample did not see: the values differ
            self.node.nixl_baseline()
        record = self.node.boundaries["baseline"]
        self.assertEqual(record["attempts"][0]["outcome"], "rejected")
        self.assertTrue(any("own scrape differs" in p for p in record["attempts"][0]["problems"]), record["attempts"][0]["problems"])
        self.assertEqual(record["outcome"], "ok")
        self.assertEqual(record["accepted"]["agents"][next(iter(record["accepted"]["agents"]))]["fields"]["memory_registered_bytes"], 4096)


class SavedEvidence(Rig):
    """``check`` judges the saved evidence again: tampering with any part of it is found."""

    def good(self):
        self.node.nixl_baseline()
        self.workload(2)
        self.node.nixl_final()
        self.assertEqual(self.node.boundaries["final"]["outcome"], "ok")
        return self.saved()

    def edit_boundaries(self, output, change):
        path = output / "collector" / "n" / "boundaries.json"
        saved = json.loads(path.read_text())
        change(saved)
        path.write_text(json.dumps(saved))

    def test_a_missing_marker_another_sample_and_a_harness_scrape_that_differs_are_found(self):
        output = self.good()
        self.assertEqual(self.verdict(output)["problems"], [])

        def other(saved):
            saved["final"]["accepted"]["sample"] = "elsewhere"
        self.edit_boundaries(output, other)
        self.assertTrue(any("marker is missing from the capture or names another sample" in p for p in self.verdict(output)["problems"]))
        output = self.good_again()

        def differs(saved):
            one = next(iter(saved["baseline"]["accepted"]["harness"]["agents"].values()))
            one["fields"]["rx_bytes"] = 5
        self.edit_boundaries(output, differs)
        self.assertTrue(any("own baseline scrape differs" in p for p in self.verdict(output)["problems"]))

    def good_again(self):
        self.setUp()
        return self.good()

    def test_changed_start_ticks_a_changed_step_log_and_a_prefill_that_moved_are_found(self):
        output = self.good()

        def restart(saved):
            saved["table_start"][0]["start"] += 1
        self.edit_boundaries(output, restart)
        self.assertTrue(any("is not the inventory's process at both boundaries" in p for p in self.verdict(output)["problems"]))
        output = self.good_again()
        (output / self.run.decode.step_log("decode", 0, 0)).write_text(step_log(3))
        self.assertTrue(any("rx_requests changed by 2, the step log's transfers by 3" in p for p in self.verdict(output)["problems"]))
        output = self.good_again()
        self.run = pd.Run(**{**{f: getattr(self.run, f) for f in self.run.__dataclass_fields__}})
        self.edit_boundaries(output, lambda saved: saved.update(role="prefill"))
        problems = collector_check_role(self, output, "prefill")
        self.assertTrue(any("a prefill rank moved bytes" in p for p in problems), problems)


def collector_check_role(rig, output, role):
    """The same saved evidence judged as a prefill node: the decode agents' receive counters moved."""
    run = pd.Run(**{**{f: getattr(rig.run, f) for f in rig.run.__dataclass_fields__},
                    "prefill": rig.run.decode, "decode": pd.Side([], "data0")})
    verdict = {"problems": []}
    collector.check(output, run, {}, verdict)
    return verdict["problems"]


class SavedInterval(SavedEvidence):
    """The claimed stable interval and the boundary ordering are rebuilt from the raw samples: a stored outcome proves nothing."""

    def capture_path(self, output):
        return next((output / "collector" / "n" / "capture").glob("*/*/events.jsonl"))

    def edit_capture(self, output, change):
        path = self.capture_path(output)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        change(rows)
        path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))

    def expect(self, output, fragment):
        found = self.verdict(output)["problems"]
        self.assertTrue(any(fragment in p for p in found), (fragment, found[:4]))

    def test_the_untouched_evidence_passes_and_the_stretch_is_made_of_real_consecutive_samples(self):
        output = self.good()
        self.assertEqual(self.verdict(output)["problems"], [])
        saved = json.loads((output / "collector" / "n" / "boundaries.json").read_text())
        self.assertGreaterEqual(len(saved["final"]["stretch"]), 3)
        self.assertEqual(nixl.final_interval_problems(saved, self.events_of(output), "nixl-n", 2.0), [])

    def events_of(self, output):
        return [json.loads(line) for line in self.capture_path(output).read_text().splitlines()]

    def test_samples_that_do_not_exist_a_late_cut_and_a_stretch_that_is_not_consecutive_are_found(self):
        output = self.good()
        self.edit_boundaries(output, lambda saved: saved["final"].update(stretch=["x-1", "x-2", "x-3"]))
        self.expect(output, "names samples that are not in the capture")
        output = self.good_again()
        self.edit_boundaries(output, lambda saved: saved["final"].update(after_ns=10 ** 30))
        self.expect(output, "did not begin after the recorded end of the workload")
        output = self.good_again()

        def skip(saved):
            ids = saved["final"]["stretch"]
            saved["final"]["stretch"] = [ids[0], ids[2], ids[-1]] if len(ids) > 3 else ids
        self.edit_boundaries(output, skip)
        self.expect(output, "not made of consecutive samples")

    def test_a_stretch_too_short_in_time_a_counter_that_moved_inside_it_and_the_workload_end_are_found(self):
        output = self.good()
        ordered = nixl.samples(self.events_of(output), "nixl-n")
        first = {s["id"]: i for i, s in enumerate(ordered)}[json.loads((output / "collector" / "n" / "boundaries.json").read_text())
                                                              ["final"]["stretch"][0]]
        self.edit_boundaries(output, lambda saved: saved["final"].update(stretch=[s["id"] for s in ordered[first:first + 3]]))
        self.expect(output, "less than 0.5 s")
        output = self.good_again()
        stretch = json.loads((output / "collector" / "n" / "boundaries.json").read_text())["final"]["stretch"]

        def move(rows):
            for row in rows:
                if row.get("type") == "event" and row.get("sample_id") == stretch[1] and row.get("kind_id") == 27:
                    row["fields"]["rx_requests"] = 99
        self.edit_capture(output, move)
        self.expect(output, "the counters moved inside the final stretch")
        output = self.good_again()
        (output / self.run.decode.step_log("decode", 0, 0)).write_text(step_log(2, ends_at=10 ** 12))
        self.expect(output, "began before the workload's own last step ended")

    def test_a_marker_noticed_after_its_pass_and_independent_scrapes_out_of_order_are_found(self):
        output = self.good()

        def late(rows):
            for row in rows:
                if row.get("type") == "boundary" and row["name"] == "final":
                    row["detected_ns"] = row["pass"]["end_ns"] + 10 ** 9
        self.edit_capture(output, late)
        self.expect(output, "request, pass and sample are not in order")
        output = self.good_again()
        self.edit_boundaries(output, lambda saved: saved["final"]["accepted"]["harness"].update(begin_ns=1, end_ns=2))
        self.expect(output, "own final scrape did not run after the pass")
        output = self.good_again()
        self.edit_boundaries(output, lambda saved: saved["baseline"]["accepted"]["harness"].update(begin_ns=1, end_ns=2))
        self.expect(output, "own baseline scrape did not run after the pass")
        output = self.good_again()
        self.edit_boundaries(output, lambda saved: saved["final"].update(after_ns=1))
        self.expect(output, "recorded as ending before the baseline was complete")


class Export(Rig):
    """The capture's kinds 27 and 28 and its boundary markers in the Perfetto export (the profiler package loaded by
    path: the host has no vLLM)."""

    def test_the_collectors_nixl_rows_become_counter_tracks_and_the_markers_do_not_break_the_reader(self):
        root = pd.REPOSITORY / "vllm" / "kv_profiling"
        spec = importlib.util.spec_from_file_location("kvp_t", root / "__init__.py", submodule_search_locations=[str(root)])
        package = importlib.util.module_from_spec(spec)
        sys.modules["kvp_t"] = package
        spec.loader.exec_module(package)
        from kvp_t import perfetto, schema

        self.assertEqual(schema.COLLECTOR_KIND_NAMES[27], "nixl_telemetry")
        self.assertEqual(schema.COLLECTOR_KIND_NAMES[28], "nixl_telemetry_events")
        self.node.nixl_baseline()
        self.finish()
        path = next((self.root / "out").glob("*/*/events.jsonl"))
        documents = perfetto.export_traces([str(path)], str(self.root / "perfetto"), None)
        document = json.loads(next(iter(documents.values())).read_text())
        tracks = {e["name"] for e in document["traceEvents"] if e.get("ph") == "C"}
        self.assertTrue(any(name.startswith("nixl_telemetry ") for name in tracks), tracks)
        self.assertTrue(any(name.startswith("nixl_telemetry_events ") for name in tracks), tracks)


class ArmCorrectness(unittest.TestCase):
    """The checks an exporter-less arm must pass before its measurements mean anything."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.out = Path(self.directory.name)
        self.run = self.make("emissions_disabled")
        self.write_good()

    @staticmethod
    def make(arm):
        return pd.Run(name="t", topology="t", model=Path("/m"), prefill=pd.Side("p", "data0"), decode=pd.Side("d", "data0"),
                      client="c", control_network="control", router=pd.Router("gw"),
                      server=pd.Server(connector="nixl", nixl_backend="ucx", timing="linear", nixl_telemetry_arm=arm))

    def write_good(self):
        commands = {role: {node: pd.server_command(self.run, role, ["10.1.0.1"])}
                    for role, node in (("prefill", "p"), ("decode", "d"))}
        (self.out / "servers.json").write_text(json.dumps(commands))
        (self.out / "nixl-arm.json").write_text(json.dumps({"arm": "emissions_disabled", "nodes": {
            "p": {"outcome": "refused", "status": None, "body": None, "error": "URLError(ConnectionRefusedError)"},
            "d": {"outcome": "refused", "status": None, "body": None, "error": "URLError(ConnectionRefusedError)"}}}))
        (self.out / self.run.decode.step_log("decode", 0, 0)).write_text(step_log(3, xfer_us=40))
        for role, node in (("prefill", "p"), ("decode", "d")):
            (self.out / "logs" / role / node).mkdir(parents=True, exist_ok=True)
            (self.out / "logs" / role / node / f"{role}.log").write_text("INFO fine\n")

    def problems(self):
        return nixl.arm_problems(self.out, self.run)

    def test_a_clean_arm_passes_and_reports_the_dropped_count_as_not_measured(self):
        issues, report = self.problems()
        self.assertEqual(issues, [])
        self.assertEqual(report["arm"], "emissions_disabled")
        self.assertIn("not emitted", report["dropped_events"])
        self.assertEqual(report["decode_transfers"], {0: 3})
        self.run = self.make("default")
        self.write_good()
        issues, report = self.problems()
        self.assertEqual(issues, [])
        self.assertTrue(report["dropped_events"].startswith("unavailable"), "never a measured zero")

    def test_the_wrong_environment_an_answering_endpoint_and_missing_probe_are_problems(self):
        commands = json.loads((self.out / "servers.json").read_text())
        commands["decode"]["d"].insert(1, "NIXL_TELEMETRY_EXPORTER=prometheus_mp")
        (self.out / "servers.json").write_text(json.dumps(commands))
        self.assertTrue(any("ran with telemetry settings" in p for p in self.problems()[0]))
        self.write_good()
        commands = json.loads((self.out / "servers.json").read_text())
        index = commands["prefill"]["p"].index("-u")
        del commands["prefill"]["p"][index:index + 2]
        (self.out / "servers.json").write_text(json.dumps(commands))
        self.assertTrue(any("ran with telemetry settings" in p for p in self.problems()[0]), "an inherited setting not cleared")
        self.write_good()
        probe = json.loads((self.out / "nixl-arm.json").read_text())
        probe["nodes"]["d"].update(outcome="answered", body="# metrics")
        (self.out / "nixl-arm.json").write_text(json.dumps(probe))
        self.assertTrue(any("the endpoint probe on d was 'answered'" in p for p in self.problems()[0]))
        (self.out / "nixl-arm.json").unlink()
        self.assertTrue(any("no endpoint probe" in p for p in self.problems()[0]))

    def test_missing_server_evidence_is_a_problem_not_a_pass(self):
        (self.out / "servers.json").write_text("{}")
        self.assertTrue(any("servers.json lists" in p for p in self.problems()[0]), self.problems()[0])
        self.write_good()
        commands = json.loads((self.out / "servers.json").read_text())
        del commands["decode"]
        (self.out / "servers.json").write_text(json.dumps(commands))
        self.assertTrue(any("servers.json lists" in p for p in self.problems()[0]))
        self.write_good()
        commands = json.loads((self.out / "servers.json").read_text())
        commands["prefill"]["p"] = ["env", "true"]
        (self.out / "servers.json").write_text(json.dumps(commands))
        self.assertTrue(any("is not a vllm command" in p for p in self.problems()[0]))
        self.write_good()
        for role, node in (("prefill", "p"), ("decode", "d")):
            (self.out / "logs" / role / node / f"{role}.log").unlink()
        found = " ".join(self.problems()[0])
        self.assertIn("the prefill log on p is not readable", found)
        self.assertIn("the decode log on d is not readable", found)
        self.write_good()
        (self.out / "logs" / "decode" / "d" / "decode.log").write_text("  \n")
        self.assertTrue(any("the decode log on d is empty" in p for p in self.problems()[0]))
        self.write_good()
        probe = json.loads((self.out / "nixl-arm.json").read_text())
        probe["nodes"].pop("d")
        (self.out / "nixl-arm.json").write_text(json.dumps(probe))
        self.assertTrue(any("was probed on ['p']" in p for p in self.problems()[0]))

    def test_broken_per_transfer_statistics_and_a_telemetry_failure_in_a_log_are_problems(self):
        (self.out / self.run.decode.step_log("decode", 0, 0)).write_text(step_log(2, xfer_us=0))
        self.assertTrue(any("without a positive integer-microsecond duration" in p for p in self.problems()[0]))
        (self.out / self.run.decode.step_log("decode", 0, 0)).write_text(step_log(0))
        self.assertTrue(any("logged 0 transfers" in p for p in self.problems()[0]))
        self.write_good()
        (self.out / "logs" / "decode" / "d" / "decode.log").write_text("ERROR transfer_exception NIXL_ERR_NO_TELEMETRY\n")
        found = " ".join(self.problems()[0])
        self.assertIn("transfer_exception", found)
        self.assertIn("NIXL_ERR_NO_TELEMETRY", found)


class ProbeToValidator(unittest.TestCase):
    """The probe script itself (an HTTP 503, a timeout, a closed port, an answer) through ``probe_endpoint`` into the arm check."""

    def serve(self, status):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(status)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close()))
        return server.server_address[1]

    def probe(self, port, timeout=None):
        script = [sys.executable, "-c", nixl.OWN_SCRAPE, f"http://127.0.0.1:{port}/metrics", *([str(timeout)] if timeout else [])]
        done = subprocess.run(script, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads(done.stdout.strip().splitlines()[-1])

    def free_port(self):
        import socket
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def test_the_probe_classifies_refusal_http_errors_timeouts_and_answers(self):
        self.assertEqual(self.probe(self.free_port())["outcome"], "refused")
        found = self.probe(self.serve(503))
        self.assertEqual((found["outcome"], found["status"]), ("http_error", 503))
        found = self.probe(self.serve(200))
        self.assertEqual((found["outcome"], found["status"], found["body"]), ("answered", 200, "ok"))
        import socket
        silent = socket.socket()
        silent.bind(("127.0.0.1", 0))
        silent.listen(1)  # accepts the connection (in the backlog) and never answers
        self.addCleanup(silent.close)
        self.assertEqual(self.probe(silent.getsockname()[1], timeout=0.5)["outcome"], "timeout")

    def validator(self, port):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            case = ArmCorrectness("test_a_clean_arm_passes_and_reports_the_dropped_count_as_not_measured")
            case.out, case.run = out, ArmCorrectness.make("emissions_disabled")
            case.write_good()
            from harness import topology

            with mock.patch.object(topology, "exec_in", local_exec):
                probe = {"arm": "emissions_disabled", "nodes": {"p": nixl.probe_endpoint("c-p", port), "d": nixl.probe_endpoint("c-d", port)}}
            (out / "nixl-arm.json").write_text(json.dumps(probe))
            return nixl.arm_problems(out, case.run)[0]

    def test_only_a_positive_refusal_passes_the_arm_check(self):
        self.assertEqual(self.validator(self.free_port()), [])
        for port, outcome in ((self.serve(503), "http_error"), (self.serve(200), "answered")):
            found = self.validator(port)
            self.assertTrue(any(f"was {outcome!r}" in p and "not a connection refusal" in p for p in found), (outcome, found))


class Wiring(unittest.TestCase):
    RUN = pd.Run(name="t", topology="t", model=Path("/m"), prefill=pd.Side("p", "data0"), decode=pd.Side("d", "data0"),
                 client="c", control_network="control", router=pd.Router("gw"),
                 server=pd.Server(connector="nixl", nixl_backend="ucx", timing="linear"))

    def with_collector(self, **options):
        return pd.Run(**{**{f: getattr(self.RUN, f) for f in self.RUN.__dataclass_fields__},
                         "collector": pd.Collector(**options)})

    def test_the_option_needs_nixl_data_parallel_size_one_and_a_source(self):
        pd_parse = lambda raw: pd.parse(raw, Path("/r.yaml"))  # noqa: E731
        base = {"name": "t", "topology": "t", "model": "/m", "client": "c", "control_network": "control",
                "prefill": {"node": "p", "kv_network": "data0"}, "decode": {"node": "d", "kv_network": "data0"},
                "router": {"gateway": "gw"}, "server": {"connector": "nixl", "timing": "linear"}}
        with mock.patch.object(pd.models, "resolve", lambda model, fetch=False: Path(model)):
            run = pd_parse({**base, "collector": {"tent": False, "nixl_telemetry": True}})
            self.assertEqual((run.collector.tent, run.collector.nixl_telemetry), (False, True))
            for options, server, message in (
                    ({"tent": False, "nixl_telemetry": False}, None, "needs a source"),
                    ({}, {"connector": "nixl", "nixl_backend": "ucx", "timing": "linear"}, "collector needs TENT's metrics"),
                    ({"tent": False, "nixl_telemetry": True}, {"connector": "mooncake", "timing": "linear"}, "needs server.connector nixl"),
                    ({"tent": "yes", "nixl_telemetry": True}, None, "true or false")):
                with self.assertRaisesRegex(ValueError, message):
                    pd_parse({**base, "collector": options, **({"server": server} if server else {})})
            dp = {**base, "decode": {"node": "d", "kv_network": "data0", "data_parallel_size": 2},
                  "collector": {"tent": False, "nixl_telemetry": True}}
            with self.assertRaisesRegex(ValueError, "data parallelism is not qualified"):
                pd_parse(dp)
            both = pd_parse({**base, "server": {"connector": "nixl", "nixl_backend": "mooncake", "timing": "linear"},
                             "collector": {"nixl_telemetry": True}})
            self.assertTrue(both.collector.tent and both.collector.nixl_telemetry)

    @staticmethod
    def env_of(command):
        """(set, cleared) of a server command's telemetry settings."""
        head = command[1:command.index(pd.VLLM)]
        cleared = {head[i + 1] for i, part in enumerate(head) if part == "-u"}
        given = dict(part.split("=", 1) for part in head if "=" in part)
        return {k: v for k, v in given.items() if k.startswith("NIXL_TELEMETRY")}, cleared

    def test_the_servers_get_the_exporter_environment_and_only_with_the_option(self):
        run = self.with_collector(tent=False, nixl_telemetry=True)
        given, cleared = self.env_of(pd.server_command(run, "decode", ["10.1.0.1"]))
        self.assertEqual(given, {"NIXL_TELEMETRY_EXPORTER": "prometheus_mp",
                                 "NIXL_TELEMETRY_MULTIPROC_DIR": pd.NIXL_TELEMETRY_DIRECTORY,
                                 "NIXL_TELEMETRY_PROMETHEUS_PORT": str(pd.NIXL_TELEMETRY_PORT),
                                 "NIXL_TELEMETRY_PROMETHEUS_LOCAL": "1"})
        self.assertEqual(cleared, set(pd.NIXL_TELEMETRY_VARS) - set(given), "everything else inherited is cleared")
        for other in (self.RUN, self.with_collector(tent=False, nixl_telemetry=False)):
            plain = pd.server_command(other, "decode", ["10.1.0.1"])
            self.assertFalse(any("NIXL_TELEMETRY" in part for part in plain))

    def test_the_arms_without_an_exporter_set_what_they_need_and_clear_every_other_telemetry_setting(self):
        for arm, wanted in (("default", {}),
                            ("emissions_disabled", {"NIXL_TELEMETRY_ENABLED_METRICS": "__vllm_nixl_metrics_disabled__"})):
            run = pd.Run(**{**{f: getattr(self.RUN, f) for f in self.RUN.__dataclass_fields__},
                            "server": pd.Server(connector="nixl", nixl_backend="ucx", timing="linear", nixl_telemetry_arm=arm)})
            for role in ("prefill", "decode"):
                given, cleared = self.env_of(pd.server_command(run, role, ["10.1.0.1"]))
                self.assertEqual((given, cleared), (wanted, set(pd.NIXL_TELEMETRY_VARS) - set(wanted)), arm)
        self.assertEqual(len(pd.NIXL_TELEMETRY_VARS), len(set(pd.NIXL_TELEMETRY_VARS)))
        # the token matches none of NIXL's metric names (an empty allowlist would enable everything)
        names = ("agent_tx_bytes", "agent_rx_bytes", "agent_tx_requests_num", "agent_rx_requests_num",
                 "agent_memory_registered", "agent_memory_deregistered", "agent_xfer_time", "agent_xfer_post_time",
                 "agent_telemetry_events_dropped", *(f"agent_err_{s}" for s in STATUS_NAMES))
        import fnmatch
        self.assertEqual(len(names), 21)
        self.assertFalse(any(fnmatch.fnmatch(name, pd.NIXL_METRICS_DISABLED_TOKEN) for name in names))

    def test_an_arm_needs_nixl_excludes_the_exporter_arm_and_owns_the_telemetry_variables(self):
        base = {"name": "t", "topology": "t", "model": "/m", "client": "c", "control_network": "control",
                "prefill": {"node": "p", "kv_network": "data0"}, "decode": {"node": "d", "kv_network": "data0"},
                "router": {"gateway": "gw"}}
        nixl_server = {"connector": "nixl", "nixl_backend": "ucx", "timing": "linear"}
        with mock.patch.object(pd.models, "resolve", lambda model, fetch=False: Path(model)):
            ok = pd.parse({**base, "server": {**nixl_server, "nixl_telemetry_arm": "emissions_disabled"}}, Path("/r.yaml"))
            self.assertEqual(pd.nixl_telemetry_arm(ok), "emissions_disabled")
            for server, extra, message in (
                    ({**nixl_server, "nixl_telemetry_arm": "off"}, {}, "server.nixl_telemetry_arm is one of"),
                    ({"connector": "mooncake", "timing": "linear", "nixl_telemetry_arm": "default"}, {}, "needs server.connector nixl"),
                    ({**nixl_server, "nixl_telemetry_arm": "default"}, {"collector": {"tent": False, "nixl_telemetry": True}},
                     "are different arms"),
                    ({**nixl_server, "nixl_telemetry_arm": "default", "env": {"NIXL_TELEMETRY_ENABLE": "1"}}, {},
                     "variables the arm controls")):
                with self.assertRaisesRegex(ValueError, message):
                    pd.parse({**base, "server": server, **extra}, Path("/r.yaml"))
            both = pd.parse({**base, "server": nixl_server, "collector": {"tent": False, "nixl_telemetry": True}}, Path("/r.yaml"))
            self.assertEqual(pd.nixl_telemetry_arm(both), "exporter")

    def test_the_manifest_has_one_nixl_target_a_node_beside_or_instead_of_the_tent_ranks(self):
        run = self.with_collector(tent=False, nixl_telemetry=True)
        self.assertEqual(collector.targets(run, "p"), [{"id": "nixl-p", "role": "prefill", "source": "nixl",
                                                        "endpoint": f"127.0.0.1:{pd.NIXL_TELEMETRY_PORT}"}])
        self.assertEqual(collector.targets(run, "d")[0]["role"], "decode")
        mooncake = pd.Run(**{**{f: getattr(run, f) for f in run.__dataclass_fields__},
                             "server": pd.Server(connector="nixl", nixl_backend="mooncake", timing="linear"),
                             "collector": pd.Collector(nixl_telemetry=True)})
        ids = [t["id"] for t in collector.targets(mooncake, "p")]
        self.assertEqual(ids, ["prefill-dp0-tp0", "nixl-p"])
        self.assertIn("--boundary-dir", collector.arguments(mooncake.collector, "r"))
        self.assertNotIn("--boundary-dir", collector.arguments(pd.Collector(), "r"))

    def test_the_pid_table_script_reads_the_process_titles_and_start_ticks(self):
        done = subprocess.run([sys.executable, "-c", nixl.PROCESS_TABLE], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIsInstance(nixl.parse_table(done.stdout), list)
        with self.assertRaises(nixl.InventoryError):
            nixl.parse_table('[{"pid": "x"}]')


if __name__ == "__main__":
    unittest.main()
