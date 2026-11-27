"""Prefill/decode runs: the checks alone, and a whole run on docker with
E2E_VLLM=1 (several minutes, most of the machine's cores)."""

from __future__ import annotations

import atexit
import contextlib
from dataclasses import asdict
import io
import json
import os
import re
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from harness import llmd, pd, spec, topology


class SyntheticRouting:
    """For test classes whose saved or live runs are synthetic fixtures with
    no router evidence: the router's checks (`llmd.recheck`, tested in
    test_llmd) are stubbed for the class, and only for it. A run on docker
    must not use this: its routing is checked for real."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        stub = mock.patch.object(llmd, "recheck")
        stub.start()
        cls.addClassCleanup(stub.stop)


E2E = Path(__file__).resolve().parent.parent


def _model_dir(config: dict) -> Path:
    directory = tempfile.TemporaryDirectory()
    atexit.register(directory.cleanup)
    (Path(directory.name) / "config.json").write_text(json.dumps(config))
    return Path(directory.name)


MOE_MODEL = _model_dir({"model_type": "qwen3_moe", "num_experts": 8})
DENSE_MODEL = _model_dir({"model_type": "qwen3", "num_hidden_layers": 2})
RUN = pd.Run(name="t", topology="t", model=Path("/m"), prefill=pd.Side("p", "data0"),
             decode=pd.Side("d", "data0"), client="c", control_network="control", router=pd.Router("gw"),
             workload=pd.Workload(num_prompts=2))


class ProfilerConfig(unittest.TestCase):
    """The KV network profiler's run options: both servers get one run ID and
    the connector's ``network_profiler`` namespace; off, nothing changes."""

    NIXL = pd.Server(connector="nixl", timing="linear")

    def run_with(self, **changes):
        return pd.Run(**{**{f: getattr(RUN, f) for f in RUN.__dataclass_fields__},
                         "server": self.NIXL, **changes})

    def extra(self, run, role, profile_id="id-1"):
        command = pd.server_command(run, role, ["10.100.0.1"], profile_id=profile_id)
        transfer = json.loads(command[command.index("--kv-transfer-config") + 1])
        return transfer["kv_connector_extra_config"]

    def test_both_servers_share_one_profile_id_and_the_options(self):
        run = self.run_with(profiler=pd.Profiler(clock_port_offset=2000, flush="background"))
        expected = {"level": "l1", "run_id": "id-1", "output_dir": pd.PROFILES,
                    "clock_port_offset": 2000, "flush": "background", "shutdown_timeout_s": 10}
        for role in ("prefill", "decode"):
            self.assertEqual(self.extra(run, role)["network_profiler"], expected)

    def test_a_profiled_server_gets_the_shutdown_grace_its_recorders_close_in(self):
        """vLLM kills an engine process at once unless a worker-shutdown grace is set: the profiled
        servers get one, above two recorders' close deadlines and under the orderly stop's wait."""
        grace = "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=30"
        for role in ("prefill", "decode"):
            command = pd.server_command(self.run_with(profiler=pd.Profiler()), role, ["10.100.0.1"], profile_id="id-1")
            self.assertIn(grace, command)
            self.assertNotIn(grace, pd.server_command(self.run_with(), role, ["10.100.0.1"]))
        self.assertGreater(pd.PROFILER_SHUTDOWN_GRACE_S, 2 * self.extra(
            self.run_with(profiler=pd.Profiler()), "prefill")["network_profiler"]["shutdown_timeout_s"])
        import inspect
        self.assertLess(pd.PROFILER_SHUTDOWN_GRACE_S, inspect.signature(pd.stop_orderly).parameters["timeout"].default)
        override = self.run_with(profiler=pd.Profiler(), server=pd.Server(
            connector="nixl", timing="linear", env={"VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": 5}))
        command = pd.server_command(override, "prefill", ["10.100.0.1"], profile_id="id-1")
        self.assertEqual([c for c in command if c.startswith("VLLM_WORKER_SHUTDOWN")],
                         ["VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=5"])

    def test_l3_passes_only_the_named_endpoints_probe_settings(self):
        run = self.run_with(profiler=pd.Profiler(
            level="l3", probes={"p": {"interfaces": ["lo"]}},
            observer_cpus={"p": [13]}))
        command = pd.server_command(run, "prefill", ["10.100.0.1"],
                                    profile_id="id-1", node="p")
        extra = json.loads(command[command.index("--kv-transfer-config") + 1])
        settings = extra["kv_connector_extra_config"]["network_profiler"]
        self.assertEqual(settings["level"], "l3")
        self.assertEqual(settings["probes"], {"interfaces": ["lo"]})
        with self.assertRaisesRegex(ValueError, "endpoint node"):
            pd.server_command(run, "prefill", ["10.100.0.1"], profile_id="id-1")

    def test_a_profiled_ucx_server_logs_ucx_protocol_selection(self):
        """UCX's protocol tables are on exactly when a UCX server is profiled; a run's
        own UCX_PROTO_INFO wins."""
        ucx = pd.Server(connector="nixl", timing="linear", nixl_backend="ucx")
        profiled = self.run_with(server=ucx, profiler=pd.Profiler())
        for role in ("prefill", "decode"):
            command = pd.server_command(profiled, role, ["10.100.0.1"], profile_id="id-1")
            self.assertIn("UCX_PROTO_INFO=y", command)
            self.assertIn("UCX_LOG_FILE=stderr", command)
            self.assertFalse([c for c in pd.server_command(self.run_with(server=ucx), role, ["10.100.0.1"])
                              if c.startswith("UCX_PROTO_INFO")])
            self.assertFalse([c for c in pd.server_command(self.run_with(profiler=pd.Profiler()), role,
                                                           ["10.100.0.1"], profile_id="id-1")
                              if c.startswith("UCX_PROTO_INFO")])
        override = self.run_with(profiler=pd.Profiler(), server=pd.Server(
            connector="nixl", timing="linear", nixl_backend="ucx", env={"UCX_PROTO_INFO": "used"}))
        self.assertEqual([c for c in pd.server_command(override, "prefill", ["10.100.0.1"], profile_id="id-1")
                          if c.startswith("UCX_PROTO_INFO")], ["UCX_PROTO_INFO=used"])

    def test_ucx_protocol_tables_are_parsed_per_node_and_joined_to_recorders(self):
        """After a profiled UCX run, each server log's tables land in ucx-protocols/<role>/<node>.json,
        with each process joined by PID to its recorder; a log without tables is unavailable."""
        import tempfile
        table = "\n".join("[1700000000.000001] [node:42  :0]   " + line for line in (
            "+--------+-------------+----------+", "| vllm inter-node cfg#1           |",
            "| remote memory read by ucp_get*(multi) into host memory from host |",
            "+--------+-------------+----------+", "| Range  | Description | Config   |",
            "+--------+-------------+----------+", "| 0..inf | zero-copy   | tcp/kv0  |",
            "+--------+-------------+----------+"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for role, node, text in (("prefill", "p", "INFO start\n" + table + "\n"), ("decode", "d", "INFO\n")):
                (output / "logs" / role / node).mkdir(parents=True)
                (output / "logs" / role / node / f"{role}.log").write_text(text)
            recorder = output / "profiles" / "prefill" / "p" / "run" / "proc"
            recorder.mkdir(parents=True)
            (recorder / "events.jsonl").write_text(json.dumps(
                {"pid": 42, "run_id": "run", "process_id": "proc", "context": {"role": "worker", "rank": 0}}) + "\n")
            malformed = recorder.parent / "other"
            malformed.mkdir()  # an unrelated, malformed recorder entry: skipped, never fatal
            (malformed / "events.jsonl").write_text(json.dumps({"pid": 7, "run_id": "run", "context": 7}) + "\n")
            def docker_cp(argv, check=False):  # the containers' final logs: prefill's has one more table
                source, local = argv[2], Path(argv[3])
                self.assertEqual(source.split(":", 1)[1], f"{pd.WORK}/logs/{local.name.split('.')[0]}.log")
                if "p" in source.split(":", 1)[0]:
                    local.write_text("INFO start\n" + table + "\n" + table.replace("cfg#1", "cfg#2") + "\n")
                    return ""
                # decode's copy fails, as shell.run reports it: the earlier copy is parsed
                raise pd.shell.CommandError(argv, 1, "no such container")

            with mock.patch.object(pd.shell, "run", side_effect=docker_cp):
                pd.ucx_protocols(self.run_with(), {"p": "c-p", "d": "c-d"}, output, "run")
            prefill = json.loads((output / "ucx-protocols" / "prefill" / "p.json").read_text())
            endpoint, after_stop = prefill["endpoint_configurations"]
            self.assertEqual((endpoint["locality"], endpoint["ep_config"], endpoint["recorder"]["process_id"]),
                             ("inter-node", 1, "proc"))
            self.assertEqual(after_stop["ep_config"], 2)  # only in the log copied after the stop
            self.assertEqual(prefill["log"], {"path": "logs/prefill/p/prefill.final.log", "after_stop": True})
            self.assertEqual(endpoint["operations"][0]["ranges"][0]["lanes"][0]["device"], "kv0")
            self.assertNotIn("unavailable", prefill)
            self.assertEqual((prefill["tables"], prefill["recorders_skipped"]), (2, 1))
            decode = json.loads((output / "ucx-protocols" / "decode" / "d.json").read_text())
            self.assertEqual(decode["unavailable"], "no UCX_PROTO_INFO table in the server log")
            self.assertEqual(decode["log"], {"path": "logs/decode/d/decode.log", "after_stop": False})

    def test_a_profiled_data_parallel_server_gets_a_shutdown_window(self):
        """With data parallelism each API server is a child process that vLLM's
        default ``--shutdown-timeout 0`` force-kills right after SIGTERM, before
        its frontend recorder closes: profiled DP servers get the profiler's
        grace; unprofiled or DP-1 servers are unchanged."""
        def flag(run, role):
            command = pd.server_command(run, role, ["10.100.0.1"], profile_id="id-1")
            i = command.index("--shutdown-timeout") if "--shutdown-timeout" in command else None
            return None if i is None else command[i + 1]

        dp = {"prefill": pd.Side("p", "data0", data_parallel_size=2),
              "decode": pd.Side("d", "data0", data_parallel_size=2)}  # fmt: skip
        profiled = self.run_with(profiler=pd.Profiler(), model=MOE_MODEL, **dp)
        for role in ("prefill", "decode"):
            self.assertEqual(flag(profiled, role), str(pd.PROFILER_SHUTDOWN_GRACE_S))
            self.assertIsNone(flag(self.run_with(profiler=pd.Profiler()), role))
            command = pd.server_command(self.run_with(model=MOE_MODEL, **dp), role, ["10.100.0.1"])
            self.assertNotIn("--shutdown-timeout", command)

    def test_every_recorder_of_each_dp_rank_must_be_present_and_closed(self):
        """Per server node: a frontend per API server slot of the launch, a
        scheduler per DP rank and a worker per (DP rank, TP rank) on its rank's
        engine, all of the side's KV role, each whole (stream, terminal row,
        manifest) and closed. A missing recorder, a lost terminal row, no
        profiler.json, hook errors, a worker on another rank's engine, the other
        side's frontend, a repeated slot or swapped endpoints fail the run."""
        import shutil
        import tempfile
        import uuid

        stats = dict.fromkeys(("head", "tail", "drop_full", "drop_contention",
                               "drop_ids", "drop_handles", "drop_identity",
                               "drop_conflict", "drop_closed", "unknown_handles"),
                              0)  # fmt: skip

        def recorders(root, run_id, role, node, dp):
            kv = {"kv_role": "kv_producer" if role == "prefill" else "kv_consumer"}
            for d in range(dp):
                engine = f"{node}-engine_dp{d}"
                for context in (
                    {"role": "frontend", "api_server": f"ApiServer_{d}", **kv},
                    {"role": "scheduler", "engine_id": engine, "rank": 0, "dp_rank": d, **kv},
                    {"role": "worker", "engine_id": engine, "rank": 0, "dp_rank": d, **kv},
                ):  # fmt: skip
                    process = str(uuid.uuid4())
                    meta = {"schema": 1, "type": "metadata", "run_id": run_id,
                            "process_id": process, "pid": 10 + d, "start_ticks": 100,
                            "clock_id": str(uuid.uuid4()), "clock_source": "CLOCK_MONOTONIC",
                            "boot_id": "b", "net_namespace": "n", "pid_namespace": "p",
                            "time_namespace": "t", "context": context}  # fmt: skip
                    folder = root / "profiles" / role / node / run_id / process
                    folder.mkdir(parents=True)
                    end = {"schema": 1, "type": "end", "complete": True, "stats": stats}
                    (folder / "events.jsonl").write_text(
                        json.dumps(meta) + "\n" + json.dumps(end) + "\n")
                    (folder / "manifest.json").write_text(json.dumps(
                        {**meta, "saved": True, "complete": True, "stats": stats}))
                    (folder / "profiler.json").write_text(
                        json.dumps({"complete": True, "hook_errors": 0}))

        def captures(root, side):
            for path in sorted(root.glob(f"profiles/{side}/*/*/events.jsonl")):
                yield path.parent, json.loads(path.read_text().splitlines()[0])["context"]

        def find(root, side, **match):
            return next(f for f, c in captures(root, side)
                        if all(c.get(k) == v for k, v in match.items()))

        def edit_context(folder, **changes):
            rows = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
            rows[0]["context"].update(changes)
            (folder / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest = json.loads((folder / "manifest.json").read_text())
            (folder / "manifest.json").write_text(
                json.dumps({**manifest, "context": rows[0]["context"]}))

        def other_side_frontend(root):
            shutil.rmtree(find(root, "decode/d", api_server="ApiServer_1"))
            source = find(root, "prefill/p", api_server="ApiServer_1")
            shutil.copytree(source, root / "profiles/decode/d" / source.parent.name / source.name)

        def swapped(root):
            profiles = root / "profiles"
            (profiles / "prefill/p").rename(profiles / "held")
            (profiles / "decode/d").rename(profiles / "prefill/p")
            (profiles / "held").rename(profiles / "decode/d")

        worker = lambda root: find(root, "decode/d", role="worker", dp_rank=0)  # noqa: E731
        faults = {
            "clean": lambda root: None,
            "missing": lambda root: shutil.rmtree(find(root, "decode/d", role="frontend")),
            "no terminal row": lambda root: (worker(root) / "events.jsonl").write_text(
                (worker(root) / "events.jsonl").read_text().splitlines()[0] + "\n"),
            "never closed": lambda root: (worker(root) / "profiler.json").unlink(),
            "hook errors": lambda root: (worker(root) / "profiler.json").write_text(
                json.dumps({"complete": True, "hook_errors": 1})),
            "other engine": lambda root: edit_context(worker(root), engine_id="d-engine_dp1"),
            "other side's frontend": other_side_frontend,
            "repeated slot": lambda root: edit_context(
                find(root, "decode/d", api_server="ApiServer_1"), api_server="ApiServer_0"),
            "swapped endpoints": swapped,
        }
        dp = {"prefill": pd.Side("p", "data0", data_parallel_size=2),
              "decode": pd.Side("d", "data0", data_parallel_size=2)}  # fmt: skip
        run = self.run_with(profiler=pd.Profiler(), **dp)
        self.assertEqual(pd.api_server_slots(run, run.decode), ["ApiServer_0", "ApiServer_1"])
        self.assertEqual(pd.api_server_slots(self.run_with(), self.run_with().decode),
                         ["MainProcess"])  # fmt: skip
        for label, fault in faults.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                root, run_id = Path(tmp), str(uuid.uuid4())
                (root / "profiler.json").write_text(json.dumps({"run_id": run_id}))
                for role, node in (("prefill", "p"), ("decode", "d")):
                    recorders(root, run_id, role, node, 2)
                fault(root)
                verdict = {"problems": []}
                pd.check_profiles(root, run, verdict)
                found = verdict["problems"]
                self.assertEqual(found == [], label == "clean", found)
                if label != "swapped endpoints":
                    self.assertTrue(all(p.startswith("decode d") for p in found), found)

    def test_off_leaves_the_connector_configuration_unchanged(self):
        run = self.run_with()
        for role in ("prefill", "decode"):
            self.assertNotIn("network_profiler", self.extra(run, role))
        with self.assertRaisesRegex(ValueError, "profile ID"):
            pd.server_command(self.run_with(profiler=pd.Profiler()), "prefill", ["10.100.0.1"])

    def test_a_run_file_can_name_the_profiler_and_rejects_the_rest(self):
        raw = {"name": "t", "topology": "t", "model": "x", "prefill": {"node": "p", "kv_network": "data0"},
               "decode": {"node": "d", "kv_network": "data0"}, "client": "c", "control_network": "control",
               "router": {"gateway": "gw"}, "server": {"connector": "nixl", "timing": "linear"}}
        with mock.patch.object(pd.models, "resolve", return_value=Path("/m")):
            run = pd.parse({**raw, "profiler": {"clock_port_offset": 1500}}, Path("/r.yaml"))
            self.assertEqual(run.profiler, pd.Profiler(clock_port_offset=1500))
            self.assertIsNone(pd.parse(raw, Path("/r.yaml")).profiler)
            for bad, message in (({"profiler": {"flush": "never"}}, "profiler.flush"),
                                 ({"profiler": {"transport": "tcp"}}, "unexpected keyword"),
                                 ({"profiler": {"clock_port_offset": 0}}, "clock_port_offset"),
                                 ({"profiler": {"nope": 1}}, "unexpected keyword")):
                with self.assertRaisesRegex(ValueError, message):
                    pd.parse({**raw, **bad}, Path("/r.yaml"))
            mooncake = pd.parse({**raw, "profiler": {}, "server": {"connector": "mooncake", "timing": "linear"}},
                                Path("/r.yaml"))
            self.assertEqual(mooncake.profiler, pd.Profiler())
            # The Mooncake connector profiles at L3 too, with each endpoint's probes and observer CPUs.
            probes = {node: {"interfaces": ["data0"], "required": ["netdev"]} for node in ("p", "d")}
            l3 = pd.parse({**raw, "server": {"connector": "mooncake", "timing": "linear"},
                           "profiler": {"level": "l3", "probes": probes, "observer_cpus": {"p": [12], "d": [13]}}},
                          Path("/r.yaml"))
            self.assertEqual(l3.profiler.level, "l3")
            dp = pd.parse({**raw, "profiler": {}, "server": {"connector": "mooncake", "timing": "linear"},
                           "decode": {"node": "d", "kv_network": "data0", "data_parallel_size": 2}}, Path("/r.yaml"))
            self.assertEqual(dp.decode.data_parallel_size, 2)

    MOONCAKE = pd.Server(connector="mooncake", timing="linear")
    INSTANCES = {"prefill": {"p0": {"control": "10.250.0.2"}, "p1": {"control": "10.250.0.3"}},
                 "decode": {"d0": {"control": "10.250.0.4"}}}

    def test_profiled_mooncake_servers_pin_nothing_for_their_clocks(self):
        """The servers' clocks are served and advertised by the servers themselves: no engine ID, clock endpoint
        or side-channel offset is given to a profiled Mooncake server, with or without data parallelism."""
        for dp in (1, 2):
            side = {"prefill": pd.Side("p0", "data0", data_parallel_size=dp),
                    "decode": pd.Side("d0", "data0", data_parallel_size=dp)}  # fmt: skip
            run = self.run_with(server=self.MOONCAKE, profiler=pd.Profiler(), model=MOE_MODEL, **side)
            for role in ("prefill", "decode"):
                with self.subTest(dp=dp, role=role):
                    command = pd.server_command(run, role, ["10.100.0.1"], profile_id="id-1", node=f"{role[0]}0")
                    transfer = json.loads(command[command.index("--kv-transfer-config") + 1])
                    profile = transfer["kv_connector_extra_config"]["network_profiler"]
                    self.assertNotIn("engine_id", transfer)
                    for key in ("clock_bind", "clock_peers", "clock_port_offset"):
                        self.assertNotIn(key, profile)
        nixl = pd.server_command(self.run_with(profiler=pd.Profiler()), "prefill", ["10.100.0.1"], profile_id="id-1")
        self.assertNotIn("engine_id", json.loads(nixl[nixl.index("--kv-transfer-config") + 1]))

    def test_each_mooncake_clock_calibration_is_answered_by_the_producer_it_names(self):
        """From the captures alone: every prefill engine a decode scheduler handed off from (its recorded handoffs)
        calibrates once, ok, against that engine's TP0/PP0 worker clock (served and advertised), with a mapping file
        of the same pair. A wrong or swapped clock, a missing, failed or duplicate calibration, a calibration of an
        engine never handed off from, or an unadvertised clock fail; a start-only mapping passes."""
        import tempfile
        import uuid

        engines = ("prefill-p_dp0", "prefill-p_dp1")

        def write(root, change=None):
            run_id = str(uuid.uuid4())
            clocks = {}
            for engine in engines:
                clocks[engine] = str(uuid.uuid4())
                folder = root / "profiles/prefill/p" / run_id / str(uuid.uuid4())
                folder.mkdir(parents=True)
                (folder / "events.jsonl").write_text(json.dumps({"run_id": run_id, "clock_id": clocks[engine],
                    "context": {"role": "worker", "engine_id": engine, "tp_rank": 0, "pp_rank": 0}}) + "\n")  # fmt: skip
                report = {"server": {"state": "available"}, "advertised": {"state": "ok"}}
                if change == "unadvertised" and engine == engines[1]:
                    report["advertised"] = {"state": "failed", "detail": "HTTP 404"}
                (folder / "profiler.json").write_text(json.dumps({"clock": report}))
            local = str(uuid.uuid4())
            folder = root / "profiles/decode/d" / run_id / str(uuid.uuid4())
            folder.mkdir(parents=True)
            rows = [{"run_id": run_id, "clock_id": local, "context": {"role": "scheduler", "engine_id": "decode-d"}}]
            for key, engine in enumerate(engines, 1):
                rows.append({"schema": 1, "sequence": 2 * key - 1, "type": "dictionary_fragment", "id": key,
                             "length": len(engine), "offset": 0, "encoding": "utf8", "payload": engine.encode().hex()})
                rows.append({"schema": 1, "sequence": 2 * key, "type": "event", "kind": "handoff", "peer_ref": key})
            (folder / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            peers = []
            for i, engine in enumerate(engines):
                sample = {"run_id": run_id, "local_generation": local, "remote_generation": clocks[engine],
                          "t1": 1, "t2": 2, "t3": 3, "t4": 4}
                (folder / f"calibration-{i}.json").write_text(json.dumps({"start": [sample], "end": [sample]}))
                peers.append({"engine_id": engine, "address": ["10.0.0.1", 40000 + i], "status": "ok",
                              "advertised_generation": clocks[engine], "remote_generation": clocks[engine],
                              "calibration_file": f"calibration-{i}.json"})
            if change == "wrong clock":
                peers[1].update(remote_generation=clocks[engines[0]], advertised_generation=clocks[engines[0]],
                                calibration_file="calibration-0.json")
            elif change == "swapped labels":
                peers[0]["engine_id"], peers[1]["engine_id"] = engines[1], engines[0]
            elif change == "failed":
                peers[1] = {"engine_id": engines[1], "status": "failed", "detail": "not advertised"}
            elif change == "missing":
                peers.pop()
            elif change == "duplicate":
                peers.append(dict(peers[0]))
            elif change == "never handed off":
                peers.append({"engine_id": "prefill-q_dp0", "status": "failed"})
            if change == "unresolved handoffs":  # the dictionary lost: nothing to verify against
                rows = [row for row in rows if row.get("type") != "dictionary_fragment"]
                peers = []
                (folder / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            (folder / "calibration.json").write_text(json.dumps({"run_id": run_id, "local_generation": local,
                                                                 "peers": peers}))
            return folder

        run = self.run_with(server=self.MOONCAKE, profiler=pd.Profiler())
        for change in (None, "wrong clock", "swapped labels", "failed", "missing", "duplicate", "never handed off",
                       "unadvertised", "unresolved handoffs"):
            with self.subTest(change), tempfile.TemporaryDirectory() as tmp:
                write(Path(tmp), change)
                verdict = {"problems": []}
                pd.check_mooncake_clocks(Path(tmp), run, verdict)
                self.assertEqual(verdict["problems"] == [], change is None, verdict["problems"])
        # anchors: an end without a start cannot align the producer; a start-only mapping can (drift unknown)
        for label, anchors, clean in (("end only", ("end",), False), ("start only", ("start",), True)):
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                file = write(Path(tmp)) / "calibration-1.json"
                mapping = json.loads(file.read_text())
                file.write_text(json.dumps({"start": mapping["start"] if "start" in anchors else [],
                                            "end": mapping["end"] if "end" in anchors else []}))
                verdict = {"problems": []}
                pd.check_mooncake_clocks(Path(tmp), run, verdict)
                self.assertEqual(verdict["problems"] == [], clean, verdict["problems"])

    def test_the_mooncake_clock_peer_count_is_checked_before_any_server_starts(self):
        """The decode holds one clock peer per prefill engine: more peers than the profiler's default bound are
        refused when the run is parsed (the clocks' own ports are ephemeral, never derived)."""
        raw = {"name": "t", "topology": "t", "model": "/m", "client": "c", "control_network": "control",
               "decode": {"node": "d", "kv_network": "data0"}, "router": {"gateway": "gw"},
               "server": {"connector": "mooncake", "timing": "linear"}}
        with mock.patch.object(pd.models, "resolve", lambda model, fetch=False: Path(model)):
            def parse(dp, **profiler):
                return pd.parse({**raw, "prefill": {"node": "p", "kv_network": "data0", "data_parallel_size": dp},
                                 "profiler": profiler}, Path("/r.yaml"))

            self.assertEqual(parse(pd.MOONCAKE_CLOCK_MAX_PEERS).prefill.data_parallel_size, pd.MOONCAKE_CLOCK_MAX_PEERS)
            with self.assertRaisesRegex(ValueError, "more clock peers than the profiler's clock_max_peers"):
                parse(pd.MOONCAKE_CLOCK_MAX_PEERS + 1)

    def test_a_saved_profiled_run_loads_again(self):
        run = self.run_with(profiler=pd.Profiler(), model=Path("/m"))
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "run.json").write_text(json.dumps(asdict(run), default=str))
            self.assertEqual(pd.saved_run(directory).profiler, pd.Profiler())

    def test_the_chat_endpoint_is_a_workload_option(self):
        chat = self.run_with(workload=pd.Workload(endpoint="chat", num_prompts=2))
        command = pd.bench_command(chat, "http://gw")
        self.assertEqual(command[command.index("--backend") + 1:command.index("--backend") + 4:2],
                         ["openai-chat", "/v1/chat/completions"])
        plain = pd.bench_command(RUN, "http://gw")
        self.assertEqual(plain[plain.index("--endpoint") + 1], "/v1/completions")

    def test_measure_adds_the_percentile_flags_only_when_asked(self):
        measured = self.run_with(workload=pd.Workload(measure=True, num_prompts=2))
        command = pd.bench_command(measured, "http://gw")
        self.assertEqual(command[command.index("--metric-percentiles") + 1], "50,95,99")
        self.assertNotIn("--metric-percentiles", pd.bench_command(RUN, "http://gw"))

    def test_the_workload_window_is_the_servers_own(self):
        busy = line(step={"requests": [{"req_id": "a"}]}, t_start=100.0, t_deadline=100.4)
        later = line(step={"requests": [{"req_id": "b"}]}, t_start=140.0, t_deadline=141.5)
        idle = line(step={"requests": []}, t_start=1.0, t_deadline=2.0)
        layout = line(kv_layout={}, t=5.0)
        self.assertEqual(pd.workload_window([busy + idle + layout, later]), (100.0, 141.5))
        self.assertIsNone(pd.workload_window([idle + layout]))


def sample(t, *processes):
    return {"t": t, "processes": [{"pid": p[0], "start": p[1], "kind": p[2], "cpu_s": p[3], "rss_kb": 10,
                                   "hwm_kb": 20} for p in processes]}


class CpuAccounting(unittest.TestCase):
    """The workload's CPU from samples: by identity, interpolated at the window's
    edges, setup and shutdown apart."""

    def series(self, setup=100.0, work=10.0, helper_per_s=0.1):
        """A server (1 CPU-s per second while busy) and a writer, sampled every second
        through ``setup`` idle seconds, a ``work`` window and 3 seconds after."""
        samples = []
        for second in range(int(setup + work + 4)):
            busy = min(max(second - setup, 0), work)
            samples.append(sample(float(second), (1, 1, "serving", busy * 1.0),
                                  (2, 1, "writer", 0.05 * second * 0 + busy * helper_per_s + 0.5),
                                  (3, 1, "clock", 0.0)))
        return samples, (setup, setup + work)

    def test_a_long_setup_does_not_change_the_workloads_helper_cores(self):
        for setup in (0.0, 100.0):
            samples, window = self.series(setup=setup)
            report = pd.cpu_report({"d": samples}, window, {"d": [1000.0]}, 1.0)["d"]
            self.assertAlmostEqual(report["serving_s"], 10.0)
            self.assertAlmostEqual(report["helpers_s"], 1.0)
            self.assertAlmostEqual(report["helpers_s"] / report["workload_s"], 0.1)
            self.assertTrue(report["covered"] and not report["lost"])
            self.assertAlmostEqual(report["max_gap_s"], 1.0)

    def test_setup_is_its_own_number_and_the_edges_are_interpolated(self):
        samples, _ = self.series(setup=20.0)
        report = pd.cpu_report({"d": samples}, (20.5, 28.25), {"d": [1000.0]}, 1.0)["d"]
        self.assertAlmostEqual(report["serving_s"], 7.75)
        self.assertAlmostEqual(report["setup_serving_s"], 0.5)  # half of the second that straddles the edge
        self.assertAlmostEqual(report["setup_helpers_s"], 0.05)
        self.assertAlmostEqual(report["setup_serving_s"] + report["serving_s"], 8.25)

    def test_the_writers_end_flush_is_its_lifetime_cpu_beyond_the_workload_and_never_double_counted(self):
        samples, window = self.series()
        report = pd.cpu_report({"d": samples}, window, {"d": [100_000.0 + 0.5 + 1.0]}, 1.0)["d"]
        self.assertAlmostEqual(report["writers_at_end_s"], 0.5 + 1.0)
        self.assertAlmostEqual(report["flush_s"], 100_000.0)
        self.assertAlmostEqual(report["writer_lifetime_cpu_s"], 100_001.5)
        self.assertEqual((report["writers_seen"], report["writer_manifests_with_cpu"]), (1, 1))

    def test_a_profiler_off_node_has_an_explicit_zero_writer_cost(self):
        no_writer = [sample(float(i), (1, 1, "serving", float(i))) for i in range(10)]
        report = pd.cpu_report({"d": no_writer}, (2.0, 8.0), {"d": []}, 1.0)["d"]
        self.assertEqual((report["writers_seen"], report["writer_lifetime_cpu_s"], report["flush_s"]), (0, 0, 0))
        self.assertEqual(pd.cpu_report({"d": no_writer}, (2.0, 8.0), {}, 1.0)["d"]["flush_s"], 0)

    def test_missing_or_inconsistent_writer_totals_are_unmeasured_never_zero(self):
        samples, window = self.series()
        for values in ([None], [], [float("nan")], [-1.0], [True], [0.5, 9.0]):
            report = pd.cpu_report({"d": samples}, window, {"d": values}, 1.0)["d"]
            self.assertIsNone(report["flush_s"], values)
            self.assertIsNone(report["writer_lifetime_cpu_s"], values)
        # A lifetime below what the samples saw the writers use is inconsistent beyond the tolerance.
        report = pd.cpu_report({"d": samples}, window, {"d": [0.5]}, 1.0)["d"]
        self.assertEqual((report["flush_s"], report["writer_lifetime_cpu_s"]), (None, None))
        # ...and within a couple of ticks of it is the counters' resolution: no flush, still valid.
        report = pd.cpu_report({"d": samples}, window, {"d": [1.49]}, 1.0)["d"]
        self.assertEqual((report["flush_s"], report["writer_lifetime_cpu_s"]), (0.0, 1.49))

    def test_a_process_that_vanished_or_appeared_is_listed_not_guessed(self):
        samples = [sample(0.0, (1, 1, "serving", 1.0), (2, 1, "serving", 1.0)),
                   sample(1.0, (1, 1, "serving", 2.0), (2, 1, "serving", 3.0)),
                   sample(2.0, (1, 1, "serving", 3.0), (9, 5, "serving", 0.5)),
                   sample(3.0, (1, 1, "serving", 4.0), (9, 5, "serving", 1.5))]
        report = pd.cpu_over(samples, 0.5, 2.5)
        self.assertEqual((report["lost"], report["new"]), (["2:1"], ["9:5"]))
        self.assertAlmostEqual(report["serving_s"], (3.5 - 1.5) + (3.0 - 2.0) + 1.0)
        self.assertTrue(report["covered"])
        self.assertFalse(pd.cpu_over(samples, 0.5, 9.0)["covered"])


class ProcessTree(unittest.TestCase):
    """The probes and the orderly stop on real processes: a stub ``vllm serve``
    owning a stub ``python -m vllm.kv_profiling.sink`` helper that ignores
    SIGTERM and finishes its flush only once its owner has closed."""

    OWNER = """
import os, signal, subprocess, sys, time
here = os.path.dirname(os.path.abspath(__file__))
helper = subprocess.Popen([sys.executable, "-m", "vllm.kv_profiling.sink", here], cwd=here,
                          env={**os.environ, "PYTHONPATH": os.path.join(here, "..", "lib")})
stubborn = len(sys.argv) > 2 and sys.argv[2] == "stubborn"
def term(*_):
    if stubborn:
        return
    open(os.path.join(here, "closing"), "w").close()
    helper.wait()
    sys.exit(0)
signal.signal(signal.SIGTERM, term)
open(os.path.join(here, "up"), "w").close()
while True:
    time.sleep(0.05)
"""
    HELPER = """
import os, signal, sys, time
here = sys.argv[1]
signal.signal(signal.SIGTERM, lambda *_: open(os.path.join(here, "sigterm"), "w").close())
open(os.path.join(here, "helper.pid"), "w").write(str(os.getpid()))
open(os.path.join(here, "helper-up"), "w").close()
while not os.path.exists(os.path.join(here, "closing")):
    time.sleep(0.05)
time.sleep(0.3)
open(os.path.join(here, "flushed"), "w").close()
"""

    def start(self, *extra):
        import subprocess
        import time

        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(root)]))
        (root / "bin").mkdir()
        # The image's Python lives under /opt/venv/; the stubs run under a Python
        # of their own, so the stop script's notion of a service is only them.
        (root / "venv" / "bin").mkdir(parents=True)
        python = root / "venv" / "bin" / "python"
        python.symlink_to(sys.executable)
        self.prefix = f"{root}/venv/bin/".encode()
        package = root / "lib" / "vllm" / "kv_profiling"
        package.mkdir(parents=True)
        (package.parent / "__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        (package / "sink.py").write_text(self.HELPER)
        (root / "bin" / "vllm").write_text(self.OWNER.replace("sys.executable", repr(str(python))))
        owner = subprocess.Popen([str(python), str(root / "bin" / "vllm"), "serve", *extra])
        def reap():
            owner.kill()
            owner.wait()
            with contextlib.suppress(OSError, ValueError):
                os.kill(int((root / "bin" / "helper.pid").read_text()), 9)

        self.addCleanup(reap)
        for _ in range(100):
            if (root / "bin" / "up").exists() and (root / "bin" / "helper-up").exists():
                break
            time.sleep(0.1)
        else:
            self.fail("the stub servers did not start")
        return root / "bin", owner

    def run_script(self, template, *args):
        import subprocess

        script = template.replace('b"/opt/venv/"', repr(self.prefix))
        done = subprocess.run([sys.executable, "-c", script, *args], capture_output=True, text=True, check=True)
        return json.loads(done.stdout.strip().splitlines()[-1])

    def test_the_sampler_sees_serving_and_the_writer_with_a_monotonic_stamp_and_never_itself(self):
        import subprocess
        import time

        bin_dir, owner = self.start()
        out = bin_dir / "samples.jsonl"
        stop = bin_dir / "samples.jsonl.stop"
        sampler = subprocess.Popen([sys.executable, "-c", pd.CPU_SAMPLER, "0.1", str(out)])
        self.addCleanup(lambda: (sampler.kill(), sampler.wait()))
        time.sleep(0.5)
        stop.touch()
        sampler.wait(timeout=10)
        samples = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertGreaterEqual(len(samples), 3)
        self.assertTrue(all(b["t"] > a["t"] for a, b in zip(samples, samples[1:])))
        self.assertLessEqual(samples[-1]["t"], time.monotonic())
        kinds = sorted(p["kind"] for p in samples[-1]["processes"])
        self.assertEqual(kinds, ["serving", "writer"])
        self.assertEqual({p["pid"] for p in samples[-1]["processes"] if p["kind"] == "serving"}, {owner.pid})

    def test_the_orderly_stop_signals_only_the_owner_and_waits_for_the_helpers_flush(self):
        bin_dir, owner = self.start()
        result = self.run_script(pd.STOP_OWNERS, "30")
        self.assertTrue(result["graceful"], result)
        self.assertEqual(result["signalled"], [owner.pid])
        self.assertTrue((bin_dir / "flushed").exists())
        self.assertFalse((bin_dir / "sigterm").exists(), "the helper was sent SIGTERM")

    def test_a_node_that_does_not_stop_is_reported_not_forced_here(self):
        bin_dir, owner = self.start("stubborn")
        result = self.run_script(pd.STOP_OWNERS, "1")
        self.assertFalse(result["graceful"])
        self.assertIn(owner.pid, [pid for pid, kind in result["remaining"]])
        self.assertFalse((bin_dir / "sigterm").exists())


class OrderlyShutdown(unittest.TestCase):
    def test_decode_stops_before_prefill_and_only_a_stuck_node_is_forced_and_recorded(self):
        run = pd.Run(**{**{f: getattr(RUN, f) for f in RUN.__dataclass_fields__},
                        "server": pd.Server(connector="nixl", timing="linear"), "profiler": pd.Profiler()})
        calls = []

        def exec_in(name, command, **kwargs):
            calls.append(("exec", name))
            ok = name != "c-p"
            return json.dumps({"signalled": [1], "graceful": ok, "remaining": [] if ok else [[7, "owner"]],
                               "seconds": 1.0})

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(pd.topology, "exec_in", exec_in), \
                mock.patch.object(pd, "stop_services", lambda name: calls.append(("force", name))):
            steps = pd.stop_orderly(run, {"p": "c-p", "d": "c-d"}, Path(directory),
                on_endpoint_closed=lambda node, result:
                    calls.append(("observer", node)))
            saved = json.loads(Path(directory, "shutdown.json").read_text())
        self.assertEqual(calls, [("exec", "c-d"), ("observer", "d"),
                                 ("exec", "c-p"), ("force", "c-p"), ("observer", "p")])
        self.assertEqual([(s["node"], s["graceful"], s.get("forced", False)) for s in steps],
                         [("d", True, False), ("p", False, True)])
        self.assertEqual(saved, steps)


class ProfilerCollection(unittest.TestCase):
    def test_profiles_are_copied_after_the_servers_stop(self):
        run = pd.Run(**{**{f: getattr(RUN, f) for f in RUN.__dataclass_fields__},
                        "server": pd.Server(connector="nixl", timing="linear"), "profiler": pd.Profiler()})
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(pd.shell, "run") as docker:
            pd._collect_profiles(run, {"p": "c-p", "d": "c-d"}, Path(directory), "id-1")
            saved = json.loads(Path(directory, "profiler.json").read_text())
        self.assertEqual(saved["run_id"], "id-1")
        sources = sorted(call.args[0][2] for call in docker.call_args_list)
        self.assertEqual(sources, [f"c-d:{pd.PROFILES}/.", f"c-p:{pd.PROFILES}/."])


def line(**record) -> str:
    return json.dumps(record) + "\n"


ONE_RANK = line(kv_layout={"tp_rank": 0, "tp_size": 1, "block_bytes": 4, "bytes": 1600})


def allocated(*requests: str) -> str:
    """Prefill's allocations of requests, under its own request ids."""
    return line(kv_events={"allocated": {f"cmpl-{r}-0-p": [1] for r in requests}})


def checked(request: str, ok: bool = True, **more) -> dict:
    """Decode's byte check of a request, under its own request id: 5 blocks
    of 50 layers, 4 bytes a block and layer."""
    return {"req_id": f"cmpl-{request}-0-d", "ok": ok, "blocks": 5, "layers": 50, "bytes": 1000,
            "mismatched_bytes": 0 if ok else 9, **more}


GOOD = {
    "bench": {"completed": 2, "errors": ["", ""], "mean_ttft_ms": 50.0},
    "prefill": ONE_RANK + allocated("a", "b") + line(kv_transfers={"send": {"count": 2, "bytes": 2000, "failed": 0}}),
    "decode": ONE_RANK + line(kv_check=checked("a")) + line(kv_check=checked("b")),
    "routers": {"control": 50, "data0": 2100},
}


class CheckTests(unittest.TestCase):
    def verdict(self, **changes) -> dict:
        given = {**GOOD, **changes}
        return pd.check(given["bench"], [given["prefill"]], [given["decode"]], given["routers"], RUN)

    def test_a_clean_run_passes(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["kv_bytes_checked"], 2000)
        self.assertEqual(verdict["transfers"], {"count": 2, "bytes": 2000, "failed": 0})

    def test_each_failure_is_reported(self) -> None:
        cases = {
            "completed": {"bench": {"completed": 1, "errors": ["", "HTTP 500"]}},
            "byte-checked 1": {"decode": ONE_RANK + line(kv_check=checked("a"))},
            "unlike prefill": {"decode": ONE_RANK + line(kv_check=checked("a", ok=False))
                               + line(kv_check=checked("b"))},
            "failed 1 transfer": {
                "prefill": ONE_RANK + allocated("a", "b")
                + line(kv_transfers={"send": {"count": 2, "bytes": 2000, "failed": 1}})},
            "moved 1000 bytes, not the 2000 decode checked": {
                "prefill": ONE_RANK + allocated("a", "b") + line(kv_transfers={"send": {"count": 1, "bytes": 1000}}),
                "routers": {"control": 50, "data0": 5000}},
            # Decode checks every block pulled: more is not the plan either.
            "moved 3000 bytes, not the 2000 decode checked": {
                "prefill": ONE_RANK + allocated("a", "b") + line(kv_transfers={"send": {"count": 2, "bytes": 3000}}),
                "routers": {"control": 50, "data0": 3100}},
            "not both produced and consumed": {"prefill": ONE_RANK + allocated("a", "b", "c")
                                               + line(kv_transfers={"send": {"count": 2, "bytes": 2000}})},
            "KV network": {"routers": {"control": 50, "data0": 100}},
            "control network carried": {"routers": {"control": 1000, "data0": 2100}},
        }
        for message, changes in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(**changes)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)

    def test_tent_stays_on_the_kv_address(self) -> None:
        tent = pd.tent_config("10.100.0.1", {"max_outstanding_tasks": 8192})
        self.assertEqual(tent["rpc_server_hostname"], "10.100.0.1")
        hp_tcp = tent["transports"]["hp_tcp"]
        self.assertEqual((hp_tcp["bind_address"], hp_tcp["advertise_address"], hp_tcp["rail_addresses"]),
                         ("10.100.0.1", "10.100.0.1", ["10.100.0.1"]))
        self.assertEqual(hp_tcp["max_outstanding_tasks"], 8192)
        self.assertTrue(all(not t["enable"] for name, t in tent["transports"].items() if name != "hp_tcp"))

    def test_the_kv_cache_dtype_reaches_both_servers(self) -> None:
        def flag(run: pd.Run, role: str, name: str) -> str:
            command = pd.server_command(run, role, ["10.100.0.1"])
            return command[command.index(name) + 1]

        self.assertEqual(flag(RUN, "prefill", "--kv-cache-dtype"), "auto")  # the model's
        fp8 = pd.Run(**{**RUN.__dict__, "server": pd.Server(kv_cache_dtype="fp8")})
        for role in ("prefill", "decode"):
            self.assertEqual(flag(fp8, role, "--kv-cache-dtype"), "fp8")
            self.assertEqual(flag(fp8, role, "--dtype"), "auto")  # the weights' unchanged

    def test_online_quantization_reaches_both_servers_when_asked(self) -> None:
        self.assertNotIn("--quantization", pd.server_command(RUN, "prefill", ["10.100.0.1"]))
        fp8 = pd.Run(**{**RUN.__dict__, "server": pd.Server(quantization="fp8")})
        for role in ("prefill", "decode"):
            command = pd.server_command(fp8, role, ["10.100.0.1"])
            self.assertEqual(command[command.index("--quantization") + 1], "fp8")

    def test_a_multimodal_model_serves_its_language_model_only_when_asked(self) -> None:
        self.assertNotIn("--language-model-only", pd.server_command(RUN, "prefill", ["10.100.0.1"]))
        text = pd.Run(**{**RUN.__dict__, "server": pd.Server(language_model_only=True)})
        for role in ("prefill", "decode"):
            self.assertIn("--language-model-only", pd.server_command(text, role, ["10.100.0.1"]))

    def test_a_repositorys_own_code_runs_only_when_asked(self) -> None:
        self.assertNotIn("--trust-remote-code", pd.server_command(RUN, "prefill", ["10.100.0.1"]))
        self.assertNotIn("--trust-remote-code", pd.bench_command(RUN, "http://gw"))
        trusted = pd.Run(**{**RUN.__dict__, "server": pd.Server(trust_remote_code=True)})
        for role in ("prefill", "decode"):
            self.assertIn("--trust-remote-code", pd.server_command(trusted, role, ["10.100.0.1"]))
        self.assertIn("--trust-remote-code", pd.bench_command(trusted, "http://gw"))
        # Only a boolean: the string "false" would turn the flags on.
        base = Path(__file__).parents[1] / "runs" / "example.yaml"
        for value, ok in (("true", True), ("false", True), ('"false"', False), ("1", False), ("yes", True)):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                spec = Path(directory) / "x.yaml"
                text = re.sub(r"^(server:.*)$", rf"\1\n  trust_remote_code: {value}",
                              base.read_text(), count=1, flags=re.M)
                spec.write_text(text)
                if ok:
                    pd.load(spec)
                else:
                    with self.assertRaisesRegex(ValueError, "trust_remote_code is true or false"):
                        pd.load(spec)

    def test_nixl_moves_the_kv_over_its_mooncake_backend_on_the_kv_address(self) -> None:
        nixl = pd.Run(**{**RUN.__dict__, "server": pd.Server(connector="nixl")})
        # A hybrid's state as NIXL locates it; a spec's env may override.
        self.assertIn("VLLM_SSM_CONV_STATE_LAYOUT=DS", pd.server_command(nixl, "prefill", ["10.100.0.1"]))
        self.assertNotIn("VLLM_SSM_CONV_STATE_LAYOUT=DS", pd.server_command(RUN, "prefill", ["10.100.0.1"]))
        sd = pd.Run(**{**RUN.__dict__, "server": pd.Server(connector="nixl", env={"VLLM_SSM_CONV_STATE_LAYOUT": "SD"})})
        self.assertIn("VLLM_SSM_CONV_STATE_LAYOUT=SD", pd.server_command(sd, "prefill", ["10.100.0.1"]))
        for role in ("prefill", "decode"):
            command = pd.server_command(nixl, role, ["10.100.0.1"])
            config = json.loads(command[command.index("--kv-transfer-config") + 1])
            self.assertEqual((config["kv_connector"], config["kv_connector_extra_config"]),
                             ("NixlConnector", {"backends": ["Mooncake"]}))
            env = dict(item.split("=", 1) for item in command[1:command.index(pd.VLLM)])
            self.assertEqual((env["VLLM_NIXL_SIDE_CHANNEL_HOST"], env["NIXL_MOONCAKE_IP_ADDR"]),
                             ("10.100.0.1", "10.100.0.1"))
            self.assertEqual(json.loads(env["MC_TENT_CONF"])["rpc_server_hostname"], "10.100.0.1")
            self.assertNotIn("VLLM_MOONCAKE_BOOTSTRAP_PORT", env)
        mooncake = pd.server_command(RUN, "prefill", ["10.100.0.1"])
        self.assertEqual(json.loads(mooncake[mooncake.index("--kv-transfer-config") + 1])["kv_connector"],
                         "MooncakeConnector")

    def test_nixl_ranks_take_their_own_addresses(self) -> None:
        nixl = pd.Run(**{**TP2.__dict__, "server": pd.Server(connector="nixl", tensor_parallel_size=2)})
        command = pd.server_command(nixl, "decode", ["10.100.0.1", "10.101.0.1"])
        config = json.loads(command[command.index("--kv-transfer-config") + 1])
        self.assertEqual(config["kv_connector_extra_config"],
                         {"backends": ["Mooncake"], "rank_addresses": ["10.100.0.1", "10.101.0.1"]})
        env = dict(item.split("=", 1) for item in command[1:command.index(pd.VLLM)])
        # Rank 0's address until each rank binds its own; one side channel.
        self.assertEqual((env["VLLM_NIXL_SIDE_CHANNEL_HOST"], env["NIXL_MOONCAKE_IP_ADDR"]),
                         ("10.100.0.1", "10.100.0.1"))

    def test_the_connector_is_mooncake_or_nixl(self) -> None:
        base = {"name": "t", "topology": "t", "model": "org/name", "client": "c", "control_network": "control",
                "router": {"gateway": "gw"},
                "prefill": {"node": "p", "kv_network": "data0"}, "decode": {"node": "d", "kv_network": "data0"}}

        def load(server: dict, **sides) -> pd.Run:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "run.yaml"
                path.write_text(json.dumps({**base, **sides, "server": {"timing": "zero", **server}}))
                return pd.load(path)

        self.assertEqual(load({}).server.connector, "mooncake")
        self.assertEqual(load({"connector": "nixl", "tensor_parallel_size": 2}).server.tensor_parallel_size, 2)
        with self.assertRaisesRegex(ValueError, "server.connector is one of mooncake, nixl"):
            load({"connector": "ucx"})
        dp = load({"connector": "nixl"}, decode={"node": "d", "kv_network": "data0", "data_parallel_size": 2})
        self.assertEqual(dp.decode.data_parallel_size, 2)

    def test_the_platform_may_choose_the_block_size(self) -> None:
        default = pd.server_command(RUN, "prefill", ["10.100.0.1"])
        self.assertEqual(default[default.index("--block-size") + 1], "16")
        unset = pd.Run(**{**RUN.__dict__, "server": pd.Server(block_size=None)})
        self.assertNotIn("--block-size", pd.server_command(unset, "prefill", ["10.100.0.1"]))

    def test_a_side_may_ask_its_own_block_size_over_nixl(self) -> None:
        def load(server: dict, **sides) -> pd.Run:
            base = {"name": "t", "topology": "t", "model": "org/name", "client": "c",
                    "control_network": "control", "router": {"gateway": "gw"},
                    "prefill": {"node": "p", "kv_network": "data0"},
                    "decode": {"node": "d", "kv_network": "data0"}}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "run.yaml"
                path.write_text(json.dumps({**base, **sides, "server": {"timing": "zero", **server}}))
                return pd.load(path)

        def block(run: pd.Run, role: str) -> str | None:
            command = pd.server_command(run, role, ["10.100.0.1"])
            return command[command.index("--block-size") + 1] if "--block-size" in command else None

        # Each side's own, as the hybrid pair whose sizes resolve to another
        # ratio needs them; a side without one inherits the server's.
        nixl = {"connector": "nixl", "block_size": 16}
        own = load(nixl, prefill={"node": "p", "kv_network": "data0", "block_size": 800},
                   decode={"node": "d", "kv_network": "data0", "block_size": 1600})
        self.assertEqual((block(own, "prefill"), block(own, "decode")), ("800", "1600"))
        inherit = load({"connector": "nixl", "block_size": 544},
                       prefill={"node": "p", "kv_network": "data0", "block_size": 544})
        self.assertEqual((block(inherit, "prefill"), block(inherit, "decode")), ("544", "544"))
        platform = load({"connector": "nixl", "block_size": None})
        self.assertEqual((block(platform, "prefill"), block(platform, "decode")), (None, None))

    def test_a_side_block_size_is_bounded(self) -> None:
        def load(server: dict, prefill: dict, decode: dict, router: dict | None = None) -> pd.Run:
            base = {"name": "t", "topology": "t", "model": "org/name", "client": "c",
                    "control_network": "control", "router": {"gateway": "gw", **(router or {})},
                    "prefill": {"node": "p", "kv_network": "data0", **prefill},
                    "decode": {"node": "d", "kv_network": "data0", **decode},
                    "server": {"timing": "zero", **server}}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "run.yaml"
                path.write_text(json.dumps(base))
                return pd.load(path)

        nixl = {"connector": "nixl", "block_size": 544}
        for bad in (0, -1, True, "544"):
            with self.assertRaisesRegex(ValueError, "block_size is a positive integer"):
                load(nixl, {"block_size": bad}, {})
        # Mooncake's write-task estimate uses the server's size: a side may
        # not change it, alone or with both sides equal to another value.
        for sides in ({"block_size": 1088}, {}):
            with self.assertRaisesRegex(ValueError, "per-side sizes need server.connector: nixl"):
                load({"block_size": 544}, {"block_size": 1088}, sides)
        load({"block_size": 544}, {"block_size": 544}, {"block_size": 544})
        load(nixl, {"block_size": 1088}, {"block_size": 1088})
        # The precise prefix index is built from server.block_size: sides
        # must equal it, and it must be set.
        router = {"prefix_index": "precise", "epp_verbosity": 5}
        precise = {"connector": "nixl", "enable_prefix_caching": True}
        with self.assertRaisesRegex(ValueError, "precise builds the index from server.block_size"):
            load({**precise, "block_size": 16}, {"block_size": 64}, {"block_size": 64}, router)
        with self.assertRaisesRegex(ValueError, "precise needs server.block_size"):
            load({**precise, "block_size": None}, {"block_size": 16}, {"block_size": 16}, router)
        load({**precise, "block_size": 64}, {}, {}, router)

    def test_a_side_may_declare_its_kernel_backend(self) -> None:
        def load(server: dict, prefill: dict) -> pd.Run:
            base = {"name": "t", "topology": "t", "model": "org/name", "client": "c",
                    "control_network": "control", "router": {"gateway": "gw"},
                    "prefill": {"node": "p", "kv_network": "data0", **prefill},
                    "decode": {"node": "d", "kv_network": "data0"},
                    "server": {"timing": "aisimulate", "gpu": "h100", **server}}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "run.yaml"
                path.write_text(json.dumps(base))
                return pd.load(path)

        def simulation(run: pd.Run, role: str) -> dict:
            command = pd.server_command(run, role, ["10.100.0.1"])
            return json.loads(command[command.index("--simulation-config") + 1])

        run = load({"connector": "nixl"}, {"kv_kernel_backend": "FLASHINFER"})
        self.assertEqual(simulation(run, "prefill")["kv_kernel_backend"], "FLASHINFER")
        self.assertNotIn("kv_kernel_backend", simulation(run, "decode"))
        for server, backend in (({"connector": "mooncake"}, "FLASHINFER"),
                                ({"connector": "nixl", "timing": "zero"}, "FLASHINFER"),
                                ({"connector": "nixl"}, 1)):
            with self.assertRaisesRegex(ValueError, "kv_kernel_backend needs a backend name, NIXL and AISimulate"):
                load(server, {"kv_kernel_backend": backend})

    def test_prefix_caching_is_off_for_byte_checks_and_settable(self) -> None:
        def flags(kv_check: bool, enabled: bool | None) -> list[str]:
            run = pd.Run(**{**RUN.__dict__, "kv_check": kv_check,
                            "server": pd.Server(enable_prefix_caching=enabled)})
            return [flag for flag in pd.server_command(run, "prefill", ["10.100.0.1"]) if "prefix-caching" in flag]

        self.assertEqual(flags(True, None), ["--no-enable-prefix-caching"])
        self.assertEqual(flags(False, None), [])  # vLLM's default
        # A hybrid model's run without byte checks keeps it off.
        self.assertEqual(flags(False, False), ["--no-enable-prefix-caching"])
        self.assertEqual(flags(False, True), ["--enable-prefix-caching"])
        # With the byte checks too: a block's pattern is then its content's.
        self.assertEqual(flags(True, True), ["--enable-prefix-caching"])
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / "run.yaml"
            spec.write_text((E2E / "runs" / "example.yaml").read_text().replace(
                "server:  # both servers", "server:  # both servers\n  enable_prefix_caching: true"))
            run = pd.load(spec)
            self.assertTrue(run.kv_check and run.server.enable_prefix_caching)

    def test_a_workload_may_share_prefixes(self) -> None:
        repetition = {"prefix_len": 768, "suffix_len": 256, "num_prefixes": 4}
        run = pd.Run(**{**RUN.__dict__, "workload": pd.Workload(input_len=1024, prefix_repetition=repetition)})
        command = pd.bench_command(run, "http://gw")
        self.assertEqual(command[command.index("--dataset-name") + 1], "prefix_repetition")
        self.assertEqual([command[command.index(f"--prefix-repetition-{name}") + 1]
                          for name in ("prefix-len", "suffix-len", "num-prefixes", "output-len")],
                         ["768", "256", "4", "16"])
        self.assertNotIn("--random-input-len", command)
        self.assertIn("--random-input-len", pd.bench_command(RUN, "http://gw"))
        base = (E2E / "runs" / "example.yaml").read_text()
        for workload, extra, error in (
                ("prefix_repetition: {prefix_len: 700, suffix_len: 256, num_prefixes: 4}", "", "input_len is"),
                ("prefix_repetition: {prefix_len: 768, suffix_len: 256}", "", "takes positive"),
                ("input_len: 1024", "expect_fragmentation: 1\n", "true or false"),
                ("input_len: 1024", "expect_fragmentation: true\nkv_check: false\n", "needs kv_check")):
            with self.subTest(workload=workload, extra=extra), tempfile.TemporaryDirectory() as directory:
                spec = Path(directory) / "run.yaml"
                text = re.sub(r"^workload:.*$", f"workload:\n  {workload}", base, count=1, flags=re.M)
                spec.write_text(re.sub(r"^kv_check:.*\n", "", text, flags=re.M) + extra)
                with self.assertRaisesRegex(ValueError, error):
                    pd.load(spec)

    def test_a_new_run_names_its_timing_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / "run.yaml"
            spec.write_text((E2E / "runs" / "example.yaml").read_text().replace("  timing: linear\n", ""))
            with self.assertRaises(ValueError) as raised:
                pd.load(spec)
            self.assertIn("needs server.timing", str(raised.exception))

    def test_the_example_run_loads(self) -> None:
        run = pd.load(E2E / "runs" / "example.yaml")
        self.assertEqual((run.prefill.node, run.decode.kv_network), ("prefill0", "data0"))
        # Named by its Hugging Face id and commit: where a run fetches it.
        self.assertEqual(run.model.parts[-3:], ("models", "Qwen--Qwen2.5-7B-Instruct",
                                                "a09a35458c702b33eeacc393d103063234e8bc28"))
        self.assertEqual(run.router, pd.Router("gateway"))
        # Its topology has the gateway as a node of its own.
        nodes = spec.load(E2E / "topologies" / f"{run.topology}.yaml").nodes
        self.assertTrue({run.router.gateway, run.client, run.prefill.node, run.decode.node} <= set(nodes))

    def test_servers_keep_idle_connections_longer_than_the_router(self) -> None:
        # vLLM's 5 s default closes connections llm-d's sidecar and Envoy reuse.
        for role in ("prefill", "decode"):
            self.assertIn("VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120", pd.server_command(RUN, role, ["10.1.0.1"]))
        asked = pd.Run(**{**RUN.__dict__, "server": pd.Server(env={"VLLM_HTTP_TIMEOUT_KEEP_ALIVE": 30})})
        command = pd.server_command(asked, "decode", ["10.1.0.1"])
        self.assertIn("VLLM_HTTP_TIMEOUT_KEEP_ALIVE=30", command)
        self.assertNotIn("VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120", command)

    def test_the_router_is_required_on_a_node_of_its_own(self) -> None:
        base = {"name": "t", "topology": "t", "model": "org/name", "client": "c", "control_network": "control",
                "prefill": {"node": "p", "kv_network": "data0"}, "decode": {"node": "d", "kv_network": "data0"},
                "server": {"timing": "zero"}}
        path = Path("/x/run.yaml")
        run = pd.parse({**base, "router": {"gateway": "gw", "epp_plugins": "plugins.yaml", "epp_verbosity": 2}}, path)
        self.assertEqual(run.router, pd.Router("gw", Path("/x/plugins.yaml"), 2))
        for router, message in ((None, "needs router.gateway"), ({}, "needs router.gateway"),
                                ({"gateway": 3}, "needs router.gateway"),
                                ({"gateway": "c"}, "a node of its own"), ({"gateway": "p"}, "a node of its own"),
                                ({"gateway": "d"}, "a node of its own"),
                                ({"gateway": "gw", "port": 1}, "router: ")):
            with self.subTest(router=router), self.assertRaisesRegex(ValueError, message):
                pd.parse({**base, **({} if router is None else {"router": router})}, path)
        # The vLLM toy proxies are gone.
        with self.assertRaisesRegex(ValueError, "unknown run key"):
            pd.parse({**base, "router": {"gateway": "gw"}, "proxy": "proxy.py"}, path)
        self.assertFalse(hasattr(pd, "ProxyFront"))


# gpt-oss-like: a 32-token window group and a full one, 2 layers each, 4
# bytes a block and layer; 1024-token prompts (64 blocks).
GROUPS_LAYOUT = line(kv_layout={
    "tp_rank": 0, "tp_size": 1, "block_bytes": 4, "bytes": 1600,
    "groups": [{"kind": "SlidingWindowSpec", "sliding_window": 32, "block_size": 16, "layers": 2, "block_bytes": 4},
               {"kind": "FullAttentionSpec", "sliding_window": None, "block_size": 16, "layers": 2,
                "block_bytes": 4}]})


def group_check(group: int, first: int, blocks: int, **more) -> dict:
    return {"group": group, "first_logical_block": first, "blocks": blocks, "layers": 2, "block_bytes": 8,
            "bytes": blocks * 8, "mismatched_bytes": 0, **more}


def grouped(request: str, window=(61, 3), full=(0, 64)) -> dict:
    """Decode's check of a request: the window's last blocks, every full one."""
    groups = [group_check(0, *window), group_check(1, *full)]
    return checked(request, blocks=window[1] + full[1], layers=4, bytes=sum(g["bytes"] for g in groups),
                   groups=groups)


def pulled(request: str, window=(61, 3), full=(0, 64), offset: int = 100) -> str:
    """Decode's pull of a request as posted: its blocks by group."""
    return line(kv_events={"reads_posted": [f"cmpl-{request}-0-d"], "allocated_groups": {f"cmpl-{request}-0-d": [
        {"group": 0, "first": window[0], "blocks": list(range(offset, offset + window[1]))},
        {"group": 1, "first": full[0], "blocks": list(range(offset + 10, offset + 10 + full[1]))}]}})


PULLS = pulled("a") + pulled("b", offset=300)
# Decode's first steps of the two 1024-token requests.
FIRST_STEPS = "".join(
    line(step={"requests": [{"req_id": f"cmpl-{r}-0-d", "num_computed_tokens": 1023,
                             "num_scheduled_tokens": 1, "num_prompt_tokens": 1024}]})
    for r in ("a", "b"))


def allocated_groups(request: str, *entries: tuple[int, int, list[int]], side: str = "p") -> str:
    return line(kv_events={"allocated_groups": {f"cmpl-{request}-0-{side}": [
        {"group": group, "first": first, "blocks": blocks} for group, first, blocks in entries]}})


class GroupCheckTests(unittest.TestCase):
    MOVED = 2 * (3 + 64) * 8

    def verdict(self, checks=None, moved=MOVED, prefill_events="", decode_events="", pulls=PULLS) -> dict:
        checks = checks or [grouped("a"), grouped("b")]
        prefill = (GROUPS_LAYOUT + allocated("a", "b") + prefill_events
                   + line(kv_transfers={"send": {"count": 2, "bytes": moved, "failed": 0}}))
        decode = GROUPS_LAYOUT + pulls + FIRST_STEPS + decode_events + "".join(line(kv_check=c) for c in checks)
        return pd.check({"completed": 2, "errors": []}, [prefill], [decode],
                        {"control": 50, "data0": moved + 100}, RUN)

    def test_groups_are_checked_and_their_evidence_kept(self) -> None:
        # Prefill gave the window block 1 again at its fourth place; decode's
        # table pads the window's head with the null block.
        verdict = self.verdict(
            prefill_events=allocated_groups("a", (0, 0, [1, 2, 3]), (1, 0, [4, 5, 6]))
            + allocated_groups("a", (0, 3, [1])),
            decode_events=allocated_groups("a", (0, 61, [7, 8, 9]), side="d")
            + allocated_groups("a", (0, 0, [0] * 61 + [7, 8, 9]), side="d"))
        self.assertEqual(verdict["problems"], [])
        window, full = verdict["ranks"][0]["groups"]["decode"][0]
        self.assertEqual((window["kind"], window["checked_blocks"], window["clipped_pulls"],
                          window["first_logical_block_max"], window["null_places"]),
                         ("SlidingWindowSpec", 6, 2, 61, 61))
        self.assertEqual((full["checked_blocks"], full["checked_bytes"], full["clipped_pulls"]), (128, 1024, 0))
        window = verdict["ranks"][0]["groups"]["prefill"][0][0]
        self.assertEqual((window["reused_blocks"], window["reallocated_blocks"]), (1, 1))
        # Listed twice at one place (a pull, then the scheduled table) is not
        # a reallocation; a block given to another request is.
        verdict = self.verdict(
            prefill_events=allocated_groups("a", (0, 0, [1, 2])) + allocated_groups("b", (0, 0, [2])),
            decode_events=allocated_groups("a", (0, 61, [7, 8, 9]), side="d")
            + allocated_groups("a", (0, 0, [0] * 61 + [7, 8, 9]), side="d"))
        decode_window = verdict["ranks"][0]["groups"]["decode"][0][0]
        self.assertEqual((decode_window["reused_blocks"], decode_window["reallocated_blocks"]), (0, 0))
        prefill_window = verdict["ranks"][0]["groups"]["prefill"][0][0]
        self.assertEqual((prefill_window["reused_blocks"], prefill_window["reallocated_blocks"]), (0, 1))

    def test_group_failures_are_reported(self) -> None:
        wrong_bytes = grouped("a")
        wrong_bytes["groups"][1] = group_check(1, 0, 64, bytes=1000)
        wrong_bytes["bytes"] = 24 + 1000
        unsummed = grouped("a")
        unsummed["bytes"] += 8
        omitted = grouped("a")
        omitted["groups"] = omitted["groups"][:1]
        omitted["bytes"] = 24
        duplicated = grouped("a")
        duplicated["groups"] = [duplicated["groups"][0], *duplicated["groups"]]
        duplicated["bytes"] += 24
        ungrouped = grouped("a")
        del ungrouped["groups"]
        cases = {
            "covers groups [0], not each of the layout's 2 once": [omitted, grouped("b")],
            "covers groups [0, 0, 1], not each of the layout's 2 once": [duplicated, grouped("b")],
            "cmpl-a-0-d's byte check lacks groups": [ungrouped, grouped("b")],
            # Within every bound, but not what was pulled.
            "group 1 checked 32 block(s) from 0, not the pull's 64 from 0": [grouped("a", full=(0, 32)),
                                                                             grouped("b")],
            "group 0 checked 3 block(s) from 1, not the pull's 3 from 61": [grouped("a", window=(1, 3)),
                                                                            grouped("b")],
            "group 1 check disagrees with the layout": [wrong_bytes, grouped("b")],
            "is not its groups' sum": [unsummed, grouped("b")],
            "pulled 4 blocks, more than its window's 3": [grouped("a", window=(60, 4)), grouped("b")],
            "group 1 (FullAttentionSpec) was pulled from block 2": [grouped("a", full=(2, 62)), grouped("b")],
            "no pull of window group 0 was clipped": [grouped("a", window=(0, 3)), grouped("b", window=(0, 3))],
        }
        for message, checks in cases.items():
            with self.subTest(message=message):
                moved = sum(c["bytes"] for c in checks)
                problems = self.verdict(checks, moved=moved)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)
        problems = self.verdict(pulls=pulled("a"))["problems"]
        self.assertTrue(any("cmpl-b-0-d's pull plan was not logged" in p for p in problems), problems)

    def test_checks_missing_fields_are_not_skipped(self) -> None:
        # A truncated full group whose aggregate blocks were dropped, or
        # both aggregates and groups: validated, not let through.
        truncated = grouped("a", full=(0, 32))
        del truncated["blocks"]
        bare = grouped("a")
        del bare["blocks"], bare["groups"]
        for message, check in (("lacks blocks", truncated), ("lacks blocks, groups", bare)):
            with self.subTest(message=message):
                problems = self.verdict([check, grouped("b")], moved=check["bytes"] + grouped("b")["bytes"])[
                    "problems"]
                self.assertTrue(any(f"cmpl-a-0-d's byte check {message}" in p for p in problems), problems)

    def test_every_group_of_a_pull_holds_blocks(self) -> None:
        # Full attention left out of the posted pull and checked empty.
        empty = grouped("a", full=(0, 0))
        pulls = line(kv_events={"reads_posted": ["cmpl-a-0-d"], "allocated_groups": {"cmpl-a-0-d": [
            {"group": 0, "first": 61, "blocks": [100, 101, 102]}]}}) + pulled("b", offset=300)
        problems = self.verdict([empty, grouped("b")], moved=empty["bytes"] + grouped("b")["bytes"],
                                pulls=pulls)["problems"]
        self.assertTrue(any("cmpl-a-0-d's group 1 checked no blocks" in p for p in problems), problems)
        self.assertTrue(any("not the pull's 0 from None" in p for p in problems), problems)

    def test_a_one_token_window_pulls_none_of_a_block_aligned_prompt(self) -> None:
        # The next token's window skips the 1024-token prompt whole: the pull
        # posts no window entry and checks none of its blocks.
        def verdict(window_blocks: int) -> dict:
            layout = GROUPS_LAYOUT.replace('"sliding_window": 32', '"sliding_window": 1')
            checks = [grouped(r, window=(64, window_blocks)) for r in ("a", "b")]
            entries = [{"group": 1, "first": 0, "blocks": list(range(10, 74))}]
            if window_blocks:
                entries.insert(0, {"group": 0, "first": 64, "blocks": list(range(window_blocks))})
            pulls = "".join(line(kv_events={"reads_posted": [f"cmpl-{r}-0-d"],
                                            "allocated_groups": {f"cmpl-{r}-0-d": entries}}) for r in ("a", "b"))
            moved = sum(c["bytes"] for c in checks)
            prefill = (layout + allocated("a", "b")
                       + line(kv_transfers={"send": {"count": 2, "bytes": moved, "failed": 0}}))
            decode = layout + pulls + FIRST_STEPS + "".join(line(kv_check=c) for c in checks)
            return pd.check({"completed": 2, "errors": []}, [prefill], [decode],
                            {"control": 50, "data0": moved + 100}, RUN)

        self.assertEqual(verdict(0)["problems"], [])
        problems = verdict(1)["problems"]
        self.assertTrue(any("1-token window group 0 checked 1 blocks from 64, not 0 from 64" in p
                            for p in problems), problems)

    def test_a_skipped_window_starts_at_the_prompts_end(self) -> None:
        # The empty group's first block is the prompt's end, cdiv'd: tokens
        # over block size, N - 1 tokens for a hybrid; its pull posts no
        # entry or an empty one.
        window = {"kind": "SlidingWindowSpec", "sliding_window": 1, "block_size": 16, "layers": 2,
                  "block_bytes": 4}
        full = {"kind": "FullAttentionSpec", "sliding_window": None, "block_size": 16, "layers": 2,
                "block_bytes": 4}
        state = {"kind": "MambaSpec", "sliding_window": None, "block_size": 4096, "layers": 2, "block_bytes": 4}

        def problems(groups: list[dict], prompt: int, first: int, posted: dict) -> list[str]:
            nested = [group_check(0, first, 0)] + [
                group_check(1, 0, 1 if groups[1] is state else -(-prompt // 16))]
            check = checked("a", blocks=sum(g["blocks"] for g in nested), layers=4,
                            bytes=sum(g["bytes"] for g in nested), groups=nested)
            plan = {"cmpl-a-0-d": {1: (0, nested[1]["blocks"]), **posted}}
            return pd._byte_check_problems({"groups": groups}, [check], prompt, plan, {"cmpl-a-0-d": prompt})

        self.assertEqual(problems([window, full], 1024, 64, {}), [])
        self.assertEqual(problems([window, full], 1024, 64, {0: (64, 0)}), [])
        bogus = problems([window, full], 1024, 999, {})
        self.assertTrue(any("checked 0 blocks from 999, not 0 from 64" in p for p in bogus), bogus)
        posted = problems([window, full], 1024, 999, {0: (999, 0)})
        self.assertTrue(any("not 0 from 64" in p for p in posted), posted)
        # A hybrid's 33-token prompt hands 32 tokens over: aligned, first 2.
        self.assertEqual(problems([window, state], 33, 2, {}), [])
        hybrid = problems([window, state], 33, 3, {})
        self.assertTrue(any("not 0 from 2" in p for p in hybrid), hybrid)


# A gated delta-net hybrid's: its state group (one 8-byte padded page a
# layer, 6 real) and a full-attention group of 16-token blocks, 2 layers
# each; 33-token prompts, of which decode pulls 32 tokens' pages.
HYBRID_LAYOUT = line(kv_layout={
    "tp_rank": 0, "tp_size": 1, "block_bytes": 8, "bytes": 1600,
    "groups": [{"kind": "MambaSpec", "sliding_window": None, "block_size": 4096, "layers": 2, "block_bytes": 8,
                "real_page_bytes": 6, "block_stride": 8, "mamba_type": "GDN_ATTN"},
               {"kind": "FullAttentionSpec", "sliding_window": None, "block_size": 16, "layers": 2,
                "block_bytes": 8, "real_page_bytes": 8, "block_stride": 8}]})


def hybrid_check(request: str, state=(0, 1), attention=(0, 2)) -> dict:
    groups = [{"group": 0, "first_logical_block": state[0], "blocks": state[1], "layers": 2, "block_bytes": 16,
               "bytes": state[1] * 16, "mismatched_bytes": 0},
              {"group": 1, "first_logical_block": attention[0], "blocks": attention[1], "layers": 2,
               "block_bytes": 16, "bytes": attention[1] * 16, "mismatched_bytes": 0}]
    return checked(request, blocks=state[1] + attention[1], layers=4, bytes=sum(g["bytes"] for g in groups),
                   groups=groups)


def hybrid_pull(request: str, state=(0, 1), attention=(0, 2), offset: int = 100) -> str:
    return line(kv_events={"reads_posted": [f"cmpl-{request}-0-d"], "allocated_groups": {f"cmpl-{request}-0-d": [
        {"group": 0, "first": state[0], "blocks": list(range(offset, offset + state[1]))},
        {"group": 1, "first": attention[0], "blocks": list(range(offset + 10, offset + 10 + attention[1]))}]}})


def stepped(request: str, side: str, computed: int, scheduled: int, prompt: int, drafts: int = 0) -> str:
    return line(step={"requests": [{"req_id": f"cmpl-{request}-0-{side}", "num_computed_tokens": computed,
                                    "num_scheduled_tokens": scheduled, "num_prompt_tokens": prompt,
                                    "num_spec_tokens": drafts}]})


class HybridCheckTests(unittest.TestCase):
    def test_handoff_uses_every_native_producer_rank(self) -> None:
        for p, d in ((2, 2), (1, 2), (2, 1), (2, 4), (4, 2)):
            joint = {("prefill", r): {"cmpl-a-0-p": [(0, 32, 32, 0)]} for r in range(p)}
            joint.update({("decode", r): {"cmpl-a-0-d": [(32, 1, 33, 0)]} for r in range(d)})
            for rank in range(d):
                sources = ([rank * p // d] if p <= d else list(range(rank * p // d, (rank + 1) * p // d)))
                with self.subTest(prefill_tp=p, decode_tp=d, decode_rank=rank):
                    result = pd._handoff_from_sources(joint, rank, sources, {"cmpl-a-0-d"})
                    self.assertEqual(result["problems"], [])
                    self.assertEqual(result["evidence"]["requests"], 1)
                    self.assertEqual(result["evidence"]["prompt_tokens"], [33, 33])

    def test_handoff_rejects_missing_or_wrong_steps_on_each_gather_source(self) -> None:
        for source in (0, 1):
            for bad_steps in ({}, {"cmpl-a-0-p": [(0, 31, 31, 0)]}):
                joint = {("prefill", r): {"cmpl-a-0-p": [(0, 32, 32, 0)]} for r in (0, 1)}
                joint[("decode", 0)] = {"cmpl-a-0-d": [(32, 1, 33, 0)]}
                joint[("prefill", source)] = bad_steps
                with self.subTest(source=source, bad_steps=bad_steps):
                    result = pd._handoff_from_sources(joint, 0, [0, 1], {"cmpl-a-0-d"})
                    self.assertEqual(len(result["problems"]), 1)
                    self.assertIn(f"prefill rank {source}:", result["problems"][0])
                    self.assertIn("no prefill steps" if not bad_steps else "prefill took", result["problems"][0])

    def verdict(self, checks=None, prefill_steps=None, decode_steps=None, pulls=None,
                layout=HYBRID_LAYOUT) -> dict:
        checks = checks or [hybrid_check("a"), hybrid_check("b")]
        moved = sum(c["bytes"] for c in checks)
        prefill_steps = prefill_steps if prefill_steps is not None else "".join(
            stepped(r, "p", 0, 32, 32) for r in ("a", "b"))
        decode_steps = decode_steps if decode_steps is not None else "".join(
            stepped(r, "d", 32, 1, 33) for r in ("a", "b"))
        pulls = pulls if pulls is not None else hybrid_pull("a") + hybrid_pull("b", offset=300)
        prefill = (layout + allocated("a", "b") + prefill_steps
                   + line(kv_transfers={"send": {"count": 2, "bytes": moved, "failed": 0}}))
        decode = layout + pulls + decode_steps + "".join(line(kv_check=c) for c in checks)
        return pd.check({"completed": 2, "errors": []}, [prefill], [decode],
                        {"control": 1, "data0": moved + 100}, RUN)

    def test_a_hybrid_handoff_passes_with_its_evidence(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["ranks"][0]["handoff"]["requests"], 2)
        self.assertEqual(verdict["ranks"][0]["handoff"]["prompt_tokens"], [33, 33])

    def test_a_speculative_resume_verifies_its_drafts(self) -> None:
        # Native MTP at depth 2: the last prompt token and two drafts.
        steps = "".join(stepped(r, "d", 32, 3, 33, drafts=2) for r in ("a", "b"))
        self.assertEqual(self.verdict(decode_steps=steps)["problems"], [])

    def test_hybrid_failures_are_reported(self) -> None:
        cases = {
            "state group 0 checked 2 blocks, not 1": {
                "checks": [hybrid_check("a", state=(0, 2)), hybrid_check("b")],
                "pulls": hybrid_pull("a", state=(0, 2)) + hybrid_pull("b", offset=300)},
            # The whole prompt's pages: not truncated.
            "group 1 checked 3 blocks, not the cdiv(32, 16) of its prompt": {
                "checks": [hybrid_check("a", attention=(0, 3)), hybrid_check("b")],
                "pulls": hybrid_pull("a", attention=(0, 3)) + hybrid_pull("b", offset=300)},
            "resumed at 33 tokens with 1 scheduled, not at 32 with 1": {
                "decode_steps": stepped("a", "d", 33, 1, 33) + stepped("b", "d", 32, 1, 33)},
            "prefill took a 33-token prompt to 33 tokens, not its 33 tokens but the last": {
                "prefill_steps": stepped("a", "p", 0, 33, 33) + stepped("b", "p", 0, 32, 32)},
            "cmpl-b-0-d has no prefill steps": {"prefill_steps": stepped("a", "p", 0, 32, 32)},
            # With speculative decoding the resume verifies its drafts too;
            # scheduled tokens beside them are not the last prompt token's.
            "resumed at 32 tokens with 3 scheduled, not at 32 with 2": {
                "decode_steps": stepped("a", "d", 32, 3, 33, drafts=1) + stepped("b", "d", 32, 1, 33)},
        }
        for message, changes in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(**changes)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)


    def test_every_consumed_request_needs_its_decode_steps(self) -> None:
        only_b = stepped("b", "d", 32, 1, 33)
        cases = {
            "complete absence": ({"decode_steps": ""}, ["cmpl-a-0-d has no decode steps",
                                                        "cmpl-b-0-d has no decode steps"]),
            "partial absence": ({"decode_steps": only_b}, ["cmpl-a-0-d has no decode steps"]),
            # One page where two are needed, behind the missing steps.
            "wrong pages": ({"decode_steps": only_b,
                             "checks": [hybrid_check("a", attention=(0, 1)), hybrid_check("b")],
                             "pulls": hybrid_pull("a", attention=(0, 1)) + hybrid_pull("b", offset=300)},
                            ["cmpl-a-0-d has no decode steps",
                             "cmpl-a-0-d's group 1 pages cannot be checked"]),
        }
        for case, (changes, messages) in cases.items():
            with self.subTest(case=case):
                verdict = self.verdict(**changes)
                for message in messages:
                    self.assertTrue(any(message in p for p in verdict["problems"]), verdict["problems"])
                self.assertLess(verdict["ranks"][0]["handoff"]["requests"], 2)

    def test_over_nixl_the_checks_follow_its_native_selection(self) -> None:
        """NIXL moves a state as its real spans in every state region, one
        holding none of the group's layers included, not the padding: the
        checks are compared with the layout's selection, not the pages."""
        raw = json.loads(HYBRID_LAYOUT)
        state, attention = raw["kv_layout"]["groups"]
        # Two state regions of 6 real bytes (one transfer-only), 2 padding
        # bytes each; the attention group's 8 bytes in each region.
        state["transfer"] = {"bytes_per_block": 12, "transfer_only_bytes_per_block": 6,
                             "padding_bytes_per_block": 4, "descriptors_per_block": 8}
        attention["transfer"] = {"bytes_per_block": 16, "transfer_only_bytes_per_block": 0,
                                 "padding_bytes_per_block": 0, "descriptors_per_block": 2}
        layout = line(**raw)

        def check(request: str, transfer_only: int = 6) -> dict:
            nixl = hybrid_check(request)
            state_check, attention_check = nixl["groups"]
            state_check.update(bytes=12, transfer_only_bytes=transfer_only, padding_bytes=4, descriptors=8)
            attention_check.update(bytes=32, transfer_only_bytes=0, padding_bytes=0, descriptors=4)
            nixl["bytes"] = 44
            return nixl

        right = self.verdict(checks=[check("a"), check("b")], layout=layout)
        self.assertEqual(right["problems"], [])
        self.assertEqual(right["kv_bytes_transfer_only"], 12)
        wrong = self.verdict(checks=[check("a", transfer_only=0), check("b")], layout=layout)["problems"]
        self.assertTrue(any("cmpl-a-0-d's group 0 check disagrees with NIXL's selection" in p
                            for p in wrong), wrong)

    def test_a_one_token_prompt_pulls_its_token(self) -> None:
        # Neither side truncates it: its one token's page is pulled, the token
        # computed again.
        checks = [hybrid_check(r, attention=(0, 1)) for r in ("a", "b")]
        verdict = self.verdict(
            checks=checks,
            prefill_steps="".join(stepped(r, "p", 0, 1, 1) for r in ("a", "b")),
            decode_steps="".join(stepped(r, "d", 0, 1, 1) for r in ("a", "b")),
            pulls=hybrid_pull("a", attention=(0, 1)) + hybrid_pull("b", attention=(0, 1), offset=300))
        self.assertEqual(verdict["problems"], [])

    def test_dp_ranks_take_prompts_from_the_joint_steps(self) -> None:
        """Decode DP=2: DP rank 0 logs both requests' joint steps, DP rank 1
        only its own KV (request a); a's pages are still checked."""

        def layout(dp: int) -> str:
            raw = json.loads(HYBRID_LAYOUT)
            raw["kv_layout"]["dp_rank"] = dp
            return line(**raw)

        def verdict(a_pages: int) -> dict:
            a = hybrid_check("a", attention=(0, a_pages))
            moved = a["bytes"] + hybrid_check("b")["bytes"]
            prefill = (layout(0) + allocated("a", "b") + stepped("a", "p", 0, 32, 32)
                       + stepped("b", "p", 0, 32, 32) + sent(moved))
            decode = [layout(0) + stepped("a", "d", 32, 1, 33) + stepped("b", "d", 32, 1, 33)
                      + hybrid_pull("b", offset=300) + line(kv_check=hybrid_check("b")),
                      layout(1) + hybrid_pull("a", attention=(0, a_pages)) + line(kv_check=a)]
            return pd.check({"completed": 2, "errors": []}, [prefill], decode,
                            {"control": 1, "data0": moved + 100}, dp_run(1, 2))

        right = verdict(2)
        self.assertEqual(right["problems"], [])
        self.assertEqual(right["ranks"][0]["handoff"]["requests"], 2)
        problems = verdict(3)["problems"]
        self.assertTrue(any("DP rank 1: cmpl-a-0-d's group 1 checked 3 blocks, not the cdiv(32, 16)" in p
                            for p in problems), problems)


    def test_without_byte_checks_decodes_allocations_require_their_steps(self) -> None:
        """Byte checks off (prefix caching off): the requests decode
        allocated, not those its steps name, need their handoff steps; on
        one DP rank or across two, the joint steps in DP rank 0's log."""

        def decode_allocated(*requests: str) -> str:
            return line(kv_events={"allocated": {f"cmpl-{r}-0-d": [1] for r in requests}})

        def layout(dp: int) -> str:
            raw = json.loads(HYBRID_LAYOUT)
            raw["kv_layout"]["dp_rank"] = dp
            return line(**raw)

        prefill = (HYBRID_LAYOUT + allocated("a", "b") + stepped("a", "p", 0, 32, 32)
                   + stepped("b", "p", 0, 32, 32) + sent(2000))
        steps = {r: stepped(r, "d", 32, 1, 33) for r in ("a", "b")}
        one = pd.Run(**{**RUN.__dict__, "kv_check": False})
        two = pd.Run(**{**dp_run(1, 2).__dict__, "kv_check": False})
        cases = {
            # (decode logs, run): the problems, the requests the handoff checked.
            "complete": ([HYBRID_LAYOUT + decode_allocated("a", "b") + steps["a"] + steps["b"]], one, [], 2),
            "partial": ([HYBRID_LAYOUT + decode_allocated("a", "b") + steps["a"]], one,
                        ["cmpl-b-0-d has no decode steps"], 1),
            "absent": ([HYBRID_LAYOUT + decode_allocated("a", "b")], one,
                       ["cmpl-a-0-d has no decode steps", "cmpl-b-0-d has no decode steps"], 0),
            "DP complete": ([layout(0) + decode_allocated("b") + steps["a"] + steps["b"],
                             layout(1) + decode_allocated("a")], two, [], 2),
            "DP partial": ([layout(0) + decode_allocated("b") + steps["b"],
                            layout(1) + decode_allocated("a")], two, ["cmpl-a-0-d has no decode steps"], 1),
            "DP absent": ([layout(0) + decode_allocated("b"), layout(1) + decode_allocated("a")], two,
                          ["cmpl-a-0-d has no decode steps", "cmpl-b-0-d has no decode steps"], 0),
        }
        for case, (decode, run, messages, checked) in cases.items():
            with self.subTest(case=case):
                prefills = [prefill if not run.decode.data_parallel_size > 1 else
                            prefill.replace(HYBRID_LAYOUT, layout(0))]
                verdict = pd.check({"completed": 2, "errors": []}, prefills, decode,
                                   {"control": 1, "data0": 2100}, run)
                self.assertEqual(verdict["kv_checks"], 0)
                if not messages:
                    self.assertEqual(verdict["problems"], [])
                for message in messages:
                    self.assertTrue(any(message in p for p in verdict["problems"]), verdict["problems"])
                self.assertEqual(verdict["ranks"][0]["handoff"]["requests"], checked)


class RecheckTests(SyntheticRouting, unittest.TestCase):
    """`python -m harness recheck` on a saved results directory."""

    def save(self, directory: Path, checks: list[dict], timing: bool = True) -> None:
        moved = sum(c["bytes"] for c in checks)
        saved = {**asdict(RUN), "model": str(RUN.model), "addresses": {}}
        if not timing:  # saved before server.timing existed (the roofline's runs)
            saved["server"] = {k: v for k, v in saved["server"].items() if k not in ("timing", "timing_args")}
        (directory / "run.json").write_text(json.dumps(saved, default=str))
        (directory / "bench.json").write_text(json.dumps({"completed": 2, "errors": []}))
        (directory / "routers.json").write_text(json.dumps({"control": 50, "data0": moved + 100}))
        (directory / "tent.json").write_text("{}")
        # The one prefill engine's bootstrap evidence (DP 1, TP 1: the engine hosts its worker).
        (directory / "bootstrap-p.json").write_text(json.dumps({
            "senders": {"7": "10.100.0.1:40000"}, "kv_addresses": ["10.100.0.1"], "query": {"0": {"engine_id": "u", "worker_addr": {"0": {"0": "tcp://10.100.0.1:40000"}}}},
            "processes": {"7": {"ppid": 1, "comm": "VLLM::EngineCor", "cmdline": "VLLM::EngineCore",
                                "listening": ["10.100.0.1:40000"]}},
            "engines": {"7": {"title": "EngineCore", "engine_id": "u"}}}))
        (directory / "control-ports.json").write_text(json.dumps({"nodes": ["gw"], "bytes": 40, "ports": {
            "p0": {"node": "c", "rx": 50, "tx": 0}, "p1": {"node": "p", "rx": 0, "tx": 25},
            "p2": {"node": "d", "rx": 0, "tx": 25}, "p3": {"node": "gw", "rx": 0, "tx": 40}}}))
        (directory / "prefill-p-steps.jsonl").write_text(
            GROUPS_LAYOUT + allocated("a", "b") + line(kv_transfers={"send": {"count": 2, "bytes": moved}}))
        (directory / "decode-d-steps.jsonl").write_text(
            GROUPS_LAYOUT + PULLS + FIRST_STEPS + "".join(line(kv_check=c) for c in checks))

    def test_recheck_exits_with_the_verdict(self) -> None:
        from harness.__main__ import main

        for checks, status in (([grouped("a"), grouped("b")], 0), ([grouped("a", full=(0, 32)), grouped("b")], 1)):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                self.save(Path(directory), checks)
                with contextlib.redirect_stdout(io.StringIO()) as printed:
                    self.assertEqual(main(["recheck", directory]), status)
                verdict = json.loads((Path(directory) / "verdict-recheck.json").read_text())
                self.assertEqual(json.loads(printed.getvalue())["problems"], verdict["problems"])
                self.assertEqual(bool(verdict["problems"]), bool(status))

    def test_control_evidence_that_is_no_object_fails_the_replay(self) -> None:
        for document in ("null", "[]", "0", '"ports"', "{}", '{"ports": null}'):
            with self.subTest(document=document), tempfile.TemporaryDirectory() as directory:
                self.save(Path(directory), [grouped("a"), grouped("b")])
                (Path(directory) / "control-ports.json").write_text(document)
                problems = pd.recheck(directory)["problems"]
                self.assertEqual(problems, ["the control network's port evidence is missing"])
        with tempfile.TemporaryDirectory() as directory:
            self.save(Path(directory), [grouped("a"), grouped("b")])
            (Path(directory) / "control-ports.json").unlink()
            with self.assertRaises(FileNotFoundError):
                pd.recheck(directory)

    def test_a_run_saved_without_a_timing_provider_still_rechecks(self) -> None:
        from harness.__main__ import main

        with tempfile.TemporaryDirectory() as directory:
            self.save(Path(directory), [grouped("a"), grouped("b")], timing=False)
            self.assertNotIn("timing", json.loads((Path(directory) / "run.json").read_text())["server"])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["recheck", directory]), 0)


TP2 = pd.Run(name="t", topology="t", model=Path("/m"), prefill=pd.Side("p", ["data0", "data1"]),
             decode=pd.Side("d", ["data0", "data1"]), client="c", control_network="control", router=pd.Router("gw"),
             server=pd.Server(tensor_parallel_size=2), workload=pd.Workload(num_prompts=2))


def rank_logs(rank: int, moved: int = 2000, ok: bool = True) -> tuple[str, str]:
    """A rank's prefill and decode step logs: its layout (4 KB a block and
    layer), two byte-checked requests of 5 blocks of 50 layers."""

    layout = line(kv_layout={"tp_rank": rank, "tp_size": 2, "shape": [8, 1, 16, 128],
                             "block_bytes": 4, "bytes": 1600})
    return (layout + allocated("a", "b") + line(kv_transfers={"send": {"count": 2, "bytes": moved, "failed": 0}}),
            layout + "".join(line(kv_check=checked(r, ok, blocks=5, layers=50)) for r in ("a", "b")))


def tent(written: float = 2000, registered: float = 1600) -> dict:
    return {"tent_write_bytes_total": written, "tent_write_requests_total": 2,
            "tent_registered_buffer_bytes": registered}


class TensorParallelCheckTests(unittest.TestCase):
    def verdict(self, logs=None, routers=None, tents=None, run=TP2) -> dict:
        logs = logs or [rank_logs(0), rank_logs(1)]
        tents = tents or {"prefill": [tent(), tent()], "decode": [tent(0), tent(0)]}
        return pd.check({"completed": 2, "errors": []}, [p for p, _ in logs], [d for _, d in logs],
                        routers or {"control": 50, "data0": 2100, "data1": 2100}, run, tents)

    def test_each_rank_is_checked_on_its_own_rail(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["requests"], 2)  # once, not per rank
        self.assertEqual(verdict["kv_bytes_checked"], 4000)
        self.assertEqual(verdict["transfers"]["bytes"], 4000)
        self.assertEqual([r["kv_layout"]["decode"]["tp_rank"] for r in verdict["ranks"]], [0, 1])

    def test_a_shared_rail_carries_every_rank(self) -> None:
        shared = pd.Run(**{**TP2.__dict__, "prefill": pd.Side("p", "data0"), "decode": pd.Side("d", "data0")})
        self.assertEqual(self.verdict(run=shared, routers={"control": 50, "data0": 4100, "data1": 0})["problems"], [])
        problems = self.verdict(run=shared, routers={"control": 50, "data0": 2100, "data1": 2000})["problems"]
        self.assertTrue(any("fewer than the 4000" in p for p in problems), problems)

    def test_each_rank_failure_is_reported(self) -> None:
        cases = {
            "rank 1: 2 request(s) received KV bytes unlike": {"logs": [rank_logs(0), rank_logs(1, ok=False)]},
            "rank 1: the connector moved 1000 bytes": {"logs": [rank_logs(0), rank_logs(1, moved=1000)],
                                                       "tents": {"prefill": [tent(), tent(1000)],
                                                                 "decode": [tent(0), tent(0)]}},
            # Beyond headers and acknowledgements (5% and 1 MiB).
            "another rank's KV": {"routers": {"control": 50, "data0": 2100 + (8 << 20), "data1": 2100}},
            "rank 0: TENT wrote 1500": {"tents": {"prefill": [tent(1500), tent()], "decode": [tent(0), tent(0)]}},
            "rank 1: decode's TENT metrics did not answer": {
                "tents": {"prefill": [tent(), tent()], "decode": [tent(0), None]}},
            "registered 100 bytes, less than its 1600": {
                "tents": {"prefill": [tent(), tent(registered=100)], "decode": [tent(0), tent(0)]}},
            # Rank 0's logs supplied for rank 1 too.
            "rank 1: prefill's log is TP rank 0 of 2, not 1 of 2": {"logs": [rank_logs(0), rank_logs(0)]},
            "rank 0: decode logged no KV layout": {
                "logs": [(rank_logs(0)[0], rank_logs(0)[1].split("\n", 1)[1]), rank_logs(1)]},
            # Counters that exist but lack what the checks need.
            "rank 1: prefill's TENT metrics lack tent_registered_buffer_bytes, tent_write_bytes_total": {
                "tents": {"prefill": [tent(), {}], "decode": [tent(0), tent(0)]}},
            "rank 0: decode's TENT metrics lack tent_registered_buffer_bytes": {
                "tents": {"prefill": [tent(), tent()], "decode": [{}, tent(0)]}},
        }
        for message, given in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(**given)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)

    def test_cleanup_finds_renamed_workers(self) -> None:
        for cmdline, comm, ours in (
            (b"/opt/venv/bin/python3\0-c\0...", b"python3", True),
            (b"VLLM::Worker_TP1\0", b"VLLM::Worker_TP", True),
            (b"VLLM::EngineCore\0", b"VLLM::EngineCor", True),
            (b"", b"VLLM::Worker_TP", True),  # an emptied command line
            (b"sleep\0infinity\0", b"sleep", False),
        ):
            with self.subTest(cmdline=cmdline):
                self.assertIs(pd.is_service(cmdline, comm), ours)
        self.assertIn("def is_service", pd.STOP)

    def test_byte_checks_must_match_the_rank_layout(self) -> None:
        prefill, decode = rank_logs(1)
        problems = self.verdict(logs=[rank_logs(0), (prefill, decode.replace('"blocks": 5', '"blocks": 4'))])[
            "problems"]
        self.assertTrue(any("disagrees with the rank's KV layout" in p for p in problems), problems)
        # An older layout's check without its blocks is not let through.
        problems = self.verdict(logs=[rank_logs(0), (prefill, decode.replace('"blocks": 5, ', ''))])["problems"]
        self.assertTrue(any("disagrees with the rank's KV layout" in p for p in problems), problems)

    def test_rank_networks_must_pair(self) -> None:
        self.assertEqual(TP2.prefill.rank_networks(2), ["data0", "data1"])
        self.assertEqual(pd.Side("p", "data0").rank_networks(2), ["data0", "data0"])
        own = " ".join(pd.server_command(TP2, "prefill", ["10.100.0.1", "10.101.0.1"]))
        self.assertIn('"rank_addresses": ["10.100.0.1", "10.101.0.1"]', own)
        shared = " ".join(pd.server_command(TP2, "prefill", ["10.100.0.1", "10.100.0.1"]))
        self.assertNotIn("rank_addresses", shared)
        self.assertIn("--tensor-parallel-size 2", shared)


def dp_layout(dp: int, tp: int = 0, tp_size: int = 1) -> str:
    return line(kv_layout={"dp_rank": dp, "tp_rank": tp, "tp_size": tp_size, "block_bytes": 4, "bytes": 1600})


def sent(nbytes: int) -> str:
    return line(kv_transfers={"send": {"count": nbytes // 1000, "bytes": nbytes, "failed": 0}})


def dp_run(prefill_dp: int, decode_dp: int, tp: int = 1, model: Path = MOE_MODEL) -> pd.Run:
    networks = "data0" if tp == 1 else ["data0", "data1"]
    return pd.Run(name="t", topology="t", model=model,
                  prefill=pd.Side("p", networks, data_parallel_size=prefill_dp),
                  decode=pd.Side("d", networks, data_parallel_size=decode_dp), client="c",
                  control_network="control", router=pd.Router("gw"), server=pd.Server(tensor_parallel_size=tp),
                  workload=pd.Workload(num_prompts=2))


class DataParallelCheckTests(unittest.TestCase):
    """Requests a and b; prefill DP=2, decode DP=1 unless a case says."""

    def verdict(self, prefill=None, decode=None, run=None, routers=None) -> dict:
        prefill = prefill or [dp_layout(0) + allocated("a") + sent(1000), dp_layout(1) + allocated("b") + sent(1000)]
        decode = decode or [dp_layout(0) + line(kv_check=checked("a")) + line(kv_check=checked("b"))]
        run = run or dp_run(len(prefill), len(decode))
        return pd.check({"completed": 2, "errors": []}, prefill, decode,
                        routers or {"control": 50, "data0": 2100}, run)

    def test_each_request_has_one_producer_and_one_consumer(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["ownership"], {"producers": {0: 1, 1: 1}, "consumers": {0: 2}})
        # Bytes counted once: each request's KV crossed data0 once.
        self.assertEqual((verdict["kv_checks"], verdict["kv_bytes_checked"], verdict["transfers"]["bytes"]),
                         (2, 2000, 2000))
        self.assertEqual([r["requests"] for r in verdict["dp_ranks"]["prefill"]], [1, 1])

    def test_ownership_failures_are_reported(self) -> None:
        cases = {
            # Both decode DP ranks checked a.
            "1 request(s) more than one of the decode DP ranks consumed": {"decode": [
                dp_layout(0) + line(kv_check=checked("a")) + line(kv_check=checked("b")),
                dp_layout(1) + line(kv_check=checked("a"))]},
            "decode logged 3 byte check(s) for 2 distinct request(s)": {"decode": [
                dp_layout(0) + line(kv_check=checked("a")) + line(kv_check=checked("b")),
                dp_layout(1) + line(kv_check=checked("a"))]},
            "1 request(s) more than one of the prefill DP ranks produced": {"prefill": [
                dp_layout(0) + allocated("a", "b") + sent(1000), dp_layout(1) + allocated("b") + sent(1000)]},
            "1 request(s) not both produced and consumed": {"decode": [dp_layout(0) + line(kv_check=checked("a"))]},
            "DP rank 1: prefill's log is DP rank 0, not 1": {"prefill": [
                dp_layout(0) + allocated("a") + sent(1000), dp_layout(0) + allocated("b") + sent(1000)]},
        }
        for message, given in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(**given)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)

    def test_every_request_is_byte_checked_once(self) -> None:
        """Two requests completed, but the logs hold only a, checked twice:
        as many checks as requests, yet b went unchecked."""
        prefill = [dp_layout(0) + allocated("a") + sent(1000), dp_layout(1) + sent(1000)]
        decode = [dp_layout(0) + line(kv_check=checked("a")) + line(kv_check=checked("a"))]
        problems = self.verdict(prefill, decode)["problems"]
        for message in ("decode byte-checked 1 distinct request(s), not 2",
                        "decode logged 2 byte check(s) for 1 distinct request(s)"):
            self.assertTrue(any(message in p for p in problems), problems)

    def test_without_byte_checks_ownership_comes_from_allocations(self) -> None:
        """kv_check off: decode's allocations name the consumers, and no
        byte check is reported (regression: they were counted as checks)."""
        run = dp_run(2, 1)
        run.kv_check = False
        decode = [dp_layout(0) + allocated("a", "b")]
        verdict = self.verdict(decode=decode, run=run)
        self.assertEqual(verdict["problems"], [])
        self.assertEqual((verdict["kv_checks"], verdict["kv_bytes_checked"]), (0, 0))
        self.assertEqual(verdict["ranks"][0]["consumed"], 2)
        self.assertEqual(verdict["ownership"], {"producers": {0: 1, 1: 1}, "consumers": {0: 2}})
        # A request decode never took is still caught.
        problems = self.verdict(decode=[dp_layout(0) + allocated("a")], run=run)["problems"]
        self.assertTrue(any("1 request(s) not both produced and consumed" in p for p in problems), problems)

    def test_every_tp_rank_of_a_request_has_the_same_owner(self) -> None:
        # TP=2: rank 1 places a and b on the other prefill DP ranks.
        prefill = [dp_layout(0, 0, 2) + allocated("a") + sent(1000), dp_layout(0, 1, 2) + allocated("b") + sent(1000),
                   dp_layout(1, 0, 2) + allocated("b") + sent(1000), dp_layout(1, 1, 2) + allocated("a") + sent(1000)]
        decode = [dp_layout(0, tp, 2) + line(kv_check=checked("a")) + line(kv_check=checked("b")) for tp in (0, 1)]
        problems = self.verdict(prefill, decode, dp_run(2, 1, tp=2),
                                {"control": 50, "data0": 2100, "data1": 2100})["problems"]
        self.assertTrue(any("rank 1: 2 request(s) with other prefill DP ranks than rank 0's" in p
                            for p in problems), problems)

    def test_ranks_follow_the_hosts_order(self) -> None:
        run = dp_run(2, 2, tp=2)
        self.assertEqual(run.prefill.ranks(2), [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual(run.decode.step_log("decode", 1, 1), "decode-d-steps.dp1.tp1.jsonl")
        self.assertEqual(run.decode.step_log("decode", 0, 0), "decode-d-steps.jsonl")
        command = " ".join(pd.server_command(run, "prefill", ["10.100.0.1", "10.101.0.1"]))
        # Local DP rank by TP rank, each TP rank on its own rail.
        self.assertIn('"rank_addresses": ["10.100.0.1", "10.101.0.1", "10.100.0.1", "10.101.0.1"]', command)
        self.assertIn("--data-parallel-size 2 --enable-expert-parallel", command)
        self.assertNotIn("--data-parallel-size", " ".join(pd.server_command(TP2, "prefill", ["a", "b"])))

    def test_a_dense_models_dp_ranks_are_independent_engines_each_with_its_tp_addresses(self) -> None:
        run = dp_run(2, 2, tp=2, model=DENSE_MODEL)
        for role in ("prefill", "decode"):
            command = " ".join(pd.server_command(run, role, ["10.100.0.1", "10.101.0.1"]))
            # Each engine lists its TP ranks (an MoE instance's list is DP by TP).
            self.assertIn('"rank_addresses": ["10.100.0.1", "10.101.0.1"]}', command)
            self.assertIn("--data-parallel-size 2 ", command)
            self.assertNotIn("--enable-expert-parallel", command)
            self.assertNotIn("--data-parallel-multi-port-external-lb", command)
        self.assertEqual(run.prefill.ranks(2), [(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_moe_is_read_from_the_config_as_vllm_counts_experts(self) -> None:
        cases = {
            "dense": ({"model_type": "qwen3"}, False),
            "scalar experts": ({"num_experts": 8}, True),
            "an alias": ({"num_local_experts": 8}, True),
            "experts of the text config": ({"text_config": {"n_routed_experts": 64}}, True),
            "the text config decides": ({"num_experts": 8, "text_config": {"model_type": "qwen3"}}, False),
            "a zero per layer list": ({"num_experts": [0, 0]}, False),
            "a per layer list": ({"num_experts": [8, 8]}, True),
            "the first alias present wins": ({"num_experts": 0, "n_routed_experts": 8}, False),
            "only in the MoE blocks": ({"block_configs": [{"block_type": "attention"},
                                                          {"block_type": "moe", "n_routed_experts": 8}]}, True),
            "no MoE block": ({"block_configs": [{"block_type": "attention"}]}, False),
            "an llm_config": ({"llm_config": {"num_experts": 8}}, True),
            "a list decides before the blocks": ({"num_experts": [0, 0], "block_configs": [
                {"block_type": "moe", "n_routed_experts": 8}]}, False),
        }
        for name, (config, expected) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "config.json").write_text(json.dumps(config))
                self.assertIs(pd.moe_of(Path(directory)), expected)

    def test_the_bootstrap_servers_published_workers_are_checked_against_the_listeners(self) -> None:
        def found(dp=2, tp=2):
            kv = [f"10.{100 + t}.0.1" for t in range(tp)]
            ports = {(d, t): 40000 + 10 * d + t for d in range(dp) for t in range(tp)}
            title = lambda t: f"Worker_TP{t}" if tp > 1 else "Worker"
            processes, engines, query = {}, {}, {}
            for d in range(dp):
                engine_pid = str(100 + d)
                engine_id = f"u_dp{d}" if dp > 1 else "u"
                processes[engine_pid] = {"ppid": 1, "comm": "VLLM::EngineCor", "listening": [],
                                         "cmdline": f"VLLM::EngineCore_DP{d}" if dp > 1 else "VLLM::EngineCore"}
                query[str(d)] = {"engine_id": engine_id, "worker_addr": {}}
                for t in range(tp):
                    pid = str(200 + 10 * d + t)
                    processes[pid] = {"ppid": int(engine_pid), "comm": "VLLM::Worker", "cmdline": f"VLLM::{title(t)}",
                                      "listening": [f"{kv[t]}:{ports[d, t]}", "0.0.0.0:8998", f"127.0.0.1:{9100 + t}"]}
                    engines[pid] = {"title": title(t), "engine_id": engine_id}
                    query[str(d)]["worker_addr"][str(t)] = {"0": f"tcp://{kv[t]}:{ports[d, t]}"}
            senders = {pid: process["listening"][0] for pid, process in processes.items() if pid in engines}
            return {"kv_addresses": kv, "query": query, "processes": processes, "engines": engines,
                    "senders": senders}

        def problems(change=None, dp=2, tp=2, **sizes):
            data = found(**sizes) if sizes else found(dp, tp)
            if change:
                change(data)
            return pd.bootstrap_problems(data, dp, tp, "prefill0")

        def has(text, found_problems):
            return any(text in p for p in found_problems)

        found_wide = lambda: found(1, 16)
        self.assertEqual(problems(), [])
        # TP 1 (a worker's title has no rank) and DP 1 (its engine id has no suffix, the engine hosts it).
        self.assertEqual(pd.bootstrap_problems(found(2, 1), 2, 1, "p"), [])
        solo = found(1, 1)
        solo["processes"]["200"]["cmdline"], solo["processes"]["200"]["ppid"] = "VLLM::EngineCore", 1
        self.assertEqual(pd.bootstrap_problems(solo, 1, 1, "p"), [])
        self.assertTrue(has("not 0..2", pd.bootstrap_problems(found(2, 2), 3, 2, "p")))
        self.assertTrue(has("TP ranks", pd.bootstrap_problems(found(2, 2), 2, 3, "p")))
        # Another engine's id with the right suffix, or the other engine's workers.
        def foreign(data):
            data["query"]["0"]["engine_id"] = "other_dp0"
        self.assertTrue(has("but its listener logged u_dp0", problems(foreign)))
        self.assertTrue(has("of different servers", problems(foreign)))
        def swapped(data):
            query = data["query"]
            query["0"], query["1"] = query["1"], query["0"]
        self.assertTrue(has("not VLLM::EngineCore_DP1", problems(swapped)) or has("EngineCore_DP", problems(swapped)))
        self.assertTrue(has("`<id>_dp<rank>`", problems(swapped)))
        # Another TP rank's address under this rank.
        def crossed(data):
            entry = data["query"]["0"]["worker_addr"]
            entry["0"], entry["1"] = entry["1"], entry["0"]
        self.assertTrue(has("not on TP rank 0's KV address", problems(crossed)))
        # The worker's HTTP bootstrap port, or its metrics', on the right host; a port nobody listens on.
        def http(data):
            data["query"]["0"]["worker_addr"]["0"]["0"] = "tcp://10.100.0.1:8998"
        self.assertTrue(has("0 process(es) logged binding that sender endpoint", problems(http)))
        def lost(data):
            data["query"]["1"]["worker_addr"]["1"]["0"] = "tcp://10.101.0.1:1"
        self.assertTrue(has("0 process(es) logged binding", problems(lost)))
        # An empty stage map, a missing rank, no registration at all.
        def empty(data):
            data["query"]["0"]["worker_addr"]["1"] = {}
        self.assertTrue(has("not stage 0", problems(empty)))
        def gone(data):
            data["query"].clear()
        self.assertTrue(has("publishes DP ranks []", problems(gone)))
        # Another socket of the same worker on the same host: not the endpoint it logged binding as its sender.
        def sibling(data):
            data["processes"]["200"]["listening"].append("10.100.0.1:46000")
            data["query"]["0"]["worker_addr"]["0"]["0"] = "tcp://10.100.0.1:46000"
        self.assertTrue(has("0 process(es) logged binding that sender endpoint", problems(sibling)))
        # Malformed entries are problems, never skipped or raised.
        for name, change in {
            "a null entry": lambda data: data["query"].__setitem__("1", None),
            "a null stage map": lambda data: data["query"]["0"]["worker_addr"].__setitem__("1", None),
            "a non-numeric rank": lambda data: data["query"]["0"]["worker_addr"].__setitem__("x", {}),
            "no engine id": lambda data: data["query"]["0"].pop("engine_id"),
        }.items():
            with self.subTest(name):
                self.assertTrue(problems(change), name)
        self.assertTrue(pd.bootstrap_problems({}, 2, 2, "p"))
        # Ranks are compared as numbers: a TP16 capture is valid.
        self.assertEqual(pd.bootstrap_problems(found_wide(), 1, 16, "p"), [])
        # A noncanonical rank key beside a null canonical one; a null title; the evidence file cut short.
        def padded(data):
            data["query"]["00"], data["query"]["1"] = data["query"].pop("0"), data["query"]["1"]
            data["query"]["0"] = None
        self.assertTrue(has("rank key '00'", problems(padded)))
        self.assertTrue(has("DP rank 0's entry is malformed", problems(padded)))
        def untitled(data):
            data["engines"]["200"]["title"] = None
        self.assertTrue(has("logged no title", problems(untitled)))
        # A worker that is not the rank's: TP rank by title.
        def retitled(data):
            data["engines"]["200"]["title"] = "Worker_TP1"
        self.assertTrue(has("its listener is Worker_TP1", problems(retitled)))

    def test_a_mooncake_runs_bootstrap_evidence_is_required_by_its_verdicts(self) -> None:
        run = dp_run(2, 2, model=DENSE_MODEL)
        run.server.connector = "mooncake"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(pd._bootstrap_verdict(run, output),
                             ["p's bootstrap evidence (bootstrap-p.json) is missing"])
            (output / "bootstrap-p.json").write_text("[]")
            self.assertEqual(pd._bootstrap_verdict(run, output), ["p's bootstrap evidence is malformed"])
            (output / "bootstrap-p.json").write_text("{}")
            self.assertEqual(pd._bootstrap_verdict(run, output), ["p's bootstrap evidence is malformed"])
            (output / "bootstrap-p.json").write_text('{"query": {"0": ')
            self.assertEqual(len(pd._bootstrap_verdict(run, output)), 1)
            self.assertIn("is malformed (JSONDecodeError", pd._bootstrap_verdict(run, output)[0])
        run.server.connector = "nixl"
        self.assertEqual(pd._bootstrap_verdict(run, Path("/nowhere")), [])

    def test_a_dp_launch_needs_the_models_config_to_tell_moe_from_dense(self) -> None:
        run = dp_run(2, 2, model=Path("/not-fetched"))
        with self.assertRaisesRegex(ValueError, "needs the model's config"):
            pd.server_command(run, "prefill", ["10.100.0.1"])


def instances_run(prefill: list[str], decode: list[str], tp: int = 1, connector: str = "mooncake") -> pd.Run:
    networks = "data0" if tp == 1 else ["data0", "data1"]
    return pd.Run(name="t", topology="t", model=Path("/m"), prefill=pd.Side(prefill, networks),
                  decode=pd.Side(decode, networks), client="c", control_network="control", router=pd.Router("gw"),
                  server=pd.Server(tensor_parallel_size=tp, connector=connector), workload=pd.Workload(num_prompts=2))


def tent_counters(moved: int | None, mover: str = "tent_write_bytes_total") -> dict:
    counters = {"tent_registered_buffer_bytes": 4096.0}
    if moved is not None:
        counters[mover] = float(moved)
    return counters


class InstanceCheckTests(unittest.TestCase):
    """Several instances of a side (``node`` a list, one engine each, as
    llm-d's router places requests): each instance owns requests as a DP
    rank would, its logs saying DP rank 0."""

    def verdict(self, prefill=None, decode=None, run=None, tent=None) -> dict:
        prefill = prefill or [dp_layout(0) + allocated("a") + sent(1000), dp_layout(0) + allocated("b") + sent(1000)]
        decode = decode or [dp_layout(0) + line(kv_check=checked("a")), dp_layout(0) + line(kv_check=checked("b"))]
        run = run or instances_run(["p0", "p1"][:len(prefill)], ["d0", "d1"][:len(decode)])
        return pd.check({"completed": 2, "errors": []}, prefill, decode, {"control": 50, "data0": 2100}, run, tent)

    def test_each_request_has_one_producing_and_one_consuming_instance(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["ownership"], {"producers": {0: 1, 1: 1}, "consumers": {0: 1, 1: 1}})
        self.assertEqual(verdict["instances"], {"prefill": ["p0", "p1"], "decode": ["d0", "d1"]})
        self.assertEqual((verdict["kv_bytes_checked"], verdict["transfers"]["bytes"]), (2000, 2000))

    def test_ownership_failures_name_the_instance(self) -> None:
        cases = {
            "1 request(s) more than one of the prefill DP ranks produced": {"prefill": [
                dp_layout(0) + allocated("a", "b") + sent(1000), dp_layout(0) + allocated("b") + sent(1000)]},
            "1 request(s) more than one of the decode DP ranks consumed": {"decode": [
                dp_layout(0) + line(kv_check=checked("a")) + line(kv_check=checked("b")),
                dp_layout(0) + line(kv_check=checked("b"))]},
            # Each instance is one engine: its log says DP rank 0.
            "instance p1: prefill's log is DP rank 1, not 0": {"prefill": [
                dp_layout(0) + allocated("a") + sent(1000), dp_layout(1) + allocated("b") + sent(1000)]},
        }
        for message, given in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(**given)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)

    def test_an_idle_instance_may_lack_its_moved_series_an_owner_may_not(self) -> None:
        prefill = [dp_layout(0) + allocated("a", "b") + sent(2000), dp_layout(0)]
        decode = [dp_layout(0) + line(kv_check=checked("a")) + line(kv_check=checked("b"))]
        run = instances_run(["p0", "p1"], ["d0"])
        verdict = self.verdict(prefill, decode, run, {"prefill": [tent_counters(2000), tent_counters(None)],
                                                      "decode": [tent_counters(None)]})
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["idle_ranks"], ["prefill p1 rank 0"])
        problems = self.verdict(prefill, decode, run, {"prefill": [tent_counters(None), tent_counters(None)],
                                                       "decode": [tent_counters(None)]})["problems"]
        self.assertTrue(any("instance p0: prefill's TENT metrics lack tent_write_bytes_total" in p
                            for p in problems), problems)

    def test_instances_name_their_step_logs_and_ports(self) -> None:
        run = instances_run(["p0", "p1"], ["d0"], tp=2)
        side = run.prefill
        self.assertEqual(side.ranks(2), [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual([side.node_of(dp) for dp, _ in side.ranks(2)], ["p0", "p0", "p1", "p1"])
        self.assertEqual(side.step_log("prefill", 1, 1), "prefill-p1-steps.tp1.jsonl")
        self.assertEqual(side.remote_step_log(1, 1), f"{pd.WORK}/steps.tp1.jsonl")
        self.assertEqual(side.local_rank(1, 1, 2), 1)
        # One instance: its DP ranks' suffixes, its node's name.
        self.assertEqual(RUN.prefill.step_log("prefill", 0, 0), "prefill-p-steps.jsonl")
        self.assertEqual(dp_run(2, 1).prefill.local_rank(1, 0, 1), 1)

    def test_an_idle_dp_rank_may_lack_its_moved_series_an_owner_may_not(self) -> None:
        # The Mooncake sidecar names any of the prefill's DP ranks: one may get none.
        prefill = [dp_layout(0) + allocated("a", "b") + sent(2000), dp_layout(1)]
        decode = [dp_layout(0) + line(kv_check=checked("a")) + line(kv_check=checked("b"))]
        run = dp_run(2, 1)
        verdict = pd.check({"completed": 2, "errors": []}, prefill, decode, {"control": 1, "data0": 2100}, run,
                           {"prefill": [tent_counters(2000), tent_counters(None)], "decode": [tent_counters(None)]})
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["idle_ranks"], ["prefill DP rank 1 rank 0"])
        problems = pd.check({"completed": 2, "errors": []}, prefill, decode, {"control": 1, "data0": 2100}, run,
                            {"prefill": [tent_counters(None), tent_counters(None)],
                             "decode": [tent_counters(None)]})["problems"]
        self.assertTrue(any("DP rank 0: prefill's TENT metrics lack tent_write_bytes_total" in p for p in problems),
                        problems)

    def test_node_lists_are_validated(self) -> None:
        base = {"name": "t", "topology": "t", "model": "org/name", "client": "c", "control_network": "control",
                "router": {"gateway": "gw"},
                "prefill": {"node": ["p0", "p1"], "kv_network": "data0"},
                "decode": {"node": ["d0"], "kv_network": "data0"}, "server": {"timing": "zero"}}
        self.assertEqual(pd.parse(base, Path("/x/run.yaml")).prefill.nodes, ["p0", "p1"])
        cases = {
            "distinct nodes": {"prefill": {"node": ["p0", "p0"], "kv_network": "data0"}},
            "one engine each": {"prefill": {"node": ["p0", "p1"], "kv_network": "data0", "data_parallel_size": 2}},
            "distinct nodes ": {"decode": {"node": ["p1"], "kv_network": "data0"}},
            "equal TP sizes": {"decode": {"node": ["d0"], "kv_network": "data0", "tensor_parallel_size": 2}},
        }
        for message, change in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message.strip()):
                pd.parse({**base, **change}, Path("/x/run.yaml"))

    def test_the_front_ends_control_traffic_is_left_out_of_the_guard(self) -> None:
        run = instances_run(["p0", "p1"], ["d0", "d1"])
        prefill = [dp_layout(0) + allocated("a") + sent(1000), dp_layout(0) + allocated("b") + sent(1000)]
        decode = [dp_layout(0) + line(kv_check=checked("a")), dp_layout(0) + line(kv_check=checked("b"))]
        routers = {"control": 450, "data0": 2100}
        problems = pd.check({"completed": 2, "errors": []}, prefill, decode, routers, run)["problems"]
        self.assertTrue(any("control network carried 450 bytes, too much" in p for p in problems), problems)
        # 450 bytes received at the router: 300 of them to or from the gateway.
        verdict = pd.check({"completed": 2, "errors": []}, prefill, decode, routers, run, None, None,
                           self.exemption())
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["control_network"], {"bytes": 450, "front_end_bytes": 300,
                                                      "front_end_nodes": ["gw"], "guarded_bytes": 150})
        # What is left must still stay under the guard.
        exempt = self.exemption()
        exempt["ports"]["c"]["rx"] += 100
        problems = pd.check({"completed": 2, "errors": []}, prefill, decode, routers, run, None, None,
                            exempt)["problems"]
        self.assertTrue(any("carried 250 bytes besides the front end's 300" in p for p in problems), problems)

    @staticmethod
    def exemption() -> dict:
        ports = {"gw": (100, 200), "p0": (50, 25), "p1": (50, 25), "d0": (50, 25), "d1": (50, 25), "c": (150, 150)}
        return {"nodes": ["gw"], "bytes": 300,
                "ports": {node: {"node": node, "rx": rx, "tx": tx} for node, (rx, tx) in ports.items()}}

    def test_the_control_evidence_is_derived_from_its_ports_and_validated(self) -> None:
        run = instances_run(["p0", "p1"], ["d0", "d1"])
        self.assertEqual(pd.control_evidence(self.exemption(), run), (450, 300, []))

        def drop(node):
            def change(exempt): del exempt["ports"][node]
            return change

        def counter(node, key, value):
            def change(exempt): exempt["ports"][node][key] = value
            return change

        cases = {
            "no ports": (lambda e: e.update(ports={}, bytes=0), "lacks the nodes"),
            "no port collection": (lambda e: e.pop("ports"), "port evidence is missing"),
            "an inflated subtotal": (lambda e: e.update(bytes=3_000_000_000), "its ports give 300"),
            "a zeroed subtotal": (lambda e: e.update(bytes=0), "its ports give 300"),
            "no gateway port": (drop("gw"), r"lacks the nodes \['gw'\]"),
            "no server port": (drop("d1"), r"lacks the nodes \['d1'\]"),
            "no client port": (drop("c"), r"lacks the nodes \['c'\]"),
            "other exempt nodes": (lambda e: e.update(nodes=["gw", "c"]), "the front end's are"),
            "no exempt nodes": (lambda e: e.update(nodes=[]), "the front end's are"),
            "a negative counter": (counter("p0", "rx", -5), "invalid rx counter"),
            "a counter that is no integer": (counter("gw", "tx", "200"), "invalid tx counter"),
            "a boolean counter": (counter("gw", "tx", True), "invalid tx counter"),
            "a share above the total": (lambda e: (e["ports"]["gw"].update(tx=5000), e.update(bytes=5100)),
                                        "exceed the control network's 450"),
        }
        prefill = [dp_layout(0) + allocated("a") + sent(1000), dp_layout(0) + allocated("b") + sent(1000)]
        decode = [dp_layout(0) + line(kv_check=checked("a")), dp_layout(0) + line(kv_check=checked("b"))]
        for name, (change, message) in cases.items():
            with self.subTest(name):
                exempt = self.exemption()
                change(exempt)
                self.assertRegex("; ".join(pd.control_evidence(exempt, run)[2]), message)
                # And through the checks: a problem, never a pass.
                problems = pd.check({"completed": 2, "errors": []}, prefill, decode,
                                    {"control": 450, "data0": 2100}, run, None, None, exempt)["problems"]
                self.assertRegex("; ".join(problems), message)
        # The exempt node is the run's gateway, whatever the saved one says.
        other = pd.Run(**{**run.__dict__, "router": pd.Router("c")})
        self.assertRegex("; ".join(pd.control_evidence(self.exemption(), other)[2]),
                         r"the front end's are \['c'\]")

    def test_the_front_ends_share_is_its_ports_bytes_both_ways(self) -> None:
        before = {"p1": {"node": "gw", "rx": 10, "tx": 20}, "p2": {"node": "d0", "rx": 5, "tx": 5}}
        after = {"p1": {"node": "gw", "rx": 110, "tx": 220}, "p2": {"node": "d0", "rx": 1005, "tx": 905}}
        exempt = pd.control_exemption(before, after, ["gw"])
        self.assertEqual(exempt["bytes"], 300)
        self.assertEqual(exempt["ports"]["p2"]["rx"], 1000)
        self.assertEqual(pd.control_exemption(before, after, [])["bytes"], 0)


# DeepSeek-V4-like packed blocks: every 20-byte block is all of one group's
# 2 layers, 8-byte pages of 6 real bytes; a 16-token window and compressed
# latents (16 tokens a block), decode recomputing the prompt's last token.
def packed_group(kind: str, window: int | None, **changes) -> dict:
    return {"kind": kind, "sliding_window": window, "block_size": 16, "layers": 2, "block_bytes": 8,
            "real_page_bytes": 6, "block_stride": 20,
            "members": [{"kind": kind, "layers": 2, "offsets": [0, 8], "page_bytes": 8, "real_page_bytes": 6,
                         "dtype": "uint8", "tokens_per_state": 1.0}],
            "checked_bytes_per_block": 12, "transfer_bytes_per_block": 20, **changes}


def packed_layout(window=None, full=None, **changes) -> str:
    return line(kv_layout={"tp_rank": 0, "tp_size": 1, "block_bytes": 20, "bytes": 2000, "prefill_backoff": 1,
                           "groups": [window or packed_group("SlidingWindowMLASpec", 16),
                                      full or packed_group("MLAAttentionSpec", None)], **changes})


def packed_check(request: str, window=(1, 1), full=(0, 2)) -> dict:
    groups = [{"group": 0, "first_logical_block": window[0], "blocks": window[1], "layers": 2, "block_bytes": 12,
               "bytes": window[1] * 12, "mismatched_bytes": 0},
              {"group": 1, "first_logical_block": full[0], "blocks": full[1], "layers": 2, "block_bytes": 12,
               "bytes": full[1] * 12, "mismatched_bytes": 0}]
    return checked(request, blocks=window[1] + full[1], layers=4, bytes=sum(g["bytes"] for g in groups),
                   groups=groups)


def packed_pull(request: str, window=(1, 1), full=(0, 2), offset: int = 100) -> str:
    return line(kv_events={"reads_posted": [f"cmpl-{request}-0-d"], "allocated_groups": {f"cmpl-{request}-0-d": [
        {"group": 0, "first": window[0], "blocks": list(range(offset, offset + window[1]))},
        {"group": 1, "first": full[0], "blocks": list(range(offset + 10, offset + 10 + full[1]))}]}})


class PackedCheckTests(unittest.TestCase):
    """33-token prompts: decode loads 32 tokens (2 blocks of the latents,
    the window's last block from place 1) and computes the last."""

    def verdict(self, layout=None, checks=None, loaded=None, prompt=33, pulls=None) -> dict:
        layout = layout or packed_layout()
        checks = checks or [packed_check("a"), packed_check("b")]
        loaded = loaded if loaded is not None else 2 * 3 * 20
        decode = (layout + (pulls or packed_pull("a") + packed_pull("b", offset=300))
                  + "".join(stepped(r, "d", prompt - 1, 1, prompt) for r in ("a", "b"))
                  + line(kv_transfers={"load": {"count": 2, "bytes": loaded, "failed": 0}})
                  + "".join(line(kv_check=c) for c in checks))
        prefill = layout + allocated("a", "b") + "".join(stepped(r, "p", 0, prompt - 1, prompt - 1) for r in ("a", "b"))
        return pd.check({"completed": 2, "errors": []}, [prefill], [decode], {"control": 1, "data0": loaded + 100},
                        NIXL)

    def test_checked_pages_and_moved_blocks_reconcile(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["kv_bytes_checked"], 2 * 3 * 12)  # real pages only
        self.assertEqual(verdict["ranks"][0]["kv_bytes_spanned"], 2 * 3 * 20)  # whole blocks
        self.assertIsNotNone(verdict["ranks"][0]["handoff"])  # N - 1 handed over

    def test_each_packed_failure_is_reported(self) -> None:
        malformed = packed_group("MLAAttentionSpec", None, transfer_bytes_per_block=None)
        cases = {
            # Moving checked bytes only is not the connector's whole blocks.
            "moved 72 bytes, not the 120 decode's checked blocks span (72 checked)": {"loaded": 72},
            "check disagrees with the layout (2 layers, 12 bytes checked a block)": {
                "checks": [packed_check("a"), {**packed_check("b"), "bytes": 30, "groups": [
                    {**packed_check("b")["groups"][0], "bytes": 6}, packed_check("b")["groups"][1]]}]},
            "group 1 has a malformed packed layout": {"layout": packed_layout(full=malformed)},
            "group 0 checks 10 bytes a block, not its members' 12 real bytes": {
                "layout": packed_layout(window=packed_group("SlidingWindowMLASpec", 16, checked_bytes_per_block=10))},
            "group 1 packs 1 layers, not its 2": {"layout": packed_layout(full=packed_group(
                "MLAAttentionSpec", None,
                members=[{**packed_group("x", None)["members"][0], "layers": 1, "offsets": [0]}]))},
            "prefill_backoff is 2, not 1": {"layout": packed_layout(prefill_backoff=2)},
            # The latents hold the 32 tokens handed over, not all 33.
            "checked 3 blocks, not the cdiv(32, 16)": {
                "checks": [packed_check("a", full=(0, 3)), packed_check("b")],
                "pulls": packed_pull("a", full=(0, 3)) + packed_pull("b", offset=300), "loaded": 140},
        }
        for message, given in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(**given)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)

    def test_malformed_member_placements_are_reported(self) -> None:
        member = packed_group("x", None)["members"][0]

        def with_members(*members, **changes) -> str:
            return packed_layout(full=packed_group("MLAAttentionSpec", None, members=list(members), **changes))

        stripped = {key: value for key, value in member.items()
                    if key not in ("offsets", "page_bytes", "kind", "dtype", "tokens_per_state")}
        cases = {
            "group 1 has a malformed packed member 0 (needs": with_members(stripped),
            "group 1 has a malformed packed member 0 (": with_members({**member, "offsets": [0]}),
            "member 0's page at 0 overlaps member 0's": with_members({**member, "offsets": [0, 0]}),
            "member 0's page at 1000 ends beyond the 20-byte block": with_members({**member, "offsets": [0, 1000]}),
            "member 0 has 9 real bytes in a 8-byte page": with_members(
                {**member, "real_page_bytes": 9}, checked_bytes_per_block=18),
            "group 1 has a malformed packed layout": with_members(member, checked_bytes_per_block=None),
            "moves 16 bytes a block, less than its 20-byte block": with_members(
                member, transfer_bytes_per_block=16),
        }
        for message, layout in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(layout=layout)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)

    def test_a_single_token_prompt_is_handed_over_whole(self) -> None:
        # One token: nothing is backed off; its latents' block is pulled.
        checks = [packed_check(r, window=(0, 1), full=(0, 1)) for r in ("a", "b")]
        pulls = packed_pull("a", (0, 1), (0, 1)) + packed_pull("b", (0, 1), (0, 1), offset=300)
        decode = (packed_layout() + pulls + "".join(stepped(r, "d", 0, 1, 1) for r in ("a", "b"))
                  + line(kv_transfers={"load": {"count": 2, "bytes": 80, "failed": 0}})
                  + "".join(line(kv_check=c) for c in checks))
        prefill = packed_layout() + allocated("a", "b") + "".join(stepped(r, "p", 0, 1, 1) for r in ("a", "b"))
        short = pd.Run(**{**NIXL.__dict__, "workload": pd.Workload(num_prompts=2, input_len=1)})
        verdict = pd.check({"completed": 2, "errors": []}, [prefill], [decode], {"control": 1, "data0": 180},
                           short)
        self.assertEqual(verdict["problems"], [])


NIXL = pd.Run(**{**RUN.__dict__, "server": pd.Server(connector="nixl")})


class NixlCheckTests(unittest.TestCase):
    """NIXL's decode reads the KV: its connector's loads and TENT's reads
    count the bytes, prefill moves none."""

    def verdict(self, loads=None, decode_tent=None, routers=None, run=NIXL) -> dict:
        loads = loads or {"count": 2, "bytes": 2000, "failed": 0}
        decode = ONE_RANK + line(kv_transfers={"load": loads}) + line(kv_check=checked("a")) + line(
            kv_check=checked("b"))
        tents = {"prefill": [{"tent_registered_buffer_bytes": 1600}],
                 "decode": [decode_tent if decode_tent is not None else {
                     "tent_read_bytes_total": 2000, "tent_registered_buffer_bytes": 1600}]}
        return pd.check({"completed": 2, "errors": []}, [ONE_RANK + allocated("a", "b")], [decode],
                        routers or {"control": 50, "data0": 2100}, run, tents)

    def test_decode_reads_the_bytes_it_checks(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["transfers"], {"count": 2, "bytes": 2000, "failed": 0})

    def test_each_failure_is_reported(self) -> None:
        cases = {
            "decode's connector failed 1 transfer": {"loads": {"count": 2, "bytes": 2000, "failed": 1}},
            "moved 1000 bytes, not the 2000 decode checked": {
                "loads": {"count": 1, "bytes": 1000, "failed": 0}, "routers": {"control": 50, "data0": 5000}},
            "TENT read 1500 bytes, not the 2000": {
                "decode_tent": {"tent_read_bytes_total": 1500, "tent_registered_buffer_bytes": 1600}},
            # Unavailable telemetry is a problem, not zero.
            "decode's TENT metrics lack tent_read_bytes_total": {
                "decode_tent": {"tent_registered_buffer_bytes": 1600}},
            "fewer than the 2000 its ranks moved": {"routers": {"control": 50, "data0": 100}},
        }
        for message, given in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(**given)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)
        # Mooncake's decode needs no reads; its prefill needs its writes.
        problems = self.verdict(run=RUN)["problems"]
        self.assertTrue(any("prefill's TENT metrics lack tent_write_bytes_total" in p for p in problems), problems)


class SpeculativeCheckTests(unittest.TestCase):
    """Native MTP over NIXL: both instances take it, decode schedules drafts,
    every request returns exactly its output length."""

    SPEC = {"method": "mtp", "num_speculative_tokens": 3, "acceptance_rates": [1.0], "seed": 7}

    def run_with(self, **spec) -> pd.Run:
        return pd.Run(**{**NIXL.__dict__, "server": pd.Server(connector="nixl", speculative={**self.SPEC, **spec})})

    def test_both_instances_take_the_speculative_config(self) -> None:
        for role in ("prefill", "decode"):
            command = pd.server_command(self.run_with(), role, ["10.100.0.1"])
            self.assertEqual(json.loads(command[command.index("--speculative-config") + 1]),
                             {"method": "mtp", "num_speculative_tokens": 3})
            self.assertEqual(command[command.index("--seed") + 1], "7")
            simulation = json.loads(command[command.index("--simulation-config") + 1])
            self.assertEqual(simulation["spec_acceptance_rates"], [1.0])
        plain = pd.server_command(NIXL, "decode", ["10.100.0.1"])
        self.assertNotIn("--speculative-config", plain)
        self.assertNotIn("spec_acceptance_rates", json.loads(plain[plain.index("--simulation-config") + 1]))

    def test_only_native_mtp_with_valid_rates(self) -> None:
        """Over either connector: the simulator decides which models' MTP
        Mooncake moves (DeepSeek-V3, V3.2, GLM-5)."""
        for connector in ("nixl", "mooncake"):
            pd._check_speculative(pd.Server(connector=connector, speculative=self.SPEC))
        for spec, connector, message in (
            ({"method": "ngram"}, "nixl", "native MTP"),
            ({"num_speculative_tokens": 6}, "nixl", "native MTP"),
            ({"acceptance_rates": []}, "nixl", "1 to 3 probabilities"),
            ({"acceptance_rates": [1.5]}, "nixl", "1 to 3 probabilities"),
            ({"acceptance_rates": [0.5] * 4}, "nixl", "1 to 3 probabilities"),
            ({"seed": "x"}, "nixl", "seed is an integer"),
            ({"draft": "x"}, "nixl", "server.speculative takes"),
            ({"num_speculative_tokens": 6}, "mooncake", "native MTP"),
        ):
            with self.subTest(spec=spec, connector=connector):
                with self.assertRaisesRegex(ValueError, message):
                    pd._check_speculative(pd.Server(connector=connector, speculative={**self.SPEC, **spec}))

    def verdict(self, steps, lengths=(16, 16), extra=(), owner=0, **spec) -> dict:
        """Decode's steps: (drafts, the step's own counts, the rank's logged
        (drafts, accepted) pairs; "same": the counts' pairs)."""
        loads = {"count": 2, "bytes": 2000, "failed": 0}
        decode = ONE_RANK + line(kv_transfers={"load": loads}) + line(kv_check=checked("a")) + line(
            kv_check=checked("b"))
        for i, (drafts, counts, logged) in enumerate(steps):
            requests = [f"cmpl-{r}-0-d" for r in "ab"[:len(drafts)]]
            decode += line(step={"step_index": i, "requests": [
                {"req_id": r, "num_computed_tokens": 100 + i, "num_scheduled_tokens": 1 + d,
                 "num_prompt_tokens": 100, "num_spec_tokens": d, "dp_rank": owner}
                for r, d in zip(requests, drafts)]},
                cost={"num_accepted": counts})
            if logged == "same":
                logged = [[d, a] for d, a in zip(drafts, counts)]
            pairs = {r: pair for r, pair, d in zip(requests, logged or [], drafts) if d}
            if pairs:
                decode += line(spec_acceptance={"step_index": i, "requests": pairs})
        decode += "".join(line(spec_acceptance=record) for record in extra)
        tents = {"prefill": [{"tent_registered_buffer_bytes": 1600}],
                 "decode": [{"tent_read_bytes_total": 2000, "tent_registered_buffer_bytes": 1600}]}
        return pd.check({"completed": 2, "errors": [], "output_lens": list(lengths)},
                        [ONE_RANK + allocated("a", "b")], [decode], {"control": 50, "data0": 2100},
                        self.run_with(**spec), tents)

    def test_drafts_are_scheduled_and_accepted_as_the_rates_say(self) -> None:
        verdict = self.verdict([([0, 0], [0, 0], None), ([3, 3], [3, 3], "same"), ([3, 1], [3, 1], "same")])
        self.assertEqual(verdict["problems"], [])
        self.assertEqual({k: verdict["speculation"][k] for k in ("steps_with_drafts", "drafted", "accepted")},
                         {"steps_with_drafts": 2, "drafted": 10, "accepted": 10})
        self.assertEqual(verdict["speculation"]["seed"], 7)
        # Rates all 0: none accepted. A joint MoE-DP step has no counts of its
        # own; its rank's acceptance records cover it.
        verdict = self.verdict([([3, 3], [0, 0], "same"), ([3], None, [[3, 0]])], acceptance_rates=[0.0])
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["speculation"]["drafts_logged"], 9)

    def test_each_failure_is_reported(self) -> None:
        cases = {
            "decode scheduled no drafts": {"steps": [([0, 0], [0, 0], None)]},
            "5 of 6 drafts accepted at acceptance rates 1": {"steps": [([3, 3], [3, 2], "same")]},
            "2 drafts accepted at acceptance rates 0": {"steps": [([3, 3], [2, 0], "same")],
                                                       "acceptance_rates": [0.0]},
            "accepted [2, 1] of drafts [1, 3]": {"steps": [([1, 3], [2, 1], [[1, 1], [3, 1]])],
                                                 "acceptance_rates": [0.5]},
            "output lengths [15, 16] of 2 request(s), not 16 each": {"steps": [([3, 3], [3, 3], "same")],
                                                                    "lengths": (16, 15)},
            # Unobserved acceptance does not pass, even at rates all 1.
            "2 drafted request step(s) lack their owner's acceptance": {"steps": [([3, 3], None, None)]},
            "1 drafted request step(s) lack their owner's acceptance": {"steps": [([3, 3], None, [[3, 3]])]},
            # Each record matches its own step: the step's drafts and counts.
            "request cmpl-a-0-d logged [3, 3], the step drafted 3 and accepted 0": {
                "steps": [([3, 3], [0, 0], [[3, 3], [3, 3]])]},
            "step 0: request cmpl-a-0-d logged [3, 3], the step drafted 1": {
                "steps": [([1, 1], None, [[3, 3], [3, 3]]), ([3, 3], None, [[1, 1], [1, 1]])]},
            "request cmpl-a-0-d logged twice": {
                "steps": [([3, 3], None, [[3, 3], [3, 3]])],
                "extra": [{"step_index": 0, "requests": {"cmpl-a-0-d": [3, 3]}}]},
            "1 acceptance record(s) for no drafted request step of their DP rank, e.g. DP rank 0 step 7": {
                "steps": [([3, 3], None, [[3, 3], [3, 3]])],
                "extra": [{"step_index": 7, "requests": {"cmpl-a-0-d": [3, 3]}}]},
            # A step's counts: one per request, nonnegative.
            "accepted [] of drafts [3, 3]": {"steps": [([3, 3], [], [[3, 1], [3, 1]])], "acceptance_rates": [0.5]},
            "accepted [-1, 1] of drafts [3, 3]": {"steps": [([3, 3], [-1, 1], [[3, 0], [3, 0]])],
                                                  "acceptance_rates": [0.0]},
            "request cmpl-a-0-d accepted [3, -1]": {"steps": [([3, 3], None, [[3, -1], [3, 1]])],
                                                    "acceptance_rates": [0.0]},
            "request cmpl-b-0-d accepted [3, 4]": {"steps": [([3, 3], None, [[3, 3], [3, 4]])]},
        }
        for message, given in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(**given)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)
        # A record in another DP rank's log than the request's owner.
        problems = self.verdict([([3, 3], None, [[3, 3], [3, 3]])], owner=1)["problems"]
        for message in ("2 drafted request step(s) lack their owner's acceptance (spec_acceptance), e.g. DP rank 1",
                        "2 acceptance record(s) for no drafted request step of their DP rank, e.g. DP rank 0"):
            self.assertTrue(any(message in p for p in problems), problems)
        # A rates-1 run with a request's acceptance missing does not pass on
        # the drafts that were logged.
        problems = self.verdict([([3, 3], None, [[3, 3]])])["problems"]
        self.assertTrue(any("3 of 6 drafts accepted at acceptance rates 1" in p for p in problems), problems)

@unittest.skipUnless(os.environ.get("E2E_VLLM") == "1", "set E2E_VLLM=1 to run vLLM in docker")
class PrefillDecodeRunTests(unittest.TestCase):
    def test_a_byte_checked_run_keeps_its_kv_on_the_data_rail(self) -> None:
        parsed = spec.load(E2E / "topologies" / "example.yaml")
        topology.down(parsed.name)
        topology.up(parsed)
        try:
            verdict = pd.execute(pd.load(E2E / "runs" / "example.yaml"))
        finally:
            topology.down(parsed.name)
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["kv_checks"], 40)
        # The other rail stayed idle.
        self.assertLess(verdict["router_bytes"]["data1"], 1 << 20)
        # Routed by llm-d, checked for real: every request's route, the
        # components where their nodes are, nothing left behind.
        routed = verdict["llmd"]
        self.assertEqual((routed["completion_requests"], routed["producer_released"], routed["consumer_received"]),
                         (40, 40, 40))
        self.assertEqual(routed["pairs"], {"prefill0->decode0": 40})
        self.assertEqual(routed["cleanup"]["remaining"], [])
        self.assertEqual(len(routed["cleanup"]["removed"]), 3)
        placed = routed["components"]
        self.assertEqual({name.rsplit("-", 1)[1]: (c["node"], c["cpuset"]) for name, c in placed.items()},
                         {"envoy": ("example-gateway", "15"), "epp": ("example-gateway", "15"),
                          "sidecar": ("example-decode0", "7-13")})
        self.assertTrue(all(c["cpuset"] == c["node_cpuset"] for c in placed.values()))


class RoutingCheckScopeTests(unittest.TestCase):
    """The router's checks are stubbed for synthetic fixtures only: a stub
    reaching the docker run would let it pass without its routing checked."""

    def test_the_docker_run_checks_its_routing_for_real(self) -> None:
        self.assertFalse(issubclass(PrefillDecodeRunTests, SyntheticRouting))
        self.assertFalse(hasattr(sys.modules[__name__], "setUpModule"))
        self.assertNotIsInstance(llmd.recheck, mock.Mock)
        # Whatever ran before it, in this module or another.
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecheckTests)
        unittest.TextTestRunner(stream=io.StringIO()).run(suite)
        self.assertNotIsInstance(llmd.recheck, mock.Mock)

    def test_a_run_without_router_evidence_fails_unstubbed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            RecheckTests().save(Path(directory), [grouped("a"), grouped("b")])
            problems = pd.recheck(directory)["problems"]
        self.assertTrue(any("missing routing evidence" in p for p in problems), problems)



class NativeMtpStateTests(unittest.TestCase):
    """Real decode records of Qwen3.5's native MTP over NIXL (two processes,
    MTP depths 1 and 3): the clipped read's metadata gives a state's running
    slot (physically before its trailing scratch slots) index k, the depth;
    the checks agree with the layout there."""

    DATA = Path(__file__).parent / "data"

    def records(self, k: int) -> list[dict]:
        path = self.DATA / f"nixl-qwen3.5-mtp-k{k}-decode-steps.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def problems(self, records: list[dict]) -> list[str]:
        layout = next(r["kv_layout"] for r in records if "kv_layout" in r)
        checks = [r["kv_check"] for r in records if "kv_check" in r]
        return pd._byte_check_problems(layout, checks, 200, pd._pull_plans(records),
                                       {c["req_id"]: 199 for c in checks})

    def test_states_start_at_the_mtp_depth(self) -> None:
        for k in (1, 3):
            with self.subTest(k=k):
                records = self.records(k)
                layout = next(r["kv_layout"] for r in records if "kv_layout" in r)
                self.assertEqual([g.get("speculative_blocks") for g in layout["groups"]], [k, k, None])
                self.assertEqual(self.problems(records), [])

    def test_other_starts_are_reported(self) -> None:
        for k in (1, 3):
            records = self.records(k)
            check = next(r["kv_check"] for r in records if "kv_check" in r)
            # A state's check placed at index 0, where the clipped read's
            # metadata gives k.
            check["groups"][0]["first_logical_block"] = 0
            # Attention still starts at 0: not moved by the depth.
            check["groups"][2]["first_logical_block"] = k
            problems = self.problems(records)
            with self.subTest(k=k):
                self.assertTrue(any(f"group 0 (MambaSpec) was pulled from block 0, not {k}" in p
                                    for p in problems), problems)
                self.assertTrue(any(f"group 2 (FullAttentionSpec) was pulled from block {k}, not 0" in p
                                    for p in problems), problems)



class SparseMlaLayoutTests(unittest.TestCase):
    """A real decode record of a tiny DeepSeek-V3.2 over NIXL (two
    processes): each layer's BF16 latent and indexer keys, pages of two
    sizes in one group, each layer its own region; a check's bytes are its
    blocks of every layer's page."""

    PATH = Path(__file__).parent / "data" / "nixl-deepseek-v3.2-decode-steps.jsonl"

    def problems(self, change=None) -> list[str]:
        records = [json.loads(line) for line in self.PATH.read_text().splitlines()]
        layout = next(r["kv_layout"] for r in records if "kv_layout" in r)
        checks = [r["kv_check"] for r in records if "kv_check" in r]
        if change:
            change(layout["groups"][0], checks[0])
        return pd._byte_check_problems(layout, checks, 200, pd._pull_plans(records),
                                       {c["req_id"]: 200 for c in checks})

    def test_a_sparse_mla_check_counts_each_layers_page(self) -> None:
        self.assertEqual(self.problems(), [])

    def test_its_failures_are_reported(self) -> None:
        def kinds_short(group, check):
            group["kinds"][0]["layers"] = 1

        def checked_wrong(group, check):
            group["checked_bytes_per_block"] += 1

        def latent_pages_only(group, check):
            # Every layer counted at the latent's page.
            check["groups"][0]["bytes"] = check["groups"][0]["blocks"] * 4 * 73728

        cases = {
            "has kinds of 3 layers, not its 4": kinds_short,
            "checks 164353 bytes a block, not its kinds' pages": checked_wrong,
            "check disagrees with the layout": latent_pages_only,
        }
        for message, change in cases.items():
            with self.subTest(message=message):
                problems = self.problems(change)
                self.assertTrue(any(message in p for p in problems), problems)


# ----- a hybrid's unequal TP: attention head-v1 beside a Mamba2 state group (state-v1) -----

STATE = {"family": "mamba2", "components": [["x", 16], ["B", 4], ["C", 4]], "conv_rows": 3, "conv_itemsize": 2,
         "ssm_heads": 4, "ssm_head_bytes": 8}
ATTN_HEAD = 10  # bytes of one KV head over a request's blocks of the attention group


def _hybrid_expectations(prefill_tp: int, decode_tp: int, kv_heads: int, only_regions: int = 1, logical_regions: int = 1):
    """Per decode rank and producer: attention heads, the state's classes
    (component -> bytes of a request's one block), written here from the
    ownership rule (a producer's range overlap) for the fixture alone."""
    out = {}
    for d in range(decode_tp):
        cands = ([d * prefill_tp // decode_tp] if prefill_tp <= decode_tp
                 else list(range(d * (prefill_tp // decode_tp), (d + 1) * (prefill_tp // decode_tp))))
        for p in cands:
            heads = [h for h in pd.global_heads(d, decode_tp, kv_heads)
                     if h in pd.global_heads(p, prefill_tp, kv_heads)
                     and p == next(q for q in cands if h in pd.global_heads(q, prefill_tp, kv_heads))]
            classes = {}
            for kind, name, width, unit in ([("conv", n, w, 6) for n, w in STATE["components"]]
                                            + [("ssm", "ssm", STATE["ssm_heads"], STATE["ssm_head_bytes"])]):
                lo, hi = d * width // decode_tp, (d + 1) * width // decode_tp
                a, b = p * width // prefill_tp, (p + 1) * width // prefill_tp
                if max(lo, a) < min(hi, b):
                    n = (min(hi, b) - max(lo, a)) * unit
                    for tag, regs in (("logical", logical_regions), ("transfer-only", only_regions)):
                        if regs:
                            classes[f"g1.{kind}.{name}.{tag}"] = n * regs
            if heads or classes:
                out[(d, p)] = (heads, classes)
    return out


def hybrid_logs(prefill_tp, decode_tp, kv_heads=4, requests=("a", "b"), only_regions=1):
    """Logs of a hybrid's unequal-TP run over NIXL consistent with ownership:
    each READ's bytes, its class table, descriptors and length table."""
    expect = _hybrid_expectations(prefill_tp, decode_tp, kv_heads, only_regions)
    tps = (("prefill", prefill_tp), ("decode", decode_tp))

    def layout(rank, tp):
        spans = pd._state_spans(STATE, rank, tp)
        heads = pd.global_heads(rank, tp, kv_heads)
        return line(kv_layout={"tp_rank": rank, "tp_size": tp, "block_bytes": 20, "bytes": 1600, "groups": [
            {"kind": "FullAttentionSpec", "block_size": 16, "layers": 2, "block_bytes": ATTN_HEAD * len(heads),
             "real_page_bytes": ATTN_HEAD * len(heads), "pattern": "head-v1", "global_heads": heads},
            {"kind": "MambaSpec", "block_size": 16, "layers": 2, "pattern": "state-v1", "global_heads": None,
             "state": {**STATE, "spans": spans}}]})

    loads = {d: [] for d in range(decode_tp)}
    checks_of = {d: [] for d in range(decode_tp)}
    sent = {p: 0 for p in range(prefill_tp)}
    received = {d: 0 for d in range(decode_tp)}
    for d in range(decode_tp):
        spans = pd._state_spans(STATE, d, decode_tp)
        span_bytes = sum(sp[3] for sp in spans)
        mine = pd.global_heads(d, decode_tp, kv_heads)
        for r in requests:
            reads = []
            for (dd, p), (heads, classes) in sorted(expect.items()):
                if dd != d:
                    continue
                total = ATTN_HEAD * len(heads) + sum(classes.values())
                a = total // 2
                table = {a: 1, total - a: 1} if a != total - a else {a: 2}
                classes_log = {k: [1, v] for k, v in classes.items()}
                if heads:
                    classes_log["g0.attention.logical"] = [1, ATTN_HEAD * len(heads)]
                reads.append({"rank": p, "completed": True, "descriptors": 2, "bytes": total, "classes": classes_log})
                loads[d].append([total, 0.01, 0.0, 2, 10.0, p, [0, 2, 2, total, table]])
                sent[p] += total
                received[d] += total
            state_bytes = 1 * (1 + only_regions) * span_bytes
            checks_of[d].append(checked(
                r, blocks=6, layers=4, bytes=ATTN_HEAD * len(mine) + state_bytes, posted_matches_checked=True,
                coverage={"problems": [], "reads": reads},
                groups=[{"group": 0, "blocks": 5, "bytes": ATTN_HEAD * len(mine), "mismatched_bytes": 0,
                         "pattern": "head-v1", "global_heads": mine},
                        {"group": 1, "blocks": 1, "regions": 1 + only_regions, "bytes": state_bytes,
                         "transfer_only_bytes": only_regions * span_bytes, "mismatched_bytes": 0,
                         "pattern": "state-v1", "global_heads": None}]))
    prefill = [layout(p, prefill_tp) + allocated(*requests) for p in range(prefill_tp)]
    decode = [layout(d, decode_tp) + "".join(stepped(r, "d", 80, 1, 80) for r in requests)
              + line(kv_transfers={"load": {"count": len(loads[d]), "bytes": sum(e[0] for e in loads[d]),
                                            "each": loads[d]}})
              + "".join(line(kv_check=c) for c in checks_of[d]) for d in range(decode_tp)]
    ports = {"kv": {**{f"p{p}": {"node": "p", "slot": p, "rx": sent[p] + 100, "tx": 40} for p in range(prefill_tp)},
                    **{f"p{prefill_tp + d}": {"node": "d", "slot": d, "rx": 40, "tx": received[d] + 100}
                       for d in range(decode_tp)}}}
    return prefill, decode, ports


class HybridHeterogeneousTpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.model = Path(self.directory.name)
        (self.model / "config.json").write_text(json.dumps({"num_key_value_heads": 4}))

    def tearDown(self) -> None:
        self.directory.cleanup()

    def reconcile(self, prefill_tp, decode_tp, change=None, nixl=True, kv_heads=4, **kw) -> dict:
        prefill, decode, ports = hybrid_logs(prefill_tp, decode_tp, kv_heads, **kw)
        if change:
            prefill, decode, ports = change(prefill, decode, ports)
        run = pd.Run(**{**RUN.__dict__, "model": self.model,
                        "prefill": pd.Side("p", "kv", tensor_parallel_size=prefill_tp),
                        "decode": pd.Side("d", "kv", tensor_parallel_size=decode_tp),
                        "server": pd.Server(connector="nixl" if nixl else "mooncake")})
        logs = {"prefill": {(0, r): pd._records(text) for r, text in enumerate(prefill)},
                "decode": {(0, r): pd._records(text) for r, text in enumerate(decode)}}
        moved = lambda records: sum(((r.get("kv_transfers") or {}).get("load") or {}).get("bytes", 0) for r in records)
        ranks = {role: {key: {"transfers": {"bytes": moved(records)}, "requests": ["a", "b"]}
                        for key, records in logs[role].items()} for role in logs}
        return pd._hetero_tp(run, {"prefill": prefill_tp, "decode": decode_tp}, logs, ranks, ports)

    def test_state_and_attention_pair_every_direction(self) -> None:
        for sizes in ((1, 2), (2, 1), (2, 4), (4, 2), (1, 4), (4, 1)):
            with self.subTest(sizes=sizes):
                hetero = self.reconcile(*sizes)
                self.assertEqual(hetero["problems"], [], sizes)
                self.assertTrue(hetero["hybrid"])
        # TP2 <- TP4 (K=4): decode 0 takes heads 0,1 and half its state from 0 and 1.
        self.assertEqual(self.reconcile(4, 2)["sources"], {0: [0, 1], 1: [2, 3]})
        self.assertEqual(self.reconcile(2, 4)["sources"], {0: [0], 1: [0], 2: [1], 3: [1]})
        classes = self.reconcile(2, 1)["byte_classes"]
        self.assertGreater(classes["state_logical"], 0)
        self.assertEqual(classes["state_transfer_only"], classes["state_logical"])

    def test_transfer_only_regions_are_apart_from_logical_state(self) -> None:
        none = self.reconcile(1, 2, only_regions=0)
        self.assertEqual(none["problems"], [])
        self.assertEqual(none["byte_classes"]["state_transfer_only"], 0)
        two = self.reconcile(1, 2, only_regions=2)
        self.assertEqual(two["problems"], [])
        self.assertEqual(two["byte_classes"]["state_transfer_only"], 2 * two["byte_classes"]["state_logical"])

    def test_mooncake_does_not_pair_a_hybrid(self) -> None:
        problems = self.reconcile(1, 2, nixl=False)["problems"]
        self.assertTrue(any("NIXL only" in p for p in problems), problems)

    def mutate(self, text, find, replace, count=1):
        self.assertIn(find, text)
        return text.replace(find, replace, count)

    def test_state_spans_must_be_the_models_layout(self) -> None:
        # A rank's span offsets or lengths other than the layout's.
        def spans(prefill, decode, ports):
            return prefill, [self.mutate(decode[0], '"spans": [["conv", "x", 0, 48', '"spans": [["conv", "x", 0, 54'),
                             *decode[1:]], ports
        problems = self.reconcile(1, 2, change=spans)["problems"]
        self.assertTrue(any("state spans" in p for p in problems), problems)
        # Another layout on one side.
        def other(prefill, decode, ports):
            return [self.mutate(prefill[0], '"ssm_heads": 4', '"ssm_heads": 8'), *prefill[1:]], decode, ports
        self.assertTrue(any("differs from" in p or "state spans" in p for p in self.reconcile(1, 2, change=other)["problems"]))

    def test_the_posted_read_record_is_required_and_must_be_clean(self) -> None:
        def strip(prefill, decode, ports):
            return prefill, [self.mutate(decode[0], '"coverage": {"problems": [], "reads"', '"nocoverage": {"reads"'),
                             *decode[1:]], ports
        self.assertTrue(any("no READ coverage" in p for p in self.reconcile(1, 2, change=strip)["problems"]))
        def flagged(prefill, decode, ports):
            return prefill, [self.mutate(decode[0], '"coverage": {"problems": []',
                                         '"coverage": {"problems": ["a shard twice"]'), *decode[1:]], ports
        self.assertTrue(any("a shard twice" in p for p in self.reconcile(1, 2, change=flagged)["problems"]))
        def unfinished(prefill, decode, ports):
            return prefill, [self.mutate(decode[0], '"completed": true', '"completed": false'), *decode[1:]], ports
        self.assertTrue(any("not complete" in p for p in self.reconcile(1, 2, change=unfinished)["problems"]))
        def unmatched(prefill, decode, ports):
            return prefill, [self.mutate(decode[0], '"posted_matches_checked": true', '"posted_matches_checked": false'),
                             *decode[1:]], ports
        self.assertTrue(any("did not match its posted READs" in p for p in self.reconcile(1, 2, change=unmatched)["problems"]))

    def test_a_reads_bytes_lengths_and_counts_are_exact(self) -> None:
        def bytes_off(prefill, decode, ports):  # a READ's bytes differ from its selected descriptors' sum
            return prefill, [re.sub(r'\[(\d+), 0.01, 0.0, 2, 10.0, 0, \[0, 2, 2, (\d+)', lambda m: f"[{int(m.group(1)) + 8}, 0.01, 0.0, 2, 10.0, 0, [0, 2, 2, {m.group(2)}",
                                    decode[0], count=1), *decode[1:]], ports
        self.assertTrue(any("selected descriptors sum" in p for p in self.reconcile(1, 2, change=bytes_off)["problems"]))
        def merged(prefill, decode, ports):  # predicted != measured
            return prefill, [self.mutate(decode[0], ", 2, 10.0, 0, [0, 2, 2,", ", 2, 10.0, 0, [0, 2, 1,"), *decode[1:]], ports
        self.assertTrue(any("predicted" in p for p in self.reconcile(1, 2, change=merged)["problems"]))
        # A merged READ (predicted = measured < selected) is fine.
        def success(prefill, decode, ports):
            text = self.mutate(decode[0], ", 2, 10.0, 0, [0, 2, 2,", ", 1, 10.0, 0, [0, 2, 1,")
            return prefill, [text, *decode[1:]], ports
        self.assertEqual(self.reconcile(1, 2, change=success)["problems"], [])

    def test_a_swapped_source_or_wrong_class_bytes_are_rejected(self) -> None:
        def swapped(prefill, decode, ports):
            text = decode[0].replace('"rank": 0', '"rank": 9', 1)
            return prefill, [text, *decode[1:]], ports
        self.assertTrue(self.reconcile(1, 2, change=swapped)["problems"])
        def wrong_class(prefill, decode, ports):
            return prefill, [self.mutate(decode[0], '"g1.conv.x.logical": [1, 48]', '"g1.conv.x.logical": [1, 42]'),
                             *decode[1:]], ports
        self.assertTrue(self.reconcile(1, 2, change=wrong_class)["problems"])


if __name__ == "__main__":
    unittest.main()


# Prefix caching: hits checked on both instances, pulls past decode's hit.
PREFIX = pd.Run(**{**RUN.__dict__, "server": pd.Server(enable_prefix_caching=True)})
FULL_LAYOUT = line(kv_layout={
    "tp_rank": 0, "tp_size": 1, "block_bytes": 8, "bytes": 1600,
    "groups": [{"kind": "FullAttentionSpec", "sliding_window": None, "block_size": 16, "layers": 2,
                "block_bytes": 4}]})


def admit(request: str, side: str, computed: int, blocks: list[int]) -> str:
    """A request admitted (new or resumed) with `computed` tokens and its
    blocks then."""
    rid = f"cmpl-{request}-0-{side}"
    return line(kv_events={"allocated": {rid: [b for b in blocks if b]}, "admitted": {rid: computed},
                           "allocated_groups": {rid: [{"group": 0, "first": 0, "blocks": blocks}]}})


def post(request: str, first: int, blocks: list[int]) -> str:
    """Decode's pull of a request's blocks from place `first`, posted."""
    rid = f"cmpl-{request}-0-d"
    return line(kv_events={"allocated": {rid: blocks}, "reads_posted": [rid],
                           "allocated_groups": {rid: [{"group": 0, "first": first, "blocks": blocks}]}})


def hit(request: str, side: str, blocks: int, ok: bool = True, first_block: int | None = None) -> dict:
    """A hit check of a request's first `blocks` places: prefill's blocks
    from 1, decode's from 101, unless `first_block`."""
    first_block = (1 if side == "p" else 101) if first_block is None else first_block
    return {"req_id": f"cmpl-{request}-0-{side}", "kind": "hit", "blocks": blocks, "bytes": blocks * 8,
            "groups": [{"group": 0, "logical_blocks": list(range(blocks)),
                        "block_ids": list(range(first_block, first_block + blocks)), "blocks": blocks,
                        "bytes": blocks * 8, "mismatched_bytes": 0 if ok else 3, "written_this_step": []}],
            "mismatched_bytes": 0 if ok else 3, "ok": ok}


def blocks(first: int, count: int) -> list[int]:
    return list(range(first, first + count))


# b's 1024-token prompt shares a's first 48 blocks (of 64): prefill admits b
# with them cached; decode too, and pulls b's last 16.
PREFILL_ADMISSIONS = admit("a", "p", 0, blocks(1, 64)) + admit("b", "p", 768, blocks(1, 48) + blocks(201, 16))
DECODE_ADMISSIONS = (post("a", 0, blocks(101, 64)) + post("b", 48, blocks(301, 16))
                     + admit("a", "d", 1023, blocks(101, 64)) + admit("b", "d", 1023, blocks(101, 48) + blocks(301, 16)))
A_CHECK = checked("a", blocks=64, layers=2, bytes=64 * 8, groups=[group_check(0, 0, 64)])
B_CHECK = checked("b", blocks=16, layers=2, bytes=16 * 8, groups=[group_check(0, 48, 16)])


class WaitTests(unittest.TestCase):
    def test_a_server_whose_engine_failed_is_reported_at_once(self) -> None:
        """A vLLM server that dies at startup never answers: its log's
        failure line ends the wait, instead of the whole timeout."""
        calls = []

        def exec_in(container, command, **kwargs):
            calls.append((container, command[0]))
            if command[0] == "grep":
                self.assertEqual((container, command[-2:]), ("prefill0", [pd.ENGINE_FAILED, "/e2e/logs/prefill.log"]))
                return "1\n"
            return "connection refused\n" if container == "client" else "ValueError: no such layout\n"

        with mock.patch.object(pd.topology, "exec_in", exec_in), mock.patch.object(pd.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "failed to start; prefill0:/e2e/logs/prefill.log ends with:"
                                                      "\nValueError: no such layout"):
                pd._wait("client", "http://10.0.0.1:8000/health", 600, "prefill0", "/e2e/logs/prefill.log")
        sleep.assert_not_called()
        self.assertEqual([c[1] for c in calls if c[0] == "prefill0"], ["grep", "tail"])

    def test_a_server_still_starting_is_waited_for(self) -> None:
        answers = iter(["refused\n", "refused\n", "up\n"])

        def exec_in(container, command, **kwargs):
            return next(answers) if container == "client" else "0\n"

        with mock.patch.object(pd.topology, "exec_in", exec_in), mock.patch.object(pd.time, "sleep") as sleep:
            pd._wait("client", "http://10.0.0.1:8000/health", 600, "prefill0", "/e2e/logs/prefill.log")
        self.assertEqual(sleep.call_count, 2)


class PrefixCachingCheckTests(unittest.TestCase):
    def verdict(self, decode_checks=None, prefill_checks=None, prefill_events="", decode_events="",
                run=PREFIX, moved=(64 + 16) * 8, prefill_admissions=PREFILL_ADMISSIONS,
                decode_admissions=DECODE_ADMISSIONS, decode_steps=None) -> dict:
        decode_checks = [A_CHECK, hit("b", "d", 48), B_CHECK] if decode_checks is None else decode_checks
        prefill_checks = [hit("b", "p", 48)] if prefill_checks is None else prefill_checks
        decode_steps = decode_steps or stepped("a", "d", 1023, 1, 1024) + stepped("b", "d", 1023, 1, 1024)
        prefill = (FULL_LAYOUT + prefill_admissions + prefill_events
                   + "".join(line(kv_check=c) for c in prefill_checks)
                   + line(kv_transfers={"send": {"count": 2, "bytes": moved, "failed": 0}}))
        decode = (FULL_LAYOUT + decode_steps + decode_admissions + decode_events
                  + "".join(line(kv_check=c) for c in decode_checks))
        return pd.check({"completed": run.workload.num_prompts, "errors": []}, [prefill], [decode],
                        {"control": 50, "data0": moved + 100}, run)

    def assertProblem(self, verdict: dict, *messages: str) -> None:
        for message in messages:
            self.assertTrue(any(message in problem for problem in verdict["problems"]), verdict["problems"])

    def test_prefix_routing_reports_where_requests_hit_and_may_expect_hits(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        # Request b hit its prefix on both instances; no prefix groups in this workload, so no ceiling.
        self.assertEqual(verdict["prefix_routing"], {
            "prefill": {"hit_requests": 1, "by_owner": {"p": 1}}, "decode": {"hit_requests": 1, "by_owner": {"d": 1}},
            "ceiling": None, "expected_at_prefill": None})
        grouped_run = pd.Run(**{**PREFIX.__dict__, "workload": pd.Workload(
            num_prompts=2, prefix_repetition={"prefix_len": 768, "suffix_len": 256, "num_prefixes": 1})})
        self.assertEqual(self.verdict(run=grouped_run)["prefix_routing"]["ceiling"], 1)
        met = pd.Run(**{**PREFIX.__dict__, "expect_prefill_hit_requests": 1})
        self.assertEqual(self.verdict(run=met)["problems"], [])
        unmet = pd.Run(**{**PREFIX.__dict__, "expect_prefill_hit_requests": 2})
        verdict = self.verdict(run=unmet)
        self.assertProblem(verdict, "1 request(s) hit a cached prefix at prefill, fewer than the 2 expected")
        self.assertEqual(verdict["prefix_routing"]["expected_at_prefill"], 2)
        # Without prefix caching there is nothing to report.
        self.assertIsNone(pd.check({"completed": 2, "errors": []}, [GROUPS_LAYOUT + allocated("a", "b") + line(
            kv_transfers={"send": {"count": 2, "bytes": 8}})], [GROUPS_LAYOUT], {"control": 1, "data0": 10},
            RUN)["prefix_routing"])

    def test_a_run_may_expect_no_prefix_hit_at_decode(self) -> None:
        """A model whose pulls leave decode nothing to hit (clipped windows, a
        hybrid over Mooncake) says so: no hit is then expected at decode, and
        one found is a problem; prefill's hits are required either way."""
        none = pd.Run(**{**PREFIX.__dict__, "expect_decode_prefix_hits": False})
        # Decode pulls all of b's 64 blocks: no prefix of its own reused.
        pulled_whole = {
            "decode_checks": [A_CHECK, checked("b", blocks=64, layers=2, bytes=64 * 8, groups=[group_check(0, 0, 64)])],
            "decode_admissions": (post("a", 0, blocks(101, 64)) + post("b", 0, blocks(301, 64))
                                  + admit("a", "d", 1023, blocks(101, 64)) + admit("b", "d", 1023, blocks(301, 64))),
            "moved": (64 + 64) * 8,
        }
        self.assertEqual(self.verdict(run=none, **pulled_whole)["problems"], [])
        self.assertProblem(self.verdict(**pulled_whole), "decode showed no checked prefix hit")
        self.assertProblem(self.verdict(run=none),
                           "decode showed 1 prefix hit(s), the run expecting none (expect_decode_prefix_hits: false)")
        self.assertProblem(self.verdict(run=none, prefill_checks=[], **pulled_whole),
                           "prefill showed no checked prefix hit")
        base = {"name": "t", "topology": "t", "model": "org/name", "client": "c", "control_network": "control",
                "router": {"gateway": "gw"}, "prefill": {"node": "p", "kv_network": "data0"},
                "decode": {"node": "d", "kv_network": "data0"},
                "server": {"timing": "zero", "enable_prefix_caching": True}, "workload": {"num_prompts": 10}}
        path = Path("/x/run.yaml")
        self.assertTrue(pd.parse(base, path).expect_decode_prefix_hits)
        self.assertFalse(pd.parse({**base, "expect_decode_prefix_hits": False}, path).expect_decode_prefix_hits)
        for message, change in {
            "true or false": {"expect_decode_prefix_hits": "no"},
            "needs server.enable_prefix_caching and kv_check": {"expect_decode_prefix_hits": False,
                                                                "server": {"timing": "zero"}},
            "needs server.enable_prefix_caching and kv_check ": {"expect_decode_prefix_hits": False, "kv_check": False},
        }.items():
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message.strip()):
                pd.parse({**base, **change}, path)

    def test_the_hit_expectation_and_the_calibration_are_validated(self) -> None:
        base = {"name": "t", "topology": "t", "model": "org/name", "client": "c", "control_network": "control",
                "router": {"gateway": "gw"}, "prefill": {"node": "p", "kv_network": "data0"},
                "decode": {"node": "d", "kv_network": "data0"},
                "server": {"timing": "zero", "enable_prefix_caching": True}, "workload": {"num_prompts": 10}}
        path = Path("/x/run.yaml")
        run = pd.parse({**base, "expect_prefill_hit_requests": 7,
                        "router": {"gateway": "gw", "peak_prefill_throughput": 46002.8}}, path)
        self.assertEqual((run.expect_prefill_hit_requests, run.router.peak_prefill_throughput), (7, 46002.8))
        cases = {
            "a positive integer": {"expect_prefill_hit_requests": 0},
            "a positive integer ": {"expect_prefill_hit_requests": True},
            "a positive integer  ": {"expect_prefill_hit_requests": "7"},
            "at most workload.num_prompts": {"expect_prefill_hit_requests": 11},
            "needs server.enable_prefix_caching": {"expect_prefill_hit_requests": 7, "server": {"timing": "zero"}},
            "needs server.enable_prefix_caching ": {"expect_prefill_hit_requests": 7, "kv_check": False},
            "a positive, finite number of tokens": {"router": {"gateway": "gw", "peak_prefill_throughput": 0}},
            "a positive, finite number of tokens ": {"router": {"gateway": "gw", "peak_prefill_throughput": "fast"}},
            "a positive, finite number of tokens  ": {"router": {"gateway": "gw", "peak_prefill_throughput": -5.0}},
            "a positive, finite number of tokens   ": {"router": {"gateway": "gw",
                                                                   "peak_prefill_throughput": float("nan")}},
            "a positive, finite number of tokens    ": {"router": {"gateway": "gw",
                                                                    "peak_prefill_throughput": float("inf")}},
            "a positive, finite number of tokens     ": {"router": {"gateway": "gw",
                                                                     "peak_prefill_throughput": float("-inf")}},
            "sets it in that file": {"router": {"gateway": "gw", "peak_prefill_throughput": 5.0,
                                                "epp_plugins": "plugins.yaml"}},
        }
        for message, change in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message.strip()):
                pd.parse({**base, **change}, path)
        # As a YAML file gives them.
        for text in (".nan", ".inf", "-.inf"):
            with self.subTest(yaml=text), tempfile.TemporaryDirectory() as directory:
                spec_path = Path(directory) / "run.yaml"
                spec_path.write_text(json.dumps(base).replace('"gateway": "gw"',
                                                              f'"gateway": "gw", "peak_prefill_throughput": {text}'))
                with self.assertRaisesRegex(ValueError, "a positive, finite number of tokens"):
                    pd.load(spec_path)

    def test_a_mooncake_request_write_above_the_task_limit_is_refused(self) -> None:
        """One request's write to one decode rank, estimated at one task a block
        of a region (no block assumed contiguous), is refused above hp_tcp's
        limit: once merged, a write above it is never admitted. Requests
        sharing a write, or writes overlapping, depend on timing: accepted,
        the verdict reports them."""
        from harness import models
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        patch = mock.patch.object(models, "MODELS", root)
        patch.start()
        self.addCleanup(patch.stop)

        def config(**fields) -> None:
            (root / "org--name" / "main").mkdir(parents=True, exist_ok=True)
            (root / "org--name" / "main" / "config.json").write_text(json.dumps(fields))

        def run(input_len: int, decode_tp: int = 2, **server):
            return pd.parse({
                "name": "t", "topology": "t", "model": "org/name", "client": "c", "control_network": "control",
                "router": {"gateway": "gw"}, "prefill": {"node": "p", "kv_network": "kv", "tensor_parallel_size": 2},
                "decode": {"node": "d", "kv_network": "kv", "tensor_parallel_size": decode_tp},
                "server": {"timing": "zero", "block_size": 16, **server},
                "workload": {"input_len": input_len, "concurrency": 4}}, Path("/x/run.yaml"))

        config(num_hidden_layers=36, num_key_value_heads=8)
        self.assertEqual(pd.request_write_tasks(run(1808))[0], 36 * 113)
        with self.assertRaisesRegex(ValueError, r"can be 4104 hp_tcp tasks \(36 layers x 1 region\(s\) x 114 "
                                                r"blocks of 16 tokens for 1809, none joined\), above "
                                                r"max_outstanding_tasks 4096.*max_outstanding_tasks: N\} \(N >= 4104\)"):
            run(1809)
        # A head slice (TP 2 -> 4) shortens each copy, not their number.
        self.assertEqual(pd.request_write_tasks(run(1808, decode_tp=4))[0], 36 * 113)
        with self.assertRaisesRegex(ValueError, "can be 4104 hp_tcp tasks"):
            run(1809, decode_tp=4)
        # Four concurrent requests may share a write of 4 x 4068 tasks, as
        # timing decides: not refused.
        self.assertEqual(run(1808, decode_tp=4).workload.concurrency, 4)
        self.assertEqual(run(1809, hp_tcp_args={"max_outstanding_tasks": 16384}).server.hp_tcp_args,
                         {"max_outstanding_tasks": 16384})
        with self.assertRaisesRegex(ValueError, "above max_outstanding_tasks 4104"):
            run(1825, hp_tcp_args={"max_outstanding_tasks": 4104})
        # A limit MC_TENT_CONF replaces is unknown, and NIXL's READs are not
        # this write: neither is checked.
        run(1809, env={"MC_TENT_CONF": "{}"})
        run(1809, connector="nixl")
        # Sparse MLA: the latent's and the indexer keys' regions.
        config(num_hidden_layers=61, kv_lora_rank=512, index_topk=2048)
        self.assertEqual(pd.request_write_tasks(run(448, block_size=64))[0], 61 * 2 * 7)
        with self.assertRaisesRegex(ValueError, r"61 layers x 2 region\(s\) x 34 blocks of 64 tokens"):
            run(2176, block_size=64)
        # A native MTP adds its one draft layer's regions, whatever its depth:
        # DeepSeek-V3 at 67 blocks crosses the limit only with it.
        config(num_hidden_layers=61, kv_lora_rank=512, num_nextn_predict_layers=1)
        self.assertEqual(pd.request_write_tasks(run(1072))[0], 61 * 67)
        spec = {"method": "mtp", "acceptance_rates": [0.8], "seed": 0}
        for k in (1, 2):
            with self.subTest(k=k), self.assertRaisesRegex(
                    ValueError, r"can be 4154 hp_tcp tasks \(61 layers \+ 1 MTP layer x 1 region\(s\) x 67 blocks"):
                run(1072, speculative={**spec, "num_speculative_tokens": k})
        # A sparse draft's indexer keys too.
        config(num_hidden_layers=61, kv_lora_rank=512, index_topk=2048, num_nextn_predict_layers=1)
        self.assertEqual(pd.request_write_tasks(run(448, block_size=64, speculative={
            **spec, "num_speculative_tokens": 1}))[0], 62 * 2 * 7)
        # A mixed physical layout with a startup-selected size is unknown.
        self.assertIsNone(pd.request_write_tasks(run(100000, block_size=None)))

    def _parse_write_estimate(self, config: dict, input_len: int, kv_check: bool = True,
                              **server) -> tuple[int, str] | None:
        with tempfile.TemporaryDirectory() as root:
            model = Path(root) / "org--name" / "main"
            model.mkdir(parents=True)
            (model / "config.json").write_text(json.dumps(config))
            with mock.patch.object(pd.models, "MODELS", Path(root)):
                run = pd.parse({
                    "name": "t", "topology": "t", "model": "org/name", "client": "c",
                    "control_network": "control", "router": {"gateway": "gw"},
                    "prefill": {"node": "p", "kv_network": "kv"},
                    "decode": {"node": "d", "kv_network": "kv"},
                    "server": {"timing": "zero", **server},
                    "kv_check": kv_check, "workload": {"input_len": input_len},
                }, Path(root) / "run.yaml")
                return pd.request_write_tasks(run)

    def test_hybrid_write_estimates_count_one_raw_page_per_state_layer(self) -> None:
        # Use resolved attention block sizes here. A state page includes the
        # convolution and temporal state, and is not split into token blocks.
        cases = [
            ("Mamba2", {"model_type": "nemotron_h", "num_hidden_layers": 52,
                        "hybrid_override_pattern": "M" * 23 + "E" * 23 + "*" * 6},
             2560, 2112, 35),
            ("GDN", {"text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 64,
                                    "layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16}},
             1024, 800, 80),
            ("KDA", {"model_type": "kimi_linear", "num_hidden_layers": 93, "linear_attn_config": {
                "kda_layers": [i for i in range(1, 93) if i % 4]}},
             1024, 768, 117),
        ]
        for name, config, tokens, size, expected in cases:
            with self.subTest(model=name):
                self.assertEqual(self._parse_write_estimate(config, tokens, block_size=size)[0], expected)
        with self.assertRaisesRegex(ValueError, r"80 hp_tcp tasks .*48 state layers x 1 page"):
            self._parse_write_estimate(cases[1][1], 1024, block_size=800, hp_tcp_args={"max_outstanding_tasks": 79})

    def test_hybrid_write_estimates_skip_unresolved_state_layouts(self) -> None:
        config = {"model_type": "qwen3_5_text", "num_hidden_layers": 2,
                  "layer_types": ["full_attention", "linear_attention"]}
        for server in ({"block_size": None}, {"enable_prefix_caching": True}, {"kv_check": False}):
            with self.subTest(server=server):
                self.assertIsNone(self._parse_write_estimate(config, 1024, **server))
        self.assertEqual(self._parse_write_estimate(config, 32, kv_check=False, enable_prefix_caching=False)[0], 3)
        self.assertIsNone(self._parse_write_estimate({"num_hidden_layers": 2, "attn_type_list": [0, 1]}, 1024))
        self.assertIsNone(self._parse_write_estimate(
            {"model_type": "kimi_linear", "num_hidden_layers": 2,
             "linear_attn_config": {"kda_layers": [0]}}, 1024))

    def test_plain_startup_write_estimate_uses_the_simulated_block_lower_bound(self) -> None:
        for config in ({"model_type": "qwen3", "num_hidden_layers": 36},
                       {"model_type": "deepseek_v3", "num_hidden_layers": 36, "kv_lora_rank": 512}):
            with self.subTest(config=config):
                bound, terms = self._parse_write_estimate(config, 1808, block_size=None)
                self.assertEqual(bound, 4068)
                self.assertIn("16 tokens for 1808 (startup lower bound)", terms)
                with self.assertRaisesRegex(ValueError, "4104 hp_tcp tasks"):
                    self._parse_write_estimate(config, 1809, block_size=None)
        self.assertIsNone(self._parse_write_estimate({"num_hidden_layers": 36}, 1809, block_size=None))

    def test_a_precise_prefix_index_needs_the_servers_events_and_render(self) -> None:
        base = {"name": "t", "topology": "t", "model": "org/name", "client": "c", "control_network": "control",
                "router": {"gateway": "gw", "prefix_index": "precise", "epp_verbosity": 5},
                "prefill": {"node": "p", "kv_network": "data0"}, "decode": {"node": "d", "kv_network": "data0"},
                "server": {"timing": "zero", "enable_prefix_caching": True}}
        path = Path("/x/run.yaml")
        run = pd.parse(base, path)
        self.assertEqual(run.router.prefix_index, "precise")
        # Each server: the render endpoints, and its events on a wildcard endpoint (vLLM binds only
        # those) under the endpoint the router knows it by.
        for role, port in (("prefill", 8000), ("decode", 8200)):
            command = pd.server_command(run, role, ["10.1.0.1"], port, ("10.250.0.9", 8000))
            self.assertIn("--enable-scale-out", command)
            events = json.loads(command[command.index("--kv-events-config") + 1])
            self.assertEqual(events, {
                "enable_kv_cache_events": True, "publisher": "zmq", "endpoint": "tcp://*:5557",
                "replay_endpoint": "tcp://*:5657", "topic": "kv@10.250.0.9:8000@sim-pd"})
        with self.assertRaisesRegex(ValueError, "needs the instance's endpoint"):
            pd.server_command(run, "prefill", ["10.1.0.1"])
        # An approximate index asks nothing of the servers.
        approximate = pd.server_command(RUN, "prefill", ["10.1.0.1"], 8000, ("10.250.0.9", 8000))
        self.assertFalse({"--enable-scale-out", "--kv-events-config"} & set(approximate))
        self.assertEqual(RUN.router.prefix_index, "approximate")

        def changed(**change):
            return {**base, **change}

        def router(**change):
            return {**base, "router": {**base["router"], **change}}

        cases = {
            "one of approximate, precise": router(prefix_index="exact"),
            "configures it in that file": router(epp_plugins="plugins.yaml"),
            "needs server.enable_prefix_caching": changed(server={"timing": "zero"}),
            "needs server.block_size": changed(server={"timing": "zero", "enable_prefix_caching": True,
                                                       "block_size": None}),
            "needs router.dp_routing: rank for prefill's 2 DP ranks": changed(
                prefill={"node": "p", "kv_network": "data0", "data_parallel_size": 2}),
            "needs router.dp_routing: rank for decode's 4 DP ranks": changed(
                decode={"node": "d", "kv_network": "data0", "data_parallel_size": 4}),
            "needs router.epp_verbosity 5": router(epp_verbosity=4),
        }
        for message, change in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                pd.parse(change, path)
        # With a server a rank, the index is of ranks: allowed, and the replay port leaves room for them.
        per_rank = {**router(dp_routing="rank"), "server": {**base["server"], "connector": "nixl"},
                    "prefill": {"node": "p", "kv_network": "data0", "data_parallel_size": 8},
                    "decode": {"node": "d", "kv_network": "data0", "data_parallel_size": 8}}
        # A dense model's ranks have no servers of their own (vLLM refuses external load balancing for it).
        with mock.patch.object(pd.models, "resolve", return_value=DENSE_MODEL), self.assertRaisesRegex(
                ValueError, "dp_routing: rank needs an MoE model"):
            pd.parse(per_rank, path)
        ranked = pd.parse(per_rank, path)
        self.assertTrue(pd.rank_routed(ranked, ranked.prefill) and ranked.router.prefix_index == "precise")
        self.assertGreaterEqual(pd.KV_EVENTS_REPLAY_PORT - pd.KV_EVENTS_PORT, 64)
        # Over Mooncake llm-d's sidecar cannot pick a rank server's bootstrap server.
        with self.assertRaisesRegex(ValueError, "selects one Mooncake bootstrap server a prefill host, not one an "
                                                "endpoint"):
            pd.parse({**router(dp_routing="rank"), "server": {**base["server"], "connector": "mooncake"},
                      "prefill": {"node": "p", "kv_network": "data0", "data_parallel_size": 2},
                      "decode": {"node": "d", "kv_network": "data0", "data_parallel_size": 2}}, path)

    def test_a_runs_containers_are_saved_as_docker_reports_them(self) -> None:
        inspected = json.dumps([{"Image": "sha256:abc", "Config": {"Image": "vllm:cpu-aisim"},
                                 "Mounts": [{"Source": "/src/simulated", "Destination": "/opt/venv/x"}]}])
        with mock.patch.object(pd.shell, "run", return_value=inspected) as run:
            found = pd.container_identities({"prefill0": "t-prefill0"})
        self.assertEqual(found, {"prefill0": {"container": "t-prefill0", "image_id": "sha256:abc",
                                              "image": "vllm:cpu-aisim", "mounts": ["/src/simulated:/opt/venv/x"]}})
        self.assertEqual(run.call_args.args[0], ["docker", "inspect", "t-prefill0"])

    def test_peak_prefill_throughput_is_an_unloaded_chunks(self) -> None:
        def step(tokens, seconds, *, cached=0, requests=1, decode=0, prefill=True):
            return line(step={"requests": [{"is_prefill": prefill, "num_cached_tokens": cached}] * requests,
                              "num_prefill_tokens": tokens, "num_decode_tokens": decode},
                        cost={"device_s": seconds, "host_s": 0.0})
        logs = [step(2048, 0.05) + step(2048, 0.04) + step(2048, 0.01, cached=320) + step(2048, 0.01, requests=2)
                + step(1024, 0.001) + step(2048, 0.01, decode=3) + step(2048, 0.0) + line(kv_events={}),
                step(2048, 0.064)]
        self.assertEqual(pd.peak_prefill_throughput(logs, 2048),
                         {"chunk_tokens": 2048, "steps": 3, "tokens_per_second": 51200.0})
        # Prompts shorter than a chunk: nothing to calibrate from.
        self.assertEqual(pd.peak_prefill_throughput([step(448, 0.01)], 2048),
                         {"chunk_tokens": 2048, "steps": 0, "tokens_per_second": None})

    def test_hits_are_checked_apart_from_transfers(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual((verdict["kv_checks"], verdict["kv_bytes_checked"]), (2, 80 * 8))
        self.assertEqual(verdict["prefix_hits"], {
            "prefill": {"kv_hit_checks": 1, "kv_hit_blocks": 48, "kv_bytes_hit_checked": 48 * 8},
            "decode": {"kv_hit_checks": 1, "kv_hit_blocks": 48, "kv_bytes_hit_checked": 48 * 8}})
        self.assertEqual(verdict["fragmentation"]["contiguous"], 2)
        # Without prefix caching a pull starts at the prompt's first block.
        off = self.verdict(run=pd.Run(**{**RUN.__dict__, "server": pd.Server(block_size=16)}))
        self.assertProblem(off, "pulled from block 48, not 0")
        self.assertIsNone(off["prefix_hits"])

    def test_prefix_caching_failures_are_reported(self) -> None:
        outside = {"req_id": "cmpl-b-0-p", "kind": "fill", "ok": False, "error": "outside the lifecycle"}
        cases = {
            "prefill: 1 prefix hit(s) held bytes unlike": {"prefill_checks": [hit("b", "p", 48, ok=False)]},
            "decode: 1 prefix hit(s) held bytes unlike": {"decode_checks": [A_CHECK, hit("b", "d", 48, False),
                                                                            B_CHECK]},
            "outside the pattern lifecycle": {"prefill_checks": [hit("b", "p", 48), outside]},
            "prefill showed no checked prefix hit": {"prefill_checks": []},
            # The pull past the hit takes the rest of the prompt, not less.
            "checked 15 blocks, not the cdiv(1024, 16) of its prompt past its 48-block prefix hit": {
                "decode_checks": [A_CHECK, hit("b", "d", 48),
                                  checked("b", blocks=15, layers=2, bytes=120, groups=[group_check(0, 48, 15)])],
                "moved": 79 * 8},
        }
        for message, changes in cases.items():
            with self.subTest(message=message):
                self.assertProblem(self.verdict(**changes), message)

    def test_hits_must_cover_each_admitted_prefix(self) -> None:
        """Response 128: a hit check must be the request's own, of exactly
        the places its admission left to the cache, its blocks there."""
        cases = {
            # b's 48-block hit missing, a's unrelated one-block hit instead.
            "unrelated": ({"decode_checks": [A_CHECK, hit("a", "d", 1), B_CHECK]},
                          ["decode: cmpl-a-0-d logged 1 prefix hit check(s), not the 0",
                           "decode: cmpl-b-0-d logged 0 prefix hit check(s), not the 1",
                           "decode showed no checked prefix hit"]),
            "missing": ({"decode_checks": [A_CHECK, B_CHECK]},
                        ["cmpl-b-0-d logged 0 prefix hit check(s), not the 1 its admissions imply ([48] block(s))"]),
            "shortened": ({"decode_checks": [A_CHECK, hit("b", "d", 47), B_CHECK]},
                          ["cmpl-b-0-d's hit check covers group 0: 47 block(s) 0:101..46:147, not its admitted "
                           "hit group 0: 48 block(s) 0:101..47:148"]),
            "other blocks": ({"prefill_checks": [hit("b", "p", 48, first_block=2)]},
                             ["prefill: cmpl-b-0-p's hit check covers group 0: 48 block(s) 0:2..47:49"]),
            "zero blocks and bytes": ({"decode_checks": [A_CHECK, hit("b", "d", 0), B_CHECK],
                                       "prefill_checks": [hit("b", "p", 0)]},
                                      ["prefill: cmpl-b-0-p's hit check covers no bytes",
                                       "decode: cmpl-b-0-d's hit check covers no bytes",
                                       "prefill showed no checked prefix hit", "decode showed no checked prefix hit"]),
            "bytes unlike the layout": ({"prefill_checks": [{**hit("b", "p", 48), "bytes": 48 * 4, "groups": [
                {**hit("b", "p", 48)["groups"][0], "bytes": 48 * 4}]}]},
                ["cmpl-b-0-p's hit check of group 0 (48 block(s), 192 bytes) disagrees with the layout (8 bytes"]),
            # The pull's own plan: its blocks are the request's at their
            # places, and it starts within the tokens it was admitted with.
            "plan unlike the table": ({"decode_admissions": DECODE_ADMISSIONS.replace(
                json.dumps(blocks(301, 16)), json.dumps(blocks(302, 16)), 2)},
                ["cmpl-b-0-d's pull of group 0 (16 block(s) from 48) is not its blocks there when admitted"]),
            "plan past the admission": ({"decode_admissions": DECODE_ADMISSIONS.replace(
                '"cmpl-b-0-d": 1023', '"cmpl-b-0-d": 700')},
                ["cmpl-b-0-d's pull of group 0 starts at block 48, past the 700 tokens"]),
        }
        for name, (changes, messages) in cases.items():
            with self.subTest(name):
                self.assertProblem(self.verdict(**changes), *messages)
        # Prefill admitted b with 48 blocks cached: a 47-block admission
        # makes its 48-block check another's.
        shorter = PREFILL_ADMISSIONS.replace('"cmpl-b-0-p": 768', '"cmpl-b-0-p": 752')
        self.assertProblem(self.verdict(prefill_admissions=shorter), "not its admitted hit group 0: 47 block(s)")

    def test_hit_bytes_count_each_of_a_sides_ranks_once(self) -> None:
        """At unequal TP sizes a side's ranks are not the pairings: a
        prefill rank feeding two decode ranks counts once, both prefill
        ranks feeding one decode rank count."""
        def side(tp_size, rank_bytes, dps=1):
            return {(dp, tp): {"kv_hit_checks": 3, "kv_hit_blocks": 48, "kv_bytes_hit_checked": rank_bytes}
                    for dp in range(dps) for tp in range(tp_size)}

        for prefill_tp, decode_tp in ((1, 2), (2, 1)):
            with self.subTest(prefill_tp=prefill_tp, decode_tp=decode_tp):
                totals = pd._prefix_hit_totals({"prefill": side(prefill_tp, 800 // prefill_tp),
                                                "decode": side(decode_tp, 800 // decode_tp)})
                self.assertEqual(totals, {role: {"kv_hit_checks": 3, "kv_hit_blocks": 48, "kv_bytes_hit_checked": 800}
                                          for role in ("prefill", "decode")})
        # Requests are counted once for each DP rank that owns them.
        self.assertEqual(pd._prefix_hit_totals({"prefill": side(2, 400, dps=2)})["prefill"],
                         {"kv_hit_checks": 6, "kv_hit_blocks": 96, "kv_bytes_hit_checked": 1600})

    def test_a_request_served_from_hits_alone_is_consumed_once(self) -> None:
        """a pulled; b's prompt (1025 tokens) is a's and one more: decode
        hits all 64 of its full blocks and pulls nothing; c hits 48 and
        pulls 16."""
        run = pd.Run(**{**PREFIX.__dict__, "workload": pd.Workload(num_prompts=3)})
        prefill = (admit("a", "p", 0, blocks(1, 64)) + admit("b", "p", 1024, blocks(1, 64) + [217])
                   + admit("c", "p", 768, blocks(1, 48) + blocks(201, 16)))
        decode = (post("a", 0, blocks(101, 64)) + admit("a", "d", 1023, blocks(101, 64))
                  + admit("b", "d", 1024, blocks(101, 64) + [400])
                  + post("c", 48, blocks(301, 16)) + admit("c", "d", 1023, blocks(101, 48) + blocks(301, 16)))
        steps = (stepped("a", "d", 1023, 1, 1024) + stepped("b", "d", 1024, 1, 1025)
                 + stepped("c", "d", 1023, 1, 1024))
        c_check = {**B_CHECK, "req_id": "cmpl-c-0-d"}
        good = dict(run=run, prefill_admissions=prefill, decode_admissions=decode, decode_steps=steps,
                    decode_checks=[A_CHECK, hit("b", "d", 64), hit("c", "d", 48), c_check],
                    prefill_checks=[hit("b", "p", 64), hit("c", "p", 48)])
        verdict = self.verdict(**good)
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["kv_checks"], 2)
        self.assertEqual(verdict["prefix_hits"]["decode"], {"kv_hit_checks": 2, "kv_hit_blocks": 112,
                                                            "kv_bytes_hit_checked": 112 * 8})
        self.assertEqual(verdict["ownership"], {"producers": {0: 3}, "consumers": {0: 3}})
        # b's hit not checked: nothing shows decode consumed it.
        self.assertProblem(self.verdict(**{**good, "decode_checks": [A_CHECK, hit("c", "d", 48), c_check]}),
                           "decode admitted cmpl-b-0-d with neither a pull nor a checked prefix hit")
        # c pulled: its byte check is still required.
        self.assertProblem(self.verdict(**{**good, "decode_checks": [A_CHECK, hit("b", "d", 64),
                                                                     hit("c", "d", 48)]}),
                           "1 pulled request(s) without a successful byte check: cmpl-c-0-d",
                           "decode logged 1 byte check(s) for 2 distinct request(s) that pulled")
        # Response 129: b reuses 768 of its 1024 tokens and prefills the
        # other 256 itself, with no pull: a missing transfer, not a full hit.
        partial = dict(run=PREFIX, decode_checks=[A_CHECK, hit("b", "d", 48)], moved=512,
                       decode_admissions=post("a", 0, blocks(101, 64)) + admit("a", "d", 1023, blocks(101, 64))
                       + admit("b", "d", 768, blocks(101, 48) + blocks(301, 16)),
                       decode_steps=stepped("a", "d", 1023, 1, 1024) + stepped("b", "d", 768, 256, 1024))
        self.assertProblem(self.verdict(**partial),
                           "decode admitted cmpl-b-0-d without a pull, 768 of its 1024 prompt tokens cached, its "
                           "first step computing from 768: it prefilled 256 tokens itself, not only the last")
        # Its admission alone claiming the whole prompt does not hide the step.
        whole = hit("b", "d", 63)
        whole["groups"][0]["block_ids"] = blocks(101, 48) + blocks(301, 15)
        claimed = {**partial, "decode_checks": [A_CHECK, whole],
                   "decode_admissions": partial["decode_admissions"].replace('"cmpl-b-0-d": 768', '"cmpl-b-0-d": 1023')}
        self.assertEqual([p for p in self.verdict(**claimed)["problems"] if "hit" in p and "prefilled" not in p], [])
        self.assertProblem(self.verdict(**claimed), "it prefilled 256 tokens itself")
        # A request decode never admitted is not consumed.
        self.assertProblem(self.verdict(**{**good, "decode_admissions": decode.replace(
            admit("b", "d", 1024, blocks(101, 64) + [400]), "")}), "decode admitted 2 distinct request(s), not 3")

    def test_fragmentation_counts_checked_matched_pulls_only(self) -> None:
        def pull(request: str, places: list[int], first: int = 2) -> dict:
            return {"kv_events": {"reads_posted": [f"cmpl-{request}-0-d"], "allocated_groups": {
                f"cmpl-{request}-0-d": [{"group": 0, "first": first, "blocks": places}]}}}

        # Prefill's tables (a chunk's allocations continue at their places).
        prefill = [pd._records(allocated_groups("a", (0, 0, [1, 2, 3, 4])) + allocated_groups("b", (0, 0, [5, 6]))
                               + allocated_groups("b", (0, 2, [9, 7]))
                               + allocated_groups("c", (0, 0, [10, 11, 12, 13]))
                               + allocated_groups("d", (0, 0, [14, 15, 17, 19])))]
        decode = [pull("a", [20, 21]), pull("b", [30, 31]), pull("c", [40, 42]),
                  pull("x", [50]),  # no producer allocation logged: unmatched
                  pull("d", [60, 62]),  # fragmented on both, but not checked
                  {"kv_events": {"reads_posted": ["cmpl-y-0-d"], "allocated": {"cmpl-y-0-d": [70, 72]}}}]
        checked_ids = {f"cmpl-{r}-0-d" for r in "abcxy"}
        self.assertEqual(pd._fragmentation(prefill, [decode], [checked_ids]), {
            "pulls": 6, "contiguous": 1, "fragmented_prefill_only": 1, "fragmented_decode_only": 1,
            "fragmented_both": 0, "unlogged": 1, "unchecked": 1, "unmatched": 1,
            "jointly_contiguous_runs": 1 + 2 + 2, "unmatched_groups": 1})
        # Checked, d is fragmented on both.
        self.assertEqual(pd._fragmentation(prefill, [decode], [checked_ids | {"cmpl-d-0-d"}])["fragmented_both"], 1)
        # A checked pull without the producer's allocation is not contiguous.
        alone = pd._fragmentation([], [[pull("missing", [50])]], [{"cmpl-missing-0-d"}])
        self.assertEqual((alone["contiguous"], alone["unmatched"], alone["unmatched_groups"]), (0, 1, 1))
        # In a run: neither an unchecked pull nor an unmatched one meets
        # expect_fragmentation (response 128's counterexamples).
        expecting = pd.Run(**{**PREFIX.__dict__, "expect_fragmentation": True})
        for prefill_events, decode_events in (
                (allocated_groups("unverified", (0, 0, [14, 15, 17, 19])), line(**pull("unverified", [60, 62]))),
                ("", line(**pull("missing", [50])))):
            verdict = self.verdict(prefill_events=prefill_events, decode_events=decode_events, run=expecting)
            self.assertProblem(verdict, "no byte-checked transfer was fragmented on both instances",
                               "1 pull(s) posted for requests decode never admitted")
            self.assertEqual((verdict["fragmentation"]["contiguous"], verdict["fragmentation"]["unchecked"]), (2, 1))


# hp_tcp's task limit: connector writes reported, TENT's failures checked.
def sends(*writes, failed=(), t=20.0) -> str:
    """A Mooncake prefill step's writes: (bytes, descriptors, start, end)
    each, and failed ones (bytes, descriptors, start), reported at `t`."""
    send = {"count": len(writes), "bytes": sum(w[0] for w in writes),
            "each": [[b, end - start, 0.0, n, end] for b, n, start, end in writes]}
    if failed:
        send |= {"failed": len(failed), "failed_each": [list(f) for f in failed]}
    return line(kv_transfers={"send": send}, t=t)


class HpTcpEvidenceTests(unittest.TestCase):
    def verdict(self, writes: str, prefill_tent=None, run=RUN) -> dict:
        prefill = ONE_RANK + allocated("a", "b") + writes
        tents = {"prefill": [{**tent(), **(prefill_tent or {})}], "decode": [tent(0)]}
        return pd.check({"completed": 2, "errors": []}, [prefill], [GOOD["decode"]],
                        {"control": 50, "data0": 2100}, run, tents)

    def hp_tcp(self, verdict: dict) -> dict:
        return verdict["dp_ranks"]["prefill"][0]["hp_tcp"]

    def test_a_mergeable_write_above_the_limit_is_reported_not_rejected(self) -> None:
        # 5,000 descriptors TENT merged into 3,000 tasks, admitted.
        verdict = self.verdict(sends((2000, 5000, 10.0, 10.2)), {"tent_transport_attempts_total": 3000})
        self.assertEqual(verdict["problems"], [])
        evidence = self.hp_tcp(verdict)
        self.assertEqual((evidence["max_connector_descriptors_per_write"], evidence["transport_attempts"],
                          evidence["task_failures"], evidence["task_limit"]), (5000, 3000, 0, 4096))
        self.assertIn("exceeds the task limit 4096 (TENT may merge them", evidence["notes"][0])
        self.assertEqual(verdict["hp_tcp_task_limit"], 4096)

    def test_a_recovered_attempt_still_fails_the_clean_transfer_check(self) -> None:
        # Refused once, then admitted: the connector saw success.
        verdict = self.verdict(sends((1000, 2288, 10.0, 10.5), (1000, 2548, 9.95, 10.1)),
                               {"tent_task_failures_total": 2288, "tent_transport_attempt_failures_total": 2288})
        self.assertEqual(verdict["problems"], [
            "prefill's TENT recorded 2288 failed task(s) and 2288 failed transport attempt(s): not a clean "
            "transfer, whether or not a retry recovered it (0 connector transfer(s) failed)"])
        self.assertEqual(self.hp_tcp(verdict)["failed_connector_writes"], 0)

    def test_an_overlap_above_the_limit_without_failures_passes(self) -> None:
        verdict = self.verdict(sends((1000, 3000, 10.0, 10.1), (1000, 2500, 10.05, 10.12)))
        self.assertEqual(verdict["problems"], [])
        evidence = self.hp_tcp(verdict)
        self.assertEqual((evidence["overlapping_call_descriptor_bound"], evidence["bound_complete"]), (5500, True))
        self.assertIn("writes of 5500 connector descriptors overlapped, above the task limit 4096 (an upper bound",
                      evidence["notes"][0])

    def test_failed_writes_count_and_unknown_spans_leave_the_bound_incomplete(self) -> None:
        # A failed write ends no later than the record reporting it.
        evidence = pd._hp_tcp_writes(pd._records(sends((1000, 100, 10.0, 10.1), failed=[(500, 3000, 10.05)])), 4096)
        self.assertEqual((evidence["overlapping_call_descriptor_bound"], evidence["bound_complete"],
                          evidence["failed_connector_writes"], evidence["max_connector_descriptors_per_write"]),
                         (3100, True, 1, 3000))
        for name, records in (
                ("no start", sends((1000, 100, 10.0, 10.1), failed=[(500, 3000, None)])),
                ("reported before its start", sends((1000, 100, 10.0, 10.1), failed=[(500, 3000, 30.0)])),
                ("counted, not listed", line(kv_transfers={"send": {"count": 0, "bytes": 0, "failed": 1}})),
                ("a write without its end", line(kv_transfers={"send": {"count": 1, "bytes": 1000,
                                                                        "each": [[1000, 0.1, 0.0, 100, None]]}}))):
            with self.subTest(name):
                evidence = pd._hp_tcp_writes(pd._records(records), 4096)
                self.assertFalse(evidence["bound_complete"])

    def test_the_c6_shape(self) -> None:
        # 2,288 refused while 2,548 were in flight, then admitted; a write of
        # two requests' 4,264 refused twice, failed.
        writes = sends((1000, 2548, 11.0, 11.2), (1000, 2288, 11.02, 11.52), failed=[(1000, 4264, 12.0)], t=14.1)
        verdict = self.verdict(writes, {"tent_task_failures_total": 10816,
                                        "tent_transport_attempt_failures_total": 10816})
        self.assertIn("prefill's connector failed 1 transfer(s)", verdict["problems"])
        self.assertTrue(any("TENT recorded 10816 failed task(s) and 10816 failed transport attempt(s)" in p
                            and "(1 connector transfer(s) failed)" in p for p in verdict["problems"]),
                        verdict["problems"])
        evidence = self.hp_tcp(verdict)
        self.assertEqual((evidence["max_connector_descriptors_per_write"], evidence["overlapping_call_descriptor_bound"],
                          evidence["failed_connector_writes"], evidence["bound_complete"]), (4264, 4836, 1, True))

    def test_the_effective_limit(self) -> None:
        conf = lambda hp_tcp: {"MC_TENT_CONF": json.dumps({"transports": {"hp_tcp": hp_tcp}})}
        for server, limit in (
                (pd.Server(), 4096),
                (pd.Server(hp_tcp_args={"max_outstanding_tasks": 16384}), 16384),
                # A spec's own TENT configuration replaces the harness's.
                (pd.Server(hp_tcp_args={"max_outstanding_tasks": 16384},
                           env=conf({"enable": True, "max_outstanding_tasks": 8192})), 8192),
                (pd.Server(env=conf({"enable": True})), 4096),
                (pd.Server(env={"MC_TENT_CONF": "{not json"}), None),
                (pd.Server(env={"MC_TENT_CONF": json.dumps({"transports": {}})}), None),
                (pd.Server(env={"MC_USE_TENT": 0}), None)):
            with self.subTest(server=server):
                self.assertEqual(pd.hp_tcp_task_limit(server), limit)
        unknown = pd.Run(**{**RUN.__dict__, "server": pd.Server(env={"MC_TENT_CONF": "{not json"})})
        verdict = self.verdict(sends((2000, 100, 10.0, 10.1)), run=unknown)
        self.assertIn("task limit is unknown", self.hp_tcp(verdict)["notes"][0])
        # NIXL (its own run: decode loads and TENT reads, prefill writes nothing): prefill's hp_tcp
        # evidence is TENT's counters only, without the connector-write keys Mooncake's always has.
        decode = ONE_RANK + line(kv_transfers={"load": {"count": 2, "bytes": 2000, "failed": 0}}) + line(
            kv_check=checked("a")) + line(kv_check=checked("b"))
        tents = {"prefill": [{"tent_registered_buffer_bytes": 1600, "tent_transport_attempts_total": 7}],
                 "decode": [{"tent_read_bytes_total": 2000, "tent_registered_buffer_bytes": 1600}]}
        nixl = pd.check({"completed": 2, "errors": []}, [ONE_RANK + allocated("a", "b")], [decode],
                        {"control": 50, "data0": 2100}, NIXL, tents)
        self.assertEqual(nixl["problems"], [])
        evidence = self.hp_tcp(nixl)
        for key in ("max_connector_descriptors_per_write", "connector_writes", "task_limit"):
            self.assertNotIn(key, evidence)
        self.assertEqual(evidence["transport_attempts"], 7)
        mooncake = self.hp_tcp(self.verdict(sends((2000, 100, 10.0, 10.1))))
        self.assertEqual((mooncake["max_connector_descriptors_per_write"], mooncake["task_limit"]), (100, 4096))


# Unequal TP: heads re-sharded, reconciled with the native pairing.
SLAB = 10  # a head's bytes in a block of a layer
PER_HEAD = 100  # a request's bytes of one head (5 blocks x 2 layers x SLAB)


def hetero_logs(prefill_tp: int, decode_tp: int, kv_heads: int, nixl: bool, requests=("a", "b"), groups=None):
    """Consistent step logs, transfers, ports and TENT scrapes of a run whose
    instances have unequal TP sizes: each request's heads moved as the
    native pairing gives them, a READ's descriptors one per block and layer,
    a replica's answers naming their requests, and each rank's TENT series
    as a scrape serves them (one never incremented absent). `groups`: KV
    cache groups of (kind, total heads, slab) each, in place of one group of
    `kv_heads` heads of SLAB bytes (Gemma4: sliding 8 heads of 10 bytes, full
    2 of 40: pages of equal bytes at TP1)."""
    groups = groups or [("FullAttentionSpec", kv_heads, SLAB)]
    tps = (("prefill", prefill_tp), ("decode", decode_tp))
    gheads = {role: [[pd.global_heads(r, n, g[1]) for g in groups] for r in range(n)] for role, n in tps}
    heads = {role: [gh[0] for gh in gheads[role]] for role in gheads}
    ratio = max(prefill_tp, decode_tp) // min(prefill_tp, decode_tp)
    paired = {d: [d * prefill_tp // decode_tp] if decode_tp >= prefill_tp else
              [d * ratio + i for i in range(ratio)] for d in range(decode_tp)}
    gsender = [{(h, d): min(p for p in paired[d] if h in gheads["prefill"][p][g])
                for d in range(decode_tp) for h in gheads["decode"][d][g]} for g in range(len(groups))]
    ph = [10 * g[2] for g in groups]  # a request's bytes of one head of each group

    def layout(rank, tp, mine):
        return line(kv_layout={"tp_rank": rank, "tp_size": tp, "block_bytes": 20, "bytes": 1600, "groups": [
            {"kind": kind, "sliding_window": None, "block_size": 16, "layers": 2,
             "block_bytes": slab * len(mine[g]), "real_page_bytes": slab * len(mine[g]), "pattern": "head-v1",
             "global_heads": mine[g]} for g, (kind, _, slab) in enumerate(groups)]})

    def group(mine):
        return [{"group": g, "first_logical_block": 0, "blocks": 5, "layers": 2, "block_bytes": groups[g][2] * len(mine[g]),
                 "bytes": ph[g] * len(mine[g]), "mismatched_bytes": 0, "pattern": "head-v1", "global_heads": mine[g]}
                for g in range(len(groups))]

    sends = {p: [] for p in range(prefill_tp)}
    loads = {d: [] for d in range(decode_tp)}
    zero = {p: [] for p in range(prefill_tp)}
    for d in range(decode_tp):
        sources = sorted({gsender[g][(h, d)] for g in range(len(groups)) for h in gheads["decode"][d][g]})
        for p in paired[d]:
            carried = [[h for h in gheads["decode"][d][g] if gsender[g][(h, d)] == p] for g in range(len(groups))]
            if not any(carried):
                zero[p] += [[d, [f"cmpl-{r}-0-d"], [[False, 0, 0, SLAB]]] for r in requests]
                continue
            plan = []
            for g, c in enumerate(carried):
                region = [True, gheads["prefill"][p][g].index(c[0]) * groups[g][2],
                          gheads["decode"][d][g].index(c[0]) * groups[g][2], groups[g][2] * len(c)]
                if c and region not in plan:
                    plan.append(region)
            for _ in requests:
                moved_bytes = sum(ph[g] * len(c) for g, c in enumerate(carried))
                entry = [moved_bytes, 0.01, 0.0, 10 * len(groups), 10.0]
                if nixl:
                    slot = sources.index(p) if len(sources) > 1 else 0
                    loads[d].append(entry + [p, [slot, 10 * len(groups), 10 * len(groups)]])
                else:
                    sends[p].append(entry + [d, plan])

    def transfers(direction, each, zero_payload=()):
        record = {"count": len(each), "bytes": sum(e[0] for e in each), "each": each}
        if zero_payload:
            record["zero_payload"] = list(zero_payload)
        return line(kv_transfers={direction: record})

    prefill = [layout(p, prefill_tp, gheads["prefill"][p]) + allocated(*requests)
               + transfers("send", sends[p], zero[p]) for p in range(prefill_tp)]
    decode = [layout(d, decode_tp, gheads["decode"][d])
              + "".join(stepped(r, "d", 80, 1, 80) for r in requests)
              + transfers("load", loads[d])
              + "".join(line(kv_check=checked(r, blocks=5, layers=2,
                                              bytes=sum(ph[g] * len(gheads["decode"][d][g]) for g in range(len(groups))),
                                              groups=group(gheads["decode"][d]))) for r in requests)
              for d in range(decode_tp)]
    sent = {p: sum(e[0] for e in sends[p]) + sum(e[0] for d in loads for e in loads[d] if e[5] == p)
            for p in range(prefill_tp)}
    received = {d: sum(e[0] for e in loads[d]) + sum(e[0] for p in sends for e in sends[p] if e[5] == d)
                for d in range(decode_tp)}
    ports = {"kv": {**{f"p{p}": {"node": "p", "slot": p, "rx": sent[p] + 100, "tx": 40} for p in range(prefill_tp)},
                    **{f"p{prefill_tp + d}": {"node": "d", "slot": d, "rx": 40, "tx": received[d] + 100}
                       for d in range(decode_tp)}}}

    if nixl:
        tents = {"prefill": [tent_scrape() for _ in range(prefill_tp)],
                 "decode": [tent_scrape(received[d], "read", len(loads[d])) for d in range(decode_tp)]}
    else:
        tents = {"prefill": [tent_scrape(sent[p], "write", len(sends[p])) if sends[p] else tent_scrape()
                             for p in range(prefill_tp)],
                 "decode": [tent_scrape() for _ in range(decode_tp)]}
    return prefill, decode, ports, sum(sent.values()), tents


def tent_scrape(moved=None, kind="write", transfers=0):
    """A rank's TENT series before and after the workload: an engine that
    never moved a byte serves no series for it."""
    def served(moved_bytes, count):
        lines = ["tent_registered_buffer_bytes 1600"]
        if moved is not None:
            lines += [f'tent_{kind}_bytes_total{{transport="tcp"}} {moved_bytes}',
                      f"tent_{kind}_requests_total {count}", f"tent_transport_attempts_total {count}"]
        return "\n".join(lines)

    return pd._tent_delta(served(0, 0), served(moved, transfers))


def dp_logs(make, owners, sizes, dps, nixl=True):
    """One fixture's logs inside data parallelism: ``make(request)`` gives
    one request's logs (prefill's and decode's, each rank's layout first,
    and ports), request r produced by prefill DP rank ``owners[r][0]`` and
    consumed by decode DP rank ``owners[r][1]``. Each rank's log is its
    layout (its DP rank's) and then its DP rank's requests' records, in
    `Side.ranks` order; link r carries rank r of every DP rank; each rank's
    TENT series its own transfers' (none served where it moved nothing)."""
    bodies = {role: {(dp, tp): [] for dp in range(dps[i]) for tp in range(sizes[role])}
              for i, role in enumerate(("prefill", "decode"))}
    layouts = {}
    payload = {role: [0] * sizes[role] for role in sizes}
    for request, (pdp, ddp) in owners.items():
        prefill, decode, ports = make(request)[:3]
        for role, texts, dp in (("prefill", prefill, pdp), ("decode", decode, ddp)):
            for tp, text in enumerate(texts):
                first, *rest = text.splitlines(keepends=True)
                layouts[(role, tp)] = json.loads(first)["kv_layout"]
                bodies[role][(dp, tp)] += rest
        for port in ports["kv"].values():
            role, direction = ("prefill", "rx") if port["node"] == "p" else ("decode", "tx")
            payload[role][port["slot"]] += port[direction] - 100
    texts = {role: [line(kv_layout={**layouts[(role, tp)], "dp_rank": dp}) + "".join(body)
                    for (dp, tp), body in bodies[role].items()] for role in bodies}
    ports = {"kv": {**{f"p{p}": {"node": "p", "slot": p, "rx": payload["prefill"][p] + 100, "tx": 40}
                       for p in range(sizes["prefill"])},
                    **{f"p{sizes['prefill'] + d}": {"node": "d", "slot": d, "rx": 40,
                                                    "tx": payload["decode"][d] + 100}
                       for d in range(sizes["decode"])}}}
    mover, kind, direction = ("decode", "read", "load") if nixl else ("prefill", "write", "send")
    tents = {role: [] for role in texts}
    for role in texts:
        for text in texts[role]:
            moves = [(r.get("kv_transfers") or {}).get(direction) or {} for r in pd._records(text)]
            count = sum(m.get("count", 0) for m in moves)
            tents[role].append(tent_scrape(sum(m.get("bytes", 0) for m in moves), kind, count)
                               if role == mover and count else tent_scrape())
    return texts["prefill"], texts["decode"], ports, sum(payload["prefill"]), tents


# Gemma4's groups: sliding layers of 8 KV heads (10-byte slabs), full ones of
# 2 (40-byte slabs): pages of equal bytes (80) at TP1, a common carried span of
# 40 bytes at TP1 <-> 2.
GEMMA = [("SlidingWindowSpec", 8, 10), ("FullAttentionSpec", 2, 40)]


class HeterogeneousTpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.model = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def run_for(self, prefill_tp, decode_tp, kv_heads, nixl, global_heads=None):
        # kv_heads 1: an MLA model's latent (DeepSeek-V3's config: 128
        # attention KV heads, one latent cached).
        config = {"num_key_value_heads": 128, "kv_lora_rank": 512} if kv_heads == 1 else {
            "num_key_value_heads": kv_heads}
        if global_heads is not None:
            config["num_global_key_value_heads"] = global_heads
        (self.model / "config.json").write_text(json.dumps(config))
        return pd.Run(**{**RUN.__dict__, "model": self.model,
                         "prefill": pd.Side("p", "kv", tensor_parallel_size=prefill_tp),
                         "decode": pd.Side("d", "kv", tensor_parallel_size=decode_tp),
                         "server": pd.Server(connector="nixl" if nixl else "mooncake")})

    def verdict(self, prefill_tp, decode_tp, kv_heads, nixl, change=None, tent_change=None, groups=None,
                global_heads=None) -> dict:
        prefill, decode, ports, moved, tents = hetero_logs(prefill_tp, decode_tp, kv_heads, nixl, groups=groups)
        if change:
            prefill, decode, ports = change(prefill, decode, ports)
        if tent_change:
            tent_change(tents)
        return pd.check({"completed": 2, "errors": []}, prefill, decode, {"control": 50, "kv": moved + 500},
                        self.run_for(prefill_tp, decode_tp, kv_heads, nixl, global_heads), tents, ports)

    def test_heads_resharded_either_way_over_both_connectors(self) -> None:
        for sizes, kv_heads in (((2, 4), 8), ((4, 2), 8), ((2, 8), 4)):
            for nixl in (False, True):
                with self.subTest(sizes=sizes, nixl=nixl):
                    verdict = self.verdict(*sizes, kv_heads, nixl)
                    self.assertEqual(verdict["problems"], [])
                    hetero = verdict["hetero_tp"]
                    self.assertEqual(hetero["sizes"], {"prefill": sizes[0], "decode": sizes[1]})
                    # Decode replicating heads (TP 8 > K 4) carries each twice.
                    self.assertEqual(hetero["replication_factor"], 2.0 if sizes == (2, 8) else 1.0)
        # Each decode rank reads half a producer's heads (2 -> 4), or
        # gathers two producers' (4 -> 2).
        self.assertEqual(self.verdict(2, 4, 8, True)["hetero_tp"]["sources"], {0: [0], 1: [0], 2: [1], 3: [1]})
        self.assertEqual(self.verdict(4, 2, 8, True)["hetero_tp"]["sources"], {0: [0, 1], 1: [2, 3]})

    def reconcile(self, prefill_tp, decode_tp, kv_heads, nixl, change=None, groups=None, global_heads=None) -> dict:
        """`_hetero_tp` alone over the fixture's logs (several groups' byte
        checks and pull events are `check`'s other rules): its evidence,
        with the problems."""
        prefill, decode, ports, _, tents = hetero_logs(prefill_tp, decode_tp, kv_heads, nixl, groups=groups)
        if change:
            prefill, decode, ports = change(prefill, decode, ports)
        run = self.run_for(prefill_tp, decode_tp, kv_heads, nixl, global_heads)
        logs = {"prefill": {(0, r): pd._records(text) for r, text in enumerate(prefill)},
                "decode": {(0, r): pd._records(text) for r, text in enumerate(decode)}}
        def moved(records):
            return sum(((r.get("kv_transfers") or {}).get(d) or {}).get("bytes", 0)
                       for r in records for d in ("send", "load"))

        ranks = {role: {key: {"transfers": {"bytes": moved(records)}, "requests": ["a", "b"]} for key, records in logs[role].items()}
                 for role in logs}
        return pd._hetero_tp(run, {"prefill": prefill_tp, "decode": decode_tp}, logs, ranks, ports)

    def test_groups_of_differing_total_heads_pair_per_group(self) -> None:
        """Gemma4-like groups (sliding 8 heads, full 2) at TP1 <-> 2 over both
        connectors: each group's heads, slab and bytes are its own, the
        carried span (40 B) is common, and the reconciliation passes."""
        for sizes in ((1, 2), (2, 1)):
            for nixl in (False, True):
                with self.subTest(sizes=sizes, nixl=nixl):
                    hetero = self.reconcile(*sizes, 8, nixl, groups=GEMMA, global_heads=2)
                    self.assertEqual(hetero["problems"], [])
                    self.assertEqual(hetero["group_kv_heads"], [8, 2])
                    self.assertEqual(hetero["slab_bytes"], [10, 40])
                    self.assertEqual(hetero["replication_factor"], 1.0)

    def test_a_wrong_group_total_is_rejected_despite_matching_slabs_and_bytes(self) -> None:
        """The slab counterexample: the full group's total taken as 8 (the
        config's global count wrong) gives the same 10-byte slab on both
        sides (80/8 at TP1, 40/4 at TP2) and the same bytes, yet its actual
        TP2 head set is [0], not four heads: rejected by the logged heads."""
        for nixl in (False, True):
            with self.subTest(nixl=nixl):
                problems = self.reconcile(1, 2, 8, nixl, groups=GEMMA, global_heads=8)["problems"]
                self.assertTrue(any("group 1 holds head-v1 heads [0], not head-v1 heads [0, 1, 2, 3]" in problem
                                    for problem in problems), problems)
                self.assertTrue(any("decode rank 0 checked" in problem and "heads [0]" in problem
                                    for problem in problems), problems)

    def test_group_totals_come_from_the_config_by_kind_or_fail(self) -> None:
        def totals(config, kinds):
            (self.model / "config.json").write_text(json.dumps(config))
            return pd.group_kv_heads(self.model, kinds)

        plain = {"num_key_value_heads": 8}
        self.assertEqual(totals(plain, ["FullAttentionSpec", "SlidingWindowSpec"]), ([8, 8], None))
        gemma = {"num_key_value_heads": 8, "num_global_key_value_heads": 2}
        self.assertEqual(totals(gemma, ["SlidingWindowSpec", "FullAttentionSpec"]), ([8, 2], None))
        # Nested text config, as a multimodal wrapper writes it.
        nested = {"text_config": {"num_key_value_heads": 8, "num_global_key_value_heads": 2}}
        self.assertEqual(totals(nested, ["FullAttentionSpec"]), ([2], None))
        # No default: an unknown kind, or an invalid count, is an error.
        self.assertIsNone(totals(gemma, ["MLAAttentionSpec"])[0])
        self.assertIn("MLAAttentionSpec", totals(gemma, ["MLAAttentionSpec"])[1])
        for bad in (0, -2, True, 2.0, "2"):
            result = totals({**gemma, "num_global_key_value_heads": bad}, ["FullAttentionSpec"])
            self.assertIsNone(result[0], bad)
        self.assertIsNone(totals({"num_global_key_value_heads": 2}, ["FullAttentionSpec"])[0])

    def test_groups_must_agree_across_ranks_and_carry_one_span(self) -> None:
        # A rank whose groups are of other kinds than the rest's.
        def other_kind(prefill, decode, ports):
            decode = list(decode)
            decode[1] = decode[1].replace('"kind": "SlidingWindowSpec"', '"kind": "FullAttentionSpec"', 1)
            return prefill, decode, ports

        found = self.reconcile(1, 2, 8, False, other_kind, groups=GEMMA, global_heads=2)["problems"]
        self.assertTrue(any("groups are" in problem for problem in found), found)
        # A READ whose bytes are not its selected descriptors' carried span.
        def off_by_one(prefill, decode, ports):
            return prefill, [log.replace("[800, 0.01", "[801, 0.01") for log in decode], ports

        found = self.reconcile(1, 2, 8, True, off_by_one, groups=GEMMA, global_heads=2)["problems"]
        self.assertTrue(any("carried span" in problem for problem in found), found)

    def test_a_replicated_producer_answers_without_a_payload(self) -> None:
        """Mooncake 8 -> 2 with 4 KV heads: ranks 0, 2, 4, 6 send; 1, 3, 5, 7
        answer with no payload, which must be recorded."""
        verdict = self.verdict(8, 2, 4, False)
        self.assertEqual(verdict["problems"], [])
        pairs = {(pair["prefill"], pair["decode"]): pair for pair in verdict["hetero_tp"]["pairs"]}
        self.assertEqual(pairs[(1, 0)], {"prefill": 1, "decode": 0, "bytes": 0, "expected": 0,
                                         "zero_payload_answers": 2})
        self.assertEqual(pairs[(2, 0)]["bytes"], 2 * PER_HEAD)

        def silent(prefill, decode, ports):
            records = [json.loads(text) for text in prefill[1].splitlines()]
            for record in records:
                record.get("kv_transfers", {}).get("send", {}).pop("zero_payload", None)
            return [prefill[0], "".join(json.dumps(r) + "\n" for r in records), *prefill[2:]], decode, ports

        problems = self.verdict(8, 2, 4, False, silent)["problems"]
        self.assertIn("prefill rank 1 never answered decode rank 0 (no payload and no zero-payload answer) for 2 "
                      "request(s), cmpl-a-0", problems)

    def test_failures_are_reported(self) -> None:
        def edit(role, rank, old, new):
            def change(prefill, decode, ports):
                logs = {"prefill": list(prefill), "decode": list(decode)}
                self.assertIn(old, logs[role][rank])
                logs[role][rank] = logs[role][rank].replace(old, new)
                return logs["prefill"], logs["decode"], ports
            return change

        def no_ports(prefill, decode, ports):
            return prefill, decode, None

        def short_port(prefill, decode, ports):
            ports["kv"]["p0"]["rx"] = 10
            return prefill, decode, ports

        cases = {
            # Decode rank 1 holding head 0's bytes, not its own.
            "decode rank 1 checked cmpl-a-0-d's heads [0]": (2, 4, 8, False, edit(
                "decode", 1, '"global_heads": [2, 3]}]}', '"global_heads": [0]}]}')),
            "decode rank 1's group 0 holds head-v1 heads [0, 1], not head-v1 heads [2, 3]": (2, 4, 8, False, edit(
                "decode", 1, '"global_heads": [2, 3]}]}}', '"global_heads": [0, 1]}]}}')),
            "prefill rank 0 -> decode rank 1 moved 400 bytes, not the 400": None,
            "plan [True, 0, 0, 20], not [True, 20, 0, 20]": (2, 4, 8, False, edit(
                "prefill", 0, "[[true, 20, 0, 20]]", "[[true, 0, 0, 20]]")),
            "a READ into slot 0, not 1": (4, 2, 8, True, edit(
                "decode", 0, "10.0, 1, [1, 10, 10]], [200", "10.0, 1, [0, 10, 10]], [200")),
            "transfer(s) logged without their peer rank": (2, 4, 8, True, edit(
                "decode", 0, "10.0, 0, [0, 10, 10]]", "10.0]")),
            "a READ of 2000 selected descriptors, more than the 1024 guard": (2, 4, 8, True, edit(
                "decode", 0, "0.0, 10, 10.0, 0, [0, 10, 10]", "0.0, 2000, 10.0, 0, [0, 2000, 2000]")),
            "prefill rank 0's router port RX 10 bytes": (2, 4, 8, False, short_port),
            "no router port counters": (2, 4, 8, False, no_ports),
        }
        cases.pop("prefill rank 0 -> decode rank 1 moved 400 bytes, not the 400")
        for message, (p, d, k, nixl, change) in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(p, d, k, nixl, change)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)
        # A pair moving other bytes than the heads it holds.
        problems = self.verdict(2, 4, 8, False, edit("prefill", 0, "[200, 0.01", "[150, 0.01"))["problems"]
        self.assertTrue(any("prefill rank 0 -> decode rank 0 moved" in p and "not the 400" in p
                            for p in problems), problems)

    @staticmethod
    def rewrite(role, rank, change):
        """A change of one rank's step log, record by record."""
        def apply(prefill, decode, ports):
            logs = {"prefill": list(prefill), "decode": list(decode)}
            records = [json.loads(text) for text in logs[role][rank].splitlines()]
            logs[role][rank] = "".join(json.dumps(record) + "\n" for record in map(change, records))
            return logs["prefill"], logs["decode"], ports
        return apply

    @staticmethod
    def each_transfer(direction, change):
        """`change(entry)` of each transfer a record logs (its new entry)."""
        def apply(record):
            transfer = (record.get("kv_transfers") or {}).get(direction)
            if transfer:
                transfer["each"] = [change(list(entry)) for entry in transfer["each"]]
            return record
        return apply

    @staticmethod
    def answers(change):
        """`change(answers)` of a prefill rank's zero-payload answers."""
        def apply(record):
            send = (record.get("kv_transfers") or {}).get("send")
            if send and "zero_payload" in send:
                send["zero_payload"] = change(send["zero_payload"])
            return record
        return apply

    def test_an_mla_latent_reaches_every_decode_rank_whole(self) -> None:
        """MLA: one latent, the same on every rank. Each decode rank receives
        it whole from one producer (decode rank d's first paired one), so the
        payload is decode TP times the unique KV: 8 at 4 -> 8, 4 at 8 -> 4.
        Mooncake's other paired producers (8 -> 4: 1, 3, 5, 7) answer without
        a payload; NIXL reads only from the first."""
        for sizes, factor in (((4, 8), 8.0), ((8, 4), 4.0)):
            for nixl in (False, True):
                with self.subTest(sizes=sizes, nixl=nixl):
                    verdict = self.verdict(*sizes, 1, nixl)
                    self.assertEqual(verdict["problems"], [])
                    hetero = verdict["hetero_tp"]
                    self.assertEqual(hetero["kv_heads"], 1)
                    self.assertEqual(hetero["replication_factor"], factor)
                    self.assertEqual(hetero["payload_bytes"], factor * hetero["unique_kv_bytes"])
                    p, d = sizes
                    if nixl or p < d:
                        self.assertEqual(hetero["sources"], {r: [r * p // d] for r in range(d)})
                    else:
                        self.assertEqual(hetero["sources"], {r: [2 * r, 2 * r + 1] for r in range(d)})
                    self.assertEqual(hetero["idle_replicas"], [1, 3, 5, 7] if (p, d) == (8, 4) and not nixl
                                     else [])
                    self.assertTrue(all(pair["bytes"] == pair["expected"] for pair in hetero["pairs"]))
        # A Mooncake replica's missing answer, as with attention heads.
        problems = self.verdict(8, 4, 1, False, self.rewrite("prefill", 3, self.answers(lambda answers: [])))[
            "problems"]
        self.assertTrue(any("prefill rank 3 never answered decode rank 1" in problem for problem in problems),
                        problems)

    def test_idle_replicas_are_accepted_only_on_their_answers(self) -> None:
        """Mooncake 8 -> 2, 4 KV heads: ranks 1, 3, 5, 7 never write, so their
        TENT scrapes serve no write series (as `_tent_delta` reads them). That
        is accepted only for a replica answering each of its decode ranks'
        requests once, with a healthy endpoint and registration."""
        verdict = self.verdict(8, 2, 4, False)
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["hetero_tp"]["idle_replicas"], [1, 3, 5, 7])
        self.assertEqual(verdict["hetero_tp"]["unwritten_accepted"], [1, 3, 5, 7])
        for rank in (1, 3, 5, 7):
            self.assertNotIn("tent_write_bytes_total", verdict["dp_ranks"]["prefill"][rank]["tent"])
            self.assertIn("tent_write_bytes_total", verdict["dp_ranks"]["prefill"][rank - 1]["tent"])
        lack = "prefill rank 1's TENT metrics lack tent_write_bytes_total, and it is not a replica"

        def problems(change=None, tent_change=None):
            return self.verdict(8, 2, 4, False, change, tent_change)["problems"]

        cases = {
            # Both answers replaced by one counting none (the old record).
            "answered decode rank 0 without a payload for unnamed requests (0)": self.rewrite(
                "prefill", 1, self.answers(lambda answers: [[0, 0, answers[0][2]]])),
            "never answered decode rank 0 (no payload and no zero-payload answer) for 1 request(s), cmpl-b-0":
                self.rewrite("prefill", 1, self.answers(lambda answers: answers[:1])),
            "answered decode rank 0 without a payload twice for 1 request(s), cmpl-a-0": self.rewrite(
                "prefill", 1, self.answers(lambda answers: answers + answers[:1])),
            "prefill rank 1 answered decode rank 99, not one it pairs with": self.rewrite(
                "prefill", 1, self.answers(lambda answers: answers + [[99, ["cmpl-a-0-d"], answers[0][2]]])),
            # Rank 1 pairs with decode rank 0 only.
            "prefill rank 1 answered decode rank 1, not one it pairs with": self.rewrite(
                "prefill", 1, self.answers(lambda answers: answers + [[1, ["cmpl-a-0-d"], answers[0][2]]])),
            "answered decode rank 0 without a payload for requests it never pulled: 1 request(s), cmpl-z-0":
                self.rewrite("prefill", 1, self.answers(
                    lambda answers: answers + [[0, ["cmpl-z-0-d"], answers[0][2]]])),
            "answered decode rank 0 without a payload under a plan that sends": self.rewrite(
                "prefill", 1, self.answers(lambda answers: [[0, a[1], [[True, 0, 0, 10]]] for a in answers])),
            # A sending rank that also answers without a payload.
            "prefill rank 0 answered decode rank 0 without a payload for 1 request(s), though it holds heads":
                self.rewrite("prefill", 0, lambda record: {**record, "kv_transfers": {"send": {
                    **record["kv_transfers"]["send"], "zero_payload": [[0, ["cmpl-a-0-d"], [[False, 0, 0, 10]]]]}}}
                    if "kv_transfers" in record else record),
        }
        for message, change in cases.items():
            with self.subTest(message=message):
                found = problems(change)
                self.assertTrue(any(message in problem for problem in found), found)
                # Rank 1's absent write series is no longer accepted.
                if "prefill rank 0" not in message:
                    self.assertTrue(any(lack in problem for problem in found), found)

        def drop(rank, *names):
            def change(tents):
                for name in names:
                    tents["prefill"][rank].pop(name)
            return change

        def silent(rank):
            def change(tents):
                tents["prefill"][rank] = None
            return change

        for message, tent_change in {
                # An active producer without its write series.
                "prefill rank 0's TENT metrics lack tent_write_bytes_total": drop(0, "tent_write_bytes_total"),
                "rank 1: prefill's TENT metrics did not answer": silent(1),
                "rank 1: prefill's TENT metrics lack tent_registered_buffer_bytes": drop(
                    1, "tent_registered_buffer_bytes")}.items():
            with self.subTest(message=message):
                found = problems(tent_change=tent_change)
                self.assertTrue(any(message in problem for problem in found), found)
        # A missing answer fails the rank's absent write series too; with the
        # series served (as zero), only the answer.
        def zero_write(tents):
            tents["prefill"][1]["tent_write_bytes_total"] = 0.0

        found = problems(self.rewrite("prefill", 1, self.answers(lambda answers: answers[:1])), zero_write)
        self.assertFalse(any(lack in problem for problem in found), found)
        self.assertTrue(any("never answered decode rank 0" in problem for problem in found), found)

    def test_plans_and_descriptors_are_measured_against_the_layouts(self) -> None:
        """A head's bytes in a block of a layer (10 here) come from both sides'
        layouts: each Mooncake region's plan must be its heads' places and
        length in them. Each NIXL READ carries three counts: the descriptors
        vLLM selected (each one block of a layer of its heads, within the
        guard), the hook's prediction of what NIXL merges them into, and what
        NIXL measured; measured = predicted <= selected, and a READ without a
        prediction fails."""
        def problems(sizes, nixl, change):
            return self.verdict(*sizes, 8, nixl, change)["problems"]

        def plans(change):
            return self.rewrite("prefill", 0, self.each_transfer(
                "send", lambda entry: entry[:6] + [change(entry[6])]))

        def reads(change):
            return self.rewrite("decode", 0, self.each_transfer("load", change))

        cases = {
            "a transfer of 200 bytes without its plan ([])": ((2, 4), False, plans(lambda plan: [])),
            "a transfer of 200 bytes without its plan (None)": ((2, 4), False, plans(lambda plan: None)),
            # A zero length while bytes moved.
            "plan [True, 0, 0, 0], not [True, 0, 0, 20]": ((2, 4), False, plans(lambda plan: [[True, 0, 0, 0]])),
            # Every length and offset scaled alike (a 64-byte head).
            "plan [True, 0, 0, 128], not [True, 0, 0, 20]": ((2, 4), False, plans(
                lambda plan: [[s, o // 10 * 64, d // 10 * 64, n // 10 * 64] for s, o, d, n in plan])),
            # A second region of another geometry.
            "plan [True, 0, 0, 10], not [True, 0, 0, 20]": ((2, 4), False, plans(
                lambda plan: plan + [[True, 0, 0, 10]])),
            "a READ without NIXL's measured descriptors (None; 10 selected)": ((2, 4), True, reads(
                lambda entry: entry[:3] + [None] + entry[4:])),
            # The hook's count alone, as before the prediction: not accepted.
            "a READ without the hook's predicted merged descriptors (None; 10 selected, NIXL measured 10)": (
                (2, 4), True, reads(lambda entry: entry[:6] + [entry[6][:2]])),
            "a READ without its selected descriptors (None)": ((2, 4), True, reads(
                lambda entry: entry[:6] + [entry[6][:1]])),
            "a READ without the hook's predicted merged descriptors (0;": ((2, 4), True, reads(
                lambda entry: entry[:6] + [[entry[6][0], 10, 0]])),
            # A wrong prediction, and a merge predicted that NIXL did not make.
            "a READ NIXL measured at 10 descriptors, predicted at 9 (10 selected)": ((2, 4), True, reads(
                lambda entry: entry[:6] + [[entry[6][0], 10, 9]])),
            "a READ NIXL measured at 10 descriptors, predicted at 2 (10 selected)": ((2, 4), True, reads(
                lambda entry: entry[:6] + [[entry[6][0], 10, 2]])),
            # More descriptors than vLLM selected, measured and predicted alike.
            "a READ NIXL measured at 12 descriptors, more than its 10 selected": ((2, 4), True, reads(
                lambda entry: entry[:3] + [12] + entry[4:6] + [[entry[6][0], 10, 12]])),
            "a READ predicted at 12 merged descriptors, more than its 10 selected": ((2, 4), True, reads(
                lambda entry: entry[:3] + [12] + entry[4:6] + [[entry[6][0], 10, 12]])),
            "prefill rank 0 -> decode rank 0: a transfer of 200 bytes without its plan (None)": (
                (4, 2), True, reads(lambda entry: entry[:6] + [None])),
            # Bytes that are not its descriptors' blocks of its heads.
            "a READ of 200 bytes in 5 selected descriptors, not 2 head(s) of 10 bytes each": ((2, 4), True, reads(
                lambda entry: entry[:3] + [5] + entry[4:6] + [[entry[6][0], 5, 5]])),
            # A layout whose heads' bytes disagree with the other side's.
            "the ranks' layouts give no single head size": ((2, 4), False, self.rewrite(
                "decode", 1, lambda record: {"kv_layout": {**record["kv_layout"], "groups": [
                    {**record["kv_layout"]["groups"][0], "real_page_bytes": 64}]}}
                if "kv_layout" in record else record)),
        }
        for message, (sizes, nixl, change) in cases.items():
            with self.subTest(message=message):
                found = problems(sizes, nixl, change)
                self.assertTrue(any(message in problem for problem in found), found)
        # Split, gather and replicated heads, both connectors, pass as logged.
        for sizes, kv_heads, nixl in (((2, 4), 8, False), ((2, 4), 8, True), ((4, 2), 8, False),
                                      ((4, 2), 8, True), ((2, 8), 4, False), ((2, 8), 4, True),
                                      ((8, 2), 4, False)):
            with self.subTest(sizes=sizes, nixl=nixl):
                verdict = self.verdict(*sizes, kv_heads, nixl)
                self.assertEqual(verdict["problems"], [])
                self.assertEqual(verdict["hetero_tp"]["slab_bytes"], SLAB)
                if nixl:
                    self.assertEqual((verdict["hetero_tp"]["max_descriptors"],
                                      verdict["hetero_tp"]["max_merged_descriptors"]), (10, 10))
        # Decode rank 0's whole-page READs, whose 10 selected descriptors NIXL
        # merged into 2, as predicted: their bytes are still the selected ones'.
        for sizes in ((2, 4), (4, 2)):
            with self.subTest(sizes=sizes, merged=True):
                verdict = self.verdict(*sizes, 8, True, reads(
                    lambda entry: entry[:3] + [2] + entry[4:6] + [[entry[6][0], 10, 2]]))
                self.assertEqual(verdict["problems"], [])
                self.assertEqual(verdict["hetero_tp"]["max_descriptors"], 10)

    def test_nixl_descriptors_follow_the_kernel_blocks_of_a_split_layout(self) -> None:
        """A layout that splits each logical block into kernel blocks has NIXL post a
        descriptor for each: the carried heads' bytes in a kernel block (here half a
        logical block's, 5 per head), which the regions' registered block length must
        confirm. The same READs against a layout without the split, or with another
        registered length, are rejected."""
        sizes = (2, 4)

        def split_layouts(lengths=lambda page: page // 2, blocks=2):
            def change(record):
                layout = record.get("kv_layout")
                if layout:
                    page = layout["groups"][0]["real_page_bytes"]
                    layout["kernel_geometry"] = {"kernel_blocks_per_logical": blocks,
                                                 "registered_region_block_lengths": [lengths(page)] * 2}
                return record

            def apply(prefill, decode, ports):
                for logs in (prefill, decode):
                    for rank, text in enumerate(logs):
                        logs[rank] = "".join(json.dumps(change(json.loads(line))) + "\n"
                                             for line in text.splitlines())
                return prefill, decode, ports
            return apply

        def halved(prefill, decode, ports):
            for rank in range(len(decode)):
                prefill, decode, ports = self.rewrite("decode", rank, self.each_transfer(
                    "load", lambda entry: [entry[0] // 2] + entry[1:]))(prefill, decode, ports)
            return prefill, decode, ports

        def chained(*changes):
            def apply(prefill, decode, ports):
                for change in changes:
                    prefill, decode, ports = change(prefill, decode, ports)
                return prefill, decode, ports
            return apply

        def geometry_problems(change):
            found = self.verdict(*sizes, 8, True, change)["problems"]
            return [problem for problem in found if "head(s) of" in problem or "registered block lengths" in problem]

        self.assertEqual(geometry_problems(chained(split_layouts(), halved)), [])
        unsplit = geometry_problems(halved)
        self.assertTrue(any("not 2 head(s) of 10 bytes each" in problem for problem in unsplit), unsplit)
        whole = geometry_problems(split_layouts())
        self.assertTrue(any("not 2 head(s) of 5 bytes each" in problem for problem in whole), whole)
        wrong = geometry_problems(chained(split_layouts(lambda page: page), halved))
        self.assertTrue(any("registered block lengths [20]" in problem or "registered block lengths [10]" in problem
                            for problem in wrong), wrong)

    def test_registered_lengths_cover_every_group_of_a_split_layout(self) -> None:
        """A layout's registered block lengths describe all its groups' regions: with sliding and full
        groups of different row lengths (10 and 20 bytes of head slab) split into two kernel rows each,
        the lengths are the kernel blocks of both groups together. The descriptors' bytes (doubled
        counts, same bytes) are accepted; a missing group's length, an extra one, and whole-page
        descriptor bytes are rejected."""
        groups = [("SlidingWindowSpec", 8, 10), ("FullAttentionSpec", 2, 20)]

        def split(lengths=lambda layout: [g["real_page_bytes"] // 2 for g in layout["groups"]
                                          for _ in range(g["layers"])], whole=False):
            def apply(prefill, decode, ports):
                for logs in (prefill, decode):
                    for rank, text in enumerate(logs):
                        records = [json.loads(line) for line in text.splitlines()]
                        layout = records[0]["kv_layout"]
                        layout["kernel_geometry"] = {"kernel_blocks_per_logical": 2,
                                                     "registered_region_block_lengths": lengths(layout)}
                        for record in records:
                            for entry in (record.get("kv_transfers") or {}).get("load", {}).get("each", []):
                                if not whole:
                                    entry[3] *= 2
                                    entry[6][1] *= 2
                                    entry[6][2] *= 2
                        logs[rank] = "".join(json.dumps(record) + "\n" for record in records)
                return prefill, decode, ports
            return apply

        def problems(change):
            return self.reconcile(2, 4, 8, True, change, groups=groups, global_heads=2)["problems"]

        self.assertEqual(problems(None), [])
        self.assertEqual(problems(split()), [])
        only_full = problems(split(lambda layout: [layout["groups"][1]["real_page_bytes"] // 2] * 2))
        self.assertTrue(any("registered block lengths [10] are not its groups' kernel blocks' [10, 20]" in p for p in only_full), only_full)
        extra = problems(split(lambda layout: [g["real_page_bytes"] // 2 for g in layout["groups"]] + [7]))
        self.assertTrue(any("registered block lengths [7, 10, 20] are not its groups' kernel blocks'" in p for p in extra), extra)
        whole = problems(split(whole=True))
        self.assertTrue(any("selected descriptors, not" in p for p in whole), whole)

    def test_unequal_sizes_need_one_switched_network(self) -> None:
        base = (E2E / "runs" / "example.yaml").read_text()
        for prefill, decode, error in (
                ("{node: prefill0, kv_network: data0, tensor_parallel_size: 2}",
                 "{node: decode0, kv_network: data0, tensor_parallel_size: 3}", "divide one another"),
                ("{node: prefill0, kv_network: [data0, data1], tensor_parallel_size: 2}",
                 "{node: decode0, kv_network: data0, tensor_parallel_size: 4}", "one switched network"),
                ("{node: prefill0, kv_network: data0, tensor_parallel_size: 0}",
                 "{node: decode0, kv_network: data0}", "positive integer")):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                spec = Path(directory) / "run.yaml"
                text = re.sub(r"^prefill:.*$", f"prefill: {prefill}", base, count=1, flags=re.M)
                spec.write_text(re.sub(r"^decode:.*$", f"decode: {decode}", text, count=1, flags=re.M))
                with self.assertRaisesRegex(ValueError, error):
                    pd.load(spec)

    def test_unequal_sizes_pair_inside_data_parallelism(self) -> None:
        """TP inside DP at unequal sizes: each side's DP ranks run its own TP
        size, TP rank r of every DP rank on link r of the switched network."""
        base = (E2E / "runs" / "example.yaml").read_text()
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / "run.yaml"
            text = re.sub(r"^prefill:.*$", "prefill: {node: prefill0, kv_network: kv, tensor_parallel_size: 2, "
                          "data_parallel_size: 2}", base, count=1, flags=re.M)
            spec.write_text(re.sub(r"^decode:.*$", "decode: {node: decode0, kv_network: kv, "
                                   "tensor_parallel_size: 4, data_parallel_size: 2}", text, count=1, flags=re.M))
            with mock.patch.object(pd.models, "resolve", return_value=MOE_MODEL):
                run = pd.load(spec)
        self.assertEqual((run.prefill.tp(run.server), run.decode.tp(run.server)), (2, 4))
        addresses = ["10.102.0.1", "10.102.0.3", "10.102.0.5", "10.102.0.7"]
        command = pd.server_command(run, "decode", addresses)
        flag = lambda name: command[command.index(name) + 1]
        self.assertEqual((flag("--tensor-parallel-size"), flag("--data-parallel-size")), ("4", "2"))
        kv_transfer = json.loads(flag("--kv-transfer-config"))
        self.assertEqual(kv_transfer["kv_connector_extra_config"]["rank_addresses"], addresses * 2)

    def test_rank_r_takes_link_r_of_a_switched_network(self) -> None:
        interfaces = {"control": {"address": "10.250.0.1", "network": "control", "slot": 0},
                      **{f"kvr{k}": {"address": f"10.102.0.{2 * k + 1}", "network": "kv", "slot": k}
                         for k in range(4)}}
        self.assertEqual(pd._kv_addresses(interfaces, pd.Side("p", "kv"), 4),
                         ["10.102.0.1", "10.102.0.3", "10.102.0.5", "10.102.0.7"])
        with self.assertRaisesRegex(ValueError, "4 links on kv, not one per rank"):
            pd._kv_addresses(interfaces, pd.Side("p", "kv"), 8)
        # A single link (or an older topology's state) serves every rank.
        self.assertEqual(pd._kv_addresses({"data0": {"address": "10.100.0.1"}}, pd.Side("p", "data0"), 2),
                         ["10.100.0.1", "10.100.0.1"])
        self.assertEqual(pd.global_heads(3, 8, 4), [1])
        self.assertEqual(pd.global_heads(1, 2, 8), [4, 5, 6, 7])


# Requests spread over both DP ranks of each side, each decode DP rank
# receiving from both producer DP ranks.
SPREAD = {"a": (0, 0), "b": (1, 1), "c": (0, 1), "d": (1, 0)}


class HeterogeneousTpDataParallelTests(unittest.TestCase):
    """Unequal TP sizes inside data parallelism: each DP rank's ranks pair
    with the producing engine's as at DP1, every rank of both sides audited."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.model = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def run_for(self, sizes, kv_heads, nixl, dps, requests):
        config = {"num_key_value_heads": 128, "kv_lora_rank": 512} if kv_heads == 1 else {
            "num_key_value_heads": kv_heads}
        (self.model / "config.json").write_text(json.dumps(config))
        return pd.Run(**{**RUN.__dict__, "model": self.model,
                         "prefill": pd.Side("p", "kv", tensor_parallel_size=sizes[0], data_parallel_size=dps[0]),
                         "decode": pd.Side("d", "kv", tensor_parallel_size=sizes[1], data_parallel_size=dps[1]),
                         "server": pd.Server(connector="nixl" if nixl else "mooncake"),
                         "workload": pd.Workload(num_prompts=requests)})

    def verdict(self, sizes, kv_heads, nixl, owners=None, dps=(2, 2), change=None) -> dict:
        owners = owners or {r: tuple(dp if n > 1 else 0 for dp, n in zip(o, dps)) for r, o in SPREAD.items()}
        prefill, decode, ports, moved, tents = dp_logs(
            lambda r: hetero_logs(*sizes, kv_heads, nixl, requests=(r,)), owners,
            {"prefill": sizes[0], "decode": sizes[1]}, dps, nixl)
        if change:
            prefill, decode, ports = change(prefill, decode, ports)
        return pd.check({"completed": len(owners), "errors": []}, prefill, decode,
                        {"control": 1, "kv": moved + 500},
                        self.run_for(sizes, kv_heads, nixl, dps, len(owners)), tents, ports)

    @staticmethod
    def at(sizes, dps, role, dp, tp):
        """A rank's index in its side's log list (`Side.ranks` order)."""
        return dp * sizes[0 if role == "prefill" else 1] + tp

    def test_heads_resharded_either_way_inside_dp_over_both_connectors(self) -> None:
        for sizes, kv_heads in (((4, 2), 4), ((2, 4), 4), ((4, 2), 2), ((4, 2), 1)):
            for nixl in (True, False):
                for dps in ((2, 2), (1, 2), (2, 1)):
                    with self.subTest(sizes=sizes, kv_heads=kv_heads, nixl=nixl, dps=dps):
                        verdict = self.verdict(sizes, kv_heads, nixl, dps=dps)
                        self.assertEqual(verdict["problems"], [])
                        hetero = verdict["hetero_tp"]
                        self.assertEqual(sum(verdict["dp_pairs"].values()), 4)
                        self.assertEqual(len(verdict["dp_pairs"]), dps[0] * dps[1])
                        self.assertTrue(all(pair["bytes"] == pair["expected"] for pair in hetero["pairs"]))
                        if nixl and dps[1] > 1:
                            self.assertEqual({pair["decode_dp"] for pair in hetero["pairs"]}, {0, 1})
        # NIXL: each decode DP rank's READs, as at DP1.
        hetero = self.verdict((4, 2), 4, True)["hetero_tp"]
        self.assertEqual(hetero["sources"], {0: [0, 1], 1: [2, 3]})
        self.assertEqual(self.verdict((4, 2), 4, True)["dp_pairs"], {"0->0": 1, "1->1": 1, "0->1": 1, "1->0": 1})

    def test_every_producer_rank_has_one_owner_of_each_request(self) -> None:
        """Both requests on DP rank 0, producer rank 1 of DP rank 1 also
        allocating them. At 4 -> 2 the source map names producer ranks 0 and
        2 first: ownership is checked on every producer and consumer rank,
        not only those."""
        owners = {"a": (0, 0), "b": (0, 0)}
        sizes, dps = (4, 2), (2, 2)

        def duplicate(prefill, decode, ports):
            prefill = list(prefill)
            prefill[self.at(sizes, dps, "prefill", 1, 1)] += allocated("a", "b")
            return prefill, decode, ports

        self.assertEqual(self.verdict(sizes, 4, True, owners)["problems"], [])
        problems = self.verdict(sizes, 4, True, owners, change=duplicate)["problems"]
        self.assertIn("prefill rank 1: 2 request(s) more than one of the prefill DP ranks produced: cmpl-a-0",
                      problems)
        self.assertIn("prefill rank 1: 2 request(s) with other prefill DP ranks than rank 0's: cmpl-a-0", problems)

        def swap(prefill, decode, ports):
            # Request a's allocation on producer rank 3 moved to DP rank 1.
            prefill = list(prefill)
            mine, other = self.at(sizes, dps, "prefill", 0, 3), self.at(sizes, dps, "prefill", 1, 3)
            self.assertIn(allocated("a"), prefill[mine])
            prefill[mine] = prefill[mine].replace(allocated("a"), "", 1)
            prefill[other] += allocated("a")
            return prefill, decode, ports

        problems = self.verdict(sizes, 4, True, owners, change=swap)["problems"]
        self.assertIn("prefill rank 3: 1 request(s) with other prefill DP ranks than rank 0's: cmpl-a-0", problems)

        def swap_consumer(prefill, decode, ports):
            # Request a's byte check on decode rank 1 under DP rank 1 instead.
            decode = list(decode)
            mine, other = self.at(sizes, dps, "decode", 0, 1), self.at(sizes, dps, "decode", 1, 1)
            records = pd._records(decode[mine])
            moved = [r for r in records if (r.get("kv_check") or {}).get("req_id") == "cmpl-a-0-d"]
            decode[mine] = "".join(line(**r) for r in records if r not in moved)
            decode[other] += "".join(line(**r) for r in moved)
            return prefill, decode, ports

        problems = self.verdict(sizes, 4, True, owners, change=swap_consumer)["problems"]
        self.assertIn("decode rank 1: 1 request(s) with other decode DP ranks than rank 0's: cmpl-a-0", problems)

    def test_nixl_bytes_are_checked_per_decode_dp_rank(self) -> None:
        """Compensating byte errors on decode DP ranks 0 and 1 leave the sum
        unchanged: each DP rank's READs are compared on their own."""
        sizes, dps = (4, 2), (2, 2)

        def compensate(prefill, decode, ports):
            decode = list(decode)
            for dp, delta in ((0, 10), (1, -10)):
                index = self.at(sizes, dps, "decode", dp, 0)
                records = pd._records(decode[index])
                for record in records:
                    load = (record.get("kv_transfers") or {}).get("load")
                    if load and load["each"]:
                        load["each"][0][0] += delta
                        break
                decode[index] = "".join(line(**r) for r in records)
            return prefill, decode, ports

        problems = self.verdict(sizes, 4, True, change=compensate)["problems"]
        for dp in (0, 1):
            self.assertTrue(any(f"prefill rank 0 -> decode DP rank {dp} rank 0 moved" in p for p in problems),
                            problems)

    def test_every_dp_ranks_layout_holds_its_heads(self) -> None:
        sizes, dps = (2, 4), (2, 2)

        def heads(prefill, decode, ports):
            decode = list(decode)
            index = self.at(sizes, dps, "decode", 1, 1)
            records = pd._records(decode[index])
            records[0]["kv_layout"]["groups"][0]["global_heads"] = [0]
            decode[index] = "".join(line(**r) for r in records)
            return prefill, decode, ports

        problems = self.verdict(sizes, 4, True, change=heads)["problems"]
        self.assertTrue(any("decode DP rank 1 rank 1's group 0 holds head-v1 heads [0], not head-v1 heads [1]" in p
                            for p in problems), problems)

    def test_a_link_carries_its_rank_of_every_dp_rank(self) -> None:
        def dp0_only(prefill, decode, ports):
            port = ports["kv"]["p0"]
            return prefill, decode, {"kv": {**ports["kv"], "p0": {**port, "rx": port["rx"] // 2}}}

        problems = self.verdict((4, 2), 4, True, change=dp0_only)["problems"]
        self.assertTrue(any("prefill rank 0's router port RX" in p for p in problems), problems)

    def test_a_producer_rank_lacking_a_request_fails(self) -> None:
        sizes, dps = (4, 2), (2, 2)

        def drop(prefill, decode, ports):
            prefill = list(prefill)
            index = self.at(sizes, dps, "prefill", 1, 2)
            self.assertIn(allocated("b"), prefill[index])
            prefill[index] = prefill[index].replace(allocated("b"), "", 1)
            return prefill, decode, ports

        problems = self.verdict(sizes, 4, True, change=drop)["problems"]
        self.assertIn("prefill rank 2 produced 3 request(s), not 4", problems)

    def test_mooncake_replicas_answer_for_their_own_engines_requests(self) -> None:
        """Mooncake 4 -> 2, 2 KV heads: producer ranks 1 and 3 of each DP
        rank hold heads ranks 0 and 2 send, and answer without a payload for
        the requests their own engine produced, once each."""
        sizes, dps = (4, 2), (2, 2)
        verdict = self.verdict(sizes, 2, False)
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["hetero_tp"]["idle_replicas"], [[0, 1], [0, 3], [1, 1], [1, 3]])
        self.assertEqual(verdict["hetero_tp"]["unwritten_accepted"], [[0, 1], [0, 3], [1, 1], [1, 3]])

        def answers(rank, edit):
            """Edit the zero-payload answers of prefill (DP rank, rank) ``rank``,
            all of them at once (the edited list logged in its first record)."""
            def change(prefill, decode, ports):
                prefill = list(prefill)
                index = self.at(sizes, dps, "prefill", *rank)
                records = pd._records(prefill[index])
                sends = [(r.get("kv_transfers") or {}).get("send") or {} for r in records]
                sends = [send for send in sends if send.get("zero_payload")]
                edited = edit([answer for send in sends for answer in send["zero_payload"]])
                for position, send in enumerate(sends):
                    send["zero_payload"] = edited if position == 0 else []
                prefill[index] = "".join(line(**r) for r in records)
                return prefill, decode, ports
            return change

        lack = "prefill DP rank 0 rank 1's TENT metrics lack tent_write_bytes_total, and it is not a replica"
        # DP rank 0's rank 1 produced a and c (SPREAD): an answer for b, DP
        # rank 1's, does not prove it idle, nor does a duplicate or a
        # missing one.
        cases = {
            "prefill DP rank 0 rank 1 answered decode rank 0 without a payload for requests another DP rank "
            "produced: 1 request(s), cmpl-b-0": answers((0, 1), lambda z: z + [[0, ["cmpl-b-0-d"], z[0][2]]]),
            "prefill DP rank 0 rank 1 answered decode rank 0 without a payload twice for 1 request(s), cmpl-a-0":
                answers((0, 1), lambda z: z + z[:1]),
            "prefill DP rank 0 rank 1 never answered decode rank 0 (no payload and no zero-payload answer) for "
            "1 request(s), cmpl-a-0": answers((0, 1), lambda z: [a for a in z if a[1] != ["cmpl-a-0-d"]]),
        }
        for message, change in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(sizes, 2, False, change=change)["problems"]
                self.assertIn(message, problems)
                self.assertTrue(any(lack in p for p in problems), problems)

        def moved(prefill, decode, ports):
            # Request a's answer given by DP rank 1's rank 1, not its own engine's.
            prefill, decode, ports = answers((0, 1), lambda z: [a for a in z if a[1] != ["cmpl-a-0-d"]])(
                prefill, decode, ports)
            return answers((1, 1), lambda z: z + [[0, ["cmpl-a-0-d"], z[0][2]]])(prefill, decode, ports)

        problems = self.verdict(sizes, 2, False, change=moved)["problems"]
        self.assertIn("prefill DP rank 1 rank 1 answered decode rank 0 without a payload for requests another DP "
                      "rank produced: 1 request(s), cmpl-a-0", problems)
        self.assertIn("prefill DP rank 0 rank 1 never answered decode rank 0 (no payload and no zero-payload "
                      "answer) for 1 request(s), cmpl-a-0", problems)

    def test_an_idle_dp_engine_is_apart_from_an_idle_replica(self) -> None:
        """Every request on prefill DP rank 0: DP rank 1's engine, given none,
        moved nothing on any rank (an idle engine, in `idle_ranks`), while
        DP rank 0's ranks 1 and 3 are its replicas, proven by their answers."""
        owners = {"a": (0, 0), "b": (0, 1)}
        verdict = self.verdict((4, 2), 2, False, owners)
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["hetero_tp"]["idle_replicas"], [[0, 1], [0, 3]])
        self.assertEqual(sorted(verdict["idle_ranks"]),
                         [f"prefill DP rank 1 rank {r}" for r in range(4)])
        self.assertEqual(verdict["dp_pairs"], {"0->0": 1, "0->1": 1})


class HybridHeterogeneousTpDataParallelTests(unittest.TestCase):
    """A hybrid's unequal TP sizes inside DP over NIXL: each decode DP rank's
    state and attention READs reconciled on their own."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.model = Path(self.directory.name)
        (self.model / "config.json").write_text(json.dumps({"num_key_value_heads": 4}))

    def tearDown(self) -> None:
        self.directory.cleanup()

    def reconcile(self, sizes, owners, change=None) -> dict:
        dps = (2, 2)
        prefill, decode, ports, _, _ = dp_logs(lambda r: hybrid_logs(*sizes, requests=(r,)), owners,
                                               {"prefill": sizes[0], "decode": sizes[1]}, dps)
        if change:
            prefill, decode, ports = change(prefill, decode, ports)
        run = pd.Run(**{**RUN.__dict__, "model": self.model,
                        "prefill": pd.Side("p", "kv", tensor_parallel_size=sizes[0], data_parallel_size=2),
                        "decode": pd.Side("d", "kv", tensor_parallel_size=sizes[1], data_parallel_size=2),
                        "server": pd.Server(connector="nixl")})
        logs = {"prefill": dict(zip(run.prefill.ranks(sizes[0]), map(pd._records, prefill))),
                "decode": dict(zip(run.decode.ranks(sizes[1]), map(pd._records, decode)))}
        moved = lambda records: sum(((r.get("kv_transfers") or {}).get("load") or {}).get("bytes", 0) for r in records)
        produced = lambda records: {pd._base_request_id(request) for r in records
                                    for request in ((r.get("kv_events") or {}).get("allocated") or {})}
        ranks = {role: {key: {"transfers": {"bytes": moved(records)}, "requests": produced(records)}
                        for key, records in logs[role].items()} for role in logs}
        return pd._hetero_tp(run, {"prefill": sizes[0], "decode": sizes[1]}, logs, ranks, ports)

    def test_state_and_attention_pair_inside_dp(self) -> None:
        for sizes in ((4, 2), (2, 4)):
            for owners in ({"a": (0, 0), "b": (1, 1)}, {"a": (0, 1), "b": (1, 0)}):
                with self.subTest(sizes=sizes, owners=owners):
                    hetero = self.reconcile(sizes, owners)
                    self.assertEqual(hetero["problems"], [])
                    self.assertEqual({pair["decode_dp"] for pair in hetero["pairs"]}, {0, 1})

    def test_a_dp_ranks_state_spans_must_be_the_layouts(self) -> None:
        sizes = (4, 2)

        def spans(prefill, decode, ports):
            decode = list(decode)
            index = sizes[1] + 0  # decode DP rank 1, rank 0
            records = pd._records(decode[index])
            records[0]["kv_layout"]["groups"][1]["state"]["spans"][0][3] += 2
            decode[index] = "".join(line(**r) for r in records)
            return prefill, decode, ports

        problems = self.reconcile(sizes, {"a": (0, 0), "b": (1, 1)}, change=spans)["problems"]
        self.assertTrue(any(p.startswith("decode DP rank 1 rank 0's group 1 state spans") for p in problems),
                        problems)

    def test_a_dp_ranks_read_is_its_own(self) -> None:
        """A READ logged by decode DP rank 1 under DP rank 0 instead: each DP
        rank's checked READs must be its logged ones."""
        sizes = (4, 2)

        def move(prefill, decode, ports):
            decode = list(decode)
            src, dst = sizes[1] + 0, 0
            records = pd._records(decode[src])
            loads = [r for r in records if (r.get("kv_transfers") or {}).get("load")]
            decode[src] = "".join(line(**r) for r in records if r not in loads)
            decode[dst] += "".join(line(**r) for r in loads)
            return prefill, decode, ports

        problems = self.reconcile(sizes, {"a": (0, 0), "b": (1, 1)}, change=move)["problems"]
        self.assertTrue(any("decode DP rank 1 rank 0's checked READs" in p for p in problems), problems)
        self.assertTrue(any("decode DP rank 0 rank 0's checked READs" in p for p in problems), problems)


class NixlTcpBackendTests(SyntheticRouting, unittest.TestCase):
    backend = "ucx"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.model = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def fixture(self, backend=None, change=None):
        backend = backend or self.backend
        factory = HeterogeneousTpTests()
        factory.model = self.model
        run = factory.run_for(2, 4, 8, True)
        run.server.nixl_backend = backend
        prefill, decode, ports, moved, tents = hetero_logs(2, 4, 8, True)
        def identity(text):
            records = pd._records(text)
            for record in records:
                if "kv_layout" in record:
                    record["kv_layout"]["nixl_transport"] = {
                        "backends": ["UCX"], "num_threads": 4, "progress_thread": True, "binding": {"address": "10.1.0.1", "interface": "kv0",
                                                            "UCX_NET_DEVICES": "kv0", "UCX_TLS": "tcp,self"}}
                    if backend == "libfabric":
                        record["kv_layout"]["nixl_transport"] = {
                            "backends": ["LIBFABRIC"], "num_threads": 4, "progress_thread": True,
                            "thread_policy_source": "agent_config", "native_provider_observed": False,
                            "native_threads_observed": False,
                            "backend_options": {"num_threads": "4"},
                            "plugin_defaults": {"num_threads": "0", "split_batch_size": "1024"},
                            "binding": {"address": "10.1.0.1", "interface": "kv0", "FI_PROVIDER": "tcp",
                                        "FI_TCP_IFACE": "kv0", "FI_ADDR_FORMAT": "FI_SOCKADDR_IN",
                                        "FI_PREFER_SYSCONFIG": "0", "config_policy": "absent_or_empty",
                                        "config_file": "/opt/libfabric/etc/libfabric.conf"}}
            return "".join(line(**record) for record in records)
        if backend in ("ucx", "libfabric"):
            prefill, decode = [identity(t) for t in prefill], [identity(t) for t in decode]
        if change:
            prefill, decode, ports = change(prefill, decode, ports)
        return run, prefill, decode, ports, moved, tents

    def verdict(self, backend=None, change=None):
        run, p, d, ports, moved, tents = self.fixture(backend, change)
        return pd.check({"completed": 2, "errors": []}, p, d, {"control": 50, "kv": moved + 500},
                        run, tents if pd.uses_tent(run.server) else None, ports)

    def test_tcp_full_accounting_and_missing_identity(self):
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["transport"], {"backend": pd.nixl_backend(pd.Server(nixl_backend=self.backend)), "tent_applicable": False})
        self.assertIsNone(verdict["hp_tcp_task_limit"])
        for find, replace in [('"nixl_transport"', '"missing_identity"'), (json.dumps(pd.nixl_backend(pd.Server(nixl_backend=self.backend))), '"Mooncake"'),
                              (('"tcp,self"' if self.backend == 'ucx' else '"FI_SOCKADDR_IN"'), '"all"'), ('"progress_thread": true', '"progress_thread": false')]:
            problems = self.verdict(change=lambda p,d,ports: (p,[t.replace(find,replace) for t in d],ports))["problems"]
            self.assertTrue(any("backend" in s or "interface evidence" in s or "thread/progress" in s for s in problems), problems)

    def test_tcp_keeps_byte_descriptor_ownership_and_port_failures(self):
        mutations = [
            lambda p,d,ports: (p,[t.replace('"ok": true','"ok": false') for t in d],ports),
            lambda p,d,ports: (p,d,{}),
            lambda p,d,ports: (p,[t.replace('"mismatched_bytes": 0','"mismatched_bytes": 1') for t in d],ports),
        ]
        for mutate in mutations:
            self.assertTrue(self.verdict(change=mutate)["problems"])
        def wrong_count(p,d,ports):
            records=pd._records(d[0])
            for r in records:
                for e in ((r.get("kv_transfers") or {}).get("load") or {}).get("each",[]):
                    e[3]+=1
            return p,["".join(line(**r) for r in records),*d[1:]],ports
        self.assertTrue(any("measured" in s for s in self.verdict(change=wrong_count)["problems"]))
        def wrong_peer(p,d,ports):
            records=pd._records(d[0])
            for r in records:
                for e in ((r.get("kv_transfers") or {}).get("load") or {}).get("each",[]): e[5]=1
            return p,["".join(line(**r) for r in records),*d[1:]],ports
        self.assertTrue(self.verdict(change=wrong_peer)["problems"])

    def test_descriptor_guards_follow_backend_in_both_reconcilers(self):
        from unittest.mock import patch
        with patch.object(pd,"NIXL_MAX_DESCRIPTORS",1):
            self.assertEqual(self.verdict()["problems"], [])
            self.assertTrue(any("guard" in s for s in self.verdict("mooncake")["problems"]))
            hybrid=HybridHeterogeneousTpTests()
            hybrid.setUp()
            try:
                # Exercise the hybrid's third guard with exactly the same logs.
                p,d,ports=hybrid_logs(1,2)
                run=pd.Run(**{**RUN.__dict__,"model":hybrid.model,"server":pd.Server(connector="nixl",nixl_backend=self.backend),
                              "prefill":pd.Side("p","kv",tensor_parallel_size=1),"decode":pd.Side("d","kv",tensor_parallel_size=2)})
                logs={"prefill":{(0,r):pd._records(t) for r,t in enumerate(p)},"decode":{(0,r):pd._records(t) for r,t in enumerate(d)}}
                ranks={role:{key:{"transfers":{"bytes":sum(((x.get("kv_transfers") or {}).get("load") or {}).get("bytes",0) for x in records)},"requests":["a","b"]} for key,records in ls.items()} for role,ls in logs.items()}
                reconcile=lambda:pd._hetero_tp(run,{"prefill":1,"decode":2},logs,ranks,ports)
                self.assertEqual(reconcile()["problems"],[])
                run.server.nixl_backend="mooncake"
                self.assertTrue(any("guard" in s for s in reconcile()["problems"]))
                run.server.nixl_backend=self.backend
                for records in logs["decode"].values():
                    for r in records:
                        if "kv_check" in r:r["kv_check"]["coverage"]["reads"][0]["completed"]=False
                self.assertTrue(any("not complete" in s for s in reconcile()["problems"]))
            finally:hybrid.tearDown()

    def test_command_and_config_restrictions(self):
        run,*_=self.fixture()
        command=pd.server_command(run,"decode",["10.1.0.1","10.1.0.3","10.1.0.5","10.1.0.7"])
        extra=json.loads(command[command.index("--kv-transfer-config")+1])["kv_connector_extra_config"]
        self.assertEqual(extra["backends"],[pd.nixl_backend(run.server)])
        self.assertEqual(len(extra["rank_addresses"]),4)
        env=dict(v.split("=",1) for v in command[1:command.index(pd.VLLM)])
        if self.backend == "ucx":
            self.assertEqual(env["UCX_TLS"],"tcp,self")
        else:
            self.assertEqual(env["FI_PROVIDER"],"tcp")
            self.assertEqual(env["FI_ADDR_FORMAT"],"FI_SOCKADDR_IN")
            self.assertEqual(env["FI_PREFER_SYSCONFIG"],"0")
            self.assertNotIn("FI_TCP_IFACE",env)
        self.assertFalse(any(k.startswith(("MC_","TENT_","NIXL_MOONCAKE_")) for k in env))
        invalid = ({"UCX_TLS":"all"},{"UCX_NET_DEVICES":"control"},{"MC_USE_TENT":"0"},{"UCX_CONFIG_FILE":"x"},{"NIXL_UCX_DEVICE_LIST":"kv0"}) if self.backend == "ucx" else (
            {"FI_PROVIDER":"sockets"},{"FI_TCP_IFACE":"control"},{"FI_PROVIDER_PATH":""},
            {"FI_TCP_PROV_NAME":"other"},{"FI_PREFER_SYSCONFIG":"1"},{"FI_ADDR_FORMAT":"FI_SOCKADDR_IN6"},
            {"FI_TCP_TX_SIZE":"1"},{"NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG":"100"},{"MC_USE_TENT":"0"})
        for overrides in invalid:
            run.server.env=overrides
            with self.assertRaises(ValueError):pd.server_command(run,"decode",["10.1.0.1"]*4)
        run.server.env={};run.server.connector="mooncake"
        with self.assertRaises(ValueError):pd.server_command(run,"decode",["10.1.0.1"]*4)

    def test_tcp_saved_run_needs_no_tent_file_or_fetch(self):
        from unittest.mock import patch
        run,p,d,ports,moved,_=self.fixture()
        run.server.timing="linear"
        out=Path(self.directory.name)/"saved";out.mkdir()
        (out/"run.json").write_text(json.dumps({**asdict(run),"model":str(run.model)},default=str))
        (out/"control-ports.json").write_text(json.dumps({"nodes":["gw"],"bytes":0,"ports":{"p0":{"node":"c","rx":50,"tx":0},"p1":{"node":"p","rx":0,"tx":25},"p2":{"node":"d","rx":0,"tx":25},"p3":{"node":"gw","rx":0,"tx":0}}}))
        for role,texts in (("prefill",p),("decode",d)):
            for rank,text in enumerate(texts):(out/getattr(run,role).step_log(role,0,rank)).write_text(text)
        for file,data in (("bench.json",{"completed":2,"errors":[]}),("routers.json",{"control":50,"kv":moved+500}),("router-ports.json",ports)):
            (out/file).write_text(json.dumps(data))
        self.assertEqual(pd.recheck(out)["problems"],[])
        with patch.object(topology,"exec_in",side_effect=AssertionError("must not fetch TENT")):
            self.assertEqual(pd._tent_metrics(run,{}),{})
        run.server.nixl_backend="mooncake"
        raw=json.loads((out/"run.json").read_text());raw["server"]["nixl_backend"]="mooncake"
        (out/"run.json").write_text(json.dumps(raw))
        with self.assertRaises(FileNotFoundError):pd.recheck(out)

    def test_a_nixl_only_collector_saved_run_rechecks_without_tent_evidence(self):
        """UCX and LIBFABRIC runs save ``transport.json``, never ``tent.json``: the saved-run entry point must not read it for a
        collector that reads NIXL's telemetry only (the collector's own evidence is judged, and found missing here)."""
        run,p,d,ports,moved,_=self.fixture()
        run.server.timing="linear"
        run.collector=pd.Collector(tent=False,nixl_telemetry=True)
        out=Path(self.directory.name)/"saved-nixl"
        out.mkdir()
        (out/"run.json").write_text(json.dumps({**asdict(run),"model":str(run.model)},default=str))
        (out/"control-ports.json").write_text(json.dumps({"nodes":["gw"],"bytes":0,"ports":{"p0":{"node":"c","rx":50,"tx":0},"p1":{"node":"p","rx":0,"tx":25},"p2":{"node":"d","rx":0,"tx":25},"p3":{"node":"gw","rx":0,"tx":0}}}))
        for role,texts in (("prefill",p),("decode",d)):
            for rank,text in enumerate(texts):(out/getattr(run,role).step_log(role,0,rank)).write_text(text)
        for file,data in (("bench.json",{"completed":2,"errors":[]}),("routers.json",{"control":50,"kv":moved+500}),("router-ports.json",ports)):
            (out/file).write_text(json.dumps(data))
        self.assertFalse((out/"tent.json").exists())
        verdict=pd.recheck(out)
        self.assertTrue(any("L2 collector p: no usable capture" in problem for problem in verdict["problems"]),
                        verdict["problems"])
        self.assertFalse(any("tent.json" in problem for problem in verdict["problems"]))

    def test_tcp_live_collection_marks_tent_not_applicable(self):
        from unittest.mock import patch
        from harness.collector import SIGNAL_COLLECTOR as collector_script
        cleared=[]
        run,p,d,ports,moved,_=self.fixture()
        output=Path(self.directory.name)/"live"
        state={"nodes":{name:{"container":name,"interfaces":{"control":{"address":"10.250.0.1"}}}
                        for name in ("p","d","c")}}
        def collect(*_):
            (output/"bench.json").write_text(json.dumps({"completed":2,"errors":[]}))
            for role,texts in (("prefill",p),("decode",d)):
                for rank,text in enumerate(texts):(output/getattr(run,role).step_log(role,0,rank)).write_text(text)
        def addresses(*_):return ["10.1.0.1"]*4
        with patch.object(topology,"load_state",return_value=state), \
             patch.object(topology,"exec_in",side_effect=lambda name,command,**kw:
                 (cleared.append(name) or '{"signalled":[],"remaining":[]}') if command[:2]==[pd.PYTHON,"-c"] and command[2]==collector_script else ""), \
             patch.object(pd.shell,"run",return_value=""), \
             patch.object(pd,"stop_services"),patch.object(pd,"_background"),patch.object(pd,"_wait"), \
             patch.object(pd,"container_identities",return_value={}), \
             patch.object(pd,"_kv_addresses",side_effect=addresses), \
             patch.object(pd,"router_bytes",side_effect=[{"control":0,"kv":0},{"control":50,"kv":moved+500}]), \
             patch.object(pd,"router_ports",side_effect=[{},ports]), \
             patch.object(pd,"network_ports",return_value={f"p{i}":{"node":n,"rx":0,"tx":0} for i,n in enumerate(("p","d","c","gw"))}), \
             patch.object(llmd,"Front"), \
             patch.object(pd,"_port_deltas",return_value=ports), \
             patch.object(pd,"_collect",side_effect=collect),patch.object(pd.time,"sleep"):
            self.assertEqual(pd.execute(run,output)["problems"],[])
        self.assertFalse((output/"tent.json").exists())
        self.assertFalse((output/"tent-after.json").exists())
        # A run without a collector still clears any one an earlier run left on every node.
        self.assertEqual(sorted(cleared),["c","d","p"])
        started=json.loads((output/"servers.json").read_text())
        self.assertEqual({role:list(nodes) for role,nodes in started.items()},{"prefill":["p"],"decode":["d"]})
        self.assertIn("VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120",started["decode"]["d"])
        self.assertEqual(json.loads((output/"transport.json").read_text()),{"backend":pd.nixl_backend(run.server),"tent_applicable":False})

    def test_zeroed_sub_blocks_are_checked_but_not_moved(self):
        # A decode block of two producer blocks, the second uncovered: NIXL
        # zeroes it; the check compares it, the transfer never moved it.
        layout={"groups":[{"block_size":2112}]}
        checks=[{"groups":[{"group":0,"blocks":1,"bytes":1000,"zeroed_bytes":500,"block_size_ratio":2}]}]
        self.assertEqual(pd._spanned_bytes(layout,checks),500)
        self.assertEqual(pd._spanned_bytes(layout,[{"groups":[{"group":0,"blocks":1,"bytes":1000}]}]),1000)

    def test_plans_of_a_group_of_two_page_sizes_are_checked_by_size(self):
        # Sparse MLA: a latent's and an indexer's layers in one group, each
        # whole on every rank (head 0), their pages of two sizes.
        heads={"prefill":{0:[[0]]},"decode":{0:[[0]],1:[[0]]}}
        sender=[{(0,0):0,(0,1):0}]
        sources={0:[0],1:[0]}
        slab=[(8,72)]
        plan=[0,4,4,160,{"8":2,"72":2}]
        entry=[160,0.01,0,4,1,0,plan]
        args=((0,1),entry,plan,True,heads,sender,sources,slab)
        self.assertEqual(pd._plan_problems(*args,max_descriptors=None),[])
        plan[4]={"8":3,"72":1}
        self.assertTrue(any("carried spans" in s for s in pd._plan_problems(*args,max_descriptors=None)))
        del plan[4]
        self.assertTrue(any("by size" in s for s in pd._plan_problems(*args,max_descriptors=None)))
        regions=[[True,0,0,8],[True,0,0,72]]
        args=((0,1),[160,0.01,0,4,1,1,regions],regions,False,heads,sender,sources,slab)
        self.assertEqual(pd._plan_problems(*args),[])
        self.assertTrue(pd._plan_problems((0,1),[160,0.01,0,4,1,1,regions[:1]],regions[:1],False,heads,sender,sources,slab))
        wrong=[[True,0,0,8],[True,0,8,72]]
        self.assertTrue(pd._plan_problems((0,1),[160,0.01,0,4,1,1,wrong],wrong,False,heads,sender,sources,slab))

    def test_tcp_large_descriptor_plan_keeps_exact_counts_and_bytes(self):
        heads={"decode":{0:[[0]]}}
        sender=[{(0,0):0}]
        sources={0:[0]}
        plan=[0,2048,2048]
        entry=[2048,0.01,0,2048,1,0,plan]
        args=((0,0),entry,plan,True,heads,sender,sources,[1])
        self.assertEqual(pd._plan_problems(*args,max_descriptors=None),[])
        self.assertTrue(any("guard" in s for s in pd._plan_problems(*args)))
        entry[3]=2047
        self.assertTrue(any("measured" in s for s in pd._plan_problems(*args,max_descriptors=None)))
        entry[3]=2048;entry[0]=2049
        self.assertTrue(any("bytes" in s for s in pd._plan_problems(*args,max_descriptors=None)))


class PackedGroupPagesTests(unittest.TestCase):
    """What one descriptor moves of a group's block: a packed group (DeepSeek-V4's) its whole row, any
    other group its pages."""

    ROW = 1002240

    def group(self, **changes) -> dict:
        return {"kind": "UniformTypeKVCacheSpecs", "block_bytes": 37440, "real_page_bytes": 37376,
                "block_stride": self.ROW,
                "kinds": [{"page_bytes": 8640, "real_page_bytes": 8448},
                          {"page_bytes": 37440, "real_page_bytes": 37376}],
                "transfer": {"descriptors_per_block": 1, "bytes_per_block": self.ROW}, **changes}

    def test_a_packed_group_moves_its_whole_row(self):
        self.assertEqual(pd._group_pages(self.group()), [self.ROW])

    def test_other_groups_keep_their_pages(self):
        plain = self.group(block_stride=37440, transfer={"descriptors_per_block": 1, "bytes_per_block": 37440})
        self.assertEqual(pd._group_pages(plain), [37376, 8448])
        two = self.group(transfer={"descriptors_per_block": 2, "bytes_per_block": self.ROW})
        self.assertEqual(pd._group_pages(two), [37376, 8448])
        other = self.group(transfer={"descriptors_per_block": 1, "bytes_per_block": 1000})
        self.assertEqual(pd._group_pages(other), [37376, 8448])
        self.assertEqual(pd._group_pages({"block_bytes": 64, "real_page_bytes": 60}), [60])

    def test_a_read_of_whole_rows_is_checked_against_the_row(self):
        heads = {"prefill": {0: [[0]], 2: [[0]]}, "decode": {0: [[0]], 1: [[0]]}}
        sender = [{(0, 0): 0, (0, 1): 2}]
        sources = {0: [0], 1: [2]}
        slab = [self.ROW]
        plan = [0, 26, 4, 26 * self.ROW, {str(self.ROW): 26}]
        entry = [26 * self.ROW, 0.01, 0, 4, 1, 2, plan]
        self.assertEqual(pd._plan_problems((2, 1), entry, plan, True, heads, sender, sources, slab,
                                           max_descriptors=None), [])
        entry[0] = 26 * self.ROW - 64
        self.assertTrue(any("bytes" in s for s in pd._plan_problems(
            (2, 1), entry, plan, True, heads, sender, sources, slab, max_descriptors=None)))


class LibfabricBackendTests(NixlTcpBackendTests):
    backend = "libfabric"

    def test_policy_metadata_is_not_a_native_provider_or_thread_observation(self):
        for key, value in (("native_provider_observed", True), ("native_threads_observed", True),
                           ("thread_policy_source", "native"), ("num_threads", 8),
                           ("backend_options", {"num_threads":"0"}),
                           ("plugin_defaults", {"num_threads":"0", "split_batch_size":"2048"})):
            def change(p,d,ports):
                records=pd._records(d[0])
                for record in records:
                    if "kv_layout" in record:record["kv_layout"]["nixl_transport"][key]=value
                return p,["".join(line(**r) for r in records),*d[1:]],ports
            self.assertTrue(any("configured-policy" in text for text in self.verdict(change=change)["problems"]))

    def test_full_selected_span_bytes_and_length_bins_remain_mandatory(self):
        hybrid=HybridHeterogeneousTpTests();hybrid.setUp()
        try:
            p,d,ports=hybrid_logs(1,2)
            run=pd.Run(**{**RUN.__dict__,"model":hybrid.model,"server":pd.Server(connector="nixl",nixl_backend="libfabric"),
                          "prefill":pd.Side("p","kv",tensor_parallel_size=1),"decode":pd.Side("d","kv",tensor_parallel_size=2)})
            logs={"prefill":{(0,r):pd._records(t) for r,t in enumerate(p)},"decode":{(0,r):pd._records(t) for r,t in enumerate(d)}}
            ranks={role:{key:{"transfers":{"bytes":sum(((x.get("kv_transfers") or {}).get("load") or {}).get("bytes",0) for x in records)},"requests":["a","b"]} for key,records in ls.items()} for role,ls in logs.items()}
            reconcile=lambda:pd._hetero_tp(run,{"prefill":1,"decode":2},logs,ranks,ports)
            self.assertEqual(reconcile()["problems"],[])
            entry=next(e for records in logs["decode"].values() for r in records for e in ((r.get("kv_transfers") or {}).get("load") or {}).get("each",[]))
            plan=entry[6]
            for index,value in ((3,plan[3]-1),(4,{})):
                saved=plan[index];plan[index]=value
                self.assertTrue(reconcile()["problems"]);plan[index]=saved
        finally:hybrid.tearDown()


def yaml_example() -> dict:
    import yaml
    return yaml.safe_load((E2E / "runs" / "example.yaml").read_text())


class ModelTests(unittest.TestCase):
    """A run's model: a Hugging Face id, fetched without weights."""

    COMMIT = "a" * 40

    def setUp(self) -> None:
        from harness import models
        self.models = models
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, ignore_errors=True))
        patch = mock.patch.object(models, "MODELS", self.root / "models")
        patch.start()
        self.addCleanup(patch.stop)
        self.requested: list[str] = []

    def hub(self, files: dict[str, bytes], status: int | None = None):
        """urlopen of a hub holding ``files`` at COMMIT (or answering ``status``)."""
        import urllib.error

        def get(url: str):
            self.requested.append(url)
            if status:
                raise urllib.error.HTTPError(url, status, "no", None, None)
            if "/api/models/" in url:
                body = json.dumps({"sha": self.COMMIT, "siblings": [{"rfilename": n} for n in files]}).encode()
            else:
                body = files[url.split(f"/resolve/{self.COMMIT}/", 1)[1]]
            return contextlib.closing(io.BytesIO(body))
        return mock.patch.object(self.models, "_get", get)

    def test_only_a_hugging_face_id_names_a_model(self) -> None:
        for value in ("../models/org--name/abc", "/abs/model", "name", ".", "org/name/extra@x y"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "model is a Hugging Face id"):
                self.models.resolve(value, fetch=True)
        with self.assertRaisesRegex(ValueError, "model is a Hugging Face id"):
            pd.parse({**yaml_example(), "model": "../models/x"}, E2E / "runs" / "example.yaml")

    def test_an_id_is_fetched_once_without_weights_into_its_commit(self) -> None:
        files = {"config.json": b"{}", "tokenizer.json": b"t", "model-00001.safetensors": b"w",
                 "pytorch_model.bin": b"w", "nested/tokenizer.model": b"s"}
        with self.hub(files):
            found = self.models.resolve("org/name", fetch=True)
            self.assertEqual(found, self.root / "models" / "org--name" / self.COMMIT)
            self.assertEqual(sorted(str(f.relative_to(found)) for f in found.rglob("*") if f.is_file()),
                             ["config.json", "nested/tokenizer.model", "tokenizer.json"])
            self.assertFalse(any(u.endswith((".safetensors", ".bin")) for u in self.requested))
            self.assertIn("/revision/main", self.requested[0])
            # Fetched: the unpinned id is that commit from now on, with no request.
            self.requested.clear()
            self.assertEqual(self.models.resolve("org/name", fetch=True), found)
            self.assertEqual(self.models.resolve(f"org/name@{self.COMMIT}"), found)
            self.assertEqual(self.requested, [])

    def test_nothing_is_downloaded_without_fetch(self) -> None:
        with self.hub({"config.json": b"{}"}):
            self.assertEqual(self.models.resolve(f"org/name@{self.COMMIT}"),
                             self.root / "models" / "org--name" / self.COMMIT)
            self.assertEqual(self.models.resolve("org/name"), self.root / "models" / "org--name" / "main")
            self.assertEqual(self.requested, [])

    def test_several_fetched_commits_need_one_named(self) -> None:
        for commit in ("a" * 40, "b" * 40):
            (self.root / "models" / "org--name" / commit).mkdir(parents=True)
            (self.root / "models" / "org--name" / commit / "config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "2 commits are fetched .* name one as org/name@<commit>"):
            self.models.resolve("org/name")
        self.assertEqual(self.models.resolve(f"org/name@{'b' * 40}").name, "b" * 40)

    def test_a_failed_fetch_says_why_and_leaves_no_model(self) -> None:
        with self.hub({}, status=401), self.assertRaisesRegex(ValueError, "HTTP 401 .*HF_TOKEN"):
            self.models.resolve("org/gated", fetch=True)
        with self.hub({"README.md": b"x"}), self.assertRaisesRegex(ValueError, "has no config.json"):
            self.models.resolve("org/empty", fetch=True)
        self.assertEqual(list((self.root / "models").glob("*/*/config.json")), [])


DSV3_CONFIG = {"model_type": "deepseek_v3", "kv_lora_rank": 512, "qk_rope_head_dim": 64,
               "num_hidden_layers": 61}


def dcp_layout(rank: int, dcp: int, page: int = 16, tp: int = 2) -> str:
    """A rank's layout record, with its resolved decode context parallelism."""
    geometry = ({"size": dcp, "rank": rank % dcp, "page_tokens": [page],
                 "scheduler_block_tokens": [page * dcp], "interleave": page} if dcp > 1 else None)
    return line(kv_layout={"tp_rank": rank, "tp_size": tp, "block_bytes": 10, "bytes": 1600,
                           "groups": [{"kind": "MLAAttentionSpec", "sliding_window": None, "block_size": page,
                                       "layers": 1, "block_bytes": 10}],
                           "decode_context_parallel": geometry})


def dcp_pull(request: str, blocks: int, prompt: int) -> str:
    """Decode's pull of a request of `prompt` tokens into `blocks` blocks, finished."""
    req_id = f"cmpl-{request}-0-d"
    return (line(kv_events={"allocated": {req_id: list(range(1, blocks + 1))}, "reads_posted": [req_id]})
            + stepped(request, "d", prompt - 1, 1, prompt) + line(kv_events={"reads_done": [req_id]}))


def dcp_check(request: str, rank: int, dcp: int, remote: int, tokens: int, blocks: int,
              payload: int, padding: int = 0) -> dict:
    """Decode's check of a request under decode context parallelism: its
    payload, moved padding and untouched blocks of one group (10 bytes a block)."""
    moved = payload + padding
    return {"req_id": f"cmpl-{request}-0-d", "ok": True, "blocks": moved, "layers": 1, "bytes": moved * 10,
            "groups": [{"group": 0, "blocks": moved, "bytes": moved * 10, "layers": 1,
                        "payload_blocks": payload, "padding_blocks": padding,
                        "untouched_blocks": blocks - moved, "padding_bytes": padding * 10,
                        "untouched_bytes": (blocks - moved) * 10, "block_bytes": 10}],
            "dcp": {"size": dcp, "rank": rank % dcp, "remote_size": remote, "remote_num_tokens": tokens},
            "mismatched_bytes": 0}


def dcp_read(peer: int, nbytes: int) -> str:
    return line(kv_transfers={"load": {"count": 1, "bytes": nbytes,
                                       "each": [[nbytes, 0.001, 0.0, 2, None, peer, None]]}})


class DecodeContextParallelTests(unittest.TestCase):
    def run_of(self, config: dict = DSV3_CONFIG, **sides) -> pd.Run:
        with tempfile.TemporaryDirectory() as root:
            model = Path(root) / "org--name" / "main"
            model.mkdir(parents=True)
            (model / "config.json").write_text(json.dumps(config))
            raw = {"name": "t", "topology": "t", "model": "org/name", "client": "c",
                   "control_network": "control", "router": {"gateway": "gw"},
                   "prefill": {"node": "p", "kv_network": ["data0", "data1"]},
                   "decode": {"node": "d", "kv_network": ["data0", "data1"],
                              "decode_context_parallel_size": 2},
                   "server": {"timing": "zero", "connector": "nixl", "tensor_parallel_size": 2}}
            for key, value in sides.items():
                raw[key] = {**raw[key], **value} if isinstance(raw.get(key), dict) else value
            with mock.patch.object(pd.models, "MODELS", Path(root)):
                return pd.parse(raw, Path(root) / "run.yaml")

    def test_dcp_is_a_dense_mla_models_over_nixl_at_equal_tp(self) -> None:
        run = self.run_of()
        self.assertEqual((run.prefill.decode_context_parallel_size, run.decode.decode_context_parallel_size),
                         (1, 2))
        self.assertEqual(self.run_of(prefill={"decode_context_parallel_size": 2}).prefill
                         .decode_context_parallel_size, 2)
        switched = {"prefill": {"decode_context_parallel_size": 2, "kv_network": "kv"},
                    "decode": {"decode_context_parallel_size": 1, "kv_network": "kv"}}
        self.assertEqual(self.run_of(**switched).decode.kv_network, "kv")
        # Unequal TP sizes, one dividing the other, on one switched network.
        unequal = self.run_of(prefill={"tensor_parallel_size": 4, "decode_context_parallel_size": 4,
                                       "kv_network": "kv"}, decode={"kv_network": "kv"})
        self.assertEqual((unequal.prefill.tp(unequal.server), unequal.decode.tp(unequal.server)), (4, 2))
        with self.assertRaisesRegex(ValueError, "one switched network for both instances"):
            self.run_of(prefill={"decode_context_parallel_size": 2}, decode={"decode_context_parallel_size": 1})
        cases = {
            "1 or the instance's TP size": {"decode": {"decode_context_parallel_size": 4}},
            "positive integer": {"decode": {"decode_context_parallel_size": 0}},
            "mooncake connector": {"server": {"timing": "zero", "connector": "mooncake",
                                              "tensor_parallel_size": 2}},
            "must divide one another, not 3 and 2": {"prefill": {"tensor_parallel_size": 3, "kv_network": "data0"},
                                                     "decode": {"kv_network": "data0"}},
            "unequal TP sizes need one switched network": {"prefill": {"tensor_parallel_size": 1,
                                                                       "kv_network": ["data0"]}},
            "data parallelism": {"decode": {"data_parallel_size": 2}},
        }
        # Prefix caching, one setting for both instances: checked or not.
        for change in ({"server": {"timing": "zero", "connector": "nixl", "tensor_parallel_size": 2,
                                   "enable_prefix_caching": True}}, {"kv_check": False}):
            with self.subTest(prefix=change):
                self.assertEqual(self.run_of(**change).decode.decode_context_parallel_size, 2)
        for message, change in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message.strip()):
                self.run_of(**change)
        for config in ({"model_type": "qwen3", "num_key_value_heads": 8},
                       {**DSV3_CONFIG, "model_type": "deepseek_v32", "index_topk": 2048}):
            with self.subTest(config=config["model_type"]), \
                    self.assertRaisesRegex(ValueError, "a dense MLA model only"):
                self.run_of(config)

    def test_dcp_sources_follow_nixls_mapping(self) -> None:
        # (decode TP, prefill TP, prefill DCP, decode DCP) -> each decode
        # rank's sources, as vLLM's `dcp_source_ranks` / `compute_tp_mapping` give them.
        cases = {
            (2, 2, 1, 2): [[0], [1]],
            (2, 2, 2, 1): [[0, 1], [0, 1]],
            (2, 2, 2, 2): [[0], [1]],
            (4, 4, 4, 4): [[0], [1], [2], [3]],
            (2, 1, 1, 2): [[0], [0]],  # replicated producer, smaller TP
            (2, 4, 1, 2): [[0], [2]],  # replicated producer, larger TP
            (4, 2, 1, 4): [[0], [0], [1], [1]],  # consumer finer by 4
            (4, 2, 2, 4): [[0], [1], [0], [1]],  # both sharded, consumer finer
            (2, 4, 4, 2): [[0, 2], [1, 3]],  # both sharded, producer finer
            (2, 4, 4, 1): [[0, 1, 2, 3], [0, 1, 2, 3]],  # replicated consumer
            (4, 2, 2, 1): [[0, 1]] * 4,
            (1, 2, 2, 1): [[0, 1]],
        }
        for (decode_tp, prefill_tp, prefill_dcp, decode_dcp), sources in cases.items():
            with self.subTest(decode_tp=decode_tp, prefill_tp=prefill_tp, prefill_dcp=prefill_dcp,
                              decode_dcp=decode_dcp):
                self.assertEqual([pd.dcp_sources(r, decode_tp, prefill_tp, prefill_dcp, decode_dcp)
                                  for r in range(decode_tp)], sources)

    PERFORMANCE = {"kv_check": False, "server": {"timing": "zero", "connector": "nixl", "tensor_parallel_size": 2,
                                                 "enable_prefix_caching": False}}

    def audit(self, decode: dict, prefill: dict | None = None, prefill_dcp: int = 1, decode_dcp: int = 2,
              checks: bool = True, unread: frozenset = frozenset(), prefill_blocks: int | None = None):
        # A coarser consumer reads across ranks: one switched network.
        network = {"kv_network": "kv"} if prefill_dcp > decode_dcp else {}
        run = self.run_of(prefill={"decode_context_parallel_size": prefill_dcp, **network},
                          decode={"decode_context_parallel_size": decode_dcp, **network},
                          **({} if checks else self.PERFORMANCE))

        def records(text: str) -> list[dict]:
            return [json.loads(x) for x in text.splitlines() if x]

        # 70 tokens: the producer's blocks of 16 x its DCP tokens.
        blocks = prefill_blocks if prefill_blocks is not None else -(-70 // (16 * prefill_dcp))
        held = (line(kv_events={"allocated": {"cmpl-a-0-p": list(range(1, blocks + 1))}})
                + line(kv_events={"sends_done": ["cmpl-a-0-p"]}))
        prefill = prefill or {r: dcp_layout(r, prefill_dcp) + held for r in range(2)}
        logs = {"prefill": {(0, r): records(t) for r, t in prefill.items()},
                "decode": {(0, r): records(t) for r, t in decode.items()}}
        return pd._dcp_audit(run, {"prefill": 2, "decode": 2}, logs, unread)

    def test_without_checks_the_pulls_bound_each_ranks_reads(self) -> None:
        # 70 tokens, 5 pages: a DCP2 rank's 3 blocks hold 3 - r payload pages.
        clean = {r: dcp_layout(r, 2) + dcp_pull("a", 3, 70) + dcp_read(r, (3 - r) * 10) for r in range(2)}
        audit = self.audit(clean, checks=False)
        self.assertEqual(audit["problems"], [])
        self.assertEqual([audit["ranks"][r]["payload_blocks"] for r in range(2)], [3, 2])
        unfinished = line(kv_events={"allocated": {"cmpl-a-0-d": [1, 2, 3]}, "reads_posted": ["cmpl-a-0-d"]})
        cases = {
            "decode rank 1 read 10 bytes, not between its 2 payload page(s) and its 3 pulled block(s)": {
                1: dcp_layout(1, 2) + dcp_pull("a", 3, 70) + dcp_read(1, 10)},
            "decode rank 0 read 0 bytes, not between its 3 payload page(s)": {
                0: dcp_layout(0, 2) + dcp_pull("a", 3, 70)},
            "decode rank 1 finished 0 of its 1 pull(s)": {
                1: dcp_layout(1, 2) + unfinished + stepped("a", "d", 69, 1, 70) + dcp_read(1, 20)},
            "decode rank 1 logged no pull": {1: dcp_layout(1, 2) + dcp_read(1, 20)},
            "1 READ(s) of no producer rank among [1]": {1: dcp_layout(1, 2) + dcp_pull("a", 3, 70) + line(
                kv_transfers={"load": {"count": 1, "bytes": 20, "each": [[20, 0.001, 0.0, 2, None, None, None]]}})},
        }
        for message, change in cases.items():
            with self.subTest(message=message):
                problems = self.audit({**clean, **change}, checks=False)["problems"]
                self.assertTrue(any(message in p for p in problems), problems)
        # A rank of payload without a read series is never accepted as notify-only.
        problems = self.audit(clean, checks=False, unread=frozenset({1}))["problems"]
        self.assertTrue(any("decode rank 1's TENT metrics lack tent_read_bytes_total" in p for p in problems),
                        problems)

    def test_each_rank_holds_its_global_pages_read_from_its_sources(self) -> None:
        # 70 tokens: 5 pages of 16; a DCP2 rank's 3 blocks hold pages r, r+2, r+4.
        clean = {r: dcp_layout(r, 2) + dcp_pull("a", 3, 70) + dcp_read(r, (3 - r) * 10)
                 + line(kv_check=dcp_check("a", r, 2, 1, 70, 3, 3 - r)) for r in range(2)}
        audit = self.audit(clean)
        self.assertEqual(audit["problems"], [])
        self.assertEqual(audit["ranks"][1], {"dcp_rank": 1, "sources": [1], "peers": [1], "reads": 1,
                                             "pulls": 1, "payload_blocks": 2, "notify_only": 0, "payload": 2,
                                             "padding": 0, "untouched": 1, "padding_bytes": 0,
                                             "untouched_bytes": 10})
        self.assertEqual(audit["released"], {0: [1, 1], 1: [1, 1]})
        cases = {
            "has 3 payload page(s), not the 2": {1: dcp_layout(1, 2) + dcp_pull("a", 3, 70) + dcp_read(1, 30)
                                                 + line(kv_check=dcp_check("a", 1, 2, 1, 70, 3, 3))},
            "1 READ(s) of no producer rank among [1]: [20, 0.001, 0.0, 2, None, 0]": {1: dcp_layout(1, 2) + dcp_pull("a", 3, 70) + dcp_read(0, 20)
                                                             + line(kv_check=dcp_check("a", 1, 2, 1, 70, 3, 2))},
            "not size 2 rank 1 from size 1": {1: dcp_layout(1, 2) + dcp_pull("a", 3, 70) + dcp_read(1, 20)
                                              + line(kv_check=dcp_check("a", 1, 2, 2, 70, 3, 2))},
            "decode rank 1's decode context parallelism is None": {
                1: dcp_layout(1, 1) + dcp_pull("a", 3, 70) + dcp_read(1, 20) + line(kv_check=dcp_check("a", 1, 2, 1, 70, 3, 2))},
        }
        for message, change in cases.items():
            with self.subTest(message=message):
                problems = self.audit({**clean, **change})["problems"]
                self.assertTrue(any(message in p for p in problems), problems)
        unreleased = {r: dcp_layout(r, 1) + line(kv_events={"allocated": {"cmpl-a-0-p": [1]}}) for r in range(2)}
        problems = self.audit(clean, prefill=unreleased)["problems"]
        self.assertTrue(any("prefill rank 0 finished sending 0 of the 1 request(s)" in p for p in problems), problems)

    def test_every_read_names_a_source_and_sums_to_its_record(self) -> None:
        clean = {r: dcp_layout(r, 2) + dcp_read(r, (3 - r) * 10)
                 + line(kv_check=dcp_check("a", r, 2, 1, 70, 3, 3 - r)) for r in range(2)}
        unnamed = line(kv_transfers={"load": {"count": 1, "bytes": 20,
                                              "each": [[20, 0.001, 0.0, 2, None, None, None]]}})
        miscounted = line(kv_transfers={"load": {"count": 2, "bytes": 20,
                                                 "each": [[20, 0.001, 0.0, 2, None, 1, None]]}})
        check = line(kv_check=dcp_check("a", 1, 2, 1, 70, 3, 2))
        cases = {
            "1 READ(s) of no producer rank among [1]": dcp_layout(1, 2) + dcp_pull("a", 3, 70) + unnamed + check,
            "load of 2 READ(s), 20 bytes, is not its 1 READ(s)' sum": dcp_layout(1, 2) + dcp_pull("a", 3, 70) + miscounted + check,
            "checked 20 bytes, not the 10 it read": dcp_layout(1, 2) + dcp_pull("a", 3, 70) + dcp_read(1, 10) + check,
        }
        for message, records in cases.items():
            with self.subTest(message=message):
                problems = self.audit({**clean, 1: records})["problems"]
                self.assertTrue(any(message in p for p in problems), problems)

    def one_page(self, payloads=(1, 0), reads=(dcp_read(0, 10), ""), read_series=(True, False),
                 checks: bool = True, answered=(True, True)) -> dict:
        """A complete verdict of one 10-token request (one 16-token page)
        from prefill DCP 1 to decode DCP 2: rank 0 reads the page, rank 1
        holds a block past the handoff, untouched (notify-only)."""
        run = self.run_of(workload={"num_prompts": 1, "input_len": 10}, **({} if checks else self.PERFORMANCE))
        layout = {"block_bytes": 10, "bytes": 1600, "groups": [
            {"kind": "MLAAttentionSpec", "sliding_window": None, "block_size": 16, "layers": 1, "block_bytes": 10}]}

        def ranked(rank: int, dcp: int) -> str:
            record = json.loads(dcp_layout(rank, dcp))["kv_layout"]
            return line(kv_layout={**record, **layout})

        def checked(rank: int, payload: int) -> str:
            check = dcp_check("a", rank, 2, 1, 10, 1, payload)
            check["groups"][0]["first_logical_block"] = 0
            return line(kv_check=check)

        pull = line(kv_events={"allocated": {"cmpl-a-0-d": [1]}, "reads_posted": ["cmpl-a-0-d"]}) + line(
            kv_events={"reads_done": ["cmpl-a-0-d"]})
        prefill = [ranked(r, 1) + line(kv_events={"allocated": {"cmpl-a-0-p": [1]}})
                   + stepped("a", "p", 0, 10, 10) + line(kv_events={"sends_done": ["cmpl-a-0-p"]}) for r in range(2)]
        decode = [ranked(r, 2) + pull + stepped("a", "d", 9, 1, 10) + reads[r]
                  + (checked(r, payloads[r]) if checks else "") for r in range(2)]
        registered = {"tent_registered_buffer_bytes": 1600}
        tents = {"prefill": [registered, registered],
                 "decode": [None if not answered[r] else {**registered, "tent_read_bytes_total": 10 * payloads[r]}
                            if read_series[r] else registered for r in range(2)]}
        return pd.check({"completed": 1, "errors": [""]}, prefill, decode,
                        {"control": 1, "data0": 200, "data1": 100}, run, tents)

    def test_a_notify_only_rank_passes_the_complete_verdict(self) -> None:
        verdict = self.one_page()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["dcp"]["unread_accepted"], [1])
        self.assertEqual((verdict["dcp"]["ranks"][1]["reads"], verdict["dcp"]["ranks"][1]["notify_only"]), (0, 1))
        # Rank 0's page under the handoff missing: no READ, its block
        # checked untouched, no read series.
        missing = self.one_page(payloads=(0, 0), reads=("", ""), read_series=(False, False))
        self.assertNotIn(0, missing["dcp"]["unread_accepted"])
        cases = {
            "decode rank 0's check of cmpl-a-0-d has 0 payload page(s), not the 1": missing,
            "decode rank 0's TENT metrics lack tent_read_bytes_total, though it posted 0 READ(s)": missing,
            # Rank 1 reads without a read series.
            "decode rank 1's TENT metrics lack tent_read_bytes_total, though it posted 1 READ(s)": self.one_page(
                payloads=(1, 1), reads=(dcp_read(0, 10), dcp_read(1, 10))),
        }
        for message, verdict in cases.items():
            with self.subTest(message=message):
                self.assertTrue(any(message in p for p in verdict["problems"]), verdict["problems"])

    def test_without_checks_a_notify_only_rank_passes_the_complete_verdict(self) -> None:
        verdict = self.one_page(checks=False)
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["dcp"]["unread_accepted"], [1])
        self.assertEqual(verdict["dcp"]["ranks"][1]["payload_blocks"], 0)
        # Rank 0's page under the handoff missing: no READ, no read series.
        missing = self.one_page(reads=("", ""), read_series=(False, False), checks=False)
        self.assertEqual(missing["dcp"]["unread_accepted"], [1])
        cases = {
            "decode rank 0 read 0 bytes, not between its 1 payload page(s)": missing,
            "decode rank 0's TENT metrics lack tent_read_bytes_total, though it posted 0 READ(s), 1 pull(s) "
            "with 1 payload page(s)": missing,
            "decode's TENT metrics did not answer": self.one_page(checks=False, answered=(True, False)),
        }
        for message, verdict in cases.items():
            with self.subTest(message=message):
                self.assertTrue(any(message in p for p in verdict["problems"]), verdict["problems"])

    def unequal(self, producer: tuple[int, int], consumer: tuple[int, int], prompt: int = 70,
                checks: bool = True, change=None) -> dict:
        """A complete verdict of one request of `prompt` tokens (16-token pages)
        from prefill (TP, DCP) `producer` to decode `consumer` on one switched
        network, every rank's records as NIXL's mapping and the pages each
        rank holds call for; `change(prefill, decode, tents)` edits them."""
        (p_tp, p_dcp), (d_tp, d_dcp) = producer, consumer
        run = self.run_of(prefill={"tensor_parallel_size": p_tp, "decode_context_parallel_size": p_dcp,
                                   "kv_network": "kv"},
                          decode={"tensor_parallel_size": d_tp, "decode_context_parallel_size": d_dcp,
                                  "kv_network": "kv"},
                          workload={"num_prompts": 1, "input_len": prompt},
                          **({} if checks else self.PERFORMANCE))
        pages = -(-prompt // 16)
        p_blocks, d_blocks = -(-prompt // (16 * p_dcp)), -(-prompt // (16 * d_dcp))
        prefill = [dcp_layout(r, p_dcp, tp=p_tp) + line(kv_events={"allocated": {
            "cmpl-a-0-p": list(range(1, p_blocks + 1))}}) + stepped("a", "p", 0, prompt, prompt)
            + line(kv_events={"sends_done": ["cmpl-a-0-p"]}) for r in range(p_tp)]
        decode, tents = [], {"prefill": [{"tent_registered_buffer_bytes": 1600}] * p_tp, "decode": []}
        for d in range(d_tp):
            mine = {k * d_dcp + d % d_dcp for k in range(d_blocks)}
            moved = set()
            records = dcp_layout(d, d_dcp, tp=d_tp) + dcp_pull("a", d_blocks, prompt)
            for source in pd.dcp_sources(d, d_tp, p_tp, p_dcp, d_dcp):
                common = mine & {k * p_dcp + source % p_dcp for k in range(p_blocks)}
                if common:
                    records += dcp_read(source, 10 * len(common))
                    moved |= common
            payload = len({g for g in mine if g < pages})
            if checks:
                check = dcp_check("a", d, d_dcp, p_dcp, prompt, d_blocks, payload, len(moved) - payload)
                check["groups"][0]["first_logical_block"] = 0
                records += line(kv_check=check)
            decode.append(records)
            tent = {"tent_registered_buffer_bytes": 1600}
            tents["decode"].append({**tent, "tent_read_bytes_total": 10 * len(moved)} if moved else tent)
        if change:
            change(prefill, decode, tents)
        return pd.check({"completed": 1, "errors": [""]}, prefill, decode, {"control": 1, "kv": 10_000}, run,
                        tents)

    def test_unequal_tp_pairs_pass_the_complete_verdict_by_nixls_mapping(self) -> None:
        # Decode ranks 2 and 3 of a TP4 consumer have no producer rank of
        # their index; a replicated consumer reads both sharded producers.
        for producer, consumer in (((2, 1), (4, 4)), ((2, 2), (4, 1))):
            for checks in (True, False):
                with self.subTest(producer=producer, consumer=consumer, checks=checks):
                    verdict = self.unequal(producer, consumer, checks=checks)
                    self.assertEqual(verdict["problems"], [])
                    sources = verdict["dcp"]["ranks"]
                    want = [pd.dcp_sources(d, 4, 2, producer[1], consumer[1]) for d in range(4)]
                    self.assertEqual([sources[d]["sources"] for d in range(4)], want)
                    self.assertEqual(verdict["dcp"]["page_tokens"], 16)
        # One page: each replicated consumer rank reads producer 0 only;
        # producer 1, mapped, holds no page it holds and is only notified.
        verdict = self.unequal((2, 2), (4, 1), prompt=10)
        self.assertEqual(verdict["problems"], [])
        self.assertEqual({d: verdict["dcp"]["ranks"][d]["peers"] for d in range(4)}, {d: [0] for d in range(4)})

    def test_a_producers_blocks_are_its_steps_allocations_combined(self) -> None:
        # Chunked prefill logs each chunk's new blocks; a resumed request's
        # blocks replace its earlier ones. The READs a finer consumer's
        # pages call for follow the producer's whole allocation.
        def allocated(blocks: int) -> str:
            return line(kv_events={"allocated": {"cmpl-a-0-p": list(range(1, blocks + 1))}})

        def chunked(prefill, decode, tents):
            for r, records in enumerate(prefill):
                blocks = len(json.loads(records.splitlines()[1])["kv_events"]["allocated"]["cmpl-a-0-p"])
                prefill[r] = records.replace(allocated(blocks), "".join(
                    line(kv_events={"allocated": {"cmpl-a-0-p": [k]}}) for k in range(1, blocks + 1)))

        def resumed(prefill, decode, tents):
            for r, records in enumerate(prefill):
                blocks = len(json.loads(records.splitlines()[1])["kv_events"]["allocated"]["cmpl-a-0-p"])
                prefill[r] = records.replace(allocated(blocks), allocated(blocks) + line(kv_events={
                    "resumed": ["cmpl-a-0-p"], "allocated": {"cmpl-a-0-p": list(range(50, 50 + blocks))}}))

        for producer in ((2, 1), (2, 2)):
            for change in (chunked, resumed):
                for checks in (True, False):
                    with self.subTest(producer=producer, change=change.__name__, checks=checks):
                        verdict = self.unequal(producer, (4, 4), checks=checks, change=change)
                        self.assertEqual(verdict["problems"], [])
        self.assertEqual(pd._allocated_blocks([json.loads(x) for x in (
            allocated(2) + line(kv_events={"allocated": {"cmpl-a-0-p": [3]}})
            + line(kv_events={"resumed": ["cmpl-a-0-p"], "allocated": {"cmpl-a-0-p": [7, 8]}})).splitlines()]),
            {"cmpl-a-0-p": 2})

    def test_unequal_tp_verdicts_reject_wrong_traffic_and_releases(self) -> None:
        def drop_read(prefill, decode, tents):
            # Producer 1 holds pages 1, 3 and 5 of 70 tokens; the rank's 1 and 3.
            assert dcp_read(1, 20) in decode[2]
            decode[2] = decode[2].replace(dcp_read(1, 20), "")

        def notified_reads(prefill, decode, tents):
            decode[3] += dcp_read(1, 10)

        def unreleased(prefill, decode, tents):
            prefill[1] = prefill[1].replace(line(kv_events={"sends_done": ["cmpl-a-0-p"]}), "")

        def expired(prefill, decode, tents):
            prefill[0] += line(kv_transfers={"lease": {"expired": 1, "late_notifications": 0}})

        def other_page(prefill, decode, tents):
            decode[1] = decode[1].replace('"block_size": 16', '"block_size": 32')

        def silent(prefill, decode, tents):
            tents["decode"][2] = None

        cases = [
            ("decode rank 2 read {0: 1} (READs by producer rank), not the {0: 1, 1: 1}", drop_read, 70),
            ("decode rank 3 read {0: 1, 1: 1} (READs by producer rank), not the {0: 1}", notified_reads, 10),
            ("prefill rank 1 finished sending 0 of the 1 request(s)", unreleased, 70),
            ("prefill rank 0 released 1 request(s) by lease expiry", expired, 70),
            ("the pair's ranks record page sizes", other_page, 70),
            ("decode's TENT metrics did not answer", silent, 70),
        ]
        for message, change, prompt in cases:
            for checks in (True, False):
                with self.subTest(message=message, checks=checks):
                    problems = self.unequal((2, 2), (4, 1), prompt=prompt, checks=checks, change=change)["problems"]
                    self.assertTrue(any(message in p for p in problems), problems)

    def prefixed(self, checks: bool = True, change=None) -> dict:
        """A complete verdict, prefix caching on, of two 70-token requests
        (16-token pages) from prefill TP2/DCP1 to decode TP2/DCP2, b sharing
        a's first 64 tokens: prefill hits 4 pages; decode hits two scheduler
        blocks (2 of each rank's blocks) and pulls one, page 4 on rank 0
        (payload), page 5 on rank 1 (padding its producer does not hold:
        untouched, the producer only notified)."""
        run = self.run_of(server={"timing": "zero", "connector": "nixl", "tensor_parallel_size": 2,
                                  "enable_prefix_caching": True},
                          workload={"num_prompts": 2, "input_len": 70}, **({} if checks else {"kv_check": False}))

        def rid(request, side):
            return f"cmpl-{request}-0-{side}"

        def admitted(request, side, computed, ids):
            return line(kv_events={"allocated": {rid(request, side): ids}, "admitted": {rid(request, side): computed},
                                   "allocated_groups": {rid(request, side): [{"group": 0, "first": 0, "blocks": ids}]}})

        def posted(request, first, ids):
            return line(kv_events={"allocated": {rid(request, "d"): ids}, "reads_posted": [rid(request, "d")],
                                   "allocated_groups": {rid(request, "d"): [
                                       {"group": 0, "first": first, "blocks": ids[first:]}]}})

        def hit_check(request, side, ids):
            return {"req_id": rid(request, side), "kind": "hit", "blocks": len(ids), "bytes": 10 * len(ids),
                    "groups": [{"group": 0, "logical_blocks": list(range(len(ids))), "block_ids": ids,
                                "blocks": len(ids), "bytes": 10 * len(ids), "mismatched_bytes": 0,
                                "written_this_step": []}], "mismatched_bytes": 0, "ok": True}

        def transfer_check(request, rank, first, count, payload):
            check = dcp_check(request, rank, 2, 1, 70, count, payload)
            check["groups"][0]["first_logical_block"] = first
            return check

        prefill = [dcp_layout(r, 1) + admitted("a", "p", 0, blocks(1, 5)) + stepped("a", "p", 0, 70, 70)
                   + admitted("b", "p", 64, blocks(1, 4) + [206]) + stepped("b", "p", 64, 6, 70)
                   + (line(kv_check=hit_check("b", "p", blocks(1, 4))) if checks else "")
                   + line(kv_events={"sends_done": [rid("a", "p"), rid("b", "p")]}) for r in range(2)]
        a_ids, b_ids = blocks(101, 3), blocks(101, 2) + [301]
        # a: rank 0 pages 0, 2, 4; rank 1 pages 1, 3 (page 5 untouched). b:
        # past the hit, rank 0 page 4; rank 1 page 5, untouched.
        reads = {0: [dcp_read(0, 30), dcp_read(0, 10)], 1: [dcp_read(1, 20), ""]}
        payloads = {0: (3, 1), 1: (2, 0)}
        decode = []
        for d in range(2):
            records = (dcp_layout(d, 2) + posted("a", 0, a_ids) + admitted("a", "d", 69, a_ids)
                       + stepped("a", "d", 69, 1, 70) + reads[d][0] + line(kv_events={"reads_done": [rid("a", "d")]})
                       + posted("b", 2, b_ids) + admitted("b", "d", 69, b_ids)
                       + stepped("b", "d", 69, 1, 70) + reads[d][1] + line(kv_events={"reads_done": [rid("b", "d")]}))
            if checks:
                records += (line(kv_check=transfer_check("a", d, 0, 3, payloads[d][0]))
                            + line(kv_check=hit_check("b", "d", blocks(101, 2)))
                            + line(kv_check=transfer_check("b", d, 2, 1, payloads[d][1])))
            decode.append(records)
        tent = {"tent_registered_buffer_bytes": 1600}
        tents = {"prefill": [tent, tent], "decode": [{**tent, "tent_read_bytes_total": 40},
                                                      {**tent, "tent_read_bytes_total": 20}]}
        if change:
            change(prefill, decode, tents)
        return pd.check({"completed": 2, "errors": ["", ""]}, prefill, decode,
                        {"control": 1, "data0": 200, "data1": 200}, run, tents)

    def test_a_prefix_hit_leaves_its_pages_out_of_the_pull_and_the_transfer_accounting(self) -> None:
        for checks in (True, False):
            with self.subTest(checks=checks):
                verdict = self.prefixed(checks=checks)
                self.assertEqual(verdict["problems"], [])
                ranks = verdict["dcp"]["ranks"]
                self.assertEqual([ranks[d]["reads"] for d in range(2)], [2, 1])
                if checks:
                    # The hit's cached bytes are not transferred ones.
                    self.assertEqual([ranks[d]["untouched"] for d in range(2)], [0, 2])
                    self.assertEqual(verdict["kv_bytes_checked"], 60)

        def hit_page_read(prefill, decode, tents):
            decode[0] += dcp_read(0, 10)
            tents["decode"][0]["tent_read_bytes_total"] = 50

        def pull_before_hit(prefill, decode, tents):
            decode[1] = decode[1].replace('"first": 2, "blocks": [301]', '"first": 1, "blocks": [102, 301]')

        def hit_per_page(prefill, decode, tents):
            decode[0] = decode[0].replace('"logical_blocks": [0, 1], "block_ids": [101, 102]',
                                          '"logical_blocks": [0, 1, 2, 3], "block_ids": [101, 102, 103, 104]')

        cases = [
            ("decode rank 0 read {0: 3} (READs by producer rank), not the {0: 2}", hit_page_read, False),
            ("decode rank 0 read {0: 3} (READs by producer rank), not the {0: 2}", hit_page_read, True),
            # Pulling page 3 too: producer rank 1 holds it, a READ is missing.
            ("decode rank 1 read {1: 1} (READs by producer rank), not the {1: 2}", pull_before_hit, False),
            ("covers 1 block(s) from 2 (payload, moved and untouched padding), not its pull's 2 from 1",
             pull_before_hit, True),
            ("hit check covers group 0: 4 block(s) 0:101..3:104, not its admitted hit group 0: 2 block(s)",
             hit_per_page, True),
        ]
        for message, change, checks in cases:
            with self.subTest(message=message, checks=checks):
                problems = self.prefixed(checks=checks, change=change)["problems"]
                self.assertTrue(any(message in p for p in problems), problems)

    def test_a_sharded_producers_ranks_all_feed_a_replicated_consumer(self) -> None:
        # Producer DCP2 to consumer DCP1: each consumer rank holds every page,
        # read from both producer ranks.
        decode = {r: dcp_layout(r, 1) + dcp_pull("a", 5, 70) + dcp_read(0, 30) + dcp_read(1, 20)
                  + line(kv_check=dcp_check("a", r, 1, 2, 70, 5, 5)) for r in range(2)}
        audit = self.audit(decode, prefill_dcp=2, decode_dcp=1)
        self.assertEqual(audit["problems"], [])
        self.assertEqual([audit["ranks"][r]["peers"] for r in range(2)], [[0, 1], [0, 1]])

    def test_the_server_is_launched_with_its_sides_dcp(self) -> None:
        run = self.run_of()
        flags = {role: pd._decode_context_parallel_flags(side) for role, side in
                 (("prefill", run.prefill), ("decode", run.decode))}
        self.assertEqual(flags, {"prefill": [], "decode": ["--decode-context-parallel-size", "2"]})

    def test_a_dcp_ranks_check_holds_its_scheduler_blocks(self) -> None:
        # 1060 tokens at 64 a page: 17 pages; a DCP2 rank holds
        # cdiv(1060, 128) = 9 blocks, rank 1's last past the handoff.
        layout = {"groups": [{"kind": "MLAAttentionSpec", "sliding_window": None, "block_size": 64,
                              "layers": 1, "block_bytes": 10}]}

        def problems(rank: int, payload: int, untouched: int) -> list[str]:
            check = dcp_check("a", rank, 2, 1, 1060, payload + untouched, payload)
            check["groups"][0]["first_logical_block"] = 0
            return pd._byte_check_problems(layout, [check], 1060, {}, {"cmpl-a-0-d": 1060})

        self.assertEqual(problems(0, 9, 0), [])
        self.assertEqual(problems(1, 8, 1), [])
        bad = problems(1, 8, 0)
        self.assertTrue(any("holds 8 blocks from 0, not the rest of the cdiv(1060, 64 * DCP 2)" in p for p in bad), bad)


class MtpStatePullTests(unittest.TestCase):
    """A hybrid's MTP with prefix caching: the scheduler admits a pulled
    state as its group's running slot, ``cdiv(computed + scheduled, block) - 1``,
    though the pull named the slot of the table it saw; attention keeps its
    places, and a pulled state group implies no decode hit."""

    GROUPS = [{"kind": "MambaSpec", "block_size": 800, "speculative_blocks": 1, "layers": 16,
               "block_bytes": 100, "transfer": {"bytes_per_block": 1600}},
              {"kind": "FullAttentionSpec", "block_size": 800, "layers": 16, "block_bytes": 100,
               "transfer": {"bytes_per_block": 1600}}]

    @staticmethod
    def admission(state_table, pull_state=(2, [7]), computed=1023, scheduled=1):
        return {"r": [{"computed": computed, "scheduled": scheduled,
                       "table": {0: state_table, 1: [4, 5]},
                       "pull": {0: pull_state, 1: (0, [4, 5])}}]}

    def problems(self, admissions):
        return pd._hit_problems({"groups": self.GROUPS}, [], admissions)[0]

    def test_the_pulled_state_admitted_as_the_running_slot(self):
        self.assertEqual(self.problems(self.admission([0, 7, 9])), [])
        self.assertEqual(pd._running_state_slot(self.GROUPS[0], {"computed": 1023, "scheduled": 1}), 1)
        # A step crossing a block keeps the running state in the next block.
        self.assertEqual(pd._running_state_slot(self.GROUPS[0], {"computed": 799, "scheduled": 2}), 1)
        self.assertEqual(pd._running_state_slot(self.GROUPS[0], {"computed": 1599, "scheduled": 1}), 1)
        self.assertIsNone(pd._running_state_slot(self.GROUPS[1], {"computed": 1023, "scheduled": 1}))

    def test_another_block_or_the_scratch_slot_is_refused(self):
        for table in ([0, 8, 9], [0, 9, 7], [7, 0, 9]):
            bad = self.problems(self.admission(table))
            self.assertTrue(any("is not its running slot 1" in p for p in bad), (table, bad))
        bad = self.problems(self.admission([0, 7, 9], pull_state=(2, [7, 8])))
        self.assertTrue(any("is not its running slot" in p for p in bad), bad)

    def test_a_pulled_state_without_its_admission_step_fails(self):
        # The pull's own place would pass (block 7 at 0), but its running slot
        # with 799 + 2 tokens is 1 (block 9): without the step it is unknown.
        for pull, table in (((0, [7]), [7, 9, 10]), ((2, [7]), [0, 7, 9])):
            bad = self.problems(self.admission(table, pull_state=pull, computed=799, scheduled=None))
            self.assertTrue(any("has no admission step" in p for p in bad), (pull, bad))
        bad = self.problems(self.admission([7, 9, 10], pull_state=(0, [7]), computed=799, scheduled=2))
        self.assertTrue(any("is not its running slot 1" in p for p in bad), bad)

    @staticmethod
    def records(admissions, delay=0.0):
        """A decode step log: each admission's step (logged first) and then
        its events, naming that step (``step_index``); scheduled at
        ``t_scheduled``, the step's modeled start ``delay`` later (a step
        queued behind the device)."""
        out = [{"kv_layout": {"groups": MtpStatePullTests.GROUPS}}]
        for index, (computed, scheduled, table) in enumerate(admissions):
            scheduled_at = 100.0 + index
            out.append({"step": {"step_index": index, "requests": [
                {"req_id": "r", "num_computed_tokens": computed, "num_scheduled_tokens": scheduled,
                 "num_prompt_tokens": 1024}]}, "t_start": scheduled_at + delay})
            out.append({"t_scheduled": scheduled_at, "step_index": index - 1, "kv_events": {
                "reads_posted": ["r"],
                "allocated_groups": {"r": [{"group": 0, "first": 2, "blocks": [7]},
                                           {"group": 1, "first": 0, "blocks": [4, 5]}]}}})
            out.append({"t_scheduled": scheduled_at, "step_index": index, "kv_events": {
                "admitted": {"r": computed},
                "allocated_groups": {"r": [{"group": 0, "first": 0, "blocks": table},
                                           {"group": 1, "first": 0, "blocks": [4, 5]}]}}})
        return out

    def test_the_step_is_its_index_not_its_modeled_start(self):
        # Scheduled at 100.0, modeled to start at 100.01: still its step.
        good = pd._admissions(self.records([(799, 2, [0, 7, 9])], delay=0.01))
        self.assertEqual(good["r"][0]["scheduled"], 2)
        self.assertEqual(self.problems(good), [])
        bad = self.problems(pd._admissions(self.records([(799, 2, [7, 9, 10])], delay=0.01)))
        self.assertTrue(any("is not its running slot 1" in p for p in bad), bad)

    def test_each_admission_takes_its_own_step(self):
        # Two admissions at 799 computed, scheduled 1 then 2: slots 0 and 1.
        good = pd._admissions(self.records([(799, 1, [7, 9]), (799, 2, [0, 7, 9])]))
        self.assertEqual([a["scheduled"] for a in good["r"]], [1, 2])
        self.assertEqual(self.problems(good), [])
        # The second still in slot 0, as the first step's count would allow.
        bad = self.problems(pd._admissions(self.records([(799, 1, [7, 9]), (799, 2, [7, 0, 9])])))
        self.assertTrue(any("is not its running slot 1" in p for p in bad), bad)

    def test_a_missing_or_ambiguous_step_is_none(self):
        records = self.records([(1023, 1, [0, 7, 9])])
        missing = [r for r in records if "step" not in r]
        self.assertIsNone(pd._admissions(missing)["r"][0]["scheduled"])
        twice = records + [dict(records[1])]
        self.assertIsNone(pd._admissions(twice)["r"][0]["scheduled"])
        unnamed = [{k: v for k, v in r.items() if k != "step_index"} for r in records]
        self.assertIsNone(pd._admissions(unnamed)["r"][0]["scheduled"])
        bad = self.problems(pd._admissions(unnamed))
        self.assertTrue(any("has no admission step" in p for p in bad), bad)

    def test_a_pulled_state_group_implies_no_decode_hit(self):
        got = pd._hit_problems({"groups": self.GROUPS}, [{"req_id": "r", "kind": "hit", "ok": True,
                                                          "groups": [{"group": 0, "logical_blocks": [1],
                                                                      "block_ids": [7], "blocks": 1,
                                                                      "bytes": 1600}],
                                                          "blocks": 1, "bytes": 1600}],
                               self.admission([0, 7, 9]))[0]
        self.assertTrue(any("logged 1 prefix hit check(s), not the 0" in p for p in got), got)


class EndpointCaptureChecks(unittest.TestCase):
    def test_missing_or_partial_observer_cannot_pass_l3(self):
        from harness.endpoint_observer import check_captures

        run = pd.Run(**{f: getattr(RUN, f) for f in RUN.__dataclass_fields__})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "profiler.json").write_text(json.dumps({"run_id": "run"}))
            verdict = {"problems": []}
            check_captures(output, run, verdict)
            self.assertEqual(len(verdict["problems"]),
                             len(run.prefill.nodes + run.decode.nodes))
            for node in run.prefill.nodes + run.decode.nodes:
                directory = output / "endpoint-probes" / node / "captures/run/process"
                directory.mkdir(parents=True)
                (directory / "manifest.json").write_text(json.dumps({
                    "run_id": "run", "complete": False}))
            verdict = {"problems": []}
            check_captures(output, run, verdict)
            self.assertTrue(all("incomplete" in item for item in verdict["problems"]))


def ebpf_package(root, omit=(), corrupt=(), manifest=None, collector_mode=0o755):
    """A built eBPF package as tools/kv-profiler-ebpf/build.sh lays it out: its files and their manifest."""
    import hashlib

    root = Path(root)
    files = {}
    for name in ("collector", "observe.bpf.o", "limits.json"):
        (root / name).write_bytes(name.encode())
        files[name] = hashlib.sha256(name.encode()).hexdigest()
    (root / "collector").chmod(collector_mode)
    whole = {"schema": 4, "mode": "tcp_tracepoints", "files": files}
    (root / "manifest.json").write_text(json.dumps(whole if manifest is None else manifest(whole)))
    for name in omit:
        (root / name).unlink()
    for name in corrupt:
        (root / name).write_bytes(b"changed")
    return str(root)


class EndpointObserverCapabilities(unittest.TestCase):
    def test_the_observer_keeps_its_endpoint_s_drops_and_inherits_only_ipc_lock(self):
        """Reading another process's /proc/<pid>/ns needs the target's capabilities to be a subset of the
        reader's: an RDMA server's added IPC_LOCK becomes the observer's too. Any other privilege the endpoint
        added is not copied, and every capability it dropped is dropped by the observer as well."""
        from harness.endpoint_observer import observer_command

        def caps(host):
            command = observer_command("o", "srv", {"Image": "img", "HostConfig": host}, [12], "/in", Path("/out"))
            return ([command[i + 1] for i, arg in enumerate(command) if arg == "--cap-add"],
                    [command[i + 1] for i, arg in enumerate(command) if arg == "--cap-drop"])

        self.assertEqual(caps({"CapAdd": ["IPC_LOCK"]}), (["IPC_LOCK"], []))
        self.assertEqual(caps({"CapAdd": ["CAP_IPC_LOCK", "SYS_ADMIN", "NET_ADMIN"], "CapDrop": ["NET_RAW", "MKNOD"]}),
                         (["IPC_LOCK"], ["MKNOD", "NET_RAW"]))
        self.assertEqual(caps({"CapAdd": ["IPC_LOCK"], "CapDrop": ["IPC_LOCK"]}), ([], ["IPC_LOCK"]))
        self.assertEqual(caps({"CapAdd": None, "CapDrop": None}), ([], []))
        self.assertEqual(caps({}), ([], []))

    def test_an_opted_in_ebpf_endpoint_gives_its_observer_only_bpf_perfmon_and_the_artifacts(self):
        """The opt-in eBPF collector needs CAP_BPF and CAP_PERFMON (never CAP_SYS_ADMIN) and its built artifact
        directory, read-only at its configured absolute path; without the explicit opt-in the observer gets
        neither, and an opted-in endpoint without a usable artifact directory is refused before anything starts."""
        import tempfile

        from harness.endpoint_observer import observer_command

        def parts(probes):
            command = observer_command("o", "srv", {"Image": "img"}, [12], "/in", Path("/out"), probes)
            caps = [command[i + 1] for i, arg in enumerate(command) if arg == "--cap-add"]
            mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "--mount"]
            return caps, mounts

        with tempfile.TemporaryDirectory() as artifacts:
            ebpf_package(artifacts)
            on = {"ebpf": {"enabled": True, "privileged_opt_in": True, "artifacts": artifacts}}
            caps, mounts = parts(on)
            self.assertEqual(caps, ["BPF", "PERFMON"])
            self.assertIn(f"type=bind,src={artifacts},dst={artifacts},readonly", mounts)
            for probes in ({"ebpf": {"enabled": True, "artifacts": artifacts}},
                           {"ebpf": {"enabled": False, "privileged_opt_in": True, "artifacts": artifacts}},
                           {"interfaces": ["data0"]}, None):
                with self.subTest(probes=probes):
                    caps, mounts = parts(probes)
                    self.assertEqual(caps, [])
                    self.assertFalse(any(artifacts in m for m in mounts))
        with tempfile.TemporaryDirectory() as scratch:
            def package(name, **kwargs):
                return ebpf_package(Path(scratch, name).mkdir() or Path(scratch, name), **kwargs)

            empty = str(Path(scratch, "empty").mkdir() or Path(scratch, "empty"))
            broken = [
                package("no-limits", omit=["limits.json"]),
                package("changed", corrupt=["observe.bpf.o"]),
                package("not-executable", collector_mode=0o644),
                package("null-manifest", manifest=lambda whole: None),
                package("list-manifest", manifest=lambda whole: [whole]),
                package("integer-files", manifest=lambda whole: {**whole, "files": 3}),
                package("previous-schema", manifest=lambda whole: {**whole, "schema": 3}),
            ]
            for artifacts in (None, "relative/dir", "/nonexistent/kvp-ebpf", empty, *broken):
                with self.subTest(artifacts=artifacts), self.assertRaisesRegex(ValueError, "probes.ebpf.artifacts"):
                    parts({"ebpf": {"enabled": True, "privileged_opt_in": True, "artifacts": artifacts}})
            # a previous generation's package is named as such, to rebuild
            with self.assertRaisesRegex(ValueError, "package schema 3, not 4: rebuild it"):
                parts({"ebpf": {"enabled": True, "privileged_opt_in": True, "artifacts": broken[-1]}})
        # a string is never consent: quoted booleans are refused, not read as true
        for bad in ({"enabled": "false", "privileged_opt_in": True}, {"enabled": True, "privileged_opt_in": "true"}):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "true or false"):
                parts({"ebpf": {**bad, "artifacts": "/x"}})

    def test_the_run_s_preflight_refuses_an_unusable_ebpf_opt_in_before_anything_starts(self):
        """An opted-in endpoint's artifact directory and literal booleans are checked when the run is parsed,
        before any endpoint is cleaned or server started."""
        raw = {"name": "t", "topology": "t", "model": "/m", "client": "c", "control_network": "control",
               "prefill": {"node": "p", "kv_network": "data0"}, "decode": {"node": "d", "kv_network": "data0"},
               "router": {"gateway": "gw"}, "server": {"connector": "nixl", "timing": "linear"}}

        def profiler(ebpf):
            return {"level": "l3", "observer_cpus": {"p": [12], "d": [13]},
                    "probes": {node: {"interfaces": ["data0"], "ebpf": ebpf} for node in ("p", "d")}}

        with mock.patch.object(pd.models, "resolve", lambda model, fetch=False: Path(model)):
            for ebpf, message in (({"enabled": True, "privileged_opt_in": True}, "probes.ebpf.artifacts"),
                                  ({"enabled": "false", "privileged_opt_in": True}, "true or false")):
                with self.subTest(ebpf=ebpf), self.assertRaisesRegex(ValueError, message):
                    pd.parse({**raw, "profiler": profiler(ebpf)}, Path("/r.yaml"))
            self.assertEqual(pd.parse({**raw, "profiler": profiler({"enabled": False})}, Path("/r.yaml")).profiler.level,
                             "l3")

    def test_an_endpoint_with_another_added_privilege_is_refused_with_its_reason(self):
        """An endpoint holding an added capability the observer does not take would make its namespaces
        unreadable: the observer refuses it by name, before anything starts, instead of failing on /proc."""
        from harness.endpoint_observer import uninherited_capabilities

        self.assertEqual(uninherited_capabilities({"HostConfig": {"CapAdd": ["IPC_LOCK"], "CapDrop": ["NET_RAW"]}}),
                         [])
        self.assertEqual(uninherited_capabilities({"HostConfig": {"CapAdd": ["IPC_LOCK", "CAP_NET_ADMIN"]}}),
                         ["NET_ADMIN"])
        self.assertEqual(uninherited_capabilities({"HostConfig": {"CapAdd": ["NET_ADMIN"], "CapDrop": ["NET_ADMIN"]}}),
                         [])


class EndpointObserverExitRecord(unittest.TestCase):
    """The Docker launcher publishes the deployment-neutral exit record only after reconciling its evidence."""

    def stop_with(self, inspect, waited="0", removed=True, captures=1):
        from harness.endpoint_observer import EndpointObserver, bundle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            output.mkdir()
            bundle(output / "observer.pyz")
            for index in range(captures):
                folder = output / "captures" / "run" / f"attempt-{index}"
                folder.mkdir(parents=True)
                (folder / "manifest.json").write_text(json.dumps({"process_id": f"attempt-{index}"}))
            (root / "inspect.json").write_text(json.dumps(inspect))
            docker = root / "docker"
            docker.write_text(
                f"#!{sys.executable}\n"
                "import pathlib, sys\n"
                "action = sys.argv[1]\n"
                f"if action == 'wait': print({waited!r})\n"
                f"elif action == 'inspect': print(pathlib.Path({str(root / 'inspect.json')!r}).read_text())\n"
                f"elif action == 'rm': sys.exit({0 if removed else 1})\n"
            )
            docker.chmod(0o755)
            observer = EndpointObserver("observer", output, "run", "decode", "owned-observer")
            with mock.patch.dict(os.environ, {"PATH": str(root)}):
                observer.stop(True)
            return json.loads((output / "exit.json").read_text())

    def state(self, **override):
        return [{"Id": "owned-observer", "State": {"ExitCode": 0, "Running": False, "OOMKilled": False, **override}}]

    def test_a_reconciled_clean_exit_is_recorded_for_this_attempt(self):
        record = self.stop_with(self.state())
        self.assertEqual((record["exit_code"], record["killed"], record["problems"], record["process_id"],
                          record["launcher"]), (0, None, [], "attempt-0", "docker"))

    def test_disagreeing_or_missing_evidence_is_never_a_clean_exit(self):
        cases = {
            "oom": (self.state(OOMKilled=True, ExitCode=137), "137", True, 1),
            "another_container": ([{"Id": "other", "State": self.state()[0]["State"]}], "0", True, 1),
            "still_running": (self.state(Running=True), "0", True, 1),
            "exit_code_disagrees": (self.state(ExitCode=2), "0", True, 1),
            "not_removed": (self.state(), "0", False, 1),
            "no_capture": (self.state(), "0", True, 0),
            # unknown or mistyped evidence is never a clean exit
            "oom_missing": ([{"Id": "owned-observer", "State": {"ExitCode": 0, "Running": False}}], "0", True, 1),
            "oom_null": (self.state(OOMKilled=None), "0", True, 1),
            "oom_string": (self.state(OOMKilled="false"), "0", True, 1),
            "exit_code_false": (self.state(ExitCode=False), "0", True, 1),
            "exit_code_float": (self.state(ExitCode=0.0), "0", True, 1),
            "running_missing": ([{"Id": "owned-observer", "State": {"ExitCode": 0, "OOMKilled": False}}], "0", True,
                                1),
            "no_state": ([{"Id": "owned-observer"}], "0", True, 1),
            "empty_state": ([{"Id": "owned-observer", "State": {}}], "0", True, 1),
        }
        for name, (inspect, waited, removed, captures) in cases.items():
            with self.subTest(name):
                record = self.stop_with(inspect, waited, removed, captures)
                if name == "oom":
                    self.assertEqual(record["killed"], "oom")
                elif name == "no_capture":
                    self.assertIsNone(record["process_id"])
                else:
                    self.assertTrue(record["problems"], record)


class EndpointObserverDiagnostics(unittest.TestCase):
    def test_stderr_only_failure_is_preserved_without_masking_exit_or_cleanup(self):
        """A fake Docker CLI keeps real subprocess stream handling, without Docker."""
        from harness.endpoint_observer import EndpointObserver, bundle

        for logs_exit in (0, 23):
            with (
                self.subTest(logs_exit=logs_exit),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                output = root / "output"
                output.mkdir()
                docker = root / "docker"
                docker.write_text(
                    f"#!{sys.executable}\n"
                    "import pathlib, sys\n"
                    "assert sys.argv[-1] == 'owned-observer'\n"
                    "action = sys.argv[1]\n"
                    "if action == 'wait': print(1)\n"
                    "elif action == 'logs':\n"
                    "    sys.stderr.buffer.write(b'only-stderr: startup failed\\n')\n"
                    f"    sys.exit({logs_exit})\n"
                    "elif action == 'inspect': print('[]')\n"
                    "elif action == 'rm':\n"
                    f"    pathlib.Path({str(root / 'removed')!r}).touch()\n"
                    "else: sys.exit(97)\n"
                )
                docker.chmod(0o755)
                bundle(output / "observer.pyz")
                observer = EndpointObserver(
                    "observer", output, "run", "decode", "owned-observer"
                )
                with mock.patch.dict(os.environ, {"PATH": str(root)}):
                    observer.stop(False)
                self.assertEqual((output / "observer.stdout.log").read_bytes(), b"")
                stderr = b"only-stderr: startup failed\n"
                self.assertEqual((output / "observer.stderr.log").read_bytes(), stderr)
                self.assertIn(stderr, (output / "observer.log").read_bytes())
                record = json.loads((output / "exit.json").read_text())
                self.assertEqual(record["exit_code"], 1)
                self.assertEqual(record["evidence"]["logs_exit_code"], logs_exit)
                self.assertFalse(record["timed_out"])
                self.assertTrue(record["evidence"]["removed"])
                self.assertIn("the container could not be inspected", record["problems"])
                self.assertTrue((root / "removed").exists())


class EndpointObserverAccounting(unittest.TestCase):
    def test_observer_is_not_a_server_child_and_its_cpu_counts_as_a_helper(self):
        command = (b"/opt/venv/bin/python\0/installed/vllm/kv_profiling/"
                   b"probe.py\0--config\0x")
        self.assertFalse(pd.is_service(command, b"python"))
        # This is also the exact function embedded in the server stop scripts.
        self.assertIn("endpoint controller", pd.STOP_OWNERS)
        samples = [sample(float(t), (1, 1, "serving", float(t)),
                          (2, 2, "probe", 0.1 * t)) for t in range(5)]
        report = pd.cpu_over(samples, 1.0, 3.0)
        self.assertAlmostEqual(report["serving_s"], 2.0)
        self.assertAlmostEqual(report["helpers_s"], 0.2)
        self.assertAlmostEqual(report["probes_s"], 0.2)
        self.assertAlmostEqual(report["probes_at_end_s"], 0.3)


class EndpointCaptureIntegrity(unittest.TestCase):
    def create(self, output, frontends=1, owner=(1, 100), recorded_owner=None):
        """A complete synthetic L3 capture. ``frontends`` API servers (each its
        own process when there are several, as under data parallelism), the
        observer configured with ``owner`` and recording ``recorded_owner``."""
        import hashlib
        import uuid
        from harness.endpoint_observer import bundle

        run_id = str(uuid.uuid4())
        probes = {
            node: {"interfaces": ["lo"], "required": ["netdev"]} for node in ("p", "d")
        }
        profiler = pd.Profiler(
            level="l3", probes=probes, observer_cpus={"p": [12], "d": [13]}
        )
        run = pd.Run(
            **{
                **{f: getattr(RUN, f) for f in RUN.__dataclass_fields__},
                "profiler": profiler,
            }
        )
        (output / "profiler.json").write_text(json.dumps({"run_id": run_id}))
        for role, node in (("prefill", "p"), ("decode", "d")):
            directory = output / "endpoint-probes" / node
            directory.mkdir(parents=True)
            source_root = Path(os.environ.get("KV_PROFILER_SOURCE_ROOT", pd.REPOSITORY))
            digest = bundle(directory / "observer.pyz", source_root=source_root)
            entries = []
            roles = ["frontend"] * frontends + ["scheduler", "worker"]
            for index, capture_role in enumerate(roles):
                process = str(uuid.uuid4())
                pid, start = ((10 + index, 200 + index)
                              if capture_role == "frontend" and frontends > 1
                              else (1, 100))  # fmt: skip
                meta = {
                    "schema": 1,
                    "type": "metadata",
                    "run_id": run_id,
                    "process_id": process,
                    "pid": pid,
                    "start_ticks": start,
                    "clock_id": str(uuid.uuid4()),
                    "clock_source": "CLOCK_MONOTONIC",
                    "boot_id": "boot",
                    "net_namespace": "net",
                    "pid_namespace": "pid",
                    "time_namespace": "time",
                    "context": {"role": capture_role, "engine_id": node, "rank": 0,
                                "kv_role": "kv_producer" if role == "prefill" else "kv_consumer"},
                }
                entry = {
                    key: value
                    for key, value in meta.items()
                    if key not in ("schema", "type")
                }
                entries.append(entry)
                folder = output / "profiles" / role / node / run_id / process
                folder.mkdir(parents=True)
                end = {
                    "schema": 1,
                    "type": "end",
                    "complete": True,
                    "stats": {
                        "head": 0,
                        "tail": 0,
                        **dict.fromkeys(
                            (
                                "drop_full",
                                "drop_contention",
                                "drop_ids",
                                "drop_handles",
                                "drop_identity",
                                "drop_conflict",
                                "drop_closed",
                                "unknown_handles",
                            ),
                            0,
                        ),
                    },
                }
                (folder / "events.jsonl").write_text(
                    json.dumps(meta) + "\n" + json.dumps(end) + "\n"
                )
                (folder / "manifest.json").write_text(
                    json.dumps(
                        {**meta, "saved": True, "complete": True, "stats": end["stats"]}
                    )
                )
            counts = {r: sum(e["context"]["role"] == r for e in entries) for r in ("frontend", "scheduler", "worker")}
            import subprocess as digest_run

            code_sha256 = digest_run.check_output(
                [sys.executable, str(directory / "observer.pyz"), "--code-digest"], text=True).strip()
            config = {
                "run_id": run_id,
                "endpoint": node,
                "role": role,
                "probes": probes[node],
                "cpus": profiler.observer_cpus[node],
                "package_sha256": digest,
                "expected": counts,
            }
            (directory / "config.json").write_text(json.dumps(config))
            # Get normalized defaults from the same immutable, stdlib-only package.
            import subprocess

            script = (
                "import sys,json;sys.path.insert(0,sys.argv[1]);"
                "from endpoint_probes.config import ProbeConfig,settings_dict;"
                "print(json.dumps(settings_dict(ProbeConfig(**json.loads(sys.argv[2]))),"
                "sort_keys=True))"
            )
            normalized = subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(directory / "observer.pyz"),
                    json.dumps(probes[node]),
                ],
                text=True,
            )
            process = str(uuid.uuid4())
            meta = {
                "schema": 1,
                "probe_schema": 1,
                "capture_format": 2,
                "type": "metadata",
                "run_id": run_id,
                "process_id": process,
                "context": {
                    "endpoint": node,
                    "endpoint_role": role,
                    "role": "endpoint_observer",
                },
                "clock_id": str(uuid.uuid4()),
                "clock_source": "CLOCK_MONOTONIC",
                "boot_id": "boot",
                "net_namespace": "net",
                "pid_namespace": "pid",
                "time_namespace": "time",
                "config_sha256": hashlib.sha256(
                    normalized.strip().encode()
                ).hexdigest(),
                "inventory": entries,
                "expected": counts,
                # what discovery froze: each frontend's parent as read
                "owner_evidence": {"frontends": [], "parent": None} if frontends == 1 else {
                    "frontends": [{"pid": e["pid"], "start_ticks": e["start_ticks"], "parent_pid": owner[0]}
                                  for e in entries if e["context"]["role"] == "frontend"],
                    "parent": {"pid": owner[0], "start_ticks": owner[1]}},
                "code_sha256": code_sha256,
                "affinity": profiler.observer_cpus[node],
                "owner": {
                    "pid": (recorded_owner or owner)[0],
                    "start_ticks": (recorded_owner or owner)[1],
                    "net_namespace": "net",
                    "pid_namespace": "pid",
                },
                "package_sha256": digest,
            }
            events = []
            for sequence, (scope, observation) in enumerate(
                (
                    ("cpu_isolation", {"verified": True}),
                    ("observer", {"cpu_s": 0.1}),
                    ("netdev", {"fields": {}}),
                ),
                1,
            ):
                events.append(
                    {
                        "schema": 1,
                        "probe_schema": 1,
                        "type": "event",
                        "kind": "probe_resource"
                        if scope == "observer"
                        else "probe_status",
                        "scope": scope,
                        "entity": node,
                        "time_ns": sequence,
                        "sample_begin_ns": sequence,
                        "sample_end_ns": sequence,
                        "sequence": sequence,
                        "request_ref": 0,
                        "handoff_ref": 0,
                        "transfer": 0,
                        "peer_ref": 0,
                        **dict.fromkeys(
                            (
                                "bytes",
                                "selected_descriptors",
                                "merged_descriptors",
                                "duration0_ns",
                                "duration1_ns",
                                "aux0",
                                "aux1",
                            )
                        ),
                        "transport": {"kind": "unknown", "tcp": None, "rdma": None},
                        "observation": observation,
                    }
                )
            end = {
                "schema": 1,
                "probe_schema": 1,
                "type": "end",
                "complete": True,
                "reason": "controller_stop",
                "stats": {
                    "head": 3,
                    "tail": 3,
                    "drop_probe_rows": 0,
                    "isolation_failures": 0,
                    "required_source_failures": 0,
                    "observation_gaps": 0,
                    "samples": 1,
                    "controller_children_closed": True,
                    "core": {"all_closed": True},
                },
            }
            folder = directory / "captures" / run_id / process
            folder.mkdir(parents=True)
            (folder / "events.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in [meta, *events, end])
            )
            (folder / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "process_id": process,
                        "saved": True,
                        "complete": True,
                        "end": end,
                    }
                )
            )
            # what a launcher publishes after reaping the observer
            (directory / "exit.json").write_text(json.dumps({
                "schema": 1, "run_id": run_id, "endpoint": node, "process_id": process, "exit_code": 0,
                "signal": None, "killed": None, "timed_out": False, "launcher": "test", "problems": [],
                "evidence": {}}))
        return run

    def test_complete_flags_do_not_replace_capture_contents_or_matching_identity(self):
        import shutil
        from harness.endpoint_observer import check_captures

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean = root / "clean"
            clean.mkdir()
            run = self.create(clean)
            verdict = {"problems": []}
            check_captures(clean, run, verdict)
            self.assertEqual(verdict["problems"], [])
            for fault in (
                "missing_events",
                "bad_json",
                "swapped",
                "stale",
                "exit",
                "timeout",
                "killed",
                "signal",
                "unreconciled",
                "copied_from_another_attempt",
                "malformed_exit",
                "missing_exit_with_docker_records",
                "earlier_capture_format",
            ):
                out = root / fault
                shutil.copytree(clean, out)
                folder = out / "endpoint-probes/d"
                if fault == "missing_events":
                    next((folder / "captures").glob("*/*/events.jsonl")).unlink()
                elif fault == "bad_json":
                    next((folder / "captures").glob("*/*/manifest.json")).write_text(
                        "["
                    )
                elif fault in ("swapped", "stale"):
                    config = json.loads((folder / "config.json").read_text())
                    config["endpoint" if fault == "swapped" else "run_id"] = "wrong"
                    (folder / "config.json").write_text(json.dumps(config))
                elif fault == "malformed_exit":
                    (folder / "exit.json").write_text(json.dumps({"schema": 1, "exit_code": "0"}))
                elif fault == "missing_exit_with_docker_records":
                    # a runtime's own records never stand in for the exit record
                    (folder / "exit.json").unlink()
                    (folder / "container.json").write_text(json.dumps(
                        [{"Id": "d", "State": {"ExitCode": 0, "Running": False, "OOMKilled": False}}]))
                    (folder / "endpoint-status.json").write_text(json.dumps(
                        {"run_id": "the-run", "endpoint": "d", "exit_code": 0, "removed": True, "timed_out": False}))
                elif fault == "earlier_capture_format":
                    events = next((folder / "captures").glob("*/*/events.jsonl"))
                    rows = [json.loads(line) for line in events.read_text().splitlines()]
                    del rows[0]["capture_format"]
                    events.write_text("".join(json.dumps(row) + "\n" for row in rows))
                else:
                    record = json.loads((folder / "exit.json").read_text())
                    record.update({
                        "exit": {"exit_code": 1}, "timeout": {"timed_out": True}, "killed": {"killed": "oom"},
                        "signal": {"exit_code": None, "signal": 9}, "unreconciled": {"problems": ["not removed"]},
                        "copied_from_another_attempt": {"process_id": "an-earlier-observer"},
                    }[fault])
                    (folder / "exit.json").write_text(json.dumps(record))
                verdict = {"problems": []}
                check_captures(out, run, verdict)
                self.assertTrue(verdict["problems"], fault)


    def test_an_observer_of_several_api_servers_follows_their_parent(self):
        """With data parallelism each API server is a child process with its
        frontend recorder: the observer's owner is their common parent, the
        process the controller configured. A recorded owner other than the
        configured one fails, and with a single API server the owner is still
        the frontend itself."""
        from harness.endpoint_observer import check_captures

        cases = (
            ("parent of two API servers", {"frontends": 2}, True),
            ("the observer's recorders are not the expected ones", {"frontends": 2}, False),
            ("one API server, owner not the frontend",
             {"frontends": 1, "owner": (5, 500)}, False),
        )  # fmt: skip
        for label, kwargs, clean in cases:
            with self.subTest(label), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                run = self.create(output, **kwargs)
                if label.startswith("the observer's recorders"):
                    for node in ("p", "d"):
                        path = output / "endpoint-probes" / node / "config.json"
                        config = json.loads(path.read_text())
                        config["expected"]["frontend"] = 3
                        path.write_text(json.dumps(config))
                verdict = {"problems": []}
                check_captures(output, run, verdict)
                self.assertEqual(verdict["problems"] == [], clean, verdict["problems"])


class EndpointProfileBacking(unittest.TestCase):
    def test_readonly_input_uses_the_exposed_backing_not_a_proc_magic_link(self):
        from harness.endpoint_observer import profile_backing

        description = {"Mounts": [
            {"Type": "bind", "Destination": "/e2e", "Source": "/shared"},
            {"Type": "volume", "Destination": "/e2e/profiles", "Source": "/captures"}],
            "GraphDriver": {"Data": {"MergedDir": "/overlay/merged"}}}
        self.assertEqual(profile_backing(description), ("/captures", "existing_mount"))
        self.assertEqual(profile_backing({"GraphDriver": description["GraphDriver"]}),
                         ("/overlay/merged/e2e/profiles", "overlay_merged"))
        with self.assertRaisesRegex(ValueError, "exposed profile"):
            profile_backing({"GraphDriver": {"Data": {}}})
