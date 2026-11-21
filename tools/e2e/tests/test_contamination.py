"""The cheap contamination monitor: the judgement on fabricated readings (a foreign load at and around each threshold,
the review's timing-mismatch counterexample, inconsistent accounting, resets, gaps, slow reads, missing coverage and
identities) and the sampler process itself on a fake cgroup directory."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from harness import contamination as c

NS = 1_000_000_000
CPUS = "0-13"
IDS = ["a" * 64, "b" * 64]


LIFE = {"started": True, "returncode": 0, "forced_kill": False, "reaped": True}


def J(rows, window, start, end, **kwargs):
    return c.judge(rows, window, start, end, **{"lifecycle": LIFE, **kwargs})


def owned(ids=IDS, cpus=CPUS):
    return {cid: {"path": f"/sys/fs/cgroup/system.slice/docker-{cid}.scope", "started": "t0", "cpus": cpus} for cid in ids}


def series(seconds=20, *, foreign=0.0, owned_cpu=3.0, hold=None, delays=None, start_ns=100 * NS, step=1.0):
    """Sampler records at one-second intervals: the measured CPUs' busy time grows by ``owned_cpu + foreign`` CPU-s a
    second, the owned cgroups' by ``owned_cpu``. ``foreign`` may be a function of the second; ``delays[i]`` delays sample i's
    cgroup reading (a read that lands later) by that many seconds, during which the owned cgroups keep working."""
    rows, busy_total, own_total = [], 0.0, 0.0
    for i in range(seconds + 1):
        t = start_ns + int(i * step * NS)
        f = foreign(i) if callable(foreign) else foreign
        if i:
            busy_total += owned_cpu + f
            own_total += owned_cpu
        delay = (delays or {}).get(i, 0.0)
        own_read = own_total + owned_cpu * delay  # the owned counters read late include the extra work
        busy_each = busy_total / 14
        cpu = {str(k): [int(busy_each / c.TICK_S), 0, 0, 1000, 0, 0, 0, 0] for k in range(14)}
        share = own_read / len(IDS)
        cg = {cid: {"user_usec": int(share * 1e6 * 0.8), "system_usec": int(share * 1e6 * 0.2), "nr_throttled": 0} for cid in IDS}
        rows.append({"i": i, "h": [t, t + 1_000_000], "c": [t + 1_000_000 + int(delay * NS), t + 2_000_000 + int(delay * NS)],
                     "cpu": cpu, "cg": cg})
    return rows


def window_of(rows, margin=3):
    return rows[margin]["h"][1] + 10, rows[-margin - 1]["h"][0] - 10


class Judgement(unittest.TestCase):
    def judge(self, rows, **kwargs):
        return J(rows, window_of(rows), owned(), owned(), **kwargs)

    def test_a_run_with_only_owned_work_is_clean(self):
        result = self.judge(series())
        self.assertEqual((result["status"], result["reasons"]), ("clean", []))
        self.assertAlmostEqual(result["mean_foreign_cpu"], 0.0, places=1)
        self.assertGreater(result["owned_cpu_mean"], 2.5)

    def test_a_decisive_foreign_load_is_contaminated_by_the_sample_threshold_and_by_the_mean(self):
        burst = self.judge(series(foreign=lambda i: 3.0 if i == 10 else 0.0))
        self.assertEqual(burst["status"], "contaminated")
        self.assertTrue(any("one-second sample" in r for r in burst["reasons"]), burst["reasons"])
        steady = self.judge(series(foreign=0.5))  # under 0.75 in every sample, above 0.20 on average
        self.assertEqual(steady["status"], "contaminated")
        self.assertTrue(any("mean foreign busy" in r for r in steady["reasons"]), steady["reasons"])
        self.assertFalse(any("one-second sample" in r for r in steady["reasons"]))

    def test_a_load_just_under_the_thresholds_is_unknown_never_clean(self):
        near = self.judge(series(foreign=0.19))
        self.assertIn(near["status"], ("unknown", "contaminated"), near)
        self.assertNotEqual(near["status"], "clean")

    def test_the_reviews_counterexample_a_late_owned_read_is_unknown_not_a_clean_07_cpu(self):
        # one 1 s interval: 1.0 CPU-s of owned work and 0.9 foreign; the owned read of the closing sample lands 200 ms late and
        # collects another 0.2 CPU-s, so the plain subtraction reports 0.7 CPU, under the 0.75 threshold
        rows = series(owned_cpu=1.0, foreign=lambda i: 0.85 if i == 11 else 0.0, delays={11: 0.2})
        a, b = rows[10], rows[11]
        def host(sample):
            return sum(r[0] + r[1] + r[2] for r in c._busy_rows(sample).values()) * c.TICK_S

        def own(sample):
            return sum(r["user_usec"] + r["system_usec"] for r in c._cg_rows(sample, IDS).values()) / 1e6

        naive = (host(b) - host(a)) - (own(b) - own(a))
        self.assertTrue(0.55 < naive < c.SAMPLE_THRESHOLD_CPU, f"the plain subtraction would pass it ({naive})")
        result = self.judge(rows)
        self.assertNotEqual(result["status"], "clean", result)
        self.assertGreaterEqual(result["decisions"]["unknown"], 1, "the uncertain sample stays unknown")
        # the same interval with prompt reads is decided (clean at 0.3 CPU of foreign work, contaminated at 0.9)
        prompt = series(owned_cpu=1.0, foreign=lambda i: 1.4 if i == 11 else 0.0)
        self.assertEqual(self.judge(prompt)["status"], "contaminated")

    def test_inconsistent_accounting_is_unknown_not_clamped_into_cleanliness(self):
        rows = series(owned_cpu=3.0)
        for row in rows[8:]:  # the owned cgroups report far more than the host's busy time
            for cid in IDS:
                row["cg"][cid]["user_usec"] += 20_000_000
        result = self.judge(rows)
        self.assertNotEqual(result["status"], "clean", result)

    def test_resets_missing_counters_gaps_and_slow_reads_are_unknown(self):
        rows = series()
        rows[10]["cg"][IDS[0]]["user_usec"] = 0
        rows[10]["cg"][IDS[0]]["system_usec"] = 0
        self.assertEqual(self.judge(rows)["status"], "unknown")
        rows = series()
        rows[10]["cg"][IDS[1]] = None
        self.assertTrue(any("could not be read" in r for r in self.judge(rows)["reasons"]))
        rows = series()
        del rows[10]["cpu"]["7"]
        self.assertTrue(any("measured CPU is missing" in r for r in self.judge(rows)["reasons"]))
        gap = series()
        for row in gap[10:]:  # every sample from the tenth on is read 4 s later than its predecessor was
            row["h"] = [v + 4 * NS for v in row["h"]]
            row["c"] = [v + 4 * NS for v in row["c"]]
        self.assertTrue(any("gap of" in r for r in J(gap, (gap[3]["h"][1] + 1, gap[-4]["h"][0] - 1), owned(), owned())["reasons"]))
        slow = series()
        for row in slow[3:9]:
            row["c"][1] += 400_000_000
        self.assertTrue(any("took more than 250 ms to read" in r for r in self.judge(slow)["reasons"]))

    def test_throttling_is_contamination(self):
        rows = series()
        for row in rows[10:]:
            row["cg"][IDS[0]]["nr_throttled"] = 5
        result = self.judge(rows)
        self.assertEqual(result["status"], "contaminated")
        self.assertTrue(any("throttled" in r for r in result["reasons"]))

    def test_coverage_of_both_ends_of_the_clients_benchmark_is_required(self):
        rows = series()
        begin, end = window_of(rows)
        for bad in ((rows[0]["h"][0] - NS, end), (begin, rows[-1]["h"][1] + NS)):
            result = J(rows, bad, owned(), owned())
            self.assertEqual(result["status"], "unknown")
            self.assertTrue(any("bracket both ends" in r for r in result["reasons"]))
        # a step-only interval inside the window is not what is judged: the whole client window is
        self.assertEqual(J(rows, (rows[5]["h"][1] + 1, rows[-6]["h"][0] - 1), owned(), owned())["status"], "clean")

    def test_the_owned_set_must_be_disjoint_stable_and_inside_the_measured_cpus(self):
        rows = series()
        window = window_of(rows)
        nested = owned()
        nested[IDS[1]]["path"] = nested[IDS[0]]["path"] + "/child"
        self.assertTrue(any("not disjoint" in r for r in J(rows, window, nested, nested)["reasons"]))
        same = owned()
        same[IDS[1]]["path"] = same[IDS[0]]["path"]
        self.assertTrue(any("share a cgroup" in r for r in J(rows, window, same, same)["reasons"]))
        changed = owned()
        changed[IDS[0]]["started"] = "t1"
        self.assertTrue(any("restarted" in r for r in J(rows, window, owned(), changed)["reasons"]))
        gone = owned(ids=IDS[:1])
        self.assertTrue(any("not the same at both ends" in r for r in J(rows, window, owned(), gone)["reasons"]))
        for cpus in ("0-15", "14", "", "x", ",", "-"):
            wide = owned(cpus=cpus)
            self.assertEqual(J(rows, window, wide, wide)["status"], "unknown", cpus)
        self.assertEqual(J(rows, window, {}, {})["status"], "unknown")

    def test_irq_and_softirq_time_is_reported_never_counted(self):
        rows = series()
        for row in rows:
            for cpu in row["cpu"].values():
                cpu[6] = row["i"] * 50  # softirq jiffies grow
        result = self.judge(rows)
        self.assertEqual(result["status"], "clean")
        self.assertGreater(result["other_cpu_mean"], 1.0)


class Review48(unittest.TestCase):
    """The five blocking findings of the review of agent-3#48, one regression each (and the neighbours that must stay unknown)."""

    def judge(self, rows, **kwargs):
        return J(rows, window_of(rows), owned(), owned(), **kwargs)

    def test_a_malformed_or_truncated_record_is_an_error_and_never_turns_a_burst_into_clean(self):
        rows = series(seconds=300, foreign=lambda i: 1.5 if i == 100 else 0.0, owned_cpu=3.0)
        self.assertEqual(self.judge(rows)["status"], "contaminated", "the burst is visible when the recording is intact")
        reviews = series(seconds=300, foreign=lambda i: 1.12 if i == 100 else 0.0, owned_cpu=3.0)
        self.assertNotEqual(self.judge(reviews)["status"], "clean", "the review's 1.12 CPU burst is never clean")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            lines = [json.dumps(r) for r in rows]
            lines[100] = lines[100][:40]  # the record holding the burst is corrupted
            path.write_text("\n".join(lines) + "\n")
            kept, errors = c.read_samples(path)
            self.assertEqual((len(kept), len(errors)), (len(rows) - 1, 1))
            judged = J(kept, window_of(rows), owned(), owned(), read_errors=errors)
            self.assertEqual(judged["status"], "unknown")
            self.assertTrue(any("not valid JSON" in r for r in judged["reasons"]))
            self.assertNotEqual(c.judge(kept, window_of(rows), owned(), owned(), lifecycle=LIFE)["status"], "clean",
                                "even without the reader's error list the missing sequence entry is found")
            path.write_text("\n".join(json.dumps(r) for r in rows))  # no final newline: a truncated last record
            _, errors = c.read_samples(path)
            self.assertTrue(any("truncated" in e for e in errors))
        self.assertEqual(c.read_samples("/nonexistent/samples.jsonl")[0], [])
        self.assertTrue(c.read_samples("/nonexistent/samples.jsonl")[1])

    def test_missing_duplicate_unordered_and_invalid_records_are_never_repaired(self):
        base = series(seconds=300)
        missing = [r for r in base if r["i"] != 100]
        duplicate = base[:50] + [base[49]] + base[50:]
        swapped = list(base)
        swapped[60], swapped[61] = swapped[61], swapped[60]
        bad_instant = series(seconds=300)
        bad_instant[70]["h"] = [5 * NS, 4 * NS]
        backwards = series(seconds=300)
        backwards[80]["h"] = [v - 5 * NS for v in backwards[80]["h"]]
        backwards[80]["c"] = [v - 5 * NS for v in backwards[80]["c"]]
        boolean = series(seconds=300)
        boolean[90]["i"] = True
        notdict = series(seconds=300)
        notdict[95] = ["not", "a", "record"]
        for name, rows in (("missing", missing), ("duplicate", duplicate), ("swapped", swapped), ("bad instant", bad_instant),
                           ("backwards", backwards), ("boolean sequence", boolean), ("not a record", notdict)):
            with self.subTest(name):
                self.assertEqual(J(rows, window_of(base), owned(), owned())["status"], "unknown")
        self.assertEqual(J([], (1, 2), owned(), owned())["status"], "unknown")

    def test_a_reset_is_found_in_the_individual_counter_before_anything_is_summed(self):
        rows = series(seconds=300, owned_cpu=3.0)
        # one cgroup's counter falls from 10.8 s to 2 s while the other's rises by as much: the sums stay equal
        for row in rows[120:]:
            row["cg"][IDS[0]]["user_usec"] -= 8_800_000
            row["cg"][IDS[1]]["user_usec"] += 8_800_000
        judged = self.judge(rows)
        self.assertEqual(judged["status"], "unknown")
        self.assertTrue(any("went backwards" in r and "cgroup" in r for r in judged["reasons"]), judged["reasons"])
        rows = series(seconds=300, owned_cpu=3.0)
        for row in rows[120:]:  # one CPU's counter falls by 78 ticks while another's rises by 78: the sums stay equal
            row["cpu"]["3"][0] -= 78
            row["cpu"]["5"][0] += 78
        judged = self.judge(rows)
        self.assertEqual(judged["status"], "unknown")
        self.assertTrue(any("CPU 3 user fell" in r for r in judged["reasons"]), judged["reasons"])
        rows = series(seconds=300)
        for row in rows[120:]:
            row["cg"][IDS[0]]["nr_throttled"] = 5
        for row in rows[130:]:
            row["cg"][IDS[0]]["nr_throttled"] = 2  # the throttling counter falls: a reset as well
        judged = self.judge(rows)
        self.assertNotEqual(judged["status"], "clean")
        self.assertTrue(any("nr_throttled fell" in r for r in judged["reasons"]), judged["reasons"])

    def test_both_ownership_snapshots_are_validated_separately_and_compared(self):
        rows = series()
        window = window_of(rows)
        for name, start, end in (
            ("start 0-15 then 0-13", owned(cpus="0-15"), owned()),
            ("end 0-15", owned(), owned(cpus="0-15")),
            ("empty parsed set at start", owned(cpus=","), owned()),
        ):
            with self.subTest(name):
                self.assertEqual(J(rows, window, start, end)["status"], "unknown")
        moved = owned()
        moved[IDS[0]]["path"] = moved[IDS[0]]["path"] + "-b"
        self.assertTrue(any("changed cgroup" in r for r in J(rows, window, owned(), moved)["reasons"]))
        fewer = owned(cpus="0-7")
        self.assertTrue(any("changed effective CPUs" in r for r in J(rows, window, owned(), fewer)["reasons"]))
        overlapping = owned()
        overlapping[IDS[1]]["path"] = overlapping[IDS[0]]["path"] + "/nested"
        judged = J(rows, window, owned(), overlapping)
        self.assertTrue(any("at the end" in r and "not disjoint" in r for r in judged["reasons"]), judged["reasons"])
        malformed = owned()
        malformed[IDS[0]] = {"path": 3}
        self.assertEqual(J(rows, window, malformed, owned())["status"], "unknown")
        self.assertEqual(J(rows, window, owned(), owned())["status"], "clean")

    def test_the_sampler_shutdown_must_be_confirmed(self):
        for name, life in (("none", None), ("nonzero exit", {**LIFE, "returncode": 7}), ("forced kill", {**LIFE, "forced_kill": True}),
                           ("not reaped", {**LIFE, "reaped": False}), ("never started", {**LIFE, "started": False}),
                           ("no status", {**LIFE, "returncode": None}), ("a string", "ok")):
            with self.subTest(name):
                rows = series()
                self.assertEqual(c.judge(rows, window_of(rows), owned(), owned(), lifecycle=life)["status"], "unknown")

    def test_stop_records_the_exit_status_and_a_forced_kill_and_reaps(self):
        with tempfile.TemporaryDirectory() as directory:
            sampler = c.Sampler(Path(directory) / "a", owned())
            sampler.process = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(7)"])
            life = sampler.stop(timeout_s=10)
            self.assertEqual((life["returncode"], life["forced_kill"], life["reaped"]), (7, False, True))
            self.assertEqual(json.loads((Path(directory) / "a" / "sampler.json").read_text()), life)
            self.assertTrue(c.lifecycle_problems(life))
            stuck = c.Sampler(Path(directory) / "b", owned())
            stuck.process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            life = stuck.stop(timeout_s=0.5)
            self.assertTrue(life["forced_kill"])
            self.assertTrue(life["reaped"])
            self.assertTrue(any("killed" in p for p in c.lifecycle_problems(life)))
            never = c.Sampler(Path(directory) / "c", owned())
            self.assertFalse(never.stop()["started"])

    def test_reap_stops_leftover_samplers_under_a_block_and_only_those(self):
        with tempfile.TemporaryDirectory() as directory:
            block = Path(directory) / "block"
            (block / "1-R" / "monitor").mkdir(parents=True)
            other = Path(directory) / "other" / "monitor"
            other.mkdir(parents=True)
            ours = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(block / "1-R" / "monitor" / "samples.jsonl")])
            theirs = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(other / "samples.jsonl")])
            try:
                report = c.reap_samplers(block)
                self.assertEqual(report["found"], [ours.pid])
                self.assertEqual(report["remaining"], [])
                ours.wait(timeout=10)
                self.assertIsNone(theirs.poll(), "a sampler of another block is not touched")
            finally:
                for process in (ours, theirs):
                    process.kill()
                    process.wait()

    def test_an_orphaned_sampler_exits_by_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "owned.json").write_text("{}")
            launcher = (
                "import subprocess, sys, os\n"
                f"p = subprocess.Popen([sys.executable, '-c', {c.SAMPLER!r}, {str(root / 'samples.jsonl')!r}, {str(root / 'owned.json')!r}, '0.1', str(os.getpid())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                "print(p.pid, flush=True)\nos._exit(0)\n")
            parent = subprocess.run([sys.executable, "-c", launcher], capture_output=True, text=True, timeout=30)
            pid = int(parent.stdout.split()[0])
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and Path(f"/proc/{pid}").exists() and "Z" not in Path(f"/proc/{pid}/stat").read_text().split(")")[-1][:3]:
                time.sleep(0.1)
            alive = Path(f"/proc/{pid}").exists() and "Z" not in Path(f"/proc/{pid}/stat").read_text().split(")")[-1][:3]
            if alive:
                os.kill(pid, 9)
            self.assertFalse(alive, "the orphaned sampler kept running")


class SamplerProcess(unittest.TestCase):
    def test_the_sampler_records_both_readings_and_ends_with_a_sample_that_began_after_the_stop_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cgroup = root / "cg"
            cgroup.mkdir()
            (cgroup / "cpu.stat").write_text("usage_usec 10\nuser_usec 7\nsystem_usec 3\nnr_periods 0\nnr_throttled 0\nthrottled_usec 0\n")
            (root / "owned.json").write_text(json.dumps({"x": str(cgroup), "gone": str(root / "missing")}))
            out = root / "samples.jsonl"
            process = subprocess.Popen([sys.executable, "-c", c.SAMPLER, str(out), str(root / "owned.json"), "0.2"])
            time.sleep(1.0)
            asked = time.monotonic_ns()
            Path(str(out) + ".stop").touch()
            process.wait(timeout=10)
            rows, errors = c.read_samples(out)
            self.assertEqual(errors, [])
        self.assertGreaterEqual(len(rows), 3)
        first = rows[0]
        self.assertEqual(sorted(first["cpu"], key=int), [str(i) for i in range(14)])
        self.assertEqual(first["cg"]["x"], {"user_usec": 7, "system_usec": 3, "nr_throttled": 0})
        self.assertIsNone(first["cg"]["gone"])
        self.assertTrue(first["h"][0] <= first["h"][1] <= first["c"][0] <= first["c"][1])
        self.assertGreaterEqual(rows[-1]["h"][0], asked, "the last sample began after the stop was requested")


class RunIntegration(unittest.TestCase):
    """The harness's side: the workload options and the monitored benchmark, with the sampler faked."""

    RUN = None

    def run_with(self, **workload):
        from harness import pd

        return pd.Run(name="t", topology="topo", model=Path("/m"), prefill=pd.Side("p", "data0"), decode=pd.Side("d", "data0"),
                      client="cl", control_network="control", router=pd.Router("gw"),
                      server=pd.Server(connector="nixl", nixl_backend="ucx", timing="linear"),
                      workload=pd.Workload(num_prompts=2, **workload))

    def test_the_options_become_benchmark_flags_and_are_validated(self):
        from harness import pd

        plain = pd.bench_command(self.run_with(), "http://gw")
        self.assertNotIn("--percentile-metrics", plain)
        self.assertNotIn("--num-warmups", plain)
        only = pd.bench_command(self.run_with(percentiles=True), "http://gw")
        self.assertEqual(only[only.index("--percentile-metrics") + 1], "ttft,tpot,itl,e2el")
        self.assertEqual(only[only.index("--metric-percentiles") + 1], "50,95,99")
        self.assertNotIn("--num-warmups", only)
        raw = {"name": "t", "topology": "t", "model": "/m", "client": "c", "control_network": "control",
               "prefill": {"node": "p", "kv_network": "data0"}, "decode": {"node": "d", "kv_network": "data0"},
               "router": {"gateway": "gw"}, "server": {"connector": "nixl", "timing": "linear"}}
        with mock.patch.object(pd.models, "resolve", lambda model, fetch=False: Path(model)):
            ok = pd.parse({**raw, "workload": {"percentiles": True, "monitor": True}}, Path("/r.yaml"))
            self.assertEqual((ok.workload.percentiles, ok.workload.monitor), (True, True))
            for bad, message in (({"percentiles": "yes"}, "percentiles is true or false"), ({"monitor": 1}, "monitor is true or false")):
                with self.assertRaisesRegex(ValueError, message):
                    pd.parse({**raw, "workload": bad}, Path("/r.yaml"))

    def fakes(self, factory):
        from harness import contamination

        class FakeSampler:
            stopped = False

            def __init__(self, directory, owned):
                Path(directory).mkdir(parents=True, exist_ok=True)
                self.directory = directory
                self.out = Path(directory) / "samples.jsonl"

            def start(self):
                pass

            def wait_first(self):
                pass

            def stop(self):
                FakeSampler.stopped = True
                self.out.write_text("".join(json.dumps(r) + "\n" for r in factory()))
                return dict(LIFE)

        return FakeSampler, [mock.patch.object(contamination, "owned_containers", lambda topology: owned()),
                             mock.patch.object(contamination, "Sampler", FakeSampler)]

    def test_a_monitored_benchmark_is_bracketed_judged_and_saved_and_without_the_monitor_nothing_runs(self):
        from harness import pd

        class Clock:
            """A host clock that advances 10 s on every read, so the benchmark's window spans many samples."""

            now = 1000 * NS

            @classmethod
            def monotonic_ns(cls):
                cls.now += 10 * NS
                return cls.now

        def samples():
            return series(seconds=60, start_ns=1000 * NS)

        fake, patches = self.fakes(samples)
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            patches.append(mock.patch.object(pd, "time", Clock))
            for patch in patches:
                patch.start()
            try:
                window = pd._monitored(self.run_with(monitor=True), out, lambda: calls.append(1))
                judgement = json.loads((out / "monitor" / "judgement.json").read_text())
                saved = json.loads((out / "bench-window.json").read_text())
                for name in ("owned-start.json", "owned-end.json"):
                    self.assertTrue((out / "monitor" / name).is_file())
            finally:
                for patch in patches:
                    patch.stop()
            self.assertEqual(calls, [1])
            self.assertTrue(fake.stopped)
            self.assertEqual(window, saved)
            self.assertLessEqual(saved["begin_ns"], saved["end_ns"])
            self.assertEqual(judgement["status"], "clean", judgement)
            self.assertEqual(judgement["constants"]["mean_threshold_cpu"], 0.20)
            quiet = Path(directory) / "off"
            quiet.mkdir()
            self.assertEqual(pd._monitored(self.run_with(), quiet, lambda: calls.append(2)), {})
            self.assertEqual(sorted(p.name for p in quiet.iterdir()), [])
            self.assertEqual(calls, [1, 2])

    def test_the_sampler_is_stopped_and_the_evidence_saved_however_the_benchmark_ends(self):
        from harness import pd

        fake, patches = self.fakes(lambda: [])
        with tempfile.TemporaryDirectory() as directory:
            for patch in patches:
                patch.start()
            try:
                def broken():
                    raise RuntimeError("the benchmark failed")
                with self.assertRaisesRegex(RuntimeError, "the benchmark failed"):
                    pd._monitored(self.run_with(monitor=True), Path(directory), broken)
            finally:
                for patch in patches:
                    patch.stop()
            window = json.loads((Path(directory) / "bench-window.json").read_text())
            judgement = json.loads((Path(directory) / "monitor" / "judgement.json").read_text())
            self.assertIn("the benchmark failed", window["benchmark_error"])
            self.assertEqual(judgement["status"], "unknown")
            for name in ("owned-start.json", "owned-end.json"):
                self.assertTrue((Path(directory) / "monitor" / name).is_file(), name)
        self.assertTrue(fake.stopped)

    def test_a_monitor_that_cannot_start_or_a_docker_that_does_not_answer_is_saved_as_unknown(self):
        from harness import contamination, pd

        for name, patched in (("start", lambda topology: (_ for _ in ()).throw(RuntimeError("docker is gone"))),):
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                with mock.patch.object(contamination, "owned_containers", patched):
                    with self.assertRaisesRegex(RuntimeError, "docker is gone"):
                        pd._monitored(self.run_with(monitor=True), Path(directory), lambda: None)
                judgement = json.loads((Path(directory) / "monitor" / "judgement.json").read_text())
                self.assertEqual(judgement["status"], "unknown")
                self.assertTrue(any("could not run" in r for r in judgement["reasons"]))
        fake, patches = self.fakes(lambda: [])
        calls = []

        def listing(topology):
            calls.append(1)
            if len(calls) > 1:
                raise RuntimeError("docker stopped answering")
            return owned()

        with tempfile.TemporaryDirectory() as directory:
            patches[0] = mock.patch.object(contamination, "owned_containers", listing)
            for patch in patches:
                patch.start()
            try:
                pd._monitored(self.run_with(monitor=True), Path(directory), lambda: None)
            finally:
                for patch in patches:
                    patch.stop()
            judgement = json.loads((Path(directory) / "monitor" / "judgement.json").read_text())
            self.assertEqual(judgement["status"], "unknown")
            self.assertTrue(any("could not be listed at the end" in r for r in judgement["reasons"]), judgement["reasons"])


if __name__ == "__main__":
    unittest.main()
