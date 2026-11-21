#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV request explorer server: the accepted viewer page over real data.

``python vllm/kv_profiling/viewer.py RUN_DIR [RUN_DIR ...] [--port N]
[--host H]``

Standard library only (no vLLM or torch import: ``vllm/__init__.py`` is never
executed — the profiler core's ``summary``/``clock``/``perfetto``/``schema``
modules are loaded by path under a synthetic package). Captures are found by
their own metadata row, never by directory name: L1 recorder captures feed the
summary's own pairing; L2 collector and L3 observer rows are decoded here, by
schema. Absent sources are reported as absent, never as zero; a metric the
records do not carry is ``unavailable``, never estimated.

Rates always divide a cumulative delta by the elapsed time between
consecutive valid readings of the same series (end to end), never by the
time spent reading the source. Transfer intervals are the connector's own:
acceptance-to-terminal for an asynchronous NIXL READ, call-entry-to-return
for a synchronous Mooncake WRITE, its terminal status qualifying its bytes.
"""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BIN_MS = 100
MAX_THREAD_STRIPS = 32
GAUGE_FIELD = "registered_buffer_bytes"

_PAGE = Path(__file__).resolve().parent / "viewer.html"

# ---------------------------------------------------------------------------
# by-path loading of the profiler core (vllm/__init__.py never executes)
# ---------------------------------------------------------------------------

_PACKAGE = "_kv_profiling_viewer"


def load_core_modules(root=None):
    """Load summary/clock/perfetto/schema by file path under one package."""
    import importlib.util

    root = Path(root or Path(__file__).resolve().parent)
    if _PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            _PACKAGE,
            root / "__init__.py",
            submodule_search_locations=[str(root)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PACKAGE] = module
        spec.loader.exec_module(module)

    def load(name):
        full = f"{_PACKAGE}.{name}"
        if full not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                full, root / f"{name}.py"
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[full] = mod
            spec.loader.exec_module(mod)
        return sys.modules[full]

    return load("summary"), load("clock"), load("perfetto"), load("schema")


SUMMARY, CLOCK, PERFETTO, SCHEMA = load_core_modules()


# ---------------------------------------------------------------------------
# capture discovery
# ---------------------------------------------------------------------------


class ViewerError(Exception):
    """A run cannot be viewed; the message is served as a JSON 500."""


def classify_capture(meta):
    """The layer, by the metadata row only."""
    if meta.get("type") != "metadata":
        raise ViewerError("first row is not a metadata row")
    if meta.get("probe_schema") == 1:
        role = (meta.get("context") or {}).get("role")
        if role != "endpoint_observer":
            raise ViewerError(f"probe metadata with role {role!r}")
        return "l3"
    role = (meta.get("context") or {}).get("role")
    if role == "collector":
        return "l2"
    if "probe_schema" in meta:
        raise ViewerError("probe schema without probe metadata")
    if role not in ("frontend", "scheduler", "worker"):
        raise ViewerError(f"recorder metadata with role {role!r}")
    return "l1"


def _fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Capture:
    layer: str
    path: Path
    meta: dict
    fingerprint: str
    run_id: str
    process_id: str
    lines: int = 0
    folded: list = field(default_factory=list)
    problems: list = field(default_factory=list)

    @property
    def context(self):
        return self.meta.get("context") or {}


def discover(run_dir):
    """Every events.jsonl under the run, folded by metadata identity.

    Same process generation with identical bytes folds into one capture
    (all copies listed); conflicting rows for one process generation is a
    named problem for that capture and excludes it.
    """
    captures = {}
    for path in sorted(Path(run_dir).rglob("events.jsonl")):
        try:
            with path.open() as stream:
                meta = json.loads(stream.readline(1 << 20))
            layer = classify_capture(meta)
        except (OSError, ValueError, ViewerError) as error:
            captures.setdefault(("?", "?", "?"), []).append(
                {
                    "path": str(path),
                    "problem": f"unreadable or unclassifiable: {error}",
                }
            )
            continue
        key = (layer, meta.get("run_id"), meta.get("process_id"))
        data = Capture(layer, path, meta, _fingerprint(path),
                       meta.get("run_id"), meta.get("process_id"))
        prior = captures.setdefault(key, [])
        same = [
            c for c in prior
            if isinstance(c, Capture) and c.fingerprint == data.fingerprint
        ]
        if same:
            same[0].folded.append(str(path))
        elif prior and isinstance(prior[0], Capture):
            prior[0].problems.append(
                f"conflicting rows for this process generation: {path}"
            )
        else:
            prior.append(data)
    folded, skipped = [], []
    for value in captures.values():
        for item in value:
            if isinstance(item, Capture):
                folded.append(item)
            else:
                skipped.append(item)
    return folded, skipped


# ---------------------------------------------------------------------------
# the run view
# ---------------------------------------------------------------------------

_CLOCK = {  # short UI labels derived from recorded roles only
    "kv_producer": "P",
    "kv_consumer": "D",
    "kv_both": "B",
    None: "?",
}


def _label_of(engine_id, registry):
    entry = registry.get(engine_id)
    if entry:
        return entry["label"]
    return f"engine-{str(engine_id or '?')[:8]}"


def engine_registry(captures):
    """engine_id -> label, from recorded contexts only (role letter + rank)."""
    registry = {}
    for capture in captures:
        if capture.layer != "l1":
            continue
        ctx = capture.context
        engine = ctx.get("engine_id")
        if not engine:
            continue
        letter = _CLOCK.get(ctx.get("kv_role"), "P" if ctx.get("role") ==
                            "scheduler" else "?")
        label = f"{letter}{ctx.get('rank', ctx.get('tp_rank', '?'))}"
        registry.setdefault(engine, {
            "label": label, "engine_id": engine,
            "kv_role": ctx.get("kv_role"), "role": ctx.get("role"),
            "process_ids": [],
        })["process_ids"].append(capture.process_id)
    return registry


def _clock_identity(meta):
    return (meta.get("boot_id"), meta.get("time_namespace"),
            meta.get("clock_source"))


class Run:
    """One RUN_DIR: discovery, the L1 summary, clocks and the view payload."""

    def __init__(self, run_dir, run_index=0):
        self.dir = Path(run_dir)
        self.index = run_index
        self.problems = []
        self.build()

    # ------------------------------------------------------------- plumbing
    def _problem(self, kind, detail):
        self.problems.append({"kind": kind, "detail": detail})

    def project(self, clock_id, ts):
        """(projected ns or None, uncertainty ns, drift known)."""
        clock = self.aliases.get(clock_id, clock_id)
        if clock == self.reference_clock:
            return ts, 0, True
        projection = (
            self.group.project(clock, ts)
            if self.group is not None and clock != self.reference_clock
            else None
        )
        if projection is None:
            return None, None, None
        return int(projection.time_ns), projection.uncertainty_ns, (
            projection.drift_known
        )

    def _load_calibrations(self, l1_captures):
        """Mapping files named by ok peers of each capture's manifest."""
        raw = []
        self.calibration_files = []
        self.calibration_status = []
        self.calibration_hashes = {}
        for capture in l1_captures:
            manifest_path = capture.path.parent / "calibration.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, ValueError) as error:
                self._problem("calibration_manifest",
                              f"{manifest_path}: {error}")
                continue
            for peer in manifest.get("peers", []):
                entry = {
                    "process_id": capture.process_id,
                    "local_generation": manifest.get("local_generation"),
                    "remote_generation": peer.get("remote_generation"),
                    "status": peer.get("status"),
                }
                # the manifest itself is a fingerprinted input, wherever it
                # sits
                self.calibration_hashes[
                    str(manifest_path)] = _fingerprint(manifest_path)
                if peer.get("status") != "ok":
                    entry["aligned"] = False
                    self.calibration_status.append(entry)
                    continue
                mapping_file = peer.get("calibration_file")
                path = (manifest_path.parent / mapping_file
                        if mapping_file else None)
                if path is None or not path.exists():
                    entry.update(aligned=False,
                                 reason="named mapping file missing")
                    self.calibration_status.append(entry)
                    continue
                # the mapping the viewer actually loads is fingerprinted at
                # its resolved path (named mappings may sit in subdirectories)
                self.calibration_hashes[str(path)] = _fingerprint(path)
                try:
                    payload = dict(json.loads(path.read_text()))
                    # A start-only mapping (no end anchor) is valid: offset
                    # projection with uncertainty, drift unknown.
                    payload.setdefault("end", [])
                    mapping = CLOCK.ClockMapping.from_json(payload)
                    import tempfile

                    # The summary's own loader requires the key; hand it the
                    # normalized copy.
                    normalized = Path(tempfile.mkdtemp(prefix="kv-viewer-")) \
                        / "mapping.json"
                    normalized.write_text(json.dumps(payload))
                except (OSError, ValueError, KeyError) as error:
                    entry.update(aligned=False, reason=f"unreadable: {error}")
                    self.calibration_status.append(entry)
                    continue
                entry.update(aligned=True, mapping=mapping,
                             drift_known=mapping.end is not None)
                self.calibration_status.append(entry)
                raw.append((entry, mapping, normalized))
        if not raw:
            return {}
        local = {e["local_generation"] for e, _, _ in raw}
        if len(local) != 1 or None in local:
            raise ViewerError(f"mixed calibration local generations: {local}")
        self.reference_clock = next(iter(local))
        unknown = [e for e, _, _ in raw if not e["drift_known"]]
        if unknown:
            self._problem(
                "start_only_calibration",
                f"{len(unknown)} mapping(s) without an end anchor: offset "
                "projection with uncertainty, drift unknown",
            )
        self.calibration_files = [path for _, _, path in raw]
        return {e["remote_generation"]: m for e, m, _ in raw
                if e["remote_generation"]}

    # ------------------------------------------------------------ summary
    def _l1_inputs(self, captures):
        return [c for c in captures if c.layer == "l1"]

    def _verify_saved(self, saved, l1):
        """A saved summary serves only with its inputs' fingerprints.

        Without fingerprints for every source the viewer rebuilds: the same
        process ids can reappear after captures grow or a calibration file
        changed, and the counts alone cannot prove the inputs identical.
        """
        if not isinstance(saved, dict) or saved.get("tool") != (
            "vllm.kv_profiling.summary"
        ):
            return None, "no saved summary document"
        if saved.get("run_id") != self.run_id:
            return None, "run id differs"
        sources = saved.get("sources") or []
        saved_procs = sorted(s.get("process_id") for s in sources)
        live_procs = sorted(c.process_id for c in l1)
        if saved_procs != live_procs:
            return None, "recorded sources differ from the captures present"
        # A summary saved before the timeline record sequences existed cannot
        # place a row in its scheduler occurrence: rebuild from the captures.
        if "timelines" not in saved or any(
            "sequences" not in t for t in saved.get("timelines") or []
        ) or any(
            "producer_handoff_sequence" not in j for j in saved.get("joins") or []
        ):
            return None, ("the saved summary predates the timeline record "
                          "sequences this viewer needs")  # fmt: skip
        fingerprints = saved.get("viewer_input_fingerprints")
        if not isinstance(fingerprints, dict):
            return None, "no input fingerprints: rebuilding is preferred"
        live = self.input_fingerprints()
        if fingerprints != live:
            return None, ("a capture, calibration or sidecar changed since "
                          "the summary was saved")
        return saved, "verified (input fingerprints)"

    def _input_sidecar_fingerprints(self):
        """Calibration manifests, the resolved mappings they name, and each
        L1 capture's profiler.json: summary-relevant inputs beyond the rows."""
        out = {}
        for capture in self.captures:
            if capture.layer != "l1" or capture.problems:
                continue
            for path in capture.path.parent.glob("*.json"):
                out[str(path.parent.parent.name) + "/" +
                    path.parent.name + "/" + path.name] = \
                    _fingerprint(path)
        out.update(self.calibration_hashes)
        return out

    def input_fingerprints(self):
        return {
            "l1": {c.process_id: c.fingerprint for c in self.captures
                   if c.layer == "l1" and not c.problems},
            "sidecars": self._input_sidecar_fingerprints(),
        }

    def build(self):
        captures, skipped = discover(self.dir)
        self.problems.extend({"kind": "unreadable_input", **item}
                             for item in skipped)
        captures = [c for c in captures if c.layer != "?"]
        if not captures:
            raise ViewerError(f"no profiler captures under {self.dir}")
        # A capture with conflicting copies cannot serve: exclude it from
        # every metric rather than silently keeping the first copy.
        conflicted = [c for c in captures if c.problems]
        for capture in conflicted:
            self._problem("capture_excluded", {
                "process_id": capture.process_id,
                "reasons": capture.problems,
            })
        captures = [c for c in captures if not c.problems]
        l1 = [c for c in captures if c.layer == "l1"]
        if not l1:
            raise ViewerError("no L1 recorder capture in the run")
        self.run_id = l1[0].run_id
        for capture in list(captures):
            if capture.layer == "l1":
                continue
            if capture.run_id != self.run_id:
                self._problem("foreign_capture", {
                    "layer": capture.layer, "run_id": capture.run_id,
                    "process_id": capture.process_id,
                })
                captures.remove(capture)
        self.captures = captures
        self._excluded = conflicted
        domains = {_clock_identity(c.meta) for c in captures}
        self.cross_node = len(domains) > 1
        # Reference clock: with calibrations, their common local generation;
        # else the consumer scheduler's clock, else any scheduler's.
        self.group = None
        self.aliases = {}
        self.reference_clock = None
        mappings = self._load_calibrations(l1)
        if self.reference_clock is None:
            sched = [c for c in l1 if c.context.get("role") == "scheduler"]
            consumers = [c for c in sched
                         if c.context.get("kv_role") == "kv_consumer"]
            pick = (consumers or sched or l1)[0]
            self.reference_clock = pick.meta["clock_id"]
            mappings = {}
        self.group = CLOCK.ClockGroup(self.reference_clock, mappings or {})
        # Every layer's clock takes part in domain resolution: collector and
        # observer clocks of one domain ride that domain's L1 clock.
        self.aliases = PERFETTO._clock_aliases(
            [c.meta for c in captures], self.group
        )
        # The summary gets L1 files and the named mapping files only.
        mapping_paths = self.calibration_files
        self.summary_source = None
        saved_path = self.dir / "summary.json"
        saved = None
        saved_reason = None
        try:
            saved = json.loads(saved_path.read_text())
        except (OSError, ValueError):
            saved = None
        if saved is not None:
            saved, why = self._verify_saved(saved, l1)
            if saved is None:
                saved_reason = why
            else:
                self.summary_source = {"source": "saved", "detail": why}
        if saved is None:
            try:
                saved = SUMMARY.summarize(
                    [c.path for c in l1],
                    calibration=mapping_paths or None,
                    reference_clock=self.reference_clock,
                )
                self.summary_source = (
                    {"source": "rebuilt", "detail": saved_reason}
                    if saved_reason else {"source": "rebuilt"}
                )
            except (ValueError, KeyError, TypeError) as error:
                raise ViewerError(
                    f"the run cannot be summarized: {error}"
                ) from error
        else:
            # still need not_available consistency
            pass
        if saved is not None and self.summary_source is None:
            self.summary_source = {"source": "rebuilt"}
        self.summary = saved
        self._index_timelines()
        self._collect_layer_rows()
        self._assemble()

    # ------------------------------------------------- request timelines
    def _index_timelines(self):
        """The summary's per-request timelines (phase 4), by scheduler source and
        request ID. A ``both`` scheduler is a consumer and a producer."""
        self.timelines = self.summary.get("timelines") or []
        self.consumer_timelines, self.producer_timelines = {}, {}
        for entry in self.timelines:
            key = (entry.get("source"), entry.get("request"))
            role = entry.get("node_role")
            if role in ("consumer", "both"):
                self.consumer_timelines.setdefault(key, []).append(entry)
            if role in ("producer", "both"):
                self.producer_timelines.setdefault(key, []).append(entry)
        self.process_clock = {
            capture.meta.get("process_id"): capture.meta.get("clock_id")
            for capture in self.captures
            if capture.layer == "l1"
        }

    @staticmethod
    def _owns(entry, sequence):
        """Whether a timeline's scheduler occurrence owns a record sequence."""
        span = entry.get("sequences") or {}
        if sequence is None or span.get("first") is None:
            return False
        return span["first"] <= sequence and (
            span.get("end") is None or sequence < span["end"]
        )

    @staticmethod
    def _ns_ms(value):
        return None if value is None else round(value / 1e6, 3)

    def _source_clock(self, index):
        sources = self.summary.get("sources") or []
        if isinstance(index, int) and 0 <= index < len(sources):
            return sources[index].get("clock_id")
        return None

    def _joins_for(self, process_id, request, occurrence):
        return [
            j for j in self.summary.get("joins") or []
            if (j.get("consumer_process_id"), j.get("consumer_request"),
                j.get("consumer_occurrence")) == (process_id, request, occurrence)
        ]  # fmt: skip

    def _incoming_handoffs(self, process_id, request, occurrence):
        """The record sequences of the connector occurrence's own incoming
        handoffs, as the summary's request entry lists them (no join needed:
        the producer's capture may be absent)."""
        return [
            handoff.get("sequence")
            for entry in self.summary.get("requests") or []
            if (entry.get("process_id"), entry.get("request"),
                entry.get("request_occurrence")) == (process_id, request, occurrence)
            for handoff in entry.get("handoffs") or []
            if handoff.get("direction") == "incoming"
            and handoff.get("sequence") is not None
        ]  # fmt: skip

    def _consumer_timeline(self, process_id, request, occurrence):
        """(the scheduler occurrence's timeline, None) or (None, reason).

        A row is a connector occurrence; its scheduler occurrence is the one
        whose record sequences contain the row's own incoming handoff (the two
        numberings can differ: a local run of the same ID before the pull is
        a scheduler occurrence of its own). Without a handoff, only an ID that
        the scheduler ran once is unambiguous."""
        listed = self.summary.get("sources") or []
        sources = {i for i, source in enumerate(listed)
                   if source.get("process_id") == process_id}  # fmt: skip
        named = [timeline for index in sources
                 for timeline in self.consumer_timelines.get((index, request), [])]
        if not named:
            return None, "no timeline for this scheduler request"
        incoming = self._incoming_handoffs(process_id, request, occurrence)
        if len(incoming) > 1:
            return None, "several incoming handoffs for this occurrence"
        if incoming:
            owners = [t for t in named if self._owns(t, incoming[0])]
            if len(owners) == 1:
                return owners[0], None
            reason = (
                "no scheduler occurrence contains the incoming handoff" if not owners
                else "several scheduler occurrences contain the handoff"
            )  # fmt: skip
            return None, reason
        if len(named) == 1:
            return named[0], None
        return None, ("several scheduler occurrences and no incoming handoff "
                      "to place this row")  # fmt: skip

    def _producer_timeline(self, joins):
        """(the producer timeline, None) through the occurrence's one confirmed
        handoff join, or (None, reason): the producer scheduler it names and the
        producer occurrence that owns the outgoing handoff's record sequence."""
        if not joins:
            return None, "no handoff join for this occurrence"
        if len(joins) > 1:
            return None, "several handoff joins for this occurrence"
        join = joins[0]
        if not join.get("joined") or join.get("ambiguous"):
            return None, "the handoff join is not confirmed"
        if join.get("identity_reused"):
            return None, ("the producer's identity is reused: several consumers "
                          "claim one outgoing handoff")  # fmt: skip
        producers = join.get("producer_sources") or []
        if len(producers) != 1:
            return None, "the handoff names no single producer scheduler"
        sequence = join.get("producer_handoff_sequence")
        if sequence is None:
            return None, "the outgoing handoff's record sequence is unknown"
        key = (producers[0], join.get("producer_request"))
        named = self.producer_timelines.get(key, [])
        owners = [t for t in named if self._owns(t, sequence)]
        if len(owners) != 1:
            reason = (
                "no producer occurrence owns the outgoing handoff" if not owners
                else "several producer occurrences own the outgoing handoff"
            )  # fmt: skip
            return None, reason
        return owners[0], None

    def _timeline_for(self, process_id, request, occurrence):
        """A request row's compute timeline, read from the summary's own
        timelines and never estimated: each value with its basis, or with its
        own reason when it is unavailable (``reasons``)."""
        if not self.timelines:
            reason = "the run's captures record no request timeline"
            return {"available": False, "reason": reason}
        joins = self._joins_for(process_id, request, occurrence)
        entry, why = self._consumer_timeline(process_id, request, occurrence)
        if entry is None:
            return {"available": False, "reason": why}
        api = entry.get("api") or {}
        stall = entry.get("stall") or {}
        intervals = entry.get("intervals_ns") or {}
        window = entry.get("decode_window") or {}
        scheduler = entry.get("scheduler") or {}
        reasons = {}
        result = {
            "available": True,
            "node_role": entry.get("node_role"),
            "state": entry.get("state"),
            "quality": entry.get("quality"),
            "kv_wait_ms": self._ns_ms(stall.get("waiting_for_kv_ns")),
            "admission_ms": self._ns_ms(intervals.get("queued_to_admitted")),
            "decode_window_ms": None,
            "time_per_token_ms": self._ns_ms(
                window.get("mean_time_per_token_after_first_ns")
            ),
            "ttft_ms": None,
            "ttft_basis": None,
            "prefill_ms": None,
            "prefill_basis": None,
            "stall": stall,
            "reasons": reasons,
        }
        if result["kv_wait_ms"] is None:
            reasons["kv_wait_ms"] = "the timeline's stall has no KV wait"
        if result["admission_ms"] is None:
            reasons["admission_ms"] = "no admission after the queue entry"
        if (window.get("first_token_output_ns") is not None
                and window.get("output_finished_ns") is not None):  # fmt: skip
            result["decode_window_ms"] = self._ns_ms(
                window["output_finished_ns"] - window["first_token_output_ns"]
            )
        else:
            reasons["decode_window_ms"] = "no first token output or finish recorded"
        if stall.get("basis") != "api_arrival":
            reasons["ttft_ms"] = (f"the stall basis is {stall.get('basis')!r}, "
                                  "not the API arrival")  # fmt: skip
        elif api.get("arrival_ns") is None or api.get("first_token_output_ns") is None:
            reasons["ttft_ms"] = "no API arrival or first token output recorded"
        else:
            result["ttft_ms"] = self._ns_ms(
                api["first_token_output_ns"] - api["arrival_ns"]
            )
            result["ttft_basis"] = "this node's API entry to its first token output"
        producer, why = self._producer_timeline(joins)
        if producer is not None:
            value = (producer.get("intervals_ns") or {}).get("prefill_window")
            result["prefill_ms"] = self._ns_ms(value)
            result["prefill_basis"] = "the producer's scheduler, its own clock"
            if value is None:
                reasons["prefill_ms"] = "the producer timeline has no prefill window"
        else:
            result["prefill_basis"] = why
            reasons["prefill_ms"] = why
        # Lane spans on the reference clock. Scheduler times are projected with
        # the scheduler's clock, frontend times with the frontend's (api.source);
        # a point that does not project is not drawn.
        scheduler_clock = (self._source_clock(entry.get("source"))
                           or self.process_clock.get(process_id))  # fmt: skip
        frontend_clock = self._source_clock(api.get("source"))

        def at(clock, value):
            if value is None or clock is None:
                return None
            return self.project(clock, value)[0]

        points = {
            "load_prepared": at(scheduler_clock, scheduler.get("load_prepared_ns")),
            "kv_received": at(scheduler_clock, scheduler.get("kv_received_ns")),
            "first_token": at(frontend_clock, api.get("first_token_output_ns")),
            "finished": at(frontend_clock, api.get("output_finished_ns")),
        }
        spans = []
        for kind, a, b in (("waiting for KV", "load_prepared", "kv_received"),
                           ("to first token", "kv_received", "first_token"),
                           ("decode", "first_token", "finished")):  # fmt: skip
            if points[a] is not None and points[b] is not None:
                spans.append({"t0": points[a], "t1": points[b], "kind": kind})
        result["spans"] = spans
        return result

    # ------------------------------------------------- layer rows (L2 / L3)
    def _collect_layer_rows(self):
        """L2/L3 rows with their real file and line, deduplicated by schema.

        L1 rows are read separately (real line numbers via ``raw_l1``), the
        summary owns their pairing.
        """
        self.rows = {"l2": [], "l3": []}
        self.l1_rows = {}  # process_id -> (rows, {sequence: line}, meta, decoder)
        self.ucx_docs = []
        for capture in self.captures:
            if capture.layer == "l1":
                self._read_l1(capture)

                continue
            path = capture.path
            rows = []
            with path.open() as stream:
                for line_no, line in enumerate(stream, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError as error:
                        capture.problems.append(f"line {line_no}: {error}")
                        continue
                    rows.append((line_no, row))
            capture.rows = rows
            self.rows[capture.layer].append(capture)
        for doc_path in sorted(self.dir.rglob("ucx-protocols/**/*.json")):
            try:
                self.ucx_docs.append({
                    "path": str(doc_path),
                    "document": json.loads(doc_path.read_text()),
                })
            except (OSError, ValueError) as error:
                self._problem("ucx_document", f"{doc_path}: {error}")

    def _read_l1(self, capture):
        meta, events, decoder, raw_rows, missing = PERFETTO._load(capture.path)
        lines = {}
        with capture.path.open() as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("type") == "event" and "sequence" in row:
                    lines[row["sequence"]] = line_no
        capture.rows = [(lines.get(r.get("sequence")), r) for r in events]
        self.l1_rows[capture.process_id] = {
            "rows": raw_rows,
            "lines": lines,
            "meta": meta,
            "decoder": decoder,
            "events": events,
            "capture": capture,
        }

    # ------------------------------------------------------------- assemble
    def _assemble(self):
        sources = self.summary.get("sources", [])
        self.sources = {s["source"]: s for s in sources}
        self.registry = engine_registry(self.captures)
        # Two engines can share a role letter and rank; disambiguate labels
        # with the engine id so pairs and lanes stay distinct.
        by_label = {}
        for engine, entry in self.registry.items():
            by_label.setdefault(entry["label"], []).append(engine)
        for label, engines in by_label.items():
            if len(engines) > 1:
                for engine in engines:
                    self.registry[engine]["label"] = (
                        label + "·" + engine[:6]
                    )
        self._build_transports()
        self._build_requests()
        self._build_lanes()
        self._build_probe_candidates()
        self._build_runkv()

    def _source_of(self, index):
        return self.sources.get(index, {})

    def _is_mooncake(self, source_index):
        meta = self._l1_meta(source_index)
        return (meta.get("context") or {}).get("connector") == "mooncake"

    def _l1_meta(self, source_index):
        source = self._source_of(source_index)
        for capture in self.captures:
            if capture.process_id == source.get("process_id"):
                return capture.meta
        return {}

    def _capture_by_process(self, process_id):
        for capture in self.captures:
            if capture.process_id == process_id:
                return capture
        return None

    # transports: one entry per physical transfer call, counted once
    def _build_transports(self):
        # Mooncake batch member shares, keyed (source index, handle): the
        # build's own per-request shares of the batch's measured totals,
        # with the summary's confirmed/planned batch fate.
        shares = {}
        for batch in self.summary.get("write_batches", []):
            shares[(batch.get("source"), batch.get("batch"))] = {
                "members": {
                    member.get("request"): (member.get("bytes"),
                                            member.get("descriptors"),
                                            member.get("ok"))
                    for member in batch.get("members", [])
                },
                # the batch's own terminal fate: written / no_data / failed
                # / incomplete — planned shares are never confirmed bytes
                "write": batch.get("write"),
                "status": batch.get("state"),
            }
        out = []
        for entry in self.summary.get("transfers", []):
            source_index = entry.get("source")
            source = self._source_of(source_index)
            mooncake = self._is_mooncake(source_index)
            # Interval: the connector's own recorded call scope. An async
            # NIXL READ sweeps acceptance to terminal; a synchronous
            # Mooncake WRITE sweeps call entry to return, its terminal
            # status qualifying the bytes.
            if mooncake:
                begin = (entry.get("post_begin_time_ns")
                         or entry.get("posted_time_ns"))
                end = entry.get("posted_time_ns") or entry.get(
                    "completed_time_ns") or entry.get("failed_time_ns")
                success = entry.get("state") == "completed" and entry.get(
                    "complete_status") in (0, 2)
            else:
                begin = entry.get("posted_time_ns")
                end = (entry.get("completed_time_ns")
                       or entry.get("failed_time_ns"))
                success = entry.get("state") == "completed"
            duration_ns = (
                end - begin
                if begin is not None and end is not None and end >= begin
                else None
            )
            bytes_value = entry.get("bytes") if success else None
            # NIXL's own integer-microsecond transfer time, when the
            # connector recorded it: the honest throughput basis (the
            # post->terminal span is the poll that saw DONE, an observation
            # interval).
            nixl_xfer_us = entry.get("nixl_xfer_us")
            throughput = None
            basis = None
            if bytes_value is not None:
                if nixl_xfer_us:
                    throughput = bytes_value / (nixl_xfer_us * 1e-6)
                    basis = "nixl_xfer_us"
                elif duration_ns:
                    throughput = bytes_value * 1_000_000_000 / duration_ns
                    basis = "measured interval"
            p0 = self.project(source.get("clock_id"), begin)[0] \
                if begin is not None else None
            p1 = self.project(source.get("clock_id"), end)[0] \
                if end is not None else None
            if p0 is None or p1 is None:
                duration_ns = None
            producer = source.get("engine_id")
            if mooncake:
                # The producer worker posts the batch; the consumer side is
                # its members' requests, resolved when the batch attaches to
                # their consumer-scheduler spines.
                batch_info = shares.get(
                    (source_index, entry.get("transfer")),
                    {"members": {}, "write": None, "status": None},
                )
                member_shares = batch_info["members"]
                # the batch's member rows on the worker carry each member's
                # shared handoff identity: (request id, handoff id)
                member_rows = []
                data = self.l1_rows.get(source.get("process_id"))
                if data:
                    handle = entry.get("transfer")
                    for event in data["events"]:
                        if not SUMMARY._is_member(event):
                            continue
                        if str(event.get("transfer")) != str(handle):
                            continue
                        member_rows.append(event)
                member_pairs = []
                for row in member_rows:
                    request_name = data["decoder"].resolve(
                        row.get("request_ref") or 0
                    )
                    shared = data["decoder"].resolve(
                        row.get("handoff_ref") or 0
                    )
                    if isinstance(request_name, str) \
                            and isinstance(shared, str):
                        member_pairs.append((request_name, shared))
                pair = None
                consumer = None
            else:
                consumer = source.get("engine_id")
                peer = entry.get("peer")
                pair = f"{peer}→{consumer}" if peer and consumer else None
                member_shares = {}
                member_pairs = []
            out.append({
                "key": f"{source_index}:{entry.get('transfer')}",
                "source_index": source_index,
                "process_id": source.get("process_id"),
                "role": source.get("role"),
                "mooncake": mooncake,
                "state": entry.get("state"),
                "request": entry.get("request"),
                "request_occurrence": entry.get("request_occurrence"),
                "producer": producer,
                "consumer": consumer,
                "member_pairs": member_pairs,
                "member_shares": member_shares,
                "pair": pair,
                "t0": p0,
                "t1": p1,
                "prep_to_post_ns": entry.get("prepared_to_posted_ns"),
                "duration_ns": duration_ns,
                "bytes": bytes_value,
                "descriptors": (
                    entry.get("merged_descriptors")
                    if entry.get("merged_descriptors") is not None
                    else entry.get("selected_descriptors")
                ),
                "throughput": throughput,
                "throughput_basis": basis,
                "nixl_xfer_us": nixl_xfer_us,
                "thread": entry.get("thread"),
                "thread_name": entry.get("thread_name"),
                "prepared_ns": entry.get("prepared_time_ns"),
                # confirmed vs planned: a batch that did not write carries
                # planned shares only
                "confirmed": (
                    batch_info.get("write") in ("written", "no_data")
                    if mooncake else (entry.get("state") == "completed")
                ),
                "planned_basis": (
                    None if mooncake and batch_info.get("write")
                    in ("written", "no_data")
                    else "planned (batch write %s)"
                    % (batch_info.get("write") or "did not complete")
                ) if mooncake else None,
                "thread_cpu_scope": "shared thread activity during the "
                "sampled interval",
            })
        self.transports = out



    # logical requests
    def _frontend_parent_edges(self):
        """Child request id -> parent request id, per frontend capture.

        The frontend records each id's parent as the request_bind's
        ``handoff_ref`` (client id, engine-composed ids, item ids); the edges
        are exact recorded identities, never string parsing.
        """
        edges = {}
        self.frontend_edges = edges
        for capture in self.captures:
            if capture.layer != "l1" or capture.context.get("role") != (
                "frontend"
            ):
                continue
            data = self.l1_rows.get(capture.process_id)
            if not data:
                continue
            per_source = edges.setdefault(capture.process_id, {})
            for event in data["events"]:
                if event.get("kind") != "request_bind" \
                        or SUMMARY._is_member(event):
                    continue
                child = data["decoder"].resolve(event.get("request_ref") or 0)
                parent = data["decoder"].resolve(event.get("handoff_ref") or 0)
                if isinstance(child, str) and isinstance(parent, str) \
                        and child != parent:
                    per_source[child] = parent

    @staticmethod
    def _ancestors(edges, request):
        """The request and its recorded parent chain (cycle-safe)."""
        chain = [request]
        seen = {request}
        current = request
        while True:
            parent = edges.get(current)
            if not parent or parent in seen:
                break
            chain.append(parent)
            seen.add(parent)
            current = parent
        return chain

    def _build_requests(self):
        """Logical requests: distinct consumer-scheduler requests, keyed by
        (scheduler process generation, id, occurrence; engine scoped).

        Attachments ride recorded identities only — the producer scheduler's
        handoff (engine + request id), the worker's transfers (engine,
        occurrence), and the frontend's own recorded parent edges. Anything
        that attaches to no scheduler request is listed separately in
        ``observations``, never folded by guesswork."""
        self._frontend_parent_edges()
        spine = {}
        spine_ids = {}
        for entry in self.summary.get("requests", []):
            source = self._source_of(entry.get("source"))
            if source.get("role") == "scheduler" and source.get(
                "kv_role"
            ) in ("kv_consumer", "kv_both"):
                tag = (source.get("process_id"), entry.get("request"),
                       entry.get("request_occurrence") or 0)
                spine[tag] = {
                    "spine": entry, "source": source, "producer": [],
                    "transports": [], "frontends": [], "others": [],
                    "transport_keys": set(),
                }
                spine_ids.setdefault(source.get("engine_id"), set()).add(
                    (entry.get("request"), entry.get("request_occurrence")
                     or 0)
                )
        producer_ids = {}
        # producer schedulers attach by the spine's incoming handoff identity
        for entry in self.summary.get("requests", []):
            source = self._source_of(entry.get("source"))
            if source.get("role") != "scheduler" or source.get(
                "kv_role"
            ) not in ("kv_producer", "kv_both"):
                continue
            for row in spine.values():
                incoming = [
                    h for h in row["spine"].get("handoffs", [])
                    if h.get("direction") == "incoming"
                    and h.get("producer_request") == entry.get("request")
                    and self._engine_matches(
                        h.get("producer_engine"), source.get("engine_id")
                    )
                ]
                if incoming:
                    row["producer"].append({"entry": entry,
                                            "source": source})
                    break
            else:
                producer_ids[(source.get("process_id"),
                              entry.get("request"),
                              entry.get("request_occurrence") or 0)] = (
                    entry, source
                )
        # worker request observations attach through their transport's
        # ownership (the same rule the click uses): a worker entry joins
        # the spine rows whose transport keys carry its request and
        # occurrence, decided after the transports attach — an entry with
        # no owned transport stays an observation
        self._pending_worker_entries = []
        for entry in self.summary.get("requests", []):
            source = self._source_of(entry.get("source"))
            if source.get("role") != "worker":
                continue
            self._pending_worker_entries.append((entry, source))
        # frontends attach through their own recorded parent edges (the
        # frontend process records no engine id, so the chain is the join):
        # an entry folds into a scheduler request when either side's
        # recorded ancestor chain contains the other's id
        all_edges = {}
        for edges in self.frontend_edges.values():
            all_edges.update(edges)
        spine_chains = {
            tag: set(self._ancestors(all_edges,
                                     row["spine"].get("request")))
            for tag, row in spine.items()
        }
        for entry in self.summary.get("requests", []):
            source = self._source_of(entry.get("source"))
            if source.get("role") != "frontend":
                continue
            chain = set(self._ancestors(all_edges, entry.get("request")))
            for tag, row in spine.items():
                if (row["spine"].get("request") in chain
                        or entry.get("request") in spine_chains[tag]):
                    row["frontends"].append({"entry": entry,
                                             "source": source})
                    break
            else:
                producer_ids[(source.get("process_id"),
                              entry.get("request"),
                              entry.get("request_occurrence") or 0)] = (
                    entry, source
                )
        # transports attach to their own request's spine row by exact
        # recorded identities (see _attach_transports)
        self._attach_transports(spine, spine_ids)
        # anything that attached to nothing stays an explicit observation
        observations = []
        # worker observations ride their transport's ownership: the entry
        # joins the spine rows that own a transport with the entry's
        # request and occurrence (exact identity; no owner ⇒ it stays an
        # observation)
        for entry, source in self._pending_worker_entries:
            request = entry.get("request")
            occurrence = entry.get("request_occurrence") or 0
            owners = [
                row for row in spine.values()
                if any(
                    transport.get("request") == request
                    and (transport.get("request_occurrence") or 0)
                    == occurrence
                    for transport in row.get("transports", [])
                )
            ]
            if owners:
                for row in owners:
                    row["others"].append(
                        {"entry": entry, "source": source,
                         "role": "worker"}
                    )
            else:
                observations.append({
                    "role": "worker",
                    "process_id": source.get("process_id"),
                    "request": request,
                    "occurrence": occurrence,
                    "state": entry.get("state"),
                })
        for (proc, request, occurrence), (entry, source) in producer_ids.\
                items():
            observations.append({
                "role": source.get("role"),
                "process_id": proc,
                "request": request,
                "occurrence": occurrence,
                "state": entry.get("state"),
            })
        self.spine = spine
        self.observations = observations

    @staticmethod
    def _spine_transfer_ids(spine_row, l1_rows, occurrence=None):
        """The shared transfer identity ids the spine's own records carry
        for one request occurrence: its scheduler handoff rows'
        `handoff_ref` (exact recorded ids), split by the request's own
        bind rows the way the summary splits occurrences."""
        capture = l1_rows.get(spine_row["source"].get("process_id"))
        if not capture:
            return set()
        request = spine_row["spine"].get("request")
        ordered = sorted(capture["events"], key=lambda e: e["sequence"])
        bind_counts = {}  # resolved request id -> binds seen (occurrence+1)
        ids = {}
        for event in ordered:
            if SUMMARY._is_member(event):
                continue
            resolved = capture["decoder"].resolve(
                event.get("request_ref") or 0
            )
            if event.get("kind") == "request_bind" and resolved == request:
                bind_counts[request] = bind_counts.get(request, 0) + 1
                continue
            if event.get("kind") != "handoff":
                continue
            shared = capture["decoder"].resolve(
                event.get("handoff_ref") or 0
            )
            if resolved == request and isinstance(shared, str):
                # the handoff belongs to the occurrence its request's bind
                # opened: the n-th bind (1-based) is occurrence n-1
                handoff_occurrence = bind_counts.get(request, 1) - 1
                ids.setdefault(handoff_occurrence, set()).add(shared)
        if occurrence is None:
            return set().union(*ids.values()) if ids else set()
        return ids.get(occurrence, set())

    def _attach_mooncake_batch(self, transport, spine):
        """One Mooncake write batch, resolved once per recorded member.

        Each member row carries (member request id, shared handoff id); a
        spine candidate is the request's occurrence whose own recorded
        handoff rows contain that shared id and whose incoming handoff came
        from this batch's producer engine. Ownership resolves per member:
        exactly one candidate occurrence takes the share (confirmed only
        when the batch's write completed); several candidates are ambiguous
        and that member stays unassigned while independently matched
        members keep their shares.
        """
        member_pairs = transport.get("member_pairs") or []
        engines = set()
        unassigned = []
        for member, shared in member_pairs:
            candidates = []
            for tag, row in spine.items():
                if row["spine"].get("request") != member:
                    continue
                occurrence = row["spine"].get("request_occurrence") or 0
                ids = self._spine_transfer_ids(
                    row, self.l1_rows, occurrence
                )
                if shared not in ids:
                    continue
                # the summary pairs both fields on each incoming row: the
                # producer engine AND the transfer id must sit on the SAME
                # incoming handoff row (a (P, X) row never matches a member
                # that names (P, A), and a genuine (P, A) survives a second
                # (P, X) row beside it). Duplicate identical rows dedupe.
                incoming = {
                    (h.get("producer_engine"), h.get("producer_request"))
                    for h in row["spine"].get("handoffs", [])
                    if h.get("direction") == "incoming"
                    and self._engine_matches(
                        transport.get("producer"),
                        h.get("producer_engine"),
                    )
                }
                matching = [
                    pair for pair in incoming
                    if pair[1] == shared
                ]
                if len(matching) != 1:
                    continue
                candidates.append((tag, row))
            if len(candidates) == 1:
                tag, row = candidates[0]
                row["transports"].append(transport)
                row["transport_keys"].add(transport["key"])
                engines.add(row["source"].get("engine_id"))
            else:
                unassigned.append({
                    "member": member,
                    "reason": (
                        f"{len(candidates)} occurrences record the shared "
                        "handoff identity; ambiguous"
                        if candidates else
                        "no occurrence records the shared handoff identity"
                    ),
                })
        if unassigned:
            self._problem("transport_unassigned", {
                "key": transport["key"],
                "reason": unassigned,
            })
        else:
            engines.add(transport.get("consumer") or "")
        engines.discard("")
        if len(engines) == 1:
            consumer = next(iter(engines))
            transport["consumer"] = consumer
            transport["pair"] = f"{transport['producer']}→{consumer}"

    def _attach_transports(self, spine, _spine_ids):
        """Exact recorded identities only; ambiguous transfers (several
        spine rows match, or a Mooncake member whose shared handoff
        identity no unique incoming producer handoff records) stay
        unassigned — visible on the lanes, attributed to no request."""
        for transport in self.transports:
            if transport["mooncake"]:
                self._attach_mooncake_batch(transport, spine)
                continue
            request = transport.get("request")
            if not request:
                continue
            occurrence = transport.get("request_occurrence") or 0
            engine = self._engine_of_transport(transport)
            matches = [
                row for row in spine.values()
                if row["spine"].get("request") == request
                and (row["spine"].get("request_occurrence") or 0)
                == occurrence
                and self._engine_matches(engine,
                                         row["source"].get("engine_id"))
            ]
            if not matches:
                continue
            if len(matches) > 1:
                self._problem("transport_unassigned", {
                    "key": transport["key"],
                    "reason": f"{len(matches)} scheduler generations "
                    "record the same request and occurrence",
                })
                continue
            matches[0]["transports"].append(transport)
            matches[0]["transport_keys"].add(transport["key"])
            transport["attached"] = True

    def _engine_of_transport(self, transport):
        source = self._source_of(transport.get("source_index"))
        return source.get("engine_id")

    @staticmethod
    def _source_engine(source):
        return source.get("engine_id")

    def _engine_matches(self, producer_engine, engine_id):
        if not producer_engine or not engine_id:
            return False
        if producer_engine == engine_id:
            return True
        entry = self.registry.get(producer_engine) or {}
        return engine_id in (entry.get("process_ids") or [])


    # ---------------------------------------------------------------- lanes
    def _build_lanes(self):
        placed = [
            t for t in self.transports
            if t["t0"] is not None and t["t1"] is not None
        ]
        times = [t["t0"] for t in placed] + [t["t1"] for t in placed]
        # every request's own records widen the window too: a frontend bind
        # or a producer handoff legitimately precedes the transport's post
        for entry in self.summary.get("requests", []):
            source = self._source_of(entry.get("source"))
            clock = source.get("clock_id")
            values = [v for k, v in entry.items()
                      if k.endswith("_time_ns") and isinstance(v, int)]
            for handoff in entry.get("handoffs", []):
                if isinstance(handoff.get("time_ns"), int):
                    values.append(handoff["time_ns"])
            for value in values:
                projected = self.project(clock, value)[0]
                if projected is not None:
                    times.append(projected)
        # frontend arrival binds are not summary entry times: scan the
        # frontend captures' own request_bind rows
        for capture in self.captures:
            if capture.layer != "l1" or capture.context.get("role") != (
                "frontend"
            ):
                continue
            data = self.l1_rows.get(capture.process_id)
            if not data:
                continue
            for event in data["events"]:
                if event.get("kind") != "request_bind" \
                        or SUMMARY._is_member(event):
                    continue
                projected = self.project(capture.meta.get("clock_id"),
                                         event.get("time_ns"))[0]
                if projected is not None:
                    times.append(projected)
        for records in self.rows["l3"]:
            for _, row in records.rows:
                begin = row.get("sample_begin_ns")
                end = row.get("sample_end_ns") or row.get("time_ns")
                if begin is None or end is None:
                    continue
                one = self.project(records.meta.get("clock_id"), begin)[0]
                two = self.project(records.meta.get("clock_id"), end)[0]
                if one is not None and two is not None:
                    times += [one, two]
        self.window = [min(times), max(times)] if times else [0, 0]
        self._lane_concurrency()
        self._lane_threads()
        self._lane_rtt()
        self._lane_interrupts()
        self._lane_ebpf()
        self._lane_collector()

    @staticmethod
    def _sweep(intervals):
        """Peak simultaneously open intervals; events swept in order."""
        events = []
        for begin, end in intervals:
            events.append((begin, 1))
            events.append((end, -1))
        peak = current = 0
        for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
            current += delta
            peak = max(peak, current)
        return peak

    def _peak_at(self, intervals, begin, end):
        count = 0
        for one, two in intervals:
            if one < end and two > begin:
                count += 1
        return count

    def _lane_concurrency(self):
        by_pair = {}
        for transport in self.transports:
            if transport["t0"] is None or transport["t1"] is None:
                continue
            by_pair.setdefault(transport["pair"] or "?", []).append(
                (transport["t0"], transport["t1"], transport["bytes"])
            )
        inflight = {}
        gbs = {}
        peaks = {}
        for pair, items in sorted(by_pair.items(), key=lambda kv: kv[0] or ""):
            intervals = [(a, b) for a, b, _ in items]
            peaks[pair] = self._sweep(intervals)
            # in-flight lane: the maximum number of intervals simultaneously
            # open anywhere inside each bin — swept within the bin, so two
            # sequential transfers in one bin never read two
            span = self.window
            nbins = int((span[1] - span[0]) / (BIN_MS * 1_000_000)) + 1
            lane = [0] * nbins
            for b in range(nbins):
                lo = span[0] + b * BIN_MS * 1_000_000
                hi = lo + BIN_MS * 1_000_000
                events = []
                for begin, end in intervals:
                    clipped_begin = max(begin, lo)
                    clipped_end = min(end, hi)
                    if clipped_begin >= clipped_end:
                        # overlapping the bin's edges still counts: an
                        # interval open across the whole bin is open
                        # somewhere inside it
                        if begin < hi and end > lo:
                            events.append((max(begin, lo), 1))
                            events.append((min(end, hi), -1))
                        continue
                    events.append((clipped_begin, 1))
                    events.append((clipped_end, -1))
                events.sort(key=lambda item: (item[0], item[1]))
                current = max_open = 0
                for _, delta in events:
                    current += delta
                    max_open = max(max_open, current)
                lane[b] = max_open
            inflight[pair] = lane
            # goodput: completed payload bytes spread uniformly over the
            # measured span (an estimate; stated as such in the payload)
            series = [0.0] * nbins
            width_s = BIN_MS / 1000.0
            for begin, end, value in items:
                if value is None:
                    continue
                total = end - begin
                if total <= 0:
                    continue
                b0 = max(0, int((begin - span[0]) / (BIN_MS * 1_000_000)))
                b1 = min(nbins - 1, int((end - span[0]) / (BIN_MS * 1_000_000)))
                for b in range(b0, b1 + 1):
                    lo = max(begin, span[0] + b * BIN_MS * 1_000_000)
                    hi = min(end, span[0] + (b + 1) * BIN_MS * 1_000_000)
                    if hi <= lo:
                        continue
                    series[b] += value * (hi - lo) / total
            gbs[pair] = [
                [span[0] + b * BIN_MS * 1_000_000, round(v / width_s / 1e9, 4)]
                if v else [span[0] + b * BIN_MS * 1_000_000, None]
                for b, v in enumerate(series)
            ]
        self.lane_inflight = inflight
        self.lane_gbs = gbs
        self.lane_peaks = peaks

    def _lane_threads(self):
        """Per-thread CPU over each sample's measured interval; identity join.

        A series maps to an L1 process only with matching boot id, network
        and PID namespaces, pid (and the recorder's start ticks when the
        observation carries them); rates divide the delta between
        consecutive readings by the end-to-end elapsed time between them.
        """
        strips = []
        l1_by_identity = {}
        for capture in self.captures:
            if capture.layer != "l1":
                continue
            meta = capture.meta
            l1_by_identity[(meta.get("boot_id"), meta.get("net_namespace"),
                            meta.get("pid_namespace"), meta.get("pid"))] = meta
        for capture in self.rows["l3"]:
            series = {}
            for _line_no, row in capture.rows:
                if (row.get("kind") != "endpoint_metrics"
                        or row.get("scope") != "threads"):
                    continue
                observation = row.get("observation") or {}
                boot = capture.meta.get("boot_id")
                pkey = (boot, capture.meta.get("net_namespace"),
                        capture.meta.get("pid_namespace"),
                        observation.get("pid"))
                # the join requires the process generation: same pid with a
                # different (or unrecorded) start ticks is a different or
                # unverifiable process and never attributes
                observed_start = observation.get("start_ticks")
                joined = (
                    pkey in l1_by_identity
                    and observed_start is not None
                    and l1_by_identity[pkey].get("start_ticks")
                    == observed_start
                )
                start_ok = joined
                hz = observation.get("clock_ticks_per_s")
                for thread in observation.get("threads", []):
                    key = (pkey, thread.get("tid"),
                           thread.get("start_ticks"))
                    series.setdefault(key, []).append(
                        (thread, row.get("sample_begin_ns"),
                         row.get("sample_end_ns"), joined, start_ok, hz)
                    )
            for key, samples in sorted(
                series.items(), key=lambda kv: str(kv[0])
            ):
                samples.sort(key=lambda item: item[2] or 0)
                points = []
                pending = []
                busy = 0.0
                prior = None
                for sample in samples:
                    thread, begin, end, joined, start_ok, hz = sample
                    if prior is None:
                        prior = sample
                        continue
                    ticks = (thread["utime_ticks"] + thread["stime_ticks"]) \
                        - (prior[0]["utime_ticks"] + prior[0]["stime_ticks"])
                    # the sampling interval: start-of-reading to
                    # start-of-reading when both are recorded (never the
                    # time spent reading procfs); end-to-end otherwise
                    begin_delta = (
                        (begin or 0) - (prior[1] or 0)
                        if begin is not None and prior[1] else None
                    )
                    elapsed_ns = (
                        begin_delta if begin_delta and begin_delta > 0
                        else (end or 0) - (prior[2] or 0)
                    )
                    hz_value = hz or prior[5] or 100
                    cpu = (
                        ticks / hz_value / (elapsed_ns / 1e9)
                        if elapsed_ns > 0 and ticks >= 0 else None
                    )
                    run_ns = thread.get("run_ns")
                    wait_ns = thread.get("wait_ns")
                    prior_run = prior[0].get("run_ns")
                    run_share = (
                        (run_ns - prior_run) / elapsed_ns
                        if run_ns is not None and prior_run is not None
                        and elapsed_ns > 0 else None
                    )
                    wait_share = (
                        (wait_ns - prior[0].get("wait_ns")) / elapsed_ns
                        if wait_ns is not None
                        and prior[0].get("wait_ns") is not None
                        and elapsed_ns > 0 else None
                    )
                    if cpu is not None:
                        busy += max(0.0, cpu) * elapsed_ns / 1e9
                    # the strip draws over the interval the rate covers:
                    # from the previous reading's projected end to this
                    # reading's projected end
                    begin_projected = self.project(
                        capture.meta.get("clock_id"), begin or 0
                    )[0]
                    end_projected = self.project(
                        capture.meta.get("clock_id"), end or 0
                    )[0]
                    prior_end_projected = self.project(
                        capture.meta.get("clock_id"), prior[2] or 0
                    )[0]
                    pending.append((
                        prior_end_projected
                        if prior_end_projected is not None
                        else begin_projected,
                        end_projected, thread.get("tid"),
                        cpu, run_share, wait_share, joined, start_ok,
                    ))
                    prior = sample
                for _index, (t0, e0, _tid, cpu, run_share, wait_share,
                             joined, start_ok) in enumerate(pending):
                    points.append({
                        "t": e0, "t0": t0, "t1": e0,
                        "cpu": None if cpu is None else round(cpu, 4),
                        "run_share": None if run_share is None
                        else round(run_share, 4),
                        "wait_share": None if wait_share is None
                        else round(wait_share, 4),
                        "joined_process": joined, "start_ticks_unverified":
                        not start_ok,
                    })
                if not points:
                    continue
                tid = key[1]
                name = self._thread_name(key[0], tid)
                strips.append({
                    "identity": {
                        "boot_id": key[0][0], "net_namespace": key[0][1],
                        "pid_namespace": key[0][2], "pid": key[0][3],
                        "tid": tid, "thread_start_ticks": key[2],
                    },
                    "label": f"{name or '?'} ({tid})",
                    "name": name,
                    "joined_process": points[0]["joined_process"],
                    "start_ticks_unverified": any(
                        p["start_ticks_unverified"] for p in points
                    ),
                    "points": points,
                    "busy_s": round(busy, 6),
                })
        strips.sort(key=lambda s: -s["busy_s"])
        self.thread_strips = strips[:MAX_THREAD_STRIPS]
        self.thread_strips_truncated = len(strips) > MAX_THREAD_STRIPS

    def _thread_name(self, process_key, tid):
        for capture in self.captures:
            if capture.layer != "l1":
                continue
            meta = capture.meta
            if (meta.get("boot_id"), meta.get("net_namespace"),
                    meta.get("pid_namespace"), meta.get("pid")) != process_key:
                continue
            for row in self.l1_rows.get(capture.process_id, {}).get(
                "rows", []
            ):
                if row.get("type") == "threads":
                    return row.get("threads", {}).get(str(tid))
        return None


    @staticmethod
    def _socket_entity(entity):
        try:
            value = json.loads(entity) if isinstance(entity, str) else entity
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    def _lane_rtt(self):
        """One series per five-tuple and socket incarnation, from the writer's
        own ``transport.tcp`` observation; never a pair."""
        series = {}
        by_entity = {}
        counter = {}
        for capture in self.rows["l3"]:
            for _line_no, row in capture.rows:
                if (row.get("kind") != "endpoint_metrics"
                        or row.get("scope") != "socket"):
                    continue
                entity = row.get("entity")
                flow = (row.get("transport") or {}).get("tcp") or {}
                info = (row.get("observation") or {}) .get("tcp") or flow
                fields = (flow.get("fields")
                          or (row.get("observation") or {}).get("fields")
                          or {})
                rtt = fields.get("rtt_us") or {}
                value = rtt.get("value") if rtt.get(
                    "available", True
                ) else None
                cookie = tuple(flow.get("cookie")
                               or info.get("cookie") or ())
                identity = (
                    capture.meta.get("net_namespace"),
                    flow.get("local_ip") or info.get("local_ip"),
                    flow.get("local_port") or info.get("local_port"),
                    flow.get("peer_ip") or info.get("peer_ip"),
                    flow.get("peer_port") or info.get("peer_port"),
                )
                incarnation = counter.setdefault(identity, {})
                if cookie and cookie not in incarnation:
                    incarnation[cookie] = len(incarnation)
                number = incarnation.get(cookie, 0)
                if (row.get("observation") or {}).get("continuity") == \
                        "counter_reset":
                    number += 1
                    incarnation[cookie] = number
                key = identity + (number,)
                end = row.get("sample_end_ns") or row.get("time_ns")
                begin = row.get("sample_begin_ns")
                projected = self.project(capture.meta.get("clock_id"), end)[0]
                projected_begin = self.project(
                    capture.meta.get("clock_id"), begin or 0
                )[0]
                series.setdefault(key, []).append({
                    "t": projected, "t0": projected_begin, "t1": projected,
                    "rtt_us": value, "incarnation": number,
                })
                by_entity.setdefault(entity, []).append(key)
        self.rtt_by_entity = by_entity
        self.rtt_series = [
            {
                "identity": {
                    "net_namespace": k[0], "local_ip": k[1],
                    "local_port": k[2], "peer_ip": k[3], "peer_port": k[4],
                    "incarnation": k[5],
                },
                "label": (
                    f"{k[1]}:{k[2]} <-> {k[3]}:{k[4]} · incarnation {k[5]}"
                ),
                "points": sorted(points, key=lambda p: p["t"] or 0),
            }
            for k, points in sorted(series.items(), key=lambda kv: str(kv[0]))
        ]

    def _lane_interrupts(self):
        """NIC IRQ and softirq rates.

        Two shapes, two rules:
        - NIC IRQ polls may be split across fragment rows; the observer
          records the split ranges per list at `observation.slices.irqs`
          (one flat `[first, end, total]` triple per fragment row). Rows
          are grouped by capture, entity and `sample_begin_ns`. A poll is
          complete only when its ranges form a disjoint exact cover of the
          recorded list (no gaps, no overlaps); a fragment reporting
          `irqs_truncated` (the source's own cap) or a NIC
          `coverage_problem` (a listed vector missing from the reading)
          makes the poll incomplete as well. Totals, reset flags and
          slices are read from the same data block as the IRQ list (the
          untyped `observation` form, or the typed `transport.tcp` form
          for schema-valid tcp-typed rows). Rates publish only against a
          common complete baseline: every IRQ of the poll carries the
          baseline of one previous complete poll, and the IRQ sets match.
          An incomplete poll clears the baseline entirely (its
          readings are never a baseline — even when it also carries a
          reset or identity-change flag), leaving a gap until the next
          complete poll establishes a new one — the transition interval
          stays unavailable rather than showing a partial or averaged
          whole-NIC rate. A zero-interval poll (two readings with the same
          end) re-baselines to the later reading without publishing.
          A complete poll marked with a reset or identity change establishes
          the baseline for the following interval.
        - Softirq readings are per (entity, CPU) counters: each keeps its
          own previous reading and its own timestamp, so an intervening
          reading of another CPU (or of NET_TX) never becomes CPU 0's
          denominator, and a counter that decreases is rebaselined: the
          reset interval stays unavailable, the decreased value is the
          baseline for the next reading, and measured zeros are kept.
        """
        lane = []
        for capture in self.rows["l3"]:
            # ---- NIC polls: group fragment rows per poll ----------------
            polls = {}  # (begin_ns) -> poll accumulator
            for _line_no, row in capture.rows:
                if (row.get("kind") != "endpoint_metrics"
                        or row.get("scope") != "interrupts"
                        or row.get("entity") != "nic"):
                    continue
                data = row.get("observation") or {}
                if not isinstance(data.get("irqs"), list):
                    data = (row.get("transport") or {}).get("tcp") or {}
                irqs_list = data.get("irqs")
                if not isinstance(irqs_list, list):
                    continue
                begin = row.get("sample_begin_ns")
                end = row.get("sample_end_ns") or row.get("time_ns")
                if begin not in polls:
                    polls[begin] = {
                        "end": end, "totals": {}, "ranges": [],
                        "rebaseline": False, "list_total": None,
                        "complete": True,
                    }
                poll = polls[begin]
                poll["end"] = max(poll["end"] or 0, end or 0)
                # the source's own cap omits whole IRQ vectors (the
                # recorded `slices` list shrinks to the slots that were
                # read): the recorded per-IRQ totals no longer cover every
                # listed vector — the whole-NIC total must be withheld, as
                # when a listed vector is missing or unreadable (a NIC
                # `coverage_problem`). The per-IRQ `per_cpu_truncated`
                # flag (a top-32 map) does NOT invalidate the poll: the
                # recorded `total` remains authoritative.
                if data.get("irqs_truncated") is True \
                        or data.get("coverage_problem"):
                    poll["complete"] = False
                if data.get("counter_reset_or_identity_change") or any(
                    item.get("counter_reset_or_identity_change")
                    for item in irqs_list
                ):
                    poll["rebaseline"] = True
                # the poll fragment ranges: the observer records
                # `slices.irqs` = [first, end, total] (a flat triple per
                # fragment row)
                group_slices = data.get("slices", {}).get("irqs") \
                    if isinstance(data.get("slices"), dict) else None
                if isinstance(group_slices, list):
                    pieces = group_slices
                    if (len(pieces) == 3 and all(
                        isinstance(v, (int, float)) for v in pieces
                    )):
                        pieces = [list(pieces)]
                    for piece in pieces:
                        if isinstance(piece, dict):
                            rng = (int(piece.get("first") or 0),
                                   int(piece.get("end") or 0),
                                   int(piece.get("total") or 0))
                        elif isinstance(piece, (list, tuple)):
                            values = list(piece) + [0, 0, 0]
                            rng = (int(values[0] or 0),
                                   int(values[1] or 0),
                                   int(values[2] or 0))
                        else:
                            rng = (int(piece), int(piece), int(piece))
                        if rng not in poll["ranges"]:
                            poll["ranges"].append(rng)
                for item in irqs_list:
                    total = item.get("total")
                    if isinstance(total, (int, float)):
                        poll["totals"][str(item.get("irq"))] = int(total)
                for piece in poll["ranges"]:
                    if piece[2]:
                        poll["list_total"] = piece[2]
                        break
            # ---- publish one rate per usable complete poll --------------
            def _rng(piece):
                if isinstance(piece, dict):
                    return (piece.get("first", 0), piece.get("end", 0),
                            piece.get("total", 0))
                values = list(piece) if isinstance(
                    piece, (list, tuple)
                ) else [piece]
                first = values[0] if values else 0
                last = values[1] if len(values) > 1 else first
                whole = values[2] if len(values) > 2 else last
                return (int(first or 0), int(last or 0), int(whole or 0))
            baseline = None  # (per-IRQ values, poll end) or None
            for poll_begin in sorted(polls, key=lambda k: (
                polls[k]["end"] or 0, k,
            )):
                poll = polls[poll_begin]
                ranges = sorted(poll["ranges"])
                # complete: an unsplit poll is complete by definition; a
                # split one must disjointly cover the recorded list
                if ranges and poll["list_total"] is not None:
                    covered = 0
                    expected_end = 0
                    ok = True
                    for beg, last, _whole in ranges:
                        if beg != expected_end:
                            ok = False
                            break
                        covered += last - beg
                        expected_end = last
                    if covered != poll["list_total"]:
                        ok = False
                    if not ok:
                        # incomplete poll: never a baseline; the gap stays
                        # until the next complete poll
                        baseline = None
                        continue
                if not poll.get("complete", True):
                    # the collector omitted IRQ vectors (irqs_truncated) or
                    # a listed vector was missing/unreadable
                    # (coverage_problem): the fragment readings cannot
                    # cover the whole list — the whole-NIC baseline clears
                    # entirely and the interval stays absent (never a
                    # partial or zero whole-NIC rate). This gate wins over
                    # a reset flag on the same poll: partial counters must
                    # never become the baseline.
                    baseline = None
                    continue
                flagged = poll.get("rebaseline", False)
                if flagged:
                    # a complete flagged poll (explicit reset or identity
                    # change) is the new baseline: its own interval is
                    # unavailable (nothing publishes for it) and the next
                    # usable interval measures against the flagged reading
                    # — patch #26 preserved that 5/s interval
                    baseline = (dict(poll["totals"]), poll["end"])
                    continue
                if baseline is None:
                    # a first poll establishes the baseline without a rate
                    baseline = (dict(poll["totals"]), poll["end"])
                    continue
                # common baseline: the IRQ sets must match — a changed set
                # re-baselines without publishing
                if set(poll["totals"]) != set(baseline[0]):
                    baseline = (dict(poll["totals"]), poll["end"])
                    continue
                rates = {}
                if poll["end"] > baseline[1]:
                    # a zero interval (two readings with the same end) can
                    # never publish; the later reading just re-baselines
                    for irq, value in sorted(poll["totals"].items()):
                        before = baseline[0][irq]
                        delta = value - before
                        if delta < 0:
                            # a decrease without a flag: no rate; the new
                            # value is the baseline for the next reading
                            rates = {}
                            break
                        rates[irq] = delta / (
                            (poll["end"] - baseline[1]) / 1e9
                        )
                if rates and poll["end"] > baseline[1]:
                    projected = self.project(
                        capture.meta.get("clock_id"), poll["end"]
                    )[0]
                    lane.append({
                        "t": projected,
                        "endpoint": capture.context.get("endpoint"),
                        "irq_total_per_s": round(sum(rates.values()), 1),
                        "entities": {
                            k: round(v, 1) for k, v in sorted(rates.items())
                        },
                    })
                baseline = (dict(poll["totals"]), poll["end"])
            # ---- softirqs: per (entity, CPU) counters -------------------
            soft = {}  # (name, cpu) -> (value, stamp)
            for _line_no, row in capture.rows:
                if (row.get("kind") != "endpoint_metrics"
                        or row.get("scope") != "interrupts"
                        or row.get("entity") != "softirqs"):
                    continue
                data = row.get("observation") or {}
                if not isinstance(data.get("softirqs"), list):
                    data = (row.get("transport") or {}).get("tcp") or {}
                items = data.get("softirqs")
                if not isinstance(items, list):
                    continue
                end = row.get("sample_end_ns") or row.get("time_ns")
                entry = None
                if data.get("counter_reset_or_identity_change"):
                    soft.clear()
                for item in items:
                    name = item.get("name")
                    rates = {}
                    for cpu, count in self._irq_cpus("softirqs", item):
                        key = (name, cpu)
                        before, stamp = soft.get(key, (None, None))
                        soft[key] = (count, end)
                        if stamp is None or before is None:
                            continue
                        if not end or end <= stamp:
                            continue
                        delta = count - before
                        if delta < 0:
                            continue
                        rates[key] = delta / ((end - stamp) / 1e9)
                    if rates:
                        projected = self.project(
                            capture.meta.get("clock_id"), end
                        )[0]
                        entry = entry or {
                            "t": projected,
                            "endpoint": capture.context.get("endpoint"),
                        }
                        if name == "NET_RX":
                            entry["net_rx_per_s"] = round(
                                sum(rates.values()), 1
                            )
                        elif name == "NET_TX":
                            entry["net_tx_per_s"] = round(
                                sum(rates.values()), 1
                            )
                if entry:
                    lane.append(entry)
        # time-ordered within each endpoint
        lane.sort(key=lambda point: (
            point.get("endpoint") or "", point.get("t") or 0))
        self.irq_lane = lane

    @staticmethod
    def _irq_cpus(source, item):
        """(cpu, count) pairs as the writer recorded them: `per_cpu` maps for
        IRQ rows, parallel `cpus`/`counts` arrays for the softirq rows."""
        if source == "irqs":
            for cpu, value in (item.get("per_cpu") or {}).items():
                yield str(cpu), value
            return
        cpus = item.get("cpus") or []
        counts = item.get("counts") or []
        for cpu, value in zip(cpus, counts):
            yield str(cpu), value

    def _lane_ebpf(self):
        """Wrapped events with qualified placement; coverage is computed
        per enabled observer and then combined: a disabled observer says
        nothing, and one partial observer keeps the run partial."""
        events = []
        enabled = False
        scoped_rows = 0
        observer_states = []
        for capture in self.rows["l3"]:
            # the writer's own enablement record: `config.ebpf` (with the
            # nested `config.probes.ebpf` as the older spelling)
            config = capture.meta.get("config") or {}
            probes = config.get("probes") or config
            capture_ebpf = bool(
                (probes.get("ebpf") or {}).get("enabled")
            )
            enabled = enabled or capture_ebpf
            has_ebpf_rows = False
            capture_failures = 0
            saw_footer = False
            for _line_no, row in capture.rows:
                if row.get("kind") != "endpoint_metrics":
                    continue
                if row.get("scope") != "ebpf":
                    continue
                scoped_rows += 1
                has_ebpf_rows = True
                enabled = True
                # the writer puts tcp-scoped payloads under `transport.tcp`;
                # `observation` carries only non-transport-scoped data
                data = (row.get("transport") or {}).get("tcp") \
                    or (row.get("observation") or {})
                if data.get("type") == "tcp_event":
                    qualified = bool(data.get("time_conversion_qualified"))
                    when = (data.get("endpoint_time_ns")
                            if qualified else data.get("timestamp_ns"))
                    projected = self.project(
                        capture.meta.get("clock_id"), when or 0
                    )[0] if qualified else None
                    events.append({
                        "t": projected,
                        "raw_kernel_ns": data.get("timestamp_ns"),
                        "placement": (
                            "qualified" if qualified
                            else "unplaced (kernel time, no qualified "
                            "conversion)"
                        ),
                        "event": data.get("event"),
                        "local_ip": data.get("local_ip"),
                        "local_port": data.get("local_port"),
                        "peer_ip": data.get("peer_ip"),
                        "peer_port": data.get("peer_port"),
                        "snapshot": data.get("snapshot"),
                    })
            stats_end = [
                r for _, r in capture.rows if r.get("type") == "end"
            ]
            if stats_end:
                saw_footer = True
                end_stats = stats_end[-1].get("stats") or {}
                capture_failures += end_stats.get("ebpf_failures", 0)
                # honor the terminal status and the loss counters: a
                # non-complete footer, or any dropped probe row, leaves the
                # channel partial
                if stats_end[-1].get("complete") is False:
                    capture_failures += 1
                if end_stats.get("drop_probe_rows"):
                    capture_failures += 1
            if not (capture_ebpf or has_ebpf_rows):
                # a disabled observer says nothing about eBPF coverage
                continue
            observer_states.append(
                "partial"
                if not (has_ebpf_rows and saw_footer
                        and capture_failures == 0)
                else "complete"
            )
        if not enabled:
            coverage = "disabled"
        elif scoped_rows == 0 or any(
            state == "partial" for state in observer_states
        ):
            coverage = "partial"
        else:
            coverage = "complete"
        self.ebpf = {
            "coverage": coverage,
            "events": sorted(
                events,
                key=lambda e: e["t"] if e["t"] is not None else 0
            ),
            "observers": observer_states,
        }

    def _lane_collector(self):
        """TENT (25/26) and NIXL telemetry (27/28): rates per target/agent.

        Cumulative deltas divided by the end-to-end elapsed time between
        consecutive valid readings of the same series — only inside one
        reset generation, only across readings the writer marked complete
        and undiscontinuous. Units are preserved (bytes stay bytes/s,
        counts stay counts/s).
        """
        lanes = []
        for capture in self.rows["l2"]:
            samples = {}
            for _line_no, row in capture.rows:
                kind = row.get("kind")
                if kind not in (
                    "transport_metrics", "transport_metrics_failures",
                    "nixl_telemetry", "nixl_telemetry_events",
                ):
                    continue
                target = row.get("target") or {}
                identity = (
                    target.get("id"), json.dumps(
                        row.get("labels") or {}, sort_keys=True
                    ), kind,
                )
                scrape = row.get("scrape") or {}
                samples.setdefault(identity, []).append({
                    "generation": int(row.get("reset_generation") or 0),
                    "partial": bool(row.get("partial")),
                    "discontinuity": bool(row.get("discontinuity")),
                    "end": scrape.get("end_ns") or row.get("time_ns"),
                    "begin": scrape.get("begin_ns"),
                    "row": row,
                })
            for identity, items in sorted(
                samples.items(), key=lambda kv: str(kv[0])
            ):
                items.sort(key=lambda item: item["end"] or 0)
                points = []
                prior = None
                for item in items:
                    units = item["row"].get("units") or {}
                    fields = item["row"].get("fields") or {}
                    rates = {}
                    gauges = {}
                    comparable = (
                        prior is not None
                        and prior["generation"] == item["generation"]
                        and not item["partial"]
                        and not prior["partial"]
                        and not item["discontinuity"]
                        and not prior["discontinuity"]
                    )
                    if comparable:
                        elapsed = (item["end"] or 0) - (prior["end"] or 0)
                        if elapsed <= 0:
                            comparable = False
                    if comparable:
                        prior_fields = prior["row"].get("fields") or {}
                        for name, value in fields.items():
                            if not isinstance(value, (int, float)):
                                continue
                            if name == GAUGE_FIELD:
                                # a gauge never becomes a rate: its sampled
                                # byte value is carried as recorded
                                gauges[name] = {
                                    "value": value,
                                    "unit": units.get(name),
                                }
                                continue
                            before = prior_fields.get(name)
                            if not isinstance(before, (int, float)):
                                continue
                            rates[name] = {
                                "rate": round(
                                    (value - before) / (elapsed / 1e9), 6
                                ),
                                "unit": units.get(name),
                            }
                    else:
                        # even an uncompared reading still carries its
                        # gauge values (a steady gauge must not read zero)
                        for name, value in fields.items():
                            if isinstance(value, (int, float)) and \
                                    name == GAUGE_FIELD:
                                gauges[name] = {
                                    "value": value,
                                    "unit": units.get(name),
                                }
                    if fields or not comparable:
                        projected = self.project(
                            capture.meta.get("clock_id"),
                            item["end"] or 0,
                        )[0]
                        points.append({
                            "t": projected,
                            "interval_ns": (
                                elapsed if comparable else None
                            ),
                            "rates": rates,
                            "gauges": gauges,
                            "reset_generation": item["generation"],
                            "partial": item["partial"],
                            "discontinuity": item["discontinuity"],
                            "scrape_begin_ns": item["begin"],
                            "scrape_end_ns": item["end"],
                        })
                    prior = item
                target, labels, kind = identity
                lanes.append({
                    "capture": capture.process_id,
                    "target": target,
                    "labels": json.loads(labels),
                    "kind": kind,
                    "label": f"{target or '?'} · {kind}",
                    "points": points,
                })
        self.collector_series = lanes

    def _entry_times(self, entry, clock_id):
        """A summary entry's recorded times, projected from its own source."""
        times = []
        for name, value in entry.items():
            if name.endswith("_time_ns") and isinstance(value, int):
                projected = self.project(clock_id, value)[0]
                if projected is not None:
                    times.append(projected)
        for handoff in entry.get("handoffs", []):
            if isinstance(handoff.get("time_ns"), int):
                projected = self.project(clock_id, handoff["time_ns"])[0]
                if projected is not None:
                    times.append(projected)
        return times

    def _request_first_time(self, row):
        """Earliest projected time among the logical request's paired
        records; every source projects from its own clock first."""
        times = []
        times += self._entry_times(row["spine"],
                                   row["source"].get("clock_id"))
        for producer in row.get("producer", []):
            times += self._entry_times(
                producer["entry"], producer["source"].get("clock_id")
            )
        for frontend in row.get("frontends", []):
            times += self._entry_times(
                frontend["entry"], frontend["source"].get("clock_id")
            )
        for part in row.get("others", []):
            times += self._entry_times(
                part["entry"], part["source"].get("clock_id")
            )
        for transport in row.get("transports", []):
            if transport.get("t0") is not None:
                times.append(transport["t0"])
        return min(times) if times else None

    def _request_row(self, tag, row):
        """One logical scheduler request; all values from its records.

        Durations are milliseconds, throughput GB/s — converted once, here.
        Mooncake member requests surface their batch's own share of the
        bytes; the physical batch counts once in the pair lanes.
        """
        source = row["source"]
        entry = row["spine"]
        request, occurrence = tag[1], tag[2]
        transports = row.get("transports", [])
        pair = None
        read_ms, gpss = [], []
        prep_ms = []
        bytes_total, desc_total = None, 0
        tids = set()
        mooncake_member = False
        planned = False
        for transport in transports:
            pair = pair or transport.get("pair")
            if transport.get("duration_ns"):
                read_ms.append(transport["duration_ns"] / 1e6)
            if transport.get("throughput"):
                gpss.append(transport["throughput"] / 1e9)
            if transport.get("prep_to_post_ns") is not None:
                prep_ms.append(transport["prep_to_post_ns"] / 1e6)
            if transport.get("mooncake") and request in transport.get(
                "member_shares", {}
            ):
                if not transport.get("confirmed"):
                    # planned, never confirmed: the batch's write did not
                    # complete, so the rows carry no confirmed bytes
                    planned = True
                else:
                    mooncake_member = True
                    share_bytes, share_desc, _ = transport["member_shares"][
                        request
                    ]
                    bytes_total = (bytes_total or 0) + (
                        share_bytes if share_bytes is not None else 0
                    )
                    if share_bytes is None and share_desc is None:
                        planned = True
                    if share_desc is not None:
                        desc_total += share_desc
            else:
                value = transport.get("bytes")
                if value is not None:
                    bytes_total = (bytes_total or 0) + value
                if transport.get("descriptors") is not None:
                    desc_total += transport["descriptors"]
            if transport.get("thread") is not None:
                tids.add(transport["thread"])
            if transport.get("planned_basis"):
                planned = True
        bases = sorted({
            transport.get("throughput_basis") for transport in transports
            if transport.get("throughput_basis")
        })
        intervals = [
            (transport["t0"], transport["t1"]) for transport in transports
            if transport.get("t0") is not None
            and transport.get("t1") is not None
        ]
        peak = self._sweep(intervals) if intervals else None
        states = [entry.get("state")]
        for part in row.get("producer", []):
            states.append(part["entry"].get("state"))
        for part in row.get("others", []):
            states.append(part["entry"].get("state"))
        failures = entry.get("failures") or []
        if transports:
            if all(t["state"] == "completed" for t in transports) \
                    and not failures:
                merged = "completed"
            elif any(t["state"] == "failed" for t in transports) or failures:
                merged = "failed"
            else:
                merged = "incomplete"
        else:
            merged = ("completed" if "completed" in states
                      else "failed" if "failed" in states else "incomplete")
        arrival = None
        for frontend in row.get("frontends", []):
            arrival = self._min_ns(arrival, self._frontend_arrival(
                frontend["source"].get("process_id"),
                frontend["entry"].get("request"),
            ))
        # posting-thread utilization: the strip is chosen by the worker
        # generation, TID and interval overlap with the transport (never
        # busiest-first), over the samples' projected intervals
        post_cpu = None
        for transport in transports:
            if transport.get("t0") is None or transport.get("t1") is None:
                continue
            interval = (transport["t0"], transport["t1"])
            for strip in self._posting_strips(
                transport.get("process_id"), transport.get("thread"),
                interval,
            ):
                for point in strip["points"]:
                    if point["t0"] is None or point["t1"] is None:
                        continue
                    if (point["t0"] < transport["t1"]
                            and point["t1"] > transport["t0"]
                            and point["cpu"] is not None):
                        post_cpu = max(post_cpu or 0, point["cpu"])
        candidates = self._transfer_candidates(transports)
        roles = ["scheduler"]
        roles += ["producer"] if row.get("producer") else []
        roles += ["frontend"] if row.get("frontends") else []
        roles += list({part["role"] for part in row.get("others", [])})
        # each operation draws at its own measured start and end
        spans = [
            {
                "t0": transport["t0"],
                "t1": transport["t1"],
                "ms": (transport["duration_ns"] / 1e6
                       if transport.get("duration_ns") else None),
                "gbs": (round(transport["throughput"] / 1e9, 3)
                        if transport.get("throughput") else None),
                "state": transport.get("state"),
                "key": transport["key"],
            }
            for transport in transports
            if transport.get("t0") is not None
        ]
        return {
            "key": {
                "run": self.index,
                "scheduler_process_id": source.get("process_id"),
                "request": request,
                "occurrence": occurrence,
            },
            "display": request + (f"#{occurrence + 1}"
                                  if entry.get("request_occurrences", 1) > 1
                                  else ""),
            "state": merged,
            "state_basis": ("merged from attached occurrences" if transports
                            else "spine occurrence"),
            "roles": roles,
            "pair": pair,
            "pair_display": self._pair_display(pair),
            "read_threads": sorted(tids),
            "transports": len(transports),
            "transport_keys": sorted(row.get("transport_keys", ())),
            "spans": spans,
            "first_record": self._request_first_time(row),
            "arrive": arrival,
            "timeline": self._timeline_for(
                source.get("process_id"), request, occurrence
            ),
            "prep_ms": prep_ms,
            "read_ms": read_ms,
            "gbs": gpss,
            "gbs_basis": bases,
            "peak_concurrency": peak,
            "mib": (round(bytes_total / 2 ** 20, 3)
                    if bytes_total is not None else None),
            "descriptors": desc_total if desc_total else None,
            "posting_thread_cpu": post_cpu,
            "bytes_basis": ("batch member share (confirmed)"
                            if mooncake_member and not planned
                            else "batch member share (planned, not "
                            "confirmed)" if planned and mooncake_member
                            else "completed payload"),
            "planned_symbol": planned,
            "failures": entry.get("failures"),
            "not_recorded": ["prefill_window", "kv_wait", "decode_window",
                             "ttft"],
            **candidates,
        }

    def _pair_display(self, pair):
        """Engine-identity pair key -> a readable, distinct label."""
        if not pair or "→" not in pair:
            return pair or "?"
        producer, consumer = pair.split("→", 1)
        left = self.registry.get(producer, {}).get("label") or (
            f"engine-{producer[:6]}" if producer != "?" else "?"
        )
        right = self.registry.get(consumer, {}).get("label") or (
            f"engine-{consumer[:6]}" if consumer != "?" else "?"
        )
        return f"{left}→{right}"

    def _transfer_candidates(self, transports):
        """RTT / retransmits through the probe join's unique candidates.

        Values reach the row only when the association is a unique temporal
        candidate; ambiguous or absent associations stay explicit.
        """
        rtt_max = retx = None
        reasons = set()
        for transport in transports:
            if transport.get("t0") is None:
                continue
            found = self.probe_candidates.get(
                (transport.get("process_id"), transport["key"].rsplit(
                    ":", 1)[-1]
                )
            )
            if found is None:
                reasons.add("no probe-join association")
                continue
            association, entries, events = found
            if association != "unique_temporal_candidate":
                reasons.add(f"association {association}")
                continue
            for entry in entries:
                for point in entry.get("points", []):
                    value = point.get("rtt_us")
                    if value is None:
                        continue
                    # the sample's own interval must overlap the transfer's
                    # measured window: both endpoints count (a sample that
                    # straddles the READ end by nanoseconds is a genuine
                    # observation of the READ)
                    if point.get("t1") is None or point.get("t0") is None:
                        continue
                    if (point["t0"] <= transport["t1"]
                            and point["t1"] >= transport["t0"]):
                        rtt_max = max(rtt_max or 0, value)
            # retransmits are counted by their event type only; state
            # changes and resets are not retransmissions
            for tcp_event in events:
                if tcp_event.get("event") == "tcp_retransmit_skb":
                    retx = (retx or 0) + 1
        if rtt_max is not None or retx is not None:
            basis = "unique_temporal_candidate"
        else:
            basis = "; ".join(sorted(reasons)) or "no probe-join association"
        return {"rtt_max_us_cand": rtt_max, "retx_cand": retx,
                "candidate_basis": basis}

    @staticmethod
    def _min_ns(current, candidate):
        if candidate is None:
            return current
        if current is None:
            return candidate
        return min(current, candidate)

    def _frontend_arrival(self, process_id, request):
        """The frontend's own request_bind time for this id (exact id)."""
        data = self.l1_rows.get(process_id)
        if not data or not request:
            return None
        earliest = None
        for event in data["events"]:
            if event.get("kind") != "request_bind":
                continue
            if SUMMARY._is_member(event):
                continue
            if data["decoder"].resolve(event.get("request_ref") or 0) \
                    != request:
                continue
            projected = self.project(
                self._capture_by_process(process_id).meta.get("clock_id"),
                event.get("time_ns"),
            )[0]
            if projected is not None:
                earliest = (projected if earliest is None
                            else min(earliest, projected))
        return earliest

    def _posting_strip(self, source, tid):
        """Single-strip convenience over :meth:`_posting_strips`."""
        strips = self._posting_strips(None, tid, None, source=source)
        return strips[0] if strips else None

    def _posting_strips(self, process_id, tid, interval, source=None):
        """All thread strips of the transfer's posting thread that overlap
        the transport's measured interval: the worker generation (its own
        process capture), TID and lifetime select the strip — never
        busiest-first."""
        if tid is None:
            return []
        process_id = process_id if process_id is not None else (
            source.get("process_id") if source else None
        )
        capture = self._capture_by_process(process_id)
        if capture is None:
            return []
        meta = capture.meta
        lo, hi = interval or (None, None)
        out = []
        for strip in self.thread_strips:
            identity = strip["identity"]
            if identity["tid"] != tid or not strip.get(
                "joined_process", False
            ):
                continue
            if not (identity["boot_id"] == meta.get("boot_id")
                    and identity["net_namespace"] == meta.get("net_namespace")
                    and identity["pid_namespace"]
                    == meta.get("pid_namespace")
                    and identity["pid"] == meta.get("pid")):
                continue
            if lo is not None and hi is not None:
                # the strip's samples must actually cover the transport's
                # measured window (two lifetimes of one TID never both do)
                covers = any(
                    point["t0"] is not None and point["t1"] is not None
                    and point["t0"] < hi and point["t1"] > lo
                    for point in strip["points"]
                )
                if not covers:
                    continue
            out.append(strip)
        return out

    def _build_probe_candidates(self):
        """The probe join's qualified local flow candidates, loaded by path.

        Keyed (worker process id, transfer handle): (association, socket
        series (matched by the writer's own recorded entity), tcp events).
        Values reach a request only through a unique temporal candidate;
        everything else stays explicit with its reason.
        """
        import importlib.util

        self.probe_candidates = {}
        root = Path(__file__).resolve().parent
        worker_files = [
            c.path for c in self.captures
            if c.layer == "l1" and not c.problems
            and c.context.get("role") == "worker"
        ]
        probe_files = [c.path for c in self.rows.get("l3", [])]
        if not worker_files or not probe_files:
            return
        name = f"{_PACKAGE}.probe_join"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                name, root / "probe_join.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        probe_join = sys.modules[name]
        try:
            results = probe_join.candidates(
                worker_files, probe_files, self.group
            )
        except (ValueError, KeyError, OSError) as error:
            self._problem("probe_join", f"{type(error).__name__}: {error}")
            return
        entity_series = self.rtt_by_entity
        for result in results:
            association = result.get("association")
            series = []
            events = []
            for candidate in result.get("candidates", []):
                for key in entity_series.get(candidate.get("entity"), []):
                    match = next((s for s in self.rtt_series
                                  if self._rtt_key(s) == key), None)
                    if match is not None:
                        series.append(match)
                for tcp_event in candidate.get("tcp_events", []):
                    # The candidate association is temporal; the event's own
                    # time stays raw (kernel time), never re-anchored here.
                    events.append({**tcp_event, "t": None})
            self.probe_candidates[(
                result.get("process_id"), str(result.get("transfer"))
            )] = (association, series, events)

    @staticmethod
    def _rtt_key(series):
        identity = series["identity"]
        return (
            identity["net_namespace"], identity["local_ip"],
            identity["local_port"], identity["peer_ip"],
            identity["peer_port"], identity["incarnation"],
        )

    def _build_runkv(self):
        rows = [self._request_row(tag, row) for tag, row in self.spine.items()]
        self.request_rows = rows
        # The header counts distinct scheduler requests — never the folded
        # frontend/producer observations, listed separately in /api/data.
        eligible = []
        throughputs = []
        for transport in self.transports:
            if transport["state"] == "completed" and transport.get(
                "duration_ns"
            ):
                eligible.append(transport["duration_ns"] / 1e6)
            if transport.get("throughput"):
                throughputs.append(transport["throughput"])
        self.runkv = {
            "requests": len(rows),
            "observations": len(self.observations),
            "transports": len(self.transports),
            "read_p50_ms": self._pct(eligible, 50),
            "read_p99_ms": self._pct(eligible, 99),
            "read_p95_ms": self._pct(eligible, 95),
            "read_max_ms": max(eligible) if eligible else None,
            "gbs_p50": self._pct(
                [t / 1e9 for t in throughputs], 50
            ) if throughputs else None,
            "gbs_p99": self._pct(
                [t / 1e9 for t in throughputs], 99
            ) if throughputs else None,
            "slow_reads": sum(
                1 for t in self.transports
                if t["state"] == "completed" and t.get("duration_ns")
                and self._pct(eligible, 50) is not None
                and t["duration_ns"] / 1e6 > 2 * self._pct(eligible, 50)
            ),
            "percentile_method": "nearest-rank over eligible completed "
            "transfers (the summary publishes p50/p95/max; p99 here)",
        }

    @staticmethod
    def _pct(samples, percent):
        if not samples:
            return None
        ordered = sorted(samples)
        return ordered[max(0, math.ceil(percent * len(ordered) / 100) - 1)]


    # ------------------------------------------------------------- payloads
    def features(self):
        """Where each request-timeline field comes from in this run."""
        if not getattr(self, "timelines", None):
            reason = "not recorded in this run's captures"
            return dict.fromkeys(
                ("prefill_window", "kv_wait", "decode_admission", "decode_window",
                 "ttft"), reason)  # fmt: skip
        return {
            "prefill_window": "producer timeline: intervals_ns.prefill_window",
            "kv_wait": "consumer timeline: stall.waiting_for_kv_ns",
            "decode_admission": "consumer timeline: intervals_ns.queued_to_admitted",
            "decode_window": "consumer timeline: decode_window, first token to finish",
            "ttft": "consumer timeline: api.arrival_ns to api.first_token_output_ns",
        }

    def api_run(self):
        captures = []
        for capture in self.captures:
            end_present = False
            complete = None
            if capture.layer == "l1":
                for row in self.l1_rows.get(capture.process_id, {}).get(
                    "rows", []
                ):
                    if row.get("type") == "end":
                        end_present = True
                        complete = row.get("complete")
            else:
                for _, row in capture.rows:
                    if row.get("type") == "end":
                        end_present = True
                        complete = row.get("complete")
            captures.append({
                "layer": capture.layer,
                "path": str(capture.path),
                "process_id": capture.process_id,
                "role": capture.context.get("role"),
                "kv_role": capture.context.get("kv_role"),
                "endpoint": capture.context.get("endpoint"),
                "pid": capture.meta.get("pid"),
                "fingerprint": capture.fingerprint[:16],
                "duplicates": capture.folded,
                "problems": capture.problems,
                "end_present": end_present,
                "complete": complete,
                "clock_id": capture.meta.get("clock_id"),
                "aligned": self.project(capture.meta.get("clock_id"),
                                        self.window[0] if self.window
                                        else 0)[0] is not None,
            })
        sources_present = {
            "l1": any(c.layer == "l1" for c in self.captures),
            "l2": bool(self.rows["l2"]),
            "l3": bool(self.rows["l3"]),
            "ebpf": self.ebpf.get("coverage") != "missing",
            "ucx_protocols": bool(self.ucx_docs),
            "client_bench": bool(list(self.dir.rglob("bench.log"))),
        }
        absent = [
            name for name, present in sources_present.items() if not present
        ]
        return {
            "run_index": self.index,
            "run_dir": str(self.dir),
            "run_id": self.run_id,
            "reference_clock": self.reference_clock,
            "cross_node": self.cross_node,
            "calibrations": [
                {k: v for k, v in entry.items() if k != "mapping"}
                for entry in self.calibration_status
            ],
            "captures": captures,
            "input_fingerprints": self.input_fingerprints(),
            "summary_source": self.summary_source,
            "problems": self.problems + self.summary.get("problems", []),
            "not_available": self.summary.get("not_available", []),
            "features": self.features(),
            "absent_sources": absent,
            "thread_strips_truncated": self.thread_strips_truncated,
            "engine_registry": self.registry,
            "percentile_method": self.runkv.get("percentile_method"),
            "thread_cpu_label": "shared thread activity during the sampled "
            "interval",
            "goodput_label": "completed payload bytes spread uniformly over "
            "each transfer's measured span (an estimate)",
            "rate_label": "deltas over the end-to-end elapsed time between "
            "consecutive readings of the same series",
        }

    def _strip_engine(self, strip):
        key = (strip["identity"]["boot_id"],
               strip["identity"]["net_namespace"],
               strip["identity"]["pid_namespace"],
               strip["identity"]["pid"])
        for capture in self.captures:
            if capture.layer != "l1":
                continue
            meta = capture.meta
            if (meta.get("boot_id"), meta.get("net_namespace"),
                    meta.get("pid_namespace"), meta.get("pid")) == key:
                return {
                    "label": _label_of(capture.context.get("engine_id"),
                                       self.registry),
                    "engine_id": capture.context.get("engine_id"),
                }
        return {"label": None, "engine_id": None}

    def api_data(self):
        pair_labels = {
            pair: self._pair_display(pair) for pair in self.lane_gbs
        }
        dots = []
        for t in sorted(self.transports, key=lambda t: t["t0"] or 0):
            if t["t0"] is None:
                continue
            dots.append({
                "t": t["t0"],
                "ms": (
                    t["duration_ns"] / 1e6 if t["duration_ns"] else None
                ),
                "pair": t["pair"],
                "state": t["state"],
                # throughput converted once: GB/s everywhere downstream
                "gbs": (round(t["throughput"] / 1e9, 3)
                        if t.get("throughput") else None),
                "thread": t.get("thread"),
                "thread_name": t.get("thread_name"),
                "key": t["key"],
            })
        return {
            "run_index": self.index,
            "window": {"t0": self.window[0], "t1": self.window[1]},
            "bin_ms": BIN_MS,
            "pairs": sorted(self.lane_gbs),
            "pair_labels": pair_labels,
            "features": self.features(),
            "dot_label": "one physical transfer call (READ or WRITE batch); "
            "duration = connector's own call/observation interval",
            "gbs_label": self.api_run()["goodput_label"],
            "dots": dots,
            "inflight": {
                pair: [[self.window[0] + b * BIN_MS * 1_000_000, v]
                       for b, v in enumerate(lane)]
                for pair, lane in self.lane_inflight.items()
            },
            "gbs": self.lane_gbs,
            "thread_strips": [
                {**strip,
                 # the identity join keys by engine id; the label is only
                 # for display
                 "engine": self._strip_engine(strip)["engine_id"]
                 or self._strip_engine(strip)["label"],
                 "engine_label": self._strip_engine(strip)["label"],
                 "engine_id": self._strip_engine(strip)["engine_id"]}
                for strip in self.thread_strips
            ],
            "rtt": self.rtt_series,
            "irq": self.irq_lane,
            "ebpf": self.ebpf,
            "collector": self.collector_series,
            "requests": self.request_rows,
            # genuine observations that attach to no scheduler request,
            # listed separately and never counted as requests
            "observations": self.observations,
            "runkv": self.runkv,
            "peak_per_pair": {k or "?": v for k, v in self.lane_peaks.items()},
        }

    def api_config(self):
        configs = []
        for capture in self.captures:
            configs.append({
                "layer": capture.layer,
                "process_id": capture.process_id,
                "role": capture.context.get("role"),
                "endpoint": capture.context.get("endpoint"),
                "metadata_config": capture.meta.get("config"),
                "caps": {
                    key: capture.meta.get(key)
                    for key in ("capabilities", "start_ticks", "clock_id",
                                "boot_id", "net_namespace", "pid_namespace")
                },
            })
        return {"configs": configs, "ucx_protocols": self.ucx_docs}

    def api_capture(self, process_id):
        capture = self._capture_by_process(process_id)
        if capture is None:
            return None
        payload = {
            "layer": capture.layer,
            "path": str(capture.path),
            "process_id": capture.process_id,
            "metadata": capture.meta,
            "duplicates": capture.folded,
            "problems": capture.problems,
        }
        if capture.layer == "l1":
            data = self.l1_rows.get(process_id, {})
            payload["threads"] = [
                row.get("threads") for row in data.get("rows", [])
                if row.get("type") == "threads"
            ]
        return payload

    def api_records(self, scheduler_process_id, request, occurrence):
        """Every record joined to one logical request; real provenance."""
        tag = (scheduler_process_id, request, occurrence)
        row = self.spine.get(tag)
        if row is None:
            return None
        records = []
        spine = row["spine"]
        source_index = spine.get("source")
        spine_source = self._source_of(source_index)
        window = []
        for transport in row.get("transports", []):
            if transport.get("t0") is not None:
                window.append(transport["t0"])
            if transport.get("t1") is not None:
                window.append(transport["t1"])
        lo, hi = (min(window), max(window)) if window else (None, None)
        attached_processes = {spine_source.get("process_id")}
        for part in row.get("producer", []) + row.get("frontends", []) \
                + row.get("others", []):
            if part.get("source", {}).get("process_id"):
                attached_processes.add(part["source"].get("process_id"))
        for transport in row.get("transports", []):
            if transport.get("process_id"):
                attached_processes.add(transport["process_id"])
        producer_ids = {
            producer["entry"].get("request")
            for producer in row.get("producer", [])
        }
        # each attached transport's worker generation: the handle's rows
        # from its own bind through retirement in that worker's records
        generation_rows = {}
        for process_id in sorted(p for p in attached_processes if p):
            data = self.l1_rows.get(process_id)
            if not data:
                continue
            bucket = {}
            for event in data["events"]:
                handle = event.get("transfer")
                if not handle:
                    continue
                if event.get("kind") in (
                    "transfer_bind", "prepared", "post_begin", "post_return",
                    "complete_observed", "transfer_retire", "failed",
                ) or SUMMARY._is_member(event):
                    bucket.setdefault(str(handle), []).append(event)
            for transport in row.get("transports", []):
                if transport.get("process_id") != process_id:
                    continue
                handle = transport["key"].rsplit(":", 1)[-1]
                rows_handle = [row for row in bucket.get(handle, [])
                               if not SUMMARY._is_member(row)]
                rows_handle.sort(key=lambda row: row.get("sequence") or 0)
                binds = [
                    row for row in rows_handle
                    if row.get("kind") == "transfer_bind"
                    and row.get("time_ns") == transport.get("prepared_ns")
                ] or [row for row in rows_handle
                      if row.get("kind") == "transfer_bind"]
                if not binds:
                    continue
                begin = rows_handle.index(binds[0])
                generation_rows[(process_id, str(handle))] = set(
                    id(row) for row in rows_handle[begin:]
                )
        batch_handles = {
            str(transport["key"].rsplit(":", 1)[-1])
            for transport in row.get("transports", [])
            if transport.get("mooncake")
        }
        for process_id in sorted(p for p in attached_processes if p):
            data = self.l1_rows.get(process_id)
            if not data:
                continue
            capture = data["capture"]
            for event in data["events"]:
                names = set()
                for ref in ("request_ref", "handoff_ref"):
                    resolved = data["decoder"].resolve(event.get(ref) or 0)
                    if isinstance(resolved, str):
                        names.add(resolved)
                match = (
                    request in names
                    or names & producer_ids
                    or (
                        event.get("kind") == "request_bind"
                        and SUMMARY._is_member(event)
                        and str(event.get("transfer")) in batch_handles
                    )
                    or id(event) in generation_rows.get(
                        (process_id, str(event.get("transfer"))), set()
                    )
                )
                if not match:
                    continue
                line = data["lines"].get(event.get("sequence"))
                projected = self.project(capture.meta.get("clock_id"),
                                         event.get("time_ns"))[0]
                records.append({
                    "layer": "L1",
                    "path": str(capture.path),
                    "line": line,
                    "t": projected,
                    "local_ns": event.get("time_ns"),
                    "clock_id": capture.meta.get("clock_id"),
                    "kind": event.get("kind"),
                    "process_id": process_id,
                    "role": capture.context.get("role"),
                    "resolved": {ref: data["decoder"].resolve(
                        event.get(ref) or 0
                    ) for ref in ("request_ref", "handoff_ref", "peer_ref")},
                    "raw": event,
                })
        for layer, captures in self.rows.items():
            for capture in captures:
                clock = capture.meta.get("clock_id")
                for line_no, event in capture.rows:
                    begin = event.get("sample_begin_ns")
                    if begin is None and event.get("scrape"):
                        begin = event["scrape"].get("begin_ns")
                    end = (event.get("sample_end_ns")
                           or event.get("time_ns"))
                    if end is None and event.get("scrape"):
                        end = event["scrape"].get("end_ns")
                    if begin is None or lo is None:
                        continue
                    one = self.project(clock, begin)[0]
                    two = self.project(clock, end)[0]
                    if one is None or two is None:
                        continue
                    if one > hi or two < lo:
                        continue
                    records.append({
                        "layer": "L3" if layer == "l3" else "L2",
                        "path": str(capture.path),
                        "line": line_no,
                        "t": one,
                        "local_ns": begin,
                        "clock_id": clock,
                        "kind": event.get("kind"),
                        "scope": event.get("scope"),
                        "process_id": capture.process_id,
                        "role": capture.context.get("role"),
                        "candidate_only": True,
                        "raw": event,
                    })
        records.sort(key=lambda r: (r["t"] if r["t"] is not None else 0))
        return {
            "key": {
                "run": self.index,
                "scheduler_process_id": scheduler_process_id,
                "request": request,
                "occurrence": occurrence,
            },
            "logical": self._request_row(tag, row),
            "records": records,
        }


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class ViewerApp:
    """The parsed runs and the handler's data source."""

    def __init__(self, run_dirs):
        self.run_dirs = [str(d) for d in run_dirs]
        self.runs = [
            [Run(run_dir, index)] for index, run_dir in enumerate(run_dirs)
        ]

    def run(self, run_index, sub=None):
        return self.runs[run_index][0]


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "kv-viewer/1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, content_type="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, payload):
            self._send(code, json.dumps(
                payload, separators=(",", ":"), allow_nan=False
            ).encode())

        def do_GET(self):
            parsed = urlparse(self.path)
            query = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
            try:
                self._route(parsed.path, query)
            except ViewerError as error:
                self._json(500, {"error": str(error)})
            except (KeyError, TypeError, ValueError) as error:
                self._json(500, {"error": f"{type(error).__name__}: {error}"})

        def _int(self, query, name, default):
            value = query.get(name)
            if value is None:
                return default
            try:
                return int(value)
            except ValueError:
                raise ViewerError(
                    f"{name} must be an integer"
                ) from None

        def _route(self, path, query):
            if path == "/" or path == "/index.html":
                try:
                    body = _PAGE.read_bytes()
                except OSError:
                    self._send(500, b"viewer page missing", "text/plain")
                    return
                self._send(200, body, "text/html")
                return
            run_index = self._int(query, "run", 0)
            if not 0 <= run_index < len(app.runs):
                raise ViewerError(f"no run {run_index}")
            run = app.run(run_index, self._int(query, "sub", None))
            if path == "/api/index":
                self._json(200, {
                    "runs": [
                        {
                            "run": i, "dir": str(app.run_dirs[i]),
                        }
                        for i in range(len(app.run_dirs))
                    ]
                })
            elif path == "/api/run":
                self._json(200, run.api_run())
            elif path == "/api/data":
                self._json(200, run.api_data())
            elif path == "/api/config":
                self._json(200, run.api_config())
            elif path == "/api/capture":
                payload = run.api_capture(query.get("process_id"))
                if payload is None:
                    self._json(404, {"error": "unknown capture"})
                else:
                    self._json(200, payload)
            elif path == "/api/records":
                if not query.get("request"):
                    raise ViewerError(
                        "request, scheduler_process_id and occurrence "
                        "identify the logical request (correction 3: "
                        "process generation in the key)"
                    )
                payload = run.api_records(
                    query.get("scheduler_process_id"),
                    query.get("request"),
                    self._int(query, "occurrence", 0),
                )
                if payload is None:
                    self._json(404, {"error": "unknown logical request"})
                else:
                    self._json(200, payload)
            else:
                self._json(404, {"error": "not found"})

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", help="one or more run dirs")
    parser.add_argument("--port", type=int, default=8321)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    app = ViewerApp(args.run_dirs)
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(app))
    server.daemon_threads = True
    print(f"kv request explorer: http://{args.host}:{args.port}/ "
          f"({len(app.runs)} run dir(s))")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
