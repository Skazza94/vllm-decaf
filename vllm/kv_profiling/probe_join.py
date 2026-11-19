# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline, qualified local flow candidates; never apportion endpoint counters."""

from pathlib import Path

from .endpoint_probes.health import inspect_trace
from .perfetto import _clock_aliases, _domain, _load


def candidates(core_paths, probe_paths, group=None):
    """Associate only validated local engine/rank owners and projected intervals.

    A unique temporal candidate is not causal attribution. Unaligned clocks,
    missing ownership and missing completion remain explicit unmatched results.
    QP observations have no proven incarnation and are not READ associations.
    """
    core_paths, probe_paths = list(core_paths), list(probe_paths)
    core = [_load(Path(path)) for path in core_paths]
    probes = [_load(Path(path)) for path in probe_paths]
    metas = [item[0] for item in core + probes]
    runs = {meta["run_id"] for meta in metas}
    if len(runs) != 1 or (group and group.run_id not in (None, next(iter(runs)))):
        raise ValueError("candidate association requires a single run")
    aliases = _clock_aliases(metas, group)
    core_health = {
        item[0]["process_id"]: inspect_trace(path)
        for path, item in zip(core_paths, core)
    }
    probe_health = {
        item[0]["process_id"]: inspect_trace(path, probe=True)
        for path, item in zip(probe_paths, probes)
    }
    results = []
    for meta, rows, _, _, missing in core:
        context = meta.get("context", {})
        if context.get("role") != "worker":
            continue
        engine, rank = context.get("engine_id"), context.get("rank")
        if not isinstance(engine, str) or type(rank) is not int:
            raise ValueError("READ engine/rank identity missing")
        starts = {}
        for row in rows:
            transfer = row["transfer"]
            if row["kind"] == "post_return" and transfer:
                starts[transfer] = row["time_ns"]
            if row["kind"] != "complete_observed" or transfer not in starts:
                continue
            start = starts.pop(transfer)
            found, reasons = {}, set()
            health_reasons = {
                "core: " + problem
                for problem in core_health[meta["process_id"]]["problems"]
            }
            if missing:
                reasons.add("core_sequence_loss")
            for observed, samples, _, _, _ in probes:
                if "probe_schema" not in observed:
                    raise ValueError("expected endpoint probe stream")
                expected_role = {"kv_producer": "prefill", "kv_consumer": "decode"}.get(
                    context.get("kv_role")
                )
                if observed["context"].get("endpoint_role") != expected_role:
                    continue
                if not observed["context"].get("endpoint"):
                    raise ValueError("probe endpoint identity missing")
                health_reasons.update(
                    "probe: " + problem
                    for problem in probe_health[observed["process_id"]]["problems"]
                )
                same = _domain(meta) == _domain(observed) and all(_domain(meta))
                for sample in samples:
                    if sample.get("scope") != "socket":
                        continue
                    flow = sample["transport"]["tcp"]
                    owners = flow.get("owners", [])
                    if not any(
                        owner.get("engine_id") == engine
                        and owner.get("rank") == rank
                        and owner.get("pid") == meta.get("pid")
                        and owner.get("start_ticks") == meta.get("start_ticks")
                        and owner.get("net_namespace") == meta.get("net_namespace")
                        and owner.get("pid_namespace") == meta.get("pid_namespace")
                        for owner in owners
                    ):
                        reasons.add("ownership_unavailable_or_other_engine")
                        continue
                    if (
                        not meta.get("boot_id")
                        or observed.get("boot_id") != meta.get("boot_id")
                        or observed.get("net_namespace") != meta.get("net_namespace")
                    ):
                        reasons.add("host_or_namespace_mismatch")
                        continue
                    left, right = start, row["time_ns"]
                    begin, end = sample["sample_begin_ns"], sample["sample_end_ns"]
                    uncertainty = 0
                    if not same:
                        projected = [
                            group.project(aliases.get(m["clock_id"], m["clock_id"]), t)
                            if group
                            else None
                            for m, t in (
                                (meta, left),
                                (meta, right),
                                (observed, begin),
                                (observed, end),
                            )
                        ]
                        if any(item is None for item in projected):
                            reasons.add("unaligned_clocks")
                            continue
                        left, right, begin, end = [p.time_ns for p in projected]
                        uncertainty = max(
                            p.uncertainty_ns for p in projected[:2]
                        ) + max(p.uncertainty_ns for p in projected[2:])
                    if end + uncertainty < left or begin - uncertainty > right:
                        continue
                    found[sample["entity"]] = {
                        "entity": sample["entity"],
                        "endpoint": observed["context"]["endpoint"],
                        "sample_interval_ns": [
                            sample["sample_begin_ns"],
                            sample["sample_end_ns"],
                        ],
                        "clock_uncertainty_ns": uncertainty,
                        "association": "temporal_candidate_not_causal",
                    }
            results.append(
                {
                    "run_id": meta["run_id"],
                    "process_id": meta["process_id"],
                    "engine_id": engine,
                    "rank": rank,
                    "transfer": transfer,
                    "association": "partial_coverage"
                    if health_reasons
                    else "unmatched"
                    if not found
                    else "unique_temporal_candidate"
                    if len(found) == 1
                    else "ambiguous",
                    "candidates": list(found.values()),
                    "coverage_complete": not health_reasons,
                    "coverage_scope": "provided_JSONL_streams",
                    "reasons": sorted(reasons | health_reasons),
                    "exact": False,
                }
            )
        for transfer in starts:
            results.append(
                {
                    "process_id": meta["process_id"],
                    "transfer": transfer,
                    "association": "unmatched",
                    "exact": False,
                    "reasons": ["missing_completion"],
                    "candidates": [],
                }
            )
    return results
