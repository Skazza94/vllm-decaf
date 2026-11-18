# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline loss-aware export. No transport summaries or live capture."""

import json
from pathlib import Path

from .clock import ClockGroup, ClockMapping
from .schema import VALUE_NAMES, DictionaryDecoder


def _load(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows or rows[0].get("type") != "metadata":
        raise ValueError("trace metadata missing")
    meta, decoder = rows[0], DictionaryDecoder()
    seen = {}
    events = []
    for row in rows:
        if row.get("schema") != 1:
            raise ValueError("unknown JSONL schema version")
        if row.get("type") not in ("dictionary_fragment", "event"):
            continue
        seq = row["sequence"]
        if seq in seen:
            if seen[seq] != row:
                raise ValueError("conflicting replay of a sequence")
            continue
        seen[seq] = row
        if row["type"] == "dictionary_fragment":
            decoder.feed(row)
        else:
            events.append(row)
    sequences = sorted(seen)
    missing = bool(sequences) and (
        sequences[0] != 1 or any(b != a + 1 for a, b in zip(sequences, sequences[1:]))
    )
    last_sequence = sequences[-1] if sequences else 0
    for row in rows:
        if row.get("type") == "end":
            stats = row.get("stats", {})
            if "head" in stats and stats["head"] != last_sequence:
                missing = True
            if row.get("complete") and stats.get("tail") != stats.get("head"):
                missing = True
    # Resolve only after checking every fragment, including conflicting ones.
    return meta, events, decoder, rows, missing


def export_traces(paths, output_dir: str, group: ClockGroup | None = None):
    """Use one origin per aligned group; export unaligned clocks separately.

    Modeled time is kept in a different document even when its numbers happen
    to match wall time. No request/connection relationship is inferred.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    streams = []
    captures = {}
    run_id = None
    calibrations = {}
    for path in paths:
        meta, rows, decoder, raw, missing = _load(Path(path))
        if run_id is not None and run_id != meta["run_id"]:
            raise ValueError("cannot merge different runs")
        run_id = meta["run_id"]
        if group and group.run_id is not None and group.run_id != run_id:
            raise ValueError("calibration belongs to another run")
        if group and meta["clock_id"] in group.mappings:
            calibrations[meta["clock_id"]] = group.mappings[meta["clock_id"]].to_json()
        loss = any(
            row.get("type") == "end" and not row.get("complete", False) for row in raw
        ) or not any(row.get("type") == "end" for row in raw)
        ends = [row for row in raw if row.get("type") == "end"]
        counter_loss = any(
            value > 0
            for end in ends
            for key, value in end.get("stats", {}).items()
            if key.startswith("drop_") or key == "unknown_handles"
        )
        loss = (
            loss
            or counter_loss
            or missing
            or any(
                row[key] and decoder.resolve(row[key]) is None
                for row in rows
                for key in ("request_ref", "handoff_ref", "peer_ref")
            )
        )
        buckets = set()
        for row in rows:
            modeled = row["kind"] == "model_step"
            projection = (
                group.project(meta["clock_id"], row["time_ns"])
                if group is not None and not modeled
                else None
            )
            bucket = (
                ("modeled", meta["clock_id"])
                if modeled
                else ("wall", "aligned")
                if projection is not None
                else ("wall", meta["clock_id"])
            )
            buckets.add(bucket)
            streams.append((bucket, meta, row, decoder, projection, loss))
        if not buckets:
            aligned = group is not None and (
                meta["clock_id"] == group.reference
                or meta["clock_id"] in group.mappings
            )
            buckets.add(
                ("modeled", meta["clock_id"])
                if meta.get("time_domain") == "modeled"
                else ("wall", "aligned" if aligned else meta["clock_id"])
            )
        sequences = [row["sequence"] for row in raw if "sequence" in row]
        for bucket in buckets:
            captures.setdefault(bucket, []).append(
                {
                    **meta,
                    "end_present": bool(ends),
                    "end_records": ends,
                    "counters": ends[-1].get("stats", {}) if ends else None,
                    "final_sequence": max(sequences, default=0),
                    "missing_sequences": missing,
                    "loss": loss,
                }
            )
    documents = {}
    for bucket in sorted(captures):
        subset = [item for item in streams if item[0] == bucket]
        origin = min(
            (
                item[4].time_ns if item[4] is not None else item[2]["time_ns"]
                for item in subset
            ),
            default=None,
        )
        events = []
        processes = {
            p: i + 1
            for i, p in enumerate(sorted({c["process_id"] for c in captures[bucket]}))
        }
        pending = {}
        requests = {}
        for capture in captures[bucket]:
            pid = processes[capture["process_id"]]
            if pid in requests:
                continue
            requests[pid] = {}
            events.append(
                {
                    "ph": "M",
                    "pid": pid,
                    "tid": 0,
                    "name": "process_name",
                    "args": {"name": str(capture.get("context", {}))},
                }
            )
        for _, meta, row, decoder, projected, loss in sorted(
            subset, key=lambda x: (x[1]["process_id"], x[2]["sequence"])
        ):
            pid = processes[meta["process_id"]]
            request_ref = row["request_ref"]
            lanes = requests[pid]
            if request_ref not in lanes:
                lanes[request_ref] = len(lanes) + 1
            tid = lanes[request_ref]
            timestamp = projected.time_ns if projected else row["time_ns"]
            ts = float((timestamp - origin) / 1000)
            request = decoder.resolve(request_ref)
            args = {
                "sequence": str(row["sequence"]),
                "raw_ns": str(row["time_ns"]),
                "clock_generation": meta["clock_id"],
                "request": request,
                "request_resolution": (
                    "known"
                    if request is not None
                    else "missing"
                    if request_ref
                    else "not_supplied"
                ),
                "handoff": decoder.resolve(row["handoff_ref"]),
                "peer": decoder.resolve(row["peer_ref"]),
                "transfer": str(row["transfer"]),
                "context": row["context"],
                "subject": str(row["subject"]),
                "status": row["status"],
                "unbound_handle": row["unbound_handle"],
                **{name: row[name] for name in VALUE_NAMES},
                "transport": row["transport"],
                "loss": loss,
                "uncertainty_ns": projected.uncertainty_ns if projected else None,
                "drift_known": projected.drift_known if projected else False,
                "extrapolated": projected.extrapolated if projected else False,
                "clock_model": (
                    "modeled_elapsed"
                    if bucket[0] == "modeled"
                    else "linear_estimate_not_a_bound"
                    if projected
                    else "unaligned_local"
                ),
            }
            events.append(
                {
                    "ph": "i",
                    "s": "t",
                    "pid": pid,
                    "tid": tid,
                    "ts": ts,
                    "name": row["kind"],
                    "args": args,
                }
            )
            if row["kind"] == "model_step" and row["duration0_ns"] is not None:
                events.append(
                    {
                        "ph": "X",
                        "pid": pid,
                        "tid": tid,
                        "ts": ts,
                        "dur": row["duration0_ns"] / 1000,
                        "name": "modeled device step",
                        "args": args,
                    }
                )
            handshake = (pid, "handshake", row["subject"], row["peer_ref"])
            if row["kind"] == "handshake_begin":
                pending[handshake] = ts
            elif row["kind"] == "handshake_end" and handshake in pending:
                start = pending.pop(handshake)
                if ts >= start:
                    events.append(
                        {
                            "ph": "X",
                            "pid": pid,
                            "tid": tid,
                            "ts": start,
                            "dur": ts - start,
                            "name": "handshake",
                            "args": args,
                        }
                    )
            key = pid, row["transfer"]
            if row["transfer"] and row["kind"] in ("post_begin", "post_return"):
                if row["kind"] == "post_return" and key in pending:
                    start, name = pending.pop(key)
                    if name == "post_begin":
                        events.append(
                            {
                                "ph": "X",
                                "pid": pid,
                                "tid": tid,
                                "ts": start,
                                "dur": ts - start,
                                "name": "post call",
                                "args": args,
                            }
                        )
                pending[key] = ts, row["kind"]
            elif row["kind"] == "complete_observed" and key in pending:
                start, phase = pending.pop(key)
                if phase != "post_return":
                    continue
                events.append(
                    {
                        "ph": "X",
                        "pid": pid,
                        "tid": tid,
                        "ts": start,
                        "dur": ts - start,
                        "name": "READ until observed DONE",
                        "args": args,
                    }
                )
        document = {
            "traceEvents": events,
            "displayTimeUnit": "ms",
            "metadata": {
                "clock_group": bucket,
                "origin_ns": str(origin) if origin is not None else None,
                "captures": captures[bucket],
                "calibrations": calibrations if bucket[0] == "wall" else {},
                "cross_process_flows": "not_inferred_without_qualified_adapter",
            },
        }
        name = f"{bucket[0]}-{len(documents)}.perfetto.json"
        path = output / name
        path.write_text(json.dumps(document))
        documents[bucket] = path
    return documents


def import_modeled_steps(step_log, output_path, *, run_id: str):
    """Convert an existing step log offline; never enable a live step logger."""
    import math
    from uuid import uuid4

    from .schema import dump_line

    metadata = {
        "type": "metadata",
        "schema": 1,
        "run_id": run_id,
        "process_id": str(uuid4()),
        "clock_id": str(uuid4()),
        "context": {"role": "modeled", "source": str(step_log)},
        "time_domain": "modeled",
    }
    elapsed = 0
    sequence = 0
    with Path(step_log).open() as source, Path(output_path).open("wb") as out:
        for line in source:
            record = json.loads(line)
            if "timing" in record:
                if sequence or "timing" in metadata:
                    raise ValueError("split restarted timing logs before import")
                metadata["timing"] = record["timing"]
                continue
            if "step" not in record or "cost" not in record:
                continue
            device = record["cost"].get("device_s")
            host = record["cost"].get("host_s", 0)
            if not all(
                type(v) in (float, int) and math.isfinite(v) and v >= 0
                for v in (device, host)
            ):
                raise ValueError("invalid modeled cost")
            if not sequence:
                out.write(dump_line(metadata))
            sequence += 1
            duration = round(device * 1_000_000_000)
            out.write(
                dump_line(
                    {
                        "schema": 1,
                        "type": "event",
                        "kind": "model_step",
                        "time_ns": elapsed,
                        "sequence": sequence,
                        "context": 0,
                        "request_ref": 0,
                        "handoff_ref": 0,
                        "peer_ref": 0,
                        "transfer": 0,
                        "subject": sequence,
                        "status": 0,
                        "unbound_handle": False,
                        "transport": {"kind": "unknown", "backend": None},
                        **dict.fromkeys(VALUE_NAMES),
                        "duration0_ns": duration,
                        "duration1_ns": round(host * 1_000_000_000),
                    }
                )
            )
            elapsed += duration
        if not sequence:
            out.write(dump_line(metadata))
        out.write(dump_line({"schema": 1, "type": "end", "complete": True}))


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-clock")
    parser.add_argument("--calibration", action="append", default=[])
    args = parser.parse_args()
    mappings = [
        ClockMapping.from_json(json.loads(Path(p).read_text()))
        for p in args.calibration
    ]
    reference = args.reference_clock or (
        mappings[0].start.local_generation if mappings else None
    )
    group = (
        ClockGroup(reference, {m.start.remote_generation: m for m in mappings})
        if reference
        else None
    )
    export_traces(args.traces, args.output_dir, group)


if __name__ == "__main__":
    from .clock import ClockMapping

    main()
