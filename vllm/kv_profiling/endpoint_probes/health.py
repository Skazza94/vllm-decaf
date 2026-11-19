# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared, streaming capture validation for the CLI, harness and offline joins."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .config import ProbeConfig
from .records import validate


def identity(meta):
    return {
        key: meta.get(key)
        for key in (
            "run_id",
            "process_id",
            "pid",
            "start_ticks",
            "boot_id",
            "net_namespace",
            "pid_namespace",
            "clock_id",
            "time_namespace",
            "clock_source",
            "context",
        )
    }


def inspect_trace(path, *, expected=None, probe=False):
    """Keep only metadata/footer and health, with bounded line and byte reads."""
    problems, meta, terminal = [], {}, None
    sequence = total = 0
    scopes = {}
    cpu_verified = 0
    try:
        with Path(path).open() as stream:
            for index, line in enumerate(
                iter(lambda: stream.readline((2 << 20) + 1), "")
            ):
                total += len(line.encode())
                if len(line) > ((2 << 20) if index == 0 else 65536) or total > 1 << 30:
                    raise ValueError("capture size/line limit")
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("schema") != 1:
                    raise ValueError("invalid capture schema")
                if probe:
                    validate(row)
                elif "probe_schema" in row:
                    raise ValueError("unexpected probe stream")
                if index == 0:
                    if row.get("type") != "metadata":
                        raise ValueError("metadata missing")
                    meta = row
                    if expected is not None and identity(meta) != expected:
                        problems.append("capture identity mismatch")
                    continue
                if terminal is not None:
                    raise ValueError("records after terminal")
                if row.get("type") == "end":
                    terminal = row
                    continue
                if row.get("type") not in ("event", "dictionary_fragment"):
                    raise ValueError("unexpected capture record")
                if (
                    type(row.get("sequence")) is not int
                    or row["sequence"] != sequence + 1
                ):
                    problems.append("sequence gap or replay")
                sequence = row.get("sequence", sequence)
                if probe and row["type"] == "event":
                    data = row.get("observation") or {}
                    scope = row.get("scope")
                    scopes[scope] = scopes.get(scope, 0) + 1
                    if scope == "cpu_isolation" and data.get("verified") is True:
                        cpu_verified += 1
                    if data.get("required") is True and data.get("available") is False:
                        problems.append("required source failure record")
                    if (
                        row.get("scope") == "cpu_isolation"
                        and data.get("verified") is not True
                    ):
                        problems.append("CPU isolation unverified/violated")
                    if data.get("missed_ticks", 0):
                        problems.append("observation gap record")
                if row["type"] == "event" and not isinstance(row.get("kind"), str):
                    raise ValueError("invalid event kind")
        if not meta:
            problems.append("metadata missing")
        if terminal is None:
            problems.append("terminal record missing")
        else:
            if terminal.get("complete") is not True:
                problems.append("terminal incomplete")
            stats = terminal.get("stats", {})
            if (
                type(stats.get("head")) is not int
                or type(stats.get("tail")) is not int
                or stats["head"] != sequence
                or stats["tail"] != sequence
            ):
                problems.append("terminal sequence mismatch")
            required_loss = (
                ("drop_probe_rows",)
                if probe
                else (
                    "drop_full",
                    "drop_contention",
                    "drop_ids",
                    "drop_handles",
                    "drop_identity",
                    "drop_conflict",
                    "drop_closed",
                    "unknown_handles",
                )
            )
            if any(type(stats.get(name)) is not int for name in required_loss):
                problems.append("terminal loss counters missing/invalid")
            for name, value in stats.items():
                if (
                    name.startswith("drop_")
                    or name
                    in (
                        "unknown_handles",
                        "required_source_failures",
                        "observation_gaps",
                        "isolation_failures",
                    )
                ) and (type(value) not in (int, float) or value != 0):
                    problems.append(f"loss/failure: {name}={value}")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        problems.append(str(error))
    return {
        "metadata": meta,
        "end": terminal,
        "problems": problems,
        "complete": not problems,
        "sequence": sequence,
        "scopes": scopes,
        "cpu_verified": cpu_verified,
    }


def read_json(path, limit=1 << 20):
    with Path(path).open() as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("JSON size limit")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def inventory(directory, run_id):
    """Freeze all observed process identities, including incomplete event files."""
    base = Path(directory) / run_id
    paths = sorted(
        {
            p.parent
            for glob in ("*/events.jsonl", "*/manifest.json")
            for p in base.glob(glob)
        }
    )
    if len(paths) > 256:
        raise ValueError("capture inventory limit")
    result = []
    for path in paths:
        with (path / "events.jsonl").open() as stream:
            raw = stream.readline(16385)
        if len(raw) > 16384:
            raise ValueError("inventory metadata limit")
        meta = json.loads(raw)
        if (
            meta.get("type") != "metadata"
            or meta.get("schema") != 1
            or meta.get("run_id") != run_id
            or meta.get("process_id") != path.name
        ):
            raise ValueError("inventory identity/path mismatch")
        result.append(identity(meta))
    return result


def validate_inventory(expected, run_id):
    if not isinstance(expected, list) or not 3 <= len(expected) <= 256:
        raise ValueError("expected frontend/scheduler/worker inventory required")
    roles, seen = set(), set()
    for meta in expected:
        if not isinstance(meta, dict) or not isinstance(meta.get("context"), dict):
            raise ValueError("invalid inventory metadata/context")
        context = meta["context"]
        role = context.get("role")
        if role not in ("frontend", "scheduler", "worker"):
            raise ValueError("unexpected inventory role")
        if meta.get("run_id") != run_id or meta.get("process_id") in seen:
            raise ValueError("inventory run/duplicate process mismatch")
        for key in (
            "process_id",
            "boot_id",
            "net_namespace",
            "pid_namespace",
            "clock_id",
            "time_namespace",
            "clock_source",
        ):
            if not isinstance(meta.get(key), str) or not meta[key]:
                raise ValueError(f"inventory missing {key}")
        for key in ("pid", "start_ticks"):
            if type(meta.get(key)) is not int or meta[key] <= 0:
                raise ValueError(f"inventory invalid {key}")
        if role != "frontend" and (
            not isinstance(context.get("engine_id"), str)
            or not context["engine_id"]
            or type(context.get("rank")) is not int
            or context["rank"] < 0
        ):
            raise ValueError("inventory engine/rank missing")
        seen.add(meta["process_id"])
        roles.add(role)
    if roles != {"frontend", "scheduler", "worker"}:
        raise ValueError("inventory lacks frontend, scheduler or worker")


def core_completion(directory, run_id, expected=None):
    problems, count = [], 0
    try:
        observed = inventory(directory, run_id)
        if expected is None:
            expected = observed
        validate_inventory(expected, run_id)
        validate_inventory(observed, run_id)
        want = {m["process_id"]: m for m in expected}
        actual = {m["process_id"]: m for m in observed}
        if want != actual:
            problems.append("expected/observed core inventory mismatch")
        for process_id, meta in want.items():
            count += 1
            base = Path(directory) / run_id / process_id
            health = inspect_trace(base / "events.jsonl", expected=meta)
            problems.extend(f"{process_id}: {p}" for p in health["problems"])
            try:
                manifest = read_json(base / "manifest.json")
                if identity(manifest) != meta:
                    problems.append(f"{process_id}: manifest identity mismatch")
                if (
                    manifest.get("complete") is not True
                    or manifest.get("saved") is not True
                ):
                    problems.append(f"{process_id}: manifest incomplete")
                if not health["end"] or manifest.get("stats") != health["end"].get(
                    "stats"
                ):
                    problems.append(f"{process_id}: manifest/terminal stats mismatch")
            except (OSError, ValueError, TypeError) as error:
                problems.append(f"{process_id}: {error}")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        problems.append(str(error))
    return {"captures": count, "all_closed": not problems, "problems": problems}


def endpoint_capture(directory, core_dir):
    """Validate saved observer data and exit status; never trust manifests alone."""
    directory = Path(directory)
    problems = []
    try:
        config = read_json(directory / "config.json")
        run, endpoint = config["run_id"], config["endpoint"]
        validate_inventory(config["inventory"], run)
        expected_hash = hashlib.sha256(
            json.dumps(asdict(ProbeConfig(**config["probes"])), sort_keys=True).encode()
        ).hexdigest()
        package = directory / "observer.pyz"
        package_hash = hashlib.sha256(package.read_bytes()).hexdigest()
        if package_hash != config["package_sha256"]:
            problems.append("observer package hash mismatch")
        paths = list((directory / "captures").glob(f"{run}/*/manifest.json"))
        if len(paths) != 1:
            raise ValueError("expected one observer capture")
        manifest = read_json(paths[0])
        health = inspect_trace(paths[0].with_name("events.jsonl"), probe=True)
        problems.extend(health["problems"])
        meta = health["metadata"]
        context = meta.get("context", {})
        front = next(
            item
            for item in config["inventory"]
            if item["context"]["role"] == "frontend"
        )
        if (
            context.get("role") != "endpoint_observer"
            or meta.get("clock_source") != "CLOCK_MONOTONIC"
            or not isinstance(meta.get("clock_id"), str)
            or not meta["clock_id"]
            or not isinstance(meta.get("time_namespace"), str)
            or any(
                meta.get(key) != front[key]
                for key in ("boot_id", "net_namespace", "pid_namespace")
            )
            or any(
                meta.get("owner", {}).get(key) != front[key]
                for key in ("pid", "start_ticks", "net_namespace", "pid_namespace")
            )
        ):
            problems.append("observer namespace/clock/owner identity mismatch")
        if (
            meta.get("run_id") != run
            or context.get("endpoint") != endpoint
            or context.get("endpoint_role") != config["role"]
            or meta.get("config_sha256") != expected_hash
            or meta.get("package_sha256") != package_hash
            or meta.get("inventory") != config["inventory"]
            or meta.get("affinity") != config["cpus"]
            or meta.get("owner", {}).get("pid") != config["owner_pid"]
            or meta.get("owner", {}).get("start_ticks") != config["owner_start"]
        ):
            problems.append("observer run/endpoint/config identity mismatch")
        if (
            manifest.get("run_id") != run
            or manifest.get("process_id") != meta.get("process_id")
            or manifest.get("complete") is not True
            or manifest.get("saved") is not True
            or manifest.get("end") != health["end"]
        ):
            problems.append("observer manifest/terminal mismatch")
        if (
            not health["cpu_verified"]
            or not health["scopes"].get("observer")
            or not health["end"]
            or health["end"].get("reason") != "controller_stop"
        ):
            problems.append("observer sampling/isolation/shutdown evidence missing")
        end_stats = (health["end"] or {}).get("stats", {})
        if (
            end_stats.get("controller_children_closed") is not True
            or end_stats.get("core", {}).get("all_closed") is not True
            or type(end_stats.get("samples")) is not int
            or end_stats["samples"] < 1
            or any(
                type(end_stats.get(key)) is not int or end_stats[key] != 0
                for key in (
                    "isolation_failures",
                    "required_source_failures",
                    "observation_gaps",
                )
            )
        ):
            problems.append("observer terminal closure/coverage proof missing")
        for source in config["probes"].get("required", []):
            scope = "socket" if source == "tcp" else source
            if not health["scopes"].get(scope):
                problems.append("required source observation missing: " + source)
        status = read_json(directory / "endpoint-status.json")
        container = json.loads((directory / "container.json").read_text())[0]
        state = container["State"]
        if (
            status.get("run_id") != run
            or status.get("endpoint") != endpoint
            or status.get("container_id") != container.get("Id")
            or status.get("timed_out") is not False
            or status.get("removed") is not True
            or type(status.get("exit_code")) is not int
            or status.get("exit_code") != 0
            or type(state.get("ExitCode")) is not int
            or state.get("ExitCode") != 0
            or state.get("Running") is not False
            or state.get("OOMKilled") is not False
        ):
            problems.append("observer exit/timeout/cleanup status invalid")
        closure = core_completion(core_dir, run, config["inventory"])
        problems.extend(closure["problems"])
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
    ) as error:
        problems.append(str(error))
    return {"complete": not problems, "problems": problems}
