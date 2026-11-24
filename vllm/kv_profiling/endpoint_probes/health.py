# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared, streaming capture validation for the CLI, harness and offline joins."""

import hashlib
import json
from pathlib import Path

from .code import code_digest
from .config import ProbeConfig, settings_dict
from .exit_record import CAPTURE_FORMAT, exit_problems
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
    ebpf_losses, ebpf_terminal = {}, {}  # recorded eBPF loss counters and counts
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
                if not probe and row.get("type") == "threads":
                    continue  # the core writer's thread inventory, unsequenced
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
                    if scope == "ebpf" and data.get("type") == "loss":
                        if isinstance(data.get("value"), int):
                            key = (data.get("scope"), data.get("counter"))
                            ebpf_losses[key] = max(
                                ebpf_losses.get(key, 0), data["value"]
                            )
                    elif scope == "ebpf" and data.get("type") == "end":
                        ebpf_terminal = data
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
    from .ebpf import event_loss

    return {
        "metadata": meta,
        "end": terminal,
        "problems": problems,
        "complete": not problems,
        "sequence": sequence,
        # derived from the recorded counters themselves, never the footer alone
        "ebpf_event_loss": event_loss(ebpf_losses, ebpf_terminal),
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


ROLES = ("frontend", "scheduler", "worker")


def recorder_counts(entries):
    """Recorders per role of an inventory."""
    return {role: sum(e["context"]["role"] == role for e in entries) for role in ROLES}


# The KV roles a side's recorders may carry.
SIDE_KV_ROLES = {
    "prefill": ("kv_producer", "kv_both"),
    "decode": ("kv_consumer", "kv_both"),
}


def side_problems(entries, endpoint_role):
    """Recorders that are not this endpoint side's: each must carry a KV role
    of its side (``kv_both`` is either)."""
    allowed = SIDE_KV_ROLES.get(endpoint_role)
    if allowed is None:
        return [f"unknown endpoint role {endpoint_role!r}"]
    wrong = sorted(
        {
            str(e["context"].get("kv_role"))
            for e in entries
            if e["context"].get("kv_role") not in allowed
        }  # fmt: skip
    )
    return (
        [f"the recorders' KV roles {wrong} are not {endpoint_role}'s"] if wrong else []
    )


# A launcher never supplies the identities the observer discovers.
RESERVED_IDENTITY_KEYS = ("inventory", "owner_pid", "owner_start")


def expected_problems(expected):
    """Why ``expected`` is not a count per role: exactly the roles, each an
    integer (not a bool or a float) of at least zero, at least one frontend."""
    if not isinstance(expected, dict) or set(expected) != set(ROLES):
        return [f"expected is a count per role {ROLES}"]
    if any(type(expected[role]) is not int or expected[role] < 0 for role in ROLES):
        return ["expected counts are non-negative integers"]
    if expected["frontend"] < 1:
        return ["expected names at least one frontend"]
    return []


def launch_config_problems(values):
    """The one launch form, for the launcher's config at launch and at check:
    the expected recorders, no supplied identity (by key, whatever its value)."""
    if not isinstance(values, dict):
        return ["the launch config is not an object"]
    problems = [
        f"the launch config supplies {key!r}; the observer discovers it"
        for key in RESERVED_IDENTITY_KEYS
        if key in values
    ]
    return problems + expected_problems(values.get("expected"))


def owner_problems(owner, frozen, evidence):
    """Why the recorded owner is not the one discovery showed: with one
    frontend, that frontend; with several, their one recorded parent, each
    frontend's evidence matching its inventory identity."""
    fronts = [e for e in frozen if e["context"]["role"] == "frontend"]
    if (
        not isinstance(owner, dict)
        or type(owner.get("pid")) is not int
        or type(owner.get("start_ticks")) is not int
    ):
        return ["owner PID and start ticks missing"]
    if not isinstance(evidence, dict) or evidence.get("supplied"):
        return ["owner evidence missing (identities supplied, not discovered)"]
    if len(fronts) == 1:
        target = (fronts[0]["pid"], fronts[0]["start_ticks"])
        if (owner["pid"], owner["start_ticks"]) != target:
            return ["the owner is not the single frontend"]
        return []
    seen = evidence.get("frontends")
    parent = evidence.get("parent")
    if not isinstance(seen, list) or not isinstance(parent, dict):
        return ["owner evidence malformed"]
    if sorted((e.get("pid"), e.get("start_ticks")) for e in seen) != sorted(
        (f["pid"], f["start_ticks"]) for f in fronts
    ):
        return ["owner evidence is not the frozen frontends'"]
    if {e.get("parent_pid") for e in seen} != {owner["pid"]}:
        return ["the owner is not the frontends' recorded parent"]
    if (parent.get("pid"), parent.get("start_ticks")) != (
        owner["pid"],
        owner["start_ticks"],
    ):
        return ["the owner's start ticks are not the recorded parent's"]
    return []


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
            "clock_source",
        ):
            if not isinstance(meta.get(key), str) or not meta[key]:
                raise ValueError(f"inventory missing {key}")
        # None: the kernel has no time namespaces (one host monotonic clock).
        time_namespace = meta.get("time_namespace", "")
        if time_namespace is not None and (
            not isinstance(time_namespace, str) or not time_namespace
        ):
            raise ValueError("inventory missing time_namespace")
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
    ebpf = {"degraded": False, "event_loss": {}}
    try:
        config = read_json(directory / "config.json")
        run, endpoint = config["run_id"], config["endpoint"]
        expected_hash = hashlib.sha256(
            json.dumps(
                settings_dict(ProbeConfig(**config["probes"])), sort_keys=True
            ).encode()
        ).hexdigest()
        # a launcher that ran a packaged zipapp names its hash; an installed
        # package has none, and the executed code's digest is checked below
        package_hash = None
        if config.get("package_sha256") is not None:
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
        if meta.get("capture_format") != CAPTURE_FORMAT:
            raise ValueError(
                f"capture format {meta.get('capture_format')!r} is an earlier "
                f"observer's, not {CAPTURE_FORMAT}: rerun the capture"
            )
        context = meta.get("context", {})
        # the identities the observer found and froze before it was ready,
        # against the recorders its launcher expected
        frozen = meta.get("inventory")
        validate_inventory(frozen, run)
        # the launch config as the CLI accepts it, and the frozen counts with
        # the same types, before any comparison
        for problem in launch_config_problems(config):
            problems.append("observer launch config: " + problem)
        for problem in expected_problems(meta.get("expected")):
            problems.append("observer frozen counts: " + problem)
        if meta.get("expected") != config.get("expected") or recorder_counts(
            frozen
        ) != config.get("expected"):
            problems.append(
                "observer inventory is not the launcher's expected recorders"
            )
        if config.get("role") not in ("prefill", "decode") or side_problems(
            frozen, config["role"]
        ):
            problems.append("observer recorders are not this endpoint side's")
        if owner_problems(meta.get("owner"), frozen, meta.get("owner_evidence")):
            problems.append(
                "observer owner not shown by its frozen evidence: "
                + "; ".join(
                    owner_problems(
                        meta.get("owner"), frozen, meta.get("owner_evidence")
                    )
                )  # fmt: skip
            )
        if meta.get("code_sha256") != code_digest():
            problems.append(
                "observer code differs from this checker's: check a capture with "
                "the observer code that made it"
            )
        # Every frontend (one API server, or one a DP rank) shares the
        # observer's boot and namespaces. The owner is the process the
        # controller configured: the frontend itself when there is one, else
        # the API servers' common parent (the serving process).
        fronts = [item for item in frozen if item["context"]["role"] == "frontend"]
        owner = meta.get("owner", {})
        if (
            not fronts
            or context.get("role") != "endpoint_observer"
            or meta.get("clock_source") != "CLOCK_MONOTONIC"
            or not isinstance(meta.get("clock_id"), str)
            or not meta["clock_id"]
            or not (
                meta.get("time_namespace") is None
                or isinstance(meta.get("time_namespace"), str)
            )
            or any(
                meta.get(key) != front[key]
                for front in fronts
                for key in ("boot_id", "net_namespace", "pid_namespace")
            )
            or (
                len(fronts) == 1
                and (owner.get("pid"), owner.get("start_ticks"))
                != (fronts[0]["pid"], fronts[0]["start_ticks"])
            )
            or any(
                owner.get(key) != fronts[0][key]
                for key in ("net_namespace", "pid_namespace")
            )
        ):
            problems.append("observer namespace/clock/owner identity mismatch")
        if (
            meta.get("run_id") != run
            or context.get("endpoint") != endpoint
            or context.get("endpoint_role") != config["role"]
            or meta.get("config_sha256") != expected_hash
            or meta.get("package_sha256") != package_hash
            or meta.get("affinity") != config["cpus"]
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
        # known eBPF event loss, from the recorded counters: the capture
        # stands, marked degraded; a footer that says otherwise is a problem
        loss = health.get("ebpf_event_loss") or {}
        ebpf = {"degraded": bool(loss), "event_loss": loss}
        if "ebpf_event_loss" in end_stats and (
            end_stats.get("ebpf_event_loss") != loss
            or end_stats.get("ebpf_degraded") is not bool(loss)
        ):
            problems.append("eBPF loss footer disagrees with the recorded counters")
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
        # How the observer ended, from whatever launched and reaped it: never
        # a container runtime's own records (any deployment writes this one).
        problems.extend(exit_problems(directory, run, endpoint, meta.get("process_id")))
        closure = core_completion(core_dir, run, frozen)
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
    return {"complete": not problems, "problems": problems, "ebpf": ebpf}
