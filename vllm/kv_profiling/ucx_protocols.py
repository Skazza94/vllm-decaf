# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline parser of UCX's protocol selection tables (``UCX_PROTO_INFO``).

With ``UCX_PROTO_INFO=y`` UCX logs, the first time it selects protocols for an
endpoint configuration and an operation, a table of the message-size ranges,
the protocol chosen for each and the lanes it uses (``UCX_PROTO_INFO=used``
logs the same tables with per-range selection counts when a worker is
destroyed). This module turns those tables, from a server log, into records:
which transport, device and share of the bytes each lane carries, per process
and per UCX endpoint configuration (``cfg#N``, numbered per UCP worker;
endpoints to peers reached the same way share one).

UCX writes every line of one table under the same ``[time] [host:pid:thread]``
prefix in a single write, so tables are grouped by that prefix even when
several processes share a log. A table that does not parse is counted with its
reason, never dropped silently. Standard library only, so the harness can run
it on a host without vLLM.
"""

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = 1
_PREFIX = re.compile(
    r"\[(?P<time>\d+\.\d+)\] \[(?P<host>[^:\]]+):\s*(?P<pid>\d+)\s*"
    r":(?P<thread>[^\]]+)\]"
    r"\s+(?P<rest>[|+].*)$"
)
_TITLE = re.compile(r"^(?:(?P<context>.*) )?(?P<locality>self|intra-node|inter-node)"
                    r" cfg#(?P<cfg>\d+)$")  # fmt: skip
_UNITS = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40,
          "P": 1 << 50, "E": 1 << 60}  # fmt: skip
_SIZE = re.compile(r"^(\d+)([KMGTPE]?)$")
_RESOURCE = re.compile(r"^(?P<transport>[^/\s]+)/(?P<device>[^/\s]+)"
                       r"(?:/path(?P<path>\d+))?$")  # fmt: skip
_SHARE = re.compile(r"(?:(?P<share>\d+)% on )?(?P<lane>[^,\s]+)")
MAX_ERRORS = 32


class TableError(ValueError):
    pass


class TrailingLines(Exception):
    """A complete table followed by lines of no recognizable table."""

    def __init__(self, consumed: int):
        super().__init__(consumed)
        self.consumed = consumed


def _size(text: str) -> int | None:
    """A UCX memory-units value (``8K``, ``inf``) in bytes; None for inf."""
    if text == "inf":
        return None
    match = _SIZE.match(text)
    if match is None:
        raise TableError(f"unrecognized size {text!r}")
    return int(match.group(1)) * _UNITS[match.group(2)]


def _range(text: str) -> tuple[int, int | None]:
    first, separator, last = text.partition("..")
    start = _size(first)
    if start is None:
        raise TableError(f"range starting at infinity: {text!r}")
    return start, _size(last) if separator else start


def lanes(config: str) -> list[dict]:
    """The lanes a protocol's configuration names, with the share of the bytes
    each carries (percent; 100 for a single lane, None where UCX gives none).

    UCX prints ``tcp/eth0``, ``50% on rc/dev0:1 and 50% on rc/dev1:1``, or, for
    lanes of one device on several paths, ``rc/dev0:1 50% on path0 and 50% on
    path1``. A configuration without a ``transport/device`` lane gives []."""
    words = config.split()
    base = None
    if len(words) > 1 and _RESOURCE.match(words[0]) and "% on " in config:
        base, config = words[0], config[len(words[0]) :].strip()
    found = []
    for part in re.split(r", | and ", config):
        match = _SHARE.fullmatch(part.strip())
        if match is None:
            continue
        lane, share = match.group("lane"), match.group("share")
        if base is not None and re.fullmatch(r"path\d+", lane):
            lane = f"{base}/{lane}"
        resource = _RESOURCE.match(lane)
        if resource is None:
            continue
        path = resource.group("path")
        found.append(
            {
                "transport": resource.group("transport"),
                "device": resource.group("device"),
                "path": int(path) if path is not None else None,
                "share_percent": int(share) if share is not None else None,
            }
        )
    if len(found) == 1 and found[0]["share_percent"] is None:
        found[0]["share_percent"] = 100
    return found


def _cells(line: str) -> list[str]:
    if not (line.startswith("|") and line.endswith("|")):
        raise TableError("row outside the table frame")
    return [cell.strip() for cell in line[1:-1].split("|")]


def parse_table(lines: list[str]) -> dict:
    """One rendered table (its lines without the log prefix)."""
    rows = [line.strip() for line in lines]
    if len(rows) < 6 or not all(rows[i].startswith("+") for i in (0, 3, 5)):
        raise TableError("not a protocol selection table")
    title = _cells(rows[1])
    operation = _cells(rows[2])
    if len(title) != 1 or len(operation) != 1:
        raise TableError("title rows are not single cells")
    match = _TITLE.match(title[0])
    if match is None:
        raise TableError(f"unrecognized endpoint configuration {title[0]!r}")
    header = _cells(rows[4])
    with_counts = header == ["Count", "Range", "Description", "Config"]
    if not with_counts and header != ["Range", "Description", "Config"]:
        raise TableError(f"unrecognized header {header!r}")
    ranges = []
    for index, row in enumerate(rows[6:], 6):
        if row.startswith("+"):
            if rows[index + 1 :]:
                # Lines after the closing border that open no frame (e.g. a
                # table cut off at the end of the log) are not this table's.
                raise TrailingLines(index + 1)
            break
        cells = _cells(row)
        if len(cells) != len(header):
            raise TableError("body row does not match the header")
        count = cells.pop(0) if with_counts else None
        if count is not None:
            if not count.isdigit():
                raise TableError(f"unrecognized count {count!r}")
            count = int(count)
        start, end = _range(cells[0])
        description = cells[1]
        estimated = description.startswith("(?) ")
        ranges.append(
            {
                "start": start,
                "end": end,
                "description": description[4:] if estimated else description,
                "estimated": estimated,
                "config": cells[2],
                "lanes": lanes(cells[2]),
                "selections": count,
            }
        )
    else:
        raise TableError("table not closed")
    if not ranges:
        raise TableError("table without ranges")
    api = re.search(r"\bucp_\w+\*?", operation[0])
    return {
        "context": match.group("context"),
        "locality": match.group("locality"),
        "ep_config": int(match.group("cfg")),
        "operation": operation[0],
        "api": api.group(0) if api else None,
        "selection_counts": with_counts,
        "ranges": ranges,
    }


def _single_cell(line: str) -> bool:
    line = line.strip()
    return line.startswith("|") and line.endswith("|") and line.count("|") == 2


def _opens_table(block: list[str], i: int) -> bool:
    """A table's frame, whatever its title says: a border, the two full-width
    title rows (the only single-cell rows a table has), then a border."""
    return (
        block[i].startswith("+")
        and _single_cell(block[i + 1])
        and _single_cell(block[i + 2])
        and block[i + 3].startswith("+")
    )


def parse_lines(lines) -> dict:
    """Every protocol selection table in a log's lines."""
    blocks: dict[tuple, list] = {}
    order = []
    for line in lines:
        match = _PREFIX.search(line.rstrip("\n"))
        if match is None:
            continue
        key = tuple(match.group(k) for k in ("time", "host", "pid", "thread"))
        if key not in blocks:
            blocks[key] = []
            order.append(key)
        blocks[key].append(match.group("rest"))
    tables, errors = [], []
    for key in order:
        time_s, host, pid, thread = key
        # One prefix may (rarely) hold several tables written in the same
        # microsecond: each opening border followed by an endpoint title starts
        # a table, so a damaged table never swallows the next one.
        block = blocks[key]
        starts = [i for i in range(len(block) - 3) if _opens_table(block, i)]
        if starts and starts[0] > 0:
            errors.append({"time_s": float(time_s), "pid": int(pid),
                           "reason": "table lines outside a table frame"})  # fmt: skip
        for index, first in enumerate(starts or [0]):
            last = starts[index + 1] if index + 1 < len(starts) else len(block)
            segment = block[first:last]
            try:
                try:
                    table = parse_table(segment)
                except TrailingLines as trailing:
                    table = parse_table(segment[: trailing.consumed])
                    errors.append(
                        {"time_s": float(time_s), "pid": int(pid),
                         "reason": "lines after a table's frame"}
                    )  # fmt: skip
            except (TableError, ValueError, IndexError) as error:
                errors.append({"time_s": float(time_s), "pid": int(pid),
                               "reason": str(error)})  # fmt: skip
                continue
            tables.append(
                {
                    "time_s": float(time_s),
                    "host": host,
                    "pid": int(pid),
                    "ucs_thread": thread.strip(),
                    **table,
                }
            )
    return {
        "tables": tables,
        "unparsed": len(errors),
        "unparsed_examples": errors[:MAX_ERRORS],
    }


def _recorders(profiles: Path, run_id: str | None) -> tuple[dict, list]:
    """The recorder processes under a profile directory (a run's directory or a
    directory of runs), by PID, restricted to ``run_id`` when given, and the
    metadata entries skipped as unreadable or malformed. The log and the
    profiles must come from the same node, so PIDs share a namespace."""
    found: dict[int, list[dict]] = {}
    skipped = []
    for path in sorted(profiles.rglob("events.jsonl")):
        try:
            with path.open() as stream:
                meta = json.loads(stream.readline(1 << 20))
        except (OSError, ValueError) as error:
            skipped.append({"path": str(path), "reason": type(error).__name__})
            continue
        if not isinstance(meta, dict):
            skipped.append({"path": str(path), "reason": "metadata is not an object"})
            continue
        if run_id is not None and meta.get("run_id") != run_id:
            continue
        context = meta.get("context")
        if type(meta.get("pid")) is not int or not isinstance(context, dict):
            skipped.append(
                {"path": str(path), "reason": "no integer pid or object context"}
            )
            continue
        found.setdefault(meta["pid"], []).append(
            {
                "run_id": meta.get("run_id"),
                "process_id": meta.get("process_id"),
                "role": context.get("role"),
                "rank": context.get("rank"),
                "engine_id": context.get("engine_id"),
            }
        )
    return found, skipped


def summarize(logs, profiles=None, run_id=None) -> dict:
    """The tables of each log, grouped by the fields UCX prints: process, UCX
    context, locality and endpoint configuration. UCX numbers ``cfg#N`` per
    UCP worker and does not print the worker, so a grouping may hold several
    workers' tables and worker identity is never resolved; a grouping where a
    collision is visible (the same label with two localities, or one operation
    selected twice) is marked ``worker_collision_detected``. With
    ``profiles``, each process is joined by PID to the recorder it ran (in
    ``run_id`` when given); a PID that matches more than one recorder stays
    unjoined, with the candidates counted, and malformed metadata entries are
    skipped and listed."""
    tables, unparsed, examples = [], 0, []
    for log in logs:
        with open(log, errors="replace") as stream:
            result = parse_lines(stream)
        for table in result["tables"]:
            table["log"] = str(log)
        tables += result["tables"]
        unparsed += result["unparsed"]
        examples += result["unparsed_examples"]
    recorders, skipped = _recorders(Path(profiles), run_id) if profiles else ({}, [])
    localities: dict[tuple, set] = {}
    for table in tables:
        label = (table["host"], table["pid"], table["context"], table["ep_config"])
        localities.setdefault(label, set()).add(table["locality"])
    endpoints: dict[tuple, dict] = {}
    for table in tables:
        label = (table["host"], table["pid"], table["context"], table["ep_config"])
        key = (*label, table["locality"])
        candidates = recorders.get(table["pid"], [])
        entry = endpoints.setdefault(
            key,
            {
                "host": table["host"],
                "pid": table["pid"],
                "recorder": candidates[0] if len(candidates) == 1 else None,
                "recorder_candidates": len(candidates),
                "context": table["context"],
                "ep_config": table["ep_config"],
                "locality": table["locality"],
                "worker_collision_detected": len(localities[label]) > 1,
                "operations": [],
            },
        )
        if any(o["operation"] == table["operation"] for o in entry["operations"]):
            entry["worker_collision_detected"] = True  # selected once per worker
        entry["operations"].append(
            {
                k: table[k]
                for k in (
                    "time_s",
                    "ucs_thread",
                    "locality",
                    "operation",
                    "api",
                    "selection_counts",
                    "ranges",
                )
            }  # fmt: skip
        )
    return {
        "schema": SCHEMA,
        "source": "UCX_PROTO_INFO",
        "run_id": run_id,
        "worker_identity": "unresolved: UCX does not print the UCP worker",
        "endpoint_configurations": list(endpoints.values()),
        "recorders_skipped": len(skipped),
        "recorders_skipped_examples": skipped[:MAX_ERRORS],
        "tables": len(tables),
        "unparsed": unparsed,
        "unparsed_examples": examples[:MAX_ERRORS],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", help="server logs with UCX_PROTO_INFO")
    parser.add_argument("--profiles", help="the node's profiler output directory")
    parser.add_argument("--run-id", help="join only recorders of this profiler run")
    parser.add_argument("--output", help="JSON file (default: standard output)")
    args = parser.parse_args(argv)
    result = summarize(args.logs, args.profiles, args.run_id)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
