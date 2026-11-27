# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline parser of the LIBFABRIC providers NIXL's agents opened, from logs.

With ``NIXL_LOG_LEVEL=INFO`` NIXL's rail manager logs, through abseil, each rail
it creates and the provider its topology chose, then the agent's
self-connection, which names the NIXL agent. With ``FI_LOG_LEVEL=info``
(``FI_LOG_PROV=core FI_LOG_SUBSYS=core`` keeps only the core's lines) libfabric
logs each fabric it opened and the provider that opened it. Both are needed:
the manager's name is NIXL's choice, each rail's open is libfabric's.

One sequence per agent, on the thread that created it (abseil prints the
thread ID; libfabric the process ID, the same for a process's main thread):
NIXL's version line when the library loads, then per rail libfabric's
``Opened fabric`` and ``Using <provider> provider`` and NIXL's
``fabric_attr->name``, then ``Created N rails using provider=P``, then
``Created self-connection for agent: A on N rails``. Each log input is parsed
on its own. A sequence whose rail counts, fabric names or order do not line up
(two processes sharing a thread ID in a merged log, a restart midway) is kept
as a diagnostic with its reason, never paired by guess; a failed rail creation
ends its sequence. Standard library only.
"""

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = 1
MAX_DIAGNOSTICS = 64
_ABSEIL = re.compile(
    r"(?:^|\s)(?P<severity>[IWEF])\d{4} (?P<time>\d{2}:\d{2}:\d{2}\.\d+)\s+"
    r"(?P<tid>\d+) (?P<file>[\w.+-]+):\d+\] (?P<message>.*)$"
)
_LIBFABRIC = re.compile(
    r"libfabric:(?P<pid>\d+):\d+::(?P<prov>[^:]*):(?P<subsys>[^:]*):"
    r"(?P<function>[^:]*)\(\):\d+<(?P<level>\w+)> (?P<message>.*)$"
)
_OPENED = re.compile(r"^Opened fabric: (?P<fabric>.+)$")
_USING = re.compile(
    r"^Using (?P<provider>\S+) provider (?P<version>\S+), path:(?P<path>.*)$"
)
_RAIL = re.compile(r"^fabric_attr->name (?P<fabric>.+)$")
_MANAGER = re.compile(
    r"^Created (?P<rails>\d+) rails using provider=(?P<provider>\S+)$"
)
_AGENT = re.compile(
    r"^Created self-connection for agent: (?P<agent>\S+) on (?P<rails>\d+) rails$"
)


_FILTER_READ = re.compile(r"^read string var provider=(?P<value>.*)$")
_FILTER_UNSET = "variable provider=<not set>"
# LIBFABRIC's own startup failures, at any log level; other errors (another
# backend's, a transfer's after startup) are not rail failures
_STARTUP_FAILURES = ("Failed to create rails", "Topology discovery failed",
                     "backend creation failed for 'LIBFABRIC'")  # fmt: skip


def _startup_failure(source, message, life) -> bool:
    if any(text in message for text in _STARTUP_FAILURES):
        return True
    if source.startswith("backend_plugin") and "libfabric" in message.lower():
        return True
    # a rail's own error while its startup sequence is in progress
    return source.startswith("libfabric_rail") and life is not None and life.active()


class _Lifetime:
    """One process's rail sequence on one thread ID, from NIXL's load."""

    def __init__(self, line):
        self.start = line
        self.fi_logging = False
        self.effective = "unknown"  # libfabric's own FI_PROVIDER, when logged
        self.reset()

    def reset(self):
        self.pending_open = None
        self.unrailed = []  # opens with their provider, awaiting NIXL's rail
        self.rails = []
        self.opens = []
        self.manager = None
        self.problem = None

    def active(self):
        return bool(self.pending_open or self.unrailed or self.rails or self.manager)

    def fail(self, problem):
        self.problem = self.problem or problem


def parse_lines(lines, log="log"):
    """Every agent's rail sequence in one log input. A sequence must follow
    the order: per rail libfabric's open, its provider and NIXL's rail (no
    open when libfabric's logging is off for the process); then the rail
    manager; then the self-connection. A thread ID with more than one
    process lifetime in the input, or none on record, pairs nothing."""
    lives: dict[int, _Lifetime] = {}
    lifetimes: dict[int, int] = {}
    records, diagnostics = [], []

    def diagnose(tid, line, reason, life=None):
        diagnostics.append({
            "log": log, "tid": tid, "line": line, "reason": reason,
            **({"opened": [o["provider"] for o in life.opens + life.unrailed],
                "rails": len(life.rails),
                "manager_provider": (life.manager or {}).get("provider")}
               if life is not None else {}),
        })  # fmt: skip

    for number, raw in enumerate(lines, 1):
        text = raw.rstrip("\n")
        fabric = _LIBFABRIC.search(text)
        if fabric is not None:
            tid = int(fabric.group("pid"))
            life = lives.get(tid)
            if life is None:
                continue  # before NIXL loaded on this thread: no lifetime
            function, message = fabric.group("function"), fabric.group("message")
            if (fabric.group("prov"), fabric.group("subsys"),
                    fabric.group("level")) != ("core", "core", "info"):
                continue  # only the core's own INFO records are evidence
            life.fi_logging = True
            if function == "fi_param_get_":
                if (read := _FILTER_READ.match(message)) is not None:
                    life.effective = read.group("value")
                elif message == _FILTER_UNSET:
                    life.effective = None
                continue
            if function != "fi_fabric_":
                continue
            if life.manager is not None:
                life.fail("a libfabric open after the rail manager")
            if (opened := _OPENED.match(message)) is not None:
                if life.pending_open is not None:
                    life.fail("two fabrics opened without a provider line")
                if life.unrailed:
                    life.fail("a second open before NIXL's rail took the first")
                life.pending_open = opened.group("fabric")
            elif (using := _USING.match(message)) is not None:
                if life.pending_open is None:
                    life.fail("a provider line without an opened fabric")
                else:
                    life.unrailed.append({
                        "fabric": life.pending_open,
                        "provider": using.group("provider"),
                        "version": using.group("version"),
                    })  # fmt: skip
                    life.pending_open = None
            continue
        native = _ABSEIL.search(text)
        if native is None:
            continue
        tid = int(native.group("tid"))
        source, message = native.group("file"), native.group("message")
        life = lives.get(tid)
        if source == "nixl_log.cpp" and message.startswith("NIXL version:"):
            if life is not None and life.active():
                diagnose(tid, number, "a new process began before the agent's "
                         "self-connection", life)  # fmt: skip
            lives[tid] = _Lifetime(number)
            lifetimes[tid] = lifetimes.get(tid, 0) + 1
            continue
        if native.group("severity") in ("E", "F") and _startup_failure(
            source, message, life
        ):
            # reported whatever the INFO logging: a backend that never came up
            diagnose(tid, number, f"rail creation failed: {message}", life)
            if life is not None:
                life.reset()
            continue
        if life is None:
            if (source, message.split(" ")[0]) in (
                ("libfabric_rail.cpp", "fabric_attr->name"),
                ("libfabric_rail_manager.cpp", "Created"),
                ("libfabric_backend.cpp", "Created"),
            ):
                diagnose(tid, number, "rail lines without a process start on record")
            continue
        if source == "libfabric_rail.cpp" and (rail := _RAIL.match(message)):
            if life.manager is not None:
                life.fail("a rail after the rail manager")
            if life.fi_logging:
                if life.pending_open is not None or not life.unrailed:
                    life.fail("a rail without its libfabric open and provider")
                elif life.unrailed[0]["fabric"] != rail.group("fabric"):
                    life.fail("the opened fabric is not the rail's")
                else:
                    life.opens.append(life.unrailed.pop(0))
            life.rails.append(rail.group("fabric"))
        elif source == "libfabric_rail_manager.cpp" and (
            manager := _MANAGER.match(message)
        ):
            if life.manager is not None:
                life.fail("two rail managers before a self-connection")
            if life.pending_open is not None or life.unrailed:
                life.fail("a libfabric open with no NIXL rail")
            life.manager = {"rails": int(manager.group("rails")),
                            "provider": manager.group("provider")}  # fmt: skip
        elif source == "libfabric_backend.cpp" and (agent := _AGENT.match(message)):
            rails = int(agent.group("rails"))
            problem = life.problem
            if problem is None and life.manager is None:
                problem = "a self-connection without the rail manager's line"
            elif problem is None and not (
                life.manager["rails"] == rails == len(life.rails)
            ):
                problem = (f"rail counts disagree: manager {life.manager['rails']}, "
                           f"agent {rails}, rails {len(life.rails)}")  # fmt: skip
            elif problem is None and life.fi_logging and len(life.opens) != rails:
                problem = "not every rail has its libfabric open"
            if problem is not None:
                diagnose(tid, number, problem, life)
            else:
                records.append({
                    "log": log, "tid": tid, "agent": agent.group("agent"),
                    "time": native.group("time"),
                    "lines": [life.start, number],
                    "manager_provider": life.manager["provider"],
                    "rails": rails, "rail_fabrics": list(life.rails),
                    "opened": list(life.opens),
                    "libfabric_logging": life.fi_logging,
                    "effective_filter": life.effective if life.fi_logging
                    else "unknown",
                })  # fmt: skip
            life.reset()
    for tid, life in lives.items():
        if life.active():
            diagnose(tid, None, "the log ends before the agent's self-connection",
                     life)  # fmt: skip
    kept = []
    for record in records:
        if lifetimes.get(record["tid"], 0) > 1:
            diagnose(record["tid"], record["lines"][1],
                     "several process lifetimes on one thread ID in this input: "
                     "the record is ambiguous")  # fmt: skip
        else:
            kept.append(record)
    return kept, diagnostics


def summarize(logs) -> dict:
    """The agents' rail records of each log, each log parsed on its own; an
    agent named by more than one record is listed under ``conflicts``."""
    records, diagnostics = [], []
    for log in logs:
        with open(log, errors="replace") as stream:
            found, problems = parse_lines(stream, str(log))
        records += found
        diagnostics += problems
    by_agent: dict[str, int] = {}
    for record in records:
        by_agent[record["agent"]] = by_agent.get(record["agent"], 0) + 1
    return {
        "schema": SCHEMA,
        "source": "NIXL_LIBFABRIC_LOG",
        "records": records,
        "conflicts": sorted(agent for agent, count in by_agent.items() if count > 1),
        "unparsed": len(diagnostics),
        "unparsed_examples": diagnostics[:MAX_DIAGNOSTICS],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", help="server logs (NIXL_LOG_LEVEL=INFO)")
    parser.add_argument("--output", help="JSON file (default: standard output)")
    args = parser.parse_args(argv)
    text = json.dumps(summarize(args.logs), indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
