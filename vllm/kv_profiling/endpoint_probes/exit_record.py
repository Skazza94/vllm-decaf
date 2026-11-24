# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The observer's exit record: how it ended, as the process that launched and
reaped it saw it. Any launcher writes it (a container runtime adapter, a pod
sidecar, a plain parent process), so a capture is checked the same way on any
deployment."""

import json
import os
import tempfile
from pathlib import Path

EXIT_RECORD = "exit.json"
EXIT_SCHEMA = 1
# The observer capture's format, recorded in its metadata when it is created.
# Format 2 is checked against an exit record; earlier captures are rerun.
CAPTURE_FORMAT = 2
KILLED = (None, "oom", "signal")


def write_exit_record(
    directory,
    *,
    run_id: str,
    endpoint: str,
    process_id: str | None,
    exit_code: int | None,
    launcher: str,
    signal: int | None = None,
    killed: str | None = None,
    timed_out: bool = False,
    problems=(),
    evidence: dict | None = None,
) -> dict:
    """Publish the record atomically, after the observer was reaped.

    ``process_id`` is the observer capture's own (its metadata's), binding the
    record to that attempt; ``problems`` are what the launcher could not
    reconcile (an unconfirmed exit, a failed cleanup), each failing the check;
    ``evidence`` is launcher-specific detail, never checked."""
    record = {
        "schema": EXIT_SCHEMA,
        "run_id": run_id,
        "endpoint": endpoint,
        "process_id": process_id,
        "exit_code": exit_code,
        "signal": signal,
        "killed": killed,
        "timed_out": timed_out,
        "launcher": launcher,
        # never converted: a mistyped field is refused below, not repaired
        "problems": list(problems) if isinstance(problems, (list, tuple)) else problems,
        "evidence": {} if evidence is None else evidence,
    }
    if not _well_formed(record):
        raise ValueError(f"not a well-formed exit record: {record!r}")
    directory = Path(directory)
    # each publication its own temporary file, so overlapping attempts never
    # rename another's or leave a partial record
    descriptor, temporary = tempfile.mkstemp(
        prefix=".exit.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(record))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / EXIT_RECORD)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return record


FIELDS = ("schema", "run_id", "endpoint", "process_id", "exit_code", "signal",
          "killed", "timed_out", "launcher", "problems", "evidence")  # fmt: skip


def _identifier(value) -> bool:
    return isinstance(value, str) and bool(value)


def _integer_or_none(value) -> bool:
    return value is None or type(value) is int  # bool is not an exit code


def _well_formed(record) -> bool:
    """Every field present with its type: an integer schema, nonempty
    identifiers (an attempt without a capture has ``process_id: null``)."""
    return (
        isinstance(record, dict)
        and all(name in record for name in FIELDS)
        and type(record["schema"]) is int
        and record["schema"] == EXIT_SCHEMA
        and _identifier(record["run_id"])
        and _identifier(record["endpoint"])
        and (record["process_id"] is None or _identifier(record["process_id"]))
        and _identifier(record["launcher"])
        and _integer_or_none(record["exit_code"])
        and _integer_or_none(record["signal"])
        and record["killed"] in KILLED
        and type(record["timed_out"]) is bool
        and isinstance(record["problems"], list)
        and all(_identifier(problem) for problem in record["problems"])
        and isinstance(record["evidence"], dict)
    )


def exit_problems(directory, run_id: str, endpoint: str, process_id) -> list[str]:
    """Why the exit record does not show a clean exit of this observer process:
    missing or malformed, another run's, endpoint's or attempt's, or a nonzero,
    signalled, killed, timed-out or unreconciled exit."""
    path = Path(directory) / EXIT_RECORD
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError:
        return ["observer exit record missing"]
    except (OSError, ValueError):
        return ["observer exit record unreadable"]
    if not _well_formed(record):
        return ["observer exit record malformed"]
    if (record["run_id"], record["endpoint"]) != (run_id, endpoint) or (
        not isinstance(process_id, str) or record["process_id"] != process_id
    ):
        return ["observer exit record belongs to another run, endpoint or attempt"]
    if (
        record["exit_code"] != 0
        or record["signal"] is not None
        or record["killed"] is not None
        or record["timed_out"]
        or record["problems"]
    ):
        return [
            "observer exit not clean: "
            + json.dumps(
                {
                    k: record[k]
                    for k in ("exit_code", "signal", "killed", "timed_out", "problems")
                }
            )  # fmt: skip
        ]
    return []
