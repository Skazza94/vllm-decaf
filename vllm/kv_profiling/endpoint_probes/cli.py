# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic real-endpoint CLI, with no harness or serving dependencies."""

import argparse
import hashlib
import json
import signal
import sys
from pathlib import Path

from .config import ProbeConfig
from .exit_record import EXIT_SCHEMA, write_exit_record
from .health import core_completion, endpoint_capture, inventory, validate_inventory
from .netdev import read_small
from .runner import Observer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=False,
        help="JSON with probe settings and explicit runtime identity",
    )
    parser.add_argument("--check-capture")
    parser.add_argument("--core-dir")
    parser.add_argument(
        "--check-core",
        help="validate a node's recorder captures (with --run-id): streams, "
        "terminal rows and manifests",
    )
    parser.add_argument("--inventory")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--record-exit",
        metavar="DIRECTORY",
        help="publish the observer's exit record in DIRECTORY, its fields as "
        "JSON on stdin (for a launcher that reaped the observer)",
    )
    args = parser.parse_args()
    if args.record_exit:
        fields = json.loads(sys.stdin.read(1 << 20))
        if not isinstance(fields, dict):
            raise ValueError("--record-exit takes a JSON object on stdin")
        # the full record may be given: its schema must be this one
        if "schema" in fields and (
            type(fields["schema"]) is not int or fields.pop("schema") != EXIT_SCHEMA
        ):
            raise ValueError(f"--record-exit writes exit record schema {EXIT_SCHEMA}")
        write_exit_record(args.record_exit, **fields)
        return 0
    if args.inventory:
        result = inventory(args.inventory, args.run_id)
        validate_inventory(result, args.run_id)
        print(json.dumps(result))
        return 0
    if args.check_core:
        result = core_completion(args.check_core, args.run_id)
        print(json.dumps(result))
        return 0 if result["all_closed"] else 2
    if args.check_capture:
        result = endpoint_capture(args.check_capture, args.core_dir)
        print(json.dumps(result))
        return 0 if result["complete"] else 2
    if args.config is None:
        parser.error("--config is required for capture")
    values = json.loads(read_small(args.config, 1 << 20))
    expected_hash = values.get("package_sha256")
    if (
        expected_hash
        and hashlib.sha256(Path(sys.argv[0]).read_bytes()).hexdigest() != expected_hash
    ):
        raise ValueError("observer package hash mismatch")
    config = ProbeConfig(**values.pop("probes"))
    stop_file = values.pop("stop_file")
    completion_file = values.pop("endpoint_complete_file")
    observer = Observer(config, **values)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: observer.stop_event.set())
    return (
        0
        if observer.run(stop_file=stop_file, endpoint_complete_file=completion_file)
        else 2
    )
