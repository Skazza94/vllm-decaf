# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic real-endpoint CLI, with no harness or serving dependencies."""

import argparse
import hashlib
import json
import signal
import sys
from pathlib import Path

from .code import code_digest
from .config import ProbeConfig
from .exit_record import EXIT_SCHEMA, write_exit_record
from .health import (
    core_completion,
    endpoint_capture,
    inventory,
    launch_config_problems,
    validate_inventory,
)
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
    parser.add_argument(
        "--sidecar",
        metavar="SETTINGS",
        help="run as the observer's launcher beside the server (a pod sidecar): "
        "SETTINGS is JSON with probes, role, expected, profiles_dir, "
        "observer_root and irq_policy; needs --run-id and --endpoint",
    )
    parser.add_argument("--endpoint", help="this endpoint's ID, with --sidecar")
    parser.add_argument(
        "--code-digest",
        action="store_true",
        help="print the digest of this observer code, as captures record it",
    )
    args = parser.parse_args()
    if args.sidecar:
        if not args.run_id or not args.endpoint:
            parser.error("--sidecar needs --run-id and --endpoint")
        from .sidecar import main as sidecar_main

        # the observer runs from the same standalone entry point as this
        return sidecar_main(
            args.sidecar, args.run_id, args.endpoint, str(Path(sys.argv[0]).absolute())
        )
    if args.code_digest:
        print(code_digest())
        return 0
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
    # one launch form: the launcher names the expected recorders and the
    # observer finds and freezes them (identities supplied by a launcher would
    # be a second, unchecked source of truth)
    if problems := launch_config_problems(values):
        raise ValueError(
            "--config names the endpoint's expected recorders per role "
            "(`expected`); the observer discovers its inventory and owner: "
            + "; ".join(problems)
        )
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
