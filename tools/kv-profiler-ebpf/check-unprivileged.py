"""Inspect artifacts and exercise privilege refusal without load or attachment."""

import json
import os
import subprocess
import sys
from pathlib import Path

from vllm.kv_profiling.endpoint_probes.ebpf import (
    effective_privilege,
    validate_artifacts,
)

root = Path(sys.argv[1])
validate_artifacts(root)
assert not effective_privilege(), "unprivileged checks require dropped BPF privileges"
limits = json.loads((root / "limits.json").read_text())
rows = [
    json.loads(line)
    for line in (root / "object-inspection.jsonl").read_text().splitlines()
]
assert {r["name"] for r in rows if r["type"] == "map"} == {
    "events",
    "control",
    "loss_counts",
}
for row in rows:
    if row["type"] == "map":
        spec = limits["maps"][row["name"]]
        assert row["entries"] == spec["max_entries"]
        assert row["key"] == spec.get("key_bytes", 0)
        assert row["value"] == spec.get("value_bytes", 0)
programs = [r for r in rows if r["type"] == "program"]
assert len(programs) == 4 and all(r["autoattach"] is False for r in programs)
assert all("tcp_probe" not in r["section"] for r in programs)
pid = os.getpid()
start = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
plan = root / "unprivileged.plan"
plan.write_text(f"P {pid} {start}\nN {os.stat('/proc/self/ns/net').st_ino}\nL \nR \nLA \nRA \n")
plan.chmod(0o600)
run = subprocess.run(
    [str(root / "collector"), "--arm", str(plan), str(root / "observe.bpf.o")],
    capture_output=True,
    text=True,
    timeout=5,
)
(root / "unprivileged.stdout").write_text(run.stdout)
(root / "unprivileged.stderr").write_text(run.stderr)
row = json.loads(run.stdout)
assert run.returncode == 2 and row["reason"] == "privilege_or_parent"
assert row["available"] is False and row["errno"] == 1
print(
    "PASS: artifact self-integrity; four passive hooks/three maps; "
    "unprivileged arm refused before object load"
)

# Count shared .text as well as entry sections, not just wrapper instructions.
code_bytes = sum(
    int(parts[2], 16)
    for line in (root / "sections.txt").read_text().splitlines()
    if len(parts := line.split()) >= 5 and parts[0].isdecimal() and "TEXT" in parts[4:]
)
assert code_bytes and code_bytes % 8 == 0
instructions = code_bytes // 8
assert instructions <= 16384 and instructions * 4 <= 65536
(root / "instruction-budget.json").write_text(
    json.dumps(
        {
            "all_executable_sections_insns": instructions,
            "conservative_per_program_insns": instructions,
            "conservative_total_insns": instructions * 4,
            "kernel_verifier_acceptance": "NOT_RUN",
        },
        indent=2,
    )
    + "\n"
)
print("PASS: shared-text-inclusive emitted instruction bounds; no kernel load")
