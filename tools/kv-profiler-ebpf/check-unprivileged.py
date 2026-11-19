"""Passive inspector and privilege-refusal checks; no BPF/perf syscall may pass."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from vllm.kv_profiling.endpoint_probes.ebpf import Channel

root = Path(sys.argv[1])
source = Path(sys.argv[2])
limits = json.loads((root / "limits.json").read_text())
expected = {m["name"]: m for m in limits["maps"]}
assert (
    hashlib.sha256((root / "limits.json").read_bytes()).hexdigest()
    == "5f9162c8a1e3a907c5ccaadfa7cb29ba09b418f4dc11326eb12701e147be4929"
)
log = root / "denied-syscalls.log"
env = {
    **os.environ,
    "LD_PRELOAD": str(root / "deny_syscalls.so"),
    "KVP_SYSCALL_LOG": str(log),
}
inspect = subprocess.run(
    [str(root / "collector"), "--inspect", str(root / "observe.bpf.o")],
    env=env,
    capture_output=True,
    text=True,
)
(root / "inspect.stdout").write_text(inspect.stdout)
(root / "inspect.stderr").write_text(inspect.stderr)
assert inspect.returncode == 0, inspect.stderr
programs = 0
for line in inspect.stdout.splitlines():
    row = json.loads(line)
    if row["type"] == "map":
        name = row["name"]
        if name in expected:
            m = expected[name]
            assert row["entries"] == m["entries"]
            assert (
                row["key"] <= m["key_bytes_cap"]
                and row["value"] <= m["value_bytes_cap"]
            )
        else:
            assert (
                name.startswith(".") and row["entries"] == 1 and row["value"] <= 16384
            )
    elif row["type"] == "program":
        programs += 1
        assert row["autoattach"] is False
assert programs == 7
pid = os.getpid()
fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
ns = os.stat("/proc/self/ns/pid")
net = os.stat("/proc/self/ns/net")
for target in (0, -1, pid):
    plan = root / f"negative-{target}.plan"
    plan.write_text(
        f"P {pid} {fields[19]}\n"
        f"N {ns.st_dev} {ns.st_ino} {net.st_ino} 0 1 1\n"
        f"B 1\nT {target} 1 77 1 1 /not/a/provider\n"
    )
    plan.chmod(0o600)
    run = subprocess.run(
        [str(root / "collector"), "--arm", str(plan), str(root / "observe.bpf.o")],
        env=env,
        capture_output=True,
        text=True,
    )
    (root / f"negative-{target}.stdout").write_text(run.stdout)
    (root / f"negative-{target}.stderr").write_text(run.stderr)
    assert run.returncode == 2
    row = json.loads(run.stdout)
    assert row["available"] is False and row["exact"] is False
    assert row["reason"] == ("privilege_or_parent" if target > 0 else "invalid_plan")
assert not log.exists() or "PROBE" not in log.read_text()
print(
    "PASS: passive map/autoattach contracts; 0/-1 TGIDs and unprivileged arm "
    "refused; zero BPF/perf syscalls observed"
)

for label, text, bound, exit_code in (
    ("missing", "map_type: 1\n", 100, 2),
    ("over", "memlock: 101\n", 100, 3),
    ("cap", "memlock: 16777217\n", 16777216, 3),
    ("equal", "memlock: 100\n", 100, 0),
):
    info = root / (label + ".fdinfo")
    info.write_text(text)
    run = subprocess.run(
        [str(root / "accounting_fixture"), str(bound), str(info)],
        env=env,
        capture_output=True,
        text=True,
    )
    (root / (label + ".accounting.stdout")).write_text(run.stdout)
    (root / (label + ".accounting.stderr")).write_text(run.stderr)
    assert run.returncode == exit_code, run.stderr
    rows = [json.loads(row) for row in run.stdout.splitlines()]
    if label in ("over", "cap"):
        assert rows[0]["outcome"] == "FAIL"
        assert rows[0]["kernel_map_bytes"] > rows[0]["bound_bytes"]
        assert rows[1]["reason"] == "kernel_map_budget_exceeded"
    elif label == "missing":
        assert rows[0]["reason"] == "kernel_map_accounting_unavailable"
    assert not log.exists() or "PROBE" not in log.read_text()
print(
    "PASS: actual collector map-accounting decisions/records; missing unavailable, "
    "observed overrun FAIL; no attachment or BPF/perf syscall"
)

# Use the actual native serializer as an external child, not a fabricated row.


child = subprocess.Popen(
    [str(root / "accounting_fixture"), "100", str(root / "over.fdinfo")],
    env=env,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
channel = Channel(child, root)
writer = SimpleNamespace(saved_sequence=0)
rows = []

deadline = time.monotonic() + 5
while channel.selector.get_map():
    assert time.monotonic() < deadline
    rows.extend(channel.poll(writer, 0.01))
result = channel.stop(writer)
(root / "accounting-channel.json").write_text(
    json.dumps({"rows": rows, "result": result}, indent=2) + "\n"
)
assert result["state"] == "failed" and result["outcome"] == "FAIL"
assert result["resource"]["kernel_map_bytes"] == 101
assert result["resource"]["bound_bytes"] == 100
assert result["native_exit"] == 3 and result["resource_complete"]
assert not log.exists() or "PROBE" not in log.read_text()
print(
    "PASS: native measured FAIL reaches controller channel with "
    "observed bytes/bound and exact child usage"
)
