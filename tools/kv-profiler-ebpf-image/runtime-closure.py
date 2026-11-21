"""Static ELF dependency closure; this never loads an audited DSO."""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

prefix = Path(sys.argv[1])
base_libraries = json.loads(Path(sys.argv[2]).read_text())
search = [
    prefix / "sysroot/usr/lib/x86_64-linux-gnu",
    prefix / "sysroot/usr/lib/llvm-18/lib",
    Path("/lib/x86_64-linux-gnu"),
    Path("/usr/lib/x86_64-linux-gnu"),
    Path("/lib64"),
]
queue = [prefix / "libexec/bpftool"] + [
    prefix / "sysroot/usr/lib/llvm-18/bin" / name
    for name in ("clang", "llvm-strip", "llvm-objdump")
]
seen = {}
while queue:
    path = queue.pop().resolve(strict=True)
    if str(path) in seen:
        continue
    info = subprocess.check_output(["readelf", "-W", "-d", str(path)], text=True)
    needed = re.findall(r"\(NEEDED\).*\[(.*?)\]", info)
    rpaths = re.findall(r"\((?:RUNPATH|RPATH)\).*\[(.*?)\]", info)
    local = [
        Path(
            p.replace("${ORIGIN}", str(path.parent)).replace(
                "$ORIGIN", str(path.parent)
            )
        )
        for row in rpaths
        for p in row.split(":")
    ]
    dependencies = {}
    for soname in needed:
        # Frozen tooling files must resolve within the private roots or exact base.
        candidates = [p / soname for p in search + local]
        found = next((p.resolve() for p in candidates if p.is_file()), None)
        if found is None:
            raise ValueError(f"unresolved {soname} for {path}")
        dependencies[soname] = str(found)
        queue.append(found)
    seen[str(path)] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "needed": dependencies,
        "rpath_runpath": rpaths,
    }
for name, record in seen.items():
    if not Path(name).is_relative_to(prefix):
        if base_libraries.get(name) != record["sha256"]:
            raise ValueError(f"dependency escaped private prefix/exact base: {name}")
print(json.dumps(seen, indent=2, sort_keys=True))
