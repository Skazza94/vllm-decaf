#!/bin/bash
# Portable artifact build and passive inspection only; never loads BPF.
set -euo pipefail
source_root=$1
output=$2
mkdir "$output"
prefix=${KVP_TOOLS_PREFIX:-/opt/kv-profiler-tools}
src="$source_root/csrc/kv_profiler/ebpf"
multiarch=$("$prefix/bin/clang" -print-target-triple)
multiarch=${multiarch/-unknown/}
multiarch=${multiarch/-pc/}
"$prefix/bin/clang" -target bpf -mcpu=v3 -O2 -g -Wall -Werror \
  -Wframe-larger-than=256 -I"$prefix/include" -I"$src" \
  -idirafter "/usr/include/$multiarch" -c "$src/observe.bpf.c" -o "$output/observe.bpf.o"
"$prefix/bin/clang" -O2 -g -Wall -Werror -I"$prefix/include" -I"$src" \
  "$src/collector.c" "$prefix/lib/libbpf.a" -l:libelf.so.1 -lz -o "$output/collector"
cp "$src/limits.json" "$output/limits.json"
/opt/venv/bin/python - "$output" "$src" <<'MANIFEST'
from pathlib import Path
import hashlib,json,sys
root,src=map(Path,sys.argv[1:])
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
data=dict(schema=3,mode='tcp_tracepoints',
 files={name:sha(root/name) for name in ['collector','observe.bpf.o','limits.json']})
(root/'manifest.json').write_text(json.dumps(data,indent=2)+'\n')
# Build input provenance is never a runtime host eligibility rule.
(root/'build-inputs.json').write_text(json.dumps({
 name:sha(src/name) for name in ['core_types.h','protocol.h','observe.bpf.c','collector.c']
},indent=2)+'\n')
MANIFEST
"$output/collector" --inspect "$output/observe.bpf.o" > "$output/object-inspection.jsonl"
"$prefix/bin/bpftool" gen skeleton "$output/observe.bpf.o" > "$output/observe.skel.h"

"$prefix/bin/llvm-objdump" --section-headers "$output/observe.bpf.o" > "$output/sections.txt"
"$prefix/bin/llvm-objdump" --disassemble "$output/observe.bpf.o" > "$output/disassembly.txt"
