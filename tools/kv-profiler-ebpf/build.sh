#!/bin/bash
# Build artifacts only; no BPF program load or attachment.
set -euo pipefail
source_root=$1
output=$2
mkdir "$output"
prefix=/opt/kv-profiler-tools
src="$source_root/csrc/kv_profiler/ebpf"
export KVP_BTF_FILE=${KVP_BTF_FILE:-/sys/kernel/btf/vmlinux}
"$prefix/bin/bpftool" btf dump file "$KVP_BTF_FILE" format c > "$output/vmlinux.h"
"$prefix/bin/clang" -target bpf -mcpu=v3 -D__TARGET_ARCH_x86 -O2 -g \
    -Wall -Werror -Wframe-larger-than=256 -I"$output" -I"$prefix/include" -I"$src" \
    -c "$src/observe.bpf.c" -o "$output/observe.bpf.o"
"$prefix/bin/clang" -O2 -g -Wall -Werror -I"$prefix/include" -I"$src" \
    "$src/collector.c" "$prefix/lib/libbpf.a" -l:libelf.so.1 -lz -o "$output/collector"
"$prefix/bin/clang" -O2 -g -Wall -Werror -pthread -I"$source_root" \
    -I"$source_root/tests/distributed/kv_transfer" -I"$src" \
    "$source_root/tests/distributed/kv_transfer/endpoint_ebpf_native.c" -o "$output/native_tests"
"$prefix/bin/clang" -O2 -shared -fPIC -Wall -Werror \
    "$source_root/tests/distributed/kv_transfer/endpoint_ebpf_syscalls.c" \
    -ldl -o "$output/deny_syscalls.so"
"$prefix/bin/clang" -O2 -g -Wall -Werror -I"$prefix/include" -I"$source_root" -I"$src" \
    "$source_root/tests/distributed/kv_transfer/endpoint_ebpf_accounting.c" \
    "$prefix/lib/libbpf.a" -l:libelf.so.1 -lz -o "$output/accounting_fixture"
cp "$src/limits.json" "$output/limits.json"
/opt/venv/bin/python - "$output" <<'MANIFEST'
from pathlib import Path
import hashlib,json,os,sys
root=Path(sys.argv[1])
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(root/'limits.json')=='5f9162c8a1e3a907c5ccaadfa7cb29ba09b418f4dc11326eb12701e147be4929'
data=dict(image='sha256:ff979851b632e2048a4a8dd141cf71b2a4701f243c0bf96fb275f7286daadc76',
 limits_sha256=sha(root/'limits.json'),kernel_release=os.uname().release,
 kernel_btf_sha256=sha(Path(os.environ['KVP_BTF_FILE'])),
 files={name:sha(root/name) for name in ['collector','observe.bpf.o','limits.json','vmlinux.h']},
 qualification='unprivileged tests only; no kernel runtime coverage')
(root/'manifest.json').write_text(json.dumps(data,indent=2)+'\n')
MANIFEST
"$output/collector" --inspect "$output/observe.bpf.o" > "$output/object-inspection.jsonl"
"$output/native_tests"
