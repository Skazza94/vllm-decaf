#!/bin/bash
# Later checkpoint, AFTER main returns the built image identity. Never attaches.
set -euo pipefail
prefix=/opt/kv-profiler-tools
output=$1
mkdir "$output"
exec > >(tee "$output/compile.log") 2>&1
"$prefix/bin/clang" --version
"$prefix/bin/llvm-strip" --version
"$prefix/bin/bpftool" version
cat > "$output/noattach.bpf.c" <<'C'
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
SEC("uprobe") int noattach(void *ctx) { return 0; }
char LICENSE[] SEC("license") = "Dual BSD/GPL";
C
"$prefix/bin/clang" -target bpf -O2 -g -I"$prefix/include" \
    -I/usr/include/x86_64-linux-gnu -c "$output/noattach.bpf.c" -o "$output/noattach.bpf.o"
"$prefix/bin/llvm-strip" -g "$output/noattach.bpf.o"
# Keep BTF for skeleton generation: -g stripping DWARF preserves .BTF.
"$prefix/bin/bpftool" gen skeleton "$output/noattach.bpf.o" name noattach > "$output/noattach.skel.h"
cat > "$output/skeleton.c" <<'C'
#include "noattach.skel.h"
int main(void) { return 0; }
C
"$prefix/bin/clang" -I"$prefix/include" -I"$output" \
    "$output/skeleton.c" -Wl,--whole-archive "$prefix/lib/libbpf.a" \
    -Wl,--no-whole-archive -l:libelf.so.1 -lz -o "$output/skeleton"
# Compile/link only: no skeleton open/load/attach or BPF/perf syscall.
sha256sum "$output"/*.o "$output"/*.h "$output/skeleton" > "$output/artifacts.sha256"
echo 'PASS: BPF object and skeleton native compile/link; no load/attachment'
