#!/bin/bash
# Executed only by main's single offline image build, never by a probe agent.
set -euo pipefail
cd /freeze
/opt/venv/bin/python /freeze/freeze.py /freeze
sha256sum /var/lib/dpkg/status > /tmp/base-status.sha256
/opt/venv/bin/python - <<'CHECK'
import hashlib, json
from pathlib import Path
lock=json.loads(Path('/freeze/inputs.lock.json').read_text())
assert hashlib.sha256(Path('/var/lib/dpkg/status').read_bytes()).hexdigest() == lock['base_dpkg_status_sha256']
CHECK
# Freeze inherited system-library bytes before adding build-stage packages.
/opt/venv/bin/python - <<'BASE'
import hashlib, json
from pathlib import Path
files = {}
for directory in ['/lib/x86_64-linux-gnu', '/usr/lib/x86_64-linux-gnu', '/lib64']:
    for path in Path(directory).glob('*.so*'):
        if path.is_file():
            resolved = path.resolve()
            files[str(resolved)] = hashlib.sha256(resolved.read_bytes()).hexdigest()
Path('/tmp/base-library-hashes.json').write_text(json.dumps(files, sort_keys=True))
BASE
# The signed-index solver chose additions only. No online apt or resolver here.
dpkg --unpack /freeze/assets/*.deb
dpkg --configure -a
/opt/venv/bin/python - <<'CHECK'
import json, subprocess
from pathlib import Path
for package in json.loads(Path('/freeze/inputs.lock.json').read_text())['packages']:
    actual=subprocess.check_output(['dpkg-query','-W','-f=${Version}',package['package']],text=True)
    assert actual == package['version'], (package['package'],actual)
CHECK
prefix=/opt/kv-profiler-tools
mkdir -p /src/bpftool /src/bpftool/libbpf "$prefix"/{bin,libexec,include,lib,manifest,sysroot,licenses}
tar -xzf /freeze/assets/bpftool.tar.gz -C /src/bpftool --strip-components=1
tar -xzf /freeze/assets/libbpf.tar.gz -C /src/bpftool/libbpf --strip-components=1
# Disable optional JIT-disassembly/libcap and internal kernel-specific skeletons.
# bpftool gen skeleton still exists. No /sys/kernel/BTF input enters this build.
make -C /src/bpftool/src -j4 CC=clang-18 HOSTCC=clang-18 \
    CLANG=clang-18 LLVM_STRIP=llvm-strip-18 VMLINUX_BTF= VMLINUX_H= \
    feature-clang-bpf-co-re=0 feature-llvm=0 feature-libcap=0 \
    feature-libbfd=0 feature-libbfd-liberty=0 feature-libbfd-liberty-z=0 \
    feature-libelf-zstd=0
cp /src/bpftool/src/bpftool "$prefix/libexec/bpftool"
cp /src/bpftool/src/libbpf/libbpf.a "$prefix/lib/libbpf.a"
cp -a /src/bpftool/src/libbpf/include/bpf "$prefix/include/"
# Keep every added package under the isolated prefix in the final image.
# All remaining dependencies are inherited from the identical pinned base.
for deb in /freeze/assets/*.deb; do dpkg-deb -x "$deb" "$prefix/sysroot"; done
cp /freeze/inputs.lock.json /freeze/recipe.sha256 "$prefix/manifest/"
mkdir -p "$prefix/licenses/bpftool" "$prefix/licenses/libbpf"
cp -a /src/bpftool/LICENSE* "$prefix/licenses/bpftool/"
cp -a /src/bpftool/libbpf/LICENSE* "$prefix/licenses/libbpf/"
for tool in clang llvm-strip llvm-objdump; do
    cat > "$prefix/bin/$tool" <<WRAPPER
#!/bin/sh
unset LD_PRELOAD
export LD_LIBRARY_PATH='$prefix/sysroot/usr/lib/x86_64-linux-gnu:$prefix/sysroot/usr/lib/llvm-18/lib'
exec '$prefix/sysroot/usr/lib/llvm-18/bin/$tool' "\$@"
WRAPPER
    chmod 755 "$prefix/bin/$tool"
done
cat > "$prefix/bin/bpftool" <<WRAPPER
#!/bin/sh
unset LD_PRELOAD
export LD_LIBRARY_PATH='$prefix/sysroot/usr/lib/x86_64-linux-gnu:$prefix/sysroot/usr/lib/llvm-18/lib'
exec '$prefix/libexec/bpftool' "\$@"
WRAPPER
chmod 755 "$prefix/bin/bpftool"
# Record the native tool's actual dependency closure, without ldd or execution.
/opt/venv/bin/python /freeze/runtime-closure.py "$prefix" \
    /tmp/base-library-hashes.json > "$prefix/manifest/elf-closure.json"
cp /tmp/base-library-hashes.json "$prefix/manifest/"
find "$prefix" -type l -printf '%p -> %l\n' | sort > "$prefix/manifest/symlinks.txt"
