# Frozen L3 tooling image (main builds, agents do not)

This is the image checkpoint after approved `agent-1#7`, not a probe
implementation. It adds an SDK under `/opt/kv-profiler-tools` only. It adds no
BPF program, daemon, bcc/bpftrace, Python dependency or transport/provider.
Neither tool presence nor a successful compilation establishes probe coverage.

## Immutable inputs and dependency closure

Both build and final stages use the already-local image ID
`sha256:c25c4f62ca7bc4433e730fc8a6238532dbe8739f13e62018a503f2c43c200dd5`
(amd64). This is the Docker image/config digest, not a registry manifest alias;
there is no mutable build-stage tag or extra frontend image. The reviewed legacy
Docker builder uses that local ID and never pulls. No fallback build is allowed.

`inputs.lock.json` records every exact version, URL, SHA-256 and size for 20
additional Debian packages and two source archives; its dependency graph has
119 selected package records, with base-provided and added dependencies
explicitly distinguished. The base's complete dpkg status is separately hashed.
The solver planned zero upgrades/removals. Native compiler, make, libc/kernel
headers, binutils and other inherited dependencies are fixed by that image ID.
No apt resolver/network access runs inside the build.

Clang/LLVM is 18.1.8 with the exact apt.llvm.org Jammy build version in the lock.
The Jammy and LLVM package indexes were signature-verified in isolated apt state;
LLVM key fingerprint is `6084F3CF814B57C1CF12EFD515CF4D18AF4F7421`.
Frozen source commits are bpftool v7.6.0
`e1bd72cee842b415b87e632425d04aa773344654` and its libbpf submodule
`da08818f4f3b6a8f6d15617184de9a6c34c5b642`.
The source tar hashes, not a tag lookup at build time, are required.
No floating apt install or Git checkout/submodule update is used.

`recipe.sha256` freezes every recipe/check file. Downloads are checked before the
build; hashes are also checked in the offline stage. All new package contents
are copied into the private sysroot; optional bpftool JIT-disassembly/libcap and
internal host-kernel-specific skeleton generation are disabled explicitly. The
bpftool skeleton **generator** remains in the tool. Static libbpf and headers are
installed in the private SDK. Source/package license material is retained.

All new DSOs remain private. Wrapper-scoped `LD_LIBRARY_PATH` and removal of
`LD_PRELOAD` affect only explicitly invoked tooling subprocesses. There is no
final-image ENV/PATH/LD_LIBRARY_PATH/loader-cache/Python setting change.
`runtime-closure.py` statically resolves each tool's DT_NEEDED closure against
private roots and the exact base; it does not load any DSO. `elf-closure.json`
records actual linked files/hashes. A pre-install system-library snapshot rejects
any dependency outside the prefix unless its bytes match the exact base. No claim that CO-RE fixes provider ABI.

## Main's single build

After approval, **main**, not an agent, runs:

```bash
bash tools/kv-profiler-ebpf-image/main-build.sh --main-run /absolute/new/build-evidence
```

Use the assigned worktree path, or main's reviewed copy with identical
`recipe.sha256`. The script verifies/downloads all inputs, takes
`/home/mariano/workspace/safir/harness.lock`, refuses existing `pd-*`/`llmd-*`
containers without touching them, then issues exactly one offline
`docker build` with 4 compiler jobs, CPUs 0–13 and 8 GiB memory. It records the
immutable resulting image ID, full inspect data and build log. A failure does
not trigger a retry or a silent dependency/version change. No serving, harness
up/down or probe attachment is in this script. The asset directory is ignored
by Git; a preverified cache may be copied there, with hashes checked again.

Static post-build checks compare runtime config/search-path inputs and hash
protected transport, wheel, provider and loader trees using unique,
never-started inspection containers. Only those containers are removed. The
base filesystem layer list must be the exact prefix of the new image's list,
with one filesystem layer added by the final-stage private-prefix COPY. Core
loader cache/config and final dpkg status remain unchanged. Missing optional
provider paths must remain missing, not become zero observations. These checks
preserve resolver inputs; they do not claim a native runtime load or RDMA test.

The current base has verbs core but no audited mlx5 provider. This recipe does
not add one. Actual node provider/UCX binaries and ABI must be supplied/frozen
before provider probes can be enabled; no guessed callback offset is accepted.

## After main returns the image identity

The next assigned agent checkpoint runs `compile-checks.sh /tmp/new-check-dir`
in that exact image, with the script mounted read-only and a writable temporary
output, unprivileged and without network. It compiles a BPF object, strips DWARF,
generates a skeleton and compiles/links a native program with the entire static
libbpf archive. It never opens/loads/attaches the skeleton or executes that native
program. Archive versions, logs, object/header/native hashes and image ID, then
report them **before** any live probing. Main's build script does not run this
later checkpoint automatically. No kernel BTF input or privilege is required for
this simple compile fixture.

Do not use `bpftool prog load`, feature probing, pinning, attach commands or a
privileged container here. Live probe correctness, real-provider coverage,
serving/A-B and physical RDMA/IRQ qualification remain separate checkpoints.
