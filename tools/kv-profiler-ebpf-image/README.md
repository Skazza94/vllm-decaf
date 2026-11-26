# eBPF tooling image

A frozen build environment for the KV profiler's opt-in eBPF collector
(`csrc/kv_profiler/ebpf`). It adds a private SDK under `/opt/kv-profiler-tools`
to a vLLM CPU serving image, and nothing else: no BPF program, daemon,
bcc/bpftrace, Python dependency, transport or provider. Build the collector's
artifacts inside it with `tools/kv-profiler-ebpf/build.sh` (see the profiler
guide, `docs/observability/kv_profiler_core.md`, "Opt-in TCP eBPF events"). The
image is for building only: the artifacts it produces run without it, and target
BTF kernels from 5.15 upward.

## What it contains

- Clang/LLVM 18.1.8, the exact apt.llvm.org Jammy build recorded in the lock.
  The package indexes were signature-verified; the LLVM key fingerprint is
  `6084F3CF814B57C1CF12EFD515CF4D18AF4F7421`.
- bpftool v7.6.0 (`e1bd72cee842b415b87e632425d04aa773344654`), with its libbpf
  submodule (`da08818f4f3b6a8f6d15617184de9a6c34c5b642`) installed as a static
  library with its headers. The skeleton generator stays in bpftool; JIT
  disassembly, libcap and host-kernel-specific skeleton generation are disabled.
- Source and package license material.

Every added shared library stays private to the prefix. The tools' wrappers set
`LD_LIBRARY_PATH` (and drop `LD_PRELOAD`) for their own subprocesses only; the
image's environment, `PATH`, loader cache and Python are unchanged.

## Frozen inputs

- **Base image:** the vLLM CPU serving image the dependency lock was solved
  against, named in `inputs.lock.json` (`base_image`) and the `Dockerfile` by its
  image ID, not a registry tag. It must already be present on the build host.
  Another base needs a new lock: the added packages and the base-provided ones
  are resolved against this exact base.
- **`inputs.lock.json`:** every added Debian package and source archive, with
  its exact version, URL, SHA-256 and size, and the dependency graph, base-provided
  and added packages distinguished. The base's dpkg status is hashed too.
- **`recipe.sha256`:** the hash of every recipe and check file. Regenerate it
  after changing any of them:

  ```bash
  cd tools/kv-profiler-ebpf-image
  git ls-files | grep -v -e '^recipe.sha256$' | LC_ALL=C sort | xargs sha256sum > recipe.sha256
  ```

No build step resolves, updates or checks out anything: downloads are checked
against the lock before the build and again in its offline stage.

## Building

```bash
bash tools/kv-profiler-ebpf-image/build-image.sh /absolute/new/output-dir
```

The script verifies the recipe and downloads the locked inputs into `assets/`
(ignored by Git; a copy of verified inputs can be placed there, and is checked
again). It then checks the base image is present and amd64/Linux, and runs one
offline `docker build` (no network, no pull) with the legacy builder, which
accepts the local base by image ID. A failed build is not retried. Optional
environment variables: `PYTHON` (the interpreter for the recipe's scripts),
`TAG` (default `vllm-kv-profiler-ebpf-tools`), `BUILD_CPUS` (a
`--cpuset-cpus` value), `BUILD_MEMORY` (default `8g`) and `BUILD_LOCK` (a file
to hold an exclusive lock on from the downloads to the end, so that builds of one
recipe on a shared host never download into `assets/` at the same time).

The output directory receives the build log, the base and built images'
inspect data, the image ID (`image.id`) and the static checks' results.

## Checks

- **Static image checks** (`check-image.py`, run by the build script) use
  never-started inspection containers, which they remove. The base's filesystem
  layers must be an exact prefix of the new image's, with one added layer (the
  private prefix). The loader cache and configuration and the dpkg status must be
  unchanged, and the protected transport, wheel, provider and loader trees
  byte-identical. Paths the base lacks must still be missing. `runtime-closure.py`
  resolves each tool's `DT_NEEDED` closure against the prefix and the base,
  without loading a library, and `elf-closure.json` records the linked files.
- **Compile smoke test** (`compile-checks.sh /new/output-dir`, run inside the
  built image, unprivileged and without network): it compiles a BPF object,
  strips its DWARF, generates a skeleton, and links a native program against the
  static libbpf. It never loads or attaches anything and needs no kernel BTF.

These checks qualify the build environment, not the collector: the collector's
own checks (unprivileged contracts, native emulation, and live runs) are
described in the profiler guide.
