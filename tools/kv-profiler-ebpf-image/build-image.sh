#!/bin/bash
# Build the eBPF tooling image once: verify (or download) the locked inputs, run
# one offline `docker build` on the recipe's base image, then the static image
# checks. It never loads or attaches a BPF program and starts no service.
#
# Usage: build-image.sh /absolute/new/output-dir
# Environment (all optional):
#   PYTHON        the interpreter for the recipe's scripts (default: python3)
#   TAG           the image tag (default: vllm-kv-profiler-ebpf-tools)
#   BUILD_CPUS    a `docker build --cpuset-cpus` value (default: unrestricted)
#   BUILD_MEMORY  a `docker build --memory` value (default: 8g)
#   BUILD_LOCK    a file to hold an exclusive flock on from the downloads to the end
set -euo pipefail
if [ "$#" != 1 ]; then
    echo 'Usage: build-image.sh /absolute/new/output-dir' >&2
    exit 2
fi
recipe=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output=$1
case "$output" in /*) ;; *) echo 'The output directory must be absolute' >&2; exit 2;; esac
mkdir "$output"
exec > >(tee "$output/build.log") 2>&1
python=${PYTHON:-python3}
# The lock covers the downloads too: builds sharing a recipe share its assets/.
if [ -n "${BUILD_LOCK:-}" ]; then
    exec 9>"$BUILD_LOCK"
    flock -w 14400 9
fi
"$python" "$recipe/freeze.py" "$recipe" --download
# The base image the lock was solved against, by image ID: it must be present.
base=$("$python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["base_image"])' \
    "$recipe/inputs.lock.json")
docker image inspect "$base" > "$output/base-image.json"
"$python" - "$output/base-image.json" <<'CHECK'
import json, sys
image = json.load(open(sys.argv[1]))[0]
assert image["Architecture"] == "amd64" and image["Os"] == "linux"
CHECK
args=(--network=none --pull=false --memory="${BUILD_MEMORY:-8g}"
      --iidfile "$output/image.id" --tag "${TAG:-vllm-kv-profiler-ebpf-tools}")
if [ -n "${BUILD_CPUS:-}" ]; then
    args+=(--cpuset-cpus="$BUILD_CPUS")
fi
# The legacy builder takes the local base by image ID and pulls nothing. A failed
# build is not retried: no dependency or version changes silently.
DOCKER_BUILDKIT=0 docker build "${args[@]}" --file "$recipe/Dockerfile" "$recipe"
image=$(cat "$output/image.id")
docker image inspect "$image" > "$output/tooling-image.json"
"$python" "$recipe/check-image.py" "$base" "$image" "$output"
echo "Build and static image checks complete: $image"
