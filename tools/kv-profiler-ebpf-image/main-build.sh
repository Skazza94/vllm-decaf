#!/bin/bash
# MAIN ONLY. Exactly one build invocation; no attachment or serving commands.
set -euo pipefail
if [ "${1:-}" != --main-run ] || [ "$#" != 2 ]; then
    echo 'Usage (main only): bash main-build.sh --main-run /absolute/new/output-dir' >&2
    exit 2
fi
recipe=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output=$2
case "$output" in /*) ;; *) echo 'Output must be absolute' >&2; exit 2;; esac
mkdir "$output"
exec > >(tee "$output/build.log") 2>&1
PY=/home/mariano/workspace/safir/kubernetes-cluster-bench/.venv/bin/python
"$PY" "$recipe/freeze.py" "$recipe" --download
base=sha256:c25c4f62ca7bc4433e730fc8a6238532dbe8739f13e62018a503f2c43c200dd5
docker image inspect "$base" > "$output/base-image.json"
exec 9>/home/mariano/workspace/safir/harness.lock
flock -w 14400 9
# Never kill or stop existing services, even if they are stale.
docker ps --format '{{.Names}}' > "$output/running-before.txt"
if grep -Eq '^(pd-|llmd-)' "$output/running-before.txt"; then
    echo 'Existing shared serving containers; refusing build without touching them' >&2
    exit 3
fi
"$PY" - "$output/base-image.json" <<'CHECK'
import json, sys
image=json.load(open(sys.argv[1]))[0]
assert image['Architecture']=='amd64' and image['Os']=='linux'
CHECK
# No syntax frontend image is pulled. The legacy builder accepts local image IDs.
# Do not retry a failed build silently: the authorized build budget is one.
DOCKER_BUILDKIT=0 docker build --network=none --pull=false --cpuset-cpus=0-13 \
    --memory=8g --iidfile "$output/image.id" \
    --tag vllm:cpu-aisim-profiling-ebpf-frozen --file "$recipe/Dockerfile" "$recipe"
image=$(cat "$output/image.id")
docker image inspect "$image" > "$output/tooling-image.json"
"$PY" "$recipe/check-image.py" "$base" "$image" "$output"
echo 'Build/static image checks complete. Return image.id; no live probing.'
# Lock FD closes on exit. No harness up/down or unrelated container cleanup.
