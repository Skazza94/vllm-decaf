"""Main's post-build static checks using only never-started containers."""

import hashlib
import json
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

base, built, output = sys.argv[1:]
output = Path(output)


def inspect(image):
    return json.loads(subprocess.check_output(["docker", "image", "inspect", image]))[0]


before, after = inspect(base), inspect(built)
assert before["Id"] == base and after["Id"] == built
assert before["RootFS"]["Layers"] == after["RootFS"]["Layers"][:-1]
assert len(after["RootFS"]["Layers"]) == len(before["RootFS"]["Layers"]) + 1
for image in (before, after):
    assert image["Architecture"] == "amd64" and image["Os"] == "linux"
configs = []
for image in (before, after):
    config = dict(image["Config"])
    config.pop("Labels", None)
    config.pop("Image", None)  # Legacy-builder parent bookkeeping, not runtime config.
    configs.append(config)
assert configs[0] == configs[1], "ordinary runtime config changed"
for key, value in (before["Config"].get("Labels") or {}).items():
    assert after["Config"]["Labels"][key] == value
# Final stage is exactly COPY to the private prefix plus a metadata label.
# No ordinary global library path/cache/config file is modified by this recipe.
paths = [
    "/opt/nixl",
    "/opt/ucx",
    "/opt/libfabric",
    "/opt/mooncake",
    "/opt/venv/lib/python3.12/site-packages/nixl",
    "/opt/venv/lib/python3.12/site-packages/nixl_cu12",
    "/opt/venv/lib/python3.12/site-packages/nixl_cu13",
    "/opt/venv/lib/python3.12/site-packages/nixl_rocm",
    "/usr/lib/x86_64-linux-gnu/libibverbs.so.1",
    "/usr/lib/x86_64-linux-gnu/libibverbs",
    "/usr/lib/x86_64-linux-gnu/libmlx5.so.1",
    "/usr/lib/x86_64-linux-gnu/librdmacm.so.1",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/var/lib/dpkg/status",
]


def manifest(path):
    entries = {}
    for child in [path] + sorted(path.rglob("*")):
        name = "." if child == path else str(child.relative_to(path))
        if child.is_symlink():
            entries[name] = {"symlink": str(child.readlink())}
        elif child.is_file():
            entries[name] = {
                "sha256": hashlib.sha256(child.read_bytes()).hexdigest(),
                "mode": stat.S_IMODE(child.stat().st_mode),
            }
    return entries


records = []
for image in (base, built):
    name = "agent1-ebpf-static-" + uuid.uuid4().hex[:16]
    container = subprocess.check_output(
        [
            "docker",
            "create",
            "--name",
            name,
            "--entrypoint",
            "sleep",
            image,
            "infinity",
        ],
        text=True,
    ).strip()
    record = {"image": image, "paths": {}}
    try:
        description = json.loads(
            subprocess.check_output(["docker", "inspect", container])
        )[0]
        assert not description["State"]["Running"] and description["State"][
            "StartedAt"
        ].startswith("0001")
        with tempfile.TemporaryDirectory(prefix="agent1-static-") as tmp:
            for index, source in enumerate(paths):
                dest = Path(tmp) / str(index)
                # -L captures the bytes at a top-level shared-library symlink.
                result = subprocess.run(
                    ["docker", "cp", "-L", f"{container}:{source}", str(dest)],
                    capture_output=True,
                    text=True,
                )
                record["paths"][source] = (
                    manifest(dest)
                    if result.returncode == 0
                    else {
                        "absent": True,
                        "copy_error": result.stderr.replace(container, "<container>"),
                    }
                )
            if image == built:
                target = output / "tool-runtime-closure.json"
                subprocess.run(
                    [
                        "docker",
                        "cp",
                        f"{container}:/opt/kv-profiler-tools/manifest/elf-closure.json",
                        str(target),
                    ],
                    check=True,
                )
                assert json.loads(target.read_text()), "missing tooling ELF closure"
    finally:
        subprocess.run(["docker", "rm", container], check=True)
    records.append(record)
    (
        output / ("base-files.json" if image == base else "tooling-files.json")
    ).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
assert records[0]["paths"] == records[1]["paths"], (
    "transport/provider/loader files changed"
)
required = paths[:6] + [paths[8], paths[12], paths[-1]]
assert all("absent" not in records[0]["paths"][p] for p in required), (
    "required audit tree absent"
)
report = {
    "status": "PASS",
    "base_image": base,
    "tooling_image": built,
    "scope": (
        "static file/config/search-path invariance; no runtime loading or attachment"
    ),
    "ordinary_resolution": (
        "same config, protected trees, loader cache/config "
        "and pinned base layers; only isolated prefix added"
    ),
    "compile_skeleton": "NOT RUN: next agent checkpoint after image identity returned",
}
(output / "image-checks.json").write_text(json.dumps(report, indent=2) + "\n")
print("PASS: transport/provider and resolver inputs unchanged; tooling prefix only")
