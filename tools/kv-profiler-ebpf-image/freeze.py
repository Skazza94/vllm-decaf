"""Verify/download the exact build inputs; never install or build anything."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, download=False):
    lock = json.loads((root / "inputs.lock.json").read_text())
    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    for item in lock["packages"] + lock["sources"]:
        path = assets / item["filename"]
        if not path.exists() and download:
            temporary = path.with_suffix(path.suffix + ".partial")
            with (
                urllib.request.urlopen(item["url"], timeout=180) as response,
                temporary.open("wb") as output,
            ):
                while block := response.read(1024 * 1024):
                    output.write(block)
            if sha(temporary) != item["sha256"]:
                raise ValueError(f"download hash mismatch: {path.name}")
            temporary.replace(path)
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError(f"input mismatch: {path.name}")
    expected = {item["filename"] for item in lock["packages"] + lock["sources"]}
    actual = {path.name for path in assets.iterdir()}
    if actual != expected:
        raise ValueError(f"unexpected/missing asset names: {actual ^ expected}")
    for line in (root / "recipe.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if sha(root / name) != digest:
            raise ValueError(f"recipe mismatch: {name}")
    print("PASS: frozen recipe, 20 debs and two source archives verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    verify(args.root.resolve(), args.download)
