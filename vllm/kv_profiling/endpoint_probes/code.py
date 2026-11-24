# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The digest of the observer code that runs, whether from a zipapp, a source
tree or an installed package."""

import hashlib
from importlib import resources


def code_digest() -> str:
    """SHA-256 over this package's modules, by name and content, and over the
    standalone launcher beside it, under the zipapp's name for it
    (``__main__.py``; ``probe.py`` in a source tree or installed package)."""
    package = resources.files(__package__)
    digest = hashlib.sha256()
    for entry in sorted(
        (e for e in package.iterdir() if e.name.endswith(".py")), key=lambda e: e.name
    ):
        digest.update(entry.name.encode() + b"\0" + entry.read_bytes() + b"\0")
    parent = package.parent  # pathlib.Path and zipfile.Path alike
    for name in ("__main__.py", "probe.py"):
        launcher = parent.joinpath(name)
        if launcher.is_file():
            digest.update(b"__main__.py\0" + launcher.read_bytes() + b"\0")
            break
    else:
        digest.update(b"__main__.py\0<none>\0")
    return digest.hexdigest()
