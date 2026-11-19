# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The run's configuration, recorded once per recorder in its metadata.

What was requested (the profiler's own settings, vLLM's parallel, cache and
KV-transfer settings, the transport environment, TENT's configuration) beside
what the transport library reports it set up (NIXL's initialized backends and
their effective options). Collected only when a recorder is opened, never on a
recording path. Values are reported as the library or the environment gives
them, except credentials: secret-named keys, and in URLs the user information
and secret-named query fields, are replaced with ``<redacted>`` at any depth; a
value whose URL boundaries are ambiguous is redacted whole.
"""

import dataclasses
import importlib.metadata
import json
import math
import os
import re
import urllib.parse
from typing import Any

# Transport and connector settings only; other environment variables are not
# the profiler's to record. FI_ is libfabric's (provider, interface).
ENV_PREFIXES = (
    "UCX_",
    "NIXL_",
    "MC_",
    "TENT_",
    "MOONCAKE_",
    "FI_",
    "VLLM_NIXL_",
    "VLLM_MOONCAKE_",
)
TENT_ENV = "MC_TENT_CONF"
MAX_BYTES = 4096
"""The snapshot's serialized bound. The recorder also checks the whole
metadata row against the reader's line limit."""
REDACTED = "<redacted>"
_DISTRIBUTIONS = ("vllm", "nixl", "mooncake-transfer-engine")
_SECRET_KEY = re.compile(
    r"token|secret|passw|credential|auth|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|cookie|session",
    re.IGNORECASE,
)
_SCHEME = re.compile(r"[a-z][a-z0-9+.-]*://", re.IGNORECASE)
# Query names also redacted beyond the key policy: URL-signing conventions.
_SECRET_QUERY = re.compile(r"key|signature|^sig$", re.IGNORECASE)
_PROFILER_PRIVATE = ("context", "run_id", "output_dir")


def _secret_name(name: str) -> bool:
    return bool(_SECRET_KEY.search(name) or _SECRET_QUERY.search(name))


def _redact_url(url: str) -> str:
    """One URL with its user information and secret-named query fields redacted;
    the path (path parameters included) and the fragment stay as they are."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return f"{url.split('://', 1)[0]}://{REDACTED}"
    netloc, changed = parts.netloc, False
    if "@" in netloc:
        netloc, changed = f"{REDACTED}@{netloc.rsplit('@', 1)[1]}", True
    fields = re.split(r"([&;])", parts.query) if parts.query else []
    for i in range(0, len(fields), 2):
        name, equals, _ = fields[i].partition("=")
        if equals and _secret_name(urllib.parse.unquote_plus(name)):
            fields[i], changed = f"{name}={REDACTED}", True
    if not changed:
        return url  # nothing to redact: the value exactly as given
    return urllib.parse.urlunsplit(
        (parts.scheme, netloc, parts.path, "".join(fields), parts.fragment)
    )


def _sanitize_text(text: str) -> str:
    """A setting that holds one URL is parsed as a whole (its boundary is the
    value's end, whatever characters the path holds); a comma-separated endpoint
    list whose every item is a URL is parsed item by item; any other value with
    several URLs has ambiguous boundaries and is redacted whole."""
    count = text.count("://")
    if count == 0:
        return text
    if count == 1:
        found = _SCHEME.search(text)
        if found is None:
            return REDACTED
        return text[: found.start()] + _redact_url(text[found.start() :])
    items = text.split(",")
    if all(_SCHEME.match(item.strip()) and item.count("://") == 1 for item in items):
        return ",".join(
            item[: len(item) - len(item.lstrip())] + _redact_url(item.strip())
            for item in items
        )
    return REDACTED


def sanitize(value: Any) -> Any:
    """A JSON-able copy with credentials redacted at any depth."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        # Strict JSON has no infinity or NaN: keep the value, as text.
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, dict):
        return {
            str(k): REDACTED if _SECRET_KEY.search(str(k)) else sanitize(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [sanitize(v) for v in value]
    return sanitize(str(value))


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _ucx_version() -> str | None:
    """UCX's own version string, only if this process already loaded libucp
    (loading it here would change the process)."""
    try:
        with open("/proc/self/maps") as maps:
            path = next(
                (
                    line.split()[-1]
                    for line in maps
                    if "/libucp.so" in line and line.split()[-1].startswith("/")
                ),
                None,
            )
    except OSError:
        return None
    if path is None:
        return None
    try:
        import ctypes

        mode = getattr(os, "RTLD_NOLOAD", 0) | ctypes.RTLD_LOCAL
        function = ctypes.CDLL(path, mode=mode).ucp_get_version_string
        function.restype = ctypes.c_char_p
        return function().decode()
    except (OSError, AttributeError):
        return None


def environment() -> dict[str, str]:
    """The transport environment; MC_TENT_CONF is parsed into ``tent_config``."""
    return {
        key: "see tent_config" if key == TENT_ENV else value
        for key, value in sorted(os.environ.items())
        if key.startswith(ENV_PREFIXES)
    }


def tent_settings() -> dict[str, Any] | None:
    """TENT's configuration as TENT reads it (a file path or inline JSON), with
    where it came from; None when MC_TENT_CONF is unset."""
    value = os.environ.get(TENT_ENV)
    if not value:
        return None
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import tent_config

    if os.path.exists(value):
        source = {"source": "file", "path": value}
    else:
        source = {"source": "inline"}
    settings = tent_config.load()
    return {**source, "settings": settings if settings is not None else "unreadable"}


def profiler_settings(config) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        k: v
        for k, v in dataclasses.asdict(config).items()
        if k not in _PROFILER_PRIVATE
    }


def vllm_settings(vllm_config) -> dict[str, Any]:
    parallel = vllm_config.parallel_config
    cache = vllm_config.cache_config
    kv = vllm_config.kv_transfer_config
    extra = dict(getattr(kv, "kv_connector_extra_config", None) or {})
    extra.pop("network_profiler", None)  # recorded under "profiler"
    return {
        "parallel": {
            name: getattr(parallel, name, None)
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "data_parallel_index",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        },
        "cache": {
            "block_size": getattr(cache, "block_size", None),
            "cache_dtype": getattr(cache, "cache_dtype", None),
        },
        "kv_transfer": {
            "kv_connector": getattr(kv, "kv_connector", None),
            "kv_role": getattr(kv, "kv_role", None),
            "kv_buffer_device": getattr(kv, "kv_buffer_device", None),
            "extra_config": extra,
        },
    }


def nixl_agent(agent, requested: dict[str, Any]) -> dict[str, Any]:
    """What NIXL set up, as its agent reports it, beside what was requested."""
    initialized = getattr(agent, "backends", None)
    backends = {}
    for name in sorted(initialized) if isinstance(initialized, dict) else ():
        entry = {}
        for key, method in (
            ("options", "get_backend_params"),
            ("plugin_defaults", "get_plugin_params"),
        ):
            try:
                entry[key] = getattr(agent, method)(name)
            except Exception as error:
                # The library's own failure, by type only: its message can echo
                # the options it was given, credentials included.
                entry[key] = None
                entry[f"{key}_error"] = type(error).__name__
        backends[name] = entry
    return {"requested": requested, "initialized": backends}


def _size(value) -> int:
    return len(json.dumps(value, separators=(",", ":")))


def bounded(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The sanitized snapshot within ``MAX_BYTES``: the single largest
    environment value, connector extra-config entry or TENT setting goes first,
    then the next, each one named in ``dropped``, never silently."""
    snapshot = sanitize(snapshot)
    dropped: dict[str, list[str]] = {}
    extra = ((snapshot.get("vllm") or {}).get("kv_transfer") or {}).get("extra_config")
    tent = (snapshot.get("tent_config") or {}).get("settings")
    mappings = {
        name: mapping
        for name, mapping in (
            ("environment", snapshot.get("environment")),
            ("extra_config", extra),
            ("tent_config", tent),
        )
        if isinstance(mapping, dict)
    }
    while _size(snapshot) > MAX_BYTES:
        candidates = [(_size(m[k]), name, k) for name, m in mappings.items() for k in m]
        if not candidates:
            break
        _, name, key = max(candidates)
        mappings[name].pop(key)
        dropped.setdefault(name, []).append(key)
    if dropped:
        snapshot["dropped"] = {k: sorted(v) for k, v in dropped.items()}
    if _size(snapshot) > MAX_BYTES:
        return {"unavailable": f"configuration snapshot over {MAX_BYTES} bytes"}
    return snapshot


def snapshot(
    vllm_config, profiler_config=None, transport: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The bounded, sanitized configuration snapshot of one recorder's process."""
    return bounded(
        {
            "profiler": profiler_settings(profiler_config),
            "vllm": vllm_settings(vllm_config),
            "environment": environment(),
            "tent_config": tent_settings(),
            "libraries": {
                **{name: _version(name) for name in _DISTRIBUTIONS},
                "ucx": _ucx_version(),
            },
            "transport": transport,
        }
    )
