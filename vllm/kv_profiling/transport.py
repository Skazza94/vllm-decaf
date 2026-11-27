# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV transport each backend's configuration allows, derived per recorder.

A backend's configuration constrains its transport; it does not say which one
the backend then selects. Each entry therefore names the ``candidates`` its
configuration allows (``tcp``, ``rdma``; None when unconstrained, empty when
neither), the ``source`` setting it was read from and the ``reason`` it is not
more. Only evidence of the transport in use resolves one (see the summary).
"""

import os
from typing import Any

KINDS = ("tcp", "rdma")

_UCX_NETWORK = {
    "tcp": "tcp",
    "ib": "rdma",
    "rc": "rdma",
    "rc_v": "rdma",
    "rc_x": "rdma",
    "rc_verbs": "rdma",
    "rc_mlx5": "rdma",
    "dc": "rdma",
    "dc_x": "rdma",
    "dc_mlx5": "rdma",
    "ud": "rdma",
    "ud_v": "rdma",
    "ud_x": "rdma",
    "ud_verbs": "rdma",
    "ud_mlx5": "rdma",
}
_UCX_LOCAL = {"self", "sm", "shm", "mm", "posix", "sysv", "cma", "knem", "xpmem"}
_UCX_LOCAL_PREFIXES = ("cuda", "rocm", "gdr", "ze")
# libfabric core providers by wire transport: None is neither TCP nor RDMA
# (udp). Any other provider is unclassified: psm3 or efa alone need not be
# hardware RDMA, so it leaves the candidates unconstrained.
_FI_PROVIDERS = {"tcp": "tcp", "sockets": "tcp", "verbs": "rdma", "udp": None}
# TENT's network transports, their kind, and whether TENT enables one its
# configuration leaves unset (hp_tcp is opt-in; rdma needs an RDMA NIC too).
_TENT_NETWORK = {"rdma": ("rdma", True), "tcp": ("tcp", True), "hp_tcp": ("tcp", False)}
# A derived value or reason is recorded within these bounds, redacted as the
# configuration snapshot is: the recorder's metadata must stay openable.
MAX_TEXT = 256
MAX_NAMES = 8


def _bounded(value):
    from .snapshot import sanitize

    value = sanitize(value)
    if isinstance(value, str) and len(value) > MAX_TEXT:
        return value[:MAX_TEXT] + f"... ({len(value)} characters)"
    if isinstance(value, dict):
        return {key: _bounded(item) for key, item in list(value.items())[:MAX_NAMES]}
    if isinstance(value, list):
        kept = [_bounded(item) for item in value[:MAX_NAMES]]
        return kept + ([f"... ({len(value)} items)"] if len(value) > MAX_NAMES else [])
    return value


def _names(names) -> str:
    shown = sorted(names)[:MAX_NAMES]
    more = len(names) - len(shown)
    return f"{shown}" + (f" and {more} more" if more > 0 else "")


def _entry(backend, candidates, reason, name, value, **extra) -> dict[str, Any]:
    return {
        "backend": backend,
        "candidates": None if candidates is None else sorted(candidates),
        "reason": _bounded(reason),
        "source": {"name": name, "value": _bounded(value)},
        **{key: _bounded(item) for key, item in extra.items()},
    }


def ucx_lane_class(transport: str) -> str:
    """A UCX lane's transport (``tcp``, ``rc_mlx5``, ``sysv``, …) as ``tcp``,
    ``rdma``, ``local`` (shared memory, self, devices) or ``unclassified``."""
    name = transport.split(":")[0]
    if name in _UCX_NETWORK:
        return _UCX_NETWORK[name]
    if name.startswith(("rc_", "dc_", "ud_")):
        return "rdma"
    if name in _UCX_LOCAL or name.startswith(_UCX_LOCAL_PREFIXES):
        return "local"
    return "unclassified"


def ucx(environment) -> dict[str, Any]:
    """UCX selects among the transports UCX_TLS allows, at runtime."""
    value = environment.get("UCX_TLS")
    runtime = "UCX selects among the allowed transports at runtime"
    if not value or value.strip() == "all":
        return _entry("UCX", None, f"UCX_TLS unset or all; {runtime}",
                      "UCX_TLS", value)  # fmt: skip
    if value.startswith("^"):
        return _entry("UCX", None, f"UCX_TLS excludes transports; {runtime}",
                      "UCX_TLS", value)  # fmt: skip
    kinds, unclassified = set(), []
    for token in filter(None, (t.strip() for t in value.split(","))):
        name = token.split(":")[0]
        if name in _UCX_NETWORK:
            kinds.add(_UCX_NETWORK[name])
        elif name not in _UCX_LOCAL and not name.startswith(_UCX_LOCAL_PREFIXES):
            unclassified.append(name)
    if unclassified:
        return _entry("UCX", None, f"UCX_TLS names unclassified transports "
                      f"{_names(set(unclassified))}; {runtime}", "UCX_TLS",
                      value)  # fmt: skip
    return _entry("UCX", kinds, runtime, "UCX_TLS", value)


def libfabric(environment) -> dict[str, Any]:
    """FI_PROVIDER filters the providers libfabric may select."""
    value = environment.get("FI_PROVIDER")
    if not value:
        return _entry("LIBFABRIC", None, "FI_PROVIDER unset: libfabric selects "
                      "the provider", "FI_PROVIDER", value)  # fmt: skip
    if value.startswith("^"):
        return _entry("LIBFABRIC", None, "FI_PROVIDER excludes providers: "
                      "libfabric selects among the rest", "FI_PROVIDER",
                      value)  # fmt: skip
    providers = [p.strip().split(";")[0] for p in value.split(",") if p.strip()]
    unclassified = {p for p in providers if p not in _FI_PROVIDERS}
    reason = (
        "FI_PROVIDER filters the providers; the provider libfabric selected is "
        "not recorded"
    )
    kinds = {_FI_PROVIDERS[p] for p in providers if p in _FI_PROVIDERS} - {None}
    if unclassified:
        reason += f"; the wire transport of {_names(unclassified)} is unclassified"
        kinds = None
    return _entry("LIBFABRIC", kinds, reason, "FI_PROVIDER", value,
                  providers=",".join(providers))  # fmt: skip


def _get(settings, key, accept=None):
    """TENT's lookup: the '/'-separated key as a nested path, else the exact
    flat key; a null is absent. With ``accept`` (Config::get<T>), a nested
    value of the wrong type falls back to the flat key too, then to None
    (the caller's default)."""
    node = settings
    for segment in key.split("/"):
        if not isinstance(node, dict) or segment not in node:
            node = None
            break
        node = node[segment]
    if node is not None and (accept is None or accept(node)):
        return node
    flat = settings.get(key)
    if flat is not None and (accept is None or accept(flat)):
        return flat
    return None


def _is_bool(value) -> bool:
    return isinstance(value, bool)


def _tent_enabled(settings):
    """Each network transport's enable as TENT reads it. tcp and rdma use
    Config::get, which falls back to the default on a type mismatch; hp_tcp's
    own parser reads only the ``transports/hp_tcp`` object (nested or that
    exact flat key), and refuses a non-boolean enable (its type is kept)."""
    flags = {}
    for name in ("tcp", "rdma"):
        flags[name] = _get(settings, f"transports/{name}/enable", _is_bool)
    hp_tcp = _get(settings, "transports/hp_tcp")
    if hp_tcp is None:
        flags["hp_tcp"] = None
    elif not isinstance(hp_tcp, dict):
        flags["hp_tcp"] = f"invalid: {type(hp_tcp).__name__}"
    elif "enable" not in hp_tcp:
        flags["hp_tcp"] = None  # absent: off by default
    else:
        enable = hp_tcp["enable"]  # present: a boolean, else invalid (null too)
        flags["hp_tcp"] = (
            enable if isinstance(enable, bool)
            else f"invalid: {type(enable).__name__}"
        )  # fmt: skip
    return flags


def tent(backend, tent_settings) -> dict[str, Any]:
    """TENT enables its transports from MC_TENT_CONF, its defaults (tcp and
    rdma on, hp_tcp off) and the devices it finds (rdma needs an RDMA NIC)."""
    reason = (
        "TENT enables transports from MC_TENT_CONF and the devices it finds; "
        "its per-transport byte counters are the evidence"
    )
    if tent_settings is None:
        return _entry(backend, None, "MC_TENT_CONF unset: " + reason,
                      "MC_TENT_CONF", None, engine="tent")  # fmt: skip
    settings = tent_settings.get("settings")
    if not isinstance(settings, dict):
        return _entry(backend, None, "MC_TENT_CONF unreadable: " + reason,
                      "MC_TENT_CONF", None, engine="tent")  # fmt: skip
    flags = _tent_enabled(settings)
    if isinstance(flags["hp_tcp"], str):
        return _entry(backend, None, "an invalid transports/hp_tcp in "
                      "MC_TENT_CONF: " + reason, "MC_TENT_CONF",
                      {f"{name}.enable": flags[name] for name in sorted(flags)},
                      engine="tent")  # fmt: skip
    kinds = {
        _TENT_NETWORK[name][0]
        for name, enable in flags.items()
        if (enable if enable is not None else _TENT_NETWORK[name][1])
    }
    return _entry(backend, kinds, reason, "MC_TENT_CONF",
                  {f"{name}.enable": flags[name] for name in sorted(flags)},
                  engine="tent")  # fmt: skip


def _uses_tent(environment) -> bool:
    """The Transfer Engine selects TENT when either variable is present, whatever
    its value (empty and 0 included)."""
    return "MC_USE_TENT" in environment or "MC_USE_TEV1" in environment


def classic(backend, environment, extra=None) -> dict[str, Any]:
    """The classic engine discovers its transport (TCP even under the default
    ``mooncake_protocol`` rdma) unless MC_FORCE_TCP forces TCP."""
    protocol = (extra or {}).get("mooncake_protocol")
    if "MC_FORCE_TCP" in environment:
        return _entry(backend, {"tcp"}, "MC_FORCE_TCP forces the classic Transfer "
                      "Engine onto TCP; it records no per-transport counters",
                      "MC_FORCE_TCP", environment["MC_FORCE_TCP"],
                      engine="classic", protocol=protocol)  # fmt: skip
    return _entry(backend, None, "the classic Transfer Engine discovers its "
                  "transport whatever its protocol; it records no per-transport "
                  "counters", "mooncake_protocol", protocol,
                  engine="classic")  # fmt: skip


def mooncake_engine(environment, tent_settings, extra) -> dict[str, Any]:
    """The Mooncake connector's engine: TENT when selected, else classic."""
    if _uses_tent(environment):
        return tent("Mooncake", tent_settings)
    return classic("Mooncake", environment, extra)


def derive(
    backends, connector, environment=None, tent_settings=None, extra=None
) -> list[dict[str, Any]]:
    """One entry per backend, from this process's configuration as the backend
    starts with it (call after the connector has bound its transport)."""
    environment = os.environ if environment is None else environment
    if connector == "mooncake":
        return [mooncake_engine(environment, tent_settings, extra)]
    entries = []
    for backend in backends or []:
        if backend == "UCX":
            entries.append(ucx(environment))
        elif backend == "LIBFABRIC":
            entries.append(libfabric(environment))
        elif backend == "Mooncake" and _uses_tent(environment):
            entries.append(tent("Mooncake", tent_settings))
        elif backend == "Mooncake":
            entries.append(classic(backend, environment))
        else:
            entries.append(_entry(backend, None, "no transport derivation for "
                                  "this backend", None, None))  # fmt: skip
    return entries


def combined(entries) -> str:
    """The recorder's transport kind: a kind only when every entry resolves to
    it (``kind``); entries that resolve to different kinds are ``mixed``."""
    kinds = {entry.get("kind", "unknown") for entry in entries or []}
    if len(kinds) == 1:
        return kinds.pop()
    return "mixed" if len(kinds - {"unknown"}) > 1 else "unknown"
