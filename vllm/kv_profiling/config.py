# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone opt-in configuration; no connector is attached here."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

LEVELS = ("off", "l1", "l3")
"""The levels the shared configuration validates."""
ENABLED_LEVELS = frozenset({"l1", "l3"})
"""The levels that attach a recorder: ``off`` attaches nothing. A connector asks
``config.enabled``, never a literal level, so a level added to both sets attaches
everywhere at once."""


def _endpoint(value, name: str) -> tuple[str, int]:
    """A ``host:port`` endpoint with a nonempty host and a valid port."""
    if type(value) is not str or value.count(":") < 1:
        raise ValueError(f"{name} must be host:port")
    host, _, port = value.rpartition(":")
    host = host.strip("[]")
    if not host or any(c.isspace() for c in host):
        raise ValueError(f"{name} needs a nonempty host")
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError(f"{name} needs a port in 1..65535")
    return host, int(port)


@dataclass(frozen=True)
class NetworkProfilerConfig:
    level: str = "off"
    run_id: str | None = None
    output_dir: str | None = None
    shared_dir: str = "/dev/shm"
    ring_records: int = 65536
    id_slots: int = 16384
    id_bytes: int = 8 << 20
    handle_slots: int = 4096
    flush: str = "background"
    transport: str = "unknown"
    clock: bool = True
    clock_port_offset: int = 1000
    clock_samples: int = 7
    clock_max_peers: int = 64
    clock_end_budget_s: float = 2
    batch_records: int = 256
    shutdown_timeout_s: float = 2
    step_window_s: float = 1.0
    """Length of the step rollup windows (0 disables the rollups)."""
    clock_discovery_s: float = 30
    """Mooncake consumer: how long a remote engine's advertised clock endpoint is
    looked up (with backoff, off the request path) before the peer fails."""
    inflight_cap: int | None = None
    """Mooncake: bound on generations open at once (a write batch is one); None is
    1024 or the handle slots, whichever is smaller."""
    context: dict[str, Any] = field(default_factory=dict)
    probes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        import math

        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {', '.join(LEVELS)}")
        if self.level == "l3":
            from .endpoint_probes.config import ProbeConfig

            ProbeConfig(**self.probes)
        elif self.probes:
            raise ValueError("probe settings require level l3")
        if self.flush not in ("end", "background"):
            raise ValueError("flush must be end or background")
        if self.transport not in ("tcp", "rdma", "unknown"):
            raise ValueError("transport must be tcp, rdma or unknown")
        if type(self.clock) is not bool:
            raise ValueError("clock must be a boolean")
        for name, low, high in (
            ("clock_port_offset", 1, 65535),
            ("clock_samples", 1, 32),
            ("clock_max_peers", 1, 256),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in {low}..{high}")
        for name, maximum in (
            ("ring_records", 1 << 20),
            ("id_slots", 65536),
            ("handle_slots", 65536),
        ):
            value = getattr(self, name)
            if (
                type(value) is not int
                or value < 2
                or value > maximum
                or value & (value - 1)
            ):
                raise ValueError(f"{name} must be a bounded power of two")
        if type(self.id_bytes) is not int or not 1 <= self.id_bytes <= 64 << 20:
            raise ValueError("id_bytes must be 1..64 MiB")
        if type(self.batch_records) is not int or not 1 <= self.batch_records <= 4096:
            raise ValueError("batch_records must be 1..4096")
        if (
            type(self.shutdown_timeout_s) not in (int, float)
            or not math.isfinite(self.shutdown_timeout_s)
            or not 0 < self.shutdown_timeout_s <= 10
        ):
            raise ValueError("shutdown_timeout_s must be finite in (0, 10]")
        if (
            type(self.step_window_s) not in (int, float)
            or not math.isfinite(self.step_window_s)
            or not 0 <= self.step_window_s <= 60
        ):
            raise ValueError("step_window_s must be finite in [0, 60]")
        import json

        if (
            type(self.clock_end_budget_s) not in (int, float)
            or not math.isfinite(self.clock_end_budget_s)
            or not 0 < self.clock_end_budget_s <= 10
        ):
            raise ValueError("clock_end_budget_s must be finite in (0, 10]")
        if len(json.dumps(self.context).encode()) > 2048:
            raise ValueError("context exceeds 2 KiB")
        if (
            type(self.clock_discovery_s) not in (int, float)
            or not math.isfinite(self.clock_discovery_s)
            or not 0 < self.clock_discovery_s <= 600
        ):
            raise ValueError("clock_discovery_s must be finite in (0, 600]")
        if self.inflight_cap is not None and (
            type(self.inflight_cap) is not int
            or not 1 <= self.inflight_cap <= self.handle_slots
        ):
            raise ValueError("inflight_cap must be an integer in 1..handle_slots")
        if self.enabled:
            if self.run_id is None or self.output_dir is None:
                raise ValueError("l1/l3 requires run_id and output_dir")
            UUID(self.run_id)
            if not Path(self.shared_dir).is_absolute():
                raise ValueError("shared_dir must be absolute")

    @property
    def effective_inflight_cap(self) -> int:
        """The bound on open generations (the default stays within the handle slots)."""
        if self.inflight_cap is not None:
            return self.inflight_cap
        return min(1024, self.handle_slots)

    @property
    def enabled(self) -> bool:
        """Whether this configuration attaches a recorder (its level is enabled)."""
        return self.level in ENABLED_LEVELS

    @classmethod
    def from_extra_config(cls, extra: dict[str, Any]):
        """Parse the reserved namespace without changing the connector config."""
        values = dict(extra.get("network_profiler", {}))
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown network_profiler keys: {unknown}")
        return cls(**values)


def profiler_config(extra: dict[str, Any] | None, context: dict[str, Any]):
    """The validated configuration of a connector, or None when it attaches nothing.

    ``extra`` is the connector's ``kv_connector_extra_config``. An absent namespace
    means off; a present one is validated even when its level is off, and a level
    that is valid but not enabled attaches nothing. Every connector uses this, so
    the enabled-level rule lives here once.
    """
    if not extra or "network_profiler" not in extra:
        return None
    values = extra["network_profiler"]
    if not isinstance(values, dict):
        raise ValueError("network_profiler must be a dict")
    values = dict(values)
    config = NetworkProfilerConfig.from_extra_config({"network_profiler": values})
    if not config.enabled:
        return None
    values["context"] = {**values.get("context", {}), **context}
    return NetworkProfilerConfig.from_extra_config({"network_profiler": values})
