# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone opt-in configuration; no connector is attached here."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID


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
    flush: str = "end"
    transport: str = "unknown"
    clock: bool = True
    clock_port_offset: int = 1000
    clock_samples: int = 7
    clock_max_peers: int = 64
    clock_end_budget_s: float = 2
    batch_records: int = 256
    shutdown_timeout_s: float = 2
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        import math

        if self.level not in ("off", "l1"):
            raise ValueError("level must be off or l1")
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
        import json

        if (
            type(self.clock_end_budget_s) not in (int, float)
            or not math.isfinite(self.clock_end_budget_s)
            or not 0 < self.clock_end_budget_s <= 10
        ):
            raise ValueError("clock_end_budget_s must be finite in (0, 10]")
        if len(json.dumps(self.context).encode()) > 2048:
            raise ValueError("context exceeds 2 KiB")
        if self.level == "l1":
            if self.run_id is None or self.output_dir is None:
                raise ValueError("l1 requires run_id and output_dir")
            UUID(self.run_id)
            if not Path(self.shared_dir).is_absolute():
                raise ValueError("shared_dir must be absolute")

    @classmethod
    def from_extra_config(cls, extra: dict[str, Any]):
        """Parse the reserved namespace without changing the connector config."""
        values = dict(extra.get("network_profiler", {}))
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown network_profiler keys: {unknown}")
        return cls(**values)
