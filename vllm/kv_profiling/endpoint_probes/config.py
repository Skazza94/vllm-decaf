# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit endpoint selectors and bounded observer configuration."""

import ipaddress
import math
from dataclasses import asdict, dataclass, field


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")


def component(value):
    if (
        not isinstance(value, str)
        or not value
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for c in value
        )
        or value in (".", "..")
    ):
        raise ValueError("expected a device/interface name, not a path")


@dataclass(frozen=True)
class ProbeConfig:
    interval_s: float = 1.0
    slow_interval_s: float = 5.0
    tcp: dict = field(default_factory=dict)
    interfaces: list[str] = field(default_factory=list)
    driver_statistics: bool = False
    rdma_ports: list[dict] = field(default_factory=list)
    rdma_required_families: list[str] = field(default_factory=lambda: ["counters"])
    qp: bool = False
    required: list[str] = field(default_factory=list)
    max_sockets: int = 1024
    queue_rows: int = 512
    output_bytes: int = 256 << 20
    ebpf: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("interval_s", "slow_interval_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not 0.1 <= value <= 3600:
                raise ValueError(f"{name} must be in 0.1..3600 seconds")
        integer(self.max_sockets, 1, 1024, "max_sockets")
        integer(self.queue_rows, 1, 512, "queue_rows")
        integer(self.output_bytes, 4096, 1 << 30, "output_bytes")
        if type(self.driver_statistics) is not bool or type(self.qp) is not bool:
            raise ValueError("driver_statistics and qp must be booleans")
        if not isinstance(self.tcp, dict) or set(self.tcp) - {
            "local_ips",
            "peer_ips",
            "local_ports",
            "peer_ports",
            "exclude_ports",
        }:
            raise ValueError("unknown TCP selector")
        for key in ("local_ips", "peer_ips"):
            values = self.tcp.get(key, [])
            if not isinstance(values, list) or len(values) > 8:
                raise ValueError("IP selectors must be lists of at most eight")
            for value in values:
                if not isinstance(value, str):
                    raise ValueError("IP selector must be a string")
                ipaddress.ip_address(value)
            if self.tcp and not values:
                raise ValueError("TCP requires local_ips and peer_ips allowlists")
        for key in ("local_ports", "peer_ports", "exclude_ports"):
            values = self.tcp.get(key, [])
            if not isinstance(values, list) or len(values) > 64:
                raise ValueError("port selectors must be lists of at most 64")
            for port in values:
                integer(port, 1, 65535, key)
        for values in (self.interfaces, self.rdma_ports):
            if not isinstance(values, list) or len(values) > 8:
                raise ValueError("at most eight interfaces or RDMA ports")
        for name in self.interfaces:
            component(name)
        for spec in self.rdma_ports:
            if not isinstance(spec, dict) or set(spec) != {"device", "port"}:
                raise ValueError("RDMA port requires device and port")
            component(spec["device"])
            integer(spec["port"], 1, 255, "RDMA port")
        if not isinstance(self.ebpf, dict):
            raise ValueError("ebpf must be a mapping")
        if self.ebpf.get("enabled", False) is False:
            if set(self.ebpf) - {"enabled"}:
                raise ValueError("disabled ebpf accepts only enabled=false")
        else:
            from .ebpf import Settings

            Settings(**self.ebpf)
        enabled = set()
        if self.ebpf.get("enabled"):
            enabled.add("ebpf")
        if self.tcp:
            enabled.add("tcp")
        if self.interfaces:
            enabled.add("netdev")
            if self.driver_statistics:
                enabled.add("driver")
        if self.rdma_ports:
            enabled.add("rdma")
            if self.qp:
                enabled.add("qp")
        if not isinstance(self.required, list) or set(self.required) - enabled:
            raise ValueError("required sources must be explicitly enabled")
        if (
            not isinstance(self.rdma_required_families, list)
            or not self.rdma_required_families
            or set(self.rdma_required_families) - {"counters", "hw_counters"}
        ):
            raise ValueError("RDMA required families must name counters or hw_counters")
        if self.qp and not self.rdma_ports:
            raise ValueError("QP queries require explicit RDMA device/ports")
        if self.driver_statistics and not self.interfaces:
            raise ValueError("driver statistics require explicit interfaces")


def settings_dict(config):
    """Preserve the pre-eBPF identity of disabled configurations and saved runs."""
    result = asdict(config)
    if not config.ebpf.get("enabled", False):
        result.pop("ebpf")
    return result
