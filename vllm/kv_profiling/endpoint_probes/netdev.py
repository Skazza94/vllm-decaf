# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only sysfs and legacy ethtool statistics; never change device settings."""

import ctypes
import fcntl
import socket
import struct
from pathlib import Path

from .config import component
from .netlink import ObservationError


def read_small(path, limit=4096):
    with Path(path).open("rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ObservationError(f"oversize sysfs value: {path}")
    return value.decode().strip()


def number(path):
    value = int(read_small(path))
    if value < 0:
        raise ObservationError(f"negative counter: {path}")
    return value


def interface(name, root=Path("/sys/class/net")):
    component(name)
    base = root / name
    index = number(base / "ifindex")
    values = {}
    for key in (
        "rx_bytes",
        "tx_bytes",
        "rx_packets",
        "tx_packets",
        "rx_errors",
        "tx_errors",
        "rx_dropped",
        "tx_dropped",
    ):
        values[key] = {
            "value": number(base / "statistics" / key),
            "meaning": "cumulative",
            "unit": "bytes"
            if key.endswith("bytes")
            else "packets"
            if key.endswith(("packets", "dropped"))
            else "events",
        }
    if number(base / "ifindex") != index:
        raise ObservationError("interface changed during read")
    return {"name": name, "ifindex": index, "fields": values}


def driver(name):
    component(name)
    encoded = name.encode()
    if len(encoded) >= 16:
        raise ValueError("interface name exceeds IFNAMSIZ")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:

        def call(command, size, prefix=b""):
            buf = ctypes.create_string_buffer(size)
            buf.raw = (struct.pack("=I", command) + prefix).ljust(size, b"\0")
            # P is a native pointer; Linux ifreq has a 24-byte union on LP64.
            request = struct.pack("@16sP", encoded, ctypes.addressof(buf)).ljust(
                40, b"\0"
            )
            fcntl.ioctl(sock.fileno(), 0x8946, request)
            return buf.raw

        info = call(3, 196)
        count = struct.unpack_from("=I", info, 180)[0]
        if not 0 < count <= 4096:
            raise ObservationError(
                "driver statistic count unavailable or exceeds limit"
            )
        names = call(0x1B, 12 + count * 32, struct.pack("=II", 1, count))
        values = call(0x1D, 8 + count * 8, struct.pack("=I", count))
        if (
            struct.unpack_from("=I", names, 8)[0] != count
            or struct.unpack_from("=I", values, 4)[0] != count
            or struct.unpack_from("=I", call(3, 196), 180)[0] != count
        ):
            raise ObservationError("driver statistics layout changed")
        result = []
        for i in range(count):
            result.append(
                {
                    "name": names[12 + 32 * i : 44 + 32 * i]
                    .split(b"\0")[0]
                    .decode(errors="replace"),
                    "value": struct.unpack_from("=Q", values, 8 + 8 * i)[0],
                    "meaning": "driver_defined",
                    "unit": "unknown",
                }
            )
        return {"name": name, "fields": result}
