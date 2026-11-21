# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit device/port observations; hardware qualification is separate."""

import struct
from pathlib import Path

from .config import component
from .netdev import number, read_small
from .netlink import ObservationError, attribute, attributes, query


def port(spec, root=Path("/sys/class/infiniband")):
    component(spec["device"])
    device = root / spec["device"]
    base = device / "ports" / str(spec["port"])
    identity = {
        key: read_small(device / key) for key in ("node_guid", "sys_image_guid")
    }
    fields, unavailable = [], []
    for directory in ("counters", "hw_counters"):
        try:
            paths = sorted((base / directory).iterdir())
        except OSError as error:
            unavailable.append({"source": directory, "error": str(error)})
            continue
        if len(paths) > 4096:
            raise ObservationError("RDMA counter count exceeds limit")
        for path in paths:
            try:
                value = number(path)
            except (OSError, ValueError, ObservationError) as error:
                unavailable.append({"source": str(path.name), "error": str(error)})
                continue
            unit = "unknown"
            meaning = "device_defined"
            converted = None
            if directory == "counters":
                meaning = "cumulative"
                if path.name in ("port_xmit_data", "port_rcv_data"):
                    unit, converted = "four_octets", value * 4
                elif path.name in ("port_xmit_packets", "port_rcv_packets"):
                    unit = "packets"

            if path.name == "lifespan":
                unit, meaning = "ms", "refresh_configuration"
            fields.append(
                {
                    "name": directory + "/" + path.name,
                    "value": value,
                    "unit": unit,
                    "meaning": meaning,
                    "converted_bytes": converted,
                }
            )
    if not fields:
        raise ObservationError("no readable RDMA counters")
    return {
        **spec,
        "identity": identity,
        "fields": fields,
        "unavailable": unavailable,
        "coverage": {
            family: any(
                field["name"].startswith(family + "/")
                and field["meaning"] != "refresh_configuration"
                for field in fields
            )
            for family in ("counters", "hw_counters")
        },
        "qp_lifetime_proven": False,
    }


def qps(spec, root=Path("/sys/class/infiniband")):
    """Read NLDEV RES_QP_GET only; identifiers are snapshots, not incarnations."""
    index = number(root / spec["device"] / "index")
    request = attribute(1, struct.pack("=I", index))
    request += attribute(3, struct.pack("=I", spec["port"]))
    rows = []
    # UAPI RDMA_NL_NLDEV=5, RES_QP_GET=10; no RAW or mutation commands.
    for body in query(20, (5 << 10) | 10, request):
        attrs = dict(attributes(body))
        if struct.unpack("=I", attrs.get(1, bytes(4)))[0] != index:
            raise ObservationError("QP dump device mismatch")
        for _, entry in attributes(attrs.get(19, b"")):
            values = dict(attributes(entry))
            port_raw = values.get(3, attrs.get(3))
            exported_port = struct.unpack("=I", port_raw)[0] if port_raw else None
            if exported_port is not None and exported_port != spec["port"]:
                continue
            row = {
                **spec,
                "port": exported_port,
                "requested_port": spec["port"],
                "port_association": "exported" if exported_port else "unavailable",
                "identity_scope": "snapshot_only",
            }
            for key, kind, code in (
                ("local_qpn", 21, "I"),
                ("remote_qpn", 22, "I"),
                ("type", 26, "B"),
                ("state", 27, "B"),
                ("pid", 28, "I"),
            ):
                raw = values.get(kind)
                row[key] = struct.unpack("=" + code, raw)[0] if raw else None
            rows.append(row)
            if len(rows) > 1024:
                raise ObservationError("QP observation limit")
    return rows
