# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TCP_INFO gauges and cumulative fields with explicit ABI length gating."""

import ipaddress
import socket
import struct
import time

from .netlink import ObservationError, attribute, attributes, query

# Linux UAPI tcp_info offsets, not an assumed fixed kernel structure size.
FIELDS = {
    "rtt_us": (68, "I", "gauge", "us"),
    "rttvar_us": (72, "I", "gauge", "us"),
    "cwnd_segments": (80, "I", "gauge", "segments"),
    "unacked_segments": (24, "I", "gauge", "segments"),
    "lost_segments": (32, "I", "gauge", "segments"),
    "retrans_segments": (36, "I", "gauge", "segments"),
    "total_retrans_segments": (100, "I", "cumulative", "segments"),
    "pacing_rate_Bps": (104, "Q", "rate", "B/s"),
    "bytes_acked": (120, "Q", "cumulative", "bytes"),
    "bytes_received": (128, "Q", "cumulative", "bytes"),
    "notsent_bytes": (144, "I", "gauge", "bytes"),
    "delivery_rate_Bps": (160, "Q", "rate", "B/s"),
    "bytes_sent": (200, "Q", "cumulative", "bytes"),
    "bytes_retrans": (208, "Q", "cumulative", "bytes"),
}


def tcp_info(data):
    result = {}
    for name, (offset, code, meaning, unit) in FIELDS.items():
        present = offset + struct.calcsize("=" + code) <= len(data)
        value = struct.unpack_from("=" + code, data, offset)[0] if present else None
        result[name] = {
            "value": value,
            "unit": unit,
            "meaning": meaning,
            "available": present,
            "sentinel": "unlimited_or_unset"
            if code == "Q" and meaning == "rate" and value == (1 << 64) - 1
            else None,
        }
    return result


INET_DIAG_SHUTDOWN = 8


def record_kind(state, timer, inode, attrs):
    """A full socket's record, or one of the kernel's compact records, which
    carry no TCP_INFO: a time-wait socket (``inet_twsk_diag_fill``: its
    substate, TIME_WAIT or FIN_WAIT2, timer 3) or a request socket
    (``inet_req_diag_fill``: SYN_RECV, timer 1). Both have inode 0 and no
    TCP_INFO (at most INET_DIAG_MARK, for a reader with CAP_NET_ADMIN); every
    full socket's record carries INET_DIAG_SHUTDOWN."""
    if INET_DIAG_SHUTDOWN in attrs or inode != 0:
        return "full"
    if timer == 3 and state in (5, 6):
        return "time_wait"
    if timer == 1 and state == 3:
        return "request"
    return "full"


def decode_socket(body):
    if len(body) < 72:
        raise ObservationError("short inet_diag_msg")
    family, state = body[:2]
    if family not in (socket.AF_INET, socket.AF_INET6):
        raise ObservationError("unknown inet_diag family")
    width = 4 if family == socket.AF_INET else 16
    src, dst = struct.unpack_from("!HH", body, 4)
    cookie = struct.unpack_from("=II", body, 44)
    attrs = dict(attributes(body[72:]))
    info = attrs.get(2, b"")
    timer, inode = body[2], struct.unpack_from("=I", body, 68)[0]
    kind = record_kind(state, timer, inode, attrs)
    return {
        "family": family,
        "protocol": "tcp",
        "state": state,
        "timer": timer,
        "record": kind,
        "tcp_info_expected": kind == "full",
        "local_ip": socket.inet_ntop(family, body[8 : 8 + width]),
        "local_port": src,
        "peer_ip": socket.inet_ntop(family, body[24 : 24 + width]),
        "peer_port": dst,
        "ifindex": struct.unpack_from("=I", body, 40)[0],
        "cookie": None if cookie == (0xFFFFFFFF, 0xFFFFFFFF) else list(cookie),
        "inode": struct.unpack_from("=I", body, 68)[0],
        "receive_queue_bytes": struct.unpack_from("=I", body, 56)[0],
        "send_queue_bytes": struct.unpack_from("=I", body, 60)[0],
        "tcp_info_length": len(info),
        "fields": tcp_info(info),
    }


def host_filter(family, local, peer):
    program = bytearray()
    hosts = []
    for opcode, address in ((7, local), (8, peer)):
        payload = struct.pack("=BBHi", family, 32 if family == 2 else 128, 0, -1)
        hosts.append((opcode, payload + socket.inet_pton(family, address)))
    length = sum(4 + len(payload) for _, payload in hosts)
    for opcode, payload in hosts:
        size = 4 + len(payload)
        program.extend(struct.pack("=BBH", opcode, size, length - len(program) + 4))
        program.extend(payload)
    return attribute(1, program)


def selected(row, selectors):
    def normalized(value):
        addr = ipaddress.ip_address(value)
        return str(getattr(addr, "ipv4_mapped", None) or addr)

    return (
        row["state"] != 10
        and normalized(row["local_ip"])
        in {normalized(x) for x in selectors["local_ips"]}
        and normalized(row["peer_ip"]) in {normalized(x) for x in selectors["peer_ips"]}
        and all(
            not selectors.get(key) or row[field] in selectors[key]
            for key, field in (
                ("local_ports", "local_port"),
                ("peer_ports", "peer_port"),
            )
        )
        and not {row["local_port"], row["peer_port"]}.intersection(
            selectors.get("exclude_ports", [])
        )
    )


class SocketReader:
    """Compile address filters once; all queries share one tick deadline."""

    def __init__(self, selectors, maximum=1024):
        self.selectors, self.maximum = selectors, maximum
        self.requests = []
        for local in selectors["local_ips"]:
            for peer in selectors["peer_ips"]:
                version = ipaddress.ip_address(local).version
                if version != ipaddress.ip_address(peer).version:
                    continue
                family = socket.AF_INET if version == 4 else socket.AF_INET6
                sockid = bytes(40) + struct.pack("=II", 0xFFFFFFFF, 0xFFFFFFFF)
                payload = (
                    struct.pack("=BBBBI", family, 6, 2, 0, 0xFFF & ~(1 << 10)) + sockid
                )
                self.requests.append(payload + host_filter(family, local, peer))

    def read(self):
        rows = {}
        deadline = time.monotonic() + 0.5
        for payload in self.requests:
            for body in query(4, 20, payload, deadline=deadline):
                row = decode_socket(body)
                if not selected(row, self.selectors):
                    continue
                key = tuple(
                    row[k]
                    for k in (
                        "family",
                        "local_ip",
                        "local_port",
                        "peer_ip",
                        "peer_port",
                    )
                )
                identity = (
                    tuple(row["cookie"])
                    if row["cookie"] is not None
                    else ("inode", row["inode"])
                )
                key += (row["ifindex"], identity)
                rows[key] = row
                if len(rows) > self.maximum:
                    raise ObservationError("socket selector limit exceeded")
        return list(rows.values())


def sockets(selectors, maximum=1024):
    """One read for callers that do not retain a sampler."""
    return SocketReader(selectors, maximum).read()
