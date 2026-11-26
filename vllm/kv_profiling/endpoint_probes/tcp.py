# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TCP_INFO gauges and cumulative fields with explicit ABI length gating."""

import socket
import struct

from .netlink import ObservationError, attributes

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
