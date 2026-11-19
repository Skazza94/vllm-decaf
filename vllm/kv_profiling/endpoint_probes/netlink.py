# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded read-only netlink framing; incomplete dumps are errors, not emptiness."""

import errno
import socket
import struct
import time


class ObservationError(RuntimeError):
    pass


def attributes(data):
    offset = 0
    while offset < len(data):
        if len(data) - offset < 4:
            raise ObservationError("truncated attribute header")
        size, kind = struct.unpack_from("=HH", data, offset)
        if size < 4 or offset + size > len(data):
            raise ObservationError("invalid attribute length")
        yield kind & 0x3FFF, data[offset + 4 : offset + size]
        offset += (size + 3) & ~3
    if offset != len(data):
        raise ObservationError("missing attribute padding")


def attribute(kind, payload):
    size = len(payload) + 4
    return struct.pack("=HH", size, kind) + payload + bytes((-size) % 4)


def messages(data, sequence):
    offset = 0
    while offset < len(data):
        if len(data) - offset < 16:
            raise ObservationError("truncated netlink header")
        size, kind, flags, seq, _ = struct.unpack_from("=IHHII", data, offset)
        if size < 16 or size > len(data) - offset or seq != sequence:
            raise ObservationError("invalid netlink message/sequence")
        if flags & 0x10:
            raise ObservationError("interrupted dump")
        yield kind, data[offset + 16 : offset + size]
        offset += (size + 3) & ~3
    if offset != len(data):
        raise ObservationError("missing message padding")


def query(protocol, kind, payload, *, dump=True, deadline=None):
    """Yield replies within a total deadline and byte/message budget."""
    with socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, protocol) as sock:
        sock.bind((0, 0))
        deadline = deadline if deadline is not None else time.monotonic() + 0.5
        flags = 1 | (0x300 if dump else 0)
        sock.sendto(
            struct.pack(
                "=IHHII", len(payload) + 16, kind, flags, 1, sock.getsockname()[0]
            )
            + payload,
            (0, 0),
        )
        total = count = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ObservationError("netlink deadline")
            sock.settimeout(remaining)
            data, _, recvflags, sender = sock.recvmsg(65536)
            total += len(data)
            count += 1
            if sender[0] or recvflags & socket.MSG_TRUNC:
                raise ObservationError("non-kernel sender or truncated datagram")
            if not data or total > 4 << 20 or count > 4096:
                raise ObservationError("netlink dump limit")
            for response_kind, body in messages(data, 1):
                if response_kind in (2, 3):
                    if len(body) < 4:
                        raise ObservationError("missing netlink status")
                    code = struct.unpack_from("=i", body)[0]
                    if code:
                        raise OSError(
                            -code, errno.errorcode.get(-code, "netlink error")
                        )
                    if response_kind == 3 or not dump:
                        return
                    continue
                if response_kind != kind:
                    raise ObservationError("unexpected netlink response kind")
                yield body
                if not dump:
                    return
