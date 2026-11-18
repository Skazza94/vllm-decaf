# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned 128-byte records and loss-aware offline identity resolution."""

import json
import struct
from enum import IntEnum
from typing import Any

VERSION = 1
RECORD = struct.Struct("=16Q")
MAX_ID_BYTES = 512


class EventKind(IntEnum):
    DICTIONARY = 1
    REQUEST_BIND = 2
    HANDOFF = 3
    HANDSHAKE_BEGIN = 4
    HANDSHAKE_END = 5
    TRANSFER_BIND = 6
    PREPARED = 7
    POST_BEGIN = 8
    POST_RETURN = 9
    COMPLETE_OBSERVED = 10
    NOTIFY_REQUESTED = 11
    NOTIFY_SEND_RETURN = 12
    NOTIFY_OBSERVED = 13
    SENDERS_DONE = 14
    RELEASE_REQUESTED = 15
    BLOCKS_RELEASED = 16
    FAILED = 17
    TRANSFER_METRICS = 18
    CLOCK_SAMPLE = 19
    MODEL_STEP = 20
    TRANSFER_RETIRE = 21


class Transport(IntEnum):
    UNKNOWN = 0
    TCP = 1
    RDMA = 2
    MIXED = 3


VALUE_NAMES = (
    "bytes",
    "selected_descriptors",
    "merged_descriptors",
    "duration0_ns",
    "duration1_ns",
    "aux0",
    "aux1",
)


def decode_record(raw: bytes) -> dict[str, Any]:
    """Decode in the sink, never in a recording callback."""
    if len(raw) != RECORD.size:
        raise ValueError("invalid binary record size")
    w = RECORD.unpack(raw)
    if w[2] >> 48 != VERSION:
        raise ValueError("unknown binary schema version")
    kind = EventKind((w[2] >> 32) & 0xFFFF)
    base = {"schema": VERSION, "sequence": w[1], "time_ns": w[0]}
    if kind == EventKind.DICTIONARY:
        length = w[6] & 0xFFFFFFFF
        encoding = w[6] >> 32
        if length > 72 or encoding not in (0, 1):
            raise ValueError("invalid dictionary fragment")
        return {
            **base,
            "type": "dictionary_fragment",
            "id": w[3],
            "length": w[4],
            "offset": w[5],
            "encoding": "utf8" if encoding else "bytes",
            "payload": raw[56 : 56 + length].hex(),
        }
    transport = Transport((w[2] >> 24) & 255).name.lower()
    flags = w[2] & 0xFFFF
    return {
        **base,
        "type": "event",
        "kind": kind.name.lower(),
        "context": w[3],
        "request_ref": w[4],
        "handoff_ref": w[5],
        "transfer": w[6],
        "peer_ref": w[7],
        "subject": w[8],
        "transport": {
            "kind": transport,
            "backend": None,
            "provenance": "caller_reported",
            "tcp": None,
            "rdma": None,
        },
        "status": (w[2] >> 16) & 255,
        "unbound_handle": bool(flags & (1 << 15)),
        **{
            name: w[9 + i] if flags & (1 << i) else None
            for i, name in enumerate(VALUE_NAMES)
        },
    }


class DictionaryDecoder:
    """IDs are append-only. An incomplete/conflicting definition never resolves."""

    def __init__(self):
        self._parts: dict[int, dict[int, bytes]] = {}
        self._specs: dict[int, tuple[int, str]] = {}
        self._invalid: set[int] = set()

    def feed(self, row: dict[str, Any]) -> None:
        key = row["id"]
        if type(key) is not int or not 1 <= key <= 65536:
            raise ValueError("dictionary ID out of range")
        try:
            total, offset = row["length"], row["offset"]
            payload = bytes.fromhex(row["payload"])
            spec = total, row["encoding"]
            if (
                type(total) is not int
                or type(offset) is not int
                or not 0 <= total <= MAX_ID_BYTES
                or not 0 <= offset <= total
                or offset + len(payload) > total
                or len(payload) > 72
                or spec[1] not in ("utf8", "bytes")
                or (total and not payload)
                or (key in self._specs and self._specs[key] != spec)
            ):
                self._invalid.add(key)
                return
            self._specs[key] = spec
            parts = self._parts.setdefault(key, {})
            for start, value in parts.items():
                if start == offset and value == payload:
                    return
                if max(start, offset) < min(start + len(value), offset + len(payload)):
                    self._invalid.add(key)
                    return
            parts[offset] = payload
        except (KeyError, TypeError, ValueError):
            self._invalid.add(key)

    def resolve(self, key: int):
        """Return None for missing bytes; never guess a missing identity."""
        if not key or key in self._invalid or key not in self._specs:
            return None
        total, encoding = self._specs[key]
        result = bytearray()
        for start, value in sorted(self._parts[key].items()):
            if start != len(result):
                return None
            result.extend(value)
        if len(result) != total:
            return None
        if encoding == "bytes":
            return {"bytes_hex": result.hex()}
        try:
            return result.decode("utf-8")
        except UnicodeDecodeError:
            return None


def dump_line(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
