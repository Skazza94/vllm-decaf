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
    # JSONL observer kinds only; never produced by the native L1 ring.
    ENDPOINT_METRICS = 22
    PROBE_STATUS = 23
    PROBE_RESOURCE = 24


class Transport(IntEnum):
    UNKNOWN = 0
    TCP = 1
    RDMA = 2
    MIXED = 3


# TRANSFER_BIND's first value word is the NIXL worker's post pass (``post_pass``):
# decode_record renames it and leaves ``bytes`` unset; other kinds keep these names.
VALUE_NAMES = (
    "bytes",
    "selected_descriptors",
    "merged_descriptors",
    "duration0_ns",
    "duration1_ns",
    "aux0",
    "aux1",
)


class Subject(IntEnum):
    """The event-specific ``subject`` word; meaning depends on the kind."""

    # FAILED: scope of the failure.
    REQUEST = 0
    TRANSFER = 1
    # COMPLETE_OBSERVED, SENDERS_DONE, RELEASE_REQUESTED: who observed it.
    WORKER = 1
    SCHEDULER = 2
    # HANDSHAKE_END: a per-peer summary carrying NIXL's own offset sample.
    OFFSET_SAMPLE = 65535
    # Frontend REQUEST_BIND edges.
    HEADER_TO_BASE = 1
    BASE_TO_PARENT = 2
    PARENT_TO_ITEM = 3
    EXTERNAL_TO_ENGINE = 4
    CHILD_TO_PARENT_ENGINE = 5
    # Mooncake REQUEST_BIND edges: a request that is a member of a write batch
    # (the row's ``transfer`` is the batch; never a request occurrence of its
    # own), and the producer worker's pairing of its request with the shared
    # transfer ID (not a canonical handoff row).
    BATCH_MEMBER = 6
    HANDOFF_IDENTITY = 7


class Failure(IntEnum):
    """FAILED ``status`` codes; ``failure_type`` constants of the connector."""

    UNKNOWN = 1
    HANDSHAKE_FAILED = 2
    HANDSHAKE_SETUP_FAILED = 3
    TRANSFER_SETUP_FAILED = 4
    TRANSFER_FAILED = 5
    TRANSFER_EXCEPTION = 6
    TRANSFER_RELEASE_FAILED = 7
    NOTIFICATION_FAILED = 8
    LEASE_EXPIRED = 9
    LATE_NOTIFICATION = 10
    CLOCK_PEER_TABLE_FULL = 11
    # The observation stopped (profiler closing) while the generation was live:
    # not a connector-reported transport result.
    INTERRUPTED = 12
    # A request's KV was not ready on the producer before the timeout.
    READY_TIMEOUT = 13
    # A batch member rejected before any write (layout or block-count error).
    MEMBER_REJECTED = 14
    # A consumer's remote KV load failed.
    RECEIVE_FAILED = 15
    # The sender coroutine or executor call raised or was cancelled.
    SEND_EXCEPTION = 16
    # A prepared READ (a mixed-memory leg) that was never posted: an earlier leg's
    # preparation or post failed, so it was released unused. Not a transport failure.
    TRANSFER_NOT_POSTED = 17


class CollectorKind(IntEnum):
    """JSONL record kinds written only by the out-of-process collector.

    Never emitted by the native recorder, whose kinds end at ``TRANSFER_RETIRE``
    and whose ABI is unchanged. 22-24 belong to the endpoint-probe records.
    """

    TRANSPORT_METRICS = 25
    TRANSPORT_METRICS_FAILURES = 26
    NIXL_TELEMETRY = 27
    NIXL_TELEMETRY_EVENTS = 28


COLLECTOR_KIND_NAMES = {
    CollectorKind.TRANSPORT_METRICS: "transport_metrics",
    CollectorKind.TRANSPORT_METRICS_FAILURES: "transport_metrics_failures",
    CollectorKind.NIXL_TELEMETRY: "nixl_telemetry",
    CollectorKind.NIXL_TELEMETRY_EVENTS: "nixl_telemetry_events",
}


class IdSource(IntEnum):
    """Frontend header binding ``status``: where the base request ID came from."""

    HEADER = 1
    BODY = 2
    GENERATED = 3
    UNKNOWN = 4


def decode_record(raw: bytes) -> dict[str, Any]:
    """Decode in the sink, never in a recording callback."""
    if len(raw) != RECORD.size:
        raise ValueError("invalid binary record size")
    w = RECORD.unpack(raw)
    if w[2] >> 48 != VERSION:
        raise ValueError("unknown binary schema version")
    kind = EventKind((w[2] >> 32) & 0xFFFF)
    if kind in (
        EventKind.ENDPOINT_METRICS,
        EventKind.PROBE_STATUS,
        EventKind.PROBE_RESOURCE,
    ):
        raise ValueError("endpoint observations require probe JSONL schema")
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
    row = {
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
    if kind == EventKind.TRANSFER_BIND:
        # TRANSFER_BIND carries no byte count (bytes belong to the completion): its
        # first value word is the NIXL worker's post pass, never payload accounting.
        row["post_pass"], row["bytes"] = row["bytes"], None
    return row


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
