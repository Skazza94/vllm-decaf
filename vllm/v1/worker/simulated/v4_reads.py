# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4's pulled rows under the byte checks: the one READ a request
needs and what its bytes are.

V4's cache groups are packed into block rows that every TP rank registers
alike, so a decode rank pulls whole rows from one producer rank (the native
rule, `owner_rank`) whatever the two TP sizes. The generic coverage compares
ownership intervals and accepts several handles or an unknown producer; this
module states the expectation before the read from the request's destination
allocation, the registered row geometry and the handoff's engine, and
compares the one completed record exactly (`read_problems`). It also splits
a row's mismatched bytes among the layers' payloads, their alignment and the
row's tail (`RowAccount`), for the draft layer apart from the target's."""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

Piece = tuple[int, int, int, int]
"""(region, block, offset in the block, length): where a descriptor's bytes land."""


def owner_rank(tp_rank: int, local_tp: int, remote_tp: int) -> int:
    """The producer rank NIXL reads a replicated (MLA) cache from
    (`compute_tp_mapping`): the decode rank's own place among the producer's."""
    return tp_rank * remote_tp // local_tp


def held_blocks(
    tokens: int, block_size: int, window: int | None = None, retained: int = 0
) -> int:
    """The blocks of a request of `tokens` computed tokens one group keeps: all
    of them, or a sliding window's (the manager's own rule, tokens before
    `tokens - window + 1 - retained` are skipped)."""
    last = (tokens - 1) // block_size
    if window is None:
        return last + 1
    first = max(0, tokens - window + 1 - retained) // block_size
    return last - first + 1


def expected_rows(
    held: Mapping[int, Sequence[int]],
    regions: Mapping[int, Sequence[int]],
    region_bytes: Mapping[int, int],
) -> list[Piece]:
    """The whole rows a request's READ must deliver: each group's held
    (non-null) blocks, in each region that group's layers sit in, from offset
    zero, a region's registered block length."""
    return [
        (region, block, 0, region_bytes[region])
        for group, blocks in held.items()
        for region in regions[group]
        for block in blocks
    ]


def read_problems(
    engine: str | None,
    rank: int,
    expected: Sequence[Piece],
    records: Sequence,
) -> list[str]:
    """What the request's READ records get wrong against the expectation: for
    a request that pulls rows, exactly one record, of the handoff's engine and
    the owner rank, resolved, completed, whose pieces are the expected rows
    once each; for a request that pulls none, no record with pieces."""
    if not expected:
        return [
            f"a READ from rank {r.rank} for a request that pulls no rows"
            for r in records
            if r.pieces
        ]
    problems = []
    if engine is None:
        problems.append("the handoff names no producer engine")
    if len(records) != 1:
        problems.append(f"{len(records)} READ records, exactly one expected")
    for r in records:
        if r.engine is None or r.engine != engine:
            problems.append(f"a READ of engine {r.engine}, not {engine}")
        if r.rank is None or r.rank != rank:
            problems.append(f"a READ from rank {r.rank}, not rank {rank}")
        if r.pieces is None or any(p[0] < 0 for p in r.pieces):
            problems.append("a READ whose descriptors did not resolve to a region")
        if not r.done:
            problems.append("a READ that had not completed")
    got = Counter(tuple(p) for r in records for p in r.pieces or ())
    want = Counter(tuple(p) for p in expected)
    for label, delta in (
        ("missing", want - got),
        ("surplus or duplicated", got - want),
    ):
        if delta:
            shown = sorted(delta.items())[:3]
            problems.append(f"{sum(delta.values())} {label} rows, e.g. {shown}")
    return problems


@dataclass(frozen=True)
class Member:
    """One layer's registered slice of a packed row."""

    layer: str
    offset: int
    page: int
    real: int
    draft: bool = False


DRAFT_LAYER = re.compile(r"model\.layers\.(\d+)\.")


def is_draft(layer: str, target_layers: int) -> bool:
    match = DRAFT_LAYER.match(layer)
    return match is not None and int(match.group(1)) >= target_layers


@dataclass
class RowAccount:
    """A request's checked rows, their mismatched bytes classified: a layer's
    payload, its alignment (the registered slice past its real page), or the
    row's tail (bytes of no layer), the draft's apart from the target's."""

    members: Sequence[Sequence[Member]]
    """Each transfer group's layers, by offset."""
    rows: list[tuple[int, int, int, int, np.ndarray]] = field(default_factory=list)

    def add(
        self, group: int, region: int, block: int, row_bytes: int, wrong: np.ndarray
    ) -> None:
        """One checked row of `group`: `wrong` the offsets of its bad bytes."""
        self.rows.append((group, region, block, row_bytes, wrong))

    def report(self, pieces: Sequence[Piece]) -> dict:
        rows_bytes = tail = 0
        overfull = 0
        slices = {"target": 0, "draft": 0}
        bad = {
            "target_payload": 0,
            "target_alignment": 0,
            "draft_payload": 0,
            "draft_alignment": 0,
            "tail": 0,
        }
        draft_rows = draft_read = 0
        draft_layers: set[str] = set()
        for group, region, block, row_bytes, wrong in self.rows:
            rows_bytes += row_bytes
            members = self.members[group]
            starts = np.array([m.offset for m in members], dtype=np.int64)
            row_tail = row_bytes - sum(m.page for m in members)
            overfull += row_tail < 0
            tail += row_tail
            for m in members:
                kind = "draft" if m.draft else "target"
                slices[kind] += m.page
                if m.draft:
                    draft_layers.add(m.layer)
                    draft_rows += 1
                    draft_read += _covered(
                        pieces, region, block, m.offset, m.offset + m.page
                    )
            if not len(wrong):
                continue
            index = np.searchsorted(starts, wrong, side="right") - 1
            for position, offset in zip(index.tolist(), wrong.tolist(), strict=True):
                m = members[position] if position >= 0 else None
                if m is None or offset >= m.offset + m.page:
                    bad["tail"] += 1
                    continue
                kind = "draft" if m.draft else "target"
                part = "payload" if offset < m.offset + m.real else "alignment"
                bad[f"{kind}_{part}"] += 1
        out = {
            "rows": len(self.rows),
            "bytes": rows_bytes,
            "target_slice_bytes": slices["target"],
            "draft_slice_bytes": slices["draft"],
            "tail_bytes": tail,
            "mismatched": bad,
            "reconciled": not overfull
            and slices["target"] + slices["draft"] + tail == rows_bytes,
        }
        if draft_layers:
            payload = sum(
                m.real
                for group, *_ in self.rows
                for m in self.members[group]
                if m.draft
            )
            out["draft"] = {
                "layers": sorted(draft_layers),
                "registered_slice_bytes": slices["draft"],
                "payload_bytes": payload,
                "alignment_bytes": slices["draft"] - payload,
                "read_bytes": draft_read,
                "mismatched_payload_bytes": bad["draft_payload"],
                "mismatched_alignment_bytes": bad["draft_alignment"],
            }
        return out


def _covered(pieces: Sequence[Piece], region: int, block: int, lo: int, hi: int) -> int:
    """Bytes of [lo, hi) of one block the pieces deliver."""
    return sum(
        max(0, min(hi, offset + length) - max(lo, offset))
        for r, b, offset, length in pieces
        if r == region and b == block
    )


def build_members(
    names_by_group: Sequence[Sequence[str]],
    packed: Mapping[str, tuple[int, int]],
    real_pages: Mapping[str, int],
    target_layers: int,
) -> list[list[Member]]:
    """Each group's layers' registered slices (`packed`: layer -> (offset in
    the row, registered bytes), as the connector advertises them), by offset;
    a layer of the draft is one past the target's last. Refused when two
    slices of a group overlap."""
    groups = []
    for names in names_by_group:
        members = sorted(
            (
                Member(
                    name,
                    packed[name][0],
                    packed[name][1],
                    real_pages[name],
                    is_draft(name, target_layers),
                )
                for name in names
                if name in packed
            ),
            key=lambda m: m.offset,
        )
        for before, after in zip(members, members[1:]):
            if before.offset + before.page > after.offset:
                raise ValueError(
                    f"packed slices of {before.layer} and {after.layer} overlap"
                )
        groups.append(members)
    return groups
