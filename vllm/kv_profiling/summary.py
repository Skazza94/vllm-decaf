# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline per-transfer and per-request analysis summary of profiler captures.

Reads the JSONL files the sink writes (the same loader contract as the
Perfetto exporter) and reports phase timings from the recorded control
points. Durations are computed only inside one process's recorded clock;
cross-process spans require saved clock calibrations and reuse the
exporter's projection rules, uncertainty included.

Nothing here touches serving, the connector or the transports: this is an
offline report, and every field comes from the records. A metric the
records do not carry is reported in ``not_available``, never invented.
"""

import argparse
import json
import math
from pathlib import Path

from . import timeline
from .clock import ClockGroup, ClockMapping
from .perfetto import _clock_aliases, _load
from .schema import Failure, Subject

# The loader yields kind names as lower-case strings (the JSONL rows); these
# sets are checked against schema.EventKind in the tests.
_PREPARED = frozenset(("transfer_bind", "prepared"))
_POST_BEGIN = frozenset(("post_begin",))
_POST_RETURN = frozenset(("post_return",))
_COMPLETE = frozenset(("complete_observed",))
_RETIRE = "transfer_retire"
_FAILED = "failed"
_BEGIN = "handshake_begin"
_END = "handshake_end"
# Request-scoped rows, keyed by a resolved request id.
_REQUEST_KINDS = frozenset(
    (
        "request_bind",
        "handoff",
        "notify_requested",
        "notify_send_return",
        "notify_observed",
        "senders_done",
        "release_requested",
        "blocks_released",
    )
)
_DURATION_KEYS = (
    "handshake_duration_ns",
    "prepared_to_posted_ns",
    "posted_to_completed_ns",
)
# Failure types that report on finished work (a notification, a release).
# Any other code, including unknown ones, is required work that failed.
_CLEANUP_CODES = frozenset(
    (
        Failure.NOTIFICATION_FAILED.name,
        Failure.LATE_NOTIFICATION.name,
        Failure.TRANSFER_RELEASE_FAILED.name,
    )
)


def _is_member(row):
    """A Mooncake write-batch member row: an association of a request with a batch,
    never a request occurrence of its own."""
    return row["kind"] == "request_bind" and row["subject"] == Subject.BATCH_MEMBER


def _percentiles(samples):
    """Nearest-rank p50/p95/max with the contributing sample count."""
    ordered = sorted(samples)

    def pick(percent):
        index = max(0, math.ceil(percent * len(ordered) / 100) - 1)
        return ordered[index]

    if not ordered:
        return {"samples": 0, "p50": None, "p95": None, "max": None}
    return {
        "samples": len(ordered),
        "p50": pick(50),
        "p95": pick(95),
        "max": ordered[-1],
    }


def _backend(meta):
    backends = (meta.get("context") or {}).get("backend") or []
    if len(backends) > 1:
        return "mixed"
    return backends[0] if backends else "unknown"


def _transport(rows):
    for row in rows:
        kind = row["transport"]["kind"]
        if kind != "unknown":
            return kind
    return "unknown"


def load_capture(path: Path):
    """Load one JSONL file with the exporter's rules, plus loss accounting."""
    meta, events, decoder, rows, missing = _load(path)
    ends = [row for row in rows if row.get("type") == "end"]
    stats = ends[-1].get("stats", {}) if ends else {}
    drops = {
        key: value
        for key, value in stats.items()
        if key.startswith("drop_") or key == "unknown_handles"
    }
    unresolved = sum(
        1
        for row in events
        for key in ("request_ref", "handoff_ref", "peer_ref")
        if row[key] and decoder.resolve(row[key]) is None
    )
    loss_detail = {
        "end_present": bool(ends),
        "end_complete": bool(ends and ends[-1].get("complete", False)),
        "missing_sequences": missing,
        "dropped": drops,
        "unresolved_identities": unresolved,
    }
    loss = (
        not loss_detail["end_present"]
        or not loss_detail["end_complete"]
        or missing
        or any(drops.values())
        or unresolved > 0
    )
    return {
        "path": path,
        "meta": meta,
        "events": events,
        # The writer's map of recording threads, kernel ID -> name (or None),
        # from its bounded "threads" rows.
        "threads": {
            tid: name
            for row in rows
            if row.get("type") == "threads"
            for tid, name in row["threads"].items()
        },
        "capture_rows": rows,
        "decoder": decoder,
        "loss": loss,
        "loss_detail": loss_detail,
    }


class _Source:
    """One process capture, classified into transfers, requests, handshakes."""

    def __init__(self, index, capture, problems):
        self.index = index
        self.capture = capture
        self.meta = capture["meta"]
        self.events = capture["events"]
        self.decoder = capture["decoder"]
        self.problems = problems
        self.context = self.meta.get("context") or {}
        self.role = self.context.get("role")
        self.engine_id = self.context.get("engine_id")
        self.transport = _transport(self.events)
        self.unclassified = 0
        self.classified = set()
        self.transfers = []
        self.requests = []
        self.handshakes = []
        self.members = {}
        self.write_batches = []
        self.post_groups = []
        self.view_unavailable = []
        self.mooncake = self.context.get("connector") == "mooncake"
        self.health = self._adapter_health()
        self._classify()

    def _classify(self):
        buckets = self._transfer_rows()
        for transfer, rows in sorted(buckets.items()):
            self.transfers.append(self._build_transfer(transfer, rows))
        self._collect_members()
        self._classify_handshakes()
        self._assign_request_occurrences()
        self._classify_requests()
        self._attach_handshakes_to_transfers()
        self._build_write_batches()
        self._build_post_groups()
        for row in self.events:
            if id(row) not in self.classified and row["kind"] not in (
                "model_step",
                "request_phase",
                "step_span",
            ):
                self.unclassified += 1

    def _adapter_health(self):
        """Why the adapter's observation cannot be taken as complete: the capture's
        ``profiler.json`` (a hook error disables recording without any ring drop, so
        the sink's own end record cannot show it). Empty means healthy; a missing or
        unreadable file is unknown, which is not healthy."""
        path = Path(self.capture["path"]).parent / "profiler.json"
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            return ["adapter health unknown: profiler.json missing or unreadable"]
        reasons = []
        if doc.get("hook_errors"):
            reasons.append(
                f"{doc['hook_errors']} hook error(s) disabled recording "
                f"({doc.get('first_error')})"
            )
        if doc.get("shutdown_failures"):
            reasons.append(f"shutdown failures: {doc['shutdown_failures']}")
        if doc.get("saved") is not True or doc.get("complete") is not True:
            reasons.append("the recorder did not report a saved, complete capture")
        if doc.get("observation_complete") is False:
            reasons.append("the adapter reports an incomplete observation")
        for key in (
            "interrupted_generations",
            "inflight_overflow",
            "unbound_batches",
            "late_hooks",
        ):
            if doc.get(key):
                reasons.append(f"{key} = {doc[key]}")
        return reasons

    def _collect_members(self):
        """Members of each write batch by batch generation, in recorded order."""
        for row in sorted(self.events, key=lambda row: row["sequence"]):
            if _is_member(row):
                self.members.setdefault(row["transfer"], []).append(row)

    def _assign_request_occurrences(self):
        """Give every transfer its request occurrence, from recorded order.

        Occurrences are split by the request's own bind rows; a transfer
        belongs to the occurrence its bind follows. A process without binds
        for the id has one occurrence by construction.
        """
        binds = {}
        for row in self.events:
            if (
                row["kind"] == "request_bind"
                and row["request_ref"]
                and not _is_member(row)
            ):
                binds.setdefault(row["request_ref"], []).append(row)
        self.occurrence_counts = {
            raw: len(rows) for raw, rows in binds.items()
        }
        for item in self.transfers:
            item["request_occurrence"] = 0
            raw = item.get("request_raw")
            rows = binds.get(raw, ())
            if not rows:
                continue
            before = sum(
                1 for row in rows if row["sequence"] < item[
                    "prepared_sequence"
                ]
            )
            item["request_occurrence"] = max(before - 1, 0)

    def _name(self, ref):
        return self.decoder.resolve(ref)

    def _problem(self, kind, detail):
        self.problems.append(
            {
                "source": self.index,
                "process_id": self.meta["process_id"],
                "kind": kind,
                "detail": detail,
            }
        )

    def _transfer_rows(self):
        prepared_transfers = {
            row["transfer"] for row in self.events if row["kind"] in _PREPARED
        }
        buckets = {}
        for row in self.events:
            transfer = row["transfer"]
            if row["kind"] in _PREPARED and transfer:
                buckets.setdefault(transfer, []).append(row)
                self.classified.add(id(row))
                continue
            is_transfer_event = (
                row["kind"] in _POST_BEGIN
                or row["kind"] in _POST_RETURN
                or row["kind"] == _RETIRE
                or (row["kind"] in _COMPLETE
                    and row["subject"] == Subject.WORKER)
                or (row["kind"] == _FAILED
                    and row["subject"] == Subject.TRANSFER)
            )
            if not is_transfer_event:
                continue
            self.classified.add(id(row))
            if not transfer:
                self._problem(
                    "orphan_transfer_event",
                    f"{row['kind']} has no handle generation",
                )
            elif transfer not in prepared_transfers:
                # A scheduler's COMPLETE_OBSERVED aggregates a request and is
                # never an orphan; a generation without a bind is.
                self._problem(
                    "transfer_event_without_bind",
                    f"{row['kind']} of handle generation {transfer} has no "
                    "bind in this process; the transfer's identity is "
                    "unknown and it is not summarized",
                )
            else:
                buckets.setdefault(transfer, []).append(row)
        return buckets

    def _build_transfer(self, transfer, rows):
        prepared = post_begin = pickup = post_return = None
        complete = failed = retire = None
        for row in rows:
            kind = row["kind"]
            if kind == "prepared":
                # Mooncake: the executor thread picked the batch up (queue time), not a
                # second bind.
                if pickup is None:
                    pickup = row
            elif kind in _POST_BEGIN:
                if post_begin is None:
                    post_begin = row
            elif kind in _PREPARED:
                if prepared is None:
                    prepared = row
                else:
                    self._problem(
                        "repeated_bind", f"transfer {transfer} bound again"
                    )
            elif kind in _POST_RETURN:
                if post_return is None:
                    post_return = row
                else:
                    self._problem(
                        "repeated_event", f"transfer {transfer}: repeated post"
                    )
            elif kind in _COMPLETE:
                if complete is None:
                    complete = row
                else:
                    self._problem(
                        "repeated_event",
                        f"transfer {transfer}: repeated completion",
                    )
            elif kind == _FAILED and failed is None:
                failed = row
            elif kind == _RETIRE and retire is None:
                retire = row
        state = "incomplete"
        if prepared is not None and complete is not None:
            if complete["time_ns"] >= prepared["time_ns"]:
                state = "completed"
            else:
                self._problem(
                    "ordering_violation",
                    f"transfer {transfer}: completion before its bind",
                )
        elif failed is not None:
            state = "failed"
        leg = None if self.mooncake or not prepared else prepared["aux0"]
        # The thread that issued the post, as observed by its post records; an
        # unposted transfer has no posting thread.
        poster = post_begin or post_return
        tid = poster.get("tid") if poster else None
        entry = {
            "source": self.index,
            "process_id": self.meta["process_id"],
            "transfer": transfer,
            "state": state,
            "thread": tid,
            "thread_name": self.capture["threads"].get(str(tid)) if tid else None,
            "leg": {0: "single", 1: "device", 2: "dram"}.get(leg, leg),
            "request": self._name(prepared["request_ref"])
            if prepared
            else None,
            "request_raw": prepared["request_ref"] if prepared else None,
            "request_occurrence": None,
            "handoff": self._name(prepared["handoff_ref"])
            if prepared
            else None,
            "peer": self._name(prepared["peer_ref"]) if prepared else None,
            "source_rank": prepared["subject"] if prepared else None,
            "selected_descriptors": (
                prepared["selected_descriptors"] if prepared else None
            ),
            "merged_descriptors": None if complete is None
            else complete["merged_descriptors"],
            "bytes": None if complete is None else complete["bytes"],
            "nixl_post_us": None if complete is None else complete["aux0"],
            "nixl_xfer_us": None if complete is None else complete["aux1"],
            "prepared_time_ns": prepared["time_ns"] if prepared else None,
            "prepared_sequence": prepared["sequence"] if prepared else None,
            "post_pass": prepared.get("post_pass") if prepared else None,
            "declared_members": (
                prepared["aux0"] if prepared and self.mooncake else None
            ),
            "executor_pickup_time_ns": pickup["time_ns"] if pickup else None,
            "post_begin_time_ns": post_begin["time_ns"] if post_begin else None,
            "posted_bytes": post_begin["bytes"] if post_begin else None,
            "posted_descriptors": (
                post_begin["selected_descriptors"] if post_begin else None
            ),
            "complete_status": complete["status"] if complete else None,
            "complete_bytes_word": complete["bytes"] if complete else None,
            "complete_descriptors_word": (
                complete["selected_descriptors"] if complete else None
            ),
            "posted_time_ns": post_return["time_ns"] if post_return else None,
            "completed_time_ns": complete["time_ns"] if complete else None,
            "failed_time_ns": failed["time_ns"] if failed else None,
            "failure_code": None,
            "retired_time_ns": retire["time_ns"] if retire else None,
        }
        for key in _DURATION_KEYS:
            entry[key] = None
        entry["throughput_bytes_per_s"] = None
        if prepared is not None and post_return is not None:
            entry["prepared_to_posted_ns"] = (
                post_return["time_ns"] - prepared["time_ns"]
            )
        if state == "completed" and post_return is not None:
            span = complete["time_ns"] - post_return["time_ns"]
            entry["posted_to_completed_ns"] = span
            if entry["bytes"] is not None and span > 0:
                entry["throughput_bytes_per_s"] = (
                    entry["bytes"] * 1_000_000_000 / span
                )
        if failed is not None:
            try:
                entry["failure_code"] = Failure(failed["status"]).name
            except ValueError:
                entry["failure_code"] = f"status_{failed['status']}"
        return entry

    def _classify_handshakes(self):
        """Per-attempt handshake records; a peer end joins the latest begin.

        Order matters: each begin opens an attempt for its peer, and every
        peer end attaches to the peer's latest open attempt. Several peer
        ends (ranks) of one attempt are retained and the attempt counts
        once. An unmatched peer end is a served reply only in the process
        that serves handshakes (a scheduler); elsewhere it is a missing
        begin and its duration stays unavailable.
        """
        rank_note = "remote rank for a local begin; served reply rank else"
        attempts = []
        open_attempt = {}
        unmatched = []
        for row in self.events:
            if row["kind"] == _BEGIN:
                self.classified.add(id(row))
                attempt = {
                    "peer_ref": row["peer_ref"],
                    "begin": row,
                    "ends": [],
                }
                attempts.append(attempt)
                open_attempt[row["peer_ref"]] = attempt
            elif row["kind"] == _END:
                self.classified.add(id(row))
                if row["subject"] == Subject.OFFSET_SAMPLE:
                    # NIXL's own estimate: a cross-check, never a duration.
                    continue
                if row["request_ref"]:
                    # A waiter end: an observation of the request, not the
                    # end of a peer attempt.
                    continue
                attempt = open_attempt.get(row["peer_ref"])
                if attempt is None:
                    unmatched.append(row)
                else:
                    attempt["ends"].append(row)
        for attempt in attempts:
            peer = self._name(attempt["peer_ref"])
            begin = attempt["begin"]
            ends = sorted(attempt["ends"], key=lambda row: row["sequence"])
            entry = {
                "source": self.index,
                "process_id": self.meta["process_id"],
                "peer": peer,
                "state": "completed" if ends else "unclosed",
                "begin_time_ns": begin["time_ns"],
                "duration_ns": (
                    ends[0]["time_ns"] - begin["time_ns"] if ends else None
                ),
                "ends": [
                    {"subject": row["subject"], "pp_rank": row["aux0"],
                     "time_ns": row["time_ns"]}
                    for row in ends
                ],
                "subject_note": rank_note,
            }
            self.handshakes.append(entry)
            if not ends:
                self._problem(
                    "unclosed_handshake",
                    f"handshake_begin for peer {peer!r} has no peer end",
                )
        for row in unmatched:
            peer = self._name(row["peer_ref"])
            if self.role == "scheduler":
                # A served reply has no local begin: no duration, no orphan.
                self.handshakes.append(
                    {
                        "source": self.index,
                        "process_id": self.meta["process_id"],
                        "peer": peer,
                        "state": "served_reply",
                        "begin_time_ns": None,
                        "duration_ns": None,
                        "ends": [
                            {"subject": row["subject"],
                             "pp_rank": row["aux0"],
                             "time_ns": row["time_ns"]}
                        ],
                        "subject_note": rank_note,
                    }
                )
            else:
                self._problem(
                    "missing_handshake_begin",
                    f"peer end for {peer!r} with no begin in this "
                    f"{self.role or 'process'}; its duration is unavailable",
                )

    def _attach_handshakes_to_transfers(self):
        for item in self.transfers:
            if not item["peer"]:
                continue
            candidates = [
                handshake for handshake in self.handshakes
                if handshake["peer"] == item["peer"]
                and handshake["state"] == "completed"
            ]
            if not candidates:
                # The handshake ran in another process; without calibration
                # its duration is unavailable here.
                continue
            rank = item["source_rank"]
            ends = [
                (handshake, end)
                for handshake in candidates
                for end in handshake["ends"]
            ]
            if rank is not None:
                matches = [pair for pair in ends if pair[1]["subject"] == rank]
            else:
                matches = ends
            if len(matches) == 1:
                handshake, end = matches[0]
                item["handshake_duration_ns"] = (
                    end["time_ns"] - handshake["begin_time_ns"]
                )
            # Zero or several matches by rank: the match is ambiguous and
            # stays unavailable rather than guessed.

    def _classify_requests(self):
        grouped = {}
        for row in self.events:
            is_request_row = (
                row["kind"] in _REQUEST_KINDS
                or (row["kind"] == _FAILED
                    and row["subject"] == Subject.REQUEST)
                or (row["kind"] in _COMPLETE
                    and row["subject"] == Subject.SCHEDULER)
                or (row["kind"] == _END
                    and row["request_ref"])
            )
            if not is_request_row or not row["request_ref"]:
                continue
            key = row["request_ref"]
            if key and self._name(key) is None:
                self._problem(
                    "unresolved_identity",
                    f"request id {key} never resolved",
                )
                self.classified.add(id(row))
                continue
            grouped.setdefault(key, []).append(row)
            self.classified.add(id(row))
        for key, rows in sorted(grouped.items()):
            self._build_request(self._name(key), key, rows)

    def _build_request(self, request, raw, rows):
        binds = [
            row for row in rows
            if row["kind"] == "request_bind" and not _is_member(row)
        ]
        occurrences = [rows]
        if len(binds) > 1:
            # Repeated ids stay separate occurrences, split by their binds.
            ordered = sorted(rows, key=lambda row: row["sequence"])
            occurrences = []
            current = []
            seen_binds = 0
            for row in ordered:
                is_bind = row["kind"] == "request_bind" and not _is_member(row)
                if is_bind and seen_binds:
                    occurrences.append(current)
                    current = []
                current.append(row)
                if is_bind:
                    seen_binds += 1
            occurrences.append(current)
        for index, occurrence in enumerate(occurrences):
            self._build_request_occurrence(
                request, raw, index, len(occurrences), occurrence,
                len(occurrences) > 1,
            )

    def _batch_write(self, batch):
        """How a write batch ended: written, no_data, failed or incomplete."""
        if batch is None:
            return None
        if batch["state"] == "completed":
            return "no_data" if batch["complete_status"] == 2 else "written"
        return "failed" if batch["state"] == "failed" else "incomplete"

    def _membership(self, rows):
        """A request's write-batch memberships (Mooncake).

        Associations, never request occurrences; the posted totals (members of
        batches whose write succeeded) are kept apart from the planned-but-
        unconfirmed ones (a failed or unfinished write) and from members whose
        share is unknown.
        """
        if not rows:
            return {}
        by_handle = {item["transfer"]: item for item in self.transfers}
        batches = []
        posted_bytes = posted_descriptors = 0
        planned_bytes = planned_descriptors = 0
        unknown = 0
        for row in rows:
            batch = by_handle.get(row["transfer"])
            write = self._batch_write(batch)
            known = row["bytes"] is not None and row["selected_descriptors"] is not None
            basis = (
                "unknown"
                if not known
                else "posted"
                if write in ("written", "no_data")
                else "planned"
            )
            batches.append(
                {
                    "batch": row["transfer"],
                    "ok": row["status"] == 0,
                    "write": write,
                    "bytes": row["bytes"],
                    "descriptors": row["selected_descriptors"],
                    "basis": basis,
                }
            )
            if basis == "posted" and row["status"] == 0:
                posted_bytes += row["bytes"]
                posted_descriptors += row["selected_descriptors"]
            elif basis in ("planned", "posted"):
                planned_bytes += row["bytes"]
                planned_descriptors += row["selected_descriptors"]
            else:
                unknown += 1
        return {
            "write_batches": batches,
            "write_bytes_posted": posted_bytes,
            "write_descriptors_posted": posted_descriptors,
            "write_bytes_planned_unconfirmed": planned_bytes,
            "write_descriptors_planned_unconfirmed": planned_descriptors,
            "write_shares_unknown": unknown,
        }

    def _build_write_batches(self):
        """The ``write_batch`` view (Mooncake): one entry per batch, with its
        members and whether their shares partition the batch's measured totals
        exactly.
        """
        if not self.mooncake:
            return
        loss = bool(self.capture["loss"])
        for item in self.transfers:
            rows = self.members.get(item["transfer"], [])
            members = []
            for row in rows:
                members.append(
                    {
                        "request": self._name(row["request_ref"]),
                        "transfer_id": self._name(row["handoff_ref"]),
                        "ok": row["status"] == 0,
                        "bytes": row["bytes"],
                        "descriptors": row["selected_descriptors"],
                    }
                )
            identities = [m["request"] for m in members]
            totals = None
            if item["posted_bytes"] is not None:
                totals = (item["posted_bytes"], item["posted_descriptors"])
            elif item["complete_status"] == 2:
                totals = (
                    item["complete_bytes_word"],
                    item["complete_descriptors_word"],
                )
            reasons = []
            if item["declared_members"] is None:
                reasons.append("no declared member count")
            elif item["declared_members"] != len(identities):
                reasons.append(
                    f"{item['declared_members']} members declared, "
                    f"{len(identities)} recorded"
                )
            if len(set(identities)) != len(identities):
                reasons.append("a request is recorded twice as a member")
            known = all(
                m["bytes"] is not None and m["descriptors"] is not None for m in members
            )
            bytes_sum = sum(m["bytes"] for m in members if m["bytes"] is not None)
            descriptors_sum = sum(
                m["descriptors"] for m in members if m["descriptors"] is not None
            )
            if reasons:
                status = "member_mismatch"
            elif not members or not known:
                status = "shares_unavailable"
            elif totals is None:
                status = "not_posted"
            elif (bytes_sum, descriptors_sum) != totals:
                status = "mismatch"
                reasons.append(
                    f"members sum to {bytes_sum} bytes and {descriptors_sum} "
                    f"descriptors, the batch to {totals[0]} and {totals[1]}"
                )
            elif loss or self.health:
                status = "observed_only"
                reasons.append("capture loss: a member row may be missing")
                reasons.extend(f"adapter: {r}" for r in self.health)
            else:
                status = "exact"
            confirmed = item["complete_bytes_word"]
            if (
                totals is not None
                and item["posted_bytes"] is not None
                and item["state"] == "completed"
                and (confirmed, item["complete_descriptors_word"]) != totals
            ):
                reasons.append("the completion's totals differ from the posted totals")
                status = "mismatch"
            self.write_batches.append(
                {
                    "source": self.index,
                    "process_id": self.meta["process_id"],
                    "view": "write_batch",
                    "batch": item["transfer"],
                    "state": item["state"],
                    "write": self._batch_write(item),
                    "times": {
                        "bound_ns": item.get("prepared_time_ns"),
                        "executor_pickup_ns": item.get("executor_pickup_time_ns"),
                        "write_begin_ns": item.get("post_begin_time_ns"),
                        "write_returned_ns": item.get("posted_time_ns"),
                        "completed_ns": item.get("completed_time_ns"),
                        "failed_ns": item.get("failed_time_ns"),
                    },
                    "engine_id": self.engine_id,
                    "consumer": item["peer"],
                    "consumer_tp_rank": item["source_rank"],
                    "members_declared": item["declared_members"],
                    "members_recorded": len(identities),
                    "members": members,
                    "bytes": None if totals is None else totals[0],
                    "descriptors": None if totals is None else totals[1],
                    "partition": {
                        "status": status,
                        "members_bytes": bytes_sum if known else None,
                        "members_descriptors": descriptors_sum if known else None,
                        "reasons": reasons,
                    },
                }
            )

    def _posting_ended(self, members):
        """Whether the worker is known to have left the pass that posted ``members``.

        A later pass's bind proves it for an earlier pass. For the last pass only a
        worker ``complete_observed`` recorded after its last bind does: completions
        are polled by the main thread after the step's ``start_load_kv`` returned, so
        a pass counter alone, or a clean recorder end, does not prove it.
        """
        last_bind = max(m["prepared_sequence"] for m in members)
        return any(
            row["kind"] == "complete_observed"
            and row["subject"] == Subject.WORKER
            and row["sequence"] > last_bind
            for row in self.events
        )

    def _build_post_groups(self):
        """The ``post_group`` view (NIXL): the independent reads one worker pass posted
        together, keyed by this process capture and the recorded pass. Reads stay
        first-class rows; a group is a derived view. Never a transport call and never an
        additive duration: reads overlap."""
        if self.mooncake or self.role != "worker":
            return
        reads = [
            item for item in self.transfers if item["prepared_sequence"] is not None
        ]
        if not reads:
            return
        caps = self.meta.get("capabilities") or self.context.get("capabilities") or {}
        if not caps.get("post_pass") or all(
            item["post_pass"] is None for item in reads
        ):
            self.view_unavailable.append(
                {
                    "scope": f"source {self.index} (worker)",
                    "reason": "post_group unavailable: no post pass recorded "
                    "(a capture made before the pass field, or a profiler without it)",
                }
            )
            return
        detail = self.capture["loss_detail"]
        structural = bool(
            detail["missing_sequences"]
            or any(detail["dropped"].values())
            or detail["unresolved_identities"]
        )
        end_complete = detail["end_complete"]
        passes = sorted(
            {item["post_pass"] for item in reads if item["post_pass"] is not None}
        )
        last = passes[-1] if passes else None
        ungrouped = [item for item in reads if item["post_pass"] is None]
        if ungrouped:
            self.view_unavailable.append(
                {
                    "scope": f"source {self.index} (worker)",
                    "reason": f"{len(ungrouped)} read(s) carry no post pass and "
                    "are in no group",
                }
            )
        for number in passes:
            members = sorted(
                (item for item in reads if item["post_pass"] == number),
                key=lambda item: item["prepared_sequence"],
            )
            accepted = [m for m in members if m["posted_time_ns"] is not None]
            unresolved = [
                m
                for m in members
                if m["state"] == "incomplete" and m["posted_time_ns"] is None
            ]
            completed = [m for m in members if m["state"] == "completed"]
            failed = [m for m in members if m["state"] == "failed"]
            reasons = []
            if structural:
                reasons.append("capture loss: a member may be missing")
            reasons.extend(f"adapter: {r}" for r in self.health)
            if number == last and not self._posting_ended(members):
                reasons.append(
                    "no later worker record shows that posting ended: the last "
                    "pass may have been unfinished"
                )
            if number == last and not end_complete:
                reasons.append(
                    "the capture has no complete end: the last pass may not "
                    "have finished"
                )
            if unresolved:
                reasons.append(
                    f"{len(unresolved)} prepared read(s) never posted nor failed"
                )
            if any(
                m["state"] == "incomplete" and m["posted_time_ns"] is not None
                for m in members
            ):
                reasons.append("a posted read has no observed completion")
            totals_known = bool(members) and all(
                m["state"] == "completed" and m["bytes"] is not None for m in members
            )
            posting_known = bool(accepted) and not unresolved
            completion_known = bool(accepted) and all(
                m["state"] == "completed" for m in accepted
            )
            exact = not reasons
            first_prepared = min(m["prepared_time_ns"] for m in members)
            first_post = min((m["posted_time_ns"] for m in accepted), default=None)
            self.post_groups.append(
                {
                    "source": self.index,
                    "process_id": self.meta["process_id"],
                    "view": "post_group",
                    "post_pass": number,
                    "reads": len(members),
                    "requests": sorted({m["request"] for m in members if m["request"]}),
                    "source_ranks": sorted(
                        {
                            m["source_rank"]
                            for m in members
                            if m["source_rank"] is not None
                        }
                    ),
                    "members": [
                        {
                            "transfer": m["transfer"],
                            "request": m["request"],
                            "source_rank": m["source_rank"],
                            "leg": m["leg"],
                            "state": m["state"],
                            "failure": m["failure_code"],
                            "posted": m["posted_time_ns"] is not None,
                            "selected_descriptors": m["selected_descriptors"],
                            "merged_descriptors": m["merged_descriptors"],
                            "bytes": m["bytes"],
                        }
                        for m in members
                    ],
                    "outcomes": {
                        "completed": len(completed),
                        "failed": len(failed),
                        # never accepted: a leg released unused because an earlier leg
                        # failed, versus a post that raised for this leg itself
                        "unused": sum(
                            1 for m in failed
                            if m["failure_code"] == "TRANSFER_NOT_POSTED"
                        ),
                        "submission_failed": sum(
                            1 for m in failed
                            if m["posted_time_ns"] is None
                            and m["failure_code"] != "TRANSFER_NOT_POSTED"
                        ),
                        "incomplete": len(unresolved)
                        + sum(
                            1
                            for m in members
                            if m["state"] == "incomplete"
                            and m["posted_time_ns"] is not None
                        ),
                        "not_posted": sum(
                            1 for m in failed if m["posted_time_ns"] is None
                        ),
                    },
                    # Whole-pass totals and spans are claimed only for an exact
                    # group; an observed-only one reports what was seen, under
                    # the *_observed names.
                    "selected_descriptors": (
                        sum(m["selected_descriptors"] or 0 for m in members)
                        if exact
                        else None
                    ),
                    "selected_descriptors_observed": sum(
                        m["selected_descriptors"] or 0 for m in members
                    ),
                    "merged_descriptors": (
                        sum(m["merged_descriptors"] for m in members)
                        if exact
                        and totals_known
                        and all(m["merged_descriptors"] is not None for m in members)
                        else None
                    ),
                    "bytes": sum(m["bytes"] for m in members)
                    if exact and totals_known
                    else None,
                    "bytes_observed": sum(
                        m["bytes"] for m in members if m["bytes"] is not None
                    ),
                    "bytes_observed_reads": sum(
                        1 for m in members if m["bytes"] is not None
                    ),
                    "first_prepared_time_ns": first_prepared,
                    "first_post_time_ns": first_post,
                    "last_post_time_ns": max(
                        (m["posted_time_ns"] for m in accepted), default=None
                    ),
                    "last_completion_time_ns": max(
                        (m["completed_time_ns"] for m in completed), default=None
                    ),
                    "posting_span_ns": (
                        max(m["posted_time_ns"] for m in accepted) - first_prepared
                        if exact and posting_known
                        else None
                    ),
                    # first accepted post to last observed completion
                    "completion_span_ns": (
                        max(m["completed_time_ns"] for m in accepted) - first_post
                        if exact and completion_known
                        else None
                    ),
                    # sums of independent reads' own measurements, never a group
                    # transport time
                    "read_xfer_us_sum": (
                        sum(m["nixl_xfer_us"] for m in completed)
                        if exact
                        and completed
                        and all(m["nixl_xfer_us"] is not None for m in completed)
                        else None
                    ),
                    "read_post_us_sum": (
                        sum(m["nixl_post_us"] for m in completed)
                        if exact
                        and completed
                        and all(m["nixl_post_us"] is not None for m in completed)
                        else None
                    ),
                    "completeness": "observed_only" if reasons else "exact",
                    "reasons": reasons,
                }
            )

    def _transfers_of(self, request, raw, index, total):
        """The transfers of one request occurrence, never another's.

        A process without the id's own binds has a single occurrence and
        every transfer of the id; otherwise the recorded order of the binds
        decides.
        """
        count = self.occurrence_counts.get(raw, 0)
        if count <= 1:
            return [
                item for item in self.transfers
                if item.get("request_raw") == raw
            ]
        return [
            item for item in self.transfers
            if item.get("request_raw") == raw
            and item["request_occurrence"] == index
        ]

    def _build_request_occurrence(
        self, request, raw, index, total, rows, repeated
    ):
        ordered = sorted(rows, key=lambda row: row["sequence"])
        failures = [
            row for row in ordered
            if row["kind"] == _FAILED
            and row["subject"] == Subject.REQUEST
        ]
        completed = [
            row for row in ordered
            if row["kind"] == "complete_observed"
            and row["subject"] == Subject.SCHEDULER
        ]
        handoffs = []
        for row in ordered:
            if row["kind"] != "handoff":
                continue
            if row["subject"] == 1:
                handoffs.append(
                    {
                        "direction": "outgoing",
                        "time_ns": row["time_ns"],
                        "sequence": row["sequence"],
                        "generation": row["aux0"],
                        "blocks": row["aux1"],
                        "engine": self._name(row["peer_ref"]),
                        # Mooncake's outgoing handoff names the shared transfer
                        # ID, which is what a consumer's handoff names too.
                        "transfer_id": (
                            self._name(row["handoff_ref"]) if self.mooncake else None
                        ),
                    }
                )
            else:
                handoffs.append(
                    {
                        "direction": "incoming",
                        "time_ns": row["time_ns"],
                        "sequence": row["sequence"],
                        "producer_engine": self._name(row["peer_ref"]),
                        # NIXL's incoming handoff carries the producer's request;
                        # Mooncake's carries the shared transfer ID (the
                        # producer's request is then found through it).
                        "producer_request": (
                            None if self.mooncake else self._name(row["handoff_ref"])
                        ),
                        "transfer_id": (
                            self._name(row["handoff_ref"]) if self.mooncake else None
                        ),
                    }
                )
        release = [
            row for row in ordered if row["kind"] == "release_requested"
        ]
        blocks_released = [
            row for row in ordered if row["kind"] == "blocks_released"
        ]
        notified = [
            row for row in ordered if row["kind"] == "notify_observed"
        ]
        sent = [
            row for row in ordered if row["kind"] == "notify_send_return"
        ]
        done = [row for row in ordered if row["kind"] == "senders_done"]
        waits = [
            {"ok": row["status"] == 0, "time_ns": row["time_ns"]}
            for row in ordered
            if row["kind"] == _END and row["request_ref"]
        ]
        transfers = self._transfers_of(request, raw, index, total)
        byte_samples = [
            item["bytes"] for item in transfers
            if item["bytes"] is not None
        ]
        spans = []
        for item in transfers:
            start = item["prepared_time_ns"]
            end = item["completed_time_ns"]
            if end is None:
                end = item["failed_time_ns"]
            if start is not None and end is not None:
                spans.append(end - start)
        # Completion requires proof: every transfer of this occurrence
        # completed, and no known failure before that boundary. A single
        # finished READ is not the boundary, and the aggregate
        # receive-finished signal is a terminal notification that also fires
        # for failed requests: it is recorded with its time, but never
        # overrides a known failure nor establishes success by itself.
        all_completed = bool(transfers) and all(
            item["state"] == "completed" for item in transfers
        )
        failed_transfers = [
            item for item in transfers if item["state"] == "failed"
        ]
        boundary = None
        if all_completed:
            boundary = max(
                (
                    item["completed_time_ns"] for item in transfers
                    if item["completed_time_ns"] is not None
                ),
                default=None,
            )
        failure_items = []
        for row in failures:
            try:
                code = Failure(row["status"]).name
            except ValueError:
                code = f"status_{row['status']}"
            failure_items.append({"code": code, "time_ns": row["time_ns"]})
        if all_completed:
            # Success survives only an identified cleanup or notification
            # failure at or after the proven movement boundary. A setup or
            # transfer failure, an unknown error type, or anything before
            # the boundary is required work that never succeeded, even when
            # another READ finished first.
            cleanup = [
                item for item in failure_items
                if item["code"] in _CLEANUP_CODES
                and boundary is not None and item["time_ns"] >= boundary
            ]
            required = [item for item in failure_items
                        if item not in cleanup]
            state = (
                "failed"
                if failed_transfers or required
                else "completed"
            )
            if state == "failed":
                boundary = None
        elif failed_transfers or failures:
            state = "failed"
            boundary = None
        else:
            state = "incomplete"
            boundary = None
        membership = self._membership([row for row in ordered if _is_member(row)])
        entry = {
            "source": self.index,
            "process_id": self.meta["process_id"],
            "role": self.role,
            "request": request,
            "state": state,
            **membership,
            "request_occurrence": index,
            "request_occurrences": total,
            "repeated_id": repeated,
            "handoffs": handoffs,
            "handshake_waits": waits,
            "transfers": len(transfers),
            "bytes": sum(byte_samples) if byte_samples else None,
            "bytes_samples": len(byte_samples),
            # Sum of the request's per-transfer terminal intervals; overlapped
            # transfers overlap, so this is a duration sum, not a latency.
            "duration_sum_ns": sum(spans) if spans else None,
            "duration_sum_samples": len(spans),
            "first_posted_time_ns": min(
                (
                    item["posted_time_ns"] for item in transfers
                    if item["posted_time_ns"] is not None
                ),
                default=None,
            ),
            # The scheduler's permission to free blocks; not the pool return.
            "release_requested_time_ns":
                release[-1]["time_ns"] if release else None,
            "blocks_released_time_ns": (
                blocks_released[-1]["time_ns"] if blocks_released else None
            ),
            "notified": notified[-1]["aux0"] if notified else None,
            "notified_expected": notified[-1]["aux1"] if notified else None,
            "notify_sent_time_ns": sent[-1]["time_ns"] if sent else None,
            "senders_done_time_ns": done[-1]["time_ns"] if done else None,
            "recv_finished_time_ns":
                completed[-1]["time_ns"] if completed else None,
            "failures": failure_items,
            # Post-completion cleanup/notification issues only exist once the
            # relevant data movement has completed.
            "post_completion_failures": [
                item for item in failure_items
                if boundary is not None and item["time_ns"] >= boundary
            ],
        }
        self.requests.append(entry)


def _join_producers(sources):
    """Join consumer handoffs to producers by exact recorded identities.

    The consumer's incoming handoff carries (producer engine id, producer
    request id); the producer's outgoing handoff carries its own engine id
    and request id. A join is equality of these recorded fields, never of
    request text alone. Occurrence ordinals are local to each process and
    are not correlation evidence: if several outgoing handoffs or several
    consumer requests carry the same recorded identity, the join stays
    ambiguous until recorded generation/correlation information can
    disambiguate it.
    """
    incoming = [
        (source, entry, handoff)
        for source in sources
        for entry in source.requests
        for handoff in entry["handoffs"]
        if handoff["direction"] == "incoming"
    ]
    outgoing = [
        (source, entry, handoff)
        for source in sources
        for entry in source.requests
        for handoff in entry["handoffs"]
        if handoff["direction"] == "outgoing"
    ]
    producer_engines = {
        source.engine_id for source in sources if source.engine_id
    }
    joins = []
    unavailable = []
    for consumer_source, entry, handoff in incoming:
        peer = handoff.get("producer_engine")
        by_transfer = consumer_source.mooncake
        # NIXL joins on (producer engine, producer request); Mooncake, whose handoff
        # carries the shared transfer ID instead, on (producer engine, transfer ID).
        remote = handoff.get("transfer_id" if by_transfer else "producer_request")
        scope = (
            f"source {consumer_source.index} request {entry['request']!r} "
            f"occurrence {entry['request_occurrence']}"
        )
        if peer is None or remote is None:
            unavailable.append(
                {
                    "scope": scope,
                    "reason": "incoming handoff with unresolved producer "
                    "identity",
                }
            )
            continue
        if peer not in producer_engines:
            unavailable.append(
                {
                    "scope": scope,
                    "reason": f"producer engine {peer!r} is not in the input",
                }
            )
            continue
        all_matches = [
            (producer_source, producer_entry, producer_handoff)
            for producer_source, producer_entry, producer_handoff in outgoing
            if producer_source.engine_id == peer
            and producer_source.mooncake == by_transfer
            and (
                producer_handoff.get("transfer_id") == remote
                if by_transfer
                else producer_entry["request"] == remote
            )
        ]
        # Occurrence ordinals are local; they are not correlation evidence
        # across processes. If several outgoing handoffs carry the same
        # recorded identity, the join stays ambiguous until recorded
        # generation/correlation information can disambiguate them.
        matches = all_matches
        producer_request = (
            matches[0][1]["request"] if by_transfer and len(matches) == 1 else remote
        )
        joins.append(
            {
                "consumer_source": consumer_source.index,
                "consumer_process_id": consumer_source.meta["process_id"],
                "consumer_request": entry["request"],
                "consumer_occurrence": entry["request_occurrence"],
                "consumer_handoff_sequence": handoff["sequence"],
                "producer_request": None if by_transfer and not matches else (
                    producer_request
                ),
                "transfer_id": remote if by_transfer else None,
                "join_key": "transfer_id" if by_transfer else "request",
                "producer_engine": peer,
                "producer_sources": sorted(
                    {producer.index for producer, _, _ in matches}
                ),
                "producer_handoff_time_ns": (
                    matches[0][2]["time_ns"] if len(matches) == 1 else None
                ),
                "matches": len(matches),
                "ambiguous": len(matches) > 1,
                "joined": len(matches) == 1,
            }
        )
        if len(matches) > 1:
            unavailable.append(
                {
                    "scope": scope,
                    "reason": f"{len(matches)} outgoing handoffs match "
                    f"({peer!r}, {remote!r}); the join is ambiguous",
                }
            )
    reused = {}
    for join in joins:
        reused.setdefault(
            (
                join["producer_engine"],
                join["transfer_id"]
                if join["join_key"] == "transfer_id"
                else join["producer_request"],
            ),
            [],
        ).append(join)
    for (peer, remote), shared in reused.items():
        if len(shared) > 1:
            for join in shared:
                join["identity_reused"] = True
    return joins, unavailable


def _cross_process_spans(sources, joins, group, alias, not_available):
    """Handoff to first READ posted spans, only over aligned clocks.

    The READ lives in the consumer's worker, not in the process that
    recorded the incoming handoff. A candidate must match the consumer
    engine and request as well as the transfer's own recorded (peer,
    producer request) identity, and the earliest post is chosen after
    projecting every candidate onto the reference clock, never by raw
    clock values.
    """
    joined = [join for join in joins if join["joined"] and not join.get(
        "identity_reused"
    )]
    reused = [join for join in joins if join.get("identity_reused")]
    if reused:
        not_available.append(
            {
                "scope": f"{len(reused)} joined request occurrence(s)",
                "reason": "the (producer engine, producer request) identity "
                "is reused across occurrences in this input, so the READ "
                "posts cannot be ascribed to one occurrence; spans omitted",
            }
        )
    if group is None:
        if joined:
            not_available.append(
                {
                    "scope": f"{len(joined)} joined request occurrence(s)",
                    "reason": "handoff-to-first-post spans need "
                    "--calibration; clocks stay unaligned",
                }
            )
        return []
    spans = []
    for join in joined:
        consumer_engine = sources[join["consumer_source"]].engine_id
        if consumer_engine is None:
            not_available.append(
                {
                    "scope": f"join of {join['consumer_request']!r} "
                    f"occurrence {join['consumer_occurrence']}",
                    "reason": "the consumer engine identity is not recorded "
                    "for this join, so its READs cannot be told from any "
                    "other consumer's; no span",
                }
            )
            continue
        producer = sources[join["producer_sources"][0]]
        posts = [
            (item, sources[item["source"]])
            for source in sources
            for item in source.transfers
            if item["handoff"] == join["producer_request"]
            and item["peer"] == join["producer_engine"]
            and item["request"] == join["consumer_request"]
            and (
                consumer_engine is None
                or source.engine_id == consumer_engine
            )
            and item["posted_time_ns"] is not None
        ]
        if not posts or join["producer_handoff_time_ns"] is None:
            not_available.append(
                {
                    "scope": f"join of {join['consumer_request']!r} "
                    f"occurrence {join['consumer_occurrence']}",
                    "reason": "no READ post of this join's consumer and "
                    "transfer identity in the input, or no producer "
                    "handoff time",
                }
            )
            continue
        # One worker process carrying several request occurrences of the
        # same recorded identity cannot ascribe its READs to this join:
        # local ordinals are not cross-process correlation evidence.
        ambiguous_workers = [
            source.index
            for source in {worker for _, worker in posts}
            if len(
                {
                    item["request_occurrence"]
                    for item, item_source in posts
                    if item_source is source
                }
            )
            > 1
        ]
        if ambiguous_workers:
            not_available.append(
                {
                    "scope": f"join of {join['consumer_request']!r} "
                    f"occurrence {join['consumer_occurrence']}",
                    "reason": f"worker source(s) {sorted(ambiguous_workers)} "
                    "record more than one request occurrence with the same "
                    "consumer and transfer identity; the recorded rows "
                    "cannot select one, so no span",
                }
            )
            continue
        producer_clock = alias.get(
            producer.meta["clock_id"], producer.meta["clock_id"]
        )
        start = group.project(producer_clock, join["producer_handoff_time_ns"])
        if start is None:
            continue
        # Choose the earliest post only after projecting every candidate;
        # raw clock values of different generations do not order events.
        projected = []
        unprojected = 0
        for transfer, worker in posts:
            projection = group.project(
                alias.get(worker.meta["clock_id"], worker.meta["clock_id"]),
                transfer["posted_time_ns"],
            )
            if projection is None:
                unprojected += 1
            else:
                projected.append((projection, transfer, worker))
        if unprojected:
            not_available.append(
                {
                    "scope": f"join of {join['consumer_request']!r} "
                    f"occurrence {join['consumer_occurrence']}",
                    "reason": f"{unprojected} candidate READ(s) could not "
                    "be projected onto the reference clock; the earliest "
                    "is chosen among the projected ones only",
                }
            )
        if not projected:
            continue
        projection, transfer, worker = min(
            projected, key=lambda item: item[0].time_ns
        )
        spans.append(
            {
                "kind": "handoff_to_first_read_posted",
                "consumer_source": join["consumer_source"],
                "post_source": worker.index,
                "producer_source": join["producer_sources"][0],
                "consumer_request": join["consumer_request"],
                "producer_request": join["producer_request"],
                "consumer_occurrence": join["consumer_occurrence"],
                "duration_ns": round(
                    projection.time_ns - start.time_ns
                ),
                "uncertainty_ns": (
                    start.uncertainty_ns + projection.uncertainty_ns
                ),
                "drift_known": (
                    start.drift_known and projection.drift_known
                ),
                "extrapolated": (
                    start.extrapolated or projection.extrapolated
                ),
                "model": "linear_estimate_not_a_bound",
            }
        )
    if len(spans) < len(joined):
        not_available.append(
            {
                "scope": f"{len(joined) - len(spans)} joined request "
                "occurrence(s)",
                "reason": "clock projection unavailable (missing calibration "
                "generation or no READ post of the transfer identity in the "
                "input)",
            }
        )
    return spans


def _aggregates(sources, transfers):
    buckets = {}
    for item in transfers:
        source = sources[item["source"]]
        prepared = [
            row for row in source.events
            if row.get("transfer") == item["transfer"]
            and row["kind"] in _PREPARED
        ]
        key = (
            f"{_backend(source.meta)}/"
            f"{_transport(prepared) if prepared else 'unknown'}"
        )
        bucket = buckets.setdefault(
            key,
            {
                "key": key,
                "sources": [],
                "count": 0,
                "completed": 0,
                "failed": 0,
                "incomplete": 0,
                "bytes_sum": 0,
                "bytes_samples": 0,
                "handshake_duration_ns": [],
                "prepared_to_posted_ns": [],
                "posted_to_completed_ns": [],
                "throughput_bytes_per_s": [],
            },
        )
        if source.index not in bucket["sources"]:
            bucket["sources"].append(source.index)
        bucket["count"] += 1
        bucket[item["state"]] += 1
        if item["bytes"] is not None:
            bucket["bytes_sum"] += item["bytes"]
            bucket["bytes_samples"] += 1
        for metric in _DURATION_KEYS + ("throughput_bytes_per_s",):
            if item[metric] is not None:
                bucket[metric].append(item[metric])
    transfer_buckets = {}
    for key, bucket in buckets.items():
        entry = {
            key_: bucket[key_]
            for key_ in (
                "key", "sources", "count", "completed", "failed",
                "incomplete",
            )
        }
        entry["bytes"] = {
            "sum": bucket["bytes_sum"], "samples": bucket["bytes_samples"]
        }
        for metric in _DURATION_KEYS + ("throughput_bytes_per_s",):
            entry[metric] = _percentiles(bucket[metric])
        transfer_buckets[key] = entry
    handshake_buckets = {}
    for source in sources:
        for handshake in source.handshakes:
            if handshake["state"] != "completed":
                continue
            key = f"{_backend(source.meta)}/{source.transport}"
            bucket = handshake_buckets.setdefault(
                key, {"key": key, "count": 0, "duration_samples": []}
            )
            bucket["count"] += 1
            bucket["duration_samples"].append(handshake["duration_ns"])
    for bucket in handshake_buckets.values():
        bucket["duration_ns"] = _percentiles(bucket.pop("duration_samples"))
    return {"transfers": transfer_buckets, "handshakes": handshake_buckets}


_NOT_AVAILABLE = [
    {
        "scope": "per transfer",
        "reason": "completed_to_notified_ns is not computed: notification "
        "observation is recorded under the producer's request id, its count "
        "never identifies individual transfers, and it can precede the "
        "consumer's DONE poll, so any gap would be signed",
    },
    {
        "scope": "per request",
        "reason": "the block-pool return (BLOCKS_RELEASED) is recorded by no "
        "adapter; release_requested_time_ns is the scheduler's permission "
        "only, and blocks_released_time_ns would need records that do not "
        "exist yet",
    },
    {
        "scope": "all",
        "reason": "socket and queue-pair identities are not recorded",
    },
    {
        "scope": "timelines",
        "reason": "the scheduler-side first-token instant, a step's device "
        "completion (a step ends at the runner's finish_forward, the host-side "
        "end of the forward), queueing before the serving layer's entry (HTTP "
        "accept and parse), and the times of unmarked decode steps are not "
        "recorded; the first token is the frontend's first non-empty output",
    },
]


def summarize(paths, calibration=None, reference_clock=None) -> dict:
    """Summarize profiler JSONL files or directories of them.

    Args:
        paths: JSONL files or directories searched for ``events.jsonl``.
        calibration: Paths of saved ``ClockMapping`` JSON files. Required
            for cross-process spans.
        reference_clock: The reference clock generation; defaults to the
            first calibration's local generation.

    Returns:
        The summary document: sources with loss flags, transfers, requests,
        handshakes, identity joins, calibrated cross-process spans,
        aggregates per backend/transport, problems and ``not_available``.

    Raises:
        ValueError: A file has no metadata, the inputs mix run ids, or two
            inputs record the same process generation with different rows.
    """
    files = []
    seen_files = set()
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found = sorted(path.rglob("events.jsonl"))
            if not found:
                raise ValueError(f"no events.jsonl under {path}")
            candidates = found
        else:
            candidates = [path]
        for candidate in candidates:
            real = candidate.resolve()
            if real not in seen_files:
                seen_files.add(real)
                files.append(candidate)
    if not files:
        raise ValueError("no events.jsonl inputs")
    # One logical contribution per recorded process generation: identical
    # copies (a repeated path, overlapping directory and file inputs) fold
    # into one source, conflicting copies are rejected, never both counted.
    captures = []
    duplicates = {}
    by_process = {}
    for path in files:
        capture = load_capture(path)
        identity = (
            capture["meta"]["run_id"],
            capture["meta"]["process_id"],
        )
        content = json.dumps(capture["capture_rows"], sort_keys=True)
        if identity in by_process:
            if by_process[identity][0] != content:
                raise ValueError(
                    "conflicting duplicate captures for process "
                    f"{identity[1]} of run {identity[0]}: {path} vs "
                    f"{by_process[identity][1]}"
                )
            duplicates.setdefault(identity[1], []).append(str(path))
            continue
        by_process[identity] = (content, path)
        captures.append(capture)
    for capture in captures:
        capture["duplicates"] = duplicates
    run_ids = {capture["meta"]["run_id"] for capture in captures}
    if len(run_ids) > 1:
        raise ValueError(f"cannot summarize different runs: {sorted(run_ids)}")
    problems = []
    sources = []
    for index, capture in enumerate(captures):
        source = _Source(index, capture, problems)
        if capture["loss"]:
            problems.append(
                {
                    "source": index,
                    "process_id": capture["meta"]["process_id"],
                    "kind": "capture_loss",
                    "detail": json.dumps(
                        capture["loss_detail"], sort_keys=True
                    ),
                }
            )
        if source.health:
            problems.append(
                {
                    "source": index,
                    "process_id": capture["meta"]["process_id"],
                    "kind": "adapter_unhealthy",
                    "detail": "; ".join(source.health),
                }
            )
        if source.unclassified:
            problems.append(
                {
                    "source": index,
                    "process_id": capture["meta"]["process_id"],
                    "kind": "unclassified_events",
                    "detail": f"{source.unclassified} row(s) of kinds this "
                    "summary does not model",
                }
            )
        sources.append(source)
    not_available = [dict(item) for item in _NOT_AVAILABLE]
    for source in sources:
        not_available.extend(source.view_unavailable)
    for source in sources:
        if source.role in ("worker", "scheduler") and not source.capture[
            "loss_detail"
        ]["end_present"]:
            not_available.append(
                {
                    "scope": f"source {source.index} ({source.role})",
                    "reason": "no end record: capture completeness unknown. "
                    "Rows flushed before the stop are summarized above; rows "
                    "still in the open ring are lost and cannot appear "
                    "(typically a recorder that was never closed)",
                }
            )
    group = None
    if calibration:
        raw = [json.loads(Path(path).read_text()) for path in calibration]
        mappings = [ClockMapping.from_json(item) for item in raw]
        reference = reference_clock or (
            mappings[0].start.local_generation if mappings else None
        )
        group = ClockGroup(
            reference, {m.start.remote_generation: m for m in mappings}
        )
        if group.run_id is not None and group.run_id != next(iter(run_ids)):
            # The exporter refuses a foreign calibration for the same reason.
            raise ValueError(
                f"calibration belongs to run {group.run_id}, not the "
                f"captured run {next(iter(run_ids))}"
            )
    alias = _clock_aliases(
        [source.meta for source in sources], group
    )
    joins, join_unavailable = _join_producers(sources)
    not_available.extend(join_unavailable)
    spans = _cross_process_spans(sources, joins, group, alias, not_available)
    timelines = timeline.build(sources, joins, group, alias, problems, not_available)
    transferred = [
        {
            key: value
            for key, value in item.items()
            if key != "request_raw" and key != "prepared_sequence"
        }
        for source in sources
        for item in source.transfers
    ]
    duplicate_inputs = {
        process_id: paths
        for capture in captures
        for process_id, paths in capture["duplicates"].items()
    }
    return {
        "schema": 1,
        "tool": "vllm.kv_profiling.summary",
        "run_id": next(iter(run_ids), None),
        "duplicate_inputs": duplicate_inputs,
        "sources": [
            {
                "source": source.index,
                "path": str(source.capture["path"]),
                "process_id": source.meta["process_id"],
                "role": source.role,
                "kv_role": source.context.get("kv_role"),
                "engine_id": source.engine_id,
                "backends": source.context.get("backend")
                or source.context.get("backends"),
                "configured_transport": source.context.get("transport"),
                "clock_id": source.meta["clock_id"],
                "events": len(source.events),
                "loss": source.capture["loss"],
                "loss_detail": source.capture["loss_detail"],
                "adapter_health": source.health,
            }
            for source in sources
        ],
        "transfers": transferred,
        "write_batches": [item for source in sources for item in source.write_batches],
        "post_groups": [item for source in sources for item in source.post_groups],
        "requests": [
            item for source in sources for item in source.requests
        ],
        "handshakes": [
            item for source in sources for item in source.handshakes
        ],
        "joins": joins,
        "timelines": timelines["timelines"],
        "kv_writes": timelines["kv_writes"],
        "cross_process_spans": spans,
        "aggregates": _aggregates(sources, [
            item for source in sources for item in source.transfers
        ]),
        "problems": problems,
        "not_available": not_available,
    }


def _fmt(value):
    return f"{value:,.0f}" if isinstance(value, (int, float)) else "-"


def _table(summary) -> str:
    lines = []
    transfers = summary["aggregates"]["transfers"]
    if transfers:
        lines.append(
            f"{'backend/transport':<32}{'count':>6}{'done':>6}{'fail':>6}"
            f"{'bytes':>15}{'p50 post->done':>17}{'p95':>10}"
        )
        for key in sorted(transfers):
            item = transfers[key]
            lines.append(
                f"{key:<32}{item['count']:>6}{item['completed']:>6}"
                f"{item['failed']:>6}{item['bytes']['sum']:>15}"
                f"{_fmt(item['posted_to_completed_ns']['p50']):>17}"
                f"{_fmt(item['posted_to_completed_ns']['p95']):>10}"
            )
    handshakes = summary["aggregates"]["handshakes"]
    if handshakes:
        lines.append(
            f"{'handshakes':<32}{'count':>6}{'p50 ns':>17}{'p95 ns':>12}"
        )
        for key in sorted(handshakes):
            item = handshakes[key]
            lines.append(
                f"{key:<32}{item['count']:>6}"
                f"{_fmt(item['duration_ns']['p50']):>17}"
                f"{_fmt(item['duration_ns']['p95']):>12}"
            )
    if summary["problems"]:
        lines.append(f"problems: {len(summary['problems'])}")
    if summary["not_available"]:
        lines.append(f"not_available: {len(summary['not_available'])}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", help="events.jsonl files or directories of them"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--calibration", action="append", default=[],
        help="saved ClockMapping JSON; enables cross-process spans",
    )
    parser.add_argument("--reference-clock")
    args = parser.parse_args(argv)
    summary = summarize(args.inputs, args.calibration, args.reference_clock)
    output = Path(args.output)
    if output.parent != Path(""):
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(_table(summary))


if __name__ == "__main__":
    main()
