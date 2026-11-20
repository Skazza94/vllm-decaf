# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request compute timelines and decode stall attribution (offline).

Built from the ``request_phase`` and ``step_span`` rows of the scheduler, worker and
frontend captures, joined by exact recorded identities: request IDs, the scheduler
generation a step marker carries, and the existing producer/consumer join. Every
interval is computed only between times of one clock domain (same boot ID, time
namespace and clock source) or between calibrated clocks, with the projection's
uncertainty; otherwise it is reported unavailable with its reason. Nothing is
clipped, estimated or invented: a metric the records cannot carry goes to
``unavailable`` (and to the summary's ``not_available``), an inconsistency to
``problems``.

Names: ``step_host_span`` is the host time between the connector receiving a step's
metadata and the runner's ``finish_forward``; it is shared by every request in that
step's batch and is not device time. What follows it up to the first output the
frontend sees (sampling, a simulator's deadline wait, output transport, frontend
processing) is ``after_step_to_first_token``, never charged to compute.
"""

from .clock import Projection
from .compute import scope_of
from .perfetto import _domain
from .schema import (
    ADMITTED_RESUMED,
    FINISH_ABORTED,
    FINISH_ERROR,
    MARK_EMPTY,
    Failure,
    RequestPhase,
    StepSubject,
)

_PHASE = {phase.value: phase.name.lower() for phase in RequestPhase}
_CLEANUP = frozenset(
    (
        Failure.NOTIFICATION_FAILED.value,
        Failure.LATE_NOTIFICATION.value,
        Failure.TRANSFER_RELEASE_FAILED.value,
        Failure.LEASE_EXPIRED.value,
    )
)
SCHEMA = 1
_AMBIGUOUS = object()  # a pairing that several candidates could satisfy


class _At:
    """A recorded time with the clock it was read on."""

    __slots__ = ("source", "ns")

    def __init__(self, source, ns):
        self.source = source
        self.ns = ns


class Clocks:
    """Differences between recorded times: raw inside one clock domain, projected
    (with uncertainty) between calibrated clocks, otherwise unavailable."""

    def __init__(self, group, alias):
        self.group = group
        self.alias = alias

    def _project(self, at):
        if self.group is None:
            return None
        clock = at.source.meta["clock_id"]
        return self.group.project(self.alias.get(clock, clock), at.ns)

    def delta(self, a, b):
        """``(b - a, uncertainty_ns, method)`` or None."""
        if a is None or b is None:
            return None
        if _domain(a.source.meta) == _domain(b.source.meta):
            return b.ns - a.ns, 0, "same_clock_domain"
        pa, pb = self._project(a), self._project(b)
        if pa is None or pb is None:
            return None
        return (
            round(pb.time_ns - pa.time_ns),
            pa.uncertainty_ns + pb.uncertainty_ns,
            "calibrated",
        )


def _identity(source):
    context = source.context
    keys = ("tp_rank", "pp_rank", "pcp_rank")
    if all(isinstance(context.get(key), int) for key in keys):
        return tuple(context[key] for key in keys)
    return None


class _Step:
    """One scheduler step as the workers recorded it: every row of every rank (a
    rank recorded twice keeps both observations)."""

    def __init__(self):
        self.ranks = {}  # identity -> [(begin_ns, end_ns, worker source), ...]


class _Span:
    """What one request occurrence owns: its scheduler rows lie between its first row
    and the next occurrence's first in the same capture (by sequence); what other
    processes recorded is attributed by time, against the occurrences of the same
    request ID that could be confused with it (its ``rivals``)."""

    def __init__(self, scheduler, total, lo, hi, t_lo, t_end):
        self.scheduler = scheduler
        self.total = total  # occurrences of this request ID in the scheduler capture
        self.key = None
        self.lo = lo
        self.hi = hi
        self.t_lo = t_lo  # when the scheduler queued it
        self.t_end = t_end  # when the scheduler finished it (None: unknown)
        # (other span, "before" | "after" | "concurrent"): the other occurrence
        # certainly ended before this one started, certainly started after this one
        # ended, or neither is certain (independent engines that overlap, a missing
        # finish, clocks that cannot be compared).
        self.rivals = []

    def has_sequence(self, row):
        return row["sequence"] >= self.lo and (
            self.hi is None or row["sequence"] < self.hi
        )


class Timelines:
    def __init__(self, sources, joins, group, alias, problems, not_available):
        self.sources = sources
        self.joins = joins
        self.clocks = Clocks(group, alias)
        self.problems = problems
        self.not_available = not_available
        self.schedulers = {}
        self.steps = {}
        self.windows = {}
        self.writes = {}
        self.out = []
        self._referenced = set()
        self._tables = {}
        self._edges = {}
        self._members = {}
        self._quality_cache = {}
        self._scope_workers = {}
        self._handoff_counts = {}

    # ------------------------------------------------------------------ helpers
    def _problem(self, source, kind, detail):
        self.problems.append(
            {
                "source": source.index,
                "process_id": source.meta["process_id"],
                "kind": kind,
                "detail": detail,
            }
        )

    def _quality(self, source):
        """Why this capture's rows cannot be taken as the whole of what happened:
        loss, a missing or incomplete end record, an unhealthy or unknown adapter."""
        cached = self._quality_cache.get(source.index)
        if cached is None:
            reasons = []
            if source.capture["loss"]:
                detail = source.capture["loss_detail"]
                if not detail["end_present"]:
                    reasons.append("no end record (capture completeness unknown)")
                elif not detail["end_complete"]:
                    reasons.append("the end record reports an incomplete capture")
                if detail["missing_sequences"]:
                    reasons.append("missing sequences")
                if any(detail["dropped"].values()):
                    reasons.append(f"dropped records {detail['dropped']}")
                if detail["unresolved_identities"]:
                    reasons.append("unresolved identities")
            reasons.extend(f"adapter: {reason}" for reason in source.health)
            cached = self._quality_cache[source.index] = reasons
        return cached

    # ---------------------------------------------------------------- indexing
    def _rows(self, source, kind, name):
        """The rows of one kind of this source recorded under this request ID, from an
        index built once (the scan per request would be quadratic)."""
        table = self._tables.get((source.index, kind))
        if table is None:
            table = {}
            for row in source.events:
                if row["kind"] == kind and row["request_ref"]:
                    table.setdefault(source._name(row["request_ref"]), []).append(row)
            self._tables[(source.index, kind)] = table
        return table.get(name, ())

    def _index(self):
        for source in self.sources:
            if source.role == "scheduler":
                self.schedulers[scope_of(source.meta["process_id"])] = source
        for source in self.sources:
            if source.role != "worker":
                continue
            for row in source.events:
                if row["kind"] != "step_span":
                    continue
                source.classified.add(id(row))
                if row["subject"] == StepSubject.STEP:
                    self._index_step(source, row)
                elif row["subject"] == StepSubject.WORKER_WINDOW:
                    self.windows.setdefault(source.index, []).append(row)
        for source in self.sources:
            for row in source.events:
                if row["kind"] in ("request_phase", "step_span"):
                    source.classified.add(id(row))
        self._index_writes()

    def _index_step(self, source, row):
        scheduler = self.schedulers.get(row["scope"])
        if scheduler is None:
            self._problem(
                source,
                "step_scope_unknown",
                f"step {row['step']} names a scheduler generation that is not in "
                "the input; the step is not joined",
            )
            return
        if scheduler.engine_id != source.engine_id or scheduler.context.get(
            "dp_rank"
        ) != source.context.get("dp_rank"):
            self._problem(
                source,
                "step_engine_mismatch",
                f"step {row['step']} of scope {row['scope']} was recorded by a "
                "worker of another engine or data-parallel rank than its scheduler",
            )
            return
        self._scope_workers.setdefault(row["scope"], {})[source.index] = source
        step = self.steps.setdefault((row["scope"], row["step"]), _Step())
        identity = _identity(source) or ("source", source.index)
        rows = step.ranks.setdefault(identity, [])
        if rows:
            self._problem(
                source,
                "duplicate_step_row",
                f"step {row['step']} recorded more than once by worker {identity}; "
                "every observation is kept and the step is not complete",
            )
        rows.append((row["begin_ns"], row["end_ns"], source))

    def _index_writes(self):
        """Mooncake's producer-side inventory: every write batch once, with its
        recorded call and completion times and its members, from the producer's own
        records (a consumer capture is not needed to list them)."""
        for source in self.sources:
            for batch in source.write_batches:
                key = f"{source.index}:{batch['batch']}"
                entry = {
                    "key": key,
                    "source": source.index,
                    "process_id": source.meta["process_id"],
                    "engine_id": batch.get("engine_id"),
                    "batch": batch["batch"],
                    "write": batch["write"],
                    "times": batch["times"],
                    "bytes": batch["bytes"],
                    "descriptors": batch["descriptors"],
                    "partition": batch["partition"]["status"],
                    "quality": self._quality(source),
                    "members": [],
                }
                for member in batch["members"]:
                    item = {
                        "request": member["request"],
                        "transfer_id": member["transfer_id"],
                        "ok": member["ok"],
                        "share_bytes": member["bytes"],
                        "share_descriptors": member["descriptors"],
                        "consumers": [],
                    }
                    entry["members"].append(item)
                    self._members.setdefault(
                        (member["request"], member["transfer_id"]), []
                    ).append((entry, item))
                self.writes[key] = entry

    def _expected(self, scheduler):
        parallel = scheduler.context.get("parallel")
        if not isinstance(parallel, dict):
            return None
        try:
            return {
                (tp, pp, pcp)
                for tp in range(parallel["tp"])
                for pp in range(parallel["pp"])
                for pcp in range(parallel["pcp"])
            }
        except (KeyError, TypeError):
            return None

    def _worker_sources(self, scheduler):
        return [
            source
            for source in self.sources
            if source.role == "worker"
            and source.engine_id == scheduler.engine_id
            and source.context.get("dp_rank") == scheduler.context.get("dp_rank")
        ]

    def _scoped_workers(self, scheduler):
        """The workers that recorded this scheduler generation's marker: the only
        evidence that links a worker's rows to it."""
        found = self._scope_workers.get(scope_of(scheduler.meta["process_id"]))
        return list(found.values()) if found else []

    # ------------------------------------------------------------------- steps
    def _step_view(self, scheduler, scope, number, phases, scheduled):
        """One step as a request sees it: every rank's observations, and the shared
        span only when exactly one row per expected rank is backed by healthy captures
        on one clock domain."""
        step = self.steps.get((scope, number))
        expected = self._expected(scheduler)
        view = {
            "step": number,
            "phases": phases,
            "scheduled_tokens": scheduled,
            "expected_ranks": None if expected is None else len(expected),
            "present_ranks": 0,
            "complete": False,
            "rank_spans_ns": {},
            "rank_observations": {},
            "host_span_ns": None,
            "begin_skew_ns": None,
            "end_skew_ns": None,
            "host_span_observed_ns": None,
            "shared_batch": True,
            "scheduler_to_step_begin_ns": None,
            "reasons": [],
            "quality": [],
            "_times": None,
            "_sources": [],
        }
        if step is None:
            view["reasons"].append("no worker recorded this step")
            return view
        view["present_ranks"] = len(step.ranks)
        for identity, rows in sorted(step.ranks.items(), key=str):
            view["rank_observations"][_key(identity)] = [
                {"begin_ns": begin, "end_ns": end, "span_ns": end - begin}
                for begin, end, _ in rows
            ]
            if len(rows) == 1:
                view["rank_spans_ns"][_key(identity)] = rows[0][1] - rows[0][0]
        workers = self._worker_sources(scheduler)
        structural = []
        if expected is None:
            structural.append(
                "the expected worker set is unknown (the scheduler capture has no "
                "parallel layout)"
            )
        else:
            have = {_identity(w) for w in workers}
            for identity in sorted(expected - set(step.ranks)):
                structural.append(
                    f"worker {_key(identity)}: "
                    + (
                        "capture present but no row for this step"
                        if identity in have
                        else "no capture in the input"
                    )
                )
            for identity in sorted(set(step.ranks) - expected, key=str):
                structural.append(
                    f"worker {_key(identity)}: a row from a rank the layout lacks"
                )
        for identity, rows in sorted(step.ranks.items(), key=str):
            if len(rows) > 1:
                structural.append(
                    f"worker {_key(identity)}: recorded {len(rows)} times, which "
                    "rank observation is the step's is ambiguous"
                )
        rows_all = [row for rows in step.ranks.values() for row in rows]
        domains = {_domain(source.meta) for _, _, source in rows_all}
        if len(domains) > 1:
            structural.append("the workers' clocks are not one clock domain")
        # Health matters for the captures that recorded this scheduler generation's
        # marker; an earlier generation's worker has no part in this step.
        sources = {id(source): source for _, _, source in rows_all}
        for source in sources.values():
            reasons = self._quality(source)
            if reasons:
                view["quality"].append(
                    {
                        "source": source.index,
                        "role": "worker",
                        "reasons": reasons,
                    }
                )
        view["_sources"] = list(sources.values())
        view["reasons"] = structural
        if rows_all and len(domains) == 1:
            begins = [begin for begin, _, _ in rows_all]
            ends = [end for _, end, _ in rows_all]
            view["host_span_observed_ns"] = max(ends) - min(begins)
        exactly_once = expected is not None and all(
            len(step.ranks.get(identity, ())) == 1 for identity in expected
        )
        if not structural and exactly_once:
            begins = [rows[0][0] for rows in step.ranks.values()]
            ends = [rows[0][1] for rows in step.ranks.values()]
            first = min(rows_all, key=lambda item: item[0])
            view["_times"] = (_At(first[2], min(begins)), _At(first[2], max(ends)))
            if not view["quality"]:
                view["complete"] = True
                view["host_span_ns"] = max(ends) - min(begins)
                view["begin_skew_ns"] = max(begins) - min(begins)
                view["end_skew_ns"] = max(ends) - min(ends)
            else:
                view["reasons"].append(
                    "a worker capture of this engine is incomplete or unhealthy: the "
                    "shared span is only observed"
                )
        return view

    # --------------------------------------------------------------- the build
    def build(self):
        self._index()
        collected = []
        for scope, scheduler in sorted(self.schedulers.items()):
            collected.extend(self._collect(scheduler, scope))
        self._relate_occurrences(collected)
        self._count_handoffs()
        for scheduler, scope, request, index, total, rows, span in collected:
            self.out.append(
                self._timeline(scheduler, scope, request, index, total, rows, span)
            )
        orphan_steps = [key for key in self.steps if key not in self._referenced]
        for key in orphan_steps:
            step = self.steps[key]
            source = next(iter(step.ranks.values()))[0][2]
            self._problem(
                source,
                "step_without_marker_row",
                f"step {key[1]} has worker rows but no scheduler phase row "
                "referencing it (a scheduler row may have been lost)",
            )
        # What the records could not carry, once per reason with the number of
        # timelines it concerns (the per-timeline ``unavailable`` has the detail).
        counts = {}
        for entry in self.out:
            for name, reason in entry["unavailable"].items():
                key = (name.split(" ")[0], reason)
                counts[key] = counts.get(key, 0) + 1
        for (name, reason), count in sorted(counts.items()):
            self.not_available.append(
                {"scope": f"timelines: {name} ({count} request(s))", "reason": reason}
            )
        return {"timelines": self.out, "kv_writes": list(self.writes.values())}

    def _collect(self, scheduler, scope):
        """Every request occurrence of one scheduler capture, with its span."""
        by_request = {}
        for row in sorted(scheduler.events, key=lambda row: row["sequence"]):
            if row["kind"] != "request_phase":
                continue
            name = scheduler._name(row["request_ref"])
            if name is None:
                self._problem(
                    scheduler,
                    "request_phase_unresolved",
                    "a request ID did not resolve",
                )
                continue
            by_request.setdefault(name, []).append(row)
        found = []
        for request, rows in by_request.items():
            occurrences = _occurrences(rows)
            for index, occurrence in enumerate(occurrences):
                lo = occurrence[0]["sequence"]
                hi = (
                    occurrences[index + 1][0]["sequence"]
                    if index + 1 < len(occurrences)
                    else None
                )
                start = next(
                    (r for r in occurrence if r["subject"] == RequestPhase.QUEUED),
                    occurrence[0],
                )
                end = next(
                    (r for r in occurrence if r["subject"] == RequestPhase.FINISHED),
                    None,
                )
                span = _Span(
                    scheduler,
                    len(occurrences),
                    lo,
                    hi,
                    _At(scheduler, start["time_ns"]),
                    None if end is None else _At(scheduler, end["time_ns"]),
                )
                span.key = (scheduler.index, request, index)
                found.append(
                    (
                        scheduler,
                        scope,
                        request,
                        index,
                        len(occurrences),
                        occurrence,
                        span,
                    )
                )
        return found

    def _relate_occurrences(self, collected):
        """How the occurrences of one request ID in captures of the same KV role relate:
        one that certainly ended before another started is sequential (a restart, the
        next use of a completed ID); anything else is concurrent, and what another
        process recorded cannot be told apart between concurrent occurrences."""
        by_request = {}
        for item in collected:
            by_request.setdefault(item[2], []).append(item)
        for items in by_request.values():
            for scheduler, _, _, _, _, _, span in items:
                for other, _, _, _, _, _, other_span in items:
                    if other_span is span:
                        continue
                    if other.context.get("kv_role") != scheduler.context.get("kv_role"):
                        continue
                    relation = "concurrent"
                    if span.t_end is not None:
                        gap = self.clocks.delta(span.t_end, other_span.t_lo)
                        if gap is not None and gap[0] >= gap[1] and gap[0] >= 0:
                            relation = "after"
                    if relation == "concurrent" and other_span.t_end is not None:
                        gap = self.clocks.delta(other_span.t_end, span.t_lo)
                        if gap is not None and gap[0] >= gap[1] and gap[0] >= 0:
                            relation = "before"
                    span.rivals.append((other_span, relation))

    def _count_handoffs(self):
        """How many incoming handoffs name each (producer engine, transfer ID,
        request): a Mooncake write can be linked to a consumer occurrence only when
        exactly one handoff claims it."""
        for source in self.sources:
            if not source.mooncake or source.context.get("kv_role") == "kv_producer":
                continue
            for entry in source.requests:
                for handoff in entry["handoffs"]:
                    if handoff["direction"] != "incoming":
                        continue
                    key = (
                        handoff["producer_engine"],
                        handoff.get("transfer_id"),
                        entry["request"],
                    )
                    self._handoff_counts[key] = self._handoff_counts.get(key, 0) + 1

    def _classify(self, at, lo, hi, lo_closed):
        """Whether a recorded time lies between ``lo`` and ``hi`` (None: unbounded),
        half-open on the side ``lo_closed`` names: True, False, or None when the clocks
        cannot tell (not comparable, or the time is within the uncertainty of a
        boundary so that either owner is possible)."""
        if lo is not None:
            gap = self.clocks.delta(lo, at)
            if gap is None:
                return None
            distance, uncertainty, _ = gap
            if lo_closed:
                inside, outside = distance >= uncertainty, distance < -uncertainty
            else:
                inside, outside = distance > uncertainty, distance <= -uncertainty
            if outside:
                return False
            if not inside:
                return None
        if hi is not None:
            gap = self.clocks.delta(at, hi)
            if gap is None:
                return None
            distance, uncertainty, _ = gap
            if lo_closed:
                inside, outside = distance > uncertainty, distance <= -uncertainty
            else:
                inside, outside = distance >= uncertainty, distance < -uncertainty
            if outside:
                return False
            if not inside:
                return None
        return True

    def _started(self, at, span):
        """Whether a time is at or after the occurrence's queueing (closed)."""
        return self._classify(at, span.t_lo, None, lo_closed=True)

    def _inside(self, source, row, span):
        """Whether a row of another process belongs to this occurrence: True/False, or
        None when it cannot be told. It must come at or after the queueing; a
        sequential successor that had started takes it (the boundary row is the
        successor's); a concurrent occurrence of the same ID that may have started
        leaves it unowned."""
        at = _At(source, row["time_ns"])
        own = self._started(at, span)
        if own is not True:
            return own
        for rival, relation in span.rivals:
            started = self._started(at, rival)
            if relation == "after":
                if started is True:
                    return False
                if started is None:
                    return None
            elif relation == "concurrent" and started is not False:
                return None
        return True

    def _arrival_owned(self, at, span):
        """Whether an API entry belongs to this occurrence: it precedes the queueing
        and is not an earlier request's (after the start of a sequential predecessor)
        nor possibly a concurrent occurrence's."""
        before = self._classify(at, None, span.t_lo, lo_closed=False)
        if before is not True:
            return before
        for rival, relation in span.rivals:
            if relation == "before":
                after_start = self._classify(at, rival.t_lo, None, lo_closed=False)
                if after_start is not True:
                    return after_start
            elif relation == "concurrent":
                could = self._classify(at, None, rival.t_lo, lo_closed=False)
                if could is not False:
                    return None
        return True

    # ------------------------------------------------------------- one request
    def _timeline(self, scheduler, scope, request, index, total, rows, span):
        phases = {name: [] for name in _PHASE.values()}
        for row in rows:
            name = _PHASE.get(row["subject"])
            if name is None:
                self._problem(
                    scheduler, "unknown_request_phase", f"subject {row['subject']}"
                )
                continue
            phases[name].append(row)
        kv_role = scheduler.context.get("kv_role")
        unavailable = {}
        problems = []
        entry = {
            "schema": SCHEMA,
            "node_role": {
                "kv_producer": "producer",
                "kv_consumer": "consumer",
                "kv_both": "both",
            }.get(kv_role, "none"),
            "engine_id": scheduler.engine_id,
            "dp_rank": scheduler.context.get("dp_rank"),
            "source": scheduler.index,
            "process_id": scheduler.meta["process_id"],
            "request": request,
            "request_occurrence": index,
            "repeated_id": total > 1,
        }
        queued = phases["queued"][0] if phases["queued"] else None
        if queued is None:
            problems.append("no QUEUED row: the capture started after the request")
        prepared = phases["load_prepared"][0] if phases["load_prepared"] else None
        admissions = phases["admitted"]
        dones = phases["prefill_done"]
        # The first-token cycle ends at the first PREFILL_DONE; later admissions
        # (preemption after a token) stay visible and outside the first-token stall.
        first_done = dones[0] if dones else None
        in_cycle = [
            a
            for a in admissions
            if first_done is None or a["sequence"] < first_done["sequence"]
        ]
        later = [a for a in admissions if a not in in_cycle]
        cycle_start = in_cycle[-1] if in_cycle else None
        preempted_rows = [
            p
            for p in phases["preempted"]
            if first_done is None or p["sequence"] < first_done["sequence"]
        ]
        chunk_rows = [
            c
            for c in phases["prefill_chunk"]
            if cycle_start is not None
            and c["sequence"] > cycle_start["sequence"]
            and (first_done is None or c["sequence"] < first_done["sequence"])
        ]
        finished = phases["finished"][0] if phases["finished"] else None
        # The request's steps of the first-token cycle, one per step number.
        step_rows = {}
        for row in (
            ([cycle_start] if cycle_start else [])
            + chunk_rows
            + ([first_done] if first_done else [])
        ):
            number = row["step"]
            kind = _PHASE[row["subject"]]
            names, scheduled, first_row = step_rows.setdefault(
                number, ([], row["scheduled_tokens"], row)
            )
            if kind not in names:
                names.append(kind)
        views = []
        for number, (names, scheduled, first_row) in step_rows.items():
            self._referenced.add((scope, number))
            view = self._step_view(scheduler, scope, number, names, scheduled)
            if view["_times"] is not None:
                lag = self.clocks.delta(
                    _At(scheduler, first_row["time_ns"]), view["_times"][0]
                )
                view["scheduler_to_step_begin_ns"] = None if lag is None else lag[0]
            views.append(view)
        for row in rows:
            if row["subject"] in (
                RequestPhase.ADMITTED,
                RequestPhase.PREFILL_CHUNK,
                RequestPhase.PREFILL_DONE,
                RequestPhase.PREEMPTED,
            ):
                self._referenced.add((scope, row["step"]))
        # Frontend.
        frontend = self._frontend(scheduler, request, span, unavailable)
        entry["api"] = frontend
        # Scheduler times.
        received = self._received(scheduler, request, prepared, span)
        entry["scheduler"] = {
            "queued_ns": None if queued is None else queued["time_ns"],
            "load_prepared_ns": None if prepared is None else prepared["time_ns"],
            "kv_received_ns": None if received is None else received["time_ns"],
            "admissions": [_admission(a) for a in admissions],
            "prefill_done_ns": None if first_done is None else first_done["time_ns"],
            "preemptions": len(phases["preempted"]),
            "finished_ns": None if finished is None else finished["time_ns"],
            "finish_state": _finish_state(finished),
            "later_admissions": [_admission(a) for a in later],
        }
        computed = None if cycle_start is None else cycle_start["computed_tokens"]
        prefill = None if cycle_start is None else cycle_start["prefill_tokens"]
        entry["tokens"] = {
            "prompt": None if queued is None else queued["prompt_tokens"],
            "prefill_effective": prefill,
            "computed_at_admission": computed,
            "tokens_remaining_at_admission": (
                None if computed is None or prefill is None else prefill - computed
            ),
            "external": None if prepared is None else prepared["external_tokens"],
            "local_cached_at_load": None
            if prepared is None
            else prepared["computed_tokens"],
            "output_first": frontend["first_token_output_tokens"],
            "output_total": frontend["output_tokens"],
        }
        entry["kv"] = self._kv(
            scheduler,
            request,
            span,
            prepared,
            received,
            cycle_start,
            admissions,
            unavailable,
            index,
        )
        entry["steps"] = [
            {k: v for k, v in view.items() if not k.startswith("_")} for view in views
        ]
        done_view = next((v for v in views if "prefill_done" in v["phases"]), None)
        start_view = next((v for v in views if "admitted" in v["phases"]), None)
        intervals, stall = self._intervals(
            scheduler,
            queued,
            prepared,
            received,
            in_cycle,
            preempted_rows,
            cycle_start,
            first_done,
            views,
            start_view,
            done_view,
            frontend,
            unavailable,
        )
        # Capture quality: every capture a figure depends on. A figure that sums rows
        # (a stall, a complete step) is withheld, the individual observations stay.
        quality = []
        seen = set()

        def add(source, role):
            reasons = self._quality(source)
            if reasons and source.index not in seen:
                seen.add(source.index)
                quality.append(
                    {"source": source.index, "role": role, "reasons": reasons}
                )

        add(scheduler, "scheduler")
        if frontend["source"] is not None:
            add(self.sources[frontend["source"]], "frontend")
        for view in views:
            for source in view["_sources"]:
                add(source, "worker")
        entry["quality"] = quality
        if quality:
            entry["state"] = "observed_only"
            entry["intervals_ns"] = None
            entry["stall"] = None
            entry["intervals_ns_observed"] = intervals
            entry["stall_observed"] = stall
            unavailable["totals"] = (
                "a capture this timeline depends on has loss or an unhealthy adapter "
                "(see quality): a row may be missing; the figures are reported under "
                "*_observed names"
            )
        else:
            entry["state"] = _state(queued, cycle_start, first_done, finished)
            entry["intervals_ns"] = intervals
            entry["stall"] = stall
        entry["decode_window"] = self._decode_window(
            scheduler, frontend, finished, unavailable
        )
        entry["cross_node"] = self._cross_node(
            scheduler, request, span, queued, prepared, received, unavailable
        )
        entry["problems"] = problems
        entry["unavailable"] = unavailable
        for reason in {*problems}:
            self._problem(scheduler, "timeline", f"{request!r}: {reason}")
        for view in entry["steps"]:
            if not view["complete"]:
                for reason in view["reasons"]:
                    unavailable.setdefault(f"step {view['step']}", reason)
                    if (
                        "no capture" in reason
                        or "no row" in reason
                        or "no worker" in reason
                    ):
                        self._problem(
                            scheduler,
                            "rank_missing",
                            f"{request!r} step {view['step']}: {reason}",
                        )
        return entry

    # ------------------------------------------------------------- the frontend
    def _frontend(self, scheduler, request, span, unavailable):
        result = {
            "external_request": None,
            "arrival_ns": None,
            "first_token_output_ns": None,
            "first_token_output_tokens": None,
            "output_finished_ns": None,
            "output_finished_status": None,
            "output_tokens": None,
            "source": None,
            "_arrival": None,
            "_first": None,
            "_finished": None,
        }
        candidates = []
        ambiguous = False
        for source in self.sources:
            if source.role != "frontend":
                continue
            frontend_role = source.context.get("kv_role")
            if (
                frontend_role is not None
                and scheduler.context.get("kv_role") is not None
                and frontend_role != scheduler.context.get("kv_role")
            ):
                continue  # a prefill node's frontend is no candidate for decode
            rows = []
            for row in self._rows(source, "request_phase", request):
                inside = self._inside(source, row, span)
                if inside is None:
                    ambiguous = True
                elif inside:
                    rows.append(row)
            if rows:
                candidates.append((source, rows))
        if ambiguous and not candidates:
            unavailable["frontend"] = (
                "this request ID occurs more than once and the frontend's clock cannot "
                "be compared with the scheduler's: its rows cannot be ascribed to one "
                "occurrence"
            )
            return result
        if not candidates:
            unavailable["frontend"] = (
                "no frontend capture recorded this occurrence of the request ID"
            )
            return result
        kv_role = scheduler.context.get("kv_role")
        if len(candidates) > 1:
            same = [c for c in candidates if c[0].context.get("kv_role") == kv_role]
            if len(same) == 1:
                candidates = same
        if len(candidates) > 1:
            unavailable["frontend"] = (
                f"{len(candidates)} frontend captures recorded this request ID and "
                "their KV role does not select one"
            )
            return result
        source, rows = candidates[0]
        firsts = [r for r in rows if r["subject"] == RequestPhase.FIRST_TOKEN_OUTPUT]
        ends = [r for r in rows if r["subject"] == RequestPhase.OUTPUT_FINISHED]
        if len(firsts) > 1 or len(ends) > 1:
            unavailable["frontend"] = (
                "the frontend recorded this request ID more than once inside this "
                "occurrence; its rows cannot be ascribed to one occurrence"
            )
            return result
        result["source"] = source.index
        if firsts:
            result["first_token_output_ns"] = firsts[0]["time_ns"]
            result["first_token_output_tokens"] = firsts[0]["output_tokens"]
            result["_first"] = _At(source, firsts[0]["time_ns"])
        else:
            unavailable["first_token_output"] = (
                "no non-empty output reached the frontend"
            )
        if ends:
            result["output_finished_ns"] = ends[0]["time_ns"]
            result["output_finished_status"] = _finish_state(ends[0])
            result["output_tokens"] = ends[0]["output_tokens"]
            result["_finished"] = _At(source, ends[0]["time_ns"])
        owned = []
        candidates, ambiguous = self._arrivals(source, request)
        for external, entered in candidates:
            verdict = self._arrival_owned(_At(source, entered), span)
            if verdict is True:
                owned.append((external, entered))
            elif verdict is None:
                ambiguous = True
        if len(owned) == 1 and not ambiguous:
            result["external_request"] = owned[0][0]
            result["arrival_ns"] = owned[0][1]
            result["_arrival"] = _At(source, owned[0][1])
        else:
            unavailable["api_arrival"] = (
                "no recorded API entry of this ID is certainly this occurrence's "
                "(none in its window, several, or within the clocks' uncertainty of "
                "another occurrence's)"
                if owned or ambiguous
                else "the request ID chain from the engine ID to the serving layer's "
                "entry is missing in the frontend capture, or belongs to another "
                "request"
            )
        return result

    def _bind_edges(self, source):
        """The frontend's recorded bind rows of one capture, indexed by stage: engine
        binds by external ID, parent-to-item binds by item and by parent, API entries
        by parent, entry headers by base ID, and child-to-parent engine edges."""
        cached = self._edges.get(source.index)
        if cached is not None:
            return cached
        index = {
            "engine_binds": {},  # external ID -> [(time, engine ID)]
            "to_external": {},  # engine ID -> [(external ID, time)]
            "to_parent_engine": {},  # child engine ID -> [parent engine ID]
            "item_rows": {},  # item ID -> [(time, parent ID)]
            "items_of": {},  # parent ID -> [(time, item ID)]
            "entries": {},  # parent ID -> [(time, base ID)]
            "headers": {},  # base ID -> [time]
        }
        for row in source.events:
            if row["kind"] != "request_bind" or not row["request_ref"]:
                continue
            name = source._name(row["request_ref"])
            target = source._name(row["handoff_ref"])
            time, subject = row["time_ns"], row["subject"]
            if subject == 1:
                index["headers"].setdefault(name, []).append(time)
            elif subject == 2 and target is not None:
                index["entries"].setdefault(name, []).append((time, target))
            elif subject == 3 and target is not None:
                index["item_rows"].setdefault(name, []).append((time, target))
                index["items_of"].setdefault(target, []).append((time, name))
            elif subject == 4 and target is not None:
                index["engine_binds"].setdefault(target, []).append((time, name))
                index["to_external"].setdefault(name, []).append((target, time))
            elif subject == 5 and target is not None:
                index["to_parent_engine"].setdefault(name, []).append(target)
        for key in ("engine_binds", "item_rows", "items_of", "entries"):
            for rows in index[key].values():
                rows.sort()
        index["blocks"] = {}
        self._edges[source.index] = index
        return index

    @staticmethod
    def _unindexed(index, parent):
        """Whether some entry of this parent has no first-prompt bind after it: a call
        that bound its engine directly under the parent's ID (a parent can be used both
        ways, so what the other calls did says nothing about this one). Entries are
        matched in time with the first-prompt binds that follow them, one to one; the
        totals are not compared, because a declaration that belongs to a call whose
        entry was not recorded would make them look balanced."""
        entries = sorted(t for t, _ in index["entries"].get(parent, ()))
        firsts = sorted(
            t for t, item in index["items_of"].get(parent, ()) if _item_index(item) == 0
        )
        j = 0
        for entry in entries:
            while j < len(firsts) and firsts[j] < entry:
                j += 1
            if j >= len(firsts):
                return True
            j += 1
        return False

    @staticmethod
    def _overbound(index, external):
        """Whether an engine bind names this prompt ID before a prompt was declared for
        it: every indexed prompt is declared first and has at most one engine bind, so
        the k-th engine bind (in time) must not precede the k-th declaration. A bind
        that does is a direct call's, whose entry may not have been recorded; counting
        a later declaration against it (a call that began after, or one whose bind
        the capture ended before) would hide that, so the order is compared, not the
        totals."""
        cached = index.setdefault("overbound", {})
        found = cached.get(external)
        if found is None:
            declared = sorted(t for t, _ in index["item_rows"].get(external, ()))
            bound = sorted(t for t, _ in index["engine_binds"].get(external, ()))
            found = cached[external] = any(
                k >= len(declared) or declared[k] > time for k, time in enumerate(bound)
            )
        return found

    def _call_blocks(self, index, parent):
        """Which API entry each engine bind of a parent's prompts belongs to, decided
        only where the records leave no other possibility.

        A call records its entry, then one parent-to-item bind per prompt, and the
        engine binds of its prompts come after all of its item binds (the engines are
        asked only once the prompt loop is done). The events of one parent are swept
        in time and cut into blocks: a block is open from its first event until its
        engine binds balance its item binds (its entries, for a parent with no item
        stage). A block holding exactly one entry and no item ID twice belongs to that
        one call, so all its engine binds have that entry. A block with several
        entries could give any of its prompts to any of them (ambiguous); a block with
        none (an entry that was not recorded), or one that never closes, assigns
        nothing. Nothing is guessed: the unavailable arrival is the scheduler-queue
        fallback."""
        cached = index["blocks"].get(parent)
        if cached is not None:
            return cached
        items = index["items_of"].get(parent, ())
        # With an item stage the engine binds name the prompt IDs; without one they
        # name the parent. A client ID can equal another call's generated prompt ID
        # (clients ``batch`` and ``batch-0``), so the parent's own ID is never mixed
        # with prompt IDs, and a parent with no item stage whose ID is also some
        # call's prompt ID cannot tell its engine binds from that call's.
        externals = {item for _, item in items} if items else {parent}
        events = [(t, 0, (t, base)) for t, base in index["entries"].get(parent, ())]
        events += [(t, 1, item) for t, item in items]
        for external in externals:
            events += [
                (t, 2, (t, engine))
                for t, engine in index["engine_binds"].get(external, ())
            ]
        events.sort(key=lambda event: (event[0], event[1]))
        result = {}
        withhold = (
            # A parent used both ways (prompts declared under it and engine binds
            # directly under its ID beyond any prompts of that ID): the two kinds of
            # call cannot be counted apart, and a missing entry would make the counts
            # look balanced.
            (bool(items) and self._overbound(index, parent))
            # A prompt ID of this parent is also the ID of a parent used directly, or
            # has more engine binds than declarations: the surplus are direct calls'.
            or any(
                (item in index["entries"] and self._unindexed(index, item))
                or self._overbound(index, item)
                for _, item in items
            )
            # This parent's own ID is a prompt ID and some call used it directly.
            or (
                parent in index["item_rows"]
                and (self._unindexed(index, parent) or self._overbound(index, parent))
            )
        )
        if withhold:
            # Nothing is assigned: the engine binds of the prompts and of the parent's
            # own ID are all ambiguous.
            for external in {item for _, item in items} | {parent}:
                for key in index["engine_binds"].get(external, ()):
                    result[key] = _AMBIGUOUS
            index["blocks"][parent] = result
            return result
        if not items:
            # No item stage: each call is an entry then one engine bind under the
            # parent's ID. A bind pairs with the entry before it when exactly one is
            # waiting; with none waiting it is an entry-less call's (nothing assigned,
            # and it takes no entry); with several waiting it is ambiguous, and each
            # ambiguous bind consumes one unknown waiting entry until all are
            # accounted for.
            pending, unknown = [], 0
            for _, kind, key in events:
                if kind == 0:
                    pending.append(key)
                elif kind == 2:
                    if not pending:
                        result[key] = None
                    elif len(pending) == 1 and unknown == 0:
                        result[key] = pending.pop()
                    else:
                        result[key] = _AMBIGUOUS
                        unknown += 1
                        if unknown >= len(pending):
                            pending, unknown = [], 0
            index["blocks"][parent] = result
            return result
        entries, item_ids, engines = [], [], []

        def close(complete):
            if not engines:
                return
            valid = complete and len(set(item_ids)) == len(item_ids)
            if valid and len(entries) == 1:
                verdict = entries[0]
            elif len(entries) > 1 or (entries and not valid):
                verdict = _AMBIGUOUS
            else:
                verdict = None
            for key in engines:
                result[key] = verdict

        for _, kind, key in events:
            if kind == 0:
                entries.append(key)
            elif kind == 1:
                item_ids.append(key)
            else:
                engines.append(key)
                # Open until the engine binds balance the item binds and every entry has
                # its first prompt (an entry with no item bind yet is a call that has
                # not begun to bind prompts: its events are still to come).
                balanced = (
                    len(engines) == len(item_ids)
                    and sum(1 for item in item_ids if _item_index(item) == 0)
                    >= len(entries)
                    if item_ids
                    else len(engines) == len(entries)
                )
                if balanced:
                    close(True)
                    entries, item_ids, engines = [], [], []
        close(False)  # a block that never closed
        index["blocks"][parent] = result
        return result

    def _arrivals(self, source, engine_id):
        """The serving-layer entry of this engine request, matched through the call it
        belongs to (see ``_call_blocks``) and not through the external ID alone: a
        client ID reused by sequential or overlapping requests has several entries and
        several engine binds, and only a call that no other call's events interleave
        with identifies its own.

        Returns the candidates [(external ID, entry time)] and whether the match was
        ambiguous."""
        index = self._bind_edges(source)
        node, hops = engine_id, 0
        while hops < 8:
            parents = index["to_parent_engine"].get(node, ())
            if len(parents) != 1 or parents[0] == node:
                break
            node, hops = parents[0], hops + 1
        candidates, ambiguous = [], False
        for external, bind_time in index["to_external"].get(node, ()):
            rows = index["item_rows"].get(external)
            if rows and (
                (external in index["entries"] and self._unindexed(index, external))
                or self._overbound(index, external)
            ):
                ambiguous = True  # a prompt ID that is also a parent used directly
                continue
            parent = rows[0][1] if rows else external
            entry = self._call_blocks(index, parent).get((bind_time, node))
            if entry is _AMBIGUOUS:
                ambiguous = True
                continue
            if entry is None:
                continue
            entered, base = entry
            headers = [t for t in index["headers"].get(base, ()) if t <= entered]
            candidates.append((external, max(headers) if headers else entered))
        return sorted(set(candidates), key=lambda item: item[1]), ambiguous

    # ----------------------------------------------------------------- the KV
    def _received(self, scheduler, request, prepared, span):
        if prepared is None:
            return None
        rows = [
            row
            for row in self._rows(scheduler, "complete_observed", request)
            if row["subject"] == 2
            and row["sequence"] > prepared["sequence"]
            and span.has_sequence(row)
        ]
        return min(rows, key=lambda r: r["sequence"]) if rows else None

    def _failures(self, scheduler, request, span, unavailable):
        """The request-scoped failures inside this occurrence, and apart from them
        those of workers with no marker of this scheduler generation: visible, but with
        nothing linking them to this scheduler they are not assigned to the request."""
        found, unlinked = [], []
        linked = {source.index for source in self._scoped_workers(scheduler)}
        for source in [scheduler, *self._worker_sources(scheduler)]:
            for row in self._rows(source, "failed", request):
                if row["subject"] != 0 or row["status"] in _CLEANUP:
                    continue
                if source is scheduler:
                    inside = span.has_sequence(row)
                else:
                    inside = self._inside(source, row, span)
                if inside is None:
                    unavailable["kv_failures"] = (
                        "a worker's failure row cannot be placed in one occurrence "
                        "of this request ID (clocks, or a boundary's uncertainty)"
                    )
                    continue
                if not inside:
                    continue
                try:
                    code = Failure(row["status"]).name
                except ValueError:
                    code = f"status_{row['status']}"
                item = {"code": code, "time_ns": row["time_ns"], "source": source.index}
                if source is scheduler or source.index in linked:
                    found.append(item)
                else:
                    unlinked.append(item)
        if unlinked:
            unavailable["kv_failures_unlinked"] = (
                "a worker failure inside this occurrence comes from a worker that "
                "recorded no marker of this scheduler generation: nothing links it to "
                "this request, so it is listed apart and not counted"
            )
        return found, unlinked

    def _kv(
        self,
        scheduler,
        request,
        span,
        prepared,
        received,
        cycle_start,
        admissions,
        unavailable,
        index,
    ):
        failures, unlinked = self._failures(scheduler, request, span, unavailable)
        failed_receive = received is not None and received["status"] != 0
        computed = None if cycle_start is None else cycle_start["computed_tokens"]
        external = None if prepared is None else prepared["external_tokens"]
        recomputed = None
        if prepared is not None:
            if failed_receive or failures:
                # Recomputation is shown by an admission after the load that starts
                # below the position a successful load resumes at. On the ordinary
                # path the scheduler resumes at the cached prefix (what the local
                # cache held when the connector was asked plus the external tokens,
                # which are incremental to it) but never past the last prompt token,
                # which it keeps for itself: min(local + external, prefill - 1). A
                # replay policy moves that position back, and without the local count
                # the records cannot establish it either way.
                # The admission that pairs with the load is the first one after it in
                # its cycle, whichever later admission finishes the prefill.
                local = prepared["computed_tokens"]
                after_load = next(
                    (a for a in admissions if a["sequence"] > prepared["sequence"]),
                    None,
                )
                resumed_at = (
                    None if after_load is None else after_load["computed_tokens"]
                )
                prefill = None if after_load is None else after_load["prefill_tokens"]
                if after_load is None:
                    recomputed = False  # no admission followed the load: none shown
                elif scheduler.context.get("bounded_replay"):
                    recomputed = None
                    unavailable["recomputation"] = (
                        "a replay policy moves the position a successful load "
                        "resumes at: recomputation cannot be established"
                    )
                elif (
                    local is None
                    or external is None
                    or resumed_at is None
                    or prefill is None
                ):
                    recomputed = None  # the records cannot establish it
                else:
                    recomputed = resumed_at < min(local + external, prefill - 1)
                outcome = "load_failed_recomputed" if recomputed else "load_failed"
            elif received is not None:
                outcome = "received"
            else:
                outcome = "unknown"
        elif computed:
            outcome = "local_hit"
        else:
            outcome = "none"
        kv = {
            "outcome": outcome,
            "receive_reported_failed": failed_receive if received is not None else None,
            "recomputation_shown": recomputed,
            "failures": failures,
            "failures_unlinked": unlinked,
            # The invalidated token count is a scheduler-internal figure no hook
            # sees; tokens are never attributed to a failure from the admission
            # counts alone (a successful load can also leave tokens to recompute).
            "tokens_attributed_to_failure": None,
        }
        if outcome in ("load_failed", "load_failed_recomputed"):
            unavailable["tokens_attributed_to_failure"] = (
                "the invalidated token count is not recorded; see "
                "tokens_remaining_at_admission"
            )
        kv["writes"] = self._writes(scheduler, request, span, index, unavailable)
        return kv

    def _writes(self, scheduler, request, span, index, unavailable):
        """Mooncake: the producer write batch(es) this consumer occurrence is a member
        of, found through the producer engine, the transfer ID of this occurrence's
        incoming handoff and the batch's member rows (decode request and transfer ID).
        The batches themselves are listed once in ``kv_writes``; a write is linked only
        when exactly one incoming handoff names its (producer engine, transfer ID,
        request): a reused identity cannot say which occurrence owns the write."""
        if not scheduler.mooncake or scheduler.context.get("kv_role") == "kv_producer":
            return []
        handoffs = [
            handoff
            for entry in scheduler.requests
            if entry["request"] == request
            for handoff in entry["handoffs"]
            if handoff["direction"] == "incoming"
            and handoff.get("sequence") is not None
            and span.lo <= handoff["sequence"]
            and (span.hi is None or handoff["sequence"] < span.hi)
        ]
        refs = []
        for handoff in handoffs:
            transfer = handoff["transfer_id"]
            for write, member in self._members.get((request, transfer), ()):
                if write["engine_id"] != handoff["producer_engine"]:
                    continue
                claims = self._handoff_counts.get(
                    (handoff["producer_engine"], transfer, request), 0
                )
                if claims != 1:
                    unavailable["kv_write_link"] = (
                        f"{claims} handoffs name this producer engine, transfer ID and "
                        "request: the write cannot be ascribed to one consumer "
                        "occurrence (it stays in kv_writes without a link)"
                    )
                    continue
                member["consumers"].append(
                    {
                        "source": scheduler.index,
                        "request": request,
                        "request_occurrence": index,
                    }
                )
                refs.append(
                    {
                        "key": write["key"],
                        "transfer_id": transfer,
                        "share_bytes": member["share_bytes"],
                        "share_descriptors": member["share_descriptors"],
                    }
                )
        return refs

    # --------------------------------------------------------------- intervals
    def _intervals(
        self,
        scheduler,
        queued,
        prepared,
        received,
        in_cycle,
        preempted_rows,
        cycle_start,
        first_done,
        views,
        start_view,
        done_view,
        frontend,
        unavailable,
    ):
        delta = self.clocks.delta

        def sched(row):
            return None if row is None else _At(scheduler, row["time_ns"])

        def pair(a, b, name, why=None):
            result = delta(a, b)
            if result is None:
                if a is not None and b is not None:
                    unavailable.setdefault(
                        name,
                        why
                        or "the two times are not on one clock domain or calibration",
                    )
                return None
            return result

        q, lp, rc = sched(queued), sched(prepared), sched(received)
        adm = sched(cycle_start)
        first_adm = sched(in_cycle[0] if in_cycle else None)
        iv = {}

        def put(name, result):
            iv[name] = None if result is None else result[0]
            if result is not None and result[1]:
                iv[name + "_uncertainty"] = result[1]

        put("api_to_queued", pair(frontend["_arrival"], q, "api_to_queued"))
        put("queued_to_load_prepared", pair(q, lp, "queued_to_load_prepared"))
        put("load_prepared_to_received", pair(lp, rc, "load_prepared_to_received"))
        put("received_to_admitted", pair(rc, adm, "received_to_admitted"))
        put("queued_to_admitted", pair(q, adm, "queued_to_admitted"))
        if start_view and start_view["_times"] and done_view and done_view["_times"]:
            put(
                "prefill_window",
                pair(start_view["_times"][0], done_view["_times"][1], "prefill_window"),
            )
        else:
            iv["prefill_window"] = None

        def span(view):
            # A view whose workers are only observed (an unhealthy capture) still
            # has its observed span; the timeline reports it as observed.
            if view is None:
                return None
            if view["host_span_ns"] is not None:
                return view["host_span_ns"]
            return view["host_span_observed_ns"] if view["_times"] is not None else None

        sums = [span(v) for v in views]
        iv["prefill_steps_host_span_sum"] = (
            sum(sums) if views and all(s is not None for s in sums) else None
        )
        # The first and the last step of the cycle are different steps when the
        # prefill ran in several chunks.
        iv["first_step_host_span"] = span(start_view)
        iv["prefill_done_step_host_span"] = span(done_view)
        if done_view and done_view["_times"]:
            put(
                "after_step_to_first_token_output",
                pair(
                    done_view["_times"][1],
                    frontend["_first"],
                    "after_step_to_first_token_output",
                ),
            )
        else:
            iv["after_step_to_first_token_output"] = None
        lag = None
        if start_view and start_view["_times"]:
            lag = pair(adm, start_view["_times"][0], "scheduler_to_step_begin")
        iv["scheduler_to_step_begin"] = None if lag is None else lag[0]
        # Stall attribution over basis start -> first token output; every interval
        # between two recorded boundaries is counted in exactly one bucket.
        stall = {
            "basis": None,
            "end": "first_token_output",
            "pre_engine_ns": None,
            "queue_ns": None,
            "waiting_for_kv_ns": None,
            "earlier_admissions_ns": None,
            "preempted_ns": None,
            "step_host_span_ns": None,
            "between_steps_ns": None,
            "after_step_to_first_token_ns": None,
            "other_ns": None,
            "complete": False,
        }
        if frontend["_first"] is None:
            unavailable.setdefault("stall", "no first token output at the frontend")
            return iv, stall
        start = frontend["_arrival"]
        stall["basis"] = "api_arrival" if start is not None else "scheduler_queued"
        if start is None:
            start = q
        total = pair(start, frontend["_first"], "stall")
        if (
            total is None
            or cycle_start is None
            or first_done is None
            or not (views and all(v["_times"] is not None for v in views))
        ):
            if total is not None:
                unavailable.setdefault(
                    "stall",
                    "a step of the first-token cycle is incomplete or missing; "
                    "no bucket is built on it",
                )
            return iv, stall
        pre = iv["api_to_queued"] if stall["basis"] == "api_arrival" else 0
        if pre is None:
            return iv, stall
        # Before the first admission: the scheduler's queue, then (a remote load) the
        # wait for the KV, then the queue again until the load's request is admitted.
        if prepared is not None:
            waiting = iv["load_prepared_to_received"]
            head = iv["queued_to_load_prepared"]
            first_admit_tail = pair(rc, first_adm, "received_to_first_admission")
            if None in (waiting, head) or first_admit_tail is None:
                return iv, stall
            queue = head + first_admit_tail[0]
        else:
            waiting = 0
            until_first = pair(q, first_adm, "queued_to_first_admission")
            if until_first is None:
                return iv, stall
            queue = until_first[0]
        # Admissions before the one that finished the prefill: the work they did
        # (admission to the preemption) and the wait that followed (preemption to the
        # next admission), from the recorded boundaries.
        earlier = preempted = 0
        for before, after in zip(in_cycle, in_cycle[1:]):
            boundary = next(
                (
                    p
                    for p in preempted_rows
                    if before["sequence"] < p["sequence"] < after["sequence"]
                ),
                None,
            )
            if boundary is None:
                unavailable.setdefault(
                    "stall",
                    "an earlier admission has no recorded preemption before the next "
                    "one; its work and its wait cannot be told apart",
                )
                return iv, stall
            work = pair(sched(before), sched(boundary), "earlier_admission")
            wait = pair(sched(boundary), sched(after), "preempted")
            if work is None or wait is None:
                return iv, stall
            earlier += work[0]
            preempted += wait[0]
        host = iv["prefill_steps_host_span_sum"]
        window = iv["prefill_window"]
        after_step = iv["after_step_to_first_token_output"]
        if None in (host, window, after_step):
            return iv, stall
        between = window - host
        stall.update(
            pre_engine_ns=pre,
            queue_ns=queue,
            waiting_for_kv_ns=waiting,
            earlier_admissions_ns=earlier,
            preempted_ns=preempted,
            step_host_span_ns=host,
            between_steps_ns=between,
            after_step_to_first_token_ns=after_step,
        )
        other = total[0] - (
            pre + queue + waiting + earlier + preempted + host + between + after_step
        )
        stall["other_ns"] = other
        stall["complete"] = other >= 0
        if other < 0:
            self._problem(
                scheduler,
                "stall_negative_remainder",
                f"the stall buckets exceed the measured total by {-other} ns "
                "(overlapping or mismatched records); nothing was clipped",
            )
        return iv, stall

    # ------------------------------------------------------------ decode window
    def _decode_window(self, scheduler, frontend, finished, unavailable):
        first, last = frontend["_first"], frontend["_finished"]
        window = {
            "first_token_output_ns": frontend["first_token_output_ns"],
            "output_finished_ns": frontend["output_finished_ns"],
            "tokens_after_first_output": None,
            "mean_time_per_token_after_first_ns": None,
            "node_cadence": None,
            "quality": [],
        }
        if frontend["source"] is not None:
            reasons = self._quality(self.sources[frontend["source"]])
            if reasons:
                window["quality"].append(
                    {
                        "source": frontend["source"],
                        "role": "frontend",
                        "reasons": reasons,
                    }
                )
        n1, total = frontend["first_token_output_tokens"], frontend["output_tokens"]
        done = frontend["output_finished_status"] == "done"
        if n1 is not None and total is not None and done and total >= n1:
            window["tokens_after_first_output"] = total - n1
            span = self.clocks.delta(first, last)
            if span is not None and total > n1:
                window["mean_time_per_token_after_first_ns"] = span[0] / (total - n1)
        else:
            unavailable["mean_time_per_token_after_first_ns"] = (
                "needs a normal finish with the output token count at the first output "
                "and at the end; no inter-output mean is derived (a speculative "
                "output carries several tokens and outputs are not counted)"
            )
        if first is not None and last is not None:
            ranks = {}
            for source in self._scoped_workers(scheduler):
                inside = []
                for row in self.windows.get(source.index, []):
                    begin = self.clocks.delta(
                        first, _At(source, row["window_begin_ns"])
                    )
                    end = self.clocks.delta(_At(source, row["window_end_ns"]), last)
                    if begin and end and begin[0] >= 0 and end[0] >= 0:
                        inside.append(row)
                if inside:
                    steps = sum(r["steps"] for r in inside)
                    busy = sum(r["busy_ns"] for r in inside)
                    length = sum(
                        r["window_end_ns"] - r["window_begin_ns"] for r in inside
                    )
                    ranks[_key(_identity(source) or ("source", source.index))] = {
                        "windows": len(inside),
                        "steps": steps,
                        "mean_step_ns": busy / steps if steps else None,
                        "max_step_ns": max(r["max_ns"] for r in inside),
                        "busy_fraction": busy / length if length else None,
                        "empty_steps": sum(r["empty_steps"] for r in inside),
                        "quality": self._quality(source),
                    }
            window["node_cadence"] = ranks or None
            if not ranks:
                unavailable["node_cadence"] = (
                    "no worker rollup window lies entirely inside the decode window"
                )
        return window

    # -------------------------------------------------------------- cross node
    def _cross_node(
        self, scheduler, request, span, queued, prepared, received, unavailable
    ):
        matching = [
            join
            for join in self.joins
            if join["consumer_source"] == scheduler.index
            and join["consumer_request"] == request
            and join.get("consumer_handoff_sequence") is not None
            and span.lo <= join["consumer_handoff_sequence"]
            and (span.hi is None or join["consumer_handoff_sequence"] < span.hi)
        ]
        if not matching:
            return None
        if (
            len(matching) > 1
            or not matching[0]["joined"]
            or matching[0].get("identity_reused")
        ):
            unavailable["cross_node"] = (
                "the producer join is missing, ambiguous or its identity is reused"
            )
            return None
        join = matching[0]
        producer = self.sources[join["producer_sources"][0]]
        handoff = _At(producer, join["producer_handoff_time_ns"])
        result = {
            "join_key": join["join_key"],
            "producer_request": join["producer_request"],
            "transfer_id": join.get("transfer_id"),
            "consumer_queued_minus_producer_handoff_ns": None,
            "ordering": None,
            "producer_pending_ns": None,
            "uncertainty_ns": None,
            "method": None,
            "quality": [],
        }
        reasons = self._quality(producer)
        if reasons:
            result["quality"].append(
                {
                    "source": producer.index,
                    "role": "producer scheduler",
                    "reasons": reasons,
                }
            )
        if queued is None:
            return result
        diff = self.clocks.delta(handoff, _At(scheduler, queued["time_ns"]))
        if diff is None:
            unavailable["cross_node"] = (
                "the producer and consumer clocks are not calibrated and not one clock "
                "domain"
            )
            return result
        delta, unc, method = diff
        result["consumer_queued_minus_producer_handoff_ns"] = delta
        result["uncertainty_ns"] = unc
        result["method"] = method
        result["ordering"] = (
            "indeterminate"
            if abs(delta) <= unc
            else "consumer_before_handoff"
            if delta < 0
            else "after_handoff"
        )
        if prepared is not None and received is not None:
            lp = self.clocks.delta(_At(scheduler, prepared["time_ns"]), handoff)
            wait = received["time_ns"] - prepared["time_ns"]
            if lp is not None:
                result["producer_pending_ns"] = min(max(lp[0], 0), wait)
        return result


def _item_index(item):
    """The prompt index an item ID ends with (``<request ID>-<index>``), or None."""
    tail = item.rsplit("-", 1)[-1] if isinstance(item, str) else ""
    return int(tail) if tail.isdigit() else None


def _key(identity):
    return ":".join(str(part) for part in identity)


def _admission(row):
    return {
        "time_ns": row["time_ns"],
        "step": row["step"],
        "resumed": row["status"] == ADMITTED_RESUMED,
        "computed_tokens": row["computed_tokens"],
        "scheduled_tokens": row["scheduled_tokens"],
        "prefill_tokens": row["prefill_tokens"],
    }


def _finish_state(row):
    if row is None:
        return None
    status = row["status"]
    if status == FINISH_ABORTED:
        return "aborted"
    if status == FINISH_ERROR:
        return "error"
    return "done"


def _state(queued, cycle_start, first_done, finished):
    # An explicit terminal state is kept even when the request never reached a step.
    if finished is not None and finished["status"] == FINISH_ERROR:
        return "failed"
    if finished is not None and finished["status"] == FINISH_ABORTED:
        return "aborted"
    if queued is None or cycle_start is None or first_done is None or finished is None:
        return "incomplete"
    return "complete"


def _occurrences(rows):
    """Split one request ID's rows into occurrences: each QUEUED after a FINISHED or
    another QUEUED starts a new one."""
    result, current = [], []
    for row in rows:
        subject = row["subject"]
        if subject == RequestPhase.QUEUED and any(
            r["subject"] in (RequestPhase.QUEUED, RequestPhase.FINISHED)
            for r in current
        ):
            result.append(current)
            current = []
        current.append(row)
    if current:
        result.append(current)
    return result


def build(sources, joins, group, alias, problems, not_available):
    """The summary's ``timelines`` and ``kv_writes`` documents."""
    builder = Timelines(sources, joins, group, alias, problems, not_available)
    document = builder.build()
    for timeline in document["timelines"]:
        for key in ("_arrival", "_first", "_finished"):
            timeline["api"].pop(key, None)
    return document


__all__ = ["build", "Clocks", "Projection", "MARK_EMPTY"]
