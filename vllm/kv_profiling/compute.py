# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The request timeline hooks shared by the NIXL and Mooncake adapters.

Request phases (``REQUEST_PHASE``) are written at control points that already run
once per request or per prefill chunk; steps (``STEP_SPAN``) are written only for the
steps the scheduler marks as carrying a phase transition, plus one rollup per time
window. Nothing is recorded per token, and the scheduler never scans the decode
batch: a step touches its new requests, the requests still in prefill and the
preempted ones. Clock readings are ``time.monotonic_ns()`` taken by the hook itself,
the same clock the recorder stamps with; a row's own timestamp is its publication
time and is never a step bound.

The mixin runs inside ``_Profiler`` subclasses (``_record``, ``_error``, ``rank`` and
``_transport`` come from them). A hook error disables the profiler as everywhere else.
"""

import time
import uuid

from .schema import (
    FINISH_ABORTED,
    FINISH_DONE,
    FINISH_ERROR,
    MARK_EMPTY,
    MARK_PHASE,
    EventKind,
    RequestPhase,
    StepSubject,
)

_REQUEST_PHASE = int(EventKind.REQUEST_PHASE)
_STEP_SPAN = int(EventKind.STEP_SPAN)
_QUEUED = int(RequestPhase.QUEUED)
_LOAD_PREPARED = int(RequestPhase.LOAD_PREPARED)
_ADMITTED = int(RequestPhase.ADMITTED)
_PREFILL_CHUNK = int(RequestPhase.PREFILL_CHUNK)
_PREFILL_DONE = int(RequestPhase.PREFILL_DONE)
_PREEMPTED = int(RequestPhase.PREEMPTED)
_FINISHED = int(RequestPhase.FINISHED)
_FIRST_TOKEN_OUTPUT = int(RequestPhase.FIRST_TOKEN_OUTPUT)
_OUTPUT_FINISHED = int(RequestPhase.OUTPUT_FINISHED)
_STEP = int(StepSubject.STEP)
_WORKER_WINDOW = int(StepSubject.WORKER_WINDOW)
_SCHEDULER_WINDOW = int(StepSubject.SCHEDULER_WINDOW)
_FINISH_DONE = int(FINISH_DONE)
_FINISH_ABORTED = int(FINISH_ABORTED)
_FINISH_ERROR = int(FINISH_ERROR)
_MARK_PHASE = int(MARK_PHASE)
_MARK_EMPTY = int(MARK_EMPTY)

DEFAULT_WINDOW_S = 1.0
DEFAULT_TRACKED_CAP = 65536
_monotonic_ns = time.monotonic_ns


def _output_tokens(req_state):
    detokenizer = req_state.detokenizer
    return None if detokenizer is None else detokenizer.num_output_tokens()


def scope_of(process_id: str) -> int:
    """The scheduler generation (63 bits of its recorder's process ID) a step marker
    carries, so workers join only the scheduler that scheduled their step."""
    return uuid.UUID(process_id).int & ((1 << 63) - 1)


class _Tracked:
    """A request the scheduler currently follows."""

    __slots__ = ("admitted", "load_seen", "target", "done", "local")

    def __init__(self):
        self.admitted = False
        self.load_seen = False
        self.target = 0  # effective prefill length of the current admission
        self.done = 0  # tokens of it computed, the scheduler's view
        self.local = None  # tokens the local cache held when the connector was asked


class TimelineMixin:
    """Request phase and step hooks of one scheduler, worker or frontend process."""

    def _timeline_init(self, window_s=DEFAULT_WINDOW_S, cap=DEFAULT_TRACKED_CAP):
        self._tl_window_ns = int(window_s * 1e9)
        self._tl_cap = cap
        self.tracked_overflow = 0
        # Scheduler.
        self._tl: dict[str, _Tracked] = {}
        self._tl_prefilling: dict[str, _Tracked] = {}
        self._tl_step = 0
        self._tl_scope = None
        self._tl_sched_window = None
        # Worker.
        self._tl_begin = 0
        self._tl_marker = None
        self._tl_worker_window = None
        self.steps_without_marker = 0

    # ---------------------------------------------------------------- phases
    def _phase(
        self,
        phase,
        request_id,
        status=0,
        step=None,
        computed=None,
        scheduled=None,
        prefill=None,
        output=None,
        external=None,
        prompt=None,
    ):
        self._record(
            _REQUEST_PHASE, self.rank, request_id, None, None, None,
            step, computed, scheduled, prefill, output, external, prompt,
            self._transport, status, phase,
        )  # fmt: skip

    # ------------------------------------------------------------- scheduler
    def request_queued(self, request):
        """The engine core accepted the request (``on_new_request``)."""
        try:
            request_id = request.request_id
            if request_id not in self._tl:
                if len(self._tl) >= self._tl_cap:
                    self.tracked_overflow += 1
                else:
                    self._tl[request_id] = _Tracked()
            self._phase(_QUEUED, request_id, prompt=request.num_prompt_tokens)
        except Exception as exc:
            self._error(exc)

    def load_expected(self, request_id, local_tokens):
        """The connector was asked how many tokens it can load beyond the
        ``local_tokens`` the local cache holds (``get_num_new_matched_tokens``): the
        base the external count is incremental to. Records nothing."""
        try:
            state = self._tl.get(request_id)
            if state is not None:
                state.local = local_tokens
        except Exception as exc:
            self._error(exc)

    def load_prepared(self, request_id, external_tokens):
        """The connector was told blocks are allocated for an external load of
        ``external_tokens`` (``update_state_after_alloc``), once per admission cycle.

        The hook is what it says: it precedes the scheduler's park on the immediate
        path and follows it when the load is deferred."""
        try:
            if external_tokens <= 0:
                return
            state = self._tl.get(request_id)
            local = None
            if state is not None:
                if state.load_seen:
                    return
                state.load_seen = True
                local = state.local
            self._phase(
                _LOAD_PREPARED, request_id, external=external_tokens, computed=local
            )
        except Exception as exc:
            self._error(exc)

    def schedule_step(self, scheduler_output, meta):
        """One ``build_connector_meta``: record the step's phase transitions and write
        the marker into ``meta`` so workers record the same step."""
        try:
            now = _monotonic_ns()
            if self._tl_scope is None:
                self._tl_scope = scope_of(self.recorder.metadata["process_id"])
            self._tl_step += 1
            step = self._tl_step
            marked = False
            counts = scheduler_output.num_scheduled_tokens
            tl = self._tl
            preempted = scheduler_output.preempted_req_ids
            if preempted:
                for request_id in preempted:
                    state = tl.get(request_id)
                    if state is not None:
                        state.load_seen = False
                        self._tl_prefilling.pop(request_id, None)
                    self._phase(_PREEMPTED, request_id, step=step)
                    marked = True
            prefilling = self._tl_prefilling
            if prefilling:
                finished = []
                for request_id, state in prefilling.items():
                    n = counts.get(request_id)
                    if not n:
                        continue
                    state.done += n
                    if state.done >= state.target:
                        finished.append(request_id)
                        self._phase(
                            _PREFILL_DONE,
                            request_id,
                            step=step,
                            computed=state.done - n,
                            scheduled=n,
                        )
                    else:
                        self._phase(
                            _PREFILL_CHUNK,
                            request_id,
                            step=step,
                            computed=state.done - n,
                            scheduled=n,
                        )
                    marked = True
                for request_id in finished:
                    del prefilling[request_id]
            for new in scheduler_output.scheduled_new_reqs:
                request_id = new.req_id
                state = tl.get(request_id)
                if state is None:
                    if len(tl) >= self._tl_cap:
                        self.tracked_overflow += 1
                        continue
                    state = tl[request_id] = _Tracked()
                n = counts.get(request_id, 0)
                computed = new.num_computed_tokens
                target = len(new.prefill_token_ids)
                prompt = new.prompt_token_ids
                self._phase(
                    _ADMITTED,
                    request_id,
                    status=1 if state.admitted else 0,
                    step=step,
                    computed=computed,
                    scheduled=n,
                    prefill=target,
                    prompt=None if prompt is None else len(prompt),
                )
                state.admitted = True
                state.target = target
                state.done = computed + n
                if state.done >= target:
                    self._phase(
                        _PREFILL_DONE,
                        request_id,
                        step=step,
                        computed=computed,
                        scheduled=n,
                    )
                else:
                    prefilling[request_id] = state
                marked = True
            tokens = scheduler_output.total_num_scheduled_tokens
            flags = (_MARK_PHASE if marked else 0) | (0 if tokens else _MARK_EMPTY)
            meta.profiler_step = step
            meta.profiler_scope = self._tl_scope
            meta.profiler_flags = flags
            self._sched_window(now, tokens, len(counts), marked)
        except Exception as exc:
            self._error(exc)

    def request_done(self, request):
        """The scheduler's ``request_finished`` (once per request)."""
        try:
            request_id = request.request_id
            self._tl.pop(request_id, None)
            self._tl_prefilling.pop(request_id, None)
            name = request.status.name
            status = (
                _FINISH_ABORTED
                if name == "FINISHED_ABORTED"
                else _FINISH_ERROR
                if name == "FINISHED_ERROR"
                else _FINISH_DONE
            )
            self._phase(
                _FINISHED,
                request_id,
                status=status,
                output=request.num_output_tokens,
            )
        except Exception as exc:
            self._error(exc)

    def _sched_window(self, now, tokens, requests, marked):
        if not self._tl_window_ns:
            return
        window = self._tl_sched_window
        if window is None:
            window = self._tl_sched_window = [now, now, 0, 0, 0, 0, 0]
        # begin, end, steps, tokens, max requests, empty steps, marked steps
        window[1] = now
        if tokens:
            window[2] += 1
            window[3] += tokens
            if requests > window[4]:
                window[4] = requests
        else:
            window[5] += 1
        if marked:
            window[6] += 1
        if now - window[0] >= self._tl_window_ns:
            self._flush_sched_window()

    def _flush_sched_window(self):
        window = self._tl_sched_window
        if window is None:
            return
        self._tl_sched_window = None
        self._record(
            _STEP_SPAN, self.rank, None, None, None, None,
            window[2], window[3], window[0], window[1], window[4], window[5],
            window[6], self._transport, 0, _SCHEDULER_WINDOW,
        )  # fmt: skip

    # ---------------------------------------------------------------- worker
    def step_begin(self, meta):
        """The connector received the step's metadata (before the forward)."""
        try:
            self._tl_begin = _monotonic_ns()
            step = getattr(meta, "profiler_step", 0)
            if step:
                self._tl_marker = (step, meta.profiler_scope, meta.profiler_flags)
            else:
                self._tl_marker = None
        except Exception as exc:
            self._error(exc)

    def step_end(self):
        """The runner's ``finish_forward``: this step no longer reads the KV."""
        try:
            begin = self._tl_begin
            if not begin:
                return  # no open step (a second call, or no bind first)
            self._tl_begin = 0
            marker = self._tl_marker
            if marker is None:
                self.steps_without_marker += 1
                return
            end = _monotonic_ns()
            step, scope, flags = marker
            if self._tl_window_ns:
                span = end - begin
                window = self._tl_worker_window
                if window is None:
                    window = self._tl_worker_window = [begin, end, 0, 0, 0, 0, 0]
                # begin, end, steps, busy, max, empty steps, marked steps
                window[1] = end
                if flags & _MARK_EMPTY:
                    window[5] += 1
                else:
                    window[2] += 1
                    window[3] += span
                    if span > window[4]:
                        window[4] = span
                if flags & _MARK_PHASE:
                    window[6] += 1
                if end - window[0] >= self._tl_window_ns:
                    self._flush_worker_window()
            if flags & _MARK_PHASE:
                self._record(
                    _STEP_SPAN, self.rank, None, None, None, None,
                    step, scope, begin, end, None, None, None,
                    self._transport, 0, _STEP,
                )  # fmt: skip
        except Exception as exc:
            self._error(exc)

    def _flush_worker_window(self):
        window = self._tl_worker_window
        if window is None:
            return
        self._tl_worker_window = None
        self._record(
            _STEP_SPAN, self.rank, None, None, None, None,
            window[2], window[6], window[0], window[1], window[3], window[4],
            window[5], self._transport, 0, _WORKER_WINDOW,
        )  # fmt: skip

    # ------------------------------------------------------------- frontend
    def first_token_output(self, req_state, new_token_ids):
        """The first non-empty output of this request state reached the frontend."""
        try:
            self._phase(
                _FIRST_TOKEN_OUTPUT, req_state.request_id, output=len(new_token_ids)
            )
        except Exception as exc:
            self._error(exc)

    def output_finished(self, req_state, finish_reason):
        """The frontend processed the request's final output (not the delivery to the
        client): status 0 for a normal finish reason, 1 when the engine delivered an
        abort, 2 for an error."""
        try:
            name = finish_reason.name
            self._phase(
                _OUTPUT_FINISHED,
                req_state.request_id,
                status=(
                    _FINISH_ABORTED
                    if name == "ABORT"
                    else _FINISH_ERROR
                    if name == "ERROR"
                    else _FINISH_DONE
                ),
                output=_output_tokens(req_state),
            )
        except Exception as exc:
            self._error(exc)

    def output_aborted(self, req_state):
        """The frontend aborted the request locally, outside the final-output path."""
        try:
            self._phase(
                _OUTPUT_FINISHED,
                req_state.request_id,
                status=_FINISH_ABORTED,
                output=_output_tokens(req_state),
            )
        except Exception as exc:
            self._error(exc)

    def output_failed(self, req_state):
        """The engine rejected the request on arrival: nothing ran."""
        try:
            self._phase(_OUTPUT_FINISHED, req_state.request_id, status=_FINISH_ERROR)
        except Exception as exc:
            self._error(exc)

    # ---------------------------------------------------------------- close
    def _timeline_close(self):
        """Flush the open windows; runs while hooks may still record."""
        for flush in (self._flush_worker_window, self._flush_sched_window):
            try:
                flush()
            except Exception as exc:
                self._error(exc)

    def _timeline_summary(self) -> dict:
        return {
            "timeline_tracked_overflow": self.tracked_overflow,
            "timeline_steps_without_marker": self.steps_without_marker,
        }


def parallel_context(parallel_config) -> dict:
    """What a scheduler's capture says about its engine's workers: how many exist."""
    return {
        "tp": parallel_config.tensor_parallel_size,
        "pp": parallel_config.pipeline_parallel_size,
        "pcp": parallel_config.prefill_context_parallel_size,
        "workers": parallel_config.world_size,
    }


def worker_identity() -> dict:
    """A worker's tensor, pipeline and prefill-context rank in its engine."""
    from vllm.distributed.parallel_state import (
        get_pcp_group,
        get_pp_group,
        get_tp_group,
    )

    return {
        "tp_rank": get_tp_group().rank_in_group,
        "pp_rank": get_pp_group().rank_in_group,
        "pcp_rank": get_pcp_group().rank_in_group,
    }


def timeline_context(vllm_config, role: str) -> dict:
    """The identity fields the summary uses to check rank completeness; a field that
    cannot be read here is simply absent (the summary then reports it unavailable)."""
    found = {}
    try:
        if role == "scheduler":
            found["parallel"] = parallel_context(vllm_config.parallel_config)
        elif role == "worker":
            found.update(worker_identity())
    except Exception:
        pass
    return found
