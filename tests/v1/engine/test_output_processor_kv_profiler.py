# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The frontend KV profiler's hook sites in the output processor.

A recording stand-in replaces the profiler: what is under test is which request state
reaches which hook, once, and which paths (streaming input, local abort, engine
rejection) reach none or another.
"""

import pytest

from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector


class Recorder:
    def __init__(self):
        self.calls = []

    def first_token_output(self, req_state, new_token_ids):
        self.calls.append(("first", req_state.request_id, len(new_token_ids)))

    def output_finished(self, req_state, finish_reason):
        self.calls.append(("finished", req_state.request_id, finish_reason.name))

    def output_aborted(self, req_state):
        self.calls.append(("aborted", req_state.request_id))

    def output_failed(self, req_state):
        self.calls.append(("failed", req_state.request_id))


def add(processor, request_id, *, resumable=False, prompt=(1, 2, 3)):
    request = EngineCoreRequest(
        request_id=request_id,
        external_req_id=f"{request_id}-ext",
        prompt_token_ids=list(prompt),
        mm_features=None,
        sampling_params=SamplingParams(
            detokenize=False, output_kind=RequestOutputKind.DELTA
        ),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=resumable,
    )
    queue = RequestOutputCollector(RequestOutputKind.DELTA, request_id=request_id)
    processor.add_request(request, None, queue=queue)
    return queue


@pytest.fixture
def processor():
    processor = OutputProcessor(None, log_stats=False)
    processor.kv_profiler = Recorder()
    return processor


def output(request_id, tokens, finish_reason=None, **kw):
    return EngineCoreOutput(
        request_id=request_id, new_token_ids=tokens, finish_reason=finish_reason, **kw
    )


def test_nothing_is_called_without_a_profiler():
    processor = OutputProcessor(None, log_stats=False)
    assert processor.kv_profiler is None
    add(processor, "r1")
    processor.process_outputs([output("r1", [5]), output("r1", [6], FinishReason.STOP)])
    assert not processor.has_request("r1")


def test_the_first_non_empty_output_is_recorded_once_and_the_final_output_once(
    processor,
):
    add(processor, "r1")
    processor.process_outputs([output("r1", [])])  # an empty first output is no token
    processor.process_outputs(
        [output("r1", [5, 6, 7])]
    )  # speculative: three tokens at once
    processor.process_outputs([output("r1", [8])])
    processor.process_outputs([output("r1", [9], FinishReason.LENGTH)])
    assert processor.kv_profiler.calls == [
        ("first", "r1", 3),
        ("finished", "r1", "LENGTH"),
    ]


def test_each_child_of_a_parallel_request_records_its_own_first_token(processor):
    for child in ("c0", "c1"):
        add(processor, child)
    processor.process_outputs([output("c0", [1]), output("c1", [2]), output("c0", [3])])
    assert sorted(processor.kv_profiler.calls) == [
        ("first", "c0", 1),
        ("first", "c1", 1),
    ]


def test_streaming_input_resetting_the_prefill_flag_records_no_second_first_token(
    processor,
):
    add(processor, "s1", resumable=True)
    processor.add_request(
        EngineCoreRequest(
            request_id="s1",
            external_req_id="s1-ext",
            prompt_token_ids=[7, 8],
            mm_features=None,
            sampling_params=SamplingParams(
                detokenize=False, output_kind=RequestOutputKind.DELTA
            ),
            pooling_params=None,
            arrival_time=0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            resumable=True,
        ),
        None,
    )
    processor.process_outputs(
        [output("s1", [4], FinishReason.STOP)]
    )  # a chunk boundary
    state = processor.request_states["s1"]
    assert state.is_prefilling is True  # the next chunk reset it
    processor.process_outputs([output("s1", [5])])
    # One first token (the session's first chunk); no terminal row at a chunk boundary.
    assert processor.kv_profiler.calls == [("first", "s1", 1)]


def test_a_local_abort_is_its_own_terminal_row_and_never_a_final_output(processor):
    add(processor, "r1")
    add(processor, "r2")
    processor.process_outputs([output("r1", [5])])
    processor.abort_requests(["r1"], internal=True)
    processor.abort_requests(["r1"], internal=True)  # already gone: nothing again
    assert processor.kv_profiler.calls == [("first", "r1", 1), ("aborted", "r1")]
    processor.process_outputs(
        [output("r1", [6], FinishReason.STOP)]
    )  # late engine output
    assert len(processor.kv_profiler.calls) == 2


def test_an_engine_rejection_on_arrival_is_a_failed_terminal_row(processor):
    from vllm.v1.engine import RequestError

    add(processor, "r1")
    processor.process_outputs(
        [
            output(
                "r1",
                [],
                FinishReason.ERROR,
                error=RequestError(
                    code="engine_paused", message="paused", retryable=True
                ),
            )
        ]
    )
    assert processor.kv_profiler.calls == [("failed", "r1")]


def test_an_error_finish_is_reported_with_its_reason(processor):
    add(processor, "r1")
    processor.process_outputs([output("r1", [], FinishReason.ERROR)])
    assert processor.kv_profiler.calls == [("finished", "r1", "ERROR")]
