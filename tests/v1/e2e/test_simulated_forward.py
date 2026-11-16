# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from statistics import pvariance

import pytest

from tests.conftest import VllmRunner
from tests.models.registry import HF_EXAMPLE_MODELS
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import HAS_TRITON
from vllm.v1.metrics.reader import Counter

pytestmark = [
    pytest.mark.cpu_model,
    pytest.mark.skipif(not HAS_TRITON, reason="Simulated forward requires Triton."),
    pytest.mark.skipif(
        not (current_platform.is_cpu() or current_platform.is_cuda_alike()),
        reason="Simulated forward needs the CPU or a CUDA-alike platform.",
    ),
]

SIMULATED_FORWARD_MODELS = [
    pytest.param("Qwen/Qwen3-235B-A22B", 1, id="qwen3-235b-a22b"),
    pytest.param("allenai/Olmo-3-7B-Think", 2, id="olmo3-hybrid-attention"),
    pytest.param("openai/gpt-oss-120b", 2, id="gpt-oss-120b"),
]

# Qwen/Qwen3-0.6B at revision c1899de289a04d12100db370d81485cdf75e47ca.
QWEN3_0_6B_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 40960,
    "max_window_layers": 28,
    "model_type": "qwen3",
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


@pytest.fixture(scope="module")
def qwen3_0_6b_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("qwen3-0.6b")
    (directory / "config.json").write_text(json.dumps(QWEN3_0_6B_CONFIG))
    return str(directory)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (
            {
                "kv_transfer_config": {
                    "kv_connector": "ExampleConnector",
                    "kv_role": "kv_both",
                }
            },
            "not ExampleConnector",
        ),
        (
            {
                "kv_transfer_config": {
                    "kv_connector": "OffloadingConnector",
                    "kv_role": "kv_both",
                }
            },
            "not OffloadingConnector",
        ),
        ({"kv_offloading_size": 1.0}, "does not support KV offloading"),
        (
            {"simulation_config": {"clock": "virtual"}, "tensor_parallel_size": 2},
            "supports one rank",
        ),
        ({"enable_lora": True}, "LoRA"),
    ],
)
def test_simulated_config_rejections(qwen3_0_6b_dir, kwargs, error) -> None:
    from vllm.engine.arg_utils import EngineArgs

    with pytest.raises(ValueError, match=error):
        EngineArgs(
            model=qwen3_0_6b_dir, max_model_len=1024, simulate_forward=True, **kwargs
        ).create_engine_config()


def test_simulate_forward_quantized_checkpoint(
    tmp_path, monkeypatch: pytest.MonkeyPatch, vllm_runner: type[VllmRunner]
) -> None:
    """A quantized checkpoint is simulated as deployed: its layers take the
    checkpoint's quantization, and its KV cache the dtype it declares."""
    from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

    # apply_model() ships a function to the workers.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    config = get_hf_file_to_dict("config.json", "Qwen/Qwen3-0.6B")
    config["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "FP8",
        "kv_cache_quant_algo": "FP8",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    with vllm_runner(
        str(tmp_path),
        tokenizer_name="Qwen/Qwen3-0.6B",
        simulate_forward=True,
        kv_cache_memory_bytes=1024**3,
        max_model_len=1024,
    ) as vllm_model:
        vllm_config = vllm_model.llm.llm_engine.vllm_config
        assert vllm_config.cache_config.cache_dtype == "fp8_e4m3"
        weight_dtypes = vllm_model.apply_model(
            lambda model: str(model.model.layers[0].self_attn.qkv_proj.weight.dtype)
        )
        outputs = vllm_model.generate_greedy([[1000, 1001, 1002]], max_tokens=4)
    assert weight_dtypes == ["torch.float8_e4m3fn"]
    assert len(outputs[0][0]) == 3 + 4


def test_simulate_forward_online_quantization(
    monkeypatch: pytest.MonkeyPatch, vllm_runner: type[VllmRunner]
) -> None:
    """Online quantization quantizes the meta weights as they load."""
    # apply_model() ships a function to the workers.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        "Qwen/Qwen3-0.6B",
        quantization="fp8",
        simulate_forward=True,
        kv_cache_memory_bytes=1024**3,
        max_model_len=1024,
    ) as vllm_model:
        weight_dtypes = vllm_model.apply_model(
            lambda model: str(model.model.layers[0].self_attn.qkv_proj.weight.dtype)
        )
        outputs = vllm_model.generate_greedy([[1000, 1001, 1002]], max_tokens=4)
    assert weight_dtypes == ["torch.float8_e4m3fn"]
    assert len(outputs[0][0]) == 3 + 4


def _check_model_available(model: str) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    except ValueError:
        return
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")


def _get_kv_cache_group_count(vllm_model: VllmRunner) -> int | None:
    engine_core_client = vllm_model.llm.llm_engine.engine_core
    engine_core = getattr(engine_core_client, "engine_core", None)
    if engine_core is None:
        return None

    kv_cache_config = getattr(engine_core.scheduler, "kv_cache_config", None)
    if kv_cache_config is None:
        return None
    return len(kv_cache_config.kv_cache_groups)


def _get_counter_value(vllm_model: VllmRunner, metric_name: str) -> int:
    vllm_model.llm.llm_engine.do_log_stats()
    value = 0
    found = False
    for metric in vllm_model.llm.get_metrics():
        if isinstance(metric, Counter) and metric.name == metric_name:
            found = True
            value += metric.value
    assert found, f"Metric not found: {metric_name}"
    return value


@pytest.mark.parametrize(("model", "min_kv_cache_groups"), SIMULATED_FORWARD_MODELS)
def test_simulate_forward_model_matrix(
    vllm_runner: type[VllmRunner],
    model: str,
    min_kv_cache_groups: int,
) -> None:
    _check_model_available(model)

    # No max_tokens: generation must stop after the trace instead of padding
    # to max_model_len.
    sampling_params = SamplingParams(
        temperature=0.0,
        detokenize=False,
        trace_decode_token_ids=[501, 502],
    )
    prompt = [1000, 1001, 1002, 1003]

    with vllm_runner(
        model,
        simulate_forward=True,
        disable_hybrid_kv_cache_manager=False,
        kv_cache_memory_bytes=16 * 1024**3,
        max_model_len=1024,
    ) as vllm_model:
        outputs = vllm_model.generate([prompt], sampling_params=sampling_params)
        assert outputs[0][0][0] == prompt + [501, 502]

        group_count = _get_kv_cache_group_count(vllm_model)
        if group_count is not None:
            assert group_count >= min_kv_cache_groups


def test_simulate_forward_prefix_cache_hybrid_retention_zero(
    monkeypatch: pytest.MonkeyPatch,
    vllm_runner: type[VllmRunner],
) -> None:
    model = "openai/gpt-oss-120b"
    _check_model_available(model)

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "0")

    sampling_params = SamplingParams(
        temperature=0.0,
        detokenize=False,
        trace_decode_token_ids=[501, 502],
    )
    prompt = [1000 + (i % 127) for i in range(512)]

    with vllm_runner(
        model,
        simulate_forward=True,
        disable_hybrid_kv_cache_manager=False,
        enable_prefix_caching=True,
        disable_log_stats=False,
        kv_cache_memory_bytes=16 * 1024**3,
        max_model_len=1024,
    ) as vllm_model:
        group_count = _get_kv_cache_group_count(vllm_model)
        if group_count is not None:
            assert group_count >= 2

        outputs = vllm_model.generate([prompt], sampling_params=sampling_params)
        assert outputs[0][0][0][-2:] == [501, 502]
        hits_after_first = _get_counter_value(vllm_model, "vllm:prefix_cache_hits")

        second_sampling_params = SamplingParams(
            temperature=0.0,
            detokenize=False,
            trace_decode_token_ids=[503, 504],
        )
        outputs = vllm_model.generate([prompt], sampling_params=second_sampling_params)
        assert outputs[0][0][0][-2:] == [503, 504]
        hits_after_second = _get_counter_value(vllm_model, "vllm:prefix_cache_hits")

        assert hits_after_second > hits_after_first


def test_simulate_offline_virtual_clock() -> None:
    """Under the virtual clock, request timestamps are exactly the constant
    timing model's step times, idle gaps are skipped, and arrivals are
    honored."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.simulated.offline import TimedRequest, simulate

    step_s = 1e-3
    args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1024,
        kv_cache_memory_bytes=1024**3,
        simulation_config={
            "simulate_forward": True,
            "clock": "virtual",
            "timing": "linear",
            "timing_args": {"fixed_us": step_s * 1e6},
        },
    )
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    requests = [TimedRequest(list(range(16)), params, 0.0) for _ in range(4)]
    requests.append(TimedRequest(list(range(16)), params, 10.0))

    result = simulate(args, requests)

    for timing in result.requests:
        # One prefill step yields the first token, then one token per step.
        expected = [timing.arrival_time + k * step_s for k in range(1, 9)]
        assert timing.num_output_tokens == 8
        assert timing.token_times == pytest.approx(expected)
        assert timing.finish_time == pytest.approx(expected[-1])
    assert result.duration == pytest.approx(10.0 + 8 * step_s)
    assert result.summary()["mean_tpot_ms"] == pytest.approx(step_s * 1e3)

    # A frontend latency delays each request's admission, so it adds to TTFT
    # and end-to-end latency but not to TPOT.
    latency_s = 5e-3
    shifted = simulate(args, requests, frontend_latency_s=latency_s)
    for before, after in zip(result.requests, shifted.requests):
        assert after.arrival_time == before.arrival_time
        assert after.ttft == pytest.approx(before.ttft + latency_s)
        assert after.e2el == pytest.approx(before.e2el + latency_s)
        assert after.tpot == pytest.approx(before.tpot)


def test_simulate_offline_host_time_holds_engine_thread() -> None:
    """A step whose host time exceeds its device time paces the run by the
    host time: the next step cannot start before the engine thread is free."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.simulated.offline import TimedRequest, simulate

    args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1024,
        kv_cache_memory_bytes=1024**3,
        simulation_config={
            "simulate_forward": True,
            "clock": "virtual",
            "timing": "linear",
            "timing_args": {"fixed_us": 1000, "host_us": 3000},
        },
    )
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    result = simulate(args, [TimedRequest(list(range(16)), params)])
    # The 3 ms host time, not the 1 ms device time, paces the steps.
    assert result.requests[0].itls[:6] == pytest.approx([3e-3] * 6)


def test_simulate_offline_sessions() -> None:
    """A session's next turn arrives a think time after the previous one
    finished, extends its prompt and output, and reuses their KV cache."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.simulated.offline import TimedSession, Turn, simulate

    args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1024,
        kv_cache_memory_bytes=1024**3,
        simulation_config={
            "simulate_forward": True,
            "clock": "virtual",
            "timing": "linear",
            "timing_args": {"fixed_us": 1000, "prefill_us_per_token": 10},
        },
    )
    params = SamplingParams(max_tokens=12, ignore_eos=True)
    session = TimedSession(
        [Turn(list(range(56)), params), Turn(list(range(100, 116)), params, 0.5)],
        arrival_time=1.0,
    )
    first, second = simulate(args, [session]).requests

    assert first.ttft == pytest.approx(1000e-6 + 56 * 10e-6)
    assert second.arrival_time == pytest.approx(first.finish_time + 0.5)
    assert second.num_prompt_tokens == 56 + 12 + 16
    # The KV of the prompt and the first 8 output tokens (64 tokens, whole
    # blocks) is cached only if the output follows the prompt; 20 tokens are
    # prefilled.
    assert second.ttft == pytest.approx(1000e-6 + 20 * 10e-6)


class AcceptAllDrafts:
    """Timing provider that accepts every draft, overriding the rates."""

    def __init__(self, vllm_config, **kwargs):
        pass

    def estimate(self, step):
        from vllm.v1.worker.simulated.timing import StepCost

        return StepCost(
            device_s=1e-3,
            num_accepted=tuple(r.num_spec_tokens for r in step.requests),
        )


@pytest.mark.parametrize(
    ("rates", "timing", "chunks"),
    [
        ([1.0], "linear", 5),
        ([0.0], "linear", 16),
        ([0.0], f"{__name__}.AcceptAllDrafts", 5),
    ],
)
def test_simulate_offline_speculative_decoding(
    rates: list[float], timing: str, chunks: int
) -> None:
    """Accepted drafts follow `spec_acceptance_rates` unless the timing
    provider sets them: with every draft accepted a step yields 1 + 3 tokens,
    with none it yields one."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.simulated.offline import TimedRequest, simulate

    args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1024,
        kv_cache_memory_bytes=1024**3,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 3,
            "prompt_lookup_max": 3,
            "prompt_lookup_min": 2,
        },
        simulation_config={
            "simulate_forward": True,
            "clock": "virtual",
            "timing": timing,
            "timing_args": {"fixed_us": 1000} if timing == "linear" else {},
            "spec_acceptance_rates": rates,
        },
    )
    params = SamplingParams(max_tokens=16, ignore_eos=True)
    result = simulate(args, [TimedRequest(list(range(16)), params) for _ in range(4)])
    for request in result.requests:
        assert request.num_output_tokens == 16
        assert len(request.token_times) == chunks


@pytest.mark.parametrize(("tp", "pp"), [(2, 1), (1, 2)])
def test_simulate_forward_multi_rank(
    vllm_runner: type[VllmRunner], tp: int, pp: int
) -> None:
    """Every rank runs the bookkeeping; the output rank's trace tokens come
    back as on a single rank."""
    # CPU ranks need no devices; CUDA ranks need one each.
    if not current_platform.is_cpu() and (
        not current_platform.is_cuda_alike()
        or current_platform.device_count() < tp * pp
    ):
        pytest.skip(f"Needs the CPU platform or {tp * pp} CUDA devices.")
    prompt = list(range(1000, 1064))
    sampling_params = SamplingParams(trace_decode_token_ids=[501, 502, 503])
    with vllm_runner(
        "Qwen/Qwen3-0.6B",
        simulate_forward=True,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        kv_cache_memory_bytes=1024**3,
        max_model_len=1024,
    ) as vllm_model:
        outputs = vllm_model.generate([prompt] * 4, sampling_params=sampling_params)
    for output in outputs:
        assert output[0][0][-3:] == [501, 502, 503]


def test_simulate_forward_prefill_context_parallel(
    vllm_runner: type[VllmRunner],
) -> None:
    """PCP ranks run their shard of each prefill, and the request still
    samples once its whole prefill is done."""
    if not current_platform.is_cuda_alike() or current_platform.device_count() < 2:
        pytest.skip("Needs 2 CUDA devices.")
    sampling_params = SamplingParams(trace_decode_token_ids=[501, 502, 503])
    # PCP needs an MLA attention backend.
    with vllm_runner(
        "deepseek-ai/DeepSeek-V2-Lite",
        simulate_forward=True,
        prefill_context_parallel_size=2,
        hf_overrides={"num_hidden_layers": 1},
        trust_remote_code=True,
        kv_cache_memory_bytes=1024**3,
        max_model_len=1024,
    ) as vllm_model:
        outputs = vllm_model.generate(
            [list(range(1000, 1300))] * 4, sampling_params=sampling_params
        )
    for output in outputs:
        assert output[0][0][-3:] == [501, 502, 503]


@pytest.mark.skipif(not current_platform.is_cpu(), reason="Tests the CPU platform.")
def test_simulate_forward_cpu_hybrid_prefix_cache_multi_rank(
    vllm_runner: type[VllmRunner],
) -> None:
    """Workers raise the block size to fit a hybrid model's Mamba pages. The
    engine core reruns VllmConfig.__post_init__ after its handshake and must
    keep that size: Mamba prefill checkpoints land on its block boundaries,
    so a repeated prompt hits up to its last full block."""
    from vllm.inputs import TokensPrompt

    params = SamplingParams(max_tokens=1, detokenize=False)
    with vllm_runner(
        "Qwen/Qwen3.8-2.4T-A95B-FP8",
        hf_overrides={"num_hidden_layers": 4},
        simulate_forward=True,
        # Unset, as the platform resolves it (the runner sets one by default).
        block_size=None,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        # The whole prompt in one chunk, which no block boundary splits.
        max_num_batched_tokens=16384,
        kv_cache_memory_bytes=4 * 1024**3,
        max_model_len=16384,
    ) as vllm_model:
        block_size = vllm_model.llm.llm_engine.vllm_config.cache_config.block_size
        # Ends mid-block: the prefill must be split at the last block boundary.
        num_tokens = 4 * block_size + block_size // 2
        prompt = TokensPrompt(
            prompt_token_ids=[1000 + i % 5000 for i in range(num_tokens)]
        )
        hits = [
            vllm_model.llm.generate(prompt, params)[0].num_cached_tokens
            for _ in range(2)
        ]
    assert hits == [0, 4 * block_size]


@pytest.mark.skipif(not current_platform.is_cpu(), reason="Tests the CPU platform.")
def test_simulate_forward_cpu_decode_context_parallel(tmp_path, monkeypatch) -> None:
    """A dense MLA model at TP2 and DCP2 keeps vLLM's DCP bookkeeping under a
    simulated forward on CPU: each rank its DCP rank and its layers' DCP
    merge setup, the scheduler's blocks spanning twice the tokens of the
    local ones, as many of them (each rank caches half of every request's
    tokens, in a virtual cache),
    and the requests finish. At DCP1 the CPU MLA backend keeps its own
    attributes."""
    from tests.v1.worker.simulated.test_real_kv_transfer import DEEPSEEK_V3
    from vllm import LLM
    from vllm.distributed import cleanup_dist_env_and_memory

    def _dcp_rank_state(worker) -> dict:
        """A rank's DCP identity, its KV cache layout and its MLA layers' DCP
        setup (defined here: sent to the workers by value)."""
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
        from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_block_size

        runner = worker.model_runner
        layers = [m for m in runner.model.modules() if isinstance(m, MLAAttention)]
        (group,) = runner.kv_cache_config.kv_cache_groups
        return {
            "dcp": [runner.dcp_size, runner.dcp_rank],
            # Local blocks, and the token span the scheduler's block covers.
            "blocks": [
                runner.kv_cache_config.num_blocks,
                group.kv_cache_spec.block_size,
                resolve_dcp_kv_block_size(group.kv_cache_spec, runner.dcp_size),
            ],
            # Lists: a set in an RPC result never comes back.
            "layers": sorted(
                {
                    (
                        layer.impl.dcp_world_size,
                        layer.impl.dcp_rank,
                        layer.impl.cp_kv_cache_interleave_size,
                        layer.impl.can_return_lse_for_decode,
                        layer.impl.need_to_return_lse_for_decode,
                        layer.dcp_manager is not None,
                    )
                    for layer in layers
                }
            ),
        }

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    (tmp_path / "config.json").write_text(json.dumps(DEEPSEEK_V3))
    params = SamplingParams(max_tokens=8, ignore_eos=True, detokenize=False)
    states = {}
    for dcp in (1, 2):
        llm = LLM(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            simulate_forward=True,
            tensor_parallel_size=2,
            decode_context_parallel_size=dcp,
            kv_cache_memory_bytes=1 << 28,
            max_model_len=256,
        )
        try:
            outputs = llm.generate(
                [{"prompt_token_ids": list(range(101))}] * 2, params, use_tqdm=False
            )
            assert [len(o.outputs[0].token_ids) for o in outputs] == [8, 8]
            states[dcp] = llm.collective_rpc(_dcp_rank_state)
        finally:
            del llm
            cleanup_dist_env_and_memory()
    assert [s["dcp"] for s in states[1]] == [[1, 0], [1, 0]]
    assert [s["dcp"] for s in states[2]] == [[2, 0], [2, 1]]
    assert [s["layers"] for s in states[1]] == [[(1, 0, 1, False, False, False)]] * 2
    assert [s["layers"] for s in states[2]] == [
        [(2, 0, 1, True, True, True)],
        [(2, 1, 1, True, True, True)],
    ]
    num_blocks, block_size, _ = states[1][0]["blocks"]
    assert [s["blocks"] for s in states[1]] == [
        [num_blocks, block_size, block_size]
    ] * 2
    assert [s["blocks"] for s in states[2]] == [
        [num_blocks, block_size, 2 * block_size]
    ] * 2


@pytest.mark.skipif(not current_platform.is_cpu(), reason="Tests the CPU platform.")
@pytest.mark.parametrize(
    "model, kwargs, problem",
    [
        # GQA at a layout vLLM itself allows: DCP <= TP / KV heads.
        ("gpt-oss", {}, "a model other than dense MLA"),
        ("deepseek-v3.2", {}, "a model other than dense MLA"),
        (
            "deepseek-v3",
            {
                "kv_transfer_config": {
                    "kv_connector": "OffloadingConnector",
                    "kv_role": "kv_both",
                }
            },
            "the OffloadingConnector connector",
        ),
    ],
)
def test_simulated_cpu_decode_context_parallel_scope(
    tmp_path, model, kwargs, problem
) -> None:
    """Simulated DCP on CPU is a dense MLA model's, over TP alone, without a
    KV connector."""
    from tests.v1.worker.simulated.test_real_kv_transfer import (
        DEEPSEEK_V3,
        DEEPSEEK_V32,
        GPT_OSS,
    )
    from vllm.engine.arg_utils import EngineArgs

    config = {
        "gpt-oss": {**GPT_OSS, "num_key_value_heads": 1},
        "deepseek-v3": DEEPSEEK_V3,
        "deepseek-v3.2": DEEPSEEK_V32,
    }[model]
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = EngineArgs(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        max_model_len=256,
        tensor_parallel_size=2,
        decode_context_parallel_size=2,
        simulation_config={"simulate_forward": True},
        **kwargs,
    )
    with pytest.raises(ValueError, match=problem):
        args.create_engine_config()


def _sparse_dcp_model(tmp_path, target):
    from tests.v1.worker.simulated.test_aisimulate_timing import bundled_config

    raw = bundled_config(target)
    raw.pop("quantization_config", None)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    (tmp_path / "DERIVED.md").write_text(
        f"{target}: published geometry, quantization_config removed for meta weights."
    )
    return raw


@pytest.mark.skipif(not current_platform.is_cpu(), reason="Tests the CPU platform.")
@pytest.mark.parametrize(
    "target, gpu, tp, weight_mode",
    [
        ("deepseek-ai/DeepSeek-V3.2", "h200", 8, "fp8_block"),
        ("nvidia/GLM-5-NVFP4", "b200", 4, "nvfp4"),
    ],
    ids=["deepseek-v32", "glm-5"],
)
def test_simulate_forward_cpu_sparse_mla_dcp(
    tmp_path, monkeypatch, target, gpu, tp, weight_mode
) -> None:
    """Published DSA geometry keeps both KV roles sharded and times live steps."""
    pytest.importorskip("aisimulate_core")
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel
    from vllm import LLM
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.inputs import TokensPrompt

    def rank_state(worker) -> dict:
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
        from vllm.v1.attention.backends.mla.flashmla_sparse import (
            FlashMLASparseMetadataBuilder,
        )
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV32IndexerMetadataBuilder,
        )
        from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_block_size

        runner = worker.model_runner
        kv = runner.kv_cache_config
        layers = [m for m in runner.model.modules() if isinstance(m, MLAAttention)]
        builders = [
            builder
            for groups in runner.attn_groups
            for group in groups
            for builder in group.metadata_builders
        ]
        specs = runner.get_kv_cache_spec()
        roles = {}
        for role in ("sparse", "indexer"):
            matching = [s for s in specs.values() if s.cache_role.value == role]
            roles[role] = {
                "layers": len(matching),
                "pages": sorted(
                    {
                        (
                            s.block_size,
                            s.page_size_bytes,
                            s.dcp_sharded,
                            resolve_dcp_kv_block_size(s, runner.dcp_size),
                        )
                        for s in matching
                    }
                ),
            }
        # Every placement points into one shared backing buffer.
        (planned_bytes,) = {t.size for t in kv.kv_cache_tensors}
        return {
            "dcp": (runner.dcp_size, runner.dcp_rank),
            "roles": roles,
            "blocks": kv.num_blocks,
            "planned_bytes": planned_bytes,
            "allocated_tensors": len(runner.kv_caches),
            "weight_device": next(runner.model.parameters()).device.type,
            "layers": sorted(
                {
                    (
                        layer.get_attn_backend().get_name(),
                        layer.impl.dcp_world_size,
                        layer.impl.dcp_rank,
                        layer.impl.can_return_lse_for_decode,
                        layer.impl.need_to_return_lse_for_decode,
                        layer.dcp_manager is not None,
                    )
                    for layer in layers
                }
            ),
            "latent_metadata": sorted(
                {
                    (
                        b.dcp_world_size,
                        b.cp_kv_cache_interleave_size,
                        b.fp8_use_mixed_batch,
                    )
                    for b in builders
                    if isinstance(b, FlashMLASparseMetadataBuilder)
                }
            ),
            "indexer_metadata": sorted(
                {
                    (b.dcp_world_size, b.dcp_rank, b.cp_kv_cache_interleave_size)
                    for b in builders
                    if isinstance(b, DeepseekV32IndexerMetadataBuilder)
                }
            ),
        }

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.delenv("VLLM_DCP_Q_REPLICATE", raising=False)
    raw = _sparse_dcp_model(tmp_path, target)

    prompts = [TokensPrompt(prompt_token_ids=[7] * n) for n in (67, 73)]
    params = SamplingParams(max_tokens=3, ignore_eos=True, detokenize=False)
    budget = 1 << 27
    totals = {}
    for dcp in (1, 2):
        path = tmp_path / f"dcp{dcp}.jsonl"
        llm = LLM(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            enforce_eager=True,
            tensor_parallel_size=tp,
            decode_context_parallel_size=dcp,
            dcp_comm_backend="ag_rs",
            dcp_q_replicate=False,
            kv_cache_dtype="fp8_ds_mla",
            block_size=64,
            kv_cache_memory_bytes=budget,
            max_model_len=256,
            max_num_batched_tokens=256,
            max_num_seqs=4,
            enable_prefix_caching=False,
            simulation_config={
                "simulate_forward": True,
                "timing": "aisimulate",
                "timing_args": {"gpu": gpu, "target_model": target},
                "log_path": str(path),
            },
        )
        try:
            outputs = llm.generate(prompts, params, use_tqdm=False)
            assert all(o.finished for o in outputs)
            assert [len(o.outputs[0].token_ids) for o in outputs] == [3, 3]
            states = llm.collective_rpc(rank_state)
            capacity = llm.llm_engine.vllm_config.cache_config.kv_cache_size_tokens
        finally:
            llm.llm_engine.engine_core.shutdown()
            del llm
            cleanup_dist_env_and_memory()

        assert len(states) == tp
        for rank, state in enumerate(states):
            dcp_rank = rank % dcp
            assert state["dcp"] == (dcp, dcp_rank)
            assert state["layers"] == [
                ("CPU_SIMULATED_MLA_SPARSE", dcp, dcp_rank, True, dcp > 1, dcp > 1)
            ]
            assert state["latent_metadata"] == [(dcp, 1, True)]
            assert state["indexer_metadata"] == [(dcp, dcp_rank, 1)]
            assert state["weight_device"] == "meta"
            assert state["allocated_tensors"] == 0  # Cache geometry is virtual.
            roles = state["roles"]
            assert roles["sparse"]["layers"] == raw["num_hidden_layers"]
            assert 0 < roles["indexer"]["layers"] <= roles["sparse"]["layers"]
            assert roles["sparse"]["pages"] == [(64, 64 * 656, True, 64 * dcp)]
            assert roles["indexer"]["pages"] == [(64, 64 * 132, True, 64 * dcp)]
            assert 0 < state["planned_bytes"] <= budget
            assert state["blocks"] == states[0]["blocks"]
            assert state["planned_bytes"] == states[0]["planned_bytes"]
        totals[dcp] = (capacity, states[0]["blocks"], states[0]["planned_bytes"])

        records = [json.loads(line) for line in path.read_text().splitlines()]
        (timing,) = [r["timing"]["aisimulate"] for r in records if "timing" in r]
        assert timing["target_model"] == target
        assert timing["kv_cache_format"] == "fp8_ds_mla"
        assert (timing["source"], timing["readiness"]) == ("aic", "ready")
        assert timing["resolved_config"]["estimation_mode"] == "op_level"
        assert timing["resolved_config"]["dcp"] == (dcp if dcp > 1 else None)
        assert timing["config"]["fallback_policy"] == "deny"

        # The scheduler may start one request's decode while the other
        # finishes prefill. Check each first decode independently.
        first_context = {}
        steps = [row for row in records if "step" in row]
        for row in steps:
            requests = row["step"]["requests"]
            for request in requests:
                if not request["is_prefill"]:
                    assert request["num_scheduled_tokens"] == 1
                    first_context.setdefault(
                        request["req_id"], request["num_computed_tokens"]
                    )
        assert sorted(first_context.values()) == [67, 73]
        sdk = RustForwardPassPerfModel.best_available(
            ForwardPassPerfModelConfig(
                model=target,
                system=f"{gpu}_sxm",
                backend="vllm",
                worker_type="aggregated",
                tp=tp,
                dcp=dcp if dcp > 1 else None,
                cp_size=1,
                moe_tp_size=tp,
                moe_ep_size=1,
                gemm_quant_mode=weight_mode,
                moe_quant_mode=weight_mode,
                fmha_quant_mode="bfloat16",
                kvcache_quant_mode="fp8",
                comm_quant_mode="half",
                estimation_mode="op_level",
                database_mode="SILICON",
                fallback_policy="deny",
            )
        )
        direct_ms = []
        for row in steps:
            reqs = row["step"]["requests"]
            prefill = [q for q in reqs if q["is_prefill"]]
            decode_contexts = [
                q["num_computed_tokens"] for q in reqs if not q["is_prefill"]
            ]
            ms = sdk.estimate_forward_pass_time_ms(
                {
                    "scheduled_requests": {
                        "num_prefill_requests": len(prefill),
                        "sum_prefill_tokens": sum(
                            q["num_scheduled_tokens"] for q in prefill
                        ),
                        "var_prefill_length": pvariance(
                            [q["num_prompt_tokens"] for q in prefill] or [0]
                        ),
                        "sum_prefill_kv_tokens": sum(
                            q["num_computed_tokens"] for q in prefill
                        ),
                        "num_decode_requests": len(decode_contexts),
                        "sum_decode_kv_tokens": sum(decode_contexts),
                        "var_decode_kv_tokens": pvariance(decode_contexts or [0]),
                        "extend_lengths": [q["num_scheduled_tokens"] for q in reqs],
                        "past_kv_lengths": [q["num_computed_tokens"] for q in reqs],
                    }
                }
            )
            assert ms > 0
            assert row["cost"]["breakdown"]["aisimulate_ms"] == pytest.approx(
                ms, rel=1e-9
            )
            direct_ms.append(ms)
        print(
            "SPARSE_DCP_STATE",
            json.dumps(
                {
                    "target": target,
                    "tp": tp,
                    "dcp": dcp,
                    "capacity": capacity,
                    "workers": states,
                    "first_decode_contexts": sorted(first_context.values()),
                    "direct_ms": direct_ms,
                }
            ),
        )

    assert totals[1][1:] == totals[2][1:]
    assert totals[2][0] == 2 * totals[1][0]


@pytest.mark.skipif(not current_platform.is_cpu(), reason="Tests the CPU platform.")
@pytest.mark.parametrize(
    "timing, connector, reason",
    [
        ("linear", None, "sparse MLA DCP requires timing aisimulate"),
        ("aisimulate", "NixlConnector", "a model other than dense MLA"),
    ],
)
def test_simulated_cpu_sparse_mla_dcp_scope(tmp_path, timing, connector, reason):
    """The sparse DCP exception never admits another timer or a KV connector."""
    from tests.v1.worker.simulated.test_real_kv_transfer import DEEPSEEK_V32
    from vllm.engine.arg_utils import EngineArgs

    (tmp_path / "config.json").write_text(json.dumps(DEEPSEEK_V32))
    transfer = (
        {"kv_connector": connector, "kv_role": "kv_consumer"} if connector else None
    )
    with pytest.raises(ValueError, match=reason):
        EngineArgs(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            max_model_len=256,
            tensor_parallel_size=2,
            decode_context_parallel_size=2,
            kv_transfer_config=transfer,
            simulation_config={"simulate_forward": True, "timing": timing},
        ).create_engine_config()


@pytest.mark.skipif(not current_platform.is_cpu(), reason="Tests the CPU platform.")
@pytest.mark.parametrize(
    "target, gpu, tp, dcp, reason",
    [
        ("nvidia/GLM-5-NVFP4", "b200", 2, 2, "mixed-batch fp8 path"),
        (
            "deepseek-ai/DeepSeek-V3.2",
            "h200",
            8,
            8,
            "DCP-gathered head counts to pad to the same fp8 decode kernel envelope",
        ),
    ],
)
def test_simulated_cpu_sparse_mla_dcp_keeps_native_head_limits(
    tmp_path, monkeypatch, capfd, target, gpu, tp, dcp, reason
):
    """An SDK estimate cannot override native FlashMLA metadata limits."""
    pytest.importorskip("aisimulate_core")
    from vllm import LLM

    monkeypatch.delenv("VLLM_DCP_Q_REPLICATE", raising=False)
    _sparse_dcp_model(tmp_path, target)
    with pytest.raises(RuntimeError, match="Engine core initialization failed"):
        LLM(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            enforce_eager=True,
            tensor_parallel_size=tp,
            decode_context_parallel_size=dcp,
            dcp_comm_backend="ag_rs",
            dcp_q_replicate=False,
            kv_cache_dtype="fp8_ds_mla",
            block_size=64,
            kv_cache_memory_bytes=1 << 26,
            max_model_len=256,
            max_num_batched_tokens=256,
            max_num_seqs=4,
            enable_prefix_caching=False,
            simulation_config={
                "simulate_forward": True,
                "timing": "aisimulate",
                "timing_args": {"gpu": gpu, "target_model": target},
            },
        )
    assert reason in capfd.readouterr().err


def _tiny_dcp_llm(tmp_path, dcp: int, log_path, **kwargs):
    """A tiny DeepSeek-V3 at TP2 and DCP `dcp`: one 16-token page, prefix
    caching and chunked prefill on, its step log at `log_path`."""
    from tests.v1.worker.simulated.test_real_kv_transfer import DEEPSEEK_V3
    from vllm import LLM

    (tmp_path / "config.json").write_text(json.dumps(DEEPSEEK_V3))
    return LLM(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        tensor_parallel_size=2,
        decode_context_parallel_size=dcp,
        block_size=16,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_model_len=256,
        kv_cache_memory_bytes=1 << 28,
        # A lone instance's rank owns every DCP-th token.
        cp_kv_cache_interleave_size=1,
        simulation_config={"simulate_forward": True, "log_path": str(log_path)},
        **kwargs,
    )


def _step_rows(log_path, prompt_tokens: int) -> list[tuple[int, int, int]]:
    """A prompt's (scheduled, computed, cached) rows from a step log, in step
    order: its prefill chunks, then its decodes."""
    from vllm.v1.worker.simulated.timing import load_step_log

    return [
        (
            request["num_scheduled_tokens"],
            request["num_computed_tokens"],
            request["num_cached_tokens"],
        )
        for record in load_step_log(str(log_path))
        for request in record["step"]["requests"]
        if request["num_prompt_tokens"] == prompt_tokens
    ]


def _local_seq_lens(lengths: list[int], dcp: int) -> list[list[int]]:
    """Each DCP rank's share of global KV lengths at interleave 1, the closed
    form of vLLM's slot mapping (derived token ownership)."""
    import torch

    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

    return get_dcp_local_seq_lens(
        torch.tensor(lengths, dtype=torch.int32), dcp, None, 1
    ).tolist()


@pytest.mark.skipif(not current_platform.is_cpu(), reason="Tests the CPU platform.")
@pytest.mark.parametrize(
    "shared, new, hits",
    [
        # Ends on a DCP2 scheduler block: both layouts credit it.
        (96, 5, {1: 96, 2: 96}),
        # Page-aligned inside a DCP2 block: only DCP1 credits the page.
        (112, 5, {1: 112, 2: 96}),
        # Mid-page: DCP1 rounds down to the page, DCP2 to its block.
        (122, 5, {1: 112, 2: 96}),
    ],
)
def test_simulate_forward_cpu_dcp_prefix_cache_hit_alignment(
    tmp_path, monkeypatch, shared, new, hits
) -> None:
    """Prefix hits land on the scheduler's block, DCP times the local page: a
    hit that ends inside one is not credited at DCP2, and its tokens are
    recomputed, whatever its page alignment."""
    from vllm.distributed import cleanup_dist_env_and_memory

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    # Every token of the warm prefix distinct, its suffixes diverging at
    # `shared` (and in later blocks) from it.
    warm = [i % 128 for i in range(128)]
    prompt = warm[:shared] + [(shared + 100 + i) % 128 for i in range(new)]
    params = SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False)
    for dcp in (1, 2):
        log = tmp_path / f"steps-dcp{dcp}.jsonl"
        llm = _tiny_dcp_llm(tmp_path, dcp, log, max_num_batched_tokens=256)
        try:
            (first,) = llm.generate(
                [{"prompt_token_ids": warm}], params, use_tqdm=False
            )
            (second,) = llm.generate(
                [{"prompt_token_ids": prompt}], params, use_tqdm=False
            )
            assert [len(o.outputs[0].token_ids) for o in (first, second)] == [1, 1]
        finally:
            del llm
            cleanup_dist_env_and_memory()
        assert _step_rows(log, len(warm)) == [(len(warm), 0, 0)]
        hit = hits[dcp]
        assert _step_rows(log, len(prompt)) == [
            (len(prompt) - hit, hit, hit)
        ]
        assert second.num_cached_tokens == hit


@pytest.mark.skipif(not current_platform.is_cpu(), reason="Tests the CPU platform.")
def test_simulate_forward_cpu_dcp_chunked_prefill(tmp_path, monkeypatch) -> None:
    """Chunked prefill splits a prompt at the token budget alone, whatever the
    DCP span: each rank computes its DCP-th tokens of every chunk, and the
    chunks alone bound where they end."""
    from vllm.distributed import cleanup_dist_env_and_memory

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    # Odd length: the last chunk ends mid-page and inside a DCP2 block.
    prompt = [i % 128 for i in range(101)]
    params = SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False)
    shares = {}
    for dcp in (1, 2):
        log = tmp_path / f"steps-dcp{dcp}.jsonl"
        llm = _tiny_dcp_llm(tmp_path, dcp, log, max_num_batched_tokens=40)
        try:
            (output,) = llm.generate(
                [{"prompt_token_ids": prompt}], params, use_tqdm=False
            )
            assert len(output.outputs[0].token_ids) == 1
        finally:
            del llm
            cleanup_dist_env_and_memory()
        rows = _step_rows(log, len(prompt))
        assert rows == [(40, 0, 0), (40, 40, 0), (21, 80, 0)]
        shares[dcp] = _local_seq_lens([c + n for n, c, _ in rows], dcp)
    assert shares[1] == [[40], [80], [101]]
    assert shares[2] == [[20, 20], [40, 40], [51, 50]]


@pytest.mark.skipif(not current_platform.is_cpu(), reason="Tests the CPU platform.")
def test_simulate_forward_cpu_dcp_prefix_cache_chunked_prefill(
    tmp_path, monkeypatch
) -> None:
    """A prefix hit inside a DCP block is recomputed in the first chunk: the
    chunks that follow start from the scheduler's boundary, not from the
    shared prefix's end."""
    from vllm.distributed import cleanup_dist_env_and_memory

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    warm = [i % 128 for i in range(128)]
    shared = 112
    prompt = warm[:shared] + [(shared + 100 + i) % 128 for i in range(100)]
    params = SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False)
    hits = {1: 112, 2: 96}
    for dcp in (1, 2):
        log = tmp_path / f"steps-dcp{dcp}.jsonl"
        llm = _tiny_dcp_llm(tmp_path, dcp, log, max_num_batched_tokens=40)
        try:
            (first,) = llm.generate(
                [{"prompt_token_ids": warm}], params, use_tqdm=False
            )
            (second,) = llm.generate(
                [{"prompt_token_ids": prompt}], params, use_tqdm=False
            )
            assert [len(o.outputs[0].token_ids) for o in (first, second)] == [1, 1]
        finally:
            del llm
            cleanup_dist_env_and_memory()
        assert _step_rows(log, len(warm)) == [
            (40, 0, 0),
            (40, 40, 0),
            (40, 80, 0),
            (8, 120, 0),
        ]
        hit = hits[dcp]
        assert _step_rows(log, len(prompt)) == [
            (40, hit, hit),
            (40, hit + 40, 0),
            (len(prompt) - hit - 80, hit + 80, 0),
        ]
        assert second.num_cached_tokens == hit


@pytest.mark.skipif(not current_platform.is_cpu(), reason="Tests the CPU platform.")
@pytest.mark.parametrize(
    "target, interleave",
    [
        ("Qwen/Qwen3-30B-A3B", 1),
        ("Qwen/Qwen3-30B-A3B", 16),
        ("Qwen/Qwen3-30B-A3B", 32),
        ("Qwen/Qwen3-235B-A22B", 1),
        ("Qwen/Qwen3-235B-A22B", 16),
        ("Qwen/Qwen3-235B-A22B", 32),
    ],
    ids=["30b-i1", "30b-i16", "30b-i32", "235b-i1", "235b-i16", "235b-i32"],
)
def test_simulate_forward_cpu_gqa_dcp(tmp_path, monkeypatch, target, interleave):
    """Use the published Qwen3 geometry and inspect each real CPU worker."""
    pytest.importorskip("aisimulate_core")
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    from tests.v1.worker.simulated.test_aisimulate_timing import bundled_config
    from vllm import LLM
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.inputs import TokensPrompt

    def rank_state(worker) -> dict:
        import torch

        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_block_size
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        runner = worker.model_runner
        kv = runner.kv_cache_config
        tables = runner.block_tables
        layers = [m for m in runner.model.modules() if isinstance(m, Attention)]
        empty = torch.empty(0)
        layer = layers[0]
        try:
            result = layer.impl.forward(layer, empty, empty, empty, empty, None, empty)
        except NotImplementedError as exc:
            forward = str(exc)
        else:
            assert result is empty  # The no-metadata warmup path is a no-op.
            forward = None
        specs = runner.get_kv_cache_spec()
        (planned_bytes,) = {t.size for t in kv.kv_cache_tensors}
        return {
            "layers": len(layers),
            "impl": sorted(
                {
                    (
                        layer.get_attn_backend().get_name(),
                        layer.get_attn_backend().supports_dcp(),
                        layer.impl.dcp_world_size,
                        layer.impl.dcp_rank,
                        getattr(layer.impl, "supports_dcp", False),
                        layer.impl.can_return_lse_for_decode,
                        layer.impl.need_to_return_lse_for_decode,
                        getattr(layer.impl, "cp_kv_cache_interleave_size", None),
                        layer.impl.num_heads,
                        layer.impl.num_kv_heads,
                    )
                    for layer in layers
                }
            ),
            "forward_refusal": forward,
            "spec_layers": len(specs),
            "pages": sorted(
                {
                    (
                        isinstance(s, FullAttentionSpec),
                        s.block_size,
                        s.page_size_bytes,
                        s.num_kv_heads,
                        s.head_size,
                        str(s.dtype),
                        s.dcp_sharded,
                        resolve_dcp_kv_block_size(s, runner.dcp_size),
                    )
                    for s in specs.values()
                }
            ),
            "block_table": (
                tables.cp_size,
                tables.cp_rank,
                tables.cp_interleave,
                tables.block_sizes,
                tables.dcp_sharded.tolist(),
            ),
            "blocks": kv.num_blocks,
            "planned_bytes": planned_bytes,
            "allocated_tensors": len(runner.kv_caches),
            "weight_device": next(runner.model.parameters()).device.type,
        }

    raw = bundled_config(target)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.delenv("VLLM_DCP_Q_REPLICATE", raising=False)
    lengths = [131, 153]
    prompts = [TokensPrompt(prompt_token_ids=[7] * n) for n in lengths]
    params = SamplingParams(max_tokens=3, ignore_eos=True, detokenize=False)
    budget = 1 << 27
    totals = {}
    for dcp in (1, 2):
        log = tmp_path / f"gqa-dcp{dcp}.jsonl"
        llm = LLM(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            enforce_eager=True,
            tensor_parallel_size=8,
            decode_context_parallel_size=dcp,
            dcp_comm_backend="ag_rs",
            dcp_q_replicate=False,
            cp_kv_cache_interleave_size=interleave if dcp > 1 else 1,
            block_size=32,
            kv_cache_memory_bytes=budget,
            max_model_len=256,
            max_num_batched_tokens=128,
            max_num_seqs=4,
            enable_prefix_caching=False,
            simulation_config={
                "simulate_forward": True,
                "timing": "aisimulate",
                "timing_args": {"gpu": "h200", "target_model": target},
                "log_path": str(log),
            },
        )
        try:
            outputs = llm.generate(prompts, params, use_tqdm=False)
            assert all(o.finished for o in outputs)
            assert [len(o.outputs[0].token_ids) for o in outputs] == [3, 3]
            states = llm.collective_rpc(rank_state)
            cache = llm.llm_engine.vllm_config.cache_config
            assert cache.kv_cache_memory_bytes == budget
            capacity = cache.kv_cache_size_tokens
        finally:
            llm.llm_engine.engine_core.shutdown()
            del llm
            cleanup_dist_env_and_memory()

        assert len(states) == 8
        print("GQA_DCP_WORKERS", json.dumps({"dcp": dcp, "states": states}))
        for rank, state in enumerate(states):
            active = dcp > 1
            dcp_rank = rank % dcp
            assert state["layers"] == state["spec_layers"] == raw["num_hidden_layers"]
            assert state["impl"] == [
                (
                    "CPU_ATTN", False, dcp, dcp_rank, active, active, active,
                    interleave if active else None,
                    raw["num_attention_heads"] // 8, 1,
                )
            ]
            if active:
                assert state["forward_refusal"] == (
                    "CPU attention with simulated DCP cannot execute "
                    "an attention forward."
                )
            else:
                assert state["forward_refusal"] is None
            assert state["pages"] == [
                (True, 32, 16384, 1, 128, "torch.bfloat16", True, 32 * dcp)
            ]
            assert state["block_table"] == (
                dcp, dcp_rank, interleave if active else 1, [32], [True]
            )
            assert state["weight_device"] == "meta"
            assert state["allocated_tensors"] == 0
            assert 0 < state["planned_bytes"] <= budget
            assert state["blocks"] == states[0]["blocks"]
            assert state["planned_bytes"] == states[0]["planned_bytes"]
        totals[dcp] = (capacity, states[0]["blocks"], states[0]["planned_bytes"])

        records = [json.loads(line) for line in log.read_text().splitlines()]
        (timing,) = [r["timing"]["aisimulate"] for r in records if "timing" in r]
        assert timing["target_model"] == target
        assert (timing["source"], timing["readiness"]) == ("aic", "ready")
        assert timing["resolved_config"]["dcp"] == (2 if dcp == 2 else None)
        assert timing["resolved_config"]["backend_version"] == "0.24.0"
        sdk = RustForwardPassPerfModel.best_available(
            ForwardPassPerfModelConfig(
                model=target,
                system="h200_sxm",
                backend="vllm",
                backend_version="0.24.0",
                worker_type="aggregated",
                tp=8,
                dcp=2 if dcp == 2 else None,
                cp_size=1,
                moe_tp_size=8,
                moe_ep_size=1,
                gemm_quant_mode="bfloat16",
                moe_quant_mode="bfloat16",
                fmha_quant_mode="bfloat16",
                kvcache_quant_mode="bfloat16",
                comm_quant_mode="half",
                estimation_mode="op_level",
                database_mode="SILICON",
                fallback_policy="deny",
            )
        )
        first_context = {}
        cached_prefill = False
        direct_ms = []
        for row in records:
            if "step" not in row:
                continue
            reqs = row["step"]["requests"]
            prefill = [q for q in reqs if q["is_prefill"]]
            decode_contexts = [
                q["num_computed_tokens"] for q in reqs if not q["is_prefill"]
            ]
            for q in reqs:
                if not q["is_prefill"]:
                    assert q["num_scheduled_tokens"] == 1
                    first_context.setdefault(q["req_id"], q["num_computed_tokens"])
            cached_prefill |= any(q["num_computed_tokens"] > 0 for q in prefill)
            ms = sdk.estimate_forward_pass_time_ms(
                {
                    "scheduled_requests": {
                        "num_prefill_requests": len(prefill),
                        "sum_prefill_tokens": sum(
                            q["num_scheduled_tokens"] for q in prefill
                        ),
                        "var_prefill_length": pvariance(
                            [q["num_prompt_tokens"] for q in prefill] or [0]
                        ),
                        "sum_prefill_kv_tokens": sum(
                            q["num_computed_tokens"] for q in prefill
                        ),
                        "num_decode_requests": len(decode_contexts),
                        "sum_decode_kv_tokens": sum(decode_contexts),
                        "var_decode_kv_tokens": pvariance(decode_contexts or [0]),
                        "extend_lengths": [q["num_scheduled_tokens"] for q in reqs],
                        "past_kv_lengths": [q["num_computed_tokens"] for q in reqs],
                    }
                }
            )
            assert ms > 0
            assert row["cost"]["breakdown"]["aisimulate_ms"] == pytest.approx(
                ms, rel=1e-9
            )
            direct_ms.append(ms)
        assert sorted(first_context.values()) == lengths
        assert cached_prefill
        print(
            "GQA_DCP_TIMING",
            json.dumps(
                {
                    "dcp": dcp,
                    "capacity": capacity,
                    "first_decode_contexts": sorted(first_context.values()),
                    "direct_ms": direct_ms,
                }
            ),
        )

    assert totals[1][1:] == totals[2][1:]
    assert totals[2][0] == 2 * totals[1][0]


def mtp_dcp_engine_cell(tmp_path, monkeypatch, target, tp, dcp, k, references):
    """One real CPU engine of a qualified MTP cell: inspect each worker's
    draft, cache geometry and block spans, and independently price every
    logged step. `references` keeps the DCP1 capacity of each (target, tp)."""
    pytest.importorskip("aisimulate_core")
    import math
    import os
    from pathlib import Path

    from tests.v1.worker.simulated.test_aisimulate_timing import (
        GLM5,
        MTP_DCP_PROFILES,
        V3,
        V32,
        _mtp_dcp_direct_sdk,
        bundled_config,
    )
    from vllm import LLM
    from vllm.distributed import cleanup_dist_env_and_memory

    def rank_state(worker):
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention

        runner = worker.model_runner
        cache = runner.kv_cache_config
        specs = worker.get_kv_cache_spec()
        draft_names = sorted(runner.speculator.draft_attn_layer_names)
        layers = get_layers_from_vllm_config(worker.vllm_config, MLAAttention)

        def identity(layer):
            impl = layer.impl
            return (
                layer.get_attn_backend().get_name(),
                impl.dcp_world_size,
                impl.dcp_rank,
                impl.can_return_lse_for_decode,
                impl.need_to_return_lse_for_decode,
            )

        roles = {}
        for name, spec in specs.items():
            role = spec.cache_role.value
            roles.setdefault(role, []).append(
                (name, spec.block_size, spec.page_size_bytes, spec.dcp_sharded,
                 str(spec.dtype), spec.is_draft)
            )
        return {
            "rank": worker.rank,
            "speculator": type(runner.speculator).__name__,
            "draft_model": type(runner.speculator.model).__name__,
            "draft_meta": next(runner.speculator.model.parameters()).is_meta,
            "target_meta": next(runner.get_model().parameters()).is_meta,
            "speculator_dcp": runner.speculator.dcp_size,
            "draft_names": draft_names,
            "draft_attention": {
                n: identity(layers[n]) for n in draft_names if n in layers
            },
            "draft_marked": {
                n: layers[n].is_draft_layer for n in draft_names if n in layers
            },
            "target_impls": sorted(
                {identity(m) for n, m in layers.items() if n not in draft_names}
            ),
            "roles": {
                role: {
                    "layers": len(rows),
                    "geometry": sorted({row[1:5] for row in rows}),
                    "draft": sorted(row[0] for row in rows if row[5]),
                }
                for role, rows in roles.items()
            },
            "group_members": [sorted(g.layer_names) for g in cache.kv_cache_groups],
            "spec_pages": sum(s.page_size_bytes for s in specs.values()),
            "table": (
                runner.block_tables.cp_size,
                runner.block_tables.cp_rank,
                runner.block_tables.cp_interleave,
                runner.block_tables.block_sizes,
                runner.block_tables.dcp_sharded.tolist(),
            ),
            "blocks": cache.num_blocks,
            "planned_bytes": sorted({t.size for t in cache.kv_cache_tensors}),
            "allocated_tensors": len(runner.kv_caches),
        }

    gpu, _, _, page = MTP_DCP_PROFILES[target]
    sparse = target != V3
    source = os.getenv("ROUND22_PUBLISHED_V3")
    raw = (
        json.loads(Path(source).read_text())
        if target == V3 and source
        else bundled_config(target)
    )
    if sparse:
        raw = {key: value for key, value in raw.items() if key != "quantization_config"}
        (tmp_path / "DERIVED.md").write_text("published geometry, no weight kernels")
    (tmp_path / "config.json").write_text(json.dumps(raw))
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.delenv("VLLM_DCP_Q_REPLICATE", raising=False)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    layers_total = raw["num_hidden_layers"]
    lengths = [1024, 1031]
    prompts = [{"prompt_token_ids": [7] * n} for n in lengths]
    budget = 1 << 29
    log = tmp_path / "steps.jsonl"
    extra = {"kv_cache_dtype": "fp8_ds_mla"} if sparse else {}
    llm = LLM(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        tensor_parallel_size=tp,
        decode_context_parallel_size=dcp,
        dcp_comm_backend="ag_rs",
        dcp_q_replicate=False,
        cp_kv_cache_interleave_size=1,
        block_size=page,
        kv_cache_memory_bytes=budget,
        max_model_len=2048,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        enable_prefix_caching=False,
        seed=1234,
        speculative_config={"method": "mtp", "num_speculative_tokens": k},
        simulation_config={
            "simulate_forward": True,
            "timing": "aisimulate",
            "timing_args": {"gpu": gpu, "target_model": target},
            "spec_acceptance_rates": [0.7],
            "log_path": str(log),
        },
        **extra,
    )
    try:
        outputs = llm.generate(
            prompts,
            SamplingParams(max_tokens=16, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
        cfg = llm.llm_engine.vllm_config
        capacity = cfg.cache_config.kv_cache_size_tokens
        lookahead = cfg.num_lookahead_tokens
        resolved_page = cfg.cache_config.block_size
        states = llm.collective_rpc(rank_state)
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        cleanup_dist_env_and_memory()
    assert all(
        out.finished and out.outputs[0].finish_reason == "length" for out in outputs
    )
    emitted = {o.request_id: len(o.outputs[0].token_ids) for o in outputs}
    assert list(emitted.values()) == [16, 16]
    assert lookahead == k
    assert resolved_page == page
    assert len(states) == tp
    draft_layer = f"model.layers.{layers_total}.self_attn.attn"
    for rank, state in enumerate(states):
        assert state["rank"] == rank
        assert state["speculator"] == "MTPSpeculator"
        assert state["draft_meta"] and state["target_meta"]
        assert state["speculator_dcp"] == dcp
        assert draft_layer in state["draft_names"]
        assert state["draft_marked"] == {draft_layer: True}
        backend = "CPU_SIMULATED_MLA_SPARSE" if sparse else "CPU_MLA"
        identity = (backend, dcp, rank % dcp, dcp > 1 or sparse, dcp > 1)
        assert state["draft_attention"] == {draft_layer: identity}
        assert state["target_impls"] == [identity]
        roles = state["roles"]
        draft_attn = f"model.layers.{layers_total}.self_attn.attn"
        draft_indexer = f"model.layers.{layers_total}.self_attn.indexer.k_cache"
        # A dense MLA latent is the one "sparse"-role cache of its group.
        assert sorted(roles) == (["indexer", "sparse"] if sparse else ["sparse"])
        assert state["draft_names"] == (
            [draft_attn, draft_indexer] if sparse else [draft_attn]
        )
        for role in roles.values():
            assert [g[0] for g in role["geometry"]] == [page]
            assert all(g[2] for g in role["geometry"])
            assert role["layers"] == layers_total + 1
        assert roles["sparse"]["draft"] == [draft_attn]
        if sparse:
            assert roles["indexer"]["draft"] == [draft_indexer]
            assert [g[:3] for g in roles["sparse"]["geometry"]] == [
                (page, page * 656, True)
            ]
            assert [g[:3] for g in roles["indexer"]["geometry"]] == [
                (page, page * 132, True)
            ]
        else:
            assert roles["sparse"]["geometry"] == [
                (16, 18432, True, "torch.bfloat16")
            ]
        # Each draft cache sits in the resolved group of the target's layers,
        # which has one block table row and one block span for all of them.
        (members,) = state["group_members"]
        assert len(members) == (2 if sparse else 1) * (layers_total + 1)
        assert set(state["draft_names"]) <= set(members)
        assert "model.layers.0.self_attn.attn" in members
        assert state["table"][:3] == (dcp, rank % dcp, 1)
        assert state["table"][3] == [page] * len(state["table"][3])
        assert state["allocated_tensors"] == 0
        (planned,) = state["planned_bytes"]
        assert planned == state["blocks"] * state["spec_pages"]
        assert state["blocks"] == states[0]["blocks"]
    blocks = states[0]["blocks"]
    assert capacity == blocks * page * dcp
    key = (target, tp)
    if dcp == 1:
        references[key] = (capacity, blocks, states[0]["planned_bytes"])
    else:
        # The same planned memory holds the same blocks; each spans DCP pages.
        assert (blocks, states[0]["planned_bytes"]) == references[key][1:]
        assert capacity == dcp * references[key][0]

    records = [json.loads(line) for line in log.read_text().splitlines()]
    (timing,) = [r["timing"]["aisimulate"] for r in records if "timing" in r]
    assert timing["target_model"] == target
    assert timing["resolved_config"]["nextn"] == k
    assert timing["resolved_config"]["dcp"] == (dcp if dcp > 1 else None)
    assert timing["speculation"]["verify_width"] == k + 1
    assert timing["resolved_config"]["backend_version"] == "0.24.0"
    accepted = {
        rec["step_index"]: rec["requests"]
        for row in records if (rec := row.get("spec_acceptance"))
    }
    first_decode = {}
    produced = {}
    scheduled_drafts = accepted_drafts = 0
    boundary_horizons = set()
    peak_horizons = {}
    sdk = _mtp_dcp_direct_sdk(target, tp, dcp, k)
    checks = []
    span = page * dcp
    for row in (r for r in records if "step" in r):
        step, cost = row["step"], row["cost"]
        requests = step["requests"]
        prefill = [q for q in requests if q["is_prefill"]]
        decode = [q for q in requests if not q["is_prefill"]]
        contexts = [q["num_computed_tokens"] for q in decode]
        samples = accepted.pop(step["step_index"], {})
        assert samples.keys() == {q["req_id"] for q in decode}
        assert cost["breakdown"]["spec_requests_below_width"] == 0
        assert cost["breakdown"]["spec_draft_tokens"] == k * len(decode)
        for i, q in enumerate(requests):
            key_ = q["req_id"].rsplit("-", 1)[0]
            end = q["num_computed_tokens"] + q["num_scheduled_tokens"]
            # Calculated horizon; the simulator logs no reservations or writes.
            horizon = math.ceil(min(2048, end + lookahead) / span)
            peak_horizons[key_] = max(horizon, peak_horizons.get(key_, 0))
            if end == 1024:
                boundary_horizons.add(horizon)
            if q["is_prefill"]:
                if end == q["num_prompt_tokens"]:
                    produced.setdefault(key_, []).append(1)
                continue
            first_decode.setdefault(key_, q["num_computed_tokens"])
            assert (q["num_scheduled_tokens"], q["num_spec_tokens"]) == (k + 1, k)
            drafted, wins = samples[q["req_id"]]
            assert drafted == k and 0 <= wins <= k
            assert cost["num_accepted"][i] == wins
            scheduled_drafts += drafted
            accepted_drafts += wins
            produced.setdefault(key_, []).append(1 + wins)
        metrics = {
            "scheduled_requests": {
                "num_prefill_requests": len(prefill),
                "sum_prefill_tokens": sum(q["num_scheduled_tokens"] for q in prefill),
                "var_prefill_length": pvariance(
                    [q["num_prompt_tokens"] for q in prefill] or [0]
                ),
                "sum_prefill_kv_tokens": sum(q["num_computed_tokens"] for q in prefill),
                "num_decode_requests": len(decode),
                "sum_decode_kv_tokens": sum(contexts),
                "var_decode_kv_tokens": pvariance(contexts or [0]),
                "extend_lengths": [q["num_scheduled_tokens"] for q in requests],
                "past_kv_lengths": [q["num_computed_tokens"] for q in requests],
            }
        }
        ms = sdk.estimate_forward_pass_time_ms(metrics)
        logged = cost["breakdown"]["aisimulate_ms"]
        assert ms > 0 and logged == pytest.approx(ms, rel=1e-9)
        checks.append({"step": step["step_index"], "global_metrics": metrics,
                       "sdk_ms": ms, "logged_ms": logged})
    # The step ending at 1,024 tokens: the lookahead adds one page span.
    assert boundary_horizons == {1024 // span + 1}
    assert 0 < max(peak_horizons.values()) <= blocks
    assert not accepted
    assert 0 < accepted_drafts < scheduled_drafts
    assert sorted(first_decode.values()) == lengths
    assert produced.keys() == emitted.keys()
    discarded = {}
    for key_, counts in produced.items():
        # The step reaching max_tokens keeps only what completes the output;
        # the async scheduler can have one more full-width step in flight.
        finishing = next(
            i for i in range(len(counts)) if sum(counts[: i + 1]) >= emitted[key_]
        )
        assert sum(counts[:finishing]) < emitted[key_]
        assert len(counts) - finishing - 1 <= 1
        discarded[key_] = sum(counts) - emitted[key_]
    summary = {
        "target": target, "tp": tp, "dcp": dcp, "k": k, "page": page,
        "workers": states, "capacity": capacity, "lookahead": lookahead,
        "first_decode": first_decode, "completed": emitted,
        "calculated_boundary_horizon": sorted(boundary_horizons),
        "calculated_peak_horizons": peak_horizons,
        "samples_discarded_at_stop": discarded,
        "drafts": scheduled_drafts, "accepted": accepted_drafts, "steps": checks,
    }
    (tmp_path / "validation.json").write_text(json.dumps(summary, indent=2))
    print(f"MTP_DCP {target} TP{tp}/DCP{dcp}/k{k}: {capacity} tokens, "
          f"{len(checks)} steps, {accepted_drafts}/{scheduled_drafts} accepted")
    return summary


@pytest.mark.skipif(not current_platform.is_cpu(), reason="Tests the CPU platform.")
@pytest.mark.parametrize(
    "target, tp, dcp, k",
    [
        ("deepseek-ai/DeepSeek-V3", 2, 2, 1),
        ("deepseek-ai/DeepSeek-V3", 4, 4, 1),
        ("deepseek-ai/DeepSeek-V3", 8, 2, 2),
        ("deepseek-ai/DeepSeek-V3.2", 8, 4, 2),
        ("nvidia/GLM-5-NVFP4", 4, 2, 3),
        ("deepseek-ai/DeepSeek-V3", 8, 8, 5),
        ("deepseek-ai/DeepSeek-V3.2", 8, 2, 5),
        ("nvidia/GLM-5-NVFP4", 4, 4, 4),
    ],
    ids=["v3-tp2-dcp2-k1", "v3-tp4-dcp4-k1", "v3-tp8-dcp2-k2",
         "v32-tp8-dcp4-k2", "glm5-tp4-dcp2-k3", "v3-tp8-dcp8-k5",
         "v32-tp8-dcp2-k5", "glm5-tp4-dcp4-k4"],
)
def test_simulate_forward_cpu_mtp_dcp(tmp_path, monkeypatch, target, tp, dcp, k):
    """The published geometry's draft, cache spans and AISimulate estimates
    under DCP, against the same planned memory at DCP 1."""
    references = {}
    for size in (1, dcp):
        directory = tmp_path / f"dcp{size}"
        directory.mkdir()
        mtp_dcp_engine_cell(directory, monkeypatch, target, tp, size, k, references)



def _round23_native_peer():
    """Bounded subprocess with real workers; run as a standalone entry point."""
    import sys
    import time
    from pathlib import Path

    from vllm import LLM, SamplingParams
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlAgentMetadata,
    )

    role, model, target, spec, dest = sys.argv[1:]
    options = json.loads(spec)
    prefill = role == "prefill"
    directory = Path(dest)
    output = directory / f"{role}.json"
    pattern = options.pop("check", True)
    prompt = options.pop("prompt")
    name = options.pop("backend")
    attest_state = options.pop("attest_state", False)
    corrupt = options.pop("corrupt", False)
    sim = {
        "simulate_forward": True,
        "timing": "aisimulate",
        "timing_args": {
            "gpu": "h200" if not target.startswith("nvidia/") else "h100",
            "target_model": target,
        },
        "kv_kernel_backend": name,
        "log_path": str(directory / f"{role}.steps"),
    }
    if pattern:
        sim["kv_pattern_fill" if prefill else "kv_pattern_check"] = True
    llm = LLM(
        model=model,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=2304 if attest_state or target.startswith("nvidia/") else 512,
        max_num_batched_tokens=2304 if target.startswith("nvidia/") else 512,
        max_num_seqs=4,
        kv_cache_memory_bytes=128 << 20,
        simulation_config=sim,
        kv_transfer_config={
            "kv_connector": "NixlConnector",
            "kv_buffer_device": "cpu",
            "kv_role": "kv_producer" if prefill else "kv_consumer",
            "kv_connector_extra_config": {
                "backends": ["Mooncake"],
                "kv_lease_duration": 45,
            },
        },
        **options,
    )

    if attest_state:
        import dataclasses

        import msgspec

        def state_metadata(worker):
            connector = get_kv_transfer_group().connector_worker
            payload = connector.xfer_handshake_metadata
            return payload.agent_metadata_bytes

        def state_refusal(worker, packed, tp):
            connector = get_kv_transfer_group().connector_worker
            meta = msgspec.msgpack.decode(packed, type=NixlAgentMetadata)
            regions = connector._ssm_region_indices
            lens = list(meta.block_lens)
            strides = list(meta.block_strides)
            for region in regions:
                lens[region] = strides[region] = 1
            bad = dataclasses.replace(meta, block_lens=lens, block_strides=strides)
            try:
                connector.add_remote_agent(bad, remote_tp_size=tp)
            except ValueError as exc:
                return str(exc)
            return None

        if prefill:
            data = llm.collective_rpc(state_metadata)
            (directory / "peer-meta").write_bytes(data[0])
        else:
            reasons = llm.collective_rpc(
                state_refusal, args=((directory / "peer-meta").read_bytes(), 4)
            )
            (directory / "state-refusal.json").write_text(json.dumps(reasons))

    token_ids = [5] * prompt
    if prefill:
        handoff = {"do_remote_decode": True}
    else:
        handoff = json.loads((directory / "prefill.json").read_text())["handoff"]
    params = SamplingParams(
        max_tokens=1 if prefill else 2,
        ignore_eos=True,
        detokenize=False,
        extra_args={"kv_transfer_params": handoff},
    )
    (response,) = llm.generate(
        [{"prompt_token_ids": token_ids}], params, use_tqdm=False
    )
    value = {
        "handoff": response.kv_transfer_params,
        "tokens": list(response.outputs[0].token_ids),
    }
    if prefill and corrupt:

        def flip_prefill_byte(worker, handoff):
            cache = next(
                iter(get_kv_transfer_group().connector_worker.device_kv_caches.values())
            )
            block = handoff["remote_block_ids"][0][0] * 2
            raw = cache[block].view(-1).view(torch.uint8)
            raw[0] ^= 1
            return int(raw.numel())

        import torch

        assert all(llm.collective_rpc(flip_prefill_byte, args=(value["handoff"],)))

    staging = output.with_suffix(".tmp")
    staging.write_text(json.dumps(value))
    staging.rename(output)
    if prefill:
        core = llm.llm_engine.engine_core.engine_core
        end = time.monotonic() + 180
        while not (directory / "stop").exists() and time.monotonic() < end:
            core.step()
            time.sleep(0.02)


@pytest.mark.parametrize(
    "case",
    [
        "hybrid",
        "hybrid-unchecked",
        "hybrid-prefix",
        "dcp",
        "dcp-asymmetric",
        "dcp-corrupt",
        "gqa-declared-prefill",
        "gqa-declared-decode",
        "gqa-remote-only-split",
        "gqa-split-both",
        "gqa-split-decode-whole-producer",
        "gqa-split-unequal",
        "gqa-undeclared-split-producer",
        "gqa-undeclared-split-producer-unchecked",
        "mla-unequal-split",
        "mla-unequal-split-unchecked",
    ],
)
def test_declared_geometry_moves_real_kernel_rows(tmp_path, case):
    import os
    import socket
    import subprocess
    import sys
    import time
    from pathlib import Path

    from tests.v1.worker.simulated.test_kernel_geometry import NANO, V3, published_path

    pytest.importorskip("aisimulate_core")
    pytest.importorskip("nixl")
    hybrid = case.startswith("hybrid")
    gqa = case.startswith("gqa")
    unequal_mla = case.startswith("mla-unequal-split")
    refusal = None
    if case == "hybrid-prefix":
        refusal = (
            "Prefix caching with heterogeneous physical_blocks_per_logical"
            " is not supported for Mamba hybrid models."
        )
    elif unequal_mla:
        refusal = (
            "Simulated NIXL with split MLA kernel blocks requires equal P/D "
            "logical block sizes (local 256, remote 128)."
        )
    elif case.startswith("gqa-undeclared-split-producer"):
        refusal = (
            "Simulated NIXL from a peer whose blocks hold several kernel blocks "
            "(2 of 64 tokens) needs this instance to declare its kernel geometry "
            "too (simulation_config.kv_kernel_backend)."
        )
    target = NANO if hybrid else "Qwen/Qwen3-8B" if gqa else V3
    model = published_path(target)
    common = {
        "prompt": 1089 if hybrid else 384,
        "enable_prefix_caching": case == "hybrid-prefix",
        "check": not case.endswith("unchecked"),
        "attest_state": hybrid and case != "hybrid-prefix",
        "corrupt": case == "dcp-corrupt",
    }
    p = {
        **common,
        "tensor_parallel_size": 4 if hybrid else 2,
        "decode_context_parallel_size": 1,
        "block_size": 1088 if hybrid else 128,
        "backend": "FLASHINFER" if hybrid else "FLASHMLA",
    }
    d = {
        **common,
        "tensor_parallel_size": 2,
        "decode_context_parallel_size": 1 if hybrid else 2,
        "block_size": 2112 if hybrid else 128,
        "backend": "FLASHINFER"
        if hybrid
        else "FLASH_ATTN_MLA"
        if case == "dcp-asymmetric"
        else "FLASHMLA",
    }
    if gqa:
        p.update(prompt=256, tensor_parallel_size=1, backend="TRITON_ATTN")
        d.update(
            prompt=256,
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            backend="TRITON_ATTN",
        )
        if case == "gqa-declared-prefill":
            d["backend"] = None
        elif case == "gqa-declared-decode":
            p["backend"] = None
        elif case.startswith("gqa-undeclared-split-producer"):
            p["backend"] = "FLASHINFER"
            d["backend"] = None
        elif case == "gqa-split-decode-whole-producer":
            p.update(backend=None, block_size=64)
            d["backend"] = "FLASHINFER"
        else:
            p["backend"] = d["backend"] = "FLASHINFER"
            if case == "gqa-remote-only-split":
                d["block_size"] = 64
            elif case == "gqa-split-unequal":
                d["block_size"] = 256
    if unequal_mla:
        p.update(prompt=256, tensor_parallel_size=1)
        d.update(
            prompt=256,
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            block_size=256,
        )
    # The script imports this function from the package, so its callables are
    # importable when collectives serialize them to spawned workers.
    script = tmp_path / "peer.py"
    script.write_text(
        "from tests.v1.e2e.test_simulated_forward import _round23_native_peer\n"
        "if __name__ == '__main__':\n    _round23_native_peer()\n"
    )

    def command(role, settings):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {
            **os.environ,
            "PYTHONPATH": str(Path.cwd()),
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "VLLM_HOST_IP": "127.0.0.1",
            "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(port),
            "NIXL_MOONCAKE_IP_ADDR": "127.0.0.1",
            "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
        }
        return [
            sys.executable,
            str(script),
            role,
            str(model),
            target,
            json.dumps(settings),
            str(tmp_path),
        ], env

    producer_log = tmp_path / "prefill.log"
    producer_command, env = command("prefill", p)
    with producer_log.open("w") as f:
        producer = subprocess.Popen(
            producer_command, stdout=f, stderr=subprocess.STDOUT, env=env
        )
    try:
        end = time.monotonic() + 240
        while not (tmp_path / "prefill.json").exists():
            assert producer.poll() is None, producer_log.read_text()[-8000:]
            assert time.monotonic() < end, producer_log.read_text()[-8000:]
            time.sleep(0.2)
        consumer_command, env = command("decode", d)
        with (tmp_path / "decode.log").open("w") as f:
            decoder = subprocess.run(
                consumer_command,
                env=env,
                timeout=180,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
        log = (tmp_path / "decode.log").read_text()
        if refusal:
            assert refusal in log, log[-8000:]
        else:
            assert decoder.returncode == 0, log[-8000:]
            assert (
                len(json.loads((tmp_path / "decode.json").read_text())["tokens"]) == 2
            )
    finally:
        (tmp_path / "stop").touch()
        try:
            producer.wait(timeout=60)
        finally:
            if producer.poll() is None:
                producer.kill()

    def step_records(role, rank):
        suffix = "" if rank == 0 else f".tp{rank}"
        step_log = tmp_path / f"{role}{suffix}.steps"
        return [json.loads(line) for line in step_log.read_text().splitlines()]

    if hybrid and case != "hybrid-prefix":
        reasons = json.loads((tmp_path / "state-refusal.json").read_text())
        assert all(
            r and "are shorter than its state (533504 bytes)" in r for r in reasons
        )
    for rank in range(d["tensor_parallel_size"]):
        records = step_records("decode", rank)
        (layout,) = [r["kv_layout"] for r in records if "kv_layout" in r]
        if gqa:
            for role, settings in (("prefill", p), ("decode", d)):
                (info,) = [
                    r["kv_layout"]
                    for r in step_records(role, rank)
                    if "kv_layout" in r
                ]
                geometry = info.get("kernel_geometry")
                if settings["backend"] is None:
                    assert geometry is None
                    continue
                assert geometry["registered_backend"] == "CPU_ATTN"
                assert geometry["logical_tokens"] == settings["block_size"]
                kernel = 64 if settings["backend"] == "FLASHINFER" else 128
                assert geometry["kernel_tokens"] == kernel
                assert geometry["kernel_blocks_per_logical"] == (
                    settings["block_size"] // kernel
                )
            if refusal:
                assert not any(
                    r.get("kv_transfers", {}).get("load", {}).get("each")
                    for r in records
                )
                assert all(
                    r["kv_check"].get("error") == "receive failed"
                    for r in records
                    if "kv_check" in r
                )
                if not common["check"]:
                    assert not any("kv_check" in r for r in records)
                continue
            (check,) = [r["kv_check"] for r in records if "kv_check" in r]
            assert check["mismatched_bytes"] == 0 and check["ok"], check
            expect_bytes = 256 * 36 * 2 * 8 * 128 * 2
            assert check["bytes"] == expect_bytes
            if case in (
                "gqa-remote-only-split",
                "gqa-split-both",
                "gqa-split-decode-whole-producer",
                "gqa-split-unequal",
            ):
                coverage = check["coverage"]
                assert coverage["problems"] == []
                (read,) = coverage["reads"]
                assert (read["rank"], read["completed"]) == (0, True)
                assert (read["descriptors"], read["bytes"]) == (4 * 36, expect_bytes)
            loads = [
                entry
                for r in records
                for entry in r.get("kv_transfers", {}).get("load", {}).get("each", ())
            ]
            (load,) = loads
            assert load[0] == expect_bytes, load  # NIXL's per-transfer bytes
            continue
        geometry = layout["kernel_geometry"]
        assert geometry["logical_tokens"] == d["block_size"]
        assert geometry["registered_logical_blocks"] == layout["num_blocks"]
        assert geometry["registered_physical_blocks"] == (
            layout["num_blocks"] * geometry["kernel_blocks_per_logical"]
        )
        if hybrid:
            assert geometry["registered_backend"] == "MAMBA2_ATTN"
            assert set(geometry["runtime_backends"]) == {"MAMBA2_ATTN", "CPU_ATTN"}
        else:
            assert geometry["registered_backend"] == "CPU_MLA"
            assert geometry["runtime_backends"] == ["CPU_MLA"]
        assert geometry["allocated_unique_bytes"] == layout["bytes"]
        assert geometry["registered_bytes"] == layout["bytes"]
        if refusal:
            assert not any(
                r.get("kv_transfers", {}).get("load", {}).get("each") for r in records
            )
            continue
        if not common["check"]:
            assert not any("kv_check" in r for r in records)
            continue
        (check,) = [r["kv_check"] for r in records if "kv_check" in r]
        if case == "dcp-corrupt" and rank == 0:
            assert not check["ok"] and check["mismatched_bytes"] == 1, check
        else:
            assert check["ok"], check
        assert check["coverage"]["problems"] == []
        reads = check["coverage"]["reads"]
        if hybrid:
            assert [r["rank"] for r in reads] == [2 * rank, 2 * rank + 1]
            assert [r["bytes"] for r in reads] == [16_146_432, 12_804_096]
            assert [r["descriptors"] for r in reads] == [198, 96]
            assert sum(g.get("zeroed_bytes", 0) for g in check["groups"]) == 3_145_728
            assert sum(g.get("padding_bytes", 0) for g in check["groups"]) == 344_064
        else:
            assert [r["rank"] for r in reads] == [rank]
            assert reads[0]["bytes"] == check["bytes"] == (2 - rank) * 8_994_816
            assert reads[0]["descriptors"] == (2 - rank) * 122
            assert check["untouched_bytes"] == rank * 8_994_816
