# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from pathlib import Path

import pytest
import torch

from vllm import EngineArgs
from vllm.config import set_current_vllm_config
from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend
from vllm.v1.attention.backends.mla.cpu_mla import CPUMLABackend
from vllm.v1.worker.simulated.kernel_geometry import declared_rule, read_rule
from vllm.v1.worker.utils import select_common_block_size

V3 = "deepseek-ai/DeepSeek-V3"
NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def published_path(model):
    root = os.environ.get("R23_MODELS")
    if root is None:
        pytest.skip("set R23_MODELS to the directory of pinned published model configs")
    return next((Path(root) / model.replace("/", "--")).glob("*/config.json")).parent


def engine(model, name=None, gpu="h100", *, sim_overrides=None, **kw):
    simulation = {
        "simulate_forward": True,
        "timing": "aisimulate",
        "timing_args": {"gpu": gpu, "target_model": model},
        "kv_kernel_backend": name,
        **(sim_overrides or {}),
    }
    kwargs = {
        "model": str(published_path(model)),
        "skip_tokenizer_init": True,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "block_size": 128,
        "max_model_len": 256,
        "enable_prefix_caching": False,
        "kv_cache_memory_bytes": 128 << 20,
        "kv_transfer_config": {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer",
            "kv_buffer_device": "cpu",
        },
        "simulation_config": simulation,
        **kw,
    }
    return EngineArgs(**kwargs).create_engine_config()


@pytest.mark.parametrize(
    ("model", "name", "gpu", "logical", "kernel"),
    [
        (V3, None, "h100", 128, 128),
        (V3, "FLASHMLA", "h100", 128, 64),
        (V3, "FLASHINFER_MLA", "b200", 96, 32),
        (V3, "FLASH_ATTN_MLA", "h100", 128, 128),
        (V3, "TRITON_MLA", "a100", 128, 128),
        (NANO, None, "h100", 2112, 2112),
        (NANO, "FLASHINFER", "h100", 2112, 64),
        (NANO, "FLASHINFER", "h100", 1056, 32),
        (NANO, "TRITON_ATTN", "h100", 1056, 1056),
    ],
)
def test_selects_declared_kernel_divisor(model, name, gpu, logical, kernel):
    config = engine(model, name, gpu)
    cpu = CPUMLABackend if model == V3 else CPUAttentionBackend
    with set_current_vllm_config(config):
        assert select_common_block_size(logical, [cpu]) == kernel
    rule = declared_rule(config)
    if name is None:
        assert rule is None
    else:
        assert rule.source.startswith("vllm/v1/attention/backends/")
        assert len(rule.source_sha256) == 64


@pytest.mark.parametrize(
    ("name", "gpu", "kw", "sim", "message"),
    [
        ("32", "h100", {}, {}, "Unknown kv_kernel_backend"),
        ("FLASHMLA", "h100", {"kv_transfer_config": None}, {}, "NixlConnector"),
        ("FLASHMLA", "h100", {}, {"timing": "linear"}, "timing aisimulate"),
        ("FLASHMLA", "h100", {}, {"simulate_forward": False}, "CPU simulated forward"),
        ("FLASHMLA", "b200", {}, {}, "does not describe"),
        ("FLASHMLA", "h100", {"kv_cache_dtype": "fp8"}, {}, "requires BF16 KV"),
        ("FLASHINFER", "h100", {}, {}, "does not describe"),
        (
            "FLASHMLA",
            "h100",
            {
                "tensor_parallel_size": 2,
                "speculative_config": {
                    "method": "mtp",
                    "num_speculative_tokens": 1,
                },
            },
            {},
            "no drafts",
        ),
    ],
)
def test_unsupported_declarations_fail_in_engine_config(name, gpu, kw, sim, message):
    with pytest.raises(ValueError, match=message):
        engine(V3, name, gpu, sim_overrides=sim, **kw)


def test_cannot_invent_a_divisor():
    config = engine(V3, "FLASHMLA")
    with set_current_vllm_config(config), pytest.raises(ValueError, match="No common"):
        select_common_block_size(96, [CPUMLABackend])


def test_missing_or_nonliteral_source_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_rule("FLASHMLA", "vllm.backend.Native", tmp_path)
    (tmp_path / "backend.py").write_text(
        "class Native:\n"
        "    def get_supported_kernel_block_sizes():\n"
        "        return choose()\n"
    )
    with pytest.raises(ValueError, match="literal kernel block size list"):
        read_rule("FLASHMLA", "vllm.backend.Native", tmp_path)


def test_overridden_registry_entry_is_refused():
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    backend = AttentionBackendEnum.FLASHMLA
    register_backend(backend, class_path=backend.value)
    try:
        with pytest.raises(ValueError, match="registry entry is overridden"):
            engine(V3, "FLASHMLA")
    finally:
        backend.clear_override()


def test_token_subspan_keeps_every_heads_own_logical_stream():
    from vllm.v1.worker.simulated.kv_pattern import _attention_slice, row_pattern

    heads = (2, 3, 4, 5)
    span = row_pattern("request", "layer", 7, 256, heads=heads).reshape(4, 64)
    for sub in range(4):
        actual = _attention_slice(
            "request", "layer", 7, 256, 0, None, heads, sub, 64, 4
        )
        assert torch.equal(actual.reshape(4, 16), span[:, sub * 16 : (sub + 1) * 16])


@pytest.mark.parametrize("ratios", [(2, 1), (2, 2), (1, 2)])
def test_dcp_peer_compares_the_logical_page_product(ratios):
    from tests.v1.worker.simulated.test_real_kv_transfer import agent_metadata
    from vllm.v1.worker.simulated.kv_transfer import dcp_peer_refusal

    config = engine(
        V3, "FLASHMLA", tensor_parallel_size=2, decode_context_parallel_size=2
    )
    local, remote = ratios
    peer = agent_metadata(
        block_size=128 // remote, physical_blocks_per_logical_kv_block=remote
    )
    assert dcp_peer_refusal(config, 2, (128 // local, local), 1, peer) is None
    unequal = agent_metadata(
        block_size=256 // remote, physical_blocks_per_logical_kv_block=remote
    )
    with_same_kernel = agent_metadata(
        block_size=128 // local, physical_blocks_per_logical_kv_block=local * 2
    )
    for bad in [unequal, with_same_kernel]:
        assert "DCP requires matching P/D logical block sizes" in dcp_peer_refusal(
            config, 2, (128 // local, local), 1, bad
        )


def test_geometry_does_not_change_fixed_work_aisimulate():
    from vllm.v1.worker.simulated.aisimulate_timing import AisimulateTimingProvider
    from vllm.v1.worker.simulated.timing import SimRequestStep, StepDescriptor

    pytest.importorskip("aisimulate_core")
    providers = [
        AisimulateTimingProvider(engine(V3, name), gpu="h100", target_model=V3)
        for name in (None, "FLASHMLA")
    ]
    one = SimRequestStep(
        req_id="r",
        num_prompt_tokens=128,
        num_computed_tokens=0,
        num_scheduled_tokens=128,
        num_cached_tokens=0,
        num_spec_tokens=0,
        is_prefill=True,
    )
    step = StepDescriptor(1, (one,), 128, 0, 1, 0, 0, 128)
    assert providers[0].config == providers[1].config
    assert providers[0].estimate(step) == providers[1].estimate(step)


@pytest.mark.parametrize(
    ("actual", "message"),
    [
        ([(0, 32)], "missing bytes"),
        ([(0, 32), (0, 32), (32, 32)], "duplicated or surplus bytes"),
        ([(0, 32), (32, 32)], None),
    ],
)
def test_prepared_kernel_spans_must_cover_the_page_once(actual, message):
    from vllm.v1.worker.simulated.state_layout import ReadRecord, coverage_problems

    logical_page = {1: [(0, 4, 0, 64)]}
    spans = [(0, 4, offset, size) for offset, size in actual]
    problems = coverage_problems(
        logical_page, [ReadRecord("r", 1, 12, spans, done=True)]
    )
    if message is None:
        assert problems == []
    else:
        assert len(problems) == 1 and message in problems[0]
