# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA MTP under DCP at the KV interleaves vLLM allows for a 64-token
page: the admitted cells against direct AISimulate calls, the retained
refusals, and the token placement the interleave decides."""

import dataclasses
import functools

import pytest

from tests.v1.worker.simulated.test_aisimulate_timing import (
    GLM5,
    MTP_DCP_DEPTHS,
    MTP_DCP_LAYOUTS,
    MTP_DCP_PROFILES,
    SPECULATIVE_DCP_TRANSFERS,
    V3,
    V32,
    _mtp_dcp_direct_sdk,
    _mtp_dcp_engine_config,
    provider,
    requires_mtp_dcp,
)
from tests.v1.worker.simulated.utils import sim_req, sim_step

SPARSE_PAGE = 64
SPARSE_INTERLEAVES = (1, 2, 4, 8, 16, 32, 64)
LARGER_INTERLEAVES = SPARSE_INTERLEAVES[1:]
SPARSE_LAYOUTS = [
    (target, tp, dcp) for target in (V32, GLM5) for tp, dcp in MTP_DCP_LAYOUTS[target]
]
# The estimator's profile of each sparse target, written out: system, weight
# mode and KV mode, as the published checkpoints and the packed cache need.
EXPECTED_PROFILE = {
    V32: dict(system="h200_sxm", gemm_quant_mode="fp8_block", kv="fp8"),
    GLM5: dict(system="b200_sxm", gemm_quant_mode="nvfp4", kv="fp8"),
}
INTERLEAVE_CELLS = [
    pytest.param(
        target,
        tp,
        dcp,
        k,
        interleave,
        id=f"{target.split('/')[1]}-tp{tp}-dcp{dcp}-k{k}-i{interleave}",
    )
    for target, tp, dcp in SPARSE_LAYOUTS
    for k in MTP_DCP_DEPTHS
    for interleave in LARGER_INTERLEAVES
]


@functools.cache
def dcp_placement(length: int, dcp: int, interleave: int, page: int):
    """Every KV token's owner and local offset from the interleave's definition
    (`cp_kv_cache_interleave_size` tokens a rank, round robin), checked
    against vLLM's closed-form local lengths and the scheduler block span.
    Returns each rank's local length and local page count. Calculated, not
    executed slot mapping."""
    import torch

    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

    owned: list[list[int]] = [[] for _ in range(dcp)]
    for token in range(length):
        rank = (token // interleave) % dcp
        local = (token // (interleave * dcp)) * interleave + token % interleave
        # A rank's tokens fill its local offsets in order, without gaps.
        assert local == len(owned[rank])
        owned[rank].append(token)
        # A scheduler block of page * DCP tokens is one local page on each rank.
        assert local // page == token // (page * dcp)
    lengths = [len(tokens) for tokens in owned]
    closed_form = get_dcp_local_seq_lens(
        torch.tensor([length], dtype=torch.int32), dcp, None, interleave
    )[0].tolist()
    assert lengths == closed_form
    assert sum(lengths) == length
    pages = [-(-n // page) for n in lengths]
    # The fullest rank holds exactly the scheduler blocks the length spans.
    assert max(pages) == -(-length // (page * dcp))
    return lengths, pages


def assert_step_placement(requests, k, dcp, interleave, page, max_len=2048):
    """The placement of each scheduled request's KV after the step, with the
    speculative tail (`k` lookahead slots) the scheduler reserves."""
    for request in requests:
        end = request["num_computed_tokens"] + request["num_scheduled_tokens"]
        for length in (end, min(max_len, end + k)):
            dcp_placement(length, dcp, interleave, page)


def boundary_lengths(dcp, interleave, page, k):
    around = {
        1,
        interleave,
        dcp * interleave,
        page,
        page * dcp,
        2 * page * dcp,
        page * dcp + interleave,
        page * dcp - 1,
    }
    return sorted(
        n + d + t for n in around for d in (-1, 0, 1) for t in (0, k) if n + d + t > 0
    )


@pytest.mark.parametrize("target,tp,dcp", SPARSE_LAYOUTS)
@pytest.mark.parametrize("interleave", SPARSE_INTERLEAVES)
def test_placement_matches_independent_token_ownership(target, tp, dcp, interleave):
    """Boundaries around the interleave, DCP times it, the page and the
    scheduler block, partial pages and speculative tails, for each rank."""
    for k in (1, 5):
        for length in boundary_lengths(dcp, interleave, SPARSE_PAGE, k):
            lengths, pages = dcp_placement(length, dcp, interleave, SPARSE_PAGE)
            assert len(lengths) == len(pages) == dcp
    # Named cases: the interleave's first cycle and the first block boundary.
    lengths, _ = dcp_placement(dcp * interleave + 1, dcp, interleave, SPARSE_PAGE)
    assert lengths[0] == interleave + 1
    assert lengths[1:] == [interleave] * (dcp - 1)
    lengths, pages = dcp_placement(SPARSE_PAGE * dcp + 1, dcp, interleave, SPARSE_PAGE)
    assert lengths[0] == SPARSE_PAGE + 1 and pages[0] == 2
    assert lengths[1:] == [SPARSE_PAGE] * (dcp - 1)
    assert pages[1:] == [1] * (dcp - 1)


@requires_mtp_dcp
@pytest.mark.parametrize("target,tp,dcp,k,interleave", INTERLEAVE_CELLS)
def test_admits_sparse_mtp_dcp_interleaves_and_matches_the_sdk(
    tmp_path, monkeypatch, target, tp, dcp, k, interleave
):
    """Config, scope and timing of every admitted (layout, depth, interleave):
    the provider's profile equals the written-out one and its estimates equal
    direct SDK calls, the same as at interleave 1."""
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import dcp_scope_problems
    from vllm.v1.worker.simulated.aisimulate_timing import (
        MTP_DCP_INTERLEAVE_APPROXIMATION,
    )

    monkeypatch.setenv("VLLM_DCP_Q_REPLICATE", "0")
    options = dict(
        target=target,
        tp=tp,
        dcp=dcp,
        spec={"num_speculative_tokens": k},
        cp_kv_cache_interleave_size=interleave,
    )
    config = _mtp_dcp_engine_config(tmp_path, **options)
    assert config.parallel_config.cp_kv_cache_interleave_size == interleave
    assert dcp_scope_problems(config, transfers=False) == []
    # The retained transfer refusal is the same at every interleave.
    assert dcp_scope_problems(config, transfers=True)[-1] == SPECULATIVE_DCP_TRANSFERS
    cached = _mtp_dcp_engine_config(
        tmp_path / "cached", **options, enable_prefix_caching=True
    )
    assert dcp_scope_problems(cached, transfers=False) == []

    profile = EXPECTED_PROFILE[target]
    timed = provider(config, gpu=MTP_DCP_PROFILES[target][0], target_model=target)
    record = timed.assumptions()["aisimulate"]
    resolved = record["resolved_config"]
    expected = {
        "system": profile["system"],
        "backend": "vllm",
        "backend_version": "0.24.0",
        "worker_type": "aggregated",
        "tp": tp,
        "moe_tp_size": tp,
        "moe_ep_size": 1,
        "dcp": dcp,
        "nextn": k,
        "gemm_quant_mode": profile["gemm_quant_mode"],
        "moe_quant_mode": profile["gemm_quant_mode"],
        "kvcache_quant_mode": profile["kv"],
        "estimation_mode": "op_level",
    }
    assert {name: resolved[name] for name in expected} == expected
    assert record["decode_context_parallel"] == {
        "size": dcp,
        "comm_backend": "ag_rs",
        "q_replicate": False,
        "kv_cache_format": "fp8_ds_mla",
        "interleave": interleave,
        "sdk_backend_version": "0.24.0",
        "attention_backend": "FLASHMLA_SPARSE",
        "indexer_kv_cache_dtype": "fp8",
    }
    assert MTP_DCP_INTERLEAVE_APPROXIMATION in record["approximations"]
    assert record["speculation"]["verify_width"] == k + 1

    sdk = _mtp_dcp_direct_sdk(target, tp, dcp, k)
    width = k + 1
    decodes = [
        dataclasses.replace(sim_req("d1", width, 1025, 1025), num_spec_tokens=k),
        dataclasses.replace(sim_req("d2", width, 1041, 1041), num_spec_tokens=k),
    ]
    metrics = {
        "num_prefill_requests": 0,
        "sum_prefill_tokens": 0,
        "var_prefill_length": 0,
        "sum_prefill_kv_tokens": 0,
        "num_decode_requests": 2,
        "sum_decode_kv_tokens": 2066,
        "var_decode_kv_tokens": 64,
        "extend_lengths": [width, width],
        "past_kv_lengths": [1025, 1041],
    }
    mixed = metrics | {
        "num_prefill_requests": 1,
        "sum_prefill_tokens": 256,
        "sum_prefill_kv_tokens": 513,
        "extend_lengths": [256, width, width],
        "past_kv_lengths": [513, 1025, 1041],
    }
    for step, shape in (
        (sim_step(0, *decodes), metrics),
        (sim_step(1, sim_req("p", 256, 513, 1024), *decodes), mixed),
    ):
        ms = sdk.estimate_forward_pass_time_ms({"scheduled_requests": shape})
        assert ms > 0
        cost = timed.estimate(step)
        assert cost.device_s * 1000 == pytest.approx(ms, rel=1e-9)
        assert cost.breakdown["spec_draft_tokens"] == 2 * k


@requires_mtp_dcp
@pytest.mark.parametrize("target,tp,dcp", SPARSE_LAYOUTS)
def test_sparse_mtp_dcp_estimates_do_not_depend_on_the_interleave(
    tmp_path, monkeypatch, target, tp, dcp
):
    """AISimulate takes no interleave: one step's estimate is the same number
    at every interleave, and the interleave-1 config records no note."""
    from vllm.v1.worker.simulated.aisimulate_timing import (
        MTP_DCP_INTERLEAVE_APPROXIMATION,
    )

    monkeypatch.setenv("VLLM_DCP_Q_REPLICATE", "0")
    k = 3
    decodes = [
        dataclasses.replace(sim_req("d1", k + 1, 1500, 1500), num_spec_tokens=k),
        dataclasses.replace(sim_req("d2", k + 1, 777, 777), num_spec_tokens=k),
    ]
    estimates = {}
    for interleave in SPARSE_INTERLEAVES:
        config = _mtp_dcp_engine_config(
            tmp_path / f"i{interleave}",
            target=target,
            tp=tp,
            dcp=dcp,
            spec={"num_speculative_tokens": k},
            cp_kv_cache_interleave_size=interleave,
        )
        timed = provider(config, gpu=MTP_DCP_PROFILES[target][0], target_model=target)
        noted = (
            MTP_DCP_INTERLEAVE_APPROXIMATION
            in timed.assumptions()["aisimulate"]["approximations"]
        )
        assert noted == (interleave > 1)
        estimates[interleave] = timed.estimate(sim_step(0, *decodes)).device_s
    assert len(set(estimates.values())) == 1


@requires_mtp_dcp
@pytest.mark.parametrize("target", [V32, GLM5])
@pytest.mark.parametrize("block_size", [None, 64])
def test_sparse_mtp_dcp_interleave_with_default_and_explicit_pages(
    tmp_path, target, block_size
):
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import dcp_scope_problems

    tp, dcp = {V32: (8, 2), GLM5: (4, 2)}[target]
    config = _mtp_dcp_engine_config(
        tmp_path,
        target=target,
        tp=tp,
        dcp=dcp,
        block_size=block_size,
        cp_kv_cache_interleave_size=64,
    )
    assert config.cache_config.user_specified_block_size == (block_size is not None)
    assert dcp_scope_problems(config, transfers=False) == []


@requires_mtp_dcp
@pytest.mark.parametrize("target,tp,dcp", SPARSE_LAYOUTS)
def test_sparse_mtp_dcp_default_page_is_checked_by_the_provider_at_an_interleave(
    tmp_path, monkeypatch, target, tp, dcp
):
    """A page size left to the backend resolves to the default 64 and passes at
    any admitted interleave; a provider built on another size still refuses."""
    monkeypatch.setenv("VLLM_DCP_Q_REPLICATE", "0")
    config = _mtp_dcp_engine_config(
        tmp_path,
        target=target,
        tp=tp,
        dcp=dcp,
        block_size=None,
        cp_kv_cache_interleave_size=16,
    )
    assert not config.cache_config.user_specified_block_size
    assert config.cache_config.block_size == 64
    gpu = MTP_DCP_PROFILES[target][0]
    provider(config, gpu=gpu, target_model=target)
    config.cache_config.block_size = 16
    with pytest.raises(ValueError, match="requires 64-token pages for"):
        provider(config, gpu=gpu, target_model=target)


@requires_mtp_dcp
@pytest.mark.parametrize(
    "target,options,reason",
    [
        (V32, {"cp_kv_cache_interleave_size": 3}, "KV interleave dividing the 64"),
        (V32, {"cp_kv_cache_interleave_size": 6}, "KV interleave dividing the 64"),
        (V32, {"cp_kv_cache_interleave_size": 128}, "KV interleave dividing the 64"),
        (V32, {"cp_kv_cache_interleave_size": 0}, "KV interleave dividing the 64"),
        (V32, {"cp_kv_cache_interleave_size": -2}, "KV interleave dividing the 64"),
        (GLM5, {"cp_kv_cache_interleave_size": 3}, "KV interleave dividing the 64"),
        (GLM5, {"cp_kv_cache_interleave_size": 96}, "KV interleave dividing the 64"),
        (GLM5, {"cp_kv_cache_interleave_size": 0}, "KV interleave dividing the 64"),
        # The deprecated alias is inert without a KV transfer: only 1 or equal.
        (V32, {"dcp_kv_cache_interleave_size": 16}, "dcp_kv_cache_interleave_size"),
        (
            GLM5,
            {"cp_kv_cache_interleave_size": 16, "dcp_kv_cache_interleave_size": 32},
            "dcp_kv_cache_interleave_size",
        ),
        (
            V32,
            {"cp_kv_cache_interleave_size": 2, "dcp_kv_cache_interleave_size": 4},
            "dcp_kv_cache_interleave_size",
        ),
        # The page, whatever the interleave.
        (
            V32,
            {"block_size": 32, "cp_kv_cache_interleave_size": 16},
            "requires 64-token pages for DeepSeek-V3.2",
        ),
        (
            GLM5,
            {"block_size": 128, "cp_kv_cache_interleave_size": 64},
            "requires 64-token pages for GLM-5",
        ),
        # Dense V3 stays at interleave 1: its CPU MLA backend declares no MTP
        # support above it and vLLM refuses it.
        (V3, {"cp_kv_cache_interleave_size": 2}, "requires interleave 1 for"),
        (V3, {"cp_kv_cache_interleave_size": 16}, "requires interleave 1 for"),
        (V3, {"cp_kv_cache_interleave_size": 64}, "requires interleave 1 for"),
        (V3, {"dcp_kv_cache_interleave_size": 2}, "requires interleave 1 for"),
        # The retained scope, at an admitted interleave.
        (
            V32,
            {"tp": 8, "dcp": 8, "cp_kv_cache_interleave_size": 16},
            "requires TP/DCP 8/2, 8/4 for DeepSeek-V3.2",
        ),
        (
            GLM5,
            {"tp": 8, "dcp": 2, "cp_kv_cache_interleave_size": 16},
            "requires TP/DCP 4/2, 4/4 for GLM-5",
        ),
        (
            V32,
            {"kv_offloading_size": 1, "cp_kv_cache_interleave_size": 16},
            "KV offloading with speculative",
        ),
    ],
)
def test_sparse_mtp_dcp_keeps_other_interleaves_and_scope_refused(
    tmp_path, target, options, reason
):
    tp, dcp = {V32: (8, 2), GLM5: (4, 2)}.get(target, (2, 2))
    options = {"tp": tp, "dcp": dcp} | options
    with pytest.raises(ValueError, match=reason):
        _mtp_dcp_engine_config(tmp_path, target=target, **options)


@requires_mtp_dcp
@pytest.mark.parametrize("target,tp,dcp", SPARSE_LAYOUTS)
@pytest.mark.parametrize("interleave", LARGER_INTERLEAVES)
def test_admits_sparse_mtp_prefix_dcp_interleaves(
    tmp_path, target, tp, dcp, interleave
):
    """Relaxing the interleave also admits prefix caching, through the same
    helper: the default one-token hash proof, and a lookahead of k slots."""
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import dcp_scope_problems

    config = _mtp_dcp_engine_config(
        tmp_path,
        target=target,
        tp=tp,
        dcp=dcp,
        enable_prefix_caching=True,
        spec={"num_speculative_tokens": 2},
        cp_kv_cache_interleave_size=interleave,
    )
    assert dcp_scope_problems(config, transfers=False) == []
    assert config.num_lookahead_tokens == 2
    assert config.num_prefill_lookahead_tokens == config.num_hash_lookahead_tokens == 1


@requires_mtp_dcp
@pytest.mark.parametrize("target,tp,dcp", SPARSE_LAYOUTS)
@pytest.mark.parametrize("interleave", [2, 64])
@pytest.mark.parametrize(
    "policy", ["disable_lookahead_block_hashes", "disable_eagle_block_drop"]
)
def test_sparse_mtp_prefix_dcp_interleave_keeps_hash_policies_refused(
    tmp_path, target, tp, dcp, interleave, policy
):
    with pytest.raises(ValueError, match="prefix caching with speculative decoding"):
        _mtp_dcp_engine_config(
            tmp_path,
            target=target,
            tp=tp,
            dcp=dcp,
            enable_prefix_caching=True,
            spec={"num_speculative_tokens": 2, policy: True},
            cp_kv_cache_interleave_size=interleave,
        )
