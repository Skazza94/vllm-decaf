# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank KV transfer settings shared by the native connectors under
simulated execution with real KV transfers (MooncakeConnector, and the
NixlConnector over NIXL's Mooncake backend): each rank's own KV address
(``rank_addresses``), TENT's hp_tcp bound to it, and the tensor-parallel
sizes real transfers pair."""

import ipaddress
import os
from dataclasses import dataclass

from vllm import envs
from vllm.config import KVTransferConfig, VllmConfig
from vllm.config.simulation import NIXL_CONNECTORS, SimulationConfig, moves_real_kv
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import tent_config
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


def host_dp_ranks(parallel_config) -> tuple[int, int]:
    """This DP engine's index among, and the count of, its model's DP engines
    on this host. Engines launched together by one server say so in their
    configuration (their local rank and local size). A multi-port external-LB
    supervisor launches a server for each of the host's ranks instead, each
    configured with one local rank; it exports their place on the host
    (VLLM_DP_SUPERVISED_RANK_LOCAL, VLLM_DP_SUPERVISED_SIZE_LOCAL)."""
    rank, size = envs.VLLM_DP_SUPERVISED_RANK_LOCAL, envs.VLLM_DP_SUPERVISED_SIZE_LOCAL
    if rank is None and size is None:
        return (
            parallel_config.data_parallel_rank_local or 0,
            parallel_config.data_parallel_size_local,
        )
    if (
        rank is None
        or size is None
        or not parallel_config.data_parallel_external_lb
        or parallel_config.data_parallel_size_local != 1
        or not 0 <= rank < size <= parallel_config.data_parallel_size
    ):
        raise ValueError(
            "VLLM_DP_SUPERVISED_RANK_LOCAL and VLLM_DP_SUPERVISED_SIZE_LOCAL are "
            "set together by the multi-port external-LB supervisor, for a server "
            "of one local DP rank in external LB mode: got rank "
            f"{rank} of {size} (data_parallel_size "
            f"{parallel_config.data_parallel_size}, data_parallel_size_local "
            f"{parallel_config.data_parallel_size_local}, external LB "
            f"{parallel_config.data_parallel_external_lb})."
        )
    return rank, size


def local_kv_rank(vllm_config: VllmConfig, tp_rank: int, tp_size: int) -> tuple[int, int]:
    """This rank's place in ``rank_addresses``, and their count. An engine
    of its own (no DP, or a dense model's independent DP engine, which runs
    as DP=1) lists its TP ranks; an MoE model's DP engines on one host list
    theirs in turn, host-local DP rank by TP rank (`host_dp_ranks`)."""
    parallel_config = vllm_config.parallel_config
    if parallel_config.data_parallel_size == 1:
        return tp_rank, tp_size
    dp_local_rank, dp_local_size = host_dp_ranks(parallel_config)
    if not 0 <= dp_local_rank < dp_local_size:
        # An engine launched on its own (offline MoE DP) keeps its host-local
        # rank among engines its configuration does not count.
        raise ValueError(
            f"rank_addresses needs this host's MoE DP engines launched "
            f"together: local DP rank {dp_local_rank} is not one of its "
            f"{dp_local_size} local engines."
        )
    return dp_local_rank * tp_size + tp_rank, dp_local_size * tp_size


def this_rank_kv_address(vllm_config: VllmConfig) -> str | None:
    """`rank_kv_address` of this worker's rank, None without
    ``rank_addresses``."""
    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    if kv_transfer_config.kv_connector_extra_config.get("rank_addresses") is None:
        return None
    return rank_kv_address(
        kv_transfer_config,
        *local_kv_rank(
            vllm_config,
            get_tensor_model_parallel_rank(),
            get_tensor_model_parallel_world_size(),
        ),
    )


def rank_kv_address(
    kv_transfer_config: KVTransferConfig, rank: int, ranks: int
) -> str | None:
    """This rank's own address, from ``rank_addresses`` in the connector's
    extra config (one IPv4 address per rank of the instance on this host,
    as `local_kv_rank` orders them, the same for every PP stage): its
    handshakes and TENT's hp_tcp use it, so each rank can have a NIC of
    its own. None without it: the host's."""
    addresses = kv_transfer_config.kv_connector_extra_config.get("rank_addresses")
    if addresses is None:
        return None
    if not isinstance(addresses, list) or len(addresses) != ranks:
        raise ValueError(
            "rank_addresses needs one address per rank on this host (local "
            f"DP ranks x TP ranks: {ranks}), got {addresses!r}."
        )
    for address in addresses:
        # IPv4Address() also takes integers.
        try:
            if not isinstance(address, str):
                raise ValueError
            ipaddress.IPv4Address(address)
        except ValueError:
            raise ValueError(
                f"rank_addresses: {address!r} is not an IPv4 address."
            ) from None
    return addresses[rank]


def bind_hp_tcp(address: str) -> None:
    """Make TENT in this process use `address` only: its RPC server (control
    metadata) and hp_tcp's listening, advertised and sending address (its
    MC_TENT_CONF; other settings kept). TENT does not derive its RPC host
    from initialize()'s hostname, and MOONCAKE_LOCAL_HOSTNAME overrides the
    config's."""
    conf = tent_config.load() or {}
    if "MOONCAKE_LOCAL_HOSTNAME" in os.environ:
        os.environ["MOONCAKE_LOCAL_HOSTNAME"] = address
    for key, value in (
        ("rpc_server_hostname", address),
        ("transports/hp_tcp/bind_address", address),
        ("transports/hp_tcp/advertise_address", address),
        ("transports/hp_tcp/rail_addresses", [address]),
    ):
        tent_config.store(conf, key, value)
    tent_config.save(conf)


def plain_mla(model) -> bool:
    """MLA with one latent cache a layer, as DeepSeek-V3 and Kimi-K2 keep it:
    not DeepSeek-V3.2's or GLM-5's sparse attention (an indexer's cache
    beside the latent, which DeepSeek's code selects by the config having
    `index_topk`, set or not), not DeepSeek-V4's compressed caches, not a
    hybrid's (Kimi-K3's) recurrent states."""
    configs = (model.hf_config, model.hf_text_config)
    return (
        model.use_mla
        and not model.is_hybrid
        and not any(hasattr(config, "index_topk") for config in configs)
        and not any(
            getattr(config, "model_type", None) == "deepseek_v4" for config in configs
        )
    )


def sparse_mla(model) -> bool:
    """DeepSeek-V3.2's or GLM-5's sparse MLA (DSA): a latent and, on the
    layers that have one, an indexer's key cache, each one cache held whole
    by every rank, as plain MLA's latent is."""
    from vllm.config.simulation import DSA_MODEL_TYPES

    return (
        model.use_mla
        and not model.is_hybrid
        and getattr(model.hf_config, "model_type", None) in DSA_MODEL_TYPES
        and getattr(model.hf_text_config, "index_topk", None) is not None
    )


def gqa_dcp_model(model) -> bool:
    """The full-attention GQA model qualified for standalone CPU simulation."""
    return (
        model.architectures == ["Qwen3MoeForCausalLM"]
        and not model.use_mla
        and not model.is_hybrid
        and model.get_sliding_window() is None
    )


# Speculative decoding where KV moves under DCP: NIXL needs the DCP
# interleave at the block size (vLLM resolves it so when the engine starts,
# `VllmConfig.adjust_dcp_kv_cache_interleave_size`), and vLLM's check
# (`check_attention_cp_compatibility`) refuses MTP at an interleave above 1
# on a backend without `supports_mtp_with_cp_non_trivial_interleave_size`,
# as the simulated CPU MLA backend is.
SPECULATIVE_DCP_TRANSFERS = (
    "speculative decoding (over NIXL with DCP the interleave is the block "
    "size, and the simulated CPU MLA backend does not run MTP at an "
    "interleave above 1)"
)


# Standalone MTP under DCP, by model type: the cells AISimulate times and the
# engine validated (TP/DCP layouts, the attention backend's page size). Sparse
# MLA's FlashMLA pages and indexer hold 64 tokens, DeepSeek-V3's MLA 16.
MTP_DCP_MODELS = {
    "deepseek_v3": {
        "name": "DeepSeek-V3",
        "architecture": "DeepseekV3ForCausalLM",
        "sparse": False,
        "block_size": 16,
        "layouts": ((2, 2), (4, 2), (4, 4), (8, 2), (8, 4), (8, 8)),
    },
    "deepseek_v32": {
        "name": "DeepSeek-V3.2",
        "architecture": "DeepseekV32ForCausalLM",
        "sparse": True,
        "block_size": 64,
        "layouts": ((8, 2), (8, 4)),
    },
    "glm_moe_dsa": {
        "name": "GLM-5",
        "architecture": "GlmMoeDsaForCausalLM",
        "sparse": True,
        "block_size": 64,
        "layouts": ((4, 2), (4, 4)),
    },
}
MTP_DCP_MAX_DEPTH = 3
MTP_DCP_MODEL_PROBLEM = (
    "speculative decoding with DCP requires DeepSeek-V3, DeepSeek-V3.2 or GLM-5"
)


def mtp_dcp_page_problem(vllm_config: VllmConfig) -> str | None:
    """The page size the engine resolved against the qualified model's."""
    qualified = MTP_DCP_MODELS[vllm_config.model_config.hf_config.model_type]
    block_size = vllm_config.cache_config.block_size
    if block_size != qualified["block_size"]:
        return (
            f"MTP with DCP requires {qualified['block_size']}-token pages for "
            f"{qualified['name']}, the engine resolved {block_size}"
        )
    return None


def standalone_mtp_dcp_problems(vllm_config: VllmConfig) -> list[str]:
    """Qualify the local model and layout before building an AISimulate profile."""
    from vllm.config.simulation import DSA_FP8_CACHE_DTYPES, SimulationConfig

    sim = vllm_config.simulation_config
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    model = vllm_config.model_config
    spec = vllm_config.speculative_config
    assert spec is not None
    problems = []
    if vllm_config.kv_transfer_config is not None:
        problems.append("KV transfers with speculative decoding and DCP")
    if cache.kv_offloading_size is not None:
        problems.append("KV offloading with speculative decoding and DCP")
    if cache.enable_prefix_caching:
        problems.append("prefix caching with speculative decoding")
    if sim.timing != "aisimulate":
        problems.append("speculative decoding with DCP requires timing aisimulate")
    if parallel.nnodes != 1:
        problems.append("MTP with DCP requires one node")
    qualified = MTP_DCP_MODELS.get(getattr(model.hf_config, "model_type", None))
    if (
        qualified is None
        or model.architectures != [qualified["architecture"]]
        or not (sparse_mla(model) if qualified["sparse"] else plain_mla(model))
    ):
        return problems + [MTP_DCP_MODEL_PROBLEM]
    name = qualified["name"]
    layouts = qualified["layouts"]
    if (
        parallel.tensor_parallel_size,
        parallel.decode_context_parallel_size,
    ) not in layouts:
        cells = ", ".join(f"{tp}/{dcp}" for tp, dcp in layouts)
        problems.append(f"MTP with DCP requires TP/DCP {cells} for {name}")
    block_size = qualified["block_size"]
    # Sparse MLA's backend sets its page size when the workers start, so a
    # page size left to it is checked there (`mtp_dcp_page_problem`).
    page_ok = cache.block_size == block_size or (
        qualified["sparse"] and not cache.user_specified_block_size
    )
    if (
        parallel.cp_kv_cache_interleave_size != 1
        or parallel.dcp_kv_cache_interleave_size != 1
        or not page_ok
    ):
        problems.append(
            f"MTP with DCP requires interleave 1 and {block_size}-token pages "
            f"for {name}"
        )
    if spec.method != "mtp":
        return problems + [f"speculative method {spec.method} (requires mtp)"]
    if not 1 <= spec.num_speculative_tokens <= MTP_DCP_MAX_DEPTH:
        problems.append(
            f"{spec.num_speculative_tokens} speculative tokens with DCP "
            f"(requires 1 to {MTP_DCP_MAX_DEPTH})"
        )
    if spec.uses_dynamic_speculative_decoding() or spec.enable_adaptive_verification:
        problems.append("dynamic or adaptive speculative decoding with DCP")
    if spec.model != model.model:
        problems.append(
            f"draft checkpoint {spec.model} (requires the target checkpoint)"
        )
    draft = spec.draft_model_config
    draft_text = getattr(draft, "hf_text_config", None)
    if (
        getattr(model.hf_text_config, "num_nextn_predict_layers", None) != 1
        or getattr(draft_text, "num_nextn_predict_layers", None) != 1
    ):
        problems.append(
            "DCP requires the target and draft to have one native MTP layer"
        )

    problems.extend(
        SimulationConfig._native_mtp_mismatches(vllm_config, effective_kv=True)
    )
    draft_tp = getattr(spec.draft_parallel_config, "tensor_parallel_size", None)
    if (
        getattr(getattr(draft, "hf_config", None), "model_type", None) != "deepseek_mtp"
        or draft_tp != parallel.tensor_parallel_size
    ):
        problems.append("MTP with DCP requires DeepSeekMTPModel at the target's TP")
    draft_dtype = getattr(draft, "dtype", None)
    if qualified["sparse"]:
        # The packed FP8 latent and indexer caches; the weights' precision is
        # the published checkpoint's, which the timing provider reads.
        kv_ok = cache.cache_dtype in DSA_FP8_CACHE_DTYPES and spec.kv_cache_dtype in (
            None,
            "auto",
            cache.cache_dtype,
        )
        weights_ok = True
        precision = "BF16 activations and the packed FP8 KV cache"
    else:
        kv_ok = cache.cache_dtype in ("auto", "bfloat16") and spec.kv_cache_dtype in (
            None,
            "auto",
            "bfloat16",
        )
        weights_ok = (
            model.quantization == "fp8"
            and getattr(draft, "quantization", None) == "fp8"
            and model.quantization_config is None
            and getattr(draft, "quantization_config", None) is None
        )
        precision = "published FP8 weights and BF16 model and KV dtype"
    if (
        not kv_ok
        or not weights_ok
        or str(model.dtype) != "torch.bfloat16"
        or str(draft_dtype) != "torch.bfloat16"
    ):
        problems.append(f"MTP with DCP requires {precision}")
    if (
        spec.attention_backend is not None
        or spec.moe_backend not in (None, "auto")
        or spec.parallel_drafting
        or spec.revision not in (None, model.revision)
        or spec.code_revision not in (None, model.code_revision)
    ):
        problems.append("draft kernel, parallel drafting or revision override with DCP")
    return problems


def dcp_scope_problems(vllm_config: VllmConfig, *, transfers: bool) -> list[str]:
    """What of an instance's configuration falls outside the scope of decode
    context parallelism under simulated compute, either endpoint's when its
    KV moves (`transfers`): dense MLA, or sparse MLA or GQA with AISimulate in
    one instance. All use tensor parallelism alone (no pipeline, prefill
    context or data parallelism, no expert parallelism). Speculation is
    limited to qualified standalone MTP (`MTP_DCP_MODELS`). Prefix caching is
    admitted with transfers too (its byte checks need the same setting on
    both endpoints: their pages' patterns are content-seeded exactly when
    it is on)."""
    parallel = vllm_config.parallel_config
    problems = []
    model = vllm_config.model_config
    if not plain_mla(model):
        if not transfers and sparse_mla(model):
            if vllm_config.simulation_config.timing != "aisimulate":
                problems.append(
                    "a model other than dense MLA without AISimulate timing "
                    "(sparse MLA DCP requires timing aisimulate)"
                )
        elif not transfers and gqa_dcp_model(model):
            if vllm_config.simulation_config.timing != "aisimulate":
                problems.append("GQA DCP without timing aisimulate")
        elif transfers:
            problems.append(
                "a model other than dense MLA (sparse or compressed MLA, a "
                "hybrid, or attention without MLA)"
            )
        else:
            problems.append(
                "a model other than dense MLA or DeepSeek-V3.2-style sparse MLA"
            )
    for name, size in (
        ("pipeline parallelism", parallel.pipeline_parallel_size),
        ("prefill context parallelism", parallel.prefill_context_parallel_size),
        ("data parallelism", parallel.data_parallel_size),
    ):
        if size > 1:
            problems.append(f"{name} {size}")
    if parallel.enable_expert_parallel:
        problems.append("expert parallelism")
    if vllm_config.speculative_config is not None:
        if transfers or vllm_config.kv_transfer_config is not None:
            problems.append(SPECULATIVE_DCP_TRANSFERS)
        else:
            problems.extend(standalone_mtp_dcp_problems(vllm_config))
    return problems


def dcp_pair_refusal(local_dcp_size: int, remote_dcp_size: int, problems) -> str:
    return (
        "Simulated real KV transfers pair decode context parallelism (local DCP "
        f"{local_dcp_size}, remote DCP {remote_dcp_size}) for a dense MLA model "
        "over tensor parallelism alone on both sides, TP sizes one dividing the "
        "other, without speculative decoding, not with "
        f"{', '.join(problems)}."
    )


UNEQUAL_TP_MLA_MTP = "simulated/nixl/unequal-tp-mla-mtp/v1"
UNEQUAL_TP_MLA_MTP_PAGE = 32
UNEQUAL_TP_MLA_MTP_PAGE_BYTES = UNEQUAL_TP_MLA_MTP_PAGE * (512 + 64) * 2


def _qualifies_unequal_tp_mtp_endpoint(config: VllmConfig, page_size: int) -> bool:
    """Common constraints for a simulated NIXL pull endpoint."""
    transfer = config.kv_transfer_config
    spec = config.speculative_config
    sim, cache, pc = (
        config.simulation_config,
        config.cache_config,
        config.parallel_config,
    )
    if (
        not moves_real_kv(config)
        or transfer.kv_connector not in NIXL_CONNECTORS
        or transfer.kv_role not in ("kv_producer", "kv_consumer")
        or transfer.get_from_extra_config("bidirectional_kv_xfer", False)
        or not transfer.get_from_extra_config("enforce_handshake_compat", True)
        or spec is None
        or spec.num_speculative_tokens != 1
        or spec.uses_dynamic_speculative_decoding()
        or spec.enable_adaptive_verification
        or sim.timing != "aisimulate"
        or sim.timing_args.get("quant_modes")
        or sim.kv_kernel_backend is not None
        or cache.block_size != page_size
        or cache.enable_prefix_caching
        or pc.tensor_parallel_size not in (2, 4)
        or any(
            size != 1
            for size in (
                pc.data_parallel_size,
                pc.pipeline_parallel_size,
                pc.prefill_context_parallel_size,
                pc.decode_context_parallel_size,
                pc.nnodes,
            )
        )
        or pc.enable_expert_parallel
    ):
        return False

    model = config.model_config
    return (
        spec.draft_parallel_config is not None
        and spec.draft_parallel_config.tensor_parallel_size == pc.tensor_parallel_size
        and spec.attention_backend is None
        and spec.moe_backend in (None, "auto")
        and not spec.parallel_drafting
        and spec.revision in (None, model.revision)
        and spec.code_revision in (None, model.code_revision)
    )


def qualifies_unequal_tp_mla_mtp(config: VllmConfig) -> bool:
    """An endpoint in the verified whole-page DeepSeek-V3 MTP scope."""
    if not _qualifies_unequal_tp_mtp_endpoint(config, UNEQUAL_TP_MLA_MTP_PAGE):
        return False
    sim, cache = config.simulation_config, config.cache_config
    spec = config.speculative_config
    model = config.model_config
    draft = spec.draft_model_config
    if (
        sim.timing_args.get("gpu") != "h200"
        or sim.timing_args.get("target_model") != "deepseek-ai/DeepSeek-V3"
        or model.hf_config.model_type != "deepseek_v3"
        or model.architectures != ["DeepseekV3ForCausalLM"]
        or not plain_mla(model)
        or SimulationConfig._native_mtp_mismatches(config, effective_kv=True)
        or draft is None
        or draft.hf_config.model_type != "deepseek_mtp"
        or not plain_mla(draft)
        or cache.cache_dtype not in ("auto", "bfloat16")
        or spec.kv_cache_dtype not in (None, "auto", "bfloat16")
    ):
        return False

    # Match the published V3 KV geometry and block-FP8 weight profile on
    # both resolved model configs; explicit BF16 and BF16 through auto agree.
    for endpoint in (model, draft):
        hf = endpoint.hf_text_config
        quant = getattr(hf, "quantization_config", None) or {}
        if (
            str(endpoint.dtype) != "torch.bfloat16"
            or endpoint.quantization != "fp8"
            or endpoint.quantization_config is not None
            or quant.get("quant_method") != "fp8"
            or quant.get("weight_block_size") != [128, 128]
            or hf.num_hidden_layers != 61
            or hf.hidden_size != 7168
            or hf.q_lora_rank != 1536
            or hf.n_routed_experts != 256
            or hf.num_experts_per_tok != 8
            or hf.num_nextn_predict_layers != 1
            or hf.num_attention_heads != 128
            or hf.kv_lora_rank != 512
            or hf.qk_rope_head_dim != 64
        ):
            return False
    return True


@dataclass(frozen=True)
class UnequalTpMtpLayout:
    """The ordered ordinary regions of one qualified native MLA MTP profile."""

    capability: str
    page_size: int
    target_layers: int
    latent_page_bytes: int
    indexer_page_bytes: int = 0

    def layer_regions(self, layer: int) -> tuple[str, ...]:
        prefix = f"model.layers.{layer}.self_attn"
        latent = f"{prefix}.attn"
        if self.indexer_page_bytes:
            return latent, f"{prefix}.indexer.k_cache"
        return (latent,)

    @property
    def draft_regions(self) -> tuple[str, ...]:
        return self.layer_regions(self.target_layers)

    @property
    def names(self) -> list[str]:
        return [
            name
            for layer in range(self.target_layers + 1)
            for name in self.layer_regions(layer)
        ]

    @property
    def page_bytes(self) -> list[int]:
        per_layer = [self.latent_page_bytes]
        if self.indexer_page_bytes:
            per_layer.append(self.indexer_page_bytes)
        return per_layer * (self.target_layers + 1)


_DENSE_MTP_LAYOUT = UnequalTpMtpLayout(
    UNEQUAL_TP_MLA_MTP,
    UNEQUAL_TP_MLA_MTP_PAGE,
    61,
    UNEQUAL_TP_MLA_MTP_PAGE_BYTES,
)
_V32_MTP_LAYOUT = UnequalTpMtpLayout(
    "simulated/nixl/unequal-tp-dsa-v32-mtp/v1",
    64,
    61,
    64 * 576 * 2,
    64 * 132,
)
_GLM5_MTP_LAYOUT = UnequalTpMtpLayout(
    "simulated/nixl/unequal-tp-dsa-glm5-mtp/v1",
    64,
    78,
    64 * 656,
    64 * 132,
)
UNEQUAL_TP_MTP_CAPABILITIES = frozenset(
    layout.capability
    for layout in (_DENSE_MTP_LAYOUT, _V32_MTP_LAYOUT, _GLM5_MTP_LAYOUT)
)


def _sparse_mtp_layout(config: VllmConfig) -> UnequalTpMtpLayout | None:
    cache, sim, model = (
        config.cache_config,
        config.simulation_config,
        config.model_config,
    )
    if not _qualifies_unequal_tp_mtp_endpoint(config, 64) or not sparse_mla(model):
        return None
    spec = config.speculative_config
    draft = spec.draft_model_config
    if (
        draft is None
        or draft.hf_config.model_type != "deepseek_mtp"
        or not draft.use_mla
        or draft.is_hybrid
        or SimulationConfig._native_mtp_mismatches(config, effective_kv=True)
    ):
        return None

    glm = model.hf_config.model_type == "glm_moe_dsa"
    if glm:
        layout, arch, cache_dtype, quant_method = (
            _GLM5_MTP_LAYOUT,
            "GlmMoeDsaForCausalLM",
            "fp8_ds_mla",
            "modelopt_fp4",
        )
        timing_args = {"gpu": "b200", "target_model": "nvidia/GLM-5-NVFP4"}
    else:
        layout, arch, cache_dtype, quant_method = (
            _V32_MTP_LAYOUT,
            "DeepseekV32ForCausalLM",
            "bfloat16",
            "fp8",
        )
        timing_args = {"gpu": "h200", "target_model": "deepseek-ai/DeepSeek-V3.2"}
    # CPU resolves an explicit BF16 latent to auto on these models.
    # Below, both the target and the draft are required to be BF16.
    cache_dtypes = (cache_dtype,) if glm else ("auto", "bfloat16")
    if (
        model.architectures != [arch]
        or sim.timing_args != timing_args
        or cache.cache_dtype not in cache_dtypes
        or spec.kv_cache_dtype not in (None, *cache_dtypes)
    ):
        return None

    geometry = {
        "num_hidden_layers": layout.target_layers,
        "num_nextn_predict_layers": 1,
        "hidden_size": 6144 if glm else 7168,
        "intermediate_size": 12288 if glm else 18432,
        "moe_intermediate_size": 2048,
        "first_k_dense_replace": 3,
        "n_shared_experts": 1,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "num_attention_heads": 64 if glm else 128,
        "kv_lora_rank": 512,
        "q_lora_rank": 2048 if glm else 1536,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 192 if glm else 128,
        "v_head_dim": 256 if glm else 128,
        "index_head_dim": 128,
        "index_n_heads": 32 if glm else 64,
        "index_topk": 2048,
        "vocab_size": 154880 if glm else 129280,
    }
    for endpoint in (model, draft):
        hf = endpoint.hf_text_config
        quant = endpoint.model_arch_config.quantization_config or {}
        if (
            str(endpoint.dtype) != "torch.bfloat16"
            or endpoint.quantization != quant_method
            or endpoint.quantization_config is not None
            or any(getattr(hf, name, None) != value for name, value in geometry.items())
            or getattr(hf, "index_topk_freq", 1) != 1
            or getattr(hf, "index_topk_pattern", None) is not None
            or bool(getattr(hf, "indexer_rope_interleave", False)) != glm
        ):
            return None
        if glm:
            sidecar = quant.get("quantization") or {}
            if (
                sidecar.get("quant_algo") != "NVFP4"
                or sidecar.get("kv_cache_quant_algo") != "FP8"
                or sidecar.get("group_size") != 16
            ):
                return None
        elif (
            quant.get("quant_method") != "fp8"
            or quant.get("weight_block_size") != [128, 128]
        ):
            return None
    return layout


def unequal_tp_mtp_layout(config: VllmConfig) -> UnequalTpMtpLayout | None:
    """The layout an endpoint may offer to an unequal-TP NIXL peer."""
    if qualifies_unequal_tp_mla_mtp(config):
        return _DENSE_MTP_LAYOUT
    return _sparse_mtp_layout(config)


def unequal_tp_mla_mtp_refusal(local_tp_size: int, remote_tp_size: int) -> str:
    return (
        "Simulated real KV transfers pair unequal TP sizes for plain attention, "
        "plain or sparse MLA only (no compressed MLA, recurrent state "
        f"or draft), not local TP {local_tp_size} and remote TP {remote_tp_size}."
    )


def simulated_dcp_scope(vllm_config: VllmConfig) -> list[str] | None:
    """An instance's `dcp_scope_problems` with real KV transfers under
    simulated compute, which NIXL advertises for its peer to check whichever
    side has decode context parallelism. It also advertises the versioned
    unequal-TP MLA MTP capability when qualified. None outside simulation."""
    if not moves_real_kv(vllm_config):
        return None
    scope = dcp_scope_problems(vllm_config, transfers=True)
    if layout := unequal_tp_mtp_layout(vllm_config):
        scope.append(layout.capability)
    return scope


def simulated_tp_mismatch(
    vllm_config: VllmConfig,
    local_tp_size: int,
    remote_tp_size: int,
    *,
    nixl_consumer: bool = False,
    remote_dcp_size: int = 1,
) -> str | None:
    """Why a peer's tensor-parallel size cannot pair with ours, if it cannot:
    real KV transfers under simulated compute pair equal sizes, or unequal
    ones (one dividing the other) for a model of plain attention (sliding
    window layers beside full ones included; no recurrent state or draft),
    or, over NIXL only, a hybrid of Mamba2 or GDN states or Kimi-K3 at
    decode TP4 from prefill TP8 (`state_layout.pairing_problem`), whose
    pattern is each KV head's
    (``kv_pattern.head_pattern``) wherever the connector re-shards it, or of
    plain MLA (`plain_mla`: one latent, the same on every rank) or sparse MLA
    (`sparse_mla`: an indexer's keys beside it, the same on every rank too). NIXL's
    decode (`nixl_consumer`, the local side) refuses, as NIXL itself does, a
    larger producer whose KV heads are replicated; MLA's latent is exempt.
    A qualified DeepSeek-V3 MTP consumer can continue to NIXL's handshake,
    where the peer's capability and its registered regions are checked.
    Decode context parallelism on either side (`remote_dcp_size` the peer's,
    from NIXL's handshake) pairs TP sizes one dividing the other (NIXL maps
    each decode rank's DCP slice to the producer ranks holding it,
    `TransferTopology.dcp_source_ranks`), this instance in
    `dcp_scope_problems`'s scope whatever its own DCP size (the peer's
    scope is checked from its NIXL metadata, `simulated_scope`)."""
    if not moves_real_kv(vllm_config):
        return None
    local_dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    if local_dcp_size > 1 or remote_dcp_size > 1:
        problems = dcp_scope_problems(vllm_config, transfers=True)
        larger, smaller = max(local_tp_size, remote_tp_size), min(
            local_tp_size, remote_tp_size
        )
        if larger % smaller:
            problems.append(f"local TP {local_tp_size} and remote TP {remote_tp_size}")
        if problems:
            return dcp_pair_refusal(local_dcp_size, remote_dcp_size, problems)
        return None
    if local_tp_size == remote_tp_size:
        return None
    sizes = f"local TP {local_tp_size} and remote TP {remote_tp_size}"
    larger, smaller = max(local_tp_size, remote_tp_size), min(local_tp_size, remote_tp_size)
    if larger % smaller:
        return f"Simulated real KV transfers pair TP sizes one dividing the other, not {sizes}."
    model = vllm_config.model_config
    # Layers of different KV heads (Gemma4's global layers beside its
    # sliding ones): the connectors count one head total, so only sizes that
    # replicate no layer and split every layer evenly on both sides are
    # admitted (each total a positive integer, never a bool, at least the
    # larger size and divisible by both).
    per_layer = getattr(model.hf_text_config, "per_layer_config", None) or ()
    totals = [getattr(layer, "num_key_value_heads", None) for layer in per_layer]
    if len(set(totals)) > 1:
        if not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in totals):
            return (
                "Simulated real KV transfers pair unequal TP sizes for layers of "
                f"valid KV head counts only (this model's layers differ: {sorted(map(str, set(totals)))}), "
                f"not {sizes}."
            )
        if any(n < larger or n % local_tp_size or n % remote_tp_size for n in totals):
            return (
                "Simulated real KV transfers pair unequal TP sizes for layers of "
                "differing KV heads only where none replicates and each splits "
                f"evenly on both sides (this model's layers differ in KV heads: "
                f"{sorted(set(totals))}), not {sizes}."
            )
    if model.is_hybrid and nixl_consumer:
        from vllm.v1.worker.simulated.state_layout import pairing_problem

        problem = pairing_problem(vllm_config, local_tp_size, remote_tp_size)
        if problem is not None:
            return (
                "Simulated real KV transfers pair unequal TP sizes for a hybrid model "
                "over NIXL only for Mamba2 or GDN states that split on both sides, or "
                f"Kimi-K3 from prefill TP8 to decode TP4, not {sizes}: {problem}."
            )
        return None
    if (
        (model.use_mla and not (plain_mla(model) or sparse_mla(model)))
        or model.is_hybrid
        or vllm_config.speculative_config is not None
    ):
        if (
            nixl_consumer
            and {local_tp_size, remote_tp_size} == {2, 4}
            and local_tp_size == vllm_config.parallel_config.tensor_parallel_size
            and unequal_tp_mtp_layout(vllm_config) is not None
        ):
            return None
        return unequal_tp_mla_mtp_refusal(local_tp_size, remote_tp_size)
    if (
        nixl_consumer
        and not model.use_mla
        and remote_tp_size > local_tp_size
        and remote_tp_size > model.get_total_num_kv_heads()
    ):
        # NIXL asserts the same (nixl/base_worker.py, before reading).
        return (
            f"NIXL does not read from a larger producer whose KV heads are "
            f"replicated ({sizes}, {model.get_total_num_kv_heads()} KV heads)."
        )
    return None
