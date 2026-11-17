# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 over NIXL at unequal tensor-parallel sizes under simulated
execution with real KV transfers.

Every V4 cache layer is one replicated shard (`max_tp_shards=1`): each rank
registers the same packed block rows, so NIXL pairs a decode rank with one
producer rank, whatever the two TP sizes. An endpoint states that it
qualifies as one scope entry (`v4_capability`); the consumer parses its own
and the producer's with the same function and compares them, then compares
the registered geometry (`registration_problem`), before NIXL registers the
peer."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.config.simulation import NIXL_CONNECTORS, SimulationConfig, moves_real_kv

CAPABILITY_FAMILY = "simulated/nixl/unequal-tp-dsv4/"
CAPABILITY_VERSION = "v1"
OTHER_UNEQUAL_TP_FAMILY = "simulated/nixl/unequal-tp-"
CAPABILITY_KEYS = ("cache", "geometry", "gpu", "mtp", "prefix", "role", "tp")
TARGET_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
GPU = "b200"
CACHE_FORMAT = "fp8_ds_mla"
CACHE_DTYPES = ("auto", "fp8", "fp8_e4m3", "fp8_ds_mla")
# The tensor-parallel sizes AISimulate prices (`aisimulate_timing.py`
# TENSOR_PARALLEL_SIZES).
TENSOR_PARALLEL_SIZES = (1, 2, 4, 8)
PRODUCER, CONSUMER = "kv_producer", "kv_consumer"
GEOMETRY_FIELDS = (
    "model_type",
    "num_hidden_layers",
    "compress_ratios",
    "head_dim",
    "qk_rope_head_dim",
    "sliding_window",
    "index_head_dim",
    "index_topk",
    "num_key_value_heads",
    "num_nextn_predict_layers",
    "hidden_size",
    "expert_dtype",
)


def v4_model(model) -> bool:
    return getattr(model.hf_config, "model_type", None) == "deepseek_v4"


def geometry_digest(config: VllmConfig) -> str:
    """The published-geometry fields the cache groups depend on, hashed."""
    model = config.model_config
    text = model.hf_text_config
    fields = {name: getattr(text, name, None) for name in GEOMETRY_FIELDS}
    fields["architectures"] = list(model.architectures)
    fields["dtype"] = str(model.dtype)
    blob = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def unequal_tp_v4_problems(config: VllmConfig) -> list[str]:
    """What keeps this endpoint out of the unequal-TP DeepSeek-V4 scope (an
    empty list: it qualifies). Each entry names one requirement."""
    model = config.model_config
    if model is None or not v4_model(model):
        return ["a DeepSeek-V4 model"]
    transfer = config.kv_transfer_config
    if transfer is None or not moves_real_kv(config):
        return ["real KV transfers under simulated compute"]
    sim, cache, pc = (
        config.simulation_config,
        config.cache_config,
        config.parallel_config,
    )
    problems = []
    if transfer.kv_connector not in NIXL_CONNECTORS:
        problems.append(f"NixlConnector (pull), not {transfer.kv_connector}")
    if transfer.kv_role not in (PRODUCER, CONSUMER):
        problems.append(f"kv_role kv_producer or kv_consumer, not {transfer.kv_role}")
    if transfer.get_from_extra_config("bidirectional_kv_xfer", False):
        problems.append("no bidirectional transfer")
    if not transfer.get_from_extra_config("enforce_handshake_compat", True):
        problems.append("NIXL's handshake compatibility check on")
    if (
        sim.timing != "aisimulate"
        or sim.timing_args.get("gpu") != GPU
        or sim.timing_args.get("target_model") != TARGET_MODEL
        or sim.timing_args.get("quant_modes")
    ):
        problems.append(
            f"AISimulate timing of {TARGET_MODEL} on {GPU} without quant_modes"
        )
    if sim.kv_kernel_backend is not None:
        problems.append("no declared kernel geometry")
    if cache.cache_dtype not in CACHE_DTYPES:
        problems.append(f"the packed FP8 cache, not {cache.cache_dtype}")
    if str(model.dtype) != "torch.bfloat16":
        problems.append(f"BF16, not {model.dtype}")
    if pc.tensor_parallel_size not in TENSOR_PARALLEL_SIZES:
        problems.append(
            f"tensor parallelism in {TENSOR_PARALLEL_SIZES}, not "
            f"{pc.tensor_parallel_size}"
        )
    for name, size in (
        ("pipeline parallelism", pc.pipeline_parallel_size),
        ("prefill context parallelism", pc.prefill_context_parallel_size),
        ("decode context parallelism", pc.decode_context_parallel_size),
        ("nodes", pc.nnodes),
    ):
        if size != 1:
            problems.append(f"no {name} ({size})")
    if (spec := config.speculative_config) is not None:
        mismatches = SimulationConfig._native_mtp_mismatches(config, effective_kv=True)
        if (
            spec.uses_dynamic_speculative_decoding()
            or spec.enable_adaptive_verification
            or spec.attention_backend is not None
            or spec.moe_backend not in (None, "auto")
            or spec.parallel_drafting
            or spec.revision not in (None, model.revision)
            or spec.code_revision not in (None, model.code_revision)
        ):
            mismatches.append("a draft kernel, dynamic or parallel drafting override")
        if mismatches:
            problems.append(
                f"native single-module MTP only, not {', '.join(mismatches)}"
            )
    return problems


def v4_fields(config: VllmConfig) -> dict[str, str]:
    spec = config.speculative_config
    return {
        "cache": CACHE_FORMAT,
        "geometry": geometry_digest(config),
        "gpu": GPU,
        "mtp": "none" if spec is None else "mtp",
        "prefix": "on" if config.cache_config.enable_prefix_caching else "off",
        "role": config.kv_transfer_config.kv_role,
        "tp": str(config.parallel_config.tensor_parallel_size),
    }


def format_capability(fields: Mapping[str, str]) -> str:
    body = ";".join(f"{key}={fields[key]}" for key in CAPABILITY_KEYS)
    return f"{CAPABILITY_FAMILY}{CAPABILITY_VERSION};{body}"


def v4_capability(config: VllmConfig) -> str | None:
    """This endpoint's scope entry, or None when it does not qualify."""
    if unequal_tp_v4_problems(config):
        return None
    return format_capability(v4_fields(config))


def parse_capability(entry: str) -> dict[str, str] | str:
    """The fields of one entry, or why it is not a well-formed one."""
    head, _, body = entry.partition(";")
    if head != f"{CAPABILITY_FAMILY}{CAPABILITY_VERSION}":
        return f"version {head.removeprefix(CAPABILITY_FAMILY)!r}, not v1"
    fields: dict[str, str] = {}
    for part in body.split(";"):
        key, sep, value = part.partition("=")
        if not sep or key in fields:
            return f"an unreadable or repeated field {part!r}"
        fields[key] = value
    if set(fields) != set(CAPABILITY_KEYS):
        return f"fields {sorted(fields)}, not {sorted(CAPABILITY_KEYS)}"
    if (
        fields["role"] not in (PRODUCER, CONSUMER)
        or fields["prefix"] not in ("on", "off")
        or fields["mtp"] not in ("none", "mtp")
        or not re.fullmatch(r"[1-9][0-9]*", fields["tp"])
        or not re.fullmatch(r"[0-9a-f]{16}", fields["geometry"])
    ):
        return f"values {fields}"
    return fields


def peer_capability(scope: Sequence[str] | None) -> dict[str, str] | str:
    """The one entry a peer's scope must carry, parsed; else why not."""
    if scope is None:
        return "states no simulated scope"
    entries = [e for e in scope if e.startswith(OTHER_UNEQUAL_TP_FAMILY)]
    ours = [e for e in entries if e.startswith(CAPABILITY_FAMILY)]
    if len(ours) != len(entries):
        return "states another model family's unequal-TP capability too"
    if not ours:
        return "states no DeepSeek-V4 unequal-TP capability"
    if len(ours) > 1:
        return f"states {len(ours)} DeepSeek-V4 unequal-TP capabilities"
    parsed = parse_capability(ours[0])
    if isinstance(parsed, str):
        return f"states a malformed DeepSeek-V4 capability: {parsed}"
    return parsed


def unequal_tp_v4_refusal(
    config: VllmConfig,
    local_tp: int,
    remote_tp: int,
    nixl_consumer: bool,
    generic: str,
) -> str | None:
    """`simulated_tp_mismatch` for a DeepSeek-V4 instance whose TP size is not
    its peer's (one divides the other): admitted when this endpoint
    qualifies, as NIXL's consumer; else `generic` (the refusal every other
    model gets) and the requirements it does not meet."""
    problems = unequal_tp_v4_problems(config)
    if not nixl_consumer:
        problems.insert(0, "a NIXL pull consumer to pair")
    elif getattr(config.kv_transfer_config, "kv_role", None) != CONSUMER:
        problems.append("kv_role kv_consumer to pair")
    if not problems:
        return None
    return f"{generic} DeepSeek-V4 pairs only as {'; '.join(problems)}."


def capability_problem(
    config: VllmConfig, remote_tp: int, scope: Sequence[str] | None
) -> str | None:
    """Why the producer whose `scope` this is cannot pair with this consumer,
    if it cannot: both endpoints are parsed by one function and must agree."""
    local = unequal_tp_v4_problems(config)
    if local:
        return f"this instance is outside the scope: {'; '.join(local)}"
    mine = parse_capability(format_capability(v4_fields(config)))
    assert not isinstance(mine, str)
    if mine["role"] != CONSUMER:
        return f"this instance is {mine['role']}, not a consumer"
    theirs = peer_capability(scope)
    if isinstance(theirs, str):
        return f"the peer {theirs}"
    if theirs["role"] != PRODUCER:
        return f"the peer is {theirs['role']}, not a producer"
    for key in ("cache", "geometry", "gpu", "mtp", "prefix"):
        if theirs[key] != mine[key]:
            return f"the peer's {key} is {theirs[key]}, this instance's {mine[key]}"
    if theirs["tp"] != str(remote_tp):
        return (
            f"the peer states TP {theirs['tp']} but its handshake gives TP {remote_tp}"
        )
    return None


@dataclass(frozen=True)
class Registration:
    """What one NIXL endpoint registers, as its handshake metadata states it."""

    region_names: tuple
    block_lens: tuple
    block_strides: tuple
    group_ids: tuple
    members: tuple
    packed: tuple
    block_size: int
    factor: int

    @classmethod
    def of_worker(cls, worker) -> "Registration":
        return cls(
            tuple(worker.region_names),
            tuple(worker.block_len_per_layer),
            tuple(worker.block_stride_per_layer),
            tuple(worker.region_group_ids),
            tuple(tuple(m) for m in worker.region_members),
            tuple(
                sorted((k, tuple(v)) for k, v in worker._packed_member_layouts.items())
            ),
            worker.block_size,
            worker._physical_blocks_per_logical_kv_block,
        )

    @classmethod
    def of_metadata(cls, metadata) -> "Registration":
        return cls(
            tuple(metadata.region_names or ()),
            tuple(metadata.block_lens),
            tuple(metadata.block_strides),
            tuple(metadata.region_group_ids or ()),
            tuple(tuple(m) for m in metadata.region_members),
            tuple(
                sorted((k, tuple(v)) for k, v in metadata.packed_member_layouts.items())
            ),
            metadata.block_size,
            metadata.physical_blocks_per_logical_kv_block,
        )


def registration_problem(local: Registration, remote: Registration) -> str | None:
    """The first registered field in which the peer differs, named."""
    for name in (
        "region_names",
        "block_lens",
        "block_strides",
        "group_ids",
        "members",
        "packed",
        "block_size",
        "factor",
    ):
        mine, theirs = getattr(local, name), getattr(remote, name)
        if mine != theirs:
            shown = (
                (len(mine), len(theirs))
                if isinstance(mine, tuple) and len(repr(mine)) > 120
                else (mine, theirs)
            )
            return f"the peer registers other {name} (this instance, peer): {shown}"
    return None


def v4_peer_refusal(
    config: VllmConfig, local: Registration, remote_tp: int, metadata
) -> str | None:
    """Before this NIXL agent (`config`, registering `local`) registers a peer
    of another TP size: the capabilities, then the registered geometry. None
    for equal sizes and for any other model."""
    local_tp = config.parallel_config.tensor_parallel_size
    if not v4_model(config.model_config) or remote_tp == local_tp:
        return None
    reason = capability_problem(config, remote_tp, metadata.simulated_scope)
    if reason is None:
        reason = registration_problem(local, Registration.of_metadata(metadata))
    if reason is None:
        return None
    return (
        "Simulated NIXL pairs DeepSeek-V4 at unequal TP sizes (local TP "
        f"{local_tp}, remote TP {remote_tp}) only between endpoints that "
        f"qualify and register alike: {reason}."
    )
