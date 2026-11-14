# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read qualified GPU kernel block rules without importing GPU backends."""

import ast
from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from pathlib import Path

import torch

from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends import registry
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode

_PROFILES = {
    "FLASHINFER": (False, ("h100", "h200")),
    "TRITON_ATTN": (False, ("a100", "h100", "h200", "b200")),
    "FLASHMLA": (True, ("h100", "h200")),
    "FLASHINFER_MLA": (True, ("b200",)),
    "FLASH_ATTN_MLA": (True, ("h100", "h200")),
    "TRITON_MLA": (True, ("a100", "h100", "h200", "b200")),
}

# Only on the qualified non-SM100 profiles above. If the source acquires
# another condition, it needs a fresh review, including the default branch.
_FLASHINFER_CONDITION = ast.parse("""\
use_large_pages = False
vllm_config = get_current_vllm_config_or_none()
if vllm_config is not None and vllm_config.model_config is not None:
    pc = vllm_config.parallel_config
    mc = vllm_config.model_config
    num_qo_heads = mc.get_num_attention_heads(pc)
    num_kv_heads = mc.get_num_kv_heads(pc)
    use_large_pages = (
        num_kv_heads > 0
        and num_qo_heads // num_kv_heads > 1
        and current_platform.is_device_capability_family(100)
        and can_use_trtllm_attention(num_qo_heads, num_kv_heads)
    )
""").body


@dataclass(frozen=True)
class KernelRule:
    sizes: tuple[int | MultipleOf, ...]
    source: str
    source_sha256: str


def _positive_int(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and type(node.value) is int and node.value > 0:
        return node.value
    raise ValueError("expected a positive integer literal")


@cache
def read_rule(name: str, class_path: str, root: Path) -> KernelRule:
    module, _, class_name = class_path.rpartition(".")
    source = module.removeprefix("vllm.").replace(".", "/") + ".py"
    data = (root / source).read_bytes()
    (cls,) = [
        node
        for node in ast.parse(data).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    (method,) = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_supported_kernel_block_sizes"
    ]
    body = list(method.body)
    if (
        isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    if name == "FLASHINFER":
        if len(body) != len(_FLASHINFER_CONDITION) + 2 or ast.dump(
            ast.Module(body=body[:-2], type_ignores=[])
        ) != ast.dump(ast.Module(body=_FLASHINFER_CONDITION, type_ignores=[])):
            raise ValueError("FlashInfer's native condition has changed")
        branch = body[-2]
        if (
            not isinstance(branch, ast.If)
            or ast.unparse(branch.test) != "not use_large_pages"
            or branch.orelse
            or not isinstance(body[-1], ast.Return)
        ):
            raise ValueError("FlashInfer's native branches have changed")
        body = branch.body
    if (
        len(body) != 1
        or not isinstance(body[0], ast.Return)
        or not isinstance(body[0].value, ast.List)
        or not body[0].value.elts
    ):
        raise ValueError("expected a literal kernel block size list")
    sizes = []
    for node in body[0].value.elts:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "MultipleOf"
            and len(node.args) == 1
            and not node.keywords
        ):
            sizes.append(MultipleOf(_positive_int(node.args[0])))
        else:
            sizes.append(_positive_int(node))
    return KernelRule(tuple(sizes), f"vllm/{source}", sha256(data).hexdigest())


def declared_rule(config, spec=None) -> KernelRule | None:
    if config is None or not config.simulation_config.simulate_forward:
        return None
    sim = config.simulation_config
    name = sim.kv_kernel_backend
    if name is None:
        return None
    if name not in _PROFILES:
        raise ValueError(
            f"Unknown kv_kernel_backend {name!r}; expected one of {sorted(_PROFILES)}."
        )
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import plain_mla

    mla, systems = _PROFILES[name]
    model = config.model_config
    gpu = sim.timing_args.get("gpu", "h100")
    dtype = config.cache_config.cache_dtype
    if (
        model is None
        or config.speculative_config is not None
        or (model.use_mla and not plain_mla(model))
        or model.use_mla != mla
        or model.is_attention_free
        or model.supports_multimodal_inputs
        or gpu not in systems
        or not (
            dtype == "bfloat16" or dtype == "auto" and model.dtype == torch.bfloat16
        )
        or (
            spec is not None
            and (
                not isinstance(spec, AttentionSpec)
                or spec.dtype != torch.bfloat16
                or spec.tokens_per_state != 1
                or spec.kv_quant_mode != KVQuantMode.NONE
            )
        )
    ):
        raise ValueError(
            f"kv_kernel_backend {name} does not describe this attention/KV kind "
            f"on AISimulate GPU preset {gpu!r} (requires BF16 KV and no drafts)."
        )
    backend = AttentionBackendEnum[name]
    if backend.is_overridden():
        raise ValueError(
            f"Cannot resolve the native kernel block rule for {name}: "
            "its registry entry is overridden."
        )
    try:
        root = Path(registry.__file__).resolve().parents[3]
        return read_rule(name, backend.value, root)
    except (OSError, SyntaxError, ValueError, IndexError) as exc:
        raise ValueError(
            f"Cannot resolve the native kernel block rule for {name}: {exc}."
        ) from exc
