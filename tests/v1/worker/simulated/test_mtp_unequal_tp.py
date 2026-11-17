# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.v1.worker.simulated.test_kernel_geometry import V3, published_path

DRAFT = "model.layers.61.self_attn.attn"
PAGE_BYTES = 36_864
CHECKED_BYTES = 3 * 62 * PAGE_BYTES
REFUSAL = "plain or sparse MLA only (no compressed MLA, recurrent state or draft)"


def _engine_options(model, tp, producer, path, **case):
    check = case.pop("check", True)
    simulation = {
        "simulate_forward": True,
        "timing": "aisimulate",
        "timing_args": {"gpu": "h200", "target_model": V3},
        "spec_acceptance_rates": [1.0],
        "log_path": str(path),
    }
    if check:
        simulation["kv_pattern_fill" if producer else "kv_pattern_check"] = True
    spec = {"method": "mtp", "num_speculative_tokens": case.pop("depth", 1)}
    spec.update(case.pop("spec_overrides", {}))
    return {
        "model": str(model),
        "skip_tokenizer_init": True,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "block_size": 32,
        "max_model_len": 256,
        "max_num_batched_tokens": 256,
        "max_num_seqs": 4,
        "kv_cache_memory_bytes": 64 << 20,
        "enable_prefix_caching": False,
        "tensor_parallel_size": tp,
        "speculative_config": spec,
        "simulation_config": simulation,
        "kv_transfer_config": {
            "kv_connector": "NixlConnector",
            "kv_buffer_device": "cpu",
            "kv_role": "kv_producer" if producer else "kv_consumer",
            "kv_connector_extra_config": {"backends": ["Mooncake"]},
        },
        **case,
    }


def _inspect(worker):
    import msgspec

    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlAgentMetadata,
    )

    connector = get_kv_transfer_group().connector_worker
    data = connector.xfer_handshake_metadata.agent_metadata_bytes
    meta = msgspec.msgpack.decode(data, type=NixlAgentMetadata)
    return {
        "names": meta.region_names,
        "lengths": meta.block_lens,
        "scope": meta.simulated_scope,
        "kernel": meta.block_size,
        "factor": meta.physical_blocks_per_logical_kv_block,
        "groups": meta.region_group_ids,
    }


def _corrupt_draft(worker, block):
    import torch

    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

    if get_tensor_model_parallel_rank() == 0:
        cache = get_kv_transfer_group().connector_worker.device_kv_caches[DRAFT]
        cache[block].view(-1).view(torch.uint8)[0] ^= 1


def _peer():
    from vllm import LLM, SamplingParams

    role, model, raw_tp, raw_settings, dest = sys.argv[1:]
    p = role == "prefill"
    settings = json.loads(raw_settings)
    corrupt = settings.pop("corrupt", False)
    directory = Path(dest)
    llm = LLM(**_engine_options(model, int(raw_tp), p, directory / role, **settings))
    (directory / f"{role}.regions.json").write_text(
        json.dumps(llm.collective_rpc(_inspect))
    )
    handoff = (
        {"do_remote_decode": True}
        if p
        else json.loads((directory / "prefill.json").read_text())["handoff"]
    )
    params = SamplingParams(
        max_tokens=1 if p else 12,
        ignore_eos=True,
        detokenize=False,
        temperature=0,
        extra_args={"kv_transfer_params": handoff},
    )
    (response,) = llm.generate([{"prompt_token_ids": [5] * 81}], params, use_tqdm=False)
    if p and corrupt:
        block = response.kv_transfer_params["remote_block_ids"][0][0]
        llm.collective_rpc(_corrupt_draft, args=(block,))
    stage = directory / f"{role}.tmp"
    stage.write_text(
        json.dumps(
            {
                "handoff": response.kv_transfer_params,
                "tokens": list(response.outputs[0].token_ids),
            }
        )
    )
    stage.rename(directory / f"{role}.json")
    if p:
        core = llm.llm_engine.engine_core.engine_core
        deadline = time.monotonic() + 180
        while not (directory / "stop").exists() and time.monotonic() < deadline:
            core.step()
            time.sleep(0.02)


def _free_bases(width):
    # A side channel may offset the base by the TP rank.
    held = []
    try:
        for _ in range(2):
            while True:
                group = []
                try:
                    first = socket.socket()
                    group.append(first)
                    first.bind(("127.0.0.1", 0))
                    base = first.getsockname()[1]
                    if base + width > 65536:
                        continue
                    for port in range(base + 1, base + width):
                        sock = socket.socket()
                        group.append(sock)
                        sock.bind(("127.0.0.1", port))
                    held += group
                    break
                except OSError:
                    continue
                finally:
                    if not all(sock in held for sock in group):
                        for sock in group:
                            sock.close()
        return held[0].getsockname()[1], held[width].getsockname()[1]
    finally:
        for sock in held:
            sock.close()


def _run_peers(directory, p_tp, d_tp, p_options, d_options):
    model = published_path(V3)
    p_port, d_port = _free_bases(max(p_tp, d_tp))
    script = directory / "peer.py"
    script.write_text(
        "from tests.v1.worker.simulated.test_mtp_unequal_tp import _peer\n"
        "if __name__ == '__main__':\n    _peer()\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd()),
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "VLLM_HOST_IP": "127.0.0.1",
        "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
        "NIXL_MOONCAKE_IP_ADDR": "127.0.0.1",
    }

    def command(role, size, options):
        return [
            sys.executable,
            str(script),
            role,
            str(model),
            str(size),
            json.dumps(options),
            str(directory),
        ]

    p_log = directory / "prefill.log"
    d_log = directory / "decode.log"
    with p_log.open("w") as f:
        producer = subprocess.Popen(
            command("prefill", p_tp, p_options),
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(p_port)},
        )
    try:
        deadline = time.monotonic() + 120
        while not (directory / "prefill.json").exists():
            assert producer.poll() is None, p_log.read_text()[-6000:]
            assert time.monotonic() < deadline, p_log.read_text()[-6000:]
            time.sleep(0.15)
        with d_log.open("w") as f:
            decode = subprocess.run(
                command("decode", d_tp, d_options),
                stdout=f,
                stderr=subprocess.STDOUT,
                env={**env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(d_port)},
                timeout=120,
            )
        return decode.returncode
    finally:
        (directory / "stop").touch()
        try:
            producer.wait(timeout=20)
        except subprocess.TimeoutExpired:
            producer.kill()
            producer.wait(timeout=20)
            raise


def _records(directory, role, rank):
    path = directory / (role if rank == 0 else f"{role}.tp{rank}")
    return [json.loads(line) for line in path.read_text().splitlines()]


def _loads(records):
    return [
        entry
        for row in records
        for entry in row.get("kv_transfers", {}).get("load", {}).get("each", [])
    ]


def _checks(records):
    return [row["kv_check"] for row in records if "kv_check" in row]


@pytest.mark.parametrize(
    ("p_tp", "d_tp", "corrupt", "check_bytes"),
    [
        (4, 2, False, True),
        (2, 4, False, True),
        (2, 4, True, True),
        (4, 2, False, False),
        (2, 2, False, True),
    ],
    ids=["gather", "scatter", "scatter-corrupt", "gather-unchecked", "equal-tp"],
)
def test_native_mla_mtp_moves_unequal_tp(tmp_path, p_tp, d_tp, corrupt, check_bytes):
    rc = _run_peers(
        tmp_path,
        p_tp,
        d_tp,
        {"corrupt": corrupt, "check": check_bytes},
        {"check": check_bytes},
    )
    log = (tmp_path / "decode.log").read_text()
    assert REFUSAL not in log, log[-6000:]
    assert rc == 0, log[-6000:]
    assert len(json.loads((tmp_path / "decode.json").read_text())["tokens"]) == 12
    handoff = json.loads((tmp_path / "prefill.json").read_text())["handoff"]
    assert handoff["remote_num_tokens"] in (80, 81), handoff
    assert [len(blocks) for blocks in handoff["remote_block_ids"]] == [3]
    for role, tp in (("prefill", p_tp), ("decode", d_tp)):
        regions = json.loads((tmp_path / f"{role}.regions.json").read_text())
        assert len(regions) == tp
        for reg in regions:
            assert (reg["kernel"], reg["factor"]) == (32, 1)
            assert len(reg["names"]) == 62 and reg["names"].count(DRAFT) == 1
            assert reg["lengths"] == [PAGE_BYTES] * 62
            assert reg["groups"] == [0] * 62
            endpoint = "producer" if role == "prefill" else "consumer"
            assert (
                "simulated/nixl/unequal-tp-mla-mtp/v1/k1/b32/prefix0/f1/" + endpoint
            ) in reg["scope"]
    for rank in range(d_tp):
        rows = _records(tmp_path, "decode", rank)
        src = rank * p_tp // d_tp
        (load,) = _loads(rows)
        assert load[5] == src and load[0] == CHECKED_BYTES, load
        assert load[6][1] == 186, load
        accepted = [
            (scheduled, kept)
            for row in rows
            for scheduled, kept in row.get("spec_acceptance", {})
            .get("requests", {})
            .values()
        ]
        assert any(scheduled == kept == 1 for scheduled, kept in accepted), accepted
        if not check_bytes:
            assert not _checks(rows)
            continue
        (check,) = _checks(rows)
        assert check["bytes"] == CHECKED_BYTES
        assert check["layers"] == 62
        if p_tp == d_tp:
            assert check["ok"] and "coverage" not in check and "draft" not in check
            continue
        assert check["coverage"]["problems"] == []
        (read,) = check["coverage"]["reads"]
        assert (read["rank"], read["completed"], read["descriptors"]) == (
            src,
            True,
            186,
        )
        damaged = corrupt and src == 0
        assert check["ok"] is not damaged
        assert check["mismatched_bytes"] == int(damaged)
        assert {m[0] for m in check["mismatched_blocks"]} == (
            {DRAFT} if damaged else set()
        )
        assert check["draft"] == {
            "layer": DRAFT,
            "checked_bytes": 3 * PAGE_BYTES,
            "read_bytes": 3 * PAGE_BYTES,
            "mismatched_bytes": int(damaged),
        }
    for rank in range(p_tp):
        rows = _records(tmp_path, "prefill", rank)
        assert any(
            handoff["remote_request_id"]
            in row.get("kv_events", {}).get("sends_done", [])
            for row in rows
        ), rows
        assert not any(
            row.get("kv_transfers", {}).get("lease", {}).get("expired") for row in rows
        )


@pytest.mark.parametrize("check", [True, False], ids=["checked", "unchecked"])
@pytest.mark.parametrize(
    "p_extra",
    [{"depth": 2}, {"enable_prefix_caching": True}, {"block_size": 64}],
    ids=["depth", "prefix", "page"],
)
def test_native_mla_mtp_refuses_unqualified_producer(tmp_path, check, p_extra):
    _run_peers(tmp_path, 2, 4, {"check": check, **p_extra}, {"check": check})
    log = (tmp_path / "decode.log").read_text()
    expected = (
        "ValueError: Simulated real KV transfers pair unequal TP sizes for plain "
        f"attention, {REFUSAL}, not local TP 4 and remote TP 2."
    )
    assert expected in log, log[-6000:]
    marker = "simulated/nixl/unequal-tp-mla-mtp/v1/k1/b32/prefix0/f1/consumer"
    for role, size in (("prefill", 2), ("decode", 4)):
        regions = json.loads((tmp_path / f"{role}.regions.json").read_text())
        assert len(regions) == size
        assert all((marker in reg["scope"]) == (role == "decode") for reg in regions)
    failed = []
    checks = []
    for rank in range(4):
        rows = _records(tmp_path, "decode", rank)
        assert not _loads(rows)
        failed.extend(
            req
            for row in rows
            for req in row.get("kv_events", {}).get("reads_failed", [])
        )
        checks.extend(_checks(rows))
    # A worker's handshake failure can abort the request before another
    # worker logs its own. What must hold on every rank is zero READs.
    assert failed
    assert bool(checks) == check
    assert all(not c["ok"] and c["error"] == "receive failed" for c in checks)


def _engine_config(tmp_path, *, model=V3, tp=2, **options):
    from vllm import EngineArgs

    return EngineArgs(
        **_engine_options(
            published_path(model), tp, False, tmp_path / "steps", check=False, **options
        )
    ).create_engine_config()


@pytest.mark.parametrize("tp,peer", [(2, 4), (4, 2)])
def test_local_mla_mtp_admission(tmp_path, tp, peer):
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
        SPECULATIVE_DCP_TRANSFERS,
        simulated_dcp_scope,
        simulated_tp_mismatch,
        unequal_tp_mtp_layout,
    )

    config = _engine_config(tmp_path, tp=tp)
    assert simulated_tp_mismatch(config, tp, peer, nixl_consumer=True) is None
    assert simulated_dcp_scope(config) == [
        SPECULATIVE_DCP_TRANSFERS,
        unequal_tp_mtp_layout(config).capability,
    ]
    assert REFUSAL in simulated_tp_mismatch(config, tp, peer)
    assert simulated_tp_mismatch(config, tp, tp, nixl_consumer=True) is None
    assert simulated_tp_mismatch(config, tp, 8, nixl_consumer=True) is None
    assert "one dividing the other" in simulated_tp_mismatch(
        config, tp, 3, nixl_consumer=True
    )
    assert SPECULATIVE_DCP_TRANSFERS in simulated_tp_mismatch(
        config, tp, peer, nixl_consumer=True, remote_dcp_size=2
    )


@pytest.mark.parametrize(
    "options",
    [
        {"enable_prefix_caching": True},
        {"block_size": 64},
        {"depth": 2},
        {"kv_cache_dtype": "fp8"},
        {"dtype": "float16"},
        {"hf_overrides": {"kv_lora_rank": 256}},
        {"model": "deepseek-ai/DeepSeek-V3.2", "block_size": 64},
        {"model": "deepseek-ai/DeepSeek-V4-Flash", "block_size": 256},
        {"model": "Qwen/Qwen3.5-27B", "language_model_only": True},
        {
            "kv_transfer_config": {
                "kv_connector": "MooncakeConnector",
                "kv_role": "kv_consumer",
                "kv_buffer_device": "cpu",
            }
        },
        {
            "kv_transfer_config": {
                "kv_connector": "NixlConnector",
                "kv_role": "kv_consumer",
                "kv_buffer_device": "cpu",
                "kv_connector_extra_config": {"enforce_handshake_compat": False},
            }
        },
    ],
    ids=[
        "prefix",
        "page",
        "depth",
        "fp8-kv",
        "fp16",
        "changed-latent",
        "sparse",
        "compressed",
        "hybrid",
        "mooncake",
        "hash-off",
    ],
)
def test_mla_mtp_other_configurations_keep_their_refusal(
    tmp_path, ds_conv_layout, options
):
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
        simulated_tp_mismatch,
        unequal_tp_mtp_layout,
    )

    config = _engine_config(tmp_path, **options)
    if options in ({"enable_prefix_caching": True}, {"block_size": 64}, {"depth": 2}):
        assert unequal_tp_mtp_layout(config) is not None
        assert simulated_tp_mismatch(config, 2, 4, nixl_consumer=True) is None
        return
    assert unequal_tp_mtp_layout(config) is None
    n = options.get("kv_transfer_config", {}).get("kv_connector") != "MooncakeConnector"
    message = simulated_tp_mismatch(config, 2, 4, nixl_consumer=n)
    assert (
        "not local TP 2 and remote TP 4: speculative decoding." in message
        if options.get("model") == "Qwen/Qwen3.5-27B"
        else REFUSAL in message
    )
    assert simulated_tp_mismatch(config, 2, 2, nixl_consumer=n) is None


@pytest.mark.parametrize(
    "mode,problem",
    [
        ("clean", None),
        ("missing-draft", "nothing from producer 0 for region 61"),
        ("duplicate-draft", "duplicated or surplus bytes from producer 0 in region 61"),
        ("lookahead", "unexpected delivery from producer 0 to region 61 block 4"),
        ("wrong-producer", "unexpected delivery from producer 1"),
        ("wrong-engine", "different producer engine"),
        ("extra-read", "Expected one MLA MTP READ, got 2"),
        ("incomplete", "had not completed"),
    ],
)
def test_mla_mtp_coverage_checks_the_draft_and_source(mode, problem):
    from dataclasses import replace

    from vllm.v1.worker.simulated.kv_pattern import mtp_read_coverage
    from vllm.v1.worker.simulated.state_layout import ReadRecord

    target = (0, 1, 0, PAGE_BYTES)
    draft = (61, 1, 0, PAGE_BYTES)
    read = ReadRecord("req", 0, "read", [target, draft], done=True, engine="producer")
    records = {
        "clean": [read],
        "missing-draft": [replace(read, pieces=[target])],
        "duplicate-draft": [replace(read, pieces=[target, draft, draft])],
        "lookahead": [replace(read, pieces=[target, draft, (61, 4, 0, PAGE_BYTES)])],
        "wrong-producer": [replace(read, rank=1)],
        "wrong-engine": [replace(read, engine="not-producer")],
        "extra-read": [replace(read, pieces=[target]), replace(read, pieces=[draft])],
        "incomplete": [replace(read, done=False)],
    }[mode]
    actual = mtp_read_coverage({0: [target, draft]}, records, "producer")
    if problem is None:
        assert actual["problems"] == []
        assert actual["reads"] == [
            {"rank": 0, "completed": True, "descriptors": 2, "bytes": 2 * PAGE_BYTES}
        ]
    else:
        assert any(problem in reason for reason in actual["problems"]), actual


def test_mla_mtp_refuses_a_different_local_draft_tp(tmp_path):
    with pytest.raises(ValueError, match="not draft tensor parallelism 1"):
        _engine_config(tmp_path, spec_overrides={"draft_tensor_parallel_size": 1})


SPEC_GRID_PROFILES = {
    V3: (61, 1152, 0, (16, 32, 64, 128)),
    "deepseek-ai/DeepSeek-V3.2": (61, 1152, 132, (64, 128)),
    "nvidia/GLM-5-NVFP4": (78, 656, 132, (64, 128)),
}


def _spec_pair_options(cell, producer, path, **overrides):
    target, p_tp, d_tp, depth, block, prefix = cell
    extra = dict(
        depth=depth,
        block_size=block,
        enable_prefix_caching=prefix,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        kv_cache_memory_bytes=128 << 20,
        seed=1234,
    )
    extra.update(overrides)
    if target == V3:
        return _engine_options(
            published_path(target), p_tp if producer else d_tp, producer, path, **extra
        )
    from tests.v1.worker.simulated.test_sparse_mtp_unequal_tp import _options

    return _options(
        target,
        published_path(target),
        p_tp if producer else d_tp,
        producer,
        path,
        **extra,
    )


def _native_layout_oracle(config, target, block):
    from dataclasses import replace

    import torch

    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.mla.cpu_mla import CPUMLABackend
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
    from vllm.v1.attention.backends.utils import (
        get_supported_kv_cache_layouts,
        resolve_kv_cache_layout,
    )
    from vllm.v1.core.kv_cache_planning import (
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )
    from vllm.v1.kv_cache_interface import (
        MLAAttentionSpec,
        SparseCacheRole,
        get_kv_quant_mode,
    )
    from vllm.v1.worker.utils import select_common_block_size

    layers, latent, indexer, _ = SPEC_GRID_PROFILES[target]
    hf = config.model_config.hf_text_config
    assert hf.num_hidden_layers == layers
    width = hf.kv_lora_rank + hf.qk_rope_head_dim
    packed_bytes = (
        hf.kv_lora_rank + 2 * hf.qk_rope_head_dim + 4 * (hf.kv_lora_rank // 128)
    )
    index_width = (
        (hf.index_head_dim + 4 * (hf.index_head_dim // 128)) if indexer else 132
    )
    latent_spec = MLAAttentionSpec(
        block_size=block,
        num_kv_heads=1,
        head_size=width,
        dtype=torch.uint8 if latent == 656 else torch.bfloat16,
        state_content_bytes=packed_bytes if latent == 656 else None,
        max_tp_shards=1,
        cache_dtype_str=config.cache_config.cache_dtype,
        kv_quant_mode=get_kv_quant_mode(config.cache_config.cache_dtype),
        topk_buffer_width=hf.index_topk if indexer else None,
        is_index_group_leader=bool(indexer),
    )
    assert latent_spec.page_size_bytes == block * latent
    index_spec = MLAAttentionSpec(
        block_size=block,
        num_kv_heads=1,
        head_size=index_width,
        dtype=torch.uint8,
        max_tp_shards=1,
        cache_role=SparseCacheRole.INDEXER,
    )
    specs = {}
    for layer in range(layers + 1):
        name = f"model.layers.{layer}.self_attn"
        specs[f"{name}.attn"] = replace(latent_spec, is_draft=layer == layers)
        if indexer:
            specs[f"{name}.indexer.k_cache"] = replace(
                index_spec, is_draft=layer == layers
            )
    groups = get_kv_cache_groups(config, specs)
    assert len(groups) == 1
    assert set(groups[0].layer_names) == set(specs)
    assert specs[f"model.layers.{layers}.self_attn.attn"] == latent_spec
    backends = [CPUMLABackend]
    if indexer:
        backends.append(DeepseekV32IndexerBackend)
    with set_current_vllm_config(config):
        kernel = select_common_block_size(block, backends)
    assert kernel == (64 if indexer else block)
    resolve_kv_cache_layout(
        config,
        [[x.name for x in get_supported_kv_cache_layouts(backends)]],
        specs.values(),
    )
    plan = get_kv_cache_config_from_groups(config, groups, 128 << 20)
    assert plan.num_blocks >= -(-1024 // block) + 1
    assert plan.num_blocks == (128 << 20) // ((layers + 1) * block * (latent + indexer))
    return specs, groups, backends, kernel


def _direct_sdk_oracle(config, provider, direct_models, requests):
    import math

    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    from vllm.v1.worker.simulated.aisimulate_timing import rank_metrics
    from vllm.v1.worker.simulated.timing import StepDescriptor, parallel_summary

    key = json.dumps(provider.config, sort_keys=True)
    if key not in direct_models:
        direct_models[key] = RustForwardPassPerfModel.best_available(
            ForwardPassPerfModelConfig(**provider.config)
        )
    direct = direct_models[key]
    result = []
    for req in requests:
        step = StepDescriptor(
            0,
            (req,),
            req.num_scheduled_tokens if req.is_prefill else 0,
            0 if req.is_prefill else req.num_scheduled_tokens,
            int(req.is_prefill),
            int(not req.is_prefill),
            0,
            req.num_computed_tokens + req.num_scheduled_tokens,
            parallel_summary(config),
        )
        ms = direct.estimate_forward_pass_time_ms(rank_metrics([req]))
        assert math.isfinite(ms) and ms > 0
        assert provider.estimate(step).device_s * 1000 == pytest.approx(ms, rel=1e-9)
        result.append(ms)
    return result


def test_spec_pair_grid(tmp_path):
    """All 960 rule rows reconcile native layouts, rank maps and direct SDK cost."""
    from dataclasses import replace
    from itertools import product

    from tests.v1.worker.simulated.test_real_kv_transfer import agent_metadata
    from vllm import EngineArgs
    from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
        compute_tp_mapping,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
        simulated_tp_mismatch,
        unequal_tp_mtp_layout,
    )
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.simulated.aisimulate_timing import AisimulateTimingProvider
    from vllm.v1.worker.simulated.kv_pattern import mtp_read_coverage
    from vllm.v1.worker.simulated.state_layout import ReadRecord
    from vllm.v1.worker.simulated.timing import SimRequestStep

    probe = EngineArgs(
        **_spec_pair_options((V3, 1, 2, 1, 16, False), True, tmp_path / "probe")
    ).create_engine_config()
    assert unequal_tp_mtp_layout(probe) is not None, (
        "The native TP1/B16 MTP endpoint has no simulator layout"
    )
    from vllm.v1.worker.simulated.kv_transfer import (
        _matches_mtp_peer,
        _mtp_split_peer_matches,
    )

    endpoints, direct_models, verdicts = {}, {}, []
    endpoint_evidence = []
    maps = audits = 0
    for target, (layers, latent, indexer, blocks) in SPEC_GRID_PROFILES.items():
        for p, d, k, block, prefix in product(
            (1, 2, 4, 8), (1, 2, 4, 8), range(1, 6), blocks, (False, True)
        ):
            if p == d:
                continue
            cell = (target, p, d, k, block, prefix)
            pair = []
            for producer, tp in ((True, p), (False, d)):
                key = (target, tp, k, block, prefix, producer)
                if key not in endpoints:
                    options = _spec_pair_options(cell, producer, tmp_path / "steps")
                    cfg = EngineArgs(**options).create_engine_config()
                    layout = unequal_tp_mtp_layout(cfg)
                    assert layout is not None, ("MTP_LAYOUT_MISMATCH", key)
                    specs, groups, backends, kernel = _native_layout_oracle(
                        cfg, target, block
                    )
                    provider = AisimulateTimingProvider(
                        cfg, **options["simulation_config"]["timing_args"]
                    )
                    assumptions = provider.assumptions()["aisimulate"]
                    assert assumptions["source"] == "aic"
                    assert assumptions["readiness"] == "ready"
                    queries = [
                        SimRequestStep(
                            "cold", 2 * block + 7, 0, 2 * block + 7, 0, 0, True
                        ),
                        SimRequestStep(
                            "cached",
                            2 * block + 7,
                            3 * block,
                            5 * block + 7,
                            3 * block,
                            0,
                            True,
                        ),
                        SimRequestStep(
                            "verify", k + 1, 7 * block, 3 * block, 0, k, False
                        ),
                    ]
                    prices = _direct_sdk_oracle(cfg, provider, direct_models, queries)
                    endpoints[key] = (cfg, layout, specs, backends, kernel, prices)
                    endpoint_evidence.append(
                        dict(
                            identity=key,
                            scope=layout.capability,
                            prices_ms=prices,
                            assumptions=assumptions,
                            planner_blocks=(128 << 20)
                            // ((layers + 1) * block * (latent + indexer)),
                        )
                    )
                pair.append(endpoints[key])
            pc, pl, _, _, _, pp = pair[0]
            dc, dl, specs, backends, kernel, dp = pair[1]
            assert simulated_tp_mismatch(dc, d, p, nixl_consumer=True) is None
            assert dl.peer_capability == pl.capability
            lengths = [specs[n].page_size_bytes // (block // kernel) for n in dl.names]
            assert dl.page_bytes == lengths
            peer = agent_metadata(
                block_size=kernel,
                physical_blocks_per_logical_kv_block=block // kernel,
                region_names=dl.names,
                block_lens=lengths,
                block_strides=lengths,
                region_group_ids=[0] * len(lengths),
                simulated_scope=[pl.capability],
            )
            assert _matches_mtp_peer(dl, peer)
            assert not _matches_mtp_peer(
                dl, replace(peer, simulated_scope=[dl.capability])
            )
            assert not _matches_mtp_peer(dl, replace(peer, simulated_scope=[]))
            assert not _matches_mtp_peer(
                dl, replace(peer, block_strides=[0] * len(lengths))
            )
            local_regions = (dl.names, lengths, [0] * len(lengths), lengths)
            assert _mtp_split_peer_matches(
                dc, (kernel, block // kernel), p, peer, local_regions
            ) == bool(indexer and block == 128)
            for damaged in (
                replace(peer, simulated_scope=[]),
                replace(peer, simulated_scope=[dl.capability]),
                replace(peer, simulated_scope=[pl.capability, dl.capability]),
                replace(
                    peer,
                    simulated_scope=[
                        pl.capability,
                        "simulated/nixl/unequal-tp-unrecognized",
                    ],
                ),
                replace(peer, pcp_size=2),
                replace(peer, dcp_size=2),
                replace(peer, physical_blocks_per_logical_kv_block=3),
                replace(peer, block_strides=[0] * len(lengths)),
                replace(peer, region_group_ids=[1] * len(lengths)),
                replace(
                    peer,
                    simulated_scope=[
                        pl.capability.replace(f"/k{k}/", f"/k{k % 5 + 1}/")
                    ],
                ),
                replace(
                    peer,
                    simulated_scope=[
                        pl.capability.replace(
                            f"/prefix{int(prefix)}/", f"/prefix{int(not prefix)}/"
                        )
                    ],
                ),
            ):
                assert not _matches_mtp_peer(dl, damaged)
                assert not _mtp_split_peer_matches(
                    dc, (kernel, block // kernel), p, damaged, local_regions
                )
            assert not _mtp_split_peer_matches(
                dc, (kernel, block // kernel), d, peer, local_regions
            )
            assert not _mtp_split_peer_matches(dc, (kernel, 3), p, peer, local_regions)
            for local_damage in (
                (list(reversed(dl.names)), lengths, [0] * len(lengths), lengths),
                (dl.names, [0] * len(lengths), [0] * len(lengths), lengths),
                (dl.names, lengths, [1] * len(lengths), lengths),
                (dl.names, lengths, [0] * len(lengths), [0] * len(lengths)),
            ):
                assert not _mtp_split_peer_matches(
                    dc, (kernel, block // kernel), p, peer, local_damage
                )
            ranks = []
            for rank in range(d):
                topo = TransferTopology(
                    tp_rank=rank,
                    tp_size=d,
                    block_size=kernel,
                    engine_id="D",
                    is_mla=True,
                    is_mamba=False,
                    total_num_kv_heads=1,
                    attn_backends=backends,
                )
                native = compute_tp_mapping(topo, p, (MLAAttentionSpec,))
                source = rank * p // d
                assert native.source_ranks_per_group == ((source,),)
                assert native.all_source_ranks == (source,)
                assert native.local_consumers == max(1, d // p)
                maps += 1
                requests = []
                for first, count in [(0, 6), (3, 3)] if prefix else [(0, 3)]:
                    factor = block // kernel
                    wanted = [
                        (
                            region,
                            page,
                            0,
                            block
                            * (latent if region % 2 == 0 or not indexer else indexer),
                        )
                        for region in range(len(dl.names))
                        for page in range(first, first + count)
                    ]
                    physical = [
                        (region, page, half * size // factor, size // factor)
                        for region, page, _, size in wanted
                        for half in range(factor)
                    ]
                    record = ReadRecord(
                        "r", source, "read", physical, done=True, engine="P"
                    )
                    audit = mtp_read_coverage({source: wanted}, [record], "P")
                    assert not audit["problems"], audit
                    size = count * block * (latent + indexer) * (layers + 1)
                    assert sum(x[3] for x in physical) == size
                    assert len(physical) == count * factor * len(dl.names)
                    for mutation in (
                        replace(record, pieces=physical[:-1]),
                        replace(record, pieces=physical + [physical[-1]]),
                        replace(record, rank=p),
                        replace(record, engine="foreign"),
                        replace(record, done=False),
                    ):
                        assert mtp_read_coverage({source: wanted}, [mutation], "P")[
                            "problems"
                        ]
                    audits += 1
                    requests.append(
                        dict(
                            first=first,
                            pages=count,
                            bytes=size,
                            descriptors=len(physical),
                            draft_latent=count * block * latent,
                            draft_indexer=count * block * indexer,
                        )
                    )
                ranks.append(dict(rank=rank, source=source, requests=requests))
            verdicts.append(
                dict(
                    cell=cell,
                    verdict="admit",
                    kernel=kernel,
                    factor=block // kernel,
                    names=dl.names,
                    lengths=lengths,
                    capabilities=[pl.capability, dl.capability],
                    ranks=ranks,
                    endpoint_prices_ms=[pp, dp],
                )
            )
    assert (len(verdicts), len(endpoints), maps, audits) == (960, 640, 3600, 5400)
    result = dict(
        rows=verdicts,
        endpoint_count=len(endpoints),
        mappings=maps,
        audits=audits,
        endpoints=endpoint_evidence,
    )
    (tmp_path / "grid-verdicts.json").write_text(json.dumps(result))


SPEC_ENGINE_CASES = [
    (V3, 1, 8, 1, 16, False, "clean"),
    (V3, 8, 1, 5, 128, True, "witness"),
    (V3, 8, 1, 5, 128, True, "cached-corrupt"),
    (V3, 4, 2, 2, 32, False, "latent-corrupt"),
    (V3, 2, 4, 4, 64, True, "clean"),
    ("deepseek-ai/DeepSeek-V3.2", 1, 8, 3, 128, True, "witness"),
    ("deepseek-ai/DeepSeek-V3.2", 1, 8, 3, 128, True, "indexer-corrupt"),
    ("deepseek-ai/DeepSeek-V3.2", 8, 1, 5, 64, False, "clean"),
    ("nvidia/GLM-5-NVFP4", 1, 8, 2, 128, True, "clean"),
    ("nvidia/GLM-5-NVFP4", 8, 1, 4, 64, False, "latent-corrupt"),
]


def _make_small_spec_model(target, directory):
    import hashlib
    import shutil

    source = published_path(target)
    directory.mkdir()
    for original in source.glob("*.json"):
        shutil.copyfile(original, directory / original.name)
    path = directory / "config.json"
    raw = json.loads(path.read_text())
    raw.update(
        hidden_size=256,
        intermediate_size=1024,
        moe_intermediate_size=1024,
        n_routed_experts=8,
        n_group=1,
        topk_group=1,
        q_lora_rank=128,
    )
    path.write_text(json.dumps(raw))
    (directory / "DERIVED.md").write_text(
        f"Compute-small fixture of {target}; published KV geometry is unchanged.\n"
        "The explicit published timing target estimates published compute.\n"
    )
    (directory / "fixture-sha256.json").write_text(
        json.dumps(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.glob("*.json")
            }
        )
    )
    return directory


def _inspect_spec_pair(worker, corrupt=None):
    import hashlib

    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
    from vllm.v1.kv_cache_interface import group_kernel_blocks
    from vllm.v1.worker.simulated.kv_pattern import block_views

    native = get_kv_transfer_group().connector_worker
    if corrupt is not None:
        layer, block = corrupt
        if get_tensor_model_parallel_rank() == 0:
            cache = native.device_kv_caches[layer]
            count = worker.model_runner.kv_cache_config.num_blocks
            grouped = group_kernel_blocks(cache, count)
            view = block_views({layer: grouped}, count)[layer]
            view[block, 0] ^= 1
        return None
    cfg = worker.vllm_config
    runner = worker.model_runner
    return {
        **_inspect(worker),
        "strides": native.block_stride_per_layer,
        "dtypes": {
            name: str(cache.dtype) for name, cache in native.device_kv_caches.items()
        },
        "num_blocks": runner.kv_cache_config.num_blocks,
        "target_tp": cfg.parallel_config.tensor_parallel_size,
        "draft_tp": cfg.speculative_config.draft_parallel_config.tensor_parallel_size,
        "depth": cfg.speculative_config.num_speculative_tokens,
        "prefix": cfg.cache_config.enable_prefix_caching,
        "timing": runner.timing_provider.assumptions()["aisimulate"],
        "fixture_sha256": hashlib.sha256(
            Path(cfg.model_config.model, "config.json").read_bytes()
        ).hexdigest(),
    }


def _spec_pair_peer():
    from vllm import LLM, SamplingParams

    raw_case, role, model, dest = sys.argv[1:]
    case = json.loads(raw_case)
    target, p_tp, d_tp, k, block, prefix, mode = case
    producer = role == "prefill"
    witness = mode in ("witness", "cached-corrupt")
    directory = Path(dest)
    options = _spec_pair_options(
        case[:6], producer, directory / role, seed=0 if witness else 1234
    )
    options["model"] = model
    options["simulation_config"]["spec_acceptance_rates"] = [0.7]
    engine = LLM(**options)
    (directory / f"{role}.regions.json").write_text(
        json.dumps(engine.collective_rpc(_inspect_spec_pair))
    )
    core = engine.llm_engine.engine_core.engine_core
    count = 2 if prefix else 1
    for index in range(count):
        if producer:
            if index:
                deadline = time.monotonic() + 180
                while not (directory / "prompt-1.json").exists():
                    assert time.monotonic() < deadline, "second prompt never arrived"
                    core.step()
                    time.sleep(0.02)
            prompt = json.loads((directory / f"prompt-{index}.json").read_text())
            handoff = {"do_remote_decode": True}
        else:
            deadline = time.monotonic() + 180
            payload = directory / f"prefill-{index}.json"
            while not payload.exists():
                assert time.monotonic() < deadline, "producer handoff never arrived"
                time.sleep(0.02)
            handoff = json.loads(payload.read_text())["handoff"]
            prompt = json.loads((directory / f"prompt-{index}.json").read_text())
        max_tokens = block + 16 if witness and not index else 32 if witness else 80
        params = SamplingParams(
            max_tokens=1 if producer else max_tokens,
            ignore_eos=True,
            detokenize=False,
            temperature=0,
            extra_args={"kv_transfer_params": handoff},
        )
        (response,) = engine.generate(
            [{"prompt_token_ids": prompt}], params, use_tqdm=False
        )
        tokens = list(response.outputs[0].token_ids)
        handoff = response.kv_transfer_params
        if producer and not index and mode in ("latent-corrupt", "indexer-corrupt"):
            layers = SPEC_GRID_PROFILES[target][0]
            layer = f"model.layers.{layers}.self_attn."
            layer += "indexer.k_cache" if mode == "indexer-corrupt" else "attn"
            engine.collective_rpc(
                _inspect_spec_pair, args=((layer, handoff["remote_block_ids"][0][-1]),)
            )
        temporary = directory / f"{role}-{index}.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "handoff": handoff,
                    "tokens": tokens,
                    "batch_queue_size": core.batch_queue_size,
                    "finish_reason": response.outputs[0].finish_reason,
                }
            )
        )
        temporary.rename(directory / f"{role}-{index}.json")
        if not producer and index == 0 and prefix:
            if witness:
                assert len(tokens) == block + 16
                second = prompt + tokens[: block + 2] + [7] * (block + 7)
                if mode == "cached-corrupt":
                    rows = _records(directory, "decode", 0)
                    writes = [row["kv_write"] for row in rows if "kv_write" in row]
                    crossing = [
                        w
                        for w in writes
                        if w["logical_block"] == 2 and w["phase"] == "completion"
                    ]
                    assert crossing, "no completion-write witness before corruption"
                    block_id = crossing[0]["block_id"]
                    layer = (
                        f"model.layers.{SPEC_GRID_PROFILES[target][0]}.self_attn.attn"
                    )
                    engine.collective_rpc(_inspect_spec_pair, args=((layer, block_id),))
            else:
                second = [5] * (4 * block + 8) + [7] * (block - 1)
            temporary = directory / "prompt-1.tmp"
            temporary.write_text(json.dumps(second))
            temporary.rename(directory / "prompt-1.json")
    if producer:
        deadline = time.monotonic() + 180
        while not (directory / "stop").exists():
            assert time.monotonic() < deadline, "consumer never stopped producer"
            core.step()
            time.sleep(0.02)


def _run_spec_pair(directory, case):
    target, p_tp, d_tp, _, block, prefix, mode = case
    model = _make_small_spec_model(target, directory / "model")
    witness = mode in ("witness", "cached-corrupt")
    n = 3 * block - 2 if witness else 5 * block + 7 if prefix else 3 * block - 1
    (directory / "prompt-0.json").write_text(json.dumps([5] * n))
    p_port, d_port = _free_bases(max(p_tp, d_tp))
    launch = directory / "peer.py"
    launch.write_text(
        "from tests.v1.worker.simulated.test_mtp_unequal_tp import _spec_pair_peer\n"
        "if __name__ == '__main__':\n    _spec_pair_peer()\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd()),
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "VLLM_HOST_IP": "127.0.0.1",
        "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
        "NIXL_MOONCAKE_IP_ADDR": "127.0.0.1",
    }

    def command(role):
        return [
            sys.executable,
            str(launch),
            json.dumps(case),
            role,
            str(model),
            str(directory),
        ]

    p_log, d_log = directory / "prefill.log", directory / "decode.log"
    with p_log.open("w") as output:
        producer = subprocess.Popen(
            command("prefill"),
            stdout=output,
            stderr=subprocess.STDOUT,
            env={**env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(p_port)},
        )
    try:
        deadline = time.monotonic() + 180
        while not (directory / "prefill-0.json").exists():
            assert producer.poll() is None, p_log.read_text()[-8000:]
            assert time.monotonic() < deadline, p_log.read_text()[-8000:]
            time.sleep(0.1)
        with d_log.open("w") as output:
            consumer = subprocess.run(
                command("decode"),
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=240,
                env={**env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(d_port)},
            )
        assert consumer.returncode == 0, d_log.read_text()[-8000:]
    finally:
        (directory / "stop").touch()
        try:
            producer.wait(timeout=30)
        except subprocess.TimeoutExpired:
            producer.kill()
            producer.wait(timeout=30)
            raise
    assert producer.returncode == 0, p_log.read_text()[-8000:]


def _assert_spec_pair(path, case):
    import math

    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    from vllm.v1.worker.simulated.aisimulate_timing import rank_metrics
    from vllm.v1.worker.simulated.timing import StepDescriptor

    target, p_tp, d_tp, k, block, prefix, mode = case
    layers, latent, indexer, _ = SPEC_GRID_PROFILES[target]
    witness = mode in ("witness", "cached-corrupt")
    pages = [3, 3] if witness else [6, 3] if prefix else [3]
    factor = block // 64 if indexer else 1
    region_count = (layers + 1) * (2 if indexer else 1)
    per_page = (layers + 1) * block * (latent + indexer)
    handoffs = [
        json.loads((path / f"prefill-{i}.json").read_text())["handoff"]
        for i in range(len(pages))
    ]
    for i, handoff in enumerate(handoffs):
        n = len(json.loads((path / f"prompt-{i}.json").read_text()))
        assert handoff["remote_num_tokens"] == n
        assert [len(x) for x in handoff["remote_block_ids"]] == [-(-n // block)]
        output = json.loads((path / f"decode-{i}.json").read_text())["tokens"]
        assert len(output) == (
            block + 16 if witness and not i else 32 if witness else 80
        )
    for role, tp in (("prefill", p_tp), ("decode", d_tp)):
        registrations = json.loads((path / f"{role}.regions.json").read_text())
        assert len(registrations) == tp
        for reg in registrations:
            assert reg["kernel"] == block // factor and reg["factor"] == factor
            lengths = [block * latent // factor]
            if indexer:
                lengths += [block * indexer // factor]
            assert reg["lengths"] == reg["strides"] == lengths * (layers + 1)
            assert len(reg["names"]) == region_count
            assert reg["groups"] == [0] * region_count
            assert reg["num_blocks"] >= -(-1024 // block) + 1
            assert reg["target_tp"] == reg["draft_tp"] == tp
            assert reg["depth"] == k and reg["prefix"] == prefix
            timing = reg["timing"]
            assert timing["source"] == "aic" and timing["readiness"] == "ready"
            assert timing["target_model"] == target
            assert timing["resolved_config"]["nextn"] == k
            assert timing["local_config"]["config_sha256"] == reg["fixture_sha256"]
    for rank in range(d_tp):
        rows = _records(path, "decode", rank)
        source = rank * p_tp // d_tp
        loads, checks = (
            _loads(rows),
            [c for c in _checks(rows) if c.get("kind") != "hit"],
        )
        assert len(loads) == len(checks) == len(pages), (loads, checks)
        for i, (load, check, moved) in enumerate(zip(loads, checks, pages)):
            assert (load[0], load[5], load[6][1]) == (
                moved * per_page,
                source,
                moved * factor * region_count,
            )
            assert check["coverage"]["problems"] == [], check
            assert load[6][2] == load[3]
            assert len(check["coverage"]["destination_pieces"]) == 1
            assert (
                len(check["coverage"]["destination_pieces"][0])
                == moved * factor * region_count
            )
            assert check["target"] == {
                "checked_bytes": moved * layers * block * (latent + indexer),
                "read_bytes": moved * layers * block * (latent + indexer),
                "mismatched_bytes": 0,
            }
            assert check["bytes"] == moved * per_page
            assert check["groups"][0]["first_logical_block"] == (3 if i else 0)
            corrupt = (
                source == 0 and i == 0 and mode in ("latent-corrupt", "indexer-corrupt")
            )
            assert check["ok"] == (not corrupt), check
            assert check["mismatched_bytes"] == int(corrupt)
            if corrupt:
                name = f"model.layers.{layers}.self_attn."
                name += "indexer.k_cache" if mode == "indexer-corrupt" else "attn"
                assert {entry[0] for entry in check["mismatched_blocks"]} == {name}
            draft = check["draft"]
            if indexer:
                assert draft["latent"]["checked_bytes"] == moved * block * latent
                assert draft["indexer"]["checked_bytes"] == moved * block * indexer
                assert draft["latent"]["read_bytes"] == moved * block * latent
                assert draft["indexer"]["read_bytes"] == moved * block * indexer
            else:
                assert (
                    draft["checked_bytes"]
                    == draft["read_bytes"]
                    == moved * block * latent
                )
        accepted = [
            pair
            for row in rows
            for pair in row.get("spec_acceptance", {}).get("requests", {}).values()
        ]
        assert any(a > 0 for _, a in accepted) and any(a < n for n, a in accepted)
        assert any(n == k for n, _ in accepted)
        steps = [row["step"] for row in rows if "step" in row]
        assert any(
            req["num_spec_tokens"] == k and req["num_scheduled_tokens"] == k + 1
            for step in steps
            for req in step["requests"]
            if not req["is_prefill"]
        )
        for i, check in enumerate(checks):
            req_id = check["req_id"]
            response = json.loads((path / f"decode-{i}.json").read_text())
            assert response["finish_reason"] == "length"
            decisions = {
                row["spec_acceptance"]["step_index"]: row["spec_acceptance"][
                    "requests"
                ].get(req_id, [0, 0])[1]
                for row in rows
                if "spec_acceptance" in row
            }
            made = [
                1 + decisions.get(step["step_index"], 0)
                for step in steps
                if any(req["req_id"] == req_id for req in step["requests"])
            ]
            emitted = len(response["tokens"])
            consumed = 0
            for terminal, count in enumerate(made):
                consumed += count
                if consumed >= emitted:
                    break
            else:
                pytest.fail(
                    f"{req_id}: acceptance stream cannot supply {emitted} outputs"
                )
            terminal_clipped = consumed - emitted
            assert 0 <= terminal_clipped < made[terminal] <= k + 1
            inflight = made[terminal + 1 :]
            assert len(inflight) < response["batch_queue_size"]
            assert sum(made) == emitted + terminal_clipped + sum(inflight)
            (path / f"reconcile-{rank}-{i}.json").write_text(
                json.dumps(
                    dict(
                        emitted=emitted,
                        terminal_clipped=terminal_clipped,
                        discarded_inflight=inflight,
                        queue_size=response["batch_queue_size"],
                        total_sampled=sum(made),
                    )
                )
            )
        hits = [c for c in _checks(rows) if c.get("kind") == "hit"]
        if prefix:
            assert hits and any(c["blocks"] == 3 for c in hits), hits
            assert all(not g["written_this_step"] for c in hits for g in c["groups"])
            if mode == "cached-corrupt":
                assert any(not c["ok"] and c["mismatched_bytes"] == 1 for c in hits)
            else:
                assert all(c["ok"] for c in hits), hits
        if witness:
            writes = [row["kv_write"] for row in rows if "kv_write" in row]
            assert any(
                w["logical_block"] == 2 and w["phase"] == "completion" for w in writes
            ), writes
        if witness:
            event = next(
                w
                for w in writes
                if w["logical_block"] == 2 and w["phase"] == "completion"
            )
            assert event["req_id"] == checks[0]["req_id"] and event["end"] >= 3 * block
            later = next(c for c in hits if c["req_id"] == checks[1]["req_id"])
            group = later["groups"][0]
            assert (
                group["block_ids"][group["logical_blocks"].index(2)]
                == event["block_id"]
            )
            decision = next(
                row["spec_acceptance"]
                for row in rows
                if row.get("spec_acceptance", {}).get("step_index")
                == event["step_index"]
            )
            assert decision["requests"][event["req_id"]][1] > 0
        timing = json.loads((path / "decode.regions.json").read_text())[rank]["timing"]
        config = {**timing["config"], "model": target}
        direct = RustForwardPassPerfModel.best_available(
            ForwardPassPerfModelConfig(**config)
        )
        for row in rows:
            if "step" not in row:
                continue
            step = StepDescriptor.from_json_dict(row["step"])
            ms = direct.estimate_forward_pass_time_ms(rank_metrics(list(step.requests)))
            assert math.isfinite(ms) and ms > 0
            assert row["cost"]["device_s"] * 1000 == pytest.approx(ms, rel=1e-9)
    for rank in range(p_tp):
        rows = _records(path, "prefill", rank)
        released = [
            req
            for row in rows
            for req in row.get("kv_events", {}).get("sends_done", [])
        ]
        assert sorted(released) == sorted(h["remote_request_id"] for h in handoffs)
        assert not any(
            row.get("kv_transfers", {}).get("lease", {}).get("expired") for row in rows
        )


@pytest.mark.parametrize("case", SPEC_ENGINE_CASES, ids=[f"T{i}" for i in range(1, 11)])
def test_spec_pair_engine_mechanisms(tmp_path, case):
    """Real peers cover native rule mechanisms, split pages and draft corruption."""
    _run_spec_pair(tmp_path, case)
    _assert_spec_pair(tmp_path, case)
