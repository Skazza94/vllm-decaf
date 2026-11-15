# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.v1.worker.simulated.test_kernel_geometry import published_path
from tests.v1.worker.simulated.test_mtp_unequal_tp import (
    REFUSAL,
    _checks,
    _engine_options,
    _free_bases,
    _inspect,
    _loads,
    _records,
)

V32 = "deepseek-ai/DeepSeek-V3.2"
GLM = "nvidia/GLM-5-NVFP4"
PROFILES = {
    V32: ("h200", "bfloat16", "fp8_block", 61, 73_728, "v32"),
    GLM: ("b200", "fp8_ds_mla", "nvfp4", 78, 41_984, "glm5"),
}
INDEXER_BYTES = 8_448


def _options(target, model, tp, producer, log, **options):
    gpu, cache, _, _, _, _ = PROFILES[target]
    timing = options.pop("timing_args", {})
    options.setdefault("block_size", 64)
    options.setdefault("kv_cache_dtype", cache)
    kwargs = _engine_options(model, tp, producer, log, **options)
    kwargs["simulation_config"]["timing_args"] = {
        "gpu": gpu,
        "target_model": target,
        **timing,
    }
    return kwargs


def _region_names(layer):
    prefix = f"model.layers.{layer}.self_attn"
    return f"{prefix}.attn", f"{prefix}.indexer.k_cache"


def _inspect_profile(worker):
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from vllm.v1.worker.simulated.timing import (
        SimRequestStep,
        StepDescriptor,
        parallel_summary,
    )

    metadata = _inspect(worker)
    native = get_kv_transfer_group().connector_worker
    runner = worker.model_runner
    provider = runner.timing_provider
    req = SimRequestStep("qualification", 2, 145, 145, 0, 1, False)
    step = StepDescriptor(
        0, (req,), 0, 2, 0, 1, 0, 147, parallel_summary(worker.vllm_config)
    )
    model = worker.vllm_config.model_config
    draft = worker.vllm_config.speculative_config.draft_model_config
    return {
        **metadata,
        "dtypes": {
            name: str(cache.dtype) for name, cache in native.device_kv_caches.items()
        },
        "strides": native.block_stride_per_layer,
        "weight_methods": [model.quantization, draft.quantization],
        "cache_dtypes": [
            worker.vllm_config.cache_config.cache_dtype,
            worker.vllm_config.speculative_config.kv_cache_dtype,
        ],
        "width_two_ms": provider.estimate(step).device_s * 1000,
        "timing": provider.assumptions()["aisimulate"],
    }


def _flip_draft(worker, layer, block):
    import torch

    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

    if get_tensor_model_parallel_rank() == 0:
        cache = get_kv_transfer_group().connector_worker.device_kv_caches[layer]
        cache[block].view(-1).view(torch.uint8)[0] ^= 1


def _peer():
    from vllm import LLM, SamplingParams

    target, role, model, tp, raw_options, path = sys.argv[1:]
    producer = role == "prefill"
    dest = Path(path)
    options = json.loads(raw_options)
    corrupt = options.pop("corrupt", None)
    engine = LLM(**_options(target, model, int(tp), producer, dest / role, **options))
    (dest / f"{role}.regions.json").write_text(
        json.dumps(engine.collective_rpc(_inspect_profile))
    )
    handoff = (
        {"do_remote_decode": True}
        if producer
        else json.loads((dest / "prefill.json").read_text())["handoff"]
    )
    params = SamplingParams(
        max_tokens=1 if producer else 12,
        ignore_eos=True,
        detokenize=False,
        temperature=0,
        extra_args={"kv_transfer_params": handoff},
    )
    (response,) = engine.generate(
        [{"prompt_token_ids": [5] * 145}], params, use_tqdm=False
    )
    handoff = response.kv_transfer_params
    if corrupt:
        block = handoff["remote_block_ids"][0][0]
        engine.collective_rpc(_flip_draft, args=(corrupt, block))
    temporary = dest / f"{role}.tmp"
    temporary.write_text(
        json.dumps({"handoff": handoff, "tokens": list(response.outputs[0].token_ids)})
    )
    temporary.rename(dest / f"{role}.json")
    if producer:
        core = engine.llm_engine.engine_core.engine_core
        deadline = time.monotonic() + 180
        while not (dest / "stop").exists() and time.monotonic() < deadline:
            core.step()
            time.sleep(0.02)


def _peers(directory, target, p_tp, d_tp, p_options=None, d_options=None):
    model = published_path(target)
    p_port, d_port = _free_bases(max(p_tp, d_tp))
    launch = directory / "peer.py"
    launch.write_text(
        "from tests.v1.worker.simulated.test_sparse_mtp_unequal_tp import _peer\n"
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

    def args(role, tp, options):
        return [
            sys.executable,
            str(launch),
            target,
            role,
            str(model),
            str(tp),
            json.dumps(options or {}),
            str(directory),
        ]

    p_log = directory / "prefill.log"
    with p_log.open("w") as output:
        p = subprocess.Popen(
            args("prefill", p_tp, p_options),
            stdout=output,
            stderr=subprocess.STDOUT,
            env={**env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(p_port)},
        )
    try:
        deadline = time.monotonic() + 120
        while not (directory / "prefill.json").exists():
            assert p.poll() is None, p_log.read_text()[-6000:]
            assert time.monotonic() < deadline, p_log.read_text()[-6000:]
            time.sleep(0.10)
        with (directory / "decode.log").open("w") as output:
            return subprocess.run(
                args("decode", d_tp, d_options),
                stdout=output,
                stderr=subprocess.STDOUT,
                env={**env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(d_port)},
                timeout=120,
            ).returncode
    finally:
        (directory / "stop").touch()
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=20)
            raise


def _registrations(path, target, role, tp):
    gpu, cache, weight, layers, latent_bytes, _ = PROFILES[target]
    rows = json.loads((path / f"{role}.regions.json").read_text())
    assert len(rows) == tp
    names = [n for layer in range(layers + 1) for n in _region_names(layer)]
    sizes = [latent_bytes, INDEXER_BYTES] * (layers + 1)
    assert all((row["kernel"], row["factor"]) == (64, 1) for row in rows)
    for row in rows:
        assert row["names"] == names, row["names"]
        assert row["lengths"] == row["strides"] == sizes
        assert row["groups"] == [0] * len(names)
        dtypes = [
            "torch.bfloat16" if cache == "bfloat16" else "torch.uint8",
            "torch.uint8",
        ] * (layers + 1)
        assert [row["dtypes"][n] for n in names] == dtypes
        timing = row["timing"]
        assert timing["readiness"] == "ready"
        assert timing["source"] != "fallback_regression"
        assert timing["kv_cache_format"] == cache
        resolved = timing["resolved_config"]
        expected = {
            "model": target,
            "system": f"{gpu}_sxm",
            "tp": tp,
            "moe_tp_size": tp,
            "worker_type": role,
            "gemm_quant_mode": weight,
            "moe_quant_mode": weight,
            "kvcache_quant_mode": "fp8" if cache == "fp8_ds_mla" else cache,
            "nextn": 1,
            "fallback_policy": "deny",
            "estimation_mode": "op_level",
        }
        assert {key: resolved[key] for key in expected} == expected
        assert {probe["kind"] for probe in timing["startup_probes"]} == {
            "prefill",
            "decode",
        }
        assert all(
            probe["ms"] > 0 and math.isfinite(probe["ms"])
            for probe in timing["startup_probes"]
        )
        assert row["width_two_ms"] > 0 and math.isfinite(row["width_two_ms"])
    return rows


def _marker(target):
    suffix = PROFILES[target][-1]
    return f"simulated/nixl/unequal-tp-dsa-{suffix}-mtp/v1"


def _assert_transferred(path, target, p_tp, d_tp, code, *, checked=True, corrupt=None):
    for role, count in (("prefill", p_tp), ("decode", d_tp)):
        for row in _registrations(path, target, role, count):
            assert _marker(target) in row["scope"]
    log = (path / "decode.log").read_text()
    assert REFUSAL not in log, log[-6000:]
    assert code == 0, log[-6000:]
    response = json.loads((path / "decode.json").read_text())
    assert len(response["tokens"]) == 12
    handoff = json.loads((path / "prefill.json").read_text())["handoff"]
    assert handoff["remote_num_tokens"] in (144, 145), handoff
    assert [len(ids) for ids in handoff["remote_block_ids"]] == [3]
    layers, latent = PROFILES[target][3:5]
    size = 3 * (layers + 1) * (latent + INDEXER_BYTES)
    selected = 6 * (layers + 1)
    draft_names = _region_names(layers)
    for d in range(d_tp):
        rows = _records(path, "decode", d)
        source = d * p_tp // d_tp
        (load,) = _loads(rows)
        assert load[0] == size and load[5] == source, load
        assert load[6][1] == selected and load[6][2] == load[3], load
        safety = [r["kv_events"] for r in rows if "kv_events" in r]
        assert not any(row.get("reads_failed") for row in safety)
        [(ready, request)] = [
            (i, event["reads_done"][0])
            for i, row in enumerate(rows)
            if (event := row.get("kv_events", {})).get("reads_done")
        ]
        first, step = next(
            (i, req)
            for i, row in enumerate(rows)
            for req in row.get("step", {}).get("requests", [])
        )
        assert ready < first and step["req_id"] == request
        assert step["num_cached_tokens"] == step["num_computed_tokens"] == 144
        accepted = [
            (drafted, kept)
            for row in rows
            for drafted, kept in row.get("spec_acceptance", {})
            .get("requests", {})
            .values()
        ]
        assert accepted and all(pair == (1, 1) for pair in accepted)
        if not checked:
            assert not _checks(rows)
            continue
        (check,) = _checks(rows)
        assert check["bytes"] == size and check["layers"] == 2 * (layers + 1)
        assert check["coverage"]["problem_count"] == 0, check
        assert check["coverage"]["reads"] == [
            {"rank": source, "completed": True, "descriptors": selected, "bytes": size}
        ]
        damaged = corrupt if source == 0 else None
        assert bool(check["ok"]) == (damaged is None)
        assert check["mismatched_bytes"] == int(damaged is not None)
        assert {entry[0] for entry in check["mismatched_blocks"]} == (
            {damaged} if damaged else set()
        )
        for kind, name, length in zip(
            ("latent", "indexer"), draft_names, (latent, INDEXER_BYTES)
        ):
            assert check["draft"][kind] == {
                "layer": name,
                "checked_bytes": 3 * length,
                "read_bytes": 3 * length,
                "mismatched_bytes": int(name == damaged),
            }
    for p in range(p_tp):
        rows = _records(path, "prefill", p)
        released = [
            req
            for row in rows
            for req in row.get("kv_events", {}).get("sends_done", [])
        ]
        assert released == [handoff["remote_request_id"]], released
        for row in rows:
            lease = row.get("kv_transfers", {}).get("lease", {})
            assert not (lease.get("expired") or lease.get("late_notifications"))


@pytest.mark.parametrize("target", PROFILES, ids=["v32", "glm"])
@pytest.mark.parametrize("p_tp,d_tp", [(4, 2), (2, 4)], ids=["gather", "scatter"])
def test_sparse_mtp_moves_unequal_tp(tmp_path, target, p_tp, d_tp):
    code = _peers(tmp_path, target, p_tp, d_tp)
    _assert_transferred(tmp_path, target, p_tp, d_tp, code)


@pytest.mark.parametrize(
    "target,kind",
    [(V32, "indexer"), (GLM, "latent")],
    ids=["indexer", "packed-latent"],
)
def test_sparse_mtp_detects_corrupt_draft(tmp_path, target, kind):
    names = dict(zip(("latent", "indexer"), _region_names(PROFILES[target][3])))
    corrupt = names[kind]
    code = _peers(tmp_path, target, 2, 4, {"corrupt": corrupt})
    _assert_transferred(tmp_path, target, 2, 4, code, corrupt=corrupt)


def test_sparse_mtp_admits_without_byte_checks(tmp_path):
    code = _peers(tmp_path, GLM, 4, 2, {"check": False}, {"check": False})
    _assert_transferred(tmp_path, GLM, 4, 2, code, checked=False)


@pytest.mark.parametrize("checked", [True, False], ids=["checked", "unchecked"])
@pytest.mark.parametrize(
    "target,change",
    [
        (V32, {"depth": 2}),
        (V32, {"enable_prefix_caching": True}),
        (V32, {"block_size": 128}),
        (GLM, {"enable_prefix_caching": True}),
    ],
    ids=["v32-depth", "v32-prefix", "v32-page", "glm-prefix"],
)
def test_sparse_mtp_refuses_unqualified_producer(tmp_path, target, change, checked):
    _peers(tmp_path, target, 2, 4, {**change, "check": checked}, {"check": checked})
    log = (tmp_path / "decode.log").read_text()
    if "block_size" in change:
        expected = (
            "ValueError: Simulated NIXL from a peer whose blocks hold several "
            "kernel blocks (2 of 64 tokens) needs this instance to declare its "
            "kernel geometry too (simulation_config.kv_kernel_backend)."
        )
    else:
        expected = (
            "ValueError: Simulated real KV transfers pair unequal TP sizes for plain "
            f"attention, {REFUSAL}, not local TP 4 and remote TP 2."
        )
    assert expected in log, log[-6000:]
    for role in ("prefill", "decode"):
        registrations = json.loads((tmp_path / f"{role}.regions.json").read_text())
        assert len(registrations) == (2 if role == "prefill" else 4)
        assert all(
            (_marker(target) in item["scope"]) == (role == "decode")
            for item in registrations
        )
    failed = []
    checks = []
    for d in range(4):
        rows = _records(tmp_path, "decode", d)
        assert not _loads(rows)
        failed.extend(
            req
            for row in rows
            for req in row.get("kv_events", {}).get("reads_failed", [])
        )
        checks.extend(_checks(rows))
    # The scheduler can abort before the other ranks get to report their error.
    assert failed
    assert bool(checks) == checked
    assert all(not c["ok"] and c["error"] == "receive failed" for c in checks)
    response = tmp_path / "decode.json"
    assert not response.exists() or json.loads(response.read_text())["tokens"] == []


def _config(path, target, **options):
    from vllm import EngineArgs

    model = published_path(target)
    tp = options.pop("tp", 2)
    return EngineArgs(
        **_options(target, model, tp, False, path / "steps", check=False, **options)
    ).create_engine_config()


@pytest.mark.parametrize("target", PROFILES, ids=["v32", "glm"])
def test_sparse_mtp_admission_and_dcp_order(tmp_path, target):
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
        SPECULATIVE_DCP_TRANSFERS,
        simulated_dcp_scope,
        simulated_tp_mismatch,
    )

    config = _config(tmp_path, target)
    assert _marker(target) in simulated_dcp_scope(config)
    assert simulated_tp_mismatch(config, 2, 4, nixl_consumer=True) is None
    assert simulated_tp_mismatch(config, 2, 2, nixl_consumer=True) is None
    assert REFUSAL in simulated_tp_mismatch(config, 2, 4)
    assert REFUSAL in simulated_tp_mismatch(config, 2, 8, nixl_consumer=True)
    assert "one dividing the other" in simulated_tp_mismatch(
        config,
        2,
        3,
        nixl_consumer=True,
    )
    assert SPECULATIVE_DCP_TRANSFERS in simulated_dcp_scope(config)
    assert SPECULATIVE_DCP_TRANSFERS in simulated_tp_mismatch(
        config, 2, 4, nixl_consumer=True, remote_dcp_size=2
    )
    with pytest.raises(ValueError, match="speculative decoding"):
        _config(tmp_path, target, decode_context_parallel_size=2)


@pytest.mark.parametrize(
    "target,options",
    [
        (V32, {"tp": 1}),
        (V32, {"depth": 2}),
        (GLM, {"enable_prefix_caching": True}),
        (V32, {"block_size": 128}),
        (V32, {"kv_cache_dtype": "fp8_ds_mla"}),
        (GLM, {"kv_cache_dtype": "bfloat16"}),
        (V32, {"timing_args": {"gpu": "h100"}}),
        (GLM, {"timing_args": {"target_model": "zai-org/GLM-5.2-FP8"}}),
        (V32, {"hf_overrides": {"index_topk_freq": 2}}),
        (GLM, {"hf_overrides": {"index_n_heads": 64}}),
        (V32, {"data_parallel_size": 2, "enable_expert_parallel": True}),
        (
            V32,
            {
                "kv_transfer_config": {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": "kv_consumer",
                    "kv_buffer_device": "cpu",
                }
            },
        ),
        (
            GLM,
            {
                "kv_transfer_config": {
                    "kv_connector": "NixlConnector",
                    "kv_role": "kv_consumer",
                    "kv_buffer_device": "cpu",
                    "kv_connector_extra_config": {"enforce_handshake_compat": False},
                }
            },
        ),
    ],
    ids=[
        "tp1",
        "depth",
        "prefix",
        "page",
        "v32-fp8-kv",
        "glm-bf16-kv",
        "gpu",
        "other-glm",
        "index-skip",
        "index-heads",
        "dp-ep",
        "mooncake",
        "hash-off",
    ],
)
def test_sparse_mtp_outside_scope_is_refused(tmp_path, target, options):
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
        simulated_dcp_scope,
        simulated_tp_mismatch,
    )

    config = _config(tmp_path, target, **options)
    assert _marker(target) not in simulated_dcp_scope(config)
    local = options.get("tp", 2)
    nixl = (
        options.get("kv_transfer_config", {}).get("kv_connector") != "MooncakeConnector"
    )
    assert REFUSAL in simulated_tp_mismatch(config, local, 4, nixl_consumer=nixl)


@pytest.mark.parametrize("model", [V32, GLM], ids=["v32", "glm"])
def test_sparse_mtp_local_draft_tp_mismatch_fails_config(tmp_path, model):
    with pytest.raises(ValueError, match="not draft tensor parallelism 1"):
        _config(tmp_path, model, spec_overrides={"draft_tensor_parallel_size": 1})


@pytest.mark.parametrize(
    "damage",
    [
        "clean",
        "missing-scope",
        "foreign-scope",
        "both-scopes",
        "order",
        "indexer-size",
        "indexer-group",
        "stride",
        "split-page",
    ],
)
def test_sparse_mtp_matches_actual_peer_registration(tmp_path, damage):
    from tests.v1.worker.simulated.test_real_kv_transfer import agent_metadata
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
        unequal_tp_mtp_layout,
    )
    from vllm.v1.worker.simulated.kv_transfer import _matches_mtp_peer

    layout = unequal_tp_mtp_layout(_config(tmp_path, V32))
    names = [name for i in range(62) for name in _region_names(i)]
    lengths = [73_728, INDEXER_BYTES] * 62
    groups = [0] * 124
    strides = list(lengths)
    scopes = {
        "missing-scope": [],
        "foreign-scope": [_marker(GLM)],
        "both-scopes": [_marker(V32), _marker(GLM)],
    }
    if damage == "order":
        names[-2], names[-1] = names[-1], names[-2]
    elif damage == "indexer-size":
        lengths[-1] += 4
    elif damage == "indexer-group":
        groups[-1] = 1
    elif damage == "stride":
        strides[-1] += 4
    peer = agent_metadata(
        block_size=64,
        physical_blocks_per_logical_kv_block=2 if damage == "split-page" else 1,
        region_names=names,
        block_lens=lengths,
        region_group_ids=groups,
        block_strides=strides,
        simulated_scope=scopes.get(damage, [_marker(V32)]),
    )
    assert _matches_mtp_peer(layout, peer) == (damage == "clean")


@pytest.mark.parametrize(
    "damage",
    [
        "clean",
        "missing-indexer",
        "duplicate-indexer",
        "lookahead",
        "wrong-producer",
        "wrong-engine",
        "unfinished",
        "twice",
    ],
)
def test_sparse_mtp_audits_draft_indexer_coverage(damage):
    from dataclasses import replace

    from vllm.v1.worker.simulated.kv_pattern import mtp_read_coverage
    from vllm.v1.worker.simulated.state_layout import ReadRecord

    latent = (122, 1, 0, 73_728)
    indexer = (123, 1, 0, INDEXER_BYTES)
    record = ReadRecord("req", 0, "read", [latent, indexer], done=True, engine="P")
    records = {
        "clean": [record],
        "missing-indexer": [replace(record, pieces=[latent])],
        "duplicate-indexer": [replace(record, pieces=[latent, indexer, indexer])],
        "lookahead": [
            replace(record, pieces=[latent, indexer, (123, 2, 0, INDEXER_BYTES)])
        ],
        "wrong-producer": [replace(record, rank=1)],
        "wrong-engine": [replace(record, engine="other")],
        "unfinished": [replace(record, done=False)],
        "twice": [record, record],
    }[damage]
    audit = mtp_read_coverage({0: [latent, indexer]}, records, "P")
    assert bool(audit["problems"]) == (damage != "clean")
