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
            assert "simulated/nixl/unequal-tp-mla-mtp/v1" in reg["scope"]
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
    marker = "simulated/nixl/unequal-tp-mla-mtp/v1"
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
        UNEQUAL_TP_MLA_MTP,
        simulated_dcp_scope,
        simulated_tp_mismatch,
    )

    config = _engine_config(tmp_path, tp=tp)
    assert simulated_tp_mismatch(config, tp, peer, nixl_consumer=True) is None
    assert simulated_dcp_scope(config) == [
        SPECULATIVE_DCP_TRANSFERS,
        UNEQUAL_TP_MLA_MTP,
    ]
    assert REFUSAL in simulated_tp_mismatch(config, tp, peer)
    assert simulated_tp_mismatch(config, tp, tp, nixl_consumer=True) is None
    assert REFUSAL in simulated_tp_mismatch(config, tp, 8, nixl_consumer=True)
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
        UNEQUAL_TP_MLA_MTP,
        simulated_dcp_scope,
        simulated_tp_mismatch,
    )

    config = _engine_config(tmp_path, **options)
    assert UNEQUAL_TP_MLA_MTP not in simulated_dcp_scope(config)
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
