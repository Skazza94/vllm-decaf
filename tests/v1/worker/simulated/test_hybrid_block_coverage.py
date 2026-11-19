# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid attention ownership across native NIXL transfer block sizes."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from vllm.v1.worker.simulated import state_layout as sl
from vllm.v1.worker.simulated.kv_pattern import row_pattern

from tests.v1.worker.simulated.test_real_kv_transfer import (
    nemotron_3_nano_config,
    qwen3_5_27b_config,
    run_dcp_peers,
)

QWEN = "Qwen/Qwen3.5-27B"


def published_config(target):
    return {QWEN: qwen3_5_27b_config, NANO: nemotron_3_nano_config}[target]()


def test_multi_head_ratio_two_real_read_covers_the_whole_row(tmp_path):
    """Four heads from one producer form complete ownership before splitting."""
    common = {
        "max_model_len": 4096,
        "prompt_len": 1900,
        "language_model_only": True,
        "timing": "aisimulate",
        "timing_args": {"gpu": "h100", "target_model": QWEN},
    }
    producer, consumer = run_dcp_peers(
        published_config(QWEN),
        {**common, "block_size": 800},
        {**common, "block_size": 1600},
        tmp_path,
        memory=(2 << 30, 2 << 30),
    )
    assert consumer["tokens"] == [2], consumer["log"][-4000:]
    checks = [row["kv_check"] for row in consumer["logs"][0] if "kv_check" in row]
    assert len(checks) == 1
    check = checks[0]
    assert check["ok"], check
    assert check["coverage"]["problems"] == []
    assert check["bytes"] == 363659264
    assert sum(group.get("zeroed_bytes", 0) for group in check["groups"]) == 52428800
    assert sum(group["padding_bytes"] for group in check["groups"]) == 160628736
    assert sum(read["bytes"] for read in check["coverage"]["reads"]) == 311230464
    assert producer["released"] is True


NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


@pytest.mark.parametrize("heads", [(0,), (0, 1), (0, 1, 2, 3)])
@pytest.mark.parametrize("ratio", [2, 4])
def test_head_ownership_and_native_conversion_keep_each_sub_block(heads, ratio):
    """Coverage is in wire order; expected bytes are in final head order."""
    from transformers import Qwen3_5TextConfig
    from vllm.distributed.kv_transfer.kv_connector.utils import (
        kv_postprocess_blksize_on_receive,
    )

    text = Qwen3_5TextConfig(**published_config(QWEN)["text_config"])
    layout = sl.layout_from_config(text, 2, 4)
    size = len(heads) * ratio * 16
    row = {
        "region": 0,
        "offset": 0,
        "length": size,
        "state": False,
        "heads": heads,
        "span": None,
    }
    pieces = sl.expected_pieces([row], [7], 0, 4 // len(heads), 1, layout, 4)
    ranges = [(off, length) for _, _, off, length in pieces[0]]
    # Every span of sub-blocks a pull can cover: its head, its tail, or both
    # clipped (the producer's retained blocks start inside the block, the
    # handoff ends inside it).
    spans = [
        range(first, last)
        for first in range(ratio + 1)
        for last in range(first, ratio + 1)
    ]
    for covered in spans:
        offset, length = sl.attention_wire_prefix(
            0, list(reversed(ranges)), 0, size, ratio, covered
        )
        assert (offset, length) == (
            covered.start * size // ratio,
            len(covered) * size // ratio,
        )
        chunks = [
            row_pattern("request", "layer", 3 * ratio + k, size // ratio, heads=heads)
            if k in covered
            else torch.zeros(size // ratio, dtype=torch.uint8)
            for k in range(ratio)
        ]
        cache = torch.cat(chunks).reshape(1, len(heads), ratio * 16, 1)
        kv_postprocess_blksize_on_receive(cache, torch.tensor([0]), ratio)
        expected = row_pattern(
            "request",
            "layer",
            3,
            size,
            heads=heads,
            sub_blocks=(ratio, covered),
        )
        assert torch.equal(cache.flatten(), expected)
        records = [sl.ReadRecord("r", 0, 1, [(0, 7, offset, length)], True)]
        wanted = {0: [(0, 7, offset, length)]} if length else {}
        if not length:
            records = []
        assert sl.coverage_problems(wanted, records) == []


@pytest.mark.parametrize(
    "ranges",
    [
        [],
        [(0, 32)],
        [(1, 32), (33, 32)],
        [(0, 32), (16, 32)],
        [(0, 32), (0, 32), (32, 32)],
        [(0, 16), (32, 32)],
        [(0, 65)],
    ],
)
def test_wire_prefix_never_promotes_a_partial_or_duplicate_owner(ranges):
    with pytest.raises(ValueError, match="share of a 64-byte page"):
        sl.attention_wire_prefix(0, ranges, 0, 64, 2, range(1))


@pytest.mark.parametrize("fault", ["missing", "duplicate", "shift", "owner", "pending"])
def test_wire_coverage_rejects_wrong_or_unfinished_read_bytes(fault):
    wanted = {0: [(0, 7, 0, 32)]}
    pieces = [(0, 7, 0, 32)]
    if fault == "missing":
        pieces = [(0, 7, 0, 31)]
    elif fault == "duplicate":
        pieces *= 2
    elif fault == "shift":
        pieces = [(0, 7, 1, 32)]
    record = sl.ReadRecord(
        "r", int(fault == "owner"), 1, pieces, done=fault != "pending"
    )
    assert sl.coverage_problems(wanted, [record])


def checker_controls(worker):
    """Exercise the public check lifecycle on real registered cache views."""
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.worker.simulated.kv_pattern import NativeChecker

    original = worker.model_runner._kv_checker
    selection = original.selection
    connector = get_kv_transfer_group().connector_worker
    (remote,) = [key for key in connector.tp_mappings if key != connector.engine_id]
    blocks = [[group + 1] for group in range(len(selection.groups))]
    results = {}
    for fault in ("clean", "head", "state", "tail", "padding"):
        pieces = []
        checker = NativeChecker(
            selection,
            heads=original.heads,
            records={"r": [sl.ReadRecord("r", 0, 1, pieces, True, remote)]},
        )
        checker.expect(
            "r", "r", [0] * len(blocks), blocks, remote, [1] * len(blocks), 799
        )
        attention = state = padding = None
        for group, ids in enumerate(blocks):
            for row, block, _ in selection.select(group, ids):
                entry = selection.rows[row]
                view = entry.view[block]
                if entry.state:
                    view.copy_(
                        sl.state_span_pattern(
                            "r",
                            selection.state_identity(group, entry.region),
                            0,
                            selection.layout,
                            entry.span,
                        )
                    )
                    pieces.append((entry.region, block, entry.offset, view.numel()))
                    state = view
                    if entry.region in selection.padding:
                        padding = selection.padding[entry.region].view[block]
                else:
                    heads = original.heads[group]
                    view.copy_(
                        row_pattern(
                            "r",
                            selection.name(group, row),
                            0,
                            view.numel(),
                            heads=heads,
                            sub_blocks=(2, range(1)),
                        )
                    )
                    pieces.append(
                        (entry.region, block, entry.offset, view.numel() // 2)
                    )
                    attention = (view, len(heads))
        if fault in ("head", "tail"):
            view, heads = attention
            index = view.numel() // heads + 3 if fault == "head" else view.numel() - 1
            view[index] ^= 1
        elif fault == "state":
            state[7] ^= 1
        elif fault == "padding":
            padding[0] ^= 1
        (result,) = checker.after_step(KVConnectorOutput(finished_recving={"r"}))
        results[fault] = result
    return results


def corrupt_prefill(worker, handoff, kind):
    """Change a retained source byte before its handoff is published."""
    if worker.rank != 0:
        return None
    filler = worker.model_runner._kv_filler
    selection = filler.selection
    for group, ids in enumerate(handoff["remote_block_ids"]):
        for row, block, _ in selection.select(group, ids[:1]):
            entry = selection.rows[row]
            if entry.state != (kind == "state"):
                continue
            view = entry.view[block]
            heads = filler.heads[group]
            offset = view.numel() // len(heads) + 7 if heads else 7
            view[offset] ^= 1
            return {"kind": kind, "block": block, "offset": offset}
    raise AssertionError("no source byte selected")


def peer_main():
    """One native LLM endpoint, with no replaced worker/checker methods."""
    from vllm import LLM, SamplingParams

    role, target, settings, destination = sys.argv[1:]
    options = json.loads(settings)
    directory = Path(destination)
    producer = role == "prefill"
    prompt = options.pop("prompt")
    fault = options.pop("fault", None)
    controls = options.pop("controls", False)
    check = options.pop("check", True)
    backend = options.pop("backend", None)
    simulation = {
        "simulate_forward": True,
        "timing": "aisimulate",
        "timing_args": {"gpu": "h100", "target_model": target},
        "log_path": str(directory / f"{role}.steps"),
        "kv_kernel_backend": backend,
    }
    if producer or check:
        simulation["kv_pattern_fill" if producer else "kv_pattern_check"] = True
    llm = LLM(
        model=str(directory / "model"),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        language_model_only=target == QWEN,
        enforce_eager=True,
        max_model_len=4096,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        kv_cache_memory_bytes=2 << 30,
        enable_prefix_caching=False,
        simulation_config=simulation,
        kv_transfer_config={
            "kv_connector": "NixlConnector",
            "kv_buffer_device": "cpu",
            "kv_role": "kv_producer" if producer else "kv_consumer",
            "kv_connector_extra_config": {
                "backends": ["Mooncake"],
                "kv_lease_duration": 300,
            },
        },
        **options,
    )
    handoff = (
        {"do_remote_decode": True}
        if producer
        else json.loads((directory / "prefill.json").read_text())["handoff"]
    )
    params = SamplingParams(
        max_tokens=1 if producer else 2,
        ignore_eos=True,
        detokenize=False,
        extra_args={"kv_transfer_params": handoff},
    )
    (response,) = llm.generate(
        [{"prompt_token_ids": [5] * prompt}],
        params,
        use_tqdm=False,
    )
    value = {
        "handoff": response.kv_transfer_params,
        "tokens": list(response.outputs[0].token_ids),
    }
    if producer and fault:
        value["corrupted"] = llm.collective_rpc(
            corrupt_prefill,
            args=(value["handoff"], fault),
        )
    if not producer and controls:
        value["controls"] = llm.collective_rpc(checker_controls)
    output = directory / f"{role}.json"
    staging = output.with_suffix(".tmp")
    staging.write_text(json.dumps(value))
    staging.rename(output)
    if producer:
        core = llm.llm_engine.engine_core.engine_core
        deadline = time.monotonic() + 480
        while not (directory / "stop").exists():
            assert time.monotonic() < deadline, "consumer did not finish"
            core.step()
            if not core.scheduler.has_requests():
                (directory / "released").touch()
            time.sleep(0.02)


def native_peers(tmp_path, target, producer, consumer):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(published_config(target)))
    script = tmp_path / "peer.py"
    script.write_text(
        "from tests.v1.worker.simulated.test_hybrid_block_coverage import peer_main\n"
        "if __name__ == '__main__':\n    peer_main()\n"
    )

    def command(role, options):
        for _ in range(100):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            held = []
            try:
                for offset in range(8):
                    sock = socket.socket()
                    held.append(sock)
                    sock.bind(("127.0.0.1", port + offset))
            except OSError:
                continue
            finally:
                for sock in held:
                    sock.close()
            break
        else:
            raise AssertionError("no free NIXL port range")
        env = {
            **os.environ,
            "PYTHONPATH": str(Path.cwd()),
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "VLLM_HOST_IP": "127.0.0.1",
            "NIXL_MOONCAKE_IP_ADDR": "127.0.0.1",
            "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(port),
            "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
        }
        return [
            sys.executable,
            str(script),
            role,
            target,
            json.dumps(options),
            str(tmp_path),
        ], env

    args, env = command("prefill", producer)
    log = tmp_path / "prefill.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            args, env=env, stdout=output, stderr=subprocess.STDOUT
        )
    try:
        deadline = time.monotonic() + 300
        while not (tmp_path / "prefill.json").exists():
            assert process.poll() is None, log.read_text()[-6000:]
            assert time.monotonic() < deadline, log.read_text()[-6000:]
            time.sleep(0.2)
        args, env = command("decode", consumer)
        with (tmp_path / "decode.log").open("w") as output:
            result = subprocess.run(
                args, env=env, timeout=300, stdout=output, stderr=subprocess.STDOUT
            )
        assert result.returncode == 0, (tmp_path / "decode.log").read_text()[-8000:]
        deadline = time.monotonic() + 60
        while not (tmp_path / "released").exists():
            assert time.monotonic() < deadline, "producer did not release its blocks"
            time.sleep(0.1)
    finally:
        (tmp_path / "stop").touch()
        try:
            process.wait(timeout=30)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    value = json.loads((tmp_path / "decode.json").read_text())
    logs = {}
    for role, settings in (("prefill", producer), ("decode", consumer)):
        logs[role] = {}
        for rank in range(settings.get("tensor_parallel_size", 1)):
            suffix = f".tp{rank}" if rank else ""
            path = tmp_path / f"{role}{suffix}.steps"
            logs[role][rank] = [
                json.loads(line) for line in path.read_text().splitlines()
            ]
    return value, logs


@pytest.mark.parametrize("fault", [None, "head", "state"])
def test_multi_head_real_peer_and_independent_corruption_controls(tmp_path, fault):
    common = {"prompt": 1900, "tensor_parallel_size": 1}
    value, logs = native_peers(
        tmp_path,
        QWEN,
        {**common, "block_size": 800, "fault": fault},
        {**common, "block_size": 1600, "controls": fault is None},
    )
    assert len(value["tokens"]) == 2
    (check,) = [r["kv_check"] for r in logs["decode"][0] if "kv_check" in r]
    assert check["coverage"]["problems"] == []
    assert check["mismatched_bytes"] == int(fault is not None)
    assert check["ok"] == (fault is None)
    if fault is None:
        (controls,) = value["controls"]
        for mode, control in controls.items():
            assert control["coverage"]["problems"] == []
            assert control["ok"] == (mode == "clean"), control
            assert control["mismatched_bytes"] == int(mode != "clean"), control
            padding = sum(g["padding_mismatched_bytes"] for g in control["groups"])
            assert padding == int(mode == "padding")


@pytest.mark.parametrize(
    "target,ptp,dtp,pblock,dblock,prompt,moved,checked,zeroed,padding,sources",
    [
        (QWEN, 1, 2, 800, 1600, 1900, 155615232, 181829632, 26214400, 80314368, 1),
        (NANO, 8, 4, 544, 1056, 1089, 15612928, 18758656, 3145728, 164864, 2),
        (NANO, 4, 8, 1056, 544, 1057, 9379328, 9477632, 98304, 270848, 1),
        (QWEN, 8, 4, 416, 800, 833, 52117504, 64700416, 12582912, 835584, 2),
    ],
    ids=["two-heads", "nemotron-forward", "nemotron-reverse", "qwen-kernel"],
)
@pytest.mark.parametrize("check", [True, False], ids=["checked", "unchecked"])
def test_native_admitted_kernel_pairs_keep_exact_bytes(
    tmp_path,
    target,
    ptp,
    dtp,
    pblock,
    dblock,
    prompt,
    moved,
    checked,
    zeroed,
    padding,
    sources,
    check,
):
    declared = "FLASHINFER" if target == NANO or ptp == 8 else None
    common = {"prompt": prompt, "backend": declared}
    value, logs = native_peers(
        tmp_path,
        target,
        {**common, "tensor_parallel_size": ptp, "block_size": pblock},
        {**common, "tensor_parallel_size": dtp, "block_size": dblock, "check": check},
    )
    assert len(value["tokens"]) == 2
    for rank, rows in logs["decode"].items():
        checks = [r["kv_check"] for r in rows if "kv_check" in r]
        assert len(checks) == int(check)
        if check:
            (result,) = checks
            assert result["ok"], result
            reads = result["coverage"]["reads"]
            assert len(reads) == sources
            assert sum(read["bytes"] for read in reads) == moved
            assert result["bytes"] == checked
            assert sum(g.get("zeroed_bytes", 0) for g in result["groups"]) == zeroed
            assert sum(g["padding_bytes"] for g in result["groups"]) == padding
        loads = [
            r["kv_transfers"]["load"]
            for r in rows
            if r.get("kv_transfers", {}).get("load", {}).get("count")
        ]
        assert sum(load["bytes"] for load in loads) == moved
        assert sum(load["count"] for load in loads) == sources


@pytest.mark.parametrize("prompt", [801, 833, 1600], ids=["row0", "row1", "full"])
def test_equal_block_pair_moves_every_kernel_block_of_the_handoff_blocks(
    tmp_path, prompt
):
    """Two peers of one split kernel geometry pair no sub-blocks: NIXL reads every
    kernel block of each of the producer's blocks, so the kernel blocks past the
    handoff's last token arrive (the producer's fill of its whole block) and are
    checked, never expected zero or unread."""
    options = {"tensor_parallel_size": 1, "block_size": 800, "backend": "FLASHINFER"}
    value, logs = native_peers(
        tmp_path,
        QWEN,
        {**options, "prompt": prompt},
        {**options, "prompt": prompt, "check": True},
    )
    assert len(value["tokens"]) == 2
    (result,) = [r["kv_check"] for r in logs["decode"][0] if "kv_check" in r]
    assert result["ok"], result
    assert result["coverage"]["problems"] == []
    assert result["mismatched_bytes"] == 0
    assert sum(g.get("zeroed_bytes", 0) for g in result["groups"]) == 0
    assert result["posted_matches_checked"], result


@pytest.mark.parametrize(
    "local,remote,local_tp,remote_tp,heads",
    [
        (1056, 544, 4, 8, 2),
        (544, 1056, 8, 4, 2),
        (800, 416, 4, 8, 4),
        (416, 800, 8, 4, 4),
        (800, 1600, 1, 1, 4),
        (2112, 544, 2, 8, 2),
    ],
)
def test_whole_gqa_blocks_keep_the_native_directional_refusal(
    local,
    remote,
    local_tp,
    remote_tp,
    heads,
):
    """The native contract is directional even when reverse divisibility holds."""
    from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology

    topology = TransferTopology(
        tp_rank=0,
        tp_size=local_tp,
        block_size=local,
        engine_id="decode",
        is_mla=False,
        is_mamba=True,
        total_num_kv_heads=heads,
        attn_backends=[],
    )
    with pytest.raises(AssertionError) as error:
        topology.block_size_ratio(remote)
    assert str(error.value) == (
        f"Local block size {local} is not divisible "
        f"by remote block size {remote} or vice versa."
    )
