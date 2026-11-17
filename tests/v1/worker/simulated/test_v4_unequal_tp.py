# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 over NIXL at unequal tensor-parallel sizes, plain and with
native MTP: the admission rule over every row of its grid (real configs), the
capability and registration checks, the READ identity and row accounting, the
timing inputs, and real engine pairs."""

import dataclasses
import functools
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from tests.v1.worker.simulated.test_kernel_geometry import published_path
from tests.v1.worker.simulated.utils import sim_req, sim_step

V4 = "deepseek-ai/DeepSeek-V4-Flash"
TPS = (1, 2, 4, 8)
UNEQUAL_PAIRS = [(p, d) for p in TPS for d in TPS if p != d]
DEPTHS = (None, 1, 2, 3, 4, 5)
ROW = 1_002_240
SWA_PAGE, SWA_REAL = 37_440, 37_376
DRAFT = "model.layers.43.attn.swa_cache"
# (block size, window) of V4-Flash's five cache groups: two sliding windows of
# 128 tokens, the compressed latents and indexer keys (whole), two compressor
# states (windows of 8 and 128 tokens).
GROUPS = ((64, 128), (64, 128), (256, None), (4, 8), (8, 128))


def held_rows(tokens: int, retained: int = 0) -> list[int]:
    from vllm.v1.worker.simulated.v4_reads import held_blocks

    return [held_blocks(tokens, b, w, retained) for b, w in GROUPS]


@functools.cache
def endpoint(
    tp,
    role,
    depth=None,
    prefix=False,
    *,
    gpu="b200",
    target=V4,
    connector="NixlConnector",
    dtype="bfloat16",
    cache_dtype="auto",
    dcp=1,
    pp=1,
    hash_check=True,
    bidirectional=False,
    quant_modes=None,
):
    """A real engine configuration over V4-Flash's published config."""
    from vllm import EngineArgs

    timing_args = {"gpu": gpu, "target_model": target}
    if quant_modes:
        timing_args["quant_modes"] = dict(quant_modes)
    options = {
        "model": str(published_path(V4)),
        "skip_tokenizer_init": True,
        "dtype": dtype,
        "enforce_eager": True,
        "max_model_len": 4096,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 4,
        "kv_cache_memory_bytes": 2 << 30,
        "enable_prefix_caching": prefix,
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "decode_context_parallel_size": dcp,
        "kv_cache_dtype": cache_dtype,
        "simulation_config": {
            "simulate_forward": True,
            "timing": "aisimulate",
            "timing_args": timing_args,
        },
        "kv_transfer_config": {
            "kv_connector": connector,
            "kv_buffer_device": "cpu",
            "kv_role": role,
            "kv_connector_extra_config": {
                "enforce_handshake_compat": hash_check,
                **({"bidirectional_kv_xfer": True} if bidirectional else {}),
            },
        },
    }
    if depth:
        options["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": depth,
        }
    return EngineArgs(**options).create_engine_config()


def baseline_metadata(**fields):
    """A real handshake metadata of a small packed registration."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlAgentMetadata,
    )

    values = {
        "engine_id": "producer",
        "agent_metadata": b"",
        "kv_caches_base_addr": [0],
        "device_id": 0,
        "num_blocks": 16,
        "block_lens": [ROW],
        "block_strides": [ROW],
        "kv_cache_layout": "NHD",
        "block_size": 256,
        "ssm_sizes": (0, 0),
        "attn_backend_name": "CPU_V4_MLA_SPARSE",
        "physical_blocks_per_logical_kv_block": 1,
        "region_group_ids": [-1],
        "region_names": ["model.layers.1.attn.swa_cache"],
        "region_members": [["model.layers.1.attn.swa_cache", DRAFT]],
        "packed_member_layouts": {
            "model.layers.1.attn.swa_cache": (0, SWA_PAGE),
            DRAFT: (SWA_PAGE, SWA_PAGE),
        },
        "simulated_scope": None,
    }
    return NixlAgentMetadata(**{**values, **fields})


def pair_scope(producer, consumer):
    """Whether `consumer` pairs with `producer` as the handshake decides."""
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
        simulated_dcp_scope,
        simulated_tp_mismatch,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.v4_unequal_tp import (
        Registration,
        v4_peer_refusal,
    )

    p_tp = producer.parallel_config.tensor_parallel_size
    c_tp = consumer.parallel_config.tensor_parallel_size
    metadata = baseline_metadata(simulated_scope=simulated_dcp_scope(producer))
    mismatch = simulated_tp_mismatch(consumer, c_tp, p_tp, nixl_consumer=True)
    refusal = v4_peer_refusal(
        consumer, Registration.of_metadata(baseline_metadata()), p_tp, metadata
    )
    return mismatch, refusal


def row_record(p_tp, c_tp, p_depth, c_depth, p_prefix, c_prefix, admitted):
    producer = endpoint(p_tp, "kv_producer", p_depth, p_prefix)
    consumer = endpoint(c_tp, "kv_consumer", c_depth, c_prefix)
    mismatch, refusal = pair_scope(producer, consumer)
    good = mismatch is None and (refusal is None) == admitted
    return {
        "producer": [p_tp, p_depth, p_prefix],
        "consumer": [c_tp, c_depth, c_prefix],
        "admitted": admitted,
        "tp_rule": mismatch,
        "peer": refusal,
        "ok": good,
        "digest": hashlib.sha256(
            repr((fields_of(consumer), p_depth, c_depth)).encode()
        ).hexdigest()[:12],
    }


def write_ledger(rows):
    path = os.environ.get("V4_LEDGER")
    if path:
        with open(path, "a") as f:
            f.writelines(json.dumps(row) + "\n" for row in rows)


@pytest.mark.parametrize(("p_tp", "c_tp"), UNEQUAL_PAIRS, ids=lambda v: f"tp{v}")
def test_the_rule_over_every_row_of_an_unequal_tp_pair(p_tp, c_tp):
    """Depths of each side independent (none, or 1-5), prefix mode of each:
    admitted when MTP presence and prefix mode agree, else refused with the
    field named; the TP rule itself admits the consumer in every row."""
    rows = []
    for p_prefix in (False, True):
        for c_prefix in (False, True):
            for p_depth in DEPTHS:
                for c_depth in DEPTHS:
                    same = (p_depth is None) == (c_depth is None)
                    admitted = same and p_prefix == c_prefix
                    rows.append(
                        row_record(
                            p_tp, c_tp, p_depth, c_depth, p_prefix, c_prefix, admitted
                        )
                    )
    write_ledger(rows)
    failed = [row for row in rows if not row["ok"]]
    assert not failed, failed[:3]
    assert sum(row["admitted"] for row in rows) == 52
    for row in rows:
        if not row["admitted"]:
            p, c = row["producer"], row["consumer"]
            field = "mtp" if (p[1] is None) != (c[1] is None) else "prefix"
            assert f"the peer's {field} is" in row["peer"], row


def test_the_grid_has_the_rows_the_design_counts():
    """Per unequal pair of 144 combinations: 52 admitted (equal presence and
    prefix), 20 mixed presence, 52 prefix mismatch, 20 both; over the 12
    pairs 624, 240, 624 and 240."""
    per_pair = 4 * len(DEPTHS) ** 2
    assert per_pair == 144 == 52 + 20 + 52 + 20
    assert len(UNEQUAL_PAIRS) * 52 == 624
    assert len(UNEQUAL_PAIRS) * 20 == 240
    assert len(UNEQUAL_PAIRS) * 52 == 624


@pytest.mark.parametrize("tp", TPS)
def test_equal_tp_rows_keep_their_admission(tp):
    """The equal-TP controls, 52 a size (2 prefix modes x (plain/plain plus
    every pair of depths 1-5), 208 over the four sizes), each with both real
    endpoint configs: admitted as before and the peer check silent, as no
    capability is needed between equal sizes; the scope still carries each
    qualifying endpoint's entry."""
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
        simulated_dcp_scope,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.v4_unequal_tp import (
        CAPABILITY_FAMILY,
    )

    rows = []
    for prefix in (False, True):
        for p_depth in DEPTHS:
            for c_depth in DEPTHS:
                if (p_depth is None) != (c_depth is None):
                    continue
                producer = endpoint(tp, "kv_producer", p_depth, prefix)
                consumer = endpoint(tp, "kv_consumer", c_depth, prefix)
                mismatch, refusal = pair_scope(producer, consumer)
                scope = simulated_dcp_scope(producer) + simulated_dcp_scope(consumer)
                rows.append(
                    {
                        "equal": True,
                        "producer": [tp, p_depth, prefix],
                        "consumer": [tp, c_depth, prefix],
                        "admitted": True,
                        "tp_rule": mismatch,
                        "peer": refusal,
                        "ok": mismatch is None
                        and refusal is None
                        and sum(e.startswith(CAPABILITY_FAMILY) for e in scope) == 2,
                    }
                )
    write_ledger(rows)
    assert len(rows) == 52
    assert all(row["ok"] for row in rows), [r for r in rows if not r["ok"]][:3]


def test_a_combined_mismatch_names_the_first_field():
    producer = endpoint(2, "kv_producer", None, False)
    consumer = endpoint(4, "kv_consumer", 3, True)
    _, refusal = pair_scope(producer, consumer)
    assert "the peer's mtp is none, this instance's mtp" in refusal


@pytest.mark.parametrize(
    ("options", "problem"),
    [
        pytest.param({"gpu": "h200"}, "AISimulate timing", id="gpu"),
        pytest.param(
            {"target": "deepseek-ai/DeepSeek-V3"}, "AISimulate timing", id="target"
        ),
        pytest.param(
            {"quant_modes": (("gemm_quant_mode", "fp8"),)},
            "without quant_modes",
            id="quant-modes",
        ),
        pytest.param(
            {"role": "kv_both"}, "kv_role kv_producer or kv_consumer", id="kv-both"
        ),
        pytest.param(
            {"hash_check": False}, "handshake compatibility check", id="hash-off"
        ),
        pytest.param(
            {"bidirectional": True}, "no bidirectional transfer", id="bidirectional"
        ),
        pytest.param({"dtype": "float16"}, "BF16", id="fp16"),
    ],
)
def test_an_endpoint_outside_the_scope_is_named(options, problem):
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
        simulated_dcp_scope,
        simulated_tp_mismatch,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.v4_unequal_tp import (
        CAPABILITY_FAMILY,
        unequal_tp_v4_problems,
    )

    role = options.get("role", "kv_consumer")
    rest = {k: v for k, v in options.items() if k != "role"}
    consumer = endpoint(2, role, None, False, **rest)
    problems = unequal_tp_v4_problems(consumer)
    assert any(problem in p for p in problems), problems
    message = simulated_tp_mismatch(consumer, 2, 4, nixl_consumer=True)
    assert "plain or sparse MLA only (no compressed MLA, recurrent state" in message
    assert "DeepSeek-V4 pairs only as" in message
    assert problem in message
    assert not any(
        e.startswith(CAPABILITY_FAMILY) for e in simulated_dcp_scope(consumer)
    )


@pytest.mark.parametrize(
    ("options", "match"),
    [
        pytest.param(
            {"connector": "MooncakeConnector"},
            "NixlConnector for DeepSeek-V4",
            id="mooncake",
        ),
        pytest.param(
            {"dcp": 2},
            "supports decode context parallelism",
            id="dcp",
        ),
        pytest.param(
            {"pp": 2},
            "tensor parallelism and data-parallel attention at most",
            id="pp",
        ),
        pytest.param({"cache_dtype": "bfloat16"}, "packed FP8 cache", id="bf16-kv"),
    ],
)
def test_the_native_and_config_refusals_stay(options, match):
    with pytest.raises(ValueError, match=match):
        endpoint(2, "kv_consumer", None, False, **options)


def test_depth_six_is_refused_at_config():
    with pytest.raises(ValueError, match="6 speculative tokens"):
        endpoint(2, "kv_consumer", 6, False)


def test_tp_outside_the_priced_sizes_is_refused():
    from vllm.distributed.kv_transfer.kv_connector.v1.v4_unequal_tp import (
        unequal_tp_v4_problems,
    )

    for tp in (3, 16):
        try:
            config = endpoint(tp, "kv_consumer", None, False)
        except ValueError:
            continue
        assert any("tensor parallelism in" in p for p in unequal_tp_v4_problems(config))


def fields_of(config):
    from vllm.distributed.kv_transfer.kv_connector.v1.v4_unequal_tp import v4_fields

    return v4_fields(config)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda s, f: None, "states no simulated scope"),
        (lambda s, f: [], "states no DeepSeek-V4 unequal-TP capability"),
        (lambda s, f: [s, s], "states 2 DeepSeek-V4 unequal-TP capabilities"),
        (
            lambda s, f: [s, "simulated/nixl/unequal-tp-mla-mtp/v1"],
            "another model family's unequal-TP capability",
        ),
        (
            lambda s, f: [s.replace("/v1;", "/v2;")],
            "a malformed DeepSeek-V4 capability: version",
        ),
        (lambda s, f: [s + ";extra=1"], "fields"),
        (lambda s, f: [s + ";tp=2"], "repeated field"),
        (
            lambda s, f: [s.replace("role=kv_producer", "role=kv_consumer")],
            "not a producer",
        ),
        (lambda s, f: [s.replace("prefix=off", "prefix=on")], "prefix is on"),
        (lambda s, f: [s.replace("mtp=none", "mtp=mtp")], "mtp is mtp"),
        (lambda s, f: [s.replace("gpu=b200", "gpu=h200")], "gpu is h200"),
        (
            lambda s, f: [
                s.replace(f"geometry={f['geometry']}", "geometry=" + "0" * 16)
            ],
            "geometry is",
        ),
        (lambda s, f: [s.replace("tp=2", "tp=4")], "states TP 4 but its handshake"),
        (lambda s, f: [s.replace("role=kv_producer", "role=kv_both")], "malformed"),
    ],
)
def test_the_capability_is_checked_field_by_field(mutate, reason):
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
        simulated_dcp_scope,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.v4_unequal_tp import (
        capability_problem,
        v4_capability,
    )

    producer = endpoint(2, "kv_producer", None, False)
    consumer = endpoint(4, "kv_consumer", None, False)
    entry = v4_capability(producer)
    assert entry in simulated_dcp_scope(producer)
    scope = mutate(entry, fields_of(producer))
    problem = capability_problem(consumer, 2, scope)
    assert problem is not None and reason in problem, problem
    assert capability_problem(consumer, 2, simulated_dcp_scope(producer)) is None


def test_the_local_endpoint_and_a_consumer_only_peer_are_validated_too():
    from vllm.distributed.kv_transfer.kv_connector.v1.v4_unequal_tp import (
        capability_problem,
        v4_capability,
    )

    consumer = endpoint(4, "kv_consumer", None, False)
    other = endpoint(2, "kv_consumer", None, False)
    assert "not a producer" in capability_problem(consumer, 2, [v4_capability(other)])
    producer = endpoint(2, "kv_producer", None, False)
    local = endpoint(4, "kv_producer", None, False)
    problem = capability_problem(local, 2, [v4_capability(producer)])
    assert "this instance is kv_producer, not a consumer" in problem
    hashed_off = endpoint(4, "kv_consumer", None, False, hash_check=False)
    problem = capability_problem(hashed_off, 2, [v4_capability(producer)])
    assert "this instance is outside the scope" in problem


REGISTRATION_MUTATIONS = {
    "region_names": {"region_names": ["x"]},
    "block_lens": {"block_lens": [ROW - 64]},
    "block_strides": {"block_strides": [ROW + 64]},
    "group_ids": {"region_group_ids": [0]},
    "members": {"region_members": [["model.layers.1.attn.swa_cache"]]},
    "packed": {
        "packed_member_layouts": {
            "model.layers.1.attn.swa_cache": (0, SWA_PAGE),
            DRAFT: (SWA_PAGE + 64, SWA_PAGE),
        }
    },
    "block_size": {"block_size": 64},
    "factor": {"physical_blocks_per_logical_kv_block": 2},
}


@pytest.mark.parametrize("field", sorted(REGISTRATION_MUTATIONS))
def test_a_registered_field_that_differs_is_named(field):
    from vllm.distributed.kv_transfer.kv_connector.v1.v4_unequal_tp import (
        Registration,
        registration_problem,
    )

    local = Registration.of_metadata(baseline_metadata())
    assert registration_problem(local, local) is None
    remote = Registration.of_metadata(
        baseline_metadata(**REGISTRATION_MUTATIONS[field])
    )
    assert f"other {field}" in registration_problem(local, remote)


def test_the_registration_is_checked_before_a_peer_registers():
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
        simulated_dcp_scope,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.v4_unequal_tp import (
        Registration,
        v4_peer_refusal,
    )

    producer = endpoint(2, "kv_producer", None, False)
    consumer = endpoint(4, "kv_consumer", None, False)
    scope = simulated_dcp_scope(producer)
    local = Registration.of_metadata(baseline_metadata())
    other = baseline_metadata(simulated_scope=scope, block_lens=[ROW - 64])
    refusal = v4_peer_refusal(consumer, local, 2, other)
    assert "only between endpoints that qualify and register alike" in refusal
    assert "other block_lens" in refusal


def record(rank=1, engine="e", pieces=None, done=True):
    from vllm.v1.worker.simulated.state_layout import ReadRecord

    return ReadRecord("r", rank, object(), pieces, done, engine)


EXPECTED = [(0, b, 0, ROW) for b in (3, 4, 5, 9)]


@pytest.mark.parametrize(
    "tp", [(2, 2), (2, 4), (4, 2)], ids=["equal", "scatter", "gather"]
)
def test_a_v4_read_is_one_completed_record_of_the_owner_and_every_row(tp):
    from vllm.v1.worker.simulated.v4_reads import owner_rank, read_problems

    local, remote = tp
    rank = owner_rank(1, local, remote)
    assert rank == 1 * remote // local
    good = record(rank, "e", list(EXPECTED))
    assert read_problems("e", rank, EXPECTED, [good]) == []
    wrong = rank + 1

    def bad(*records, engine="e"):
        return read_problems(engine, rank, EXPECTED, list(records))

    cases = {
        "wrong engine": bad(record(rank, "x", list(EXPECTED))),
        "engine None": bad(record(rank, None, list(EXPECTED))),
        "no handoff engine": bad(good, engine=None),
        "wrong rank": bad(record(wrong, "e", list(EXPECTED))),
        "rank None": bad(record(None, "e", list(EXPECTED))),
        "split handles": bad(
            record(rank, "e", EXPECTED[:2]), record(rank, "e", EXPECTED[2:])
        ),
        "extra empty handle": bad(good, record(rank, "e", [])),
        "extra handle with pieces": bad(good, record(rank, "e", [EXPECTED[0]])),
        "two producers": bad(good, record(wrong, "e", list(EXPECTED))),
        "missing row": bad(record(rank, "e", EXPECTED[:-1])),
        "duplicate row": bad(record(rank, "e", [*EXPECTED, EXPECTED[0]])),
        "wrong block": bad(record(rank, "e", [*EXPECTED[:-1], (0, 10, 0, ROW)])),
        "wrong offset": bad(record(rank, "e", [*EXPECTED[:-1], (0, 9, 64, ROW)])),
        "wrong length": bad(record(rank, "e", [*EXPECTED[:-1], (0, 9, 0, ROW - 64)])),
        "another region": bad(record(rank, "e", [*EXPECTED[:-1], (1, 9, 0, ROW)])),
        "unresolved": bad(record(rank, "e", [*EXPECTED[:-1], (-1, -1, 0, ROW)])),
        "unresolved list": bad(record(rank, "e", None)),
        "incomplete": bad(record(rank, "e", list(EXPECTED), done=False)),
        "no record": bad(),
    }
    for name, problems in cases.items():
        assert problems, name


def test_a_request_that_pulls_no_rows_has_no_read():
    from vllm.v1.worker.simulated.v4_reads import read_problems

    assert read_problems("e", 0, [], []) == []
    assert read_problems("e", 0, [], [record(0, "e", [])]) == []
    assert read_problems("e", 0, [], [record(0, "e", [(0, 1, 0, ROW)])])


def packed_group(draft: bool):
    """V4-Flash's group 1: 21 sliding-window layers, plus the draft's."""
    from vllm.v1.worker.simulated.v4_reads import build_members

    names = [f"model.layers.{i}.attn.swa_cache" for i in range(1, 43, 2)]
    layers = names + ([DRAFT] if draft else [])
    packed = {n: (i * SWA_PAGE, SWA_PAGE) for i, n in enumerate(layers)}
    return build_members([layers], packed, dict.fromkeys(layers, SWA_REAL), 43)


def test_the_draft_slice_is_split_into_payload_alignment_and_the_row_tail():
    from vllm.v1.worker.simulated.v4_reads import RowAccount

    members = packed_group(True)
    draft_at = 21 * SWA_PAGE
    assert members[0][-1].offset == draft_at == 786_240
    rows = RowAccount(members)
    wrong = {
        0: np.array([], dtype=np.int64),
        1: np.array([draft_at, draft_at + 5], dtype=np.int64),
        2: np.array([draft_at + SWA_REAL + 3, 5], dtype=np.int64),
        3: np.array([22 * SWA_PAGE + 7], dtype=np.int64),
    }
    for block, offsets in wrong.items():
        rows.add(0, 0, block, ROW, offsets)
    pieces = [(0, b, 0, ROW) for b in wrong]
    report = rows.report(pieces)
    assert report["rows"] == 4 and report["bytes"] == 4 * ROW
    assert report["target_slice_bytes"] == 4 * 21 * SWA_PAGE
    assert report["draft_slice_bytes"] == 4 * SWA_PAGE
    assert report["tail_bytes"] == 4 * (ROW - 22 * SWA_PAGE) == 4 * 178_560
    assert report["reconciled"]
    assert report["mismatched"] == {
        "target_payload": 1,
        "target_alignment": 0,
        "draft_payload": 2,
        "draft_alignment": 1,
        "tail": 1,
    }
    assert report["draft"] == {
        "layers": [DRAFT],
        "registered_slice_bytes": 4 * SWA_PAGE,
        "payload_bytes": 4 * SWA_REAL,
        "alignment_bytes": 4 * 64,
        "read_bytes": 4 * SWA_PAGE,
        "mismatched_payload_bytes": 2,
        "mismatched_alignment_bytes": 1,
    }


def test_three_rows_carry_the_draft_slice_the_design_counts():
    from vllm.v1.worker.simulated.v4_reads import RowAccount

    rows = RowAccount(packed_group(True))
    for block in range(3):
        rows.add(0, 0, block, ROW, np.array([], dtype=np.int64))
    draft = rows.report([(0, b, 0, ROW) for b in range(3)])["draft"]
    assert draft["registered_slice_bytes"] == 112_320
    assert draft["payload_bytes"] == 112_128
    assert draft["alignment_bytes"] == 192


def test_a_row_the_read_did_not_cover_counts_no_draft_bytes_read():
    from vllm.v1.worker.simulated.v4_reads import RowAccount

    rows = RowAccount(packed_group(True))
    rows.add(0, 0, 0, ROW, np.array([], dtype=np.int64))
    draft = rows.report([(0, 0, 0, 100)])["draft"]
    assert draft["read_bytes"] == 0


def test_overlapping_slices_are_refused():
    from vllm.v1.worker.simulated.v4_reads import build_members

    with pytest.raises(ValueError, match="overlap"):
        build_members(
            [["a", "b"]], {"a": (0, 100), "b": (50, 100)}, {"a": 90, "b": 90}, 99
        )


@pytest.mark.parametrize(
    ("tokens", "counts"),
    [
        (600, [3, 3, 3, 2, 16]),
        (1024, [2, 2, 4, 2, 16]),
        (1025, [3, 3, 5, 3, 17]),
        (1027, [3, 3, 5, 2, 17]),
        (1031, [3, 3, 5, 2, 16]),
        (1087, [2, 2, 5, 2, 16]),
    ],
)
def test_the_held_rows_of_a_request_follow_the_windows(tokens, counts):
    assert held_rows(tokens) == counts
    assert sum(counts) * ROW == sum(held_rows(tokens)) * ROW


def test_retained_tokens_extend_the_window_edge():
    assert held_rows(1027)[3] == 2
    assert held_rows(1027, retained=8)[3] == 4


@pytest.mark.parametrize("worker", ["prefill", "decode"])
@pytest.mark.parametrize("tp", TPS)
@pytest.mark.parametrize("depth", DEPTHS)
def test_the_provider_is_the_direct_sdk_for_each_pricing_input(tp, depth, worker):
    pytest.importorskip("aisimulate_core")
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    from vllm.v1.worker.simulated.aisimulate_timing import (
        AisimulateTimingProvider,
        rank_metrics,
    )

    role = "kv_producer" if worker == "prefill" else "kv_consumer"
    config = endpoint(tp, role, depth, False)
    timed = AisimulateTimingProvider(config, gpu="b200", target_model=V4)
    record = timed.assumptions()["aisimulate"]
    resolved = record["resolved_config"]
    assert (resolved["tp"], resolved["nextn"]) == (tp, depth or 0)
    assert resolved["worker_type"] == worker
    assert resolved["estimation_mode"] == "op_level"
    assert resolved["database_mode"] == "SILICON"
    assert resolved["fallback_policy"] == "deny"
    sdk = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=V4,
            system="b200_sxm",
            backend="vllm",
            backend_version="0.24.0",
            worker_type=worker,
            tp=tp,
            pp=1,
            attention_dp=1,
            moe_tp_size=tp,
            moe_ep_size=1,
            nextn=depth or 0,
            gemm_quant_mode="fp8_block",
            moe_quant_mode="w4a8_mxfp4_mxfp8",
            fmha_quant_mode="bfloat16",
            kvcache_quant_mode="fp8",
            comm_quant_mode="half",
            estimation_mode="op_level",
            database_mode="SILICON",
            fallback_policy="deny",
        )
    )
    if worker == "prefill":
        step = sim_step(0, sim_req("p", 600, 0, 600))
    else:
        width = (depth or 0) + 1
        step = sim_step(
            0,
            *(
                dataclasses.replace(
                    sim_req(f"d{i}", width, 1024, 1024), num_spec_tokens=depth or 0
                )
                for i in range(4)
            ),
        )
    ms = sdk.estimate_forward_pass_time_ms(rank_metrics(list(step.requests)))
    assert ms > 0
    assert timed.estimate(step).device_s * 1000 == pytest.approx(ms, rel=1e-9)
    approximations = record["approximations"]
    assert any("pure sliding-window layers" in a for a in approximations)
    assert any("acceptance does not enter the cost" in a for a in approximations) == (
        bool(depth)
    )


# ---------------------------------------------------------------------------
# Real engine pairs: a producer and a consumer engine over NIXL in one
# container, as `test_mtp_unequal_tp.py` runs them.
# ---------------------------------------------------------------------------


def peer_options(tp, producer, path, depth, prefix, check=True):
    simulation = {
        "simulate_forward": True,
        "timing": "aisimulate",
        "timing_args": {"gpu": "b200", "target_model": V4},
        "log_path": str(path),
    }
    if depth:
        simulation["spec_acceptance_rates"] = [1.0]
    if check:
        simulation["kv_pattern_fill" if producer else "kv_pattern_check"] = True
    options = {
        "model": str(published_path(V4)),
        "skip_tokenizer_init": True,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "max_model_len": 4096,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 4,
        "kv_cache_memory_bytes": 2 << 30,
        "enable_prefix_caching": prefix,
        "tensor_parallel_size": tp,
        "simulation_config": simulation,
        "kv_transfer_config": {
            "kv_connector": "NixlConnector",
            "kv_buffer_device": "cpu",
            "kv_role": "kv_producer" if producer else "kv_consumer",
            "kv_connector_extra_config": {"backends": ["Mooncake"]},
        },
    }
    if depth:
        options["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": depth,
        }
    return options


def corrupt_byte(worker, layer, block, row_offset):
    """Flip one bit of a producer row, `row_offset` bytes into its block row
    (the region's storage starts at row zero)."""
    import torch

    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.distributed.kv_transfer import get_kv_transfer_group

    if get_tensor_model_parallel_rank() == 0:
        cache = get_kv_transfer_group().connector_worker.device_kv_caches[layer]
        stride = cache.stride(0) * cache.element_size()
        storage = cache.untyped_storage()
        flat = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),))
        flat[block * stride + row_offset] ^= 1


def _peer():
    from vllm import LLM, SamplingParams

    role, raw_tp, raw_settings, dest = sys.argv[1:5]
    producer = role == "prefill"
    settings = json.loads(raw_settings)
    directory = Path(dest)
    corrupt = settings.get("corrupt")
    llm = LLM(
        **peer_options(
            int(raw_tp),
            producer,
            directory / role,
            settings["depth"],
            settings["prefix"],
            settings["check"],
        )
    )
    for i, prompt in enumerate(settings["prompts"]):
        handoff = (
            {"do_remote_decode": True}
            if producer
            else json.loads((directory / f"prefill.{i}.json").read_text())["handoff"]
        )
        params = SamplingParams(
            max_tokens=1 if producer else 12,
            ignore_eos=True,
            detokenize=False,
            temperature=0,
            extra_args={"kv_transfer_params": handoff},
        )
        (response,) = llm.generate(
            [{"prompt_token_ids": prompt}], params, use_tqdm=False
        )
        if producer and corrupt and i == 0:
            group, layer, offset = corrupt
            block = response.kv_transfer_params["remote_block_ids"][group][0]
            llm.collective_rpc(corrupt_byte, args=(layer, block, offset))
        stage = directory / f"{role}.{i}.tmp"
        stage.write_text(
            json.dumps(
                {
                    "handoff": response.kv_transfer_params,
                    "tokens": list(response.outputs[0].token_ids),
                }
            )
        )
        stage.rename(directory / f"{role}.{i}.json")
    if producer:
        core = llm.llm_engine.engine_core.engine_core
        deadline = time.monotonic() + 300
        while not (directory / "stop").exists() and time.monotonic() < deadline:
            core.step()
            time.sleep(0.02)


def _free_base(width):
    held = []
    try:
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
                return base
            except OSError:
                continue
            finally:
                if not all(sock in held for sock in group):
                    for sock in group:
                        sock.close()
    finally:
        for sock in held:
            sock.close()


def run_pair(directory, p_tp, d_tp, p_settings, d_settings):
    script = directory / "peer.py"
    script.write_text(
        "from tests.v1.worker.simulated.test_v4_unequal_tp import _peer\n"
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

    def command(role, tp, settings):
        return [
            sys.executable,
            str(script),
            role,
            str(tp),
            json.dumps(settings),
            str(directory),
        ]

    last = len(p_settings["prompts"]) - 1
    with (directory / "prefill.log").open("w") as f:
        producer = subprocess.Popen(
            command("prefill", p_tp, p_settings),
            stdout=f,
            stderr=subprocess.STDOUT,
            env={
                **env,
                "VLLM_NIXL_SIDE_CHANNEL_PORT": str(_free_base(max(p_tp, d_tp))),
            },
        )
    try:
        deadline = time.monotonic() + 600
        while not (directory / f"prefill.{last}.json").exists():
            assert producer.poll() is None, (directory / "prefill.log").read_text()[
                -6000:
            ]
            assert time.monotonic() < deadline, "the producer did not finish"
            time.sleep(0.2)
        with (directory / "decode.log").open("w") as f:
            decode = subprocess.run(
                command("decode", d_tp, d_settings),
                stdout=f,
                stderr=subprocess.STDOUT,
                env={
                    **env,
                    "VLLM_NIXL_SIDE_CHANNEL_PORT": str(_free_base(max(p_tp, d_tp))),
                },
                timeout=900,
            )
        return decode.returncode
    finally:
        (directory / "stop").touch()
        try:
            producer.wait(timeout=30)
        except subprocess.TimeoutExpired:
            producer.kill()
            producer.wait(timeout=30)


def rank_rows(directory, role, rank):
    path = directory / (role if rank == 0 else f"{role}.tp{rank}")
    return [json.loads(line) for line in path.read_text().splitlines()]


def checks(rows):
    return [row["kv_check"] for row in rows if "kv_check" in row]


def prompt_of(n, seed=5):
    return [seed] * n


def settings_of(depth, prefix, prompts, **more):
    return {"depth": depth, "prefix": prefix, "check": True, "prompts": prompts, **more}


def decode_checks(directory, d_tp):
    return {rank: checks(rank_rows(directory, "decode", rank)) for rank in range(d_tp)}


def expected_counts(handoff_tokens, depth):
    return held_rows(handoff_tokens)


PAIR_CELLS = [
    pytest.param(1, 8, None, None, False, [600], id="1to8-plain"),
    pytest.param(8, 1, None, None, False, [600], id="8to1-plain"),
    pytest.param(
        8,
        2,
        5,
        5,
        True,
        [
            prompt_of(600),
            prompt_of(600),
            prompt_of(320) + prompt_of(280, 7),
            prompt_of(600, 7),
        ],
        id="8to2-mtp5-prefix",
    ),
    pytest.param(2, 4, 1, 5, False, [600], id="2to4-mtp1-to-mtp5"),
    pytest.param(4, 2, 1, 1, False, [600], id="4to2-mtp1"),
    pytest.param(2, 2, 1, 1, False, [600], id="2to2-control"),
    pytest.param(4, 2, None, None, False, [1025, 1027, 1031, 1087], id="4to2-residues"),
]


@pytest.mark.parametrize(
    ("p_tp", "d_tp", "p_depth", "d_depth", "prefix", "prompts"), PAIR_CELLS
)
def test_engine_pairs_move_the_rows_from_the_native_owner(
    tmp_path, p_tp, d_tp, p_depth, d_depth, prefix, prompts
):
    from vllm.v1.worker.simulated.v4_reads import owner_rank

    prompts = [prompt_of(p) if isinstance(p, int) else p for p in prompts]
    rc = run_pair(
        tmp_path,
        p_tp,
        d_tp,
        settings_of(p_depth, prefix, prompts),
        settings_of(d_depth, prefix, prompts),
    )
    log = (tmp_path / "decode.log").read_text()
    assert rc == 0, log[-6000:]
    per_rank = decode_checks(tmp_path, d_tp)
    for i, prompt in enumerate(prompts):
        handoff = json.loads((tmp_path / f"prefill.{i}.json").read_text())["handoff"]
        tokens = handoff["remote_num_tokens"]
        assert tokens in (len(prompt), len(prompt) - 1), (tokens, len(prompt))
        counts = held_rows(tokens)
        for rank in range(d_tp):
            check = per_rank[rank][i]
            assert check["ok"], check
            assert check["mismatched_bytes"] == 0
            assert [g["blocks"] for g in check["groups"]] == counts, (tokens, check)
            assert check["bytes"] == sum(counts) * ROW
            coverage = check["coverage"]
            assert coverage["problems"] == [], coverage
            (read,) = coverage["reads"]
            owner = owner_rank(rank, d_tp, p_tp)
            assert (read["rank"], read["completed"]) == (owner, True)
            assert read["descriptors"] == sum(counts)
            assert read["bytes"] == sum(counts) * ROW
            assert [g["pattern"] for g in check["groups"]] == ["head-v1"] * 5
            rows = check["rows"]
            assert rows["reconciled"] and rows["bytes"] == check["bytes"]
            assert sum(rows["mismatched"].values()) == 0
            if d_depth:
                draft = check["draft"]
                slices = counts[1] * SWA_PAGE
                assert draft["registered_slice_bytes"] == slices
                assert draft["payload_bytes"] == counts[1] * SWA_REAL
                assert draft["alignment_bytes"] == counts[1] * 64
                assert draft["read_bytes"] == slices
                assert draft["mismatched_payload_bytes"] == 0
            else:
                assert "draft" not in check
    for rank in range(p_tp):
        rows = rank_rows(tmp_path, "prefill", rank)
        done = [
            json.loads((tmp_path / f"prefill.{i}.json").read_text())["handoff"][
                "remote_request_id"
            ]
            for i in range(len(prompts))
        ]
        sent = {
            s for row in rows for s in row.get("kv_events", {}).get("sends_done", [])
        }
        assert set(done) <= sent, (rank, done, sent)
        assert not any(
            row.get("kv_transfers", {}).get("lease", {}).get("expired") for row in rows
        )


DRAFT_OFFSET = 21 * SWA_PAGE
CORRUPTIONS = [
    pytest.param(2, 4, None, 0, 3, {0, 1}, "target_payload", id="target-payload"),
    pytest.param(
        4, 2, 1, 1, DRAFT_OFFSET + 5, {0}, "draft_payload", id="draft-payload"
    ),
    pytest.param(
        4,
        2,
        1,
        1,
        DRAFT_OFFSET + SWA_REAL + 7,
        {0},
        "draft_alignment",
        id="draft-alignment",
    ),
    pytest.param(4, 2, 1, 1, 22 * SWA_PAGE + 11, {0}, "tail", id="row-tail"),
]


@pytest.mark.parametrize(
    ("p_tp", "d_tp", "depth", "group", "offset", "failing", "kind"), CORRUPTIONS
)
def test_a_corrupted_producer_byte_fails_exactly_the_ranks_that_read_it(
    tmp_path, p_tp, d_tp, depth, group, offset, failing, kind
):
    from vllm.v1.worker.simulated.v4_reads import owner_rank

    prompts = [prompt_of(600)]
    layer = "model.layers.1.attn.swa_cache" if group == 0 else DRAFT
    producer = settings_of(depth, False, prompts, corrupt=[group, layer, offset])
    consumer = settings_of(depth, False, prompts)
    assert run_pair(tmp_path, p_tp, d_tp, producer, consumer) == 0
    per_rank = decode_checks(tmp_path, d_tp)
    bad = {r for r in range(d_tp) if owner_rank(r, d_tp, p_tp) == 0}
    assert bad == failing
    for rank in range(d_tp):
        (check,) = per_rank[rank]
        assert check["ok"] is (rank not in failing), check
        assert check["mismatched_bytes"] == (1 if rank in failing else 0)
        if rank in failing:
            assert check["rows"]["mismatched"][kind] == 1
            assert sum(check["rows"]["mismatched"].values()) == 1


def test_mtp_on_one_side_only_is_refused_by_the_native_hash(tmp_path):
    prompts = [prompt_of(600)]
    rc = run_pair(
        tmp_path,
        2,
        4,
        settings_of(1, False, prompts),
        settings_of(None, False, prompts),
    )
    log = (tmp_path / "decode.log").read_text()
    assert "NIXL compatibility hash mismatch" in log, (rc, log[-3000:])
    assert all(
        c.get("error") == "receive failed"
        for rank in range(4)
        for c in checks(rank_rows(tmp_path, "decode", rank))
    )


@pytest.mark.parametrize("check", [True, False], ids=["checked", "unchecked"])
def test_prefix_mode_on_one_side_only_is_refused_by_the_capability(tmp_path, check):
    """Before any registration, with the byte checks on or off."""
    prompts = [prompt_of(600)]
    run_pair(
        tmp_path,
        2,
        4,
        settings_of(None, True, prompts, check=check),
        settings_of(None, False, prompts, check=check),
    )
    log = (tmp_path / "decode.log").read_text()
    assert "the peer's prefix is on, this instance's off" in log, log[-3000:]
    assert not any(
        c.get("ok")
        for rank in range(4)
        for c in checks(rank_rows(tmp_path, "decode", rank))
    )
