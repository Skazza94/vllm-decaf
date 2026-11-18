# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""#6: a hybrid's MTP with prefix caching. States are keyed by their
endpoint, so a pulled state is the producer's whichever slot NIXL writes;
the checker's append-only block tables follow the scheduler's relocation of
speculative state blocks."""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace

import pytest

from tests.v1.core.test_prefix_caching import (
    _make_hybrid_kv_cache_config,
    make_kv_cache_manager,
    make_request,
)
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.worker.simulated.kv_pattern import (
    STATE_IDENTITY,
    TokenChains,
    _ConsumerPages,
    _ExpectedGroup,
)

BLOCK = 16

def free_port():
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def chains(endpoint_states=True, **kwargs):
    return TokenChains(
        [BLOCK, BLOCK], state_groups=[1], endpoint_states=endpoint_states, **kwargs
    )


@pytest.mark.parametrize("prompt_len", [17, 32, 40, 47, 48])
def test_a_pulled_state_is_the_producers_whichever_slot(prompt_len):
    """The consumer expects the producer's running state at its handoff,
    whatever logical slot the read names (MTP's scratch slot included)."""
    prompt = list(range(5, 5 + prompt_len))
    producer, consumer = chains(), chains(backoff=1)
    producer.admit("p", prompt[:-1])
    consumer.admit("d", prompt)
    end = prompt_len - 1
    last = (end - 1) // BLOCK
    want = producer.prompt_content("p", 1, last)
    assert want.startswith(STATE_IDENTITY) and f"end{end}|" in want
    for slot in range(last, last + 3):
        assert consumer.pulled_content("d", 1, slot) == want
    # Attention keeps its blocks' content.
    assert consumer.pulled_content("d", 0, 0) == producer.prompt_content("p", 0, 0)


def test_states_keyed_by_endpoint_only_when_asked():
    prompt = list(range(64))
    plain, keyed = chains(endpoint_states=False), chains()
    plain.admit("r", prompt)
    keyed.admit("r", prompt)
    assert plain.identity("r", 1, 1, BLOCK).startswith("content|")
    assert keyed.identity("r", 1, 1, BLOCK).startswith(STATE_IDENTITY)
    assert plain.identity("r", 0, 1, BLOCK) == keyed.identity("r", 0, 1, BLOCK)
    # Another endpoint, another state.
    assert keyed.identity("r", 1, 1, BLOCK) != keyed.identity("r", 1, 2, BLOCK)


def test_a_grain_keys_endpoint_states_at_the_grain():
    prompt = list(range(64))
    grained = TokenChains(
        [32, 32], state_groups=[1], endpoint_states=True, grain=BLOCK
    )
    small = chains()
    grained.admit("r", prompt)
    small.admit("r", prompt)
    assert grained.identity("r", 1, 0, 32) == small.identity("r", 1, 1, BLOCK)


@dataclass
class Pages(_ConsumerPages):
    chains: TokenChains | None = None
    tables: dict = field(default_factory=dict)
    results: list = field(default_factory=list)
    step_writes: dict = field(default_factory=dict)
    hits_due: list = field(default_factory=list)
    pending: dict = field(default_factory=dict)
    awaiting: dict = field(default_factory=dict)
    pull_starts: dict = field(default_factory=dict)
    written: dict = field(default_factory=dict)
    write_events: list = field(default_factory=list)


def test_a_pulled_state_group_has_no_local_hit():
    """A state group the pull fills starts its hit at 0 (its earlier slots
    are null or the running slot the scheduler moves the state to); a group
    the pull leaves empty, and every group without endpoint states, keeps
    the pull's start."""
    groups = [
        _ExpectedGroup(0, 2, [7, 8], None),
        _ExpectedGroup(1, 2, [9], None),
        _ExpectedGroup(2, 2, [NULL_BLOCK_ID], None),
    ]
    three = TokenChains([BLOCK] * 3, state_groups=[1, 2], endpoint_states=True)
    pages = Pages(chains=three)
    pages._record_pull_start("r", groups)
    assert pages.pull_starts["r"] == {0: 2, 1: 0, 2: 2}
    plain = Pages(chains=TokenChains([BLOCK] * 3, state_groups=[1, 2]))
    plain._record_pull_start("r", groups)
    assert plain.pull_starts["r"] == {0: 2, 1: 2, 2: 2}


@pytest.mark.parametrize("accepted", [[1] * 30, [0] * 30, [1, 0, 0, 1, 1, 0] * 5])
def test_the_checker_tables_follow_the_scheduler_through_relocation(accepted):
    """The native cache manager (align mode, one speculative state block,
    MTP's lookahead, every checkpoint kept) relocates exclusively owned
    scratch blocks to its table's end and nulls their place; the checker's
    tables, built only from each step's new block ids as the worker's are,
    hold the scheduler's block at every place it keeps one (so a checkpoint
    it caches there is the block `_materialize` writes) and at the running
    slot, `cdiv(computed + scheduled, block) - 1`. Checkpoint caching and its
    hit are exercised on the engine
    (`test_nixl_peers_check_a_hybrids_decode_written_state_hit_with_mtp`)."""
    init_none_hash(sha256)
    config = _make_hybrid_kv_cache_config(BLOCK, 200, ["full", "mamba"])
    config.kv_cache_groups[1].kv_cache_spec = replace(
        config.kv_cache_groups[1].kv_cache_spec, num_speculative_blocks=1
    )
    manager = make_kv_cache_manager(
        kv_cache_config=config,
        max_model_len=1024,
        enable_caching=True,
        hash_block_size=BLOCK,
        retention_interval=None,
        use_eagle=True,
        num_hash_lookahead_tokens=1,
    )
    mamba = manager.coordinator.single_type_managers[1]
    request = make_request(
        "r", list(range(40)), BLOCK, sha256, num_hash_lookahead_tokens=1
    )
    pages = Pages(chains=chains())
    relocated = False

    def step(num_new: int, lookahead: int) -> None:
        nonlocal relocated
        before = list(mamba.req_to_blocks[request.request_id])
        blocks = manager.allocate_slots(
            request, num_new, num_lookahead_tokens=lookahead
        )
        assert blocks is not None
        ids = blocks.get_block_ids()
        pages._track(request.request_id, ids, replace=not pages.tables)
        after = mamba.req_to_blocks[request.request_id]
        relocated |= any(
            not a.is_null and b.is_null for a, b in zip(before, after)
        )
        table = pages.tables[request.request_id][1]
        assert len(table) == len(after)
        for place, block in enumerate(after):
            if not block.is_null:
                assert table[place] == block.block_id, (place, table, after)
        slot = -(-(request.num_computed_tokens + num_new) // BLOCK) - 1
        assert not after[slot].is_null and table[slot] == after[slot].block_id

    step(len(request.all_token_ids), 0)
    request.num_computed_tokens = len(request.all_token_ids)
    request.append_output_token_ids(0)
    for accept in accepted:
        # A decode step: the sampled token and one draft.
        step(2, 1)
        request.num_computed_tokens += 1 + accept
        request.append_output_token_ids([0] * (1 + accept))
    assert relocated, "no speculative block was relocated"


# #33: the same native MTP between instances of unequal TP sizes. The tiny
# Qwen3.5's state: conv columns Q 32, K 32, V 64 of `kernel - 1 + k` bf16
# rows (kernel 4), four 16x16 bf16 SSM heads; one attention layer and the
# draft's, 2 KV heads of 32 dims (a 16-token block page of both heads).
CONV_COLUMNS, SSM_BYTES, HEAD_PAGE = 128, 4 * 16 * 16 * 2, 16 * 2 * 32 * 2 * 2


def run_tiny_mtp_peers(tmp_path, p_tp, d_tp, k, consumer=None, config=None, **common):
    from tests.v1.worker.simulated.test_real_kv_transfer import (
        QWEN3_5_MOE_MTP,
        run_dcp_peers,
    )

    config = QWEN3_5_MOE_MTP if config is None else config

    def side(tp, depth):
        options = {
            "speculative_config": {"method": "mtp", "num_speculative_tokens": depth},
            "simulation": {"spec_acceptance_rates": [0.5] * depth},
            "prompt_len": 40,
            "tensor_parallel_size": tp,
            **common,
        }
        # Qwen3.5 is a multimodal wrapper served as its language model; the
        # plain text hybrids need no such override.
        if "text_config" in config:
            options["language_model_only"] = True
        return options

    return run_dcp_peers(
        config, side(p_tp, k), {**side(d_tp, k), **(consumer or {})}, tmp_path
    )


def checks(records, kind=None):
    return [
        r["kv_check"]
        for r in records
        if "kv_check" in r and r["kv_check"].get("kind") == kind
    ]


def layout_groups(records):
    (layout,) = [r["kv_layout"] for r in records if "kv_layout" in r][:1]
    return layout["groups"]


@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("p_tp,d_tp", [(1, 2), (2, 1)])
def test_a_hybrids_mtp_states_re_shard_at_unequal_tp(
    tmp_path, ds_conv_layout, p_tp, d_tp, k
):
    """Every decode rank reads its columns and heads of the producer's
    widened states (`kernel - 1 + k` conv rows, as vLLM allocates them) and
    the draft's attention layer beside the target's; each byte checks."""
    p, c = run_tiny_mtp_peers(tmp_path, p_tp, d_tp, k)
    assert c["tokens"] == [2], c["log"][-4000:]
    assert c["posted"] and all(c["posted"]), c["posted"]
    conv = CONV_COLUMNS // d_tp * (3 + k) * 2
    for rank in range(d_tp):
        groups = layout_groups(c["logs"][rank])
        states = [g for g in groups if g["kind"] == "MambaSpec"]
        (attention,) = [g for g in groups if g["kind"] == "FullAttentionSpec"]
        assert [g["state"]["conv_rows"] for g in states] == [3 + k] * 2
        assert [g["speculative_blocks"] for g in states] == [k] * 2
        assert attention["layers"] == 2
        (check,) = checks(c["logs"][rank])
        assert check["ok"] and check["mismatched_bytes"] == 0, check
        assert check["coverage"]["problems"] == [], check["coverage"]
        by_pattern = {}
        for group in check["groups"]:
            by_pattern.setdefault(group["pattern"], []).append(group)
        # Each group's regions: its layers', vLLM's padding layer beside the
        # one-layer group's.
        assert [g["bytes"] for g in by_pattern["state-v1"]] == [
            g["layers"] * (conv + SSM_BYTES // d_tp) for g in by_pattern["state-v1"]
        ]
        assert sorted(g["layers"] for g in by_pattern["state-v1"]) == [1, 2]
        (pulled,) = by_pattern["head-v1"]
        assert pulled["bytes"] == 3 * 2 * HEAD_PAGE // d_tp


@pytest.mark.parametrize("mode", ["conv-spec", "ssm"])
@pytest.mark.parametrize("p_tp,d_tp", [(1, 2), (2, 1)])
def test_a_flipped_widened_state_byte_fails_its_reader_only(
    tmp_path, ds_conv_layout, p_tp, d_tp, mode
):
    """A byte of a re-sharded state flipped after NIXL wrote it, in a conv
    row only MTP adds or in the SSM state: the reading rank's check fails in
    that state group alone; every other rank's passes."""
    target = d_tp - 1
    p, c = run_tiny_mtp_peers(
        tmp_path, p_tp, d_tp, 2, consumer={"corrupt": [mode, target]}
    )
    assert "corrupted" in c, c["log"][-4000:]
    (corrupted,) = [x for x in c["corrupted"] if x]
    _, group, _ = corrupted
    for rank in range(d_tp):
        (check,) = checks(c["logs"][rank])
        if rank != target:
            assert check["ok"], check
            continue
        assert not check["ok"], check
        bad = [g["group"] for g in check["groups"] if g["mismatched_bytes"]]
        assert bad == [group], check


@pytest.mark.parametrize("p_tp,d_tp", [(1, 2), (2, 1)])
def test_a_hybrids_mtp_prefix_hits_at_unequal_tp(tmp_path, ds_conv_layout, p_tp, d_tp):
    """Prefix caching on both, retention None, 16-token blocks: B shares 65
    of A's 70 tokens, the 64 of four blocks and the token after them: the
    connector's lookahead block hashes make a block's hash cover the next
    token, so a prefix ending on a block boundary does not hit its last
    block. Prefill's hit of B covers its state at logical block 3 and four
    attention blocks; decode's covers the four attention blocks, its states
    pulled (#6's rules); every check ok."""
    a = [(7 * i) % 120 + 1 for i in range(70)]
    p, c = run_tiny_mtp_peers(
        tmp_path,
        p_tp,
        d_tp,
        1,
        prompts=[a, a[:65] + [3] * 9],
        enable_prefix_caching=True,
        prefix_cache_retention_interval=None,
        max_model_len=512,
    )
    assert c["tokens"] == [2, 2], c["log"][-4000:]
    for rank in range(p_tp):
        (hit,) = checks(p["logs"][rank], "hit")
        assert hit["ok"], hit
        assert [(g["blocks"], g["logical_blocks"]) for g in hit["groups"]] == [
            (1, [3]),
            (1, [3]),
            (4, [0, 1, 2, 3]),
        ]
    for rank in range(d_tp):
        pulls = checks(c["logs"][rank])
        assert len(pulls) == 2 and all(x["ok"] for x in pulls), pulls
        (hit,) = checks(c["logs"][rank], "hit")
        assert hit["ok"] and [g["logical_blocks"] for g in hit["groups"]] == [
            [0, 1, 2, 3]
        ], hit


def test_a_grained_hybrid_mtp_pull_at_unequal_tp_and_blocks(tmp_path, ds_conv_layout):
    """`kv_pattern_grain` with MTP (#24's option): prefill TP1 at 16-token
    blocks, decode TP2 at 32 reading two producer blocks a block. B shares
    65 of A's tokens on prefill, which hits 64 of them; decode pulls B alone, cold,
    and every rank's check passes with the content keyed at 16 tokens."""
    a = [(7 * i) % 120 + 1 for i in range(70)]
    common = {
        "prompts": [a, a[:65] + [3] * 9],
        "enable_prefix_caching": True,
        "prefix_cache_retention_interval": None,
        "max_model_len": 512,
    }
    grain = {"spec_acceptance_rates": [0.5], "kv_pattern_grain": 16}
    p, c = run_tiny_mtp_peers(
        tmp_path,
        1,
        2,
        1,
        consumer={"block_size": 32, "consume": [1], "simulation": grain},
        block_size=16,
        **common,
        simulation=grain,
    )
    assert c["tokens"] == [2], c["log"][-4000:]
    (hit,) = checks(p["logs"][0], "hit")
    assert hit["ok"] and hit["groups"][-1]["logical_blocks"] == [0, 1, 2, 3], hit
    for rank in range(2):
        assert checks(c["logs"][rank], "hit") == []
        (pull,) = checks(c["logs"][rank])
        assert pull["ok"] and pull["coverage"]["problems"] == [], pull
        (attention,) = [g for g in pull["groups"] if g["pattern"] == "head-v1"]
        assert attention["block_size_ratio"] == 2, attention


@pytest.mark.parametrize("check", [True, False], ids=["checked", "unchecked"])
@pytest.mark.parametrize("p_tp,d_tp", [(1, 2), (2, 1)])
def test_unequal_tp_peers_of_other_mtp_depths_are_refused_before_a_read(
    tmp_path, ds_conv_layout, p_tp, d_tp, check
):
    """Depths 2 and 3 widen the conv state differently: decode's handshake
    compares the producer's state sizes with the model's at its TP and
    refuses it, so no READ is posted, bytes checked or not."""
    p, c = run_tiny_mtp_peers(
        tmp_path,
        p_tp,
        d_tp,
        2,
        consumer={
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
            "simulation": {"spec_acceptance_rates": [0.5] * 3},
            "check": check,
        },
    )
    assert "holds its states otherwise" in c["log"]
    assert c["posted"] == [0] * d_tp


# #33's Nemotron-H: the same native MTP on a Mamba2 hybrid. The tiny
# Nemotron-3 plus one attention + MoE MTP module; its Mamba2 states' conv
# columns x·B·C are 64 + 32 + 32 of `kernel - 1 + k` rows (kernel 4, bf16),
# four 8x16 FP32 SSM heads (NemotronH's `mamba_ssm_cache_dtype` resolves
# float32), 2 KV heads of 16 dims, and 48-token attention blocks at depth 1.
NEMOTRON_CONV_COLUMNS, NEMOTRON_SSM_BYTES, NEMOTRON_HEAD_PAGE = (
    128,
    8 * 8 * 16 * 4,
    48 * 2 * 16 * 2 * 2,
)


def nemotron_mtp_config():
    from tests.v1.worker.simulated.test_real_kv_transfer import NEMOTRON_3

    return {
        **NEMOTRON_3,
        "num_nextn_predict_layers": 1,
        "mtp_hybrid_override_pattern": "*E",
    }


@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("p_tp,d_tp", [(1, 2), (2, 1)])
def test_a_mamba2_hybrids_mtp_states_re_shard_at_unequal_tp(
    tmp_path, ds_conv_layout, p_tp, d_tp, k
):
    """Nemotron-H's native MTP pairs unequal TP like Qwen3.5's: every decode
    rank reads its columns and heads of the producer's widened Mamba2 states
    (`kernel - 1 + k` conv rows, as `mamba2_state_shape` allocates them) and
    the draft's attention layer beside the target's; each byte checks."""
    p, c = run_tiny_mtp_peers(tmp_path, p_tp, d_tp, k, config=nemotron_mtp_config())
    assert c["tokens"] == [2], c["log"][-4000:]
    assert c["posted"] and all(c["posted"]), c["posted"]
    conv = NEMOTRON_CONV_COLUMNS // d_tp * (3 + k) * 2
    for rank in range(d_tp):
        groups = layout_groups(c["logs"][rank])
        states = [g for g in groups if g["kind"] == "MambaSpec"]
        (attention,) = [g for g in groups if g["kind"] == "FullAttentionSpec"]
        assert [g["state"]["conv_rows"] for g in states] == [3 + k] * len(states)
        assert [g["speculative_blocks"] for g in states] == [k] * len(states)
        assert attention["layers"] == 2
        (check,) = checks(c["logs"][rank])
        assert check["ok"] and check["mismatched_bytes"] == 0, check
        assert check["coverage"]["problems"] == [], check["coverage"]
        by_pattern = {}
        for group in check["groups"]:
            by_pattern.setdefault(group["pattern"], []).append(group)
        # Each state group's regions: its layers', the layers per group the
        # engine picked (attention and Mamba2 layers beside each other's).
        state_bytes = conv + NEMOTRON_SSM_BYTES // d_tp
        for pattern, members in by_pattern.items():
            if pattern != "state-v1":
                continue
            for group in members:
                assert group["bytes"] == group["layers"] * state_bytes, group
        (pulled,) = by_pattern["head-v1"]
        # The attention group's two layers (the target's and the draft's),
        # each one 48-token page of both heads, K and V beside each other.
        assert pulled["bytes"] == 2 * NEMOTRON_HEAD_PAGE // d_tp, pulled


@pytest.mark.parametrize("mode", ["conv-spec", "ssm"])
@pytest.mark.parametrize("p_tp,d_tp", [(1, 2), (2, 1)])
def test_a_flipped_mamba2_widened_state_byte_fails_its_reader_only(
    tmp_path, ds_conv_layout, p_tp, d_tp, mode
):
    """Mamba2's, like Qwen3.5's: a byte of a re-sharded state flipped after
    NIXL wrote it, in a conv row only MTP adds or in the SSM state, fails
    exactly the rank that reads it."""
    target = d_tp - 1
    p, c = run_tiny_mtp_peers(
        tmp_path,
        p_tp,
        d_tp,
        2,
        consumer={"corrupt": [mode, target]},
        config=nemotron_mtp_config(),
    )
    assert "corrupted" in c, c["log"][-4000:]
    (corrupted,) = [x for x in c["corrupted"] if x]
    _, group, _ = corrupted
    for rank in range(d_tp):
        (check,) = checks(c["logs"][rank])
        if rank != target:
            assert check["ok"], check
            continue
        assert not check["ok"], check
        bad = [g["group"] for g in check["groups"] if g["mismatched_bytes"]]
        assert bad == [group], check


@pytest.mark.parametrize("p_tp,d_tp", [(1, 2), (2, 1)])
def test_a_mamba2_hybrids_mtp_prefix_hits_at_unequal_tp(
    tmp_path, ds_conv_layout, p_tp, d_tp
):
    """Mamba2 blocks, with prefix caching: Nemotron-3's tiny states key at
    48-token blocks, and a prefix must reach past the lookahead hash's token.
    B shares 150 of A's 170 tokens; both directions hit 3 attention blocks
    and the state at the third, and every check passes: the pulls and the
    hit among decode's, one hit of prefill's."""
    a = [(7 * i) % 120 + 1 for i in range(170)]
    p, c = run_tiny_mtp_peers(
        tmp_path,
        p_tp,
        d_tp,
        1,
        config=nemotron_mtp_config(),
        prompts=[a, a[:150] + [3] * 10],
        enable_prefix_caching=True,
        prefix_cache_retention_interval=None,
        max_model_len=512,
    )
    assert c["tokens"] == [2, 2], c["log"][-4000:]
    for rank in range(p_tp):
        (hit,) = checks(p["logs"][rank], "hit")
        assert hit["ok"], hit
        # The one state block B's shared prefix reaches (its 150 tokens
        # cover 48 x 3 and the token after), both Mamba2 layers; and the
        # three attention blocks, both attention layers.
        want = (10240, 36864) if p_tp == 1 else (5120, 18432)
        assert [g["bytes"] for g in hit["groups"]] == list(want), hit
        assert [(g["blocks"], g["logical_blocks"]) for g in hit["groups"]] == [
            (1, [2]),
            (3, [0, 1, 2]),
        ]
    for rank in range(d_tp):
        pulls = checks(c["logs"][rank])
        assert len(pulls) == 2 and all(x["ok"] for x in pulls), pulls
        (hit,) = checks(c["logs"][rank], "hit")
        assert hit["ok"], hit
        # Decode's hit covers what its pulls left: the attention group's
        # three blocks, both layers; its state groups
        # the pulls filled have no hit of decode's own (#6's rule).
        assert 3 * 2 * NEMOTRON_HEAD_PAGE // d_tp == hit["bytes"], hit
        assert [(g["blocks"], g["logical_blocks"]) for g in hit["groups"]] == [
            (3, [0, 1, 2]),
        ]


@pytest.mark.parametrize("check", [True, False], ids=["checked", "unchecked"])
@pytest.mark.parametrize("p_tp,d_tp", [(1, 2), (2, 1), (1, 1)])
def test_mamba2_mtp_peers_of_other_depths_are_refused_before_a_read(
    tmp_path, ds_conv_layout, p_tp, d_tp, check
):
    """Nemotron-H at unequal TP and at equal: a producer of depth 2 against
    a consumer of depth 3 holds states of other byte sizes; the
    state-size comparison refuses it before any READ, checked or not."""
    p, c = run_tiny_mtp_peers(
        tmp_path,
        p_tp,
        d_tp,
        2,
        config=nemotron_mtp_config(),
        consumer={
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
            "simulation": {"spec_acceptance_rates": [0.5] * 3},
            "check": check,
        },
    )
    assert "holds its states otherwise" in c["log"]
    assert c["posted"] == [0] * d_tp




def test_a_mamba2_hybrids_mtp_pairs_unequal_tp_its_kda_and_replayssm_do_not(
    tmp_path, tmp_path_factory, ds_conv_layout
):
    """The widened layout is shared, MTP's pairing is not arbitrary: the
    Nemotron-H Mamba2 hybrid's native MTP pairs as its without-MTP rule
    pairs (NIXL on, Mooncake off); Kimi-K3's KDA carries no draft the
    connector knows (`REAL_KV_MTP_DRAFTS` refuses the configuration) and its
    `pairing_problem` keeps MTP refused where the state layout is reached;
    and ReplaySSM stays outside real transfers."""
    from copy import deepcopy
    from types import SimpleNamespace

    from tests.v1.worker.simulated.test_real_kv_transfer import (
        KIMI_K3_8_HEADS,
        engine_config,
        write_model_dir,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
        simulated_tp_mismatch,
    )
    from vllm.v1.worker.simulated import state_layout as sl

    mtp = {"method": "mtp", "num_speculative_tokens": 2}
    model_dir = write_model_dir(
        tmp_path_factory, "tiny-nemotron-mtp-scope", nemotron_mtp_config()
    )
    plain = engine_config(str(model_dir), dtype="bfloat16", speculative_config=mtp)
    assert sl.layout_for(plain).conv_rows == 3 + 2
    # The Mamba2 hybrid pairs by the rule its without-MTP pairing uses.
    assert sl.pairing_problem(plain, 2, 1) is None
    assert simulated_tp_mismatch(plain, 2, 1, nixl_consumer=True) is None
    # Mooncake's connector re-shards no hybrid state, MTP or not.
    assert "unequal TP sizes" in simulated_tp_mismatch(plain, 2, 1)
    # Kimi-K3: the configuration gate still refuses its MTP (its KDA state
    # has no draft the connector knows), and the pairing gate's own KDA
    # line keeps the rule for the layout reached behind it.
    k3 = write_model_dir(tmp_path_factory, "tiny-kimi-k3-mtp-scope", KIMI_K3_8_HEADS)
    # vLLM builds Kimi-Linear a draft of its own architecture, not a native
    # MTP module, so the simulated configuration refuses it; the pairing
    # gate's own KDA line keeps the rule for the layout reached behind such
    # a configuration.
    with pytest.raises(ValueError, match="native single-module MTP.*kimi_linear"):
        engine_config(str(k3), dtype="bfloat16", speculative_config=mtp)
    config = engine_config(str(k3), dtype="bfloat16", tensor_parallel_size=8)
    assert sl.layout_for(config).family == "kda"
    with_spec = deepcopy(config)
    with_spec.speculative_config = SimpleNamespace(
        method="mtp", num_speculative_tokens=1
    )
    assert (
        sl.pairing_problem(with_spec, 4, 8) == "speculative decoding with a kda state"
    )
    # ReplaySSM: native Mamba2 replay state beside MTP, over a connector.
    with pytest.raises(ValueError, match="ReplaySSM|replayssm|Replay"):
        engine_config(
            str(model_dir),
            dtype="bfloat16",
            speculative_config=mtp,
            use_replayssm=True,
        )
    from tests.v1.worker.simulated.test_kv_pattern import (
        decoder,
        mamba2_mixer,
        recurrent_model,
    )
    from vllm.v1.worker.simulated.kv_transfer import check_real_kv_layers

    with pytest.raises(ValueError, match="ReplaySSM only"):
        check_real_kv_layers(
            recurrent_model([decoder(mixer=mamba2_mixer(use_replayssm=True))])
        )


@pytest.mark.parametrize(
    "acceptance, corrupt",
    [(1.0, False), (0.0, False), (1.0, True)],
    ids=["accepted", "rejected", "corrupt-state"],
)
def test_nixl_peers_check_a_mamba2_hybrids_decode_written_state_hit_with_mtp(
    tmp_path_factory, tmp_path, ds_conv_layout, acceptance, corrupt
):
    """#6 for Nemotron-H (tiny, MTP k 1, 48-token blocks): A (100 tokens) is
    pulled, its state keyed by its endpoint whichever slot NIXL writes, and
    decode generates 96 tokens, every state checkpoint kept (retention
    None). B, A's prompt, decode's 96 generated tokens and 10 more, runs on
    decode alone: its native hit (four matched blocks, none dropped for MTP:
    the connector's lookahead block hashes make the trailing block stable)
    reaches the state at 192 tokens, two blocks decode's own steps wrote
    past A's handoff, and compares equal with drafts accepted or rejected;
    with that state's block (logical 3) flipped after A, B's hit fails
    exactly there."""
    from tests.v1.worker.simulated.test_real_kv_transfer import (
        NIXL_PREFIX_PEER,
        write_model_dir,
    )

    model = write_model_dir(
        tmp_path_factory, "tiny-nemotron-mtp-prefix", nemotron_mtp_config()
    )
    a = [(7 * i) % 120 + 1 for i in range(100)]
    b = a + [0] * 96 + [3] * 10
    prompts, max_tokens = json.dumps([a, b]), json.dumps([96, 2])
    spec = {
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 1},
        "simulation": {"spec_acceptance_rates": [acceptance]},
        "prefix_cache_retention_interval": None,
    }
    consumer_kwargs = {
        **spec,
        "local_prompts": [1],
        **({"corrupt_state_block": 3} if corrupt else {}),
    }
    script = tmp_path / "peer.py"
    script.write_text(NIXL_PREFIX_PEER)
    env = {
        **os.environ,
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_HOST_IP": "127.0.0.1",
        "NIXL_MOONCAKE_IP_ADDR": "127.0.0.1",
        "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
        "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
    }
    produced, consumed = tmp_path / "producer.json", tmp_path / "consumer.json"
    producer = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "producer",
            model,
            json.dumps(spec),
            str(produced),
            str(64 * 1024**2),
            json.dumps([a]),
        ],
        env={**env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(free_port())},
    )
    try:
        deadline = time.monotonic() + 600
        while not produced.exists():
            assert producer.poll() is None, "the producer exited"
            assert time.monotonic() < deadline, "no producer output"
            time.sleep(0.5)
        handoffs = json.loads(produced.read_text())
        # B has no handoff: decode serves it from its own cache.
        remote = tmp_path / "remote.json"
        remote.write_text(json.dumps({"params": handoffs["params"] + [None]}))
        consumer = subprocess.run(
            [
                sys.executable,
                str(script),
                "consumer",
                model,
                json.dumps(consumer_kwargs),
                str(consumed),
                str(64 * 1024**2),
                prompts,
                str(remote),
                max_tokens,
            ],
            env={**env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(free_port())},
            timeout=600,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert consumer.returncode == 0, consumer.stdout[-4000:]
    finally:
        (tmp_path / "producer.json.done").touch()
        try:
            producer.wait(timeout=60)
        finally:
            producer.kill()
    result = json.loads(consumed.read_text())
    assert result["outputs"][0] == [0] * 96
    steps = [json.loads(line) for line in (tmp_path / "consumer.json.steps").open()]
    (layout,) = [r["kv_layout"] for r in steps if "kv_layout" in r][:1]
    states = {i for i, g in enumerate(layout["groups"]) if g["kind"] == "MambaSpec"}
    assert states
    assert all(layout["groups"][g]["speculative_blocks"] == 1 for g in states)
    drafts = [r["spec_acceptance"] for r in steps if "spec_acceptance" in r]
    assert drafts, "no draft verified"
    # Each admission's events name the step record that scheduled it.
    scheduled = {
        r["step"]["step_index"]: {q["req_id"] for q in r["step"]["requests"]}
        for r in steps
        if "step" in r
    }
    admitted = [r for r in steps if (r.get("kv_events") or {}).get("admitted")]
    assert len(admitted) == 2
    for record in admitted:
        assert set(record["kv_events"]["admitted"]) <= scheduled[record["step_index"]]
    c_checks = json.loads((tmp_path / "consumer.json.checks").read_text())
    pulls = [c for c in c_checks if "kind" not in c]
    assert len(pulls) == 1 and pulls[0]["ok"], pulls
    (hit,) = [c for c in c_checks if c.get("kind") == "hit"]
    by_group = {g["group"]: g for g in hit["groups"]}
    # The hit reaches a state block decode's own step wrote past A's
    # handoff (A's 100 tokens close block 1: 48 x 2): A computed 195 of its
    # 196 tokens, four full blocks, so the state at 192 (block 3).
    for group in states:
        assert by_group[group]["logical_blocks"][-1] == 3, hit
    if corrupt:
        assert not hit["ok"], hit
        assert {m[2] for m in hit["mismatched_blocks"]} == {result["corrupted"]}
    else:
        assert all(c["ok"] for c in c_checks), c_checks
