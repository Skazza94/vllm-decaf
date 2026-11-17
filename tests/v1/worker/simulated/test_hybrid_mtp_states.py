# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""#6: a hybrid's MTP with prefix caching. States are keyed by their
endpoint, so a pulled state is the producer's whichever slot NIXL writes;
the checker's append-only block tables follow the scheduler's relocation of
speculative state blocks."""

import json
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


def run_tiny_mtp_peers(tmp_path, p_tp, d_tp, k, consumer=None, **common):
    from tests.v1.worker.simulated.test_real_kv_transfer import (
        QWEN3_5_MOE_MTP,
        run_dcp_peers,
    )

    def side(tp, depth):
        return {
            "speculative_config": {"method": "mtp", "num_speculative_tokens": depth},
            "simulation": {"spec_acceptance_rates": [0.5] * depth},
            "language_model_only": True,
            "prompt_len": 40,
            "tensor_parallel_size": tp,
            **common,
        }

    return run_dcp_peers(
        QWEN3_5_MOE_MTP, side(p_tp, k), {**side(d_tp, k), **(consumer or {})}, tmp_path
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
            2 * (conv + SSM_BYTES // d_tp)
        ] * 2
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
    """Prefix caching on both, retention None, 16-token blocks: B shares 64
    of A's 70 tokens. Prefill's hit of B covers its state at logical block
    2 and three attention blocks (MTP drops the fourth); decode's covers the
    three attention blocks, its states pulled (#6's rules); every check ok."""
    a = [(7 * i) % 120 + 1 for i in range(70)]
    p, c = run_tiny_mtp_peers(
        tmp_path,
        p_tp,
        d_tp,
        1,
        prompts=[a, a[:64] + [3] * 10],
        enable_prefix_caching=True,
        prefix_cache_retention_interval=None,
        max_model_len=512,
    )
    assert c["tokens"] == [2, 2], c["log"][-4000:]
    for rank in range(p_tp):
        (hit,) = checks(p["logs"][rank], "hit")
        assert hit["ok"], hit
        assert [(g["blocks"], g["logical_blocks"]) for g in hit["groups"]] == [
            (1, [2]),
            (1, [2]),
            (3, [0, 1, 2]),
        ]
    for rank in range(d_tp):
        pulls = checks(c["logs"][rank])
        assert len(pulls) == 2 and all(x["ok"] for x in pulls), pulls
        (hit,) = checks(c["logs"][rank], "hit")
        assert hit["ok"] and [g["logical_blocks"] for g in hit["groups"]] == [
            [0, 1, 2]
        ], hit


def test_a_grained_hybrid_mtp_pull_at_unequal_tp_and_blocks(tmp_path, ds_conv_layout):
    """`kv_pattern_grain` with MTP (#24's option): prefill TP1 at 16-token
    blocks, decode TP2 at 32 reading two producer blocks a block. B shares
    64 of A's tokens on prefill, which hits them; decode pulls B alone, cold,
    and every rank's check passes with the content keyed at 16 tokens."""
    a = [(7 * i) % 120 + 1 for i in range(70)]
    common = {
        "prompts": [a, a[:64] + [3] * 10],
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
    assert hit["ok"] and hit["groups"][-1]["logical_blocks"] == [0, 1, 2], hit
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


def test_only_a_gdn_hybrids_mtp_pairs_unequal_tp(tmp_path, ds_conv_layout):
    """The widened layout is shared, MTP's qualification is not: a Mamba2
    hybrid's native MTP (tiny Nemotron-H with one MTP layer) keeps the state
    gate's refusal at unequal TP, and real transfers refuse its MTP outright
    (`REAL_KV_MTP_DRAFTS`)."""
    from tests.v1.worker.simulated.test_real_kv_transfer import (
        NEMOTRON_3,
        engine_config,
    )
    from vllm import EngineArgs
    from vllm.v1.worker.simulated import state_layout as sl

    config = {
        **NEMOTRON_3,
        "num_nextn_predict_layers": 1,
        "mtp_hybrid_override_pattern": "*E",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    mtp = {"method": "mtp", "num_speculative_tokens": 2}
    plain = EngineArgs(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=256,
        enforce_eager=True,
        speculative_config=mtp,
    ).create_engine_config()
    assert sl.layout_for(plain).conv_rows == 3 + 2
    assert sl.pairing_problem(plain, 2, 1) == "speculative decoding with a mamba2 state"
    with pytest.raises(ValueError, match="native single-module MTP"):
        engine_config(str(tmp_path), dtype="bfloat16", speculative_config=mtp)
