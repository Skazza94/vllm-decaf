# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""#6: a hybrid's MTP with prefix caching. States are keyed by their
endpoint, so a pulled state is the producer's whichever slot NIXL writes;
the checker's append-only block tables follow the scheduler's relocation of
speculative state blocks."""

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
