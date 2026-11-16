# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`kv_pattern_grain`: content identities shared by a hybrid P/D pair whose
decode attention block is a multiple of prefill's, with prefix caching."""

import pytest
import torch

from tests.v1.worker.simulated.test_real_kv_transfer import (
    NEMOTRON_3,
    engine_config,
    write_model_dir,
)
from vllm.v1.worker.simulated.kv_pattern import (
    STATE_IDENTITY,
    TokenChains,
    attention_zeros,
    check_pattern_grain,
    row_pattern,
)

LLAMA = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 128,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "max_position_embeddings": 256,
    "torch_dtype": "float32",
}
GRAIN = 8
RATIO = 2
HEADS = (2, 3)
SLAB = 6
NAME = "g0.r0.o0"


def pair(prompt, backoff=1):
    """A producer at the grain and a consumer at twice it, each an
    attention group and a state group of its block size (align mode)."""
    producer = TokenChains([GRAIN, GRAIN], grain=GRAIN, state_groups=[1])
    consumer = TokenChains(
        [GRAIN * RATIO, GRAIN * RATIO], backoff=backoff, grain=GRAIN, state_groups=[1]
    )
    # The producer's prompt is its handoff (native NIXL truncates it).
    producer.admit("p", prompt[: len(prompt) - backoff])
    consumer.admit("d", prompt)
    return producer, consumer


def producer_block(chains, block, end):
    """A producer block's bytes as its filler writes them, head by head."""
    n = min(GRAIN, end - block * GRAIN)
    identity = chains.identity("p", 0, block, n)
    return row_pattern("p", NAME, block, len(HEADS) * SLAB, 0, identity, HEADS)


@pytest.mark.parametrize("prompt_len", [1, 8, 9, 15, 16, 17, 24, 31, 33])
def test_pulled_pieces_are_the_producer_blocks(prompt_len):
    """A decode block's slab of each head is its producer blocks' slabs of
    that head in token order, a piece starting at or past the handoff zero;
    a piece the handoff ends inside is the producer's whole patterned
    block, as NIXL moves it."""
    prompt = list(range(100, 100 + prompt_len))
    producer, consumer = pair(prompt)
    end = prompt_len - 1
    for logical in range(-(-prompt_len // (GRAIN * RATIO))):
        identity = consumer.pulled_content("d", 0, logical)
        assert isinstance(identity, tuple) and len(identity) == RATIO
        got = row_pattern(
            "d", NAME, logical, len(HEADS) * SLAB * RATIO, 0, identity, HEADS
        ).reshape(len(HEADS), RATIO, SLAB)
        for k in range(RATIO):
            block = logical * RATIO + k
            if block * GRAIN >= end:
                assert identity[k] is None
                assert not got[:, k].any()
            else:
                want = producer_block(producer, block, end).reshape(len(HEADS), SLAB)
                assert torch.equal(got[:, k], want)


def test_a_block_is_the_same_pulled_computed_or_hit():
    """Full decode blocks: the identity a pull expects is the one decode
    writes when it computes the block and compares when a later request
    hits it, and a later request sharing the prefix expects the same."""
    prompt = list(range(7, 7 + 40))
    _, consumer = pair(prompt)
    pulled = consumer.pulled_content("d", 0, 0)
    computed = consumer.identity("d", 0, 0, GRAIN * RATIO)
    assert pulled == computed and None not in pulled
    consumer.admit("d2", prompt[:16] + [1, 2, 3])
    assert consumer.identity("d2", 0, 0, GRAIN * RATIO) == computed
    other = TokenChains([GRAIN * RATIO], grain=GRAIN)
    other.admit("x", prompt[:15] + [0])
    assert other.identity("x", 0, 0, GRAIN * RATIO) != computed


@pytest.mark.parametrize("end", [1, 7, 8, 9, 16, 17, 23])
def test_a_state_is_keyed_by_its_endpoint_on_either_side(end):
    """A state has one identity for the tokens it absorbed, whichever block
    of either block size holds it; another endpoint is another state."""
    prompt = list(range(50, 90))
    producer, consumer = pair(prompt, backoff=0)
    on_p = producer.identity("p", 1, (end - 1) // GRAIN, (end - 1) % GRAIN + 1)
    big = GRAIN * RATIO
    on_d = consumer.identity("d", 1, (end - 1) // big, (end - 1) % big + 1)
    assert on_p == on_d and on_p.startswith(STATE_IDENTITY)
    assert on_p != consumer.identity("d", 1, end // big, end % big + 1)


@pytest.mark.parametrize("piece", [0, 1])
def test_one_wrong_byte_in_a_piece_is_detected(piece):
    """Corrupting one byte of either piece (a hit's or a pull's) changes the
    comparison, and only that piece's slab."""
    prompt = list(range(64))
    _, consumer = pair(prompt)
    identity = consumer.identity("d", 0, 1, GRAIN * RATIO)
    want = row_pattern("d", NAME, 1, len(HEADS) * SLAB * RATIO, 0, identity, HEADS)
    got = want.clone().reshape(len(HEADS), RATIO, SLAB)
    got[1, piece, 3] ^= 0x5A
    bad = (got.flatten() != want).reshape(len(HEADS), RATIO, SLAB)
    assert int(bad.sum()) == 1 and bad[1, piece].any()


def test_another_producer_block_in_a_sub_block_is_detected():
    """The pinned planning's shift (a sub-block holding the producer block
    one earlier) never matches the content expectation."""
    prompt = list(range(200, 200 + 25))
    producer, consumer = pair(prompt)
    end = len(prompt) - 1
    want = row_pattern(
        "d",
        NAME,
        1,
        len(HEADS) * SLAB * RATIO,
        0,
        consumer.pulled_content("d", 0, 1),
        HEADS,
    ).reshape(len(HEADS), RATIO, SLAB)
    shifted = producer_block(producer, 1, end).reshape(len(HEADS), SLAB)
    assert not torch.equal(want[:, 0], shifted)
    # Past the handoff the content expects zero, whatever NIXL wrote there.
    assert not want[:, 1].any()


def test_grain_matches_each_side():
    """The producer's attention blocks are the grain; a consumer's are a
    multiple of it; states are not attention."""
    check_pattern_grain(TokenChains([8, 8], grain=8, state_groups=[1]), True)
    check_pattern_grain(TokenChains([16, 16], grain=8, state_groups=[1]), False)
    with pytest.raises(ValueError, match="equal to it on the producer"):
        check_pattern_grain(TokenChains([16, 16], grain=8, state_groups=[1]), True)
    with pytest.raises(ValueError, match="a multiple of it on the consumer"):
        check_pattern_grain(TokenChains([12, 12], grain=8, state_groups=[1]), False)


@pytest.fixture(scope="module")
def dense_dir(tmp_path_factory):
    return write_model_dir(tmp_path_factory, "grain-llama", LLAMA)


@pytest.fixture(scope="module")
def hybrid_dir(tmp_path_factory):
    return write_model_dir(tmp_path_factory, "grain-nemotron-3", NEMOTRON_3)


def test_grain_needs_the_pattern_flags(dense_dir):
    with pytest.raises(ValueError, match="needs kv_pattern_fill or kv_pattern_check"):
        engine_config(dense_dir, {"kv_pattern_grain": 32})
    with pytest.raises(ValueError, match="greater than 0"):
        engine_config(dense_dir, {"kv_pattern_grain": 0, "kv_pattern_check": True})


def test_grain_only_for_a_hybrid_with_prefix_caching(dense_dir, tmp_path):
    simulation = {
        "kv_pattern_check": True,
        "kv_pattern_grain": 32,
        "log_path": str(tmp_path / "s"),
    }
    with pytest.raises(ValueError, match="kv_pattern_grain only for a hybrid"):
        engine_config(dense_dir, simulation, enable_prefix_caching=True)


def test_grain_admits_a_hybrid_consumer_with_prefix_caching(
    hybrid_dir, ds_conv_layout, tmp_path
):
    simulation = {
        "kv_pattern_check": True,
        "kv_pattern_grain": 16,
        "log_path": str(tmp_path / "steps"),
    }
    config = engine_config(hybrid_dir, simulation, enable_prefix_caching=True)
    assert config.simulation_config.kv_pattern_grain == 16
    with pytest.raises(ValueError, match="kv_pattern_grain only for a hybrid"):
        engine_config(hybrid_dir, simulation, enable_prefix_caching=False)


def test_zero_accounting_keeps_native_and_content_apart():
    """A cold pull's uncovered last piece is both zeroed by NIXL and due
    zero; the pinned pairing's shifted tail writes it (nothing zeroed)
    while the content still expects zero; without a grain they agree."""
    size, pieces = 1024, ("content|x", None)
    assert attention_zeros(size, 2, 1, pieces) == (512, 512)
    assert attention_zeros(size, 2, 2, pieces) == (0, 512)
    assert attention_zeros(size, 2, 2, ("a", "b")) == (0, 0)
    assert attention_zeros(size, 2, 1, "content|whole") == (512, 512)
    assert attention_zeros(size, 1, 1) == (0, 0)


def test_published_kimi_k3s_hit_and_pull_use_the_same_content_coordinates():
    """Replicated MLA and TP-sharded KDA agree at the retained checkpoint;
    distinct suffixes and a shifted latent are distinguishable at the pull."""
    from tests.v1.worker.simulated.test_aisimulate_timing import bundled_config
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
    from vllm.v1.worker.simulated.state_layout import (
        layout_from_config,
        state_span_pattern,
    )

    model = KimiLinearConfig(**bundled_config("moonshotai/Kimi-K3")["text_config"])
    state = layout_from_config(model, 2, 4)
    prefill = TokenChains([768, 768], grain=768, state_groups=[1])
    decode = TokenChains([1536, 1536], backoff=1, grain=768, state_groups=[1])
    shared = list(range(1536))
    cold = shared + [4] * 512
    hit = shared + [8] * 512
    for name, prompt in (("cold", cold), ("hit", hit)):
        prefill.admit(name, prompt[:-1])
        decode.admit(name, prompt)
    slab = 768 * 576 * 2

    def latent(name, logical, width, identity):
        return row_pattern(
            name, "model.layers.3.self_attn", logical, width, 0, identity, (0,)
        )

    pieces = [
        latent("cold", b, slab, prefill.identity("cold", 0, b, 768))
        for b in (0, 1)
    ]
    hit_id = decode.identity("hit", 0, 0, 1536)
    hit_page = latent("hit", 0, 2 * slab, hit_id)
    assert torch.equal(hit_page, torch.cat(pieces))
    hit_page[slab + 1] ^= 1
    assert torch.count_nonzero(hit_page != torch.cat(pieces)).item() == 1
    first, second = decode.pulled_content("hit", 0, 1)
    assert second is None
    expected = latent("hit", 1, 2 * slab, (first, second))
    assert not expected[slab:].any()
    assert not torch.equal(expected[:slab], pieces[1])  # Native shifted piece 1.
    cold_page = latent("cold", 1, 2 * slab, decode.pulled_content("cold", 0, 1))
    assert not torch.equal(expected[:slab], cold_page[:slab])

    for pblock, ptokens, dblock, dtokens in ((1, 768, 0, 1536), (2, 511, 1, 511)):
        p = prefill.identity("hit", 1, pblock, ptokens)
        d = decode.identity("hit", 1, dblock, dtokens)
        assert p == d and p.startswith(STATE_IDENTITY)
        for rank in range(4):
            def spans(identity, size, r):
                return [
                    state_span_pattern(identity, "model.layers.0", 0, state, span)
                    for span in state.spans(r, size)
                ]

            lo, hi = spans(p, 8, rank * 2), spans(p, 8, rank * 2 + 1)
            want = torch.cat(spans(d, 4, rank))
            got = torch.cat([part for pair in zip(lo, hi) for part in pair])
            assert torch.equal(got, want)
            got[-1] ^= 1
            assert torch.count_nonzero(got != want).item() == 1
