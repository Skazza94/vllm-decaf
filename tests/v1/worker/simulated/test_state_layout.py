# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CP4 step A: a hybrid's recurrent state as the model defines it (state-v1),
ownership across unequal TP sizes, the posted-read coverage rule, and the
mutations the independent reference must fail."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.simulated import state_layout as sl
from vllm.v1.worker.simulated.kv_pattern import (
    NativeChecker,
    NativeFiller,
    native_group_heads,
)

NEMOTRON = SimpleNamespace(
    mamba_num_heads=64, mamba_head_dim=64, n_groups=8, ssm_state_size=128, conv_kernel=4
)
QWEN = SimpleNamespace(
    linear_num_key_heads=16, linear_key_head_dim=128, linear_num_value_heads=48,
    linear_value_head_dim=128, linear_conv_kernel_dim=4,
)
# Kimi-K3's published KDA fields; its text config is Kimi-Linear's.
K3_KDA = {
    "num_heads": 96, "head_dim": 128, "short_conv_kernel_size": 4,
    "kda_layers": [1, 2, 3], "full_attn_layers": [4],
}
# Decode TP4's MLA page: 1,536 tokens of a 576-wide BF16 latent, one head.
K3_LATENT = 1536 * 576 * 2


def _k3_config(**kda):
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    layers = {"kda_layers": [1, 2, 3], "full_attn_layers": [4]}
    return KimiLinearConfig(num_hidden_layers=4, linear_attn_config={**layers, **kda} if kda else K3_KDA)


def test_the_layouts_come_from_the_model_config_alone():
    n = sl.layout_from_config(NEMOTRON, 2, 4)
    assert n.family == "mamba2" and n.components == (("x", 4096), ("B", 1024), ("C", 1024))
    assert n.conv_rows == 3 and n.ssm_heads == 64 and n.ssm_head_bytes == 64 * 128 * 4
    # The saved Nemotron TP2 state spans: 12,288 / 3,072 / 3,072 / 1,048,576 bytes.
    assert [s.length for s in n.spans(0, 2)] == [12288, 3072, 3072, 1048576]
    q = sl.layout_from_config(QWEN, 2, 4)
    assert q.family == "gdn" and q.components == (("Q", 2048), ("K", 2048), ("V", 6144))
    assert q.ssm_heads == 48 and q.ssm_head_bytes == 128 * 128 * 4
    assert q.local_sizes(1) == (10240 * 3 * 2, 48 * 128 * 128 * 4)
    assert sl.layout_from_config(SimpleNamespace(hidden_size=1), 2, 4) is None
    # Offsets are per rank, components in order, then the heads.
    assert [s.offset for s in q.spans(1, 2)] == [0, 1024 * 6, 2048 * 6, (1024 * 2 + 3072) * 6]
    assert [(s.first, s.count) for s in q.spans(1, 2)] == [(1024, 1024), (1024, 1024), (3072, 3072), (24, 24)]
    assert not q.divides(5) and q.divides(8)
    with pytest.raises(ValueError):
        q.spans(0, 5)


def test_element_sizes_are_resolved_as_the_native_allocation_resolves_them():
    def config(mamba="auto", ssm="auto", text=NEMOTRON):
        return SimpleNamespace(
            speculative_config=None,
            model_config=SimpleNamespace(hf_text_config=text, dtype=torch.bfloat16),
            cache_config=SimpleNamespace(mamba_cache_dtype=mamba, mamba_ssm_cache_dtype=ssm),
        )

    sizes = lambda c: (sl.layout_for(c).conv_itemsize, sl.layout_for(c).ssm_itemsize)
    # Both "auto": the model's dtype, the SSM following the conv state.
    assert sizes(config()) == (2, 2)
    # --mamba-cache-dtype sets the conv state, and an "auto" SSM state with it.
    assert sizes(config(mamba="float32")) == (4, 4)
    assert sizes(config(mamba="float32", ssm="bfloat16")) == (4, 2)
    # An "auto" SSM dtype takes the config's own, as the hybrid configs' verification does.
    own = SimpleNamespace(**vars(NEMOTRON), mamba_ssm_cache_dtype="float32")
    assert sizes(config(text=own)) == (2, 4)


def test_kimi_k3s_kda_state_comes_from_its_linear_attn_config():
    """KDA's conv is Q|K|V (each heads x head_dim columns, kernel - 1 rows),
    its recurrent state a head_dim x head_dim matrix a head
    (`kda_state_shape`). Per rank, BF16 conv and FP32 state: (27,648,
    786,432) bytes at TP8, (55,296, 1,572,864) at TP4, as K3's runs log."""
    from transformers import PretrainedConfig

    k = sl.layout_from_config(_k3_config(), 2, 4)
    assert k.family == "kda" and k.components == (("Q", 12288), ("K", 12288), ("V", 12288))
    assert (k.conv_rows, k.ssm_heads, k.ssm_shape, k.ssm_head_bytes) == (3, 96, (128, 128), 128 * 128 * 4)
    assert k.local_sizes(8) == (27648, 786432) and k.local_sizes(4) == (55296, 1572864)
    assert [(s.offset, s.length) for s in k.spans(1, 8)] == [
        (0, 9216), (9216, 9216), (18432, 9216), (27648, 786432)
    ]
    assert [s.length for s in k.spans(3, 4)] == [18432] * 3 + [1572864]
    # Incomplete KDA fields, or the nested config under another model type: no layout.
    assert sl.layout_from_config(_k3_config(num_heads=96), 2, 4) is None
    assert sl.layout_from_config(PretrainedConfig(linear_attn_config=K3_KDA), 2, 4) is None


def _spec(layout, tp, **over):
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    conv = sum(w // tp for _, w in layout.components)
    values = dict(
        mamba_type=MambaAttentionBackendEnum.MAMBA2 if layout.family == "mamba2" else MambaAttentionBackendEnum.GDN_ATTN,
        tp_replicated=False,
        shapes=((conv, layout.conv_rows), (layout.ssm_heads // tp, *layout.ssm_shape)),
        dtypes=(torch.bfloat16, torch.float32),
    )
    values.update(over)
    return SimpleNamespace(**values)


def test_a_registered_spec_must_be_the_layout_field_by_field():
    n = sl.layout_from_config(NEMOTRON, 2, 4)
    assert sl.check_spec(n, _spec(n, 2), 2) == []
    assert sl.check_spec(n, _spec(n, 1), 1) == []
    bad = sl.check_spec(n, _spec(n, 2, dtypes=(torch.float32, torch.float32)), 2)
    assert any("conv element" in p for p in bad)
    bad = sl.check_spec(n, _spec(n, 2, dtypes=(torch.bfloat16, torch.bfloat16)), 2)
    assert any("SSM element" in p for p in bad)
    # Equal bytes, other semantics: the head axis elsewhere, flattened heads, a
    # wider state in a narrower element: each fails on its own field.
    for shape, dtype, field in (
        ((32, 64, 128), torch.float32, None),
        ((64, 32, 128), torch.float32, "SSM shape"),
        ((32, 8192), torch.float32, "SSM shape"),
        ((32, 64, 256), torch.bfloat16, "SSM"),
    ):
        spec = _spec(n, 2, dtypes=(torch.bfloat16, dtype))
        spec.shapes = (spec.shapes[0], shape)
        found = sl.check_spec(n, spec, 2)
        assert (found == []) if field is None else any(field in p for p in found), (shape, dtype, found)
    assert sl.check_spec(n, _spec(n, 2, tp_replicated=True), 2)
    spec = _spec(n, 2)
    spec.shapes = ((4096 // 2 + 513 + 512, 3), spec.shapes[1])  # a conv wider by a column
    assert any("conv shape" in p for p in sl.check_spec(n, spec, 2))
    q = sl.layout_from_config(QWEN, 2, 4)
    assert any("mamba_type" in p for p in sl.check_spec(q, _spec(n, 2), 2))



# ----- the pattern is the same bytes of the same global coordinates -----

TOY = sl.StateLayout("mamba2", (("x", 16), ("B", 4), ("C", 4)), 3, 2, 4, 8)


def _page(layout, rank, tp, block, layer="L", base="req"):
    return {
        (s.kind, s.component): sl.state_span_pattern(base, layer, block, layout, s)
        for s in layout.spans(rank, tp)
    }


def test_state_bytes_are_the_same_at_any_tp_size():
    whole = _page(TOY, 0, 1, 3)
    for tp in (2, 4):
        for key in whole:
            joined = torch.cat([_page(TOY, r, tp, 3)[key] for r in range(tp)])
            assert torch.equal(joined, whole[key]), (tp, key)
    # Another layer, block, request or component is another stream.
    assert not torch.equal(_page(TOY, 0, 1, 3, layer="M")[("ssm", "ssm")], whole[("ssm", "ssm")])
    assert not torch.equal(_page(TOY, 0, 1, 4)[("conv", "x")], whole[("conv", "x")])
    assert not torch.equal(_page(TOY, 0, 1, 3, base="other")[("conv", "B")], whole[("conv", "B")])
    assert not torch.equal(whole[("conv", "B")], whole[("conv", "C")])


def test_two_state_groups_never_share_a_stream_for_the_same_ordinal():
    """Identity is the model's layer name (or a transfer-only region's own
    identity), never an ordinal within a group: layer 0 of two groups differ."""
    a = _page(TOY, 0, 1, 1, layer="model.layers.0.mixer")
    b = _page(TOY, 0, 1, 1, layer="model.layers.7.mixer")
    x = _page(TOY, 0, 1, 1, layer="xfer.g1.r4")
    for key in a:
        assert not torch.equal(a[key], b[key]) and not torch.equal(a[key], x[key])


# ----- ownership -----


def test_ownership_covers_every_local_head_and_state_range():
    # Qwen: 4 KV heads; decode TP1 owns heads 0..3 and needs producers 0 and 1 of TP2.
    assert sl.attention_sources(0, 1, 2, 4) == {0: (0, 1), 1: (2, 3)}
    # TP2 <- TP1: both ranks take their heads from the one producer.
    assert sl.attention_sources(1, 2, 1, 4) == {0: (2, 3)}
    # Nemotron K=2: TP2 <- TP4 attention from producer 0 / 2, state from 0,1 / 2,3.
    assert sl.attention_sources(0, 2, 4, 2) == {0: (0,)}
    assert sl.attention_sources(1, 2, 4, 2) == {2: (1,)}
    assert sl.state_sources(0, 2, 4, 4096) == {0: (0, 1024), 1: (1024, 2048)}
    assert sl.state_sources(1, 2, 4, 4096) == {2: (2048, 3072), 3: (3072, 4096)}
    # TP4 <- TP2 (replicated attention on the decode side).
    assert [sl.attention_sources(r, 4, 2, 2) for r in range(4)] == [{0: (0,)}, {0: (0,)}, {1: (1,)}, {1: (1,)}]
    assert sl.state_sources(1, 4, 2, 4096) == {0: (1024, 2048)}
    assert sl.state_sources(0, 2, 2, 4096) == {0: (0, 2048)}


@pytest.mark.parametrize("kv", [2, 4])
def test_ownership_matches_nixl_compute_tp_mapping(kv):
    """The independent rule against the native mapping, every pairing of TP
    1, 2, 4, 8 (the larger a multiple of the smaller)."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import compute_tp_mapping
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    width = 4096
    for local in (1, 2, 4, 8):
        for remote in (1, 2, 4, 8):
            if max(local, remote) % min(local, remote):
                continue
            for rank in range(local):
                topo = SimpleNamespace(
                    tp_rank=rank, tp_size=local, is_mla=False, total_num_kv_heads=kv,
                    dcp_source_ranks=None, dcp_consumer_count=lambda *a: 1,
                )
                plan = compute_tp_mapping(topo, remote, (FullAttentionSpec, MambaSpec))
                assert tuple(sorted(sl.attention_sources(rank, local, remote, kv))) == plan.source_ranks_per_group[0], (local, remote, rank)
                assert tuple(sorted(sl.state_sources(rank, local, remote, width))) == plan.source_ranks_per_group[1], (local, remote, rank)


@pytest.mark.parametrize("config", [NEMOTRON, QWEN], ids=["mamba2", "gdn"])
@pytest.mark.parametrize("local,remote", [(1, 2), (1, 4), (1, 8), (2, 4), (2, 8), (4, 8)])
def test_native_split_reads_land_where_ownership_says(monkeypatch, config, local, remote):
    """P TP > D TP: the decode rank's local state descriptors as native NIXL
    builds them (`_build_mamba_local`, the conv split from `MambaSpec`), cut
    per producer by native `_build_local_splits_from_plan` under the native
    mapping, resolved to (region, block, offset, length) pieces, equal the
    independent rule's (`expected_pieces`) for every producer."""
    import types

    from vllm.distributed.kv_transfer.kv_connector.v1 import ssm_conv_transfer_utils
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        NixlBaseConnectorWorker as Native,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import compute_tp_mapping
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    monkeypatch.setattr(ssm_conv_transfer_utils, "is_conv_state_dim_first", lambda: True)
    layout = sl.layout_from_config(config, 2, 4)
    spec = _spec(layout, local)
    decomp = ssm_conv_transfer_utils.derive_mamba_conv_split(spec, local)
    page, regions, num_blocks, held = sum(decomp.ssm_sizes) + 512, 3, 6, [1, 4]
    bases = [(r + 1) << 32 for r in range(regions)]
    for rank in range(local):
        topo = SimpleNamespace(
            tp_rank=rank, tp_size=local, is_mla=False, total_num_kv_heads=8,
            dcp_source_ranks=None, dcp_consumer_count=lambda *a: 1,
        )
        plan = compute_tp_mapping(topo, remote, (FullAttentionSpec, MambaSpec))
        worker = SimpleNamespace(
            _conv_decomp=decomp, _mamba_ssm_size=decomp.ssm_sizes, _logical_num_blocks=num_blocks,
            _physical_blocks_per_logical_kv_block=1, device_id=0, _ssm_region_indices=list(range(regions)),
            block_stride_per_layer=[page] * regions, _ple_region_index=None,
            _group_spec_types=[FullAttentionSpec, MambaSpec], transfer_topo=topo,
            block_len_per_layer=[], num_regions=0, _pack_strided_descs=Native._pack_strided_descs,
            _strided_desc_starts=Native._strided_desc_starts,
        )
        worker._fa_desc_replicated = types.MethodType(Native._fa_desc_replicated, worker)
        local_descs = Native._build_mamba_local(worker, bases)
        handles = list(Native._build_local_splits_from_plan(worker, plan, local_descs, num_fa_descs=0))
        assert len(handles) == len(plan.all_source_ranks)
        native = {}
        for p_idx, handle in enumerate(handles):
            for addr, length, _, stride, count in handle.tolist():
                region = next(r for r, base in enumerate(bases) if base <= addr < base + page)
                assert stride == page and count == num_blocks
                native.setdefault(plan.all_source_ranks[p_idx], []).extend(
                    (region, b, addr - bases[region], length) for b in held
                )
        rows = [
            {"region": r, "offset": span.offset, "length": span.length, "state": True, "span": span}
            for r in range(regions)
            for span in layout.spans(rank, local)
        ]
        want = sl.expected_pieces(rows, held, rank, local, remote, layout, 8)
        assert {p: sorted(v) for p, v in native.items()} == {p: sorted(v) for p, v in want.items()}, (
            local, remote, rank,
        )


# ----- the posted-read coverage rule -----

ROWS = [
    {"region": 0, "offset": 0, "length": 64, "state": False, "heads": (0, 1, 2, 3)},
    {"region": 1, "offset": 0, "length": 8, "state": True, "span": TOY.spans(0, 1)[0]},
]


def _expected(local=1, remote=2, rank=0, kv=4, blocks=(5,)):
    rows = [dict(ROWS[0]), dict(ROWS[1])]
    rows[1]["span"] = TOY.spans(rank, local)[0]
    rows[1]["length"] = rows[1]["span"].length
    return sl.expected_pieces(rows, list(blocks), rank, local, remote, TOY, kv)


def _read(rank, pieces, done=True):
    return sl.ReadRecord("req", rank, object(), pieces, done)


def _from(expected, rank, only=None):
    pieces = [p for p in expected[rank] if only is None or only(p)]
    return pieces


def test_a_valid_attention_plus_state_read_then_a_state_only_read_is_covered():
    want = _expected()
    assert set(want) == {0, 1}
    # Producer 0: attention heads 0,1 and the first half of x; producer 1: heads 2,3 and the second.
    reads = [_read(0, want[0]), _read(1, want[1])]
    assert sl.coverage_problems(want, reads) == []
    # Producer 1's attention READ first, its state in a later state-only READ.
    attn = lambda p: p[0] == 0
    reads = [_read(0, want[0]), _read(1, _from(want, 1, attn)), _read(1, _from(want, 1, lambda p: p[0] == 1))]
    assert sl.coverage_problems(want, reads) == []
    # NIXL cutting the same bytes into more descriptors is the same coverage.
    split = [(r, b, o, 1) for r, b, o, n in want[0] if r == 1 for o in range(o, o + n)] + [p for p in want[0] if p[0] == 0]
    assert sl.coverage_problems(want, [_read(0, split), _read(1, want[1])]) == []


def test_heads_a_page_does_not_hold_are_an_ownership_error_not_a_crash():
    rows = [dict(ROWS[0], heads=(0, 1)), dict(ROWS[1], span=TOY.spans(0, 1)[0])]
    with pytest.raises(ValueError, match=r"supplies heads \[2, 3\]"):
        sl.expected_pieces(rows, [5], 0, 1, 2, TOY, 4)


def test_coverage_rejects_what_ownership_does_not_say():
    want = _expected()
    ok0, ok1 = want[0], want[1]
    state = lambda p: p[0] == 1
    cases = {
        "missing shard": [_read(0, ok0), _read(1, [p for p in ok1 if p[0] == 0])],
        "duplicated shard": [_read(0, ok0), _read(1, ok1), _read(1, _from(want, 1, state))],
        "swapped sources": [_read(1, ok0), _read(0, ok1)],
        "foreign region": [_read(0, ok0 + [(9, 5, 0, 8)]), _read(1, ok1)],
        "wrong block": [_read(0, [(r, 6, o, n) for r, b, o, n in ok0]), _read(1, ok1)],
        "unknown producer": [_read(0, ok0), _read(1, ok1), _read(7, [(1, 5, 0, 8)])],
        "state from the attention producer only": [_read(0, ok0 + _from(want, 1, state)), _read(1, [p for p in ok1 if p[0] == 0])],
    }
    for name, reads in cases.items():
        assert sl.coverage_problems(want, reads), name
    # Both READs posted, one still outstanding when the request is reported complete.
    pending = sl.coverage_problems(want, [_read(0, ok0), _read(1, ok1, done=False)])
    assert pending and "had not completed" in pending[0]


def _k3_rows(rank):
    """Decode TP4 rank `rank`'s selected rows as `NativeChecker._coverage`
    describes them: an MLA layer's latent page (group 0, its one head) and a
    KDA layer's spans (group 1)."""
    k = sl.layout_from_config(_k3_config(), 2, 4)
    latent = {
        "region": 0, "offset": 0, "length": K3_LATENT, "state": False, "heads": (0,),
        "span": None, "group": 0, "logical": True,
    }
    state = [
        {
            "region": 1, "offset": s.offset, "length": s.length, "state": True, "heads": None,
            "span": s, "group": 1, "logical": True,
        }
        for s in k.spans(rank, 4)
    ]
    return k, latent, state


@pytest.mark.parametrize("rank", range(4))
def test_kimi_k3s_decode_rank_reads_its_latent_from_2r_and_each_kda_span_from_2r_and_2r_plus_1(rank):
    """Prefill TP8 -> decode TP4: decode rank r takes the latent (whole
    pages: one head) from producer 2r alone, and each of Q, K, V and the
    recurrent state's halves from 2r and 2r + 1, once. A 2,047-token
    prefill fills 3 producer blocks of 768 tokens, 2 decode blocks of
    1,536: the last latent page read is its covered half (the cut a real
    pair's checker makes: `test_nixl_peers_check_kimi_k3s_uncovered_latent_tail_at_a_block_size_ratio`)."""
    k, latent, state = _k3_rows(rank)
    mla = sl.expected_pieces([latent], [2, 3], rank, 4, 8, k, 1)
    kda = sl.expected_pieces(state, [5], rank, 4, 8, k, 1)
    first, second = 2 * rank, 2 * rank + 1
    assert mla == {first: [(0, 2, 0, K3_LATENT), (0, 3, 0, K3_LATENT)]}
    assert set(kda) == {first, second}
    assert sorted(kda[first]) == [(1, 5, 0, 9216), (1, 5, 18432, 9216), (1, 5, 36864, 9216), (1, 5, 55296, 786432)]
    assert sorted(kda[second]) == [
        (1, 5, 9216, 9216), (1, 5, 27648, 9216), (1, 5, 46080, 9216), (1, 5, 55296 + 786432, 786432)
    ]
    cut = [(0, 2, 0, K3_LATENT), (0, 3, 0, K3_LATENT // 2)]
    want = {first: cut + kda[first], second: kda[second]}
    reads = [_read(first, want[first]), _read(second, want[second])]
    assert sl.coverage_problems(want, reads) == []
    v_of_second = sorted(kda[second])[2]
    cases = {
        "the latent from 2r + 1": [_read(first, kda[first]), _read(second, cut + kda[second])],
        "the uncovered half read": [_read(first, mla[first] + kda[first]), _read(second, kda[second])],
        "V missing from 2r + 1": [
            _read(first, want[first]), _read(second, [p for p in kda[second] if p != v_of_second])
        ],
        "V from 2r instead of 2r + 1": [
            _read(first, want[first] + [v_of_second]),
            _read(second, [p for p in kda[second] if p != v_of_second]),
        ],
        "the recurrent state twice": [
            _read(first, want[first]), _read(second, kda[second]), _read(first, [sorted(kda[first])[3]])
        ],
    }
    for name, reads in cases.items():
        assert sl.coverage_problems(want, reads), name


# The published Qwen3.5-27B's text config, the layout's source, as a real
# `Qwen3_5TextConfig`: 64 layers, 48 gated delta-net and 16 full-attention
# ones interleaved every fourth layer, 16 key heads and 48 value heads of
# 128 (conv kernel 4), 4 attention KV heads of head_dim 256.
QWEN3_5_27B_TEXT_CONFIG = None


def _qwen3_5_27b_text_config():
    global QWEN3_5_27B_TEXT_CONFIG
    if QWEN3_5_27B_TEXT_CONFIG is None:
        from transformers import Qwen3_5TextConfig

        QWEN3_5_27B_TEXT_CONFIG = Qwen3_5TextConfig(
            hidden_size=5120,
            num_hidden_layers=64,
            full_attention_interval=4,
            layer_types=[
                "full_attention" if layer % 4 == 3 else "linear_attention"
                for layer in range(64)
            ],
            num_attention_heads=24,
            num_key_value_heads=4,
            head_dim=256,
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            mamba_ssm_dtype="float32",
        )
    return QWEN3_5_27B_TEXT_CONFIG


# The 27B's 4 attention KV heads of head_dim 256, K and V, BF16: 1,024 B a
# token on a decode TP4 rank, an 800-token page of 819,200 B.
QWEN3_5_27B_ATTENTION_PAGE = 819200


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_qwen3_5s_gdn_state_splits_tp_1_2_4_and_8(tp):
    """The 27B geometry (16 key heads and 48 value heads of 128, conv kernel
    4): the GDN layout splits over every TP size the unequal pairs use, conv
    `Q|K|V` columns of 3 BF16 values, SSM heads of a 128x128 FP32 state."""
    layout = sl.layout_from_config(_qwen3_5_27b_text_config(), 2, 4)
    assert layout.family == "gdn"
    assert layout.components == (("Q", 2048), ("K", 2048), ("V", 6144))
    assert layout.ssm_heads == 48
    assert layout.ssm_head_bytes == 128 * 128 * 4
    assert layout.divides(tp)
    assert layout.local_sizes(tp) == (61440 // tp, 3145728 // tp)


def _qwen3_5_27b_state_rows(rank):
    """Decode TP4 rank `rank`'s state rows, one per conv component then the
    recurrent heads (`layout_from_config`, the published 27B's text
    config)."""
    layout = sl.layout_from_config(_qwen3_5_27b_text_config(), 2, 4)
    return [
        {
            "region": 1, "offset": s.offset, "length": s.length, "state": True,
            "heads": None, "span": s, "group": 1, "logical": True,
        }
        for s in layout.spans(rank, 4)
    ]


@pytest.mark.parametrize("rank", range(4))
def test_qwen3_5s_decode_tp4_rank_reads_its_800_token_blocks_from_prefill_tp8(rank):
    """Prefill TP8 (400-token attention blocks) -> decode TP4 (its 400
    request aligned to 800: an attention page must cover a state's page). A
    2,000-token prefill makes 5 producer attention blocks and 3 of the
    consumer's: its 800-token pages are 819,200 B a page at TP 4 (one of 4
    KV heads, head_dim 256, K and V, BF16), the last's tail half covered,
    409,600 B NIXL zeroes on receive; attention comes from 2r alone, and
    each of Q, K, V and the recurrent state's halves from 2r and 2r + 1,
    once. Kimi-K3's rule (`test_kimi_k3s_decode_rank_reads_its_latent...`)
    at the 27B's GDN components: 512 conv columns of 6 B a component per
    rank, split in two over the producers, and 12 FP32 heads of 65,536."""
    layout = sl.layout_from_config(_qwen3_5_27b_text_config(), 2, 4)
    page = QWEN3_5_27B_ATTENTION_PAGE
    attn = {
        "region": 0, "offset": 0, "length": page, "state": False,
        "heads": (rank,), "span": None, "group": 0, "logical": True,
    }
    state = _qwen3_5_27b_state_rows(rank)
    attention = sl.expected_pieces([attn], [0, 1, 2], rank, 4, 8, layout, 4)
    first, second = 2 * rank, 2 * rank + 1
    assert attention == {
        first: [(0, 0, 0, page), (0, 1, 0, page), (0, 2, 0, page)]
    }
    mba = sl.expected_pieces(state, [3], rank, 4, 8, layout, 4)
    assert sorted(mba[first]) == [
        (1, 3, 0, 1536), (1, 3, 3072, 1536), (1, 3, 6144, 4608), (1, 3, 15360, 393216)
    ]
    assert sorted(mba[second]) == [
        (1, 3, 1536, 1536), (1, 3, 4608, 1536), (1, 3, 10752, 4608),
        (1, 3, 408576, 393216),
    ]
    cut = [(0, 0, 0, page), (0, 1, 0, page), (0, 2, 0, page // 2)]
    want = {first: cut + mba[first], second: mba[second]}
    reads = [_read(first, want[first]), _read(second, want[second])]
    assert sl.coverage_problems(want, reads) == []
    v_of_second = sorted(mba[second])[2]
    cases = {
        "the attention from 2r + 1": [
            _read(first, mba[first]), _read(second, cut + mba[second])
        ],
        "the uncovered half read": [
            _read(first, attention[first] + mba[first]), _read(second, mba[second])
        ],
        "V missing from 2r + 1": [
            _read(first, want[first]),
            _read(second, [p for p in mba[second] if p != v_of_second]),
        ],
        "V from 2r instead of 2r + 1": [
            _read(first, want[first] + [v_of_second]),
            _read(second, [p for p in mba[second] if p != v_of_second]),
        ],
        "the recurrent state twice": [
            _read(first, want[first]), _read(second, mba[second]),
            _read(first, [sorted(mba[first])[3]]),
        ],
    }
    for name, reads in cases.items():
        assert sl.coverage_problems(want, reads), name


# ----- mutations: the native side moves, the semantic pattern stays -----


def _stub(layout, rank, tp, blocks=4):
    rows = [
        SimpleNamespace(
            view=torch.zeros(blocks, s.length, dtype=torch.uint8), region=7, offset=s.offset,
            state=True, span=s,
        )
        for s in layout.spans(rank, tp)
    ]
    return SimpleNamespace(
        rows=rows, layout=layout, groups=[["L"]], members=[{7}], padding={},
        select=lambda group, held: [(i, b, k) for k, b in enumerate(held) for i in range(len(rows))],
        state_identity=lambda group, region: "L",
        name=lambda group, row: f"g{group}.r7.o{rows[row].offset}",
        state_regions=lambda selected: {7},
        regions=lambda selected: {7},
        order=lambda held: [b for ids in held.values() for b in ids],
        worker=None,
    )


def _produce(tp):
    stubs = [_stub(TOY, r, tp) for r in range(tp)]
    for r, s in enumerate(stubs):
        NativeFiller(s, r, heads=((),))._write_block("cmpl-a-0", 0, 0, 2, None)
    return stubs


def _check(stub, rank):
    checker = NativeChecker(stub, tp_rank=rank, heads=((),))
    checker._check_geometry = lambda: None
    checker._coverage = lambda e, r: {"problems": [], "reads": []}  # tested apart
    checker.posted["cmpl-a-0"] = [2]
    checker.reads["cmpl-a-0"] = [[2]]
    expected = SimpleNamespace(
        remote_request_id="cmpl-a-0", block_ids=[[2]],
        groups=[SimpleNamespace(group=0, block_ids=[2], first_logical_block=3)],
    )
    return checker._compare("cmpl-b-0", expected)


def test_a_cached_state_is_checked_at_its_blocks_place():
    """Without prefix caching a request's state is its one block (place 0,
    whatever index the read's metadata gives). With it (align mode) a state
    is its block's: written and compared at the block's place among the
    request's, under its content's identity."""
    identity = "content|B4|n4|00"
    producer = _stub(TOY, 0, 1)
    NativeFiller(producer, 0, heads=((),))._write_block("cmpl-a-0", 0, 3, 2, identity)

    def compare(place, identity_of):
        consumer = _stub(TOY, 0, 1)
        for row, source in zip(consumer.rows, producer.rows):
            row.view[2] = source.view[2]
        checker = NativeChecker(consumer, tp_rank=0, heads=((),))
        checker._check_geometry = lambda: None
        checker._coverage = lambda e, r: {"problems": [], "reads": []}
        checker.posted["cmpl-a-0"] = [2]
        checker.reads["cmpl-a-0"] = [[2]]
        expected = SimpleNamespace(
            remote_request_id="cmpl-a-0", block_ids=[[2]],
            groups=[SimpleNamespace(group=0, block_ids=[2], first_logical_block=place)],
        )
        return checker._compare("cmpl-b-0", expected, identity_of)

    ok = compare(3, lambda group, logical: identity)
    assert ok["ok"] and ok["groups"][0]["pattern_first_block"] == 3, ok
    assert ok["groups"][0]["pattern"] == sl.STATE_PATTERN
    # Another place, or another content, is another state.
    assert not compare(2, lambda group, logical: identity)["ok"]
    assert not compare(3, lambda group, logical: "content|B4|n4|01")["ok"]
    # Without prefix caching the place stays the request's one state's.
    assert not compare(3, None)["ok"]


def _receive(rank, local_tp, remote_tp, mutate=None):
    """A consumer rank's state as a native split / gather would deliver it
    (components in order, each from its producers), optionally mutated."""
    producers = _produce(remote_tp)
    consumer = _stub(TOY, rank, local_tp)
    pieces = {}
    for span in TOY.spans(rank, local_tp):
        width = dict(TOY.components)[span.component] if span.kind == "conv" else TOY.ssm_heads
        parts = []
        for p, (first, last) in sorted(sl.state_sources(rank, local_tp, remote_tp, width).items()):
            source = next(
                s for s in TOY.spans(p, remote_tp) if (s.kind, s.component) == (span.kind, span.component)
            )
            row = producers[p].rows[TOY.spans(p, remote_tp).index(source)].view[2]
            parts.append(row[(first - source.first) * source.unit_bytes : (last - source.first) * source.unit_bytes])
        pieces[(span.kind, span.component)] = torch.cat(parts)
    if mutate:
        mutate(pieces, TOY.spans(rank, local_tp))
    for row, span in zip(consumer.rows, TOY.spans(rank, local_tp)):
        row.view[2] = pieces[(span.kind, span.component)]
    return _check(consumer, rank)


@pytest.mark.parametrize("local,remote", [(1, 1), (2, 2), (2, 1), (1, 2), (1, 4), (4, 1), (2, 4), (4, 2)])
def test_a_correct_split_or_gather_passes(local, remote):
    for rank in range(local):
        result = _receive(rank, local, remote)
        assert result["ok"], (local, remote, rank, result)
        assert result["groups"][0]["pattern"] == "rank-v0" or result["groups"][0]["pattern"] == "state-v1"


def test_native_offsets_component_order_and_head_slices_are_failures():
    def swap_x_b(pieces, spans):  # components in the wrong order (same bytes moved, B where x goes)
        x, b = pieces[("conv", "x")], pieces[("conv", "B")]
        pieces[("conv", "x")], pieces[("conv", "B")] = b.repeat(len(x) // len(b)), x[: len(b)]

    def shift_column(pieces, spans):  # one column late
        x = pieces[("conv", "x")]
        pieces[("conv", "x")] = torch.cat([x[6:], x[:6]])

    def shift_head(pieces, spans):  # an SSM head off by one
        h = pieces[("ssm", "ssm")]
        pieces[("ssm", "ssm")] = torch.cat([h[8:], h[:8]])

    def swap_b_c(pieces, spans):
        pieces[("conv", "B")], pieces[("conv", "C")] = pieces[("conv", "C")], pieces[("conv", "B")]

    for local, remote in ((1, 1), (2, 1), (1, 2), (2, 4)):
        assert _receive(0, local, remote)["ok"]
        for mutation in (swap_x_b, shift_column, shift_head, swap_b_c):
            assert not _receive(0, local, remote, mutation)["ok"], (local, remote, mutation.__name__)


def test_a_state_of_another_layer_with_the_same_shape_fails():
    """Two state groups of identical shape: layer 0's bytes in layer 1's place
    (a cross-group swap) must fail, ordinals notwithstanding."""
    producers = _produce(1)
    consumer = _stub(TOY, 0, 1)
    consumer.state_identity = lambda group, region: "M"  # the checker expects layer M, delivery is L's
    for row, src in zip(consumer.rows, producers[0].rows):
        row.view[2] = src.view[2]
    assert not _check(consumer, 0)["ok"]


def test_native_group_heads_admit_a_hybrid_only_with_its_state_groups():
    heads = {"a0": ((0, 1), 16), "a1": ((0, 1), 16)}
    groups = [["a0", "a1"], ["m0"], ["m1"]]
    assert native_group_heads(heads, groups, True) is None
    held = native_group_heads(heads, groups, True, frozenset({1, 2}))
    assert held == ((0, 1), (), ())
    # A state group left out, or an attention group without heads, keeps the guard.
    assert native_group_heads(heads, groups, True, frozenset({1})) is None
    assert native_group_heads({"a0": ((0, 1), 16)}, groups, True, frozenset({1, 2})) is None
    # Not a hybrid: unchanged.
    assert native_group_heads(heads, [["a0", "a1"]], False) == ((0, 1),)


# The published Nemotron-3-Nano's geometry, as a real `NemotronHConfig`:
# 52 layers in the model's hybrid pattern, 32 attention heads over 2 KV
# heads of head_dim 128, a Mamba2 mixer of 64 heads of 64 over 8 groups of a
# 128-wide state, a 4-wide conv, the state FP32.
NEMOTRON_3_NANO_CONFIG = None


def _nemotron_3_nano_config():
    global NEMOTRON_3_NANO_CONFIG
    if NEMOTRON_3_NANO_CONFIG is None:
        from transformers import NemotronHConfig

        NEMOTRON_3_NANO_CONFIG = NemotronHConfig(
            hidden_size=2688,
            num_hidden_layers=52,
            hybrid_override_pattern=(
                "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
            ),
            num_attention_heads=32,
            num_key_value_heads=2,
            head_dim=128,
            mamba_num_heads=64,
            mamba_head_dim=64,
            ssm_state_size=128,
            n_groups=8,
            conv_kernel=4,
            mamba_ssm_cache_dtype="float32",
        )
    return NEMOTRON_3_NANO_CONFIG


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_nemotron_3_nanos_mamba2_state_splits_tp_1_2_4_and_8(tp):
    """The published Nemotron-3-Nano's Mamba2 layout: conv x|B|C of 4,096 +
    1,024 + 1,024 BF16 columns, 64 SSM heads of a 64x128 FP32 state; it
    splits over every TP size the unequal pairs use."""
    layout = sl.layout_from_config(_nemotron_3_nano_config(), 2, 4)
    assert layout.family == "mamba2"
    assert layout.components == (("x", 4096), ("B", 1024), ("C", 1024))
    assert layout.ssm_heads == 64
    assert layout.ssm_head_bytes == 64 * 128 * 4
    assert layout.divides(tp)
    assert layout.local_sizes(tp) == (36864 // tp, 2097152 // tp)


def _nemotron_attn_row(rank, tp, block):
    """The attention region of a published Nemotron-3-Nano decoder rank: its
    heads' page of `block` tokens, 512 B a token (one KV head of head_dim
    128, K and V, BF16)."""
    heads = sl.head_global(tp, rank, 2)
    return {
        "region": 0, "offset": 0, "state": False, "span": None,
        "length": block * len(heads) * 128 * 2 * 2,
        "heads": heads, "group": 0, "logical": True,
    }


def _nemotron_state_rows(rank, tp):
    layout = sl.layout_from_config(_nemotron_3_nano_config(), 2, 4)
    return [
        {
            "region": 1, "offset": s.offset, "length": s.length, "state": True,
            "heads": None, "span": s, "group": 1, "logical": True,
        }
        for s in layout.spans(rank, tp)
    ]


def _qwen3_5_attn_row(rank, tp, block):
    heads = sl.head_global(tp, rank, 4)
    return {
        "region": 0, "offset": 0, "state": False, "span": None,
        "length": block * len(heads) * 256 * 2 * 2,
        "heads": heads, "group": 0, "logical": True,
    }


def _qwen3_5_27b_state_rows_at(rank, tp):
    layout = sl.layout_from_config(_qwen3_5_27b_text_config(), 2, 4)
    return [
        {
            "region": 1, "offset": s.offset, "length": s.length, "state": True,
            "heads": None, "span": s, "group": 1, "logical": True,
        }
        for s in layout.spans(rank, tp)
    ]


CANDIDATES = [
    # (id, producer TP, consumer TP, the consumer's explicit block)
    ("nemotron-3-nano p8-d4 r2", 8, 4, 1088),
    ("nemotron-3-nano p8-d2 r4", 8, 2, 2176),
    ("nemotron-3-nano p4-d8 r2", 4, 8, 2112),
    ("qwen3.5-27b p1-d8 r2", 1, 8, 1600),
    ("qwen3.5-27b p2-d8 r2", 2, 8, 1600),
    ("qwen3.5-27b p4-d8 r2", 4, 8, 1600),
]


def _candidate_rows(name, rank, consumer_tp, block):
    """One decoder rank's attention and state rows at its TP, and the
    published config's `StateLayout`."""
    if name == "nemotron":
        return (
            _nemotron_attn_row(rank, consumer_tp, block),
            _nemotron_state_rows(rank, consumer_tp),
            sl.layout_from_config(_nemotron_3_nano_config(), 2, 4),
            2,
        )
    return (
        _qwen3_5_attn_row(rank, consumer_tp, block),
        _qwen3_5_27b_state_rows_at(rank, consumer_tp),
        sl.layout_from_config(_qwen3_5_27b_text_config(), 2, 4),
        4,
    )


STATE_CANDIDATES = [
    # (id, producer TP, consumer TP, the consumer's explicit block, the rank
    # the state table works, that rank's state pieces per producer)
    (
        "nemotron-3-nano p8-d4 r2", 8, 4, 1088, 1,
        {
            2: [(1, 0, 0, 3072), (1, 0, 6144, 768), (1, 0, 7680, 768),
                (1, 0, 9216, 262144), (1, 1, 0, 3072), (1, 1, 6144, 768),
                (1, 1, 7680, 768), (1, 1, 9216, 262144)],
            3: [(1, 0, 3072, 3072), (1, 0, 6912, 768), (1, 0, 8448, 768),
                (1, 0, 271360, 262144), (1, 1, 3072, 3072),
                (1, 1, 6912, 768), (1, 1, 8448, 768),
                (1, 1, 271360, 262144)],
        },
    ),
    (
        "nemotron-3-nano p8-d2 r4", 8, 2, 2176, 0,
        {
            p: [(1, 0, 3072 * i, 3072), (1, 0, 12288 + 768 * i, 768),
                (1, 0, 15360 + 768 * i, 768),
                (1, 0, 18432 + 262144 * i, 262144),
                (1, 1, 3072 * i, 3072), (1, 1, 12288 + 768 * i, 768),
                (1, 1, 15360 + 768 * i, 768),
                (1, 1, 18432 + 262144 * i, 262144)]
            for i, p in enumerate((0, 1, 2, 3))
        },
    ),
    (
        "nemotron-3-nano p4-d8 r2", 4, 8, 2112, 3,
        {
            1: [(1, 0, 0, 3072), (1, 0, 3072, 768), (1, 0, 3840, 768),
                (1, 0, 4608, 262144), (1, 1, 0, 3072), (1, 1, 3072, 768),
                (1, 1, 3840, 768), (1, 1, 4608, 262144)],
        },
    ),
    (
        "qwen3.5-27b p4-d8 r2", 4, 8, 1600, 5,
        {
            2: [(1, 0, 0, 1536), (1, 0, 1536, 1536), (1, 0, 3072, 4608),
                (1, 0, 7680, 393216), (1, 1, 0, 1536),
                (1, 1, 1536, 1536), (1, 1, 3072, 4608),
                (1, 1, 7680, 393216)],
        },
    ),
    (
        "qwen3.5-27b p2-d8 r2", 2, 8, 1600, 0,
        {
            0: [(1, 0, 0, 1536), (1, 0, 1536, 1536), (1, 0, 3072, 4608),
                (1, 0, 7680, 393216), (1, 1, 0, 1536),
                (1, 1, 1536, 1536), (1, 1, 3072, 4608),
                (1, 1, 7680, 393216)],
        },
    ),
    (
        "qwen3.5-27b p1-d8 r2", 1, 8, 1600, 0,
        {
            0: [(1, 0, 0, 1536), (1, 0, 1536, 1536), (1, 0, 3072, 4608),
                (1, 0, 7680, 393216), (1, 1, 0, 1536),
                (1, 1, 1536, 1536), (1, 1, 3072, 4608),
                (1, 1, 7680, 393216)],
        },
    ),
]


def _model_of(table, producer_tp, consumer_tp, block):
    name = next(c[0] for c in table if c[1:4] == (producer_tp, consumer_tp, block))
    return "nemotron" if name.startswith("nemotron") else "qwen"


@pytest.mark.parametrize(
    "producer_tp,consumer_tp,block,rank,want_state",
    [c[1:] for c in STATE_CANDIDATES],
    ids=[c[0] for c in STATE_CANDIDATES],
)
def test_a_candidate_pairings_attention_and_state_are_covered(
    producer_tp, consumer_tp, block, rank, want_state
):
    """A new candidate's ownership at a worked rank: attention is whole pages
    of the rank's head(s), from the producers the head mapping picks
    (`attention_sources`), one page a block; the state is its spans once each
    from the overlapping producers, contiguous global coordinates, one piece
    a block. READs of exactly attention + state satisfy `coverage_problems`,
    and a READ from a producer ownership does not name is uncovered."""
    name = _model_of(STATE_CANDIDATES, producer_tp, consumer_tp, block)
    attn, state, layout, kv_heads = _candidate_rows(
        name, rank, consumer_tp, block
    )
    blocks = [0, 1]
    attention = sl.expected_pieces(
        [attn], blocks, rank, consumer_tp, producer_tp, layout, kv_heads
    )
    sources = sl.attention_sources(rank, consumer_tp, producer_tp, kv_heads)
    assert set(attention) == set(sources)
    for p in sources:
        assert attention[p] == [(0, b, 0, attn["length"]) for b in blocks]
    head_bytes = 128 * 2 * 2 if name == "nemotron" else 256 * 2 * 2
    heads = sl.head_global(consumer_tp, rank, kv_heads)
    assert attn["length"] == block * head_bytes * len(heads)
    got = sl.expected_pieces(
        state, blocks, rank, consumer_tp, producer_tp, layout, kv_heads
    )
    assert {p: sorted(pieces) for p, pieces in got.items()} == want_state
    want = {
        p: attention.get(p, []) + got[p] for p in set(attention) | set(got)
    }
    reads = [_read(p, want[p]) for p in sorted(want)]
    assert sl.coverage_problems(want, reads) == []
    wrong = next(p for p in range(max(producer_tp, consumer_tp)) if p not in want)
    assert sl.coverage_problems(
        want, reads + [_read(wrong, attention[min(attention)])]
    )


BLOCKED_HEAD_PIECES = [
    # (id, producer TP, consumer TP, block, the per-producer head slabs
    # `expected_pieces` returns: each a share of the page, not a block)
    ("nemotron-3-nano p4-d1 1056-2112", 4, 1, 2112,
     {
         0: [(0, 0, 0, 1081344), (0, 1, 0, 1081344)],
         2: [(0, 0, 1081344, 1081344), (0, 1, 1081344, 1081344)],
     }),
    ("nemotron-3-nano p8-d1 544-2176", 8, 1, 2176,
     {
         0: [(0, 0, 0, 1114112), (0, 1, 0, 1114112)],
         4: [(0, 0, 1114112, 1114112), (0, 1, 1114112, 1114112)],
     }),
    ("qwen3.5-27b p8-d2 416-832", 8, 2, 832,
     {
         0: [(0, 0, 0, 851968), (0, 1, 0, 851968)],
         2: [(0, 0, 851968, 851968), (0, 1, 851968, 851968)],
     }),
]


@pytest.mark.parametrize(
    "producer_tp,consumer_tp,block,pieces",
    [c[1:] for c in BLOCKED_HEAD_PIECES],
    ids=[c[0] for c in BLOCKED_HEAD_PIECES],
)
def test_a_multi_head_consumer_reads_head_slabs_nixl_refuses_its_blocks(
    producer_tp, consumer_tp, block, pieces
):
    """The blocked pair classes the stage-1 matrix deferred: at a multi-head
    consumer, `expected_pieces` still returns each KV head's share of the page
    — smaller than the page — and READs covering them satisfy
    `coverage_problems`, so the layout alone would move them. Over NIXL,
    though, the pair is refused before any READ: the connector asserts that
    head-sharded attention reads with more producer than consumer ranks and
    heterogeneous block sizes are unsupported, which
    test_real_kv_transfer.py reproduces as `receive failed`."""
    name = _model_of(BLOCKED_HEAD_PIECES, producer_tp, consumer_tp, block)
    attn, _, layout, kv_heads = _candidate_rows(name, 0, consumer_tp, block)
    blocks = [0, 1]
    got = sl.expected_pieces(
        [attn], blocks, 0, consumer_tp, producer_tp, layout, kv_heads
    )
    assert {p: sorted(pl) for p, pl in got.items()} == pieces
    assert all(
        length < attn["length"] for _, _, _, length in next(iter(got.values()))
    )
    reads = [_read(p, sorted(pl)) for p, pl in got.items()]
    assert sl.coverage_problems(got, reads) == []
