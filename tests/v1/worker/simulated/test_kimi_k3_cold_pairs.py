"""Real config/planner and offline evidence for cold Kimi-K3 unequal TP."""

import math
from dataclasses import replace
from itertools import permutations

import pytest
import torch

from tests.v1.worker.simulated.test_aisimulate_timing import bundled_config
from tests.v1.worker.simulated.test_real_kv_transfer import KIMI_K3, engine_config
from tests.v1.worker.simulated.utils import sim_req, sim_step
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineTransferInfo,
    TransferTopology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import NixlAgentMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    compute_tp_mapping,
)
from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
    KIMI_K3_MLA_LAYERS,
    kimi_k3_cold_pair,
    simulated_tp_mismatch,
)
from vllm.models.kimi_k3.nvidia.model import KimiK3ForConditionalGeneration
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.core.kv_cache_planning import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec
from vllm.v1.worker.simulated.aisimulate_timing import AisimulateTimingProvider
from vllm.v1.worker.simulated.kv_pattern import (
    SENTINEL,
    NativeChecker,
    _attention_slice,
    _Expected,
    _ExpectedGroup,
    row_pattern,
)
from vllm.v1.worker.simulated.kv_transfer import KimiK3ColdPeer
from vllm.v1.worker.simulated.state_layout import (
    ReadRecord,
    layout_for,
    paired_k3_read_problems,
)

TARGET = "moonshotai/Kimi-K3"
PAGES = {1: 5760, 2: 2944, 4: 1536, 8: 768}
# Primary selected descriptors, primary bytes, state-only bytes, zeroed
# bytes, whole copied-tail bytes (none: NIXL pairs the sub-blocks up to the
# one holding the handoff's last token), partial copied-tail bytes, padded
# state, for a handoff of 1,000 tokens.
VECTORS = {
    (1, 2): (468, 252997632, 224686080, 53084160, 0, 663552, 9326592),
    (1, 4): (348, 144193536, 112343040, 10616832, 0, 4202496, 9750528),
    (1, 8): (348, 88022016, 56171520, 10616832, 0, 4202496, 4875264),
    (2, 1): (468, 252997632, 224686080, 130940928, 0, 663552, 8478720),
    (2, 4): (468, 140654592, 112343040, 14155776, 0, 663552, 9750528),
    (2, 8): (468, 84483072, 56171520, 14155776, 0, 663552, 4875264),
    (4, 1): (348, 144193536, 112343040, 127401984, 0, 4202496, 8478720),
    (4, 2): (468, 140654592, 112343040, 53084160, 0, 663552, 9326592),
    (4, 8): (324, 98638848, 56171520, 0, 0, 14819328, 4875264),
    (8, 1): (348, 88022016, 56171520, 127401984, 0, 4202496, 8478720),
    (8, 2): (468, 84483072, 56171520, 53084160, 0, 663552, 9326592),
    (8, 4): (324, 98638848, 56171520, 0, 0, 14819328, 9750528),
}


@pytest.fixture(scope="module", autouse=True)
def _ds_layout():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
        yield


@pytest.fixture(scope="module")
def config_dir(tmp_path_factory):
    import json

    path = tmp_path_factory.mktemp("kimi-k3")
    (path / "config.json").write_text(json.dumps(bundled_config(TARGET)))
    return path


def endpoint(path, tp, producer):
    return engine_config(
        str(path),
        simulation={
            "timing": "aisimulate",
            "timing_args": {"gpu": "b200", "target_model": TARGET},
            "kv_pattern_fill": producer,
            "kv_pattern_check": not producer,
            "log_path": str(path / "steps"),
        },
        kv_transfer={"kv_role": "kv_producer" if producer else "kv_consumer"},
        dtype="bfloat16",
        max_model_len=6144,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=5 << 28,
        block_size=PAGES[tp],
        tensor_parallel_size=tp,
        language_model_only=True,
    )


@pytest.fixture(scope="module")
def endpoints(config_dir):
    roles = {}
    for producer in (False, True):
        for tp, page in PAGES.items():
            cfg = endpoint(config_dir, tp, producer)
            assert cfg.cache_config.block_size == page
            cls = KimiK3ForConditionalGeneration
            shapes = cls.get_mamba_state_shape_from_config(cfg)
            dtypes = cls.get_mamba_state_dtype_from_config(cfg)
            assert dtypes == (torch.bfloat16, torch.float32)
            mamba = MambaSpec(
                block_size=6144,
                shapes=shapes,
                dtypes=dtypes,
                page_size_padded=page * 1152,
                mamba_cache_mode="none",
                mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            )
            mla = MLAAttentionSpec(
                block_size=page, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
            )
            # Use the native resolver as the engine would, without starting it.
            assert resolve_kv_cache_layout(cfg, [["LBNHC"]], [mamba, mla]).name == (
                "LBNHC"
            )
            groups = get_kv_cache_groups(
                cfg,
                {
                    f"model.layers.{i - 1}": mla if i in KIMI_K3_MLA_LAYERS else mamba
                    for i in range(1, 94)
                },
            )
            pool = get_kv_cache_config_from_groups(cfg, groups, 5 << 28)
            assert sorted(len(g.layer_names) for g in pool.transfer_groups) == [
                23,
                23,
                23,
                24,
            ]
            assert pool.num_blocks == {1: 8, 2: 16, 4: 31, 8: 63}[tp]
            layout = layout_for(cfg)
            assert layout.local_sizes(tp) == (221184 // tp, 6291456 // tp)
            timer = AisimulateTimingProvider(cfg, gpu="b200", target_model=TARGET)
            requests = (
                sim_req("p", 1000, 0, 1000) if producer else sim_req("d", 1, 1000, 1000)
            )
            cost = timer.estimate(sim_step(0, requests)).device_s
            assert math.isfinite(cost) and cost > 0
            roles[producer, tp] = (cfg, pool, layout)
    return roles


def observation(rank, peer, expected, *, p, d, dr, unit, copied, first):
    """Construct wire samples independently using rank-relative split offsets."""
    dest, sources = [], []
    for g, (p_ids, local) in enumerate(zip(expected.remote_block_ids, expected.groups)):
        d_ids = local.block_ids
        for region, base in enumerate(peer.bases):
            if g == 3:
                if rank != first:
                    continue
                width = unit * 1152
                p_units, d_units = PAGES[p] // unit, PAGES[d] // unit
                for j in range(copied):
                    dbk, do = divmod(j, d_units)
                    pbk, po = divmod(j, p_units)
                    dest.append((region, d_ids[dbk], do * width, width))
                    sources.append(
                        (base + p_ids[pbk] * peer.stride + po * width, width)
                    )
            elif region != 23:
                for i in range(4):
                    size = (73728 if i < 3 else 6291456) // max(p, d)
                    prior = i * 73728
                    source = prior // p + (dr % max(1, d // p)) * size
                    into = prior // d + (rank % max(1, p // d)) * size
                    dest.append((region, d_ids[0], into, size))
                    sources.append((base + p_ids[0] * peer.stride + source, size))
    return ReadRecord("p", rank, None, dest, True, "p", sources)


@pytest.mark.parametrize(("p", "d"), list(permutations(PAGES, 2)))
def test_cold_k3_all_directed_pairs_offline(endpoints, p, d, record_property):
    (_, pp, _), (dc, dp, layout) = endpoints[True, p], endpoints[False, d]
    assert simulated_tp_mismatch(dc, d, p, nixl_consumer=True) is None
    assert kimi_k3_cold_pair(dc, p) == (
        None if (p, d) == (8, 4) else (PAGES[p], PAGES[d])
    )
    pages = PAGES[p], PAGES[d]
    unit = math.gcd(*pages)
    fp, fd = (page // unit for page in pages)
    members = [set(range(23)) for _ in range(3)] + [set(range(24))]
    heads = {0: (), 1: (), 2: (), 3: (0,)}
    vector = VECTORS[p, d]
    for h in (1000, unit, max(pages) + 1):
        np, nd = (math.ceil(h / page) for page in pages)
        # NIXL pairs both sides' sub-blocks up to the one holding the last
        # token; both lists start at token 0.
        cropped = math.ceil(h / unit)
        known_p = ((6,), (2,), (4,), (7, 1, 9, 5, 11, 3, 13, 8)[:np])
        if np > len(known_p[3]):  # TP8 may use eight allocations at the long bound.
            raise AssertionError((p, d, h))
        d_ids = [7, 1, 9, 5, 11, 3, 13, 8][:nd]
        expected = _Expected(
            "p",
            [_ExpectedGroup(i, 0, ids) for i, ids in enumerate([[6], [2], [4], d_ids])],
            remote_engine_id="p",
            remote_num_tokens=h,
            remote_block_ids=known_p,
        )
        for rank in range(d):
            first = rank * p // d
            ranks = list(range(first, first + max(1, p // d)))
            topo = TransferTopology(rank, d, PAGES[d], "d", True, True, 1, [])
            topo.register_remote_engine(
                "p", EngineTransferInfo(p, unit * 1152, unit, fp)
            )
            plan = compute_tp_mapping(
                transfer_topology=topo,
                remote_tp_size=p,
                group_spec_types=[MambaSpec] * 3 + [MLAAttentionSpec],
            )
            assert topo.block_size_ratio(unit) == fd
            assert plan.all_source_ranks == tuple(ranks)
            assert plan.source_ranks_per_group == (tuple(ranks),) * 3 + ((first,),)
            assert plan.local_consumers == max(1, d // p)
            peers = {}
            for pr in ranks:
                width = pages[0] * 1152
                bases = [(pr + 1) * (1 << 42) + reg * (1 << 33) for reg in range(24)]
                raw = NixlAgentMetadata(
                    "p",
                    b"",
                    bases,
                    0,
                    pp.num_blocks,
                    [width] * 24,
                    [width] * 24,
                    "LBNHC",
                    pages[0],
                    (221184 // p, 6291456 // p),
                    "CPU_MLA",
                    1,
                )
                peers[pr] = KimiK3ColdPeer(
                    pr, tuple(raw.kv_caches_base_addr), raw.num_blocks, raw.block_size
                )
            assert (
                NativeChecker._cold_k3_aperture(
                    expected, peers, pages, unit, heads, dp.num_blocks
                )
                == cropped
            )
            want = NativeChecker._cold_k3_pairs(
                expected,
                peers,
                p,
                pages,
                unit,
                cropped,
                local_tp=d,
                local_rank=rank,
                layout=layout,
                heads=heads,
                members=members,
            )
            reads = [
                observation(
                    r,
                    peers[r],
                    expected,
                    p=p,
                    d=d,
                    dr=rank,
                    unit=unit,
                    copied=cropped,
                    first=first,
                )
                for r in ranks
            ]
            assert paired_k3_read_problems(want, reads, "p") == []
            state_bytes = 69 * 6512640 // max(p, d)
            des = [len(r.pieces) for r in reads]
            moved = [sum(size for _, size in r.source_ranges) for r in reads]
            row = unit * 27648
            zeroed = (nd * fd - cropped) * row
            whole = max(0, cropped - math.ceil(h / unit)) * row
            partial = (cropped * unit - h) * 27648 - whole
            padding = 69 * (pages[1] * 1152 - 6512640 // d)
            assert des == [276 + 24 * cropped] + [276] * (len(ranks) - 1)
            assert moved == [state_bytes + cropped * row] + [state_bytes] * (
                len(ranks) - 1
            )
            assert sum(moved) + zeroed == 69 * 6512640 // d + nd * fd * row
            if h == 1000:
                assert (
                    des[0],
                    moved[0],
                    state_bytes,
                    zeroed,
                    whole,
                    partial,
                    padding,
                ) == vector
                broken = [
                    reads[1:],
                    reads + [reads[0]],
                    [replace(reads[0], rank=p), *reads[1:]],
                    [replace(reads[0], engine="other"), *reads[1:]],
                    [replace(reads[0], done=False), *reads[1:]],
                    [replace(reads[0], source_ranges=None), *reads[1:]],
                    [
                        replace(
                            reads[0],
                            pieces=reads[0].pieces[1:],
                            source_ranges=reads[0].source_ranges[1:],
                        ),
                        *reads[1:],
                    ],
                    [
                        replace(
                            reads[0],
                            source_ranges=[
                                reads[0].source_ranges[1],
                                reads[0].source_ranges[0],
                                *reads[0].source_ranges[2:],
                            ],
                        ),
                        *reads[1:],
                    ],
                    [
                        replace(
                            reads[0],
                            source_ranges=[
                                (
                                    reads[0].source_ranges[0][0] + 1,
                                    reads[0].source_ranges[0][1],
                                ),
                                *reads[0].source_ranges[1:],
                            ],
                        ),
                        *reads[1:],
                    ],
                ]
                assert all(paired_k3_read_problems(want, rs, "p") for rs in broken)
            if (p, d) in ((1, 2), (2, 1)) and h == 5761:
                assert des[0] == 1380 > 1024
        if h == 1000:
            _offline_content(pages, unit, h, cropped, nd * fd, layout, p, d)
    record_property(
        "proof", "published config/pricing + offline evidence; no native READ"
    )


def _offline_content(pages, unit, h, copied, allocated, layout, p, d):
    from vllm.v1.worker.simulated.state_layout import state_span_pattern

    pp, dp = pages
    fp = pp // unit
    width = unit * 1152
    name, request = "mla", "r"
    producer_pages = [
        row_pattern(request, name, pos, pp * 1152, 0, None, (0,))
        for pos in range(math.ceil(h / pp))
    ]
    samples = {}
    for j in range(allocated):
        page, piece = divmod(j, fp)
        if j < copied:
            actual = producer_pages[page][piece * width : (piece + 1) * width].clone()
            want = _attention_slice(
                request, name, page, pp * 1152, 0, None, (0,), piece, width, fp
            )
            start = j * unit
            if start < h < start + unit:
                samples["partial-tail"] = (actual, want, (h - start) * 1152)
            elif start >= h:
                samples.setdefault("whole-tail", (actual, want, 0))
            else:
                samples.setdefault("payload", (actual, want, 0))
        else:
            actual = torch.zeros(width, dtype=torch.uint8)
            want = torch.full_like(actual, 0)
            samples.setdefault("zero", (actual, want, 0))
        assert torch.equal(actual, want)
    for span in layout.spans(0, d):
        # Gather rank-zero's actual producer shards. When the producer is
        # wider in KV state, take the destination-width piece of its stream.
        fragments = []
        for rank in range(max(1, p // d)):
            parent = next(
                s for s in layout.spans(rank, p) if s.component == span.component
            )
            fragments.append(state_span_pattern(request, "state", 0, layout, parent))
        actual = torch.cat(fragments)[: span.length].clone()
        want = state_span_pattern(request, "state", 0, layout, span)
        assert torch.equal(actual, want)
        samples[span.component] = (actual, want, 0)
    missing = dp * 1152 - sum(layout.local_sizes(d))
    padding = torch.full((missing,), SENTINEL, dtype=torch.uint8)
    samples["padding"] = (padding, padding.clone(), 0)
    assert {
        "payload",
        "partial-tail",
        "Q",
        "K",
        "V",
        "ssm",
        "padding",
    } <= samples.keys()
    if copied > math.ceil(h / unit):
        assert "whole-tail" in samples
    if allocated > copied:
        assert "zero" in samples
    for actual, want, offset in samples.values():
        actual[offset] ^= 1
        assert int((actual != want).sum()) == 1


def test_cold_k3_rule_does_not_admit_tiny_model_or_reverse_prefix(
    tmp_path,
    config_dir,
):
    import json

    (tmp_path / "config.json").write_text(json.dumps(KIMI_K3))
    tiny = engine_config(
        str(tmp_path),
        tensor_parallel_size=4,
        dtype="bfloat16",
        block_size=1536,
        simulation={
            "timing": "aisimulate",
            "timing_args": {"gpu": "b200", "target_model": TARGET},
            "kv_pattern_check": True,
            "log_path": str(tmp_path / "steps"),
        },
    )
    assert kimi_k3_cold_pair(tiny, 2) is None
    assert simulated_tp_mismatch(tiny, 4, 2, nixl_consumer=True)
    consumer = endpoint(config_dir, 8, False)
    consumer.cache_config.enable_prefix_caching = True
    consumer.simulation_config.kv_pattern_grain = 768
    assert simulated_tp_mismatch(consumer, 8, 4, nixl_consumer=True)
