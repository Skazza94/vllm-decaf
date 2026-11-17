# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU engines of sparse MLA MTP under DCP at KV interleaves above 1, with
prefix caching off and on, each against its DCP1 control."""

import pytest

from tests.v1.e2e.test_simulated_forward import (
    _mtp_prefix_dcp_engine_cell,
    mtp_dcp_engine_cell,
)
from tests.v1.worker.simulated.test_aisimulate_timing import (
    GLM5,
    V32,
    requires_mtp_dcp,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = [
    pytest.mark.cpu_model,
    pytest.mark.skipif(not HAS_TRITON, reason="Simulated forward requires Triton."),
    pytest.mark.skipif(not current_platform.is_cpu(), reason="Tests the CPU platform."),
]

# (target, TP, DCP, KV interleave, MTP depth): both ends of each layout's
# range of interleaves and of the qualified depths.
INTERLEAVE_ENGINE_CELLS = [
    pytest.param(V32, 8, 2, 2, 5, id="v32-tp8-dcp2-i2-k5"),
    pytest.param(V32, 8, 4, 64, 2, id="v32-tp8-dcp4-i64-k2"),
    pytest.param(GLM5, 4, 2, 64, 5, id="glm5-tp4-dcp2-i64-k5"),
    pytest.param(GLM5, 4, 4, 2, 2, id="glm5-tp4-dcp4-i2-k2"),
]


def run_uncached_control(directory, monkeypatch, target, tp, k, references):
    """The DCP1 engine of a (target, TP, depth); `references` takes its capacity,
    blocks and planned memory for the candidates of the same target and TP."""
    directory.mkdir()
    return mtp_dcp_engine_cell(directory, monkeypatch, target, tp, 1, k, references)


def run_uncached_candidate(
    directory, monkeypatch, target, tp, dcp, interleave, k, references
):
    """The DCP engine at an interleave, against the control in `references`."""
    directory.mkdir()
    return mtp_dcp_engine_cell(
        directory, monkeypatch, target, tp, dcp, k, references, interleave
    )


def run_cached_engine(directory, monkeypatch, target, tp, dcp, interleave, k):
    return _mtp_prefix_dcp_engine_cell(
        directory, monkeypatch, target, tp, dcp, k, interleave
    )


def assert_cached_candidate_matches_control(actual, reference, dcp, interleave):
    """A hit lands on the scheduler block, whatever the interleave."""
    assert actual["interleave"] == interleave and reference["interleave"] == 1
    assert (actual["dcp"], reference["dcp"]) == (dcp, 1)
    assert actual["capacity"] == dcp * reference["capacity"]
    assert actual["hits"][2] == 1024 < reference["hits"][2]
    assert actual["hits"][1] == reference["hits"][1] == 1024
    assert [worker["planned_bytes"] for worker in actual["workers"]] == [
        worker["planned_bytes"] for worker in reference["workers"]
    ]


@requires_mtp_dcp
@pytest.mark.parametrize("target,tp,dcp,interleave,k", INTERLEAVE_ENGINE_CELLS)
def test_simulate_forward_cpu_sparse_mtp_dcp_interleave(
    tmp_path, monkeypatch, target, tp, dcp, interleave, k
):
    """The cell's checks at an interleave above 1, after the interleave-1
    DCP1 control: capacity, block tables, placement and every step's estimate."""
    references = {}
    run_uncached_control(tmp_path / "dcp1", monkeypatch, target, tp, k, references)
    run_uncached_candidate(
        tmp_path / f"dcp{dcp}", monkeypatch, target, tp, dcp, interleave, k, references
    )


@requires_mtp_dcp
@pytest.mark.parametrize("target,tp,dcp,interleave,k", INTERLEAVE_ENGINE_CELLS)
def test_simulate_forward_cpu_sparse_mtp_prefix_dcp_interleave(
    tmp_path, monkeypatch, target, tp, dcp, interleave, k
):
    """Warm hits, partial suffixes and chunks at an interleave above 1."""
    actual = run_cached_engine(
        tmp_path / "candidate", monkeypatch, target, tp, dcp, interleave, k
    )
    reference = run_cached_engine(
        tmp_path / "reference-dcp1", monkeypatch, target, tp, 1, 1, k
    )
    assert_cached_candidate_matches_control(actual, reference, dcp, interleave)
