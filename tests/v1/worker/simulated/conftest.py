# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


@pytest.fixture
def ds_conv_layout(monkeypatch):
    """The DS conv-state layout, its cached resolutions cleared around."""
    from vllm import envs
    from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout

    def clear():
        envs.disable_envs_cache()
        get_conv_state_layout.cache_clear()

    def use(layout):
        if layout is None:
            monkeypatch.delenv("VLLM_SSM_CONV_STATE_LAYOUT", raising=False)
        else:
            monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", layout)
        clear()

    use("DS")
    yield use
    monkeypatch.delenv("VLLM_SSM_CONV_STATE_LAYOUT", raising=False)
    clear()
