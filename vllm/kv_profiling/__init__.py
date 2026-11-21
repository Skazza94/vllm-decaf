# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in profiler primitives; importing this package attaches nothing."""


def attach_frontend(vllm_config):
    """The frontend profiler of an engine that owns an input processor, or None.

    Nothing else of this package is imported unless the connector extra
    configuration enables a level. The caller owns the result and closes it.
    """
    kv = vllm_config.kv_transfer_config
    extra = None if kv is None else kv.kv_connector_extra_config.get("network_profiler")
    if extra is None or (
        isinstance(extra, dict) and extra.get("level", "off") == "off"
    ):
        return None
    from .nixl import attach_frontend as attach

    return attach(vllm_config)
