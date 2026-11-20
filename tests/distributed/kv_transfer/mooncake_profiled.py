# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real MooncakeConnector scheduler and worker constructors, profiling off or on.

The transfer engine is the existing test double (``FakeMooncakeWrapper``); everything
else is the connector's own code. The model configuration is the published Qwen3-8B
``config.json``, written to a temporary directory: no download or mounted models.
"""

import asyncio
import atexit
import contextlib
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import zmq.asyncio

from tests.v1.kv_connector.unit.test_mooncake_connector import (
    FakeMooncakeWrapper,
    _make_test_kv_cache_config,
    _region,
    _xfer_meta,
    patch_worker_dependencies,
)
from tests.v1.kv_connector.unit.utils import create_vllm_config, maybe_update_block_size
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    KVConnectorRole,
    MooncakeConnector,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    SendBlockMeta,
)
from vllm.kv_profiling.perfetto import _load

__all__ = [
    "AsyncMock",
    "FakeMooncakeWrapper",
    "_region",
    "_xfer_meta",
    "events_of",
    "make_scheduler",
    "patch_worker_dependencies",
    "model_dir",
    "profiler_extra",
    "ready_send_meta",
    "serve",
    "sock",
    "worker_of",
]


QWEN3_8B = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "max_position_embeddings": 40960,
    "max_window_layers": 36,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}
_MODEL_DIR: list[str] = []


def model_dir() -> str:
    """A model directory holding only the published Qwen3-8B ``config.json``.

    The connector needs a model configuration, never weights, a tokenizer or a
    download."""
    if not _MODEL_DIR:
        directory = tempfile.mkdtemp(prefix="mooncake-profiler-model-")
        (Path(directory) / "config.json").write_text(json.dumps(QWEN3_8B))
        atexit.register(shutil.rmtree, directory, True)
        _MODEL_DIR.append(directory)
    return _MODEL_DIR[0]


def profiler_extra(tmp_path: Path, **config) -> dict:
    return {
        "network_profiler": {
            "level": "l1",
            "run_id": str(uuid4()),
            "output_dir": str(tmp_path / "profiles"),
            "shutdown_timeout_s": 5,
            **config,
        }
    }


def vllm_config(role: str, extra: dict | None):
    return create_vllm_config(
        model=model_dir(),
        kv_connector="MooncakeConnector",
        kv_role=role,
        block_size=maybe_update_block_size(16),
        kv_connector_extra_config=extra,
    )


@contextlib.contextmanager
def worker_of(role: str, extra: dict | None = None):
    """The real worker (profiling per ``extra``), prepared for a one-region producer."""
    config = vllm_config(role, extra)
    with (
        set_current_vllm_config(config),
        patch_worker_dependencies(),
        patch.object(MooncakeConnectorWorker, "_sync_block_size_with_kernel"),
    ):
        connector = MooncakeConnector(
            config, KVConnectorRole.WORKER, _make_test_kv_cache_config()
        )
        worker = connector.connector_worker
        worker.kv_caches_base_addr = [0x1000]
        worker.block_len_per_layer = [256]
        worker.kv_block_len_per_layer = [256]
        worker.registered_layer_names = ["model.layers.0.self_attn"]
        worker.registered_layer_indices = [0]
        try:
            yield worker
        finally:
            worker.shutdown()
            worker.is_kv_consumer = True


def make_scheduler(role: str, extra: dict | None = None, engine_id: str = "engine-a"):
    config = vllm_config(role, extra)
    return MooncakeConnectorScheduler(config, engine_id, _make_test_kv_cache_config())


def ready_send_meta(
    worker, p_req: str, transfer: str, blocks=((1,),), ready=True
) -> SendBlockMeta:
    meta = SendBlockMeta(
        p_req_id=p_req,
        transfer_id=transfer,
        local_block_ids=[list(group) for group in blocks],
        ready=asyncio.Event(),
    )
    if ready:
        meta.ready.set()
    worker.reqs_need_send[transfer] = meta
    return meta


def sock():
    return AsyncMock(spec=zmq.asyncio.Socket, send_multipart=AsyncMock())


async def serve(
    worker, req_blocks, *, remote_port=1234, tp_rank=0, connection=None, **overrides
):
    """One consumer pull answered by the producer: the response it sent."""
    regions = [_region(0x2000, row_offset=-1, layer_name="model.layers.0.self_attn")]
    metadata = _xfer_meta(
        regions,
        req_blocks,
        remote_hostname="consumer",
        remote_port=remote_port,
        remote_tp_rank=tp_rank,
        **overrides,
    )
    connection = connection or sock()
    with patch.object(worker, "sender_loop", asyncio.get_running_loop()):
        await worker.send_kv_to_decode(b"consumer", connection, metadata)
    return worker._xfer_resp_decoder.decode(
        connection.send_multipart.call_args.args[0][1]
    )


def events_of(profiler, *, timeline=False) -> list[dict]:
    """The closed capture's events with their identities resolved, in order. The request
    timeline rows (``request_phase``, ``step_span``) are left out unless asked for: these
    tests are about the transfer chain."""
    profiler.close()
    meta, events, decoder, _, missing = _load(profiler.output_dir / "events.jsonl")
    assert not missing
    out = []
    for e in events:
        if not timeline and e["kind"] in ("request_phase", "step_span"):
            continue
        out.append(
            {
                "kind": e["kind"],
                "request": decoder.resolve(e["request_ref"]),
                "handoff": decoder.resolve(e["handoff_ref"]),
                "peer": decoder.resolve(e["peer_ref"]),
                "transfer": e["transfer"],
                "subject": e["subject"],
                "status": e["status"],
                "bytes": e["bytes"],
                "selected_descriptors": e["selected_descriptors"],
                "duration0_ns": e["duration0_ns"],
                "aux0": e["aux0"],
                "aux1": e["aux1"],
                "unbound": e["unbound_handle"],
            }
        )
    return out
