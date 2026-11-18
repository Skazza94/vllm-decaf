# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One side of a real NIXL prefill/decode pair, optionally profiled.

Run as ``python -c "from tests.distributed.kv_transfer.profiled_peer import
main; main()" <role> <model> <target> <spec-json> <dest>``; the workers of a
multi-rank peer import the helpers below by module path.
"""

import gc
import json
import sys
import time
from pathlib import Path


def install_delay(worker, rank, seconds):
    """Hold one decode rank's READ posting: a real delay in front of the real
    ``_read_blocks``, which still runs unchanged."""
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

    if get_tensor_model_parallel_rank() != rank:
        return False
    connector = get_kv_transfer_group().connector_worker
    real = connector._read_blocks

    def delayed(*args, **kwargs):
        time.sleep(seconds)
        return real(*args, **kwargs)

    connector._read_blocks = delayed
    return True


def main():
    from vllm import LLM, SamplingParams

    role, model, target, spec, dest = sys.argv[1:]
    options = json.loads(spec)
    prefill = role == "prefill"
    directory = Path(dest)
    prompt = options.pop("prompt")
    name = options.pop("backend")
    profile = options.pop("profile", None)
    lease = options.pop("lease", 45)
    delay = options.pop("delay", None)
    warm = options.pop("warm", False)
    dtype = options.pop("dtype", "bfloat16")
    pattern = options.pop("pattern", True)
    sim = {
        "simulate_forward": True,
        "timing": "aisimulate",
        "timing_args": {"gpu": "h200", "target_model": target},
        "kv_kernel_backend": name,
        "log_path": str(directory / f"{role}.steps"),
    }
    if pattern:
        sim["kv_pattern_fill" if prefill else "kv_pattern_check"] = True
    extra = {"backends": ["Mooncake"], "kv_lease_duration": lease}
    if profile is not None:
        extra["network_profiler"] = profile
    llm = LLM(
        model=model,
        skip_tokenizer_init=True,
        dtype=dtype,
        enforce_eager=True,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        kv_cache_memory_bytes=128 << 20,
        simulation_config=sim,
        kv_transfer_config={
            "kv_connector": "NixlConnector",
            "kv_buffer_device": "cpu",
            "kv_role": "kv_producer" if prefill else "kv_consumer",
            "kv_connector_extra_config": extra,
        },
        **options,
    )
    token_ids = [5] * prompt
    outcome = {}
    if not prefill:
        if delay:
            llm.collective_rpc(install_delay, args=(delay["rank"], delay["seconds"]))
        if warm:
            warmup = SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False)
            llm.generate([{"prompt_token_ids": token_ids}], warmup, use_tqdm=False)
        handoff = json.loads((directory / "prefill.json").read_text())["handoff"]
    else:
        handoff = {"do_remote_decode": True}
    params = SamplingParams(
        max_tokens=1 if prefill else 2,
        ignore_eos=True,
        detokenize=False,
        extra_args={"kv_transfer_params": handoff},
    )
    try:
        (response,) = llm.generate(
            [{"prompt_token_ids": token_ids}], params, use_tqdm=False
        )
        outcome = {
            "handoff": response.kv_transfer_params,
            "tokens": list(response.outputs[0].token_ids),
            "finish": str(response.outputs[0].finish_reason),
        }
    except Exception as exc:
        outcome = {"error": type(exc).__name__, "message": str(exc)[:500]}
    staging = (directory / f"{role}.json").with_suffix(".tmp")
    staging.write_text(json.dumps(outcome))
    staging.rename(directory / f"{role}.json")
    if prefill:
        core = llm.llm_engine.engine_core.engine_core
        end = time.monotonic() + 240
        while not (directory / "stop").exists() and time.monotonic() < end:
            core.step()
            time.sleep(0.02)
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
