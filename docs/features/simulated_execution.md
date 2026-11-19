# Simulated Execution

Simulated execution runs the real vLLM engine (scheduler, KV cache manager,
prefix caching, chunked prefill, preemption, async scheduling and the OpenAI
frontend) but replaces the model forward pass with a timing model. It predicts
a deployment's TTFT, TPOT and throughput without its hardware or weights, and
exercises scheduler and KV cache behavior deterministically on CPU-only hosts.

| Page | Content |
| --- | --- |
| This page | How it works, configuration, timing providers, offline runs, parallelism, CPU hosts |
| [AISim timing](simulated_execution/aisimulate.md) | GPU step times from AISim's performance model: targets, precision, layouts, DCP, MTP |
| [Real KV transfers](simulated_execution/real_kv_transfers.md) | Disaggregated prefill/decode with real KV over NIXL or Mooncake, TCP or RDMA, byte checks |

## How it works

- The model is built on the `meta` device, so KV cache specs, attention
  metadata and hybrid layouts are vLLM's own, with no parameters in memory.
- The KV cache is virtual: blocks are accounted as usual within
  `--kv-cache-memory-bytes`, but no tensors are allocated (except with
  [real KV transfers](#real-kv-transfers)).
- Output tokens come from `SamplingParams.trace_decode_token_ids` (see
  [trace replay](../serving/online_serving/trace_replay.md)) or, without a
  trace, are `fallback_token_id`; such requests end by `max_tokens`.
- A *timing provider* prices each step: its device time (`device_s`) runs on a
  modeled device queue, overlapping the next step's scheduling as under async
  scheduling, and its optional host time (`host_s`, spent in `execute_model`
  and `sample_tokens`) holds the engine thread. The rest of the engine loop is
  the real code and takes its real time.

Simulation runs on CUDA-alike hosts and on the CPU platform (see
[Running without a GPU](#running-without-a-gpu)).

## Quick start

### Online

```bash
vllm serve Qwen/Qwen3-0.6B \
  --simulate-forward \
  --kv-cache-memory-bytes 8589934592 \
  --simulation-config '{"timing": "linear",
                        "timing_args": {"fixed_us": 3000,
                                        "prefill_us_per_token": 20,
                                        "decode_us_per_req": 40},
                        "log_path": "/tmp/sim_steps.jsonl"}'
```

Any client works unchanged, including `vllm bench serve`.

### Offline

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    simulation_config={"simulate_forward": True, "timing": "linear",
                       "timing_args": {"fixed_us": 3000, "decode_us_per_req": 40}},
    kv_cache_memory_bytes=4 * 1024**3,
)
out = llm.generate(
    [{"prompt_token_ids": list(range(300))}],
    SamplingParams(trace_decode_token_ids=[501, 502, 503]),
)
```

!!! tip
    Set `--kv-cache-memory-bytes` to the KV budget of the modeled deployment,
    as its startup log reports it: prefix-cache hit rates and preemption depend
    on it.

## Configuration

`--simulate-forward` turns simulation on; `--simulation-config` (or
`simulation_config` offline) takes:

| Key | Default | Description |
| --- | --- | --- |
| `simulate_forward` | `false` | Simulate the forward pass (what `--simulate-forward` sets). |
| `timing` | `zero` | The [timing provider](#timing-providers). |
| `timing_args` | `{}` | The provider's arguments. |
| `log_path` | none | The step log (JSONL) the run writes. |
| `clock` | `real` | The time base of [offline runs](#offline-runs-in-simulated-time). |
| `record_step_timing` | `false` | On a real deployment, record step times with CUDA events (see [Recording](#recording-a-real-deployment)). |
| `spec_acceptance_rates` | none | Draft acceptance per position, or a recorded step log (see [Speculative decoding](#speculative-decoding)). |
| `kv_pattern_fill`, `kv_pattern_check`, `kv_pattern_grain` | off | Byte checks of [real KV transfers](simulated_execution/real_kv_transfers.md#byte-checks). |
| `kv_kernel_backend` | none | GPU kernel block rules for [real KV transfers](simulated_execution/real_kv_transfers.md#block-sizes-and-kernel-geometry). |
| `fallback_token_id` | `0` | Output token of requests without `trace_decode_token_ids`. |

## Timing providers

| `timing` | Behavior | `timing_args` |
| --- | --- | --- |
| `zero` | Steps take no device time (default). | none |
| `linear` | `fixed_us + prefill_us_per_token * P + decode_us_per_req * D + decode_us_per_kv_token * K` | Those four, plus `host_us` or `host_fit_log` |
| `table` | Nearest-neighbor interpolation over a step log. | `path`, `k` |
| `replay` | Exact match on the step's batch composition, else `fallback`. | `path`, `fallback` (`knn`/`zero`), `k`, `miss_fallback_s` |
| `aisimulate` | AISim's forward-pass estimate of a published model on a GPU. | `gpu`, `target_model`, `backend_version`, `quant_modes`, `host_us` ([details](simulated_execution/aisimulate.md)) |
| `module.Class` | A `vllm.v1.worker.simulated.timing.TimingProvider`. | Its keyword arguments |

A provider receives a `StepDescriptor` and returns a `StepCost`. The
descriptor holds, per request, the tokens scheduled in the step, the KV length
before it, the prompt length, prefix-cache hits at admission, draft tokens,
whether it is prefilling and, in an MoE-DP joint step, its DP rank; plus batch
totals, common-prefix blocks and the parallel configuration.

### Step tables

`table` and `replay` read a recorded step log (see
[Recording](#recording-a-real-deployment)). A `table` can also be written by
hand, as JSONL records of
`{"num_reqs", "num_tokens", "sum_past_kv", "device_s"[, "host_s"]}`: the step's
requests and scheduled tokens, and the sum of their KV lengths before the step.

### Host time

`HostTimeModel` prices host time as
`fixed_us + per_req_us * R + per_prefill_token_us * P`, fitted to a step log
(`HostTimeModel.from_log`, or `host_fit_log` for `linear`), for providers that
model only the device.

- Under async scheduling host time matters only where it exceeds device time,
  mostly in prefills, so it mainly affects TTFT.
- Fit it to a recording of the same parallel layout and speculative decoding
  setup, made under async scheduling; terms the recording does not determine
  stay zero.

## Recording a real deployment

A real deployment writes the same step log, timed with CUDA events:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --simulation-config '{"record_step_timing": true, "log_path": "/tmp/real_steps.jsonl"}'
```

- `replay` with this log reproduces the recorded step times wherever the
  simulated scheduler forms the same batches; `table` interpolates them for
  other workloads. Both return the recorded device and host times.
- Host time matters: large prefills are often host-bound, and pricing them by
  device time alone underestimates TTFT. The recorded host time spans the
  engine thread from `execute_model` to the end of `sample_tokens`, so it
  includes engine core work between the two calls, such as structured-output
  bitmasks.

## Offline runs in simulated time

`vllm.v1.worker.simulated.offline` drives an in-process engine with timed
arrivals of requests and multi-turn sessions, and timestamps every output
token:

```python
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.simulated.offline import TimedRequest, simulate

requests = [
    TimedRequest(prompt_token_ids=list(range(512)),
                 sampling_params=SamplingParams(max_tokens=128, ignore_eos=True),
                 arrival_time=0.05 * i)
    for i in range(200)
]
result = simulate(
    EngineArgs(
        model="Qwen/Qwen3-0.6B",
        kv_cache_memory_bytes=8 * 1024**3,
        simulation_config={"simulate_forward": True, "clock": "fast_forward",
                           "timing": "replay",
                           "timing_args": {"path": "/tmp/real_steps.jsonl"}},
    ),
    requests,
)
print(result.summary())  # as vllm bench serve reports them
```

### Clocks

| `clock` | Waits (device time, idle gaps) | Engine host loop | Use |
| --- | --- | --- | --- |
| `real` (default) | slept | real time | serving |
| `fast_forward` | skipped | real time | sweeps: the latencies of `real`, faster |
| `virtual` | skipped | free | deterministic runs; underestimates host-bound workloads |

- Predicted `host_s` holds the engine thread under every clock.
- The non-real clocks need an engine from `make_engine()` (which `simulate()`
  uses), and a run then takes as long as the engine's host work.
  `make_engine()` and `run()` also drive a real engine on the real clock, to
  produce a reference run.
- [`vllm bench sweep simulate`](../benchmarking/sweeps.md#simulated-sweep)
  runs a `vllm bench sweep serve` experiment this way.
- Only the engine core runs, so `frontend_latency_s` (of `run()` and
  `simulate()`) adds the server's latency outside it (HTTP, API server, IPC)
  to each request's arrival.

### Multi-turn sessions

A `TimedSession` is a multi-turn conversation: each `Turn` appends its tokens
to the previous turn's prompt and output, and arrives `think_time` seconds
after the previous turn finished, so later turns reuse the conversation's KV
cache.

```python
from vllm.v1.worker.simulated.offline import TimedSession, Turn

params = SamplingParams(max_tokens=128, ignore_eos=True)
session = TimedSession(
    turns=[Turn(first_prompt_ids, params)]
    + [Turn(user_ids, params, think_time=3.0) for user_ids in follow_ups],
    arrival_time=0.0,
)
```

- Results have one timing per turn, in input order.
- Prompts are concatenated token ids, so chat-template tokens belong in each
  turn's `new_token_ids`, and `max_model_len` must fit a whole conversation.

## Speculative decoding

The scheduler schedules draft tokens as usual, and the simulated runner
decides how many are accepted. The draft model is built on the meta device
too, and its KV cache counts against the budget.

`spec_acceptance_rates` gives the probability that the draft at each position
is accepted given that all earlier ones were; the last value repeats. A step
log recorded with speculative decoding holds every request's accepted count,
and its path estimates the rates instead:

```python
simulation_config={
    "simulate_forward": True,
    "timing": "replay",
    "timing_args": {"path": "/tmp/real_steps.jsonl"},
    "spec_acceptance_rates": "/tmp/real_steps.jsonl",
}
```

- A provider's `StepCost.num_accepted` overrides the rates.
- Acceptance is sampled independently per step, so per-request acceptance, and
  with it P99 TPOT, varies less than on a real deployment.
- Trace replay is not available with speculative decoding.
- AISim times native MTP only (see
  [AISim timing](simulated_execution/aisimulate.md#speculative-decoding)).

## Parallelism

Tensor, pipeline, data and context (prefill and decode) parallelism run as
deployed: one simulated worker per rank builds its shard of the model, so
per-rank KV capacity is vLLM's own; `--kv-cache-memory-bytes` is per rank. No
collective runs in a step: every rank derives the same timing from the same
scheduler output.

| Parallelism | Behavior |
| --- | --- |
| Pipeline | `device_s` is a step's time through the pipeline, split evenly over the stages, so consecutive microbatches overlap. Steps recorded under pipeline parallelism carry the same through-pipeline time. The last stage's host time includes receiving intermediate tensors, which can bound the step rate under load; fit host time to a recording of the same layout. |
| Data, dense models | Independent engines; DP rank `i > 0` logs to `<log_path>.dp<i>`. |
| Data, MoE models | DP ranks run in lockstep: wherever the real runner synchronizes them, the ranks gather their step descriptors into one joint step, priced once and logged by DP rank 0, busy or idle; each request keeps its DP rank. Joint steps do not record accepted drafts. |

Only the output rank (last pipeline stage, first tensor-parallel rank) writes
the step log. On CUDA hosts each rank binds its own device, using a few MB of
its memory.

## Real KV transfers

With a KV connector, prefill and decode instances run as simulated servers
whose KV caches are real CPU tensors, moved by vLLM's own `NixlConnector`,
`NixlPullConnector` or `MooncakeConnector` over the actual network, over TCP
(TENT's `hp_tcp`, UCX, libfabric) or RDMA (UCX, TENT). Compute stays
simulated; transfers take their actual time, and every received byte can be
checked against a pattern the producer wrote.

| Topic | Where |
| --- | --- |
| Supported models (attention, MLA, sparse MLA, hybrids, DeepSeek-V4) | [Supported models](simulated_execution/real_kv_transfers.md#supported-models) |
| MTP speculative decoding | [Speculative decoding](simulated_execution/real_kv_transfers.md#speculative-decoding) |
| Equal and unequal TP, DP, DCP | [Parallel layouts](simulated_execution/real_kv_transfers.md#parallel-layouts) |
| Byte checks | [Byte checks](simulated_execution/real_kv_transfers.md#byte-checks) |
| Per-rank addresses, TCP and RDMA | [Transports](simulated_execution/real_kv_transfers.md#transports) |
| Step-log records | [Logs](simulated_execution/real_kv_transfers.md#logs) |

The [KV transfer profiler](../observability/kv_profiler_core.md) records each
request's handoff, transfers and peer clocks at the connectors and the endpoints,
for offline Perfetto export and summaries.

## Running without a GPU

A CPU build of vLLM (`VLLM_TARGET_DEVICE=cpu`, which installs Triton's CPU
backend) runs simulations on any x86-64 host:

```bash
TRITON_CPU_BACKEND=1 vllm serve Qwen/Qwen3-235B-A22B --simulate-forward \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --kv-cache-memory-bytes 68719476736 \
  --simulation-config '{"timing": "replay", "timing_args": {"path": "steps.jsonl"}}'
```

The CPU image the examples here and the harness use is built from the
repository root (it needs Python 3.11 or newer):

```bash
docker build -f docker/Dockerfile.cpu --target vllm-openai \
  --build-arg VLLM_CPU_X86=true --build-arg ENABLE_AMX_FP8=0 \
  -t vllm:cpu .
```

### What the CPU platform keeps from the GPU deployment

- **Ranks need no devices**, so a layout larger than any one GPU host runs on
  a CPU host.
- **Scheduling settings**: simulated workers share the host's cores instead of
  each taking a NUMA node (nor does a `NixlConnector` worker pin itself to
  each node's last core for its transfers), and keep the modeled GPU
  deployment's settings (async scheduling, the in-process executor) rather
  than the CPU platform's.
- **MLA** keeps its chunked prefill and prefix caching, and its blocks are not
  forced to the CPU MLA kernel's 16 tokens.
- **Block size**: the KV block size must be a multiple of 16, a GPU's
  smallest: pass `--block-size` to match the deployment.
- **Attention backends**: layers still pick the CPU platform's backends and
  kernels. `--decode-context-parallel-size` above 1 builds for a dense MLA
  model only, whose CPU backend keeps vLLM's DCP bookkeeping in a simulation.
  DSA sparse MLA (DeepSeek-V3.2, GLM-5) builds in a simulation only, keeping
  the GPU deployment's FlashMLA sparse backend for its caches (no kernel
  runs).

### Weight formats

Weight formats build as on the GPU; weight quantization changes neither the KV
cache nor what a transfer moves.

| Case | Behavior |
| --- | --- |
| A checkpoint published with quantized weights (FP8, MXFP4, MXFP8 or NVFP4, ModelOpt's or compressed-tensors': DeepSeek-V3, DeepSeek-V3.2, MiniMax-M2.5, gpt-oss, Kimi-K3, the `-NVFP4` checkpoints) | Builds as published: serve it by its name, with no edited copy of its config. |
| `--quantization` `fp8`, `fp8_per_tensor`, `fp8_per_block`, `fp8_per_channel`, `mxfp8`, `int8_per_channel_weight_only` | Quantizes the layers as they load, as on a GPU: they load on the meta device and end with the deployment's weight dtypes and no data. |
| `--quantization mxfp4`; `nvfp4_per_token` on an MoE model | Not supported: the first's emulation kernel needs the `amd-quark` package; vLLM takes the second on a Blackwell GPU only. |
| A format the CPU has no kernel for | Built and never run: linear layers take vLLM's device-independent emulation kernel, NVFP4 experts the backend a GPU deployment tries first (or `--moe-backend`), per-tensor FP8 experts the Triton one, MXFP4 experts the CPU's MXFP4 backend (which takes Kimi-K3's SiTU activation and grouped sigmoid routing in a simulation only). |
| A checkpoint that also declares a quantized KV cache | Takes that cache dtype as on its GPU: dense MLA keeps an FP8 cache (a byte an element), DSA sparse MLA FlashMLA's packed one (`fp8_ds_mla`, 656 bytes a token), whichever FP8 dtype is given. |

## Testing end to end

`tools/e2e` holds a local harness for simulated prefill/decode serving:

- it builds a topology of Docker containers joined by separate control and KV
  networks;
- serves a run behind [llm-d](https://llm-d.ai)'s router (Envoy, its endpoint
  picker and its disaggregation sidecar, without Kubernetes);
- checks the result: every transferred byte, each rank's transfers and the
  router's choices against what the engines ran.

`tools/e2e/topologies/example.yaml` and `tools/e2e/runs/example.yaml` document
its options. The harness runs over TCP; RDMA runs on real nodes, outside it.

## Limitations

- On CUDA hosts each rank needs its own device (but little of its memory); the
  CPU platform has no such limit.
- Pipeline stages are assumed balanced.
- No LoRA, pooling or diffusion models, and speculative decoding without
  adaptive verification only.
- KV connectors are the native NIXL pull connector and the Mooncake connector,
  moving real KV between served instances (see
  [Real KV transfers](#real-kv-transfers)); no KV offloading and no other
  connector.
- The `fast_forward` and `virtual` clocks need a single process with one rank;
  served and multi-rank simulations run on the real clock.
- Token ids are not meaningful; only their counts are.
- What AISim cannot time is rejected, not approximated (see
  [AISim timing](simulated_execution/aisimulate.md)).
