# AISim Timing

The `aisimulate` timing provider prices each simulated step with
[AISim](https://github.com/ai-dynamo/aisim)'s forward-pass performance model:
the scheduler's actual batch, one workload per attention data-parallel rank,
from kernel timings measured on the chosen GPU under a vLLM release. There is
no analytical fallback: what AISim cannot time is rejected, not
approximated.

See [Simulated Execution](../simulated_execution.md) for the simulator itself
and the other timing providers.

## Installation

The provider needs the optional `aisim==0.15.0.dev202611140000000105` package.
`docker/Dockerfile.aisim` adds it to the CPU image (see
[Running without a GPU](../simulated_execution.md#running-without-a-gpu) for
`vllm:cpu`). From the repository root:

```bash
docker build -f docker/Dockerfile.aisim -t vllm:cpu-aisim .
```

## Configuration

```bash
vllm serve deepseek-ai/DeepSeek-V3 --simulate-forward \
  --kv-cache-memory-bytes 68719476736 \
  --simulation-config '{"timing": "aisimulate",
                        "timing_args": {"gpu": "h200",
                                        "target_model": "deepseek-ai/DeepSeek-V3"}}'
```

| `timing_args` key | Meaning |
| --- | --- |
| `gpu` | The GPU whose measured kernel timings price the steps (`h100`, `h200`, `b200`, `a100`). |
| `target_model` | The published model timed (see [Target model](#target-model)). |
| `backend_version` | The vLLM version of AISim's timing tables, when not automatic. |
| `quant_modes` | Overrides the published precision; the target is then recorded as modified. |
| `host_us` | Host time added to each step. |

### Timing tables

Automatic table selection with this pin resolves to vLLM 0.30.0 for H100, H200
and B200; A100 keeps its 0.14.0 override. The qualified
[decode context parallelism](#decode-context-parallelism) queries keep explicit
0.24.0 tables.

!!! note
    A table change can substantially change estimates; it does not establish
    improved GPU timing accuracy.

## Target model

The deployment timed is the published model `target_model`, at that model's
published weight precision. The local model only holds the KV cache and serves
the scheduler:

- Its KV and state geometry must match the target's bundled config.
- The local KV cache precision is the one timed.
- The model's own Hugging Face id or a `<org>--<name>/<commit>` directory names
  the target; a derived copy (a directory with `DERIVED.md`) must name it
  explicitly.
- Bundled configs and quantization sidecars are read offline, using a private
  copy when the SDK needs a local model path.

### Weight precision

The published precision is read from the target's full config, a ModelOpt
checkpoint's `hf_quant_config.json` included:

| Published format | How it is read |
| --- | --- |
| Block FP8, MXFP4 experts | A fixed mapping |
| NVFP4, per-tensor FP8, compressed-tensors, mixed per-layer precision | As AISim itself reads the checkpoint |
| Anything else | Refused |

Layers the server quantizes itself are not the published weights. For an
unquantized checkpoint:

| Server option | Timed as |
| --- | --- |
| `--quantization fp8` or `fp8_per_tensor` (per-tensor weights, dynamic activations) | AISim's `fp8` precision of the published model |
| `--quantization fp8_per_block` | AISim's `fp8_block` precision |

The target is then recorded as modified, with its `online_quantization`. The
shorthand is taken alone, as vLLM resolves it: every linear layer and every
expert quantized its way. Refused before the model is built:

- a `--quantization-config` beside it that changes anything (layers ignored,
  per-layer `targets`, another weight or activation precision);
- any other online quantization: another method, `--quantization online` with
  a config, or layers quantized beside a quantized checkpoint.

### Model-specific cases

- **DeepSeek-V4** is timed with its packed FP8 cache (`auto`, `fp8`,
  `fp8_e4m3` or `fp8_ds_mla`; others are rejected) and its FP4 experts.
- **DSA models** (DeepSeek-V3.2, GLM-5): an FP8 cache under any of those FP8
  names is timed and recorded as the packed one (`kv_cache_format:
  fp8_ds_mla`, the configured dtype beside it).
- **GLM-5 on B200** has to name its cache (`fp8`, the GPU's default there, or
  `bfloat16`): `auto` is FP8 on that GPU and BF16 on a CPU host.
- **MiMo-V2-Flash**: the SDK reads its published `hybrid_layer_pattern`
  directly; its sliding layers are marked 1.

### Startup check

Startup times one prefill and one decode step, so a target AISim lacks
data for at those shapes fails before serving. A later batch of another shape
can still lack data and fail then.

## Supported layouts

| Layout | Scope |
| --- | --- |
| Tensor parallelism | Within one node: TP 1, 2, 4 or 8. |
| MoE data-parallel attention with expert parallelism | Each rank of TP 1, 2 or 4, timed as vLLM lays it out (attention on a rank's GPUs, the experts over all of them); up to 8 GPUs, all on one host, launched by one server or a server each by a multi-port supervisor (`--data-parallel-multi-port-external-lb`). Ranks on other hosts are rejected. |
| Decode context parallelism | Within a tensor-parallel group, for the models below. |

## Decode context parallelism

AISim takes global context lengths and prices each rank's share of the
decode KV and the merge of partial outputs. Every DCP configuration requires:

- `dcp_comm_backend` set to `ag_rs`;
- effective Q replication off (`VLLM_DCP_Q_REPLICATE`, when set, overrides
  `dcp_q_replicate`).

GLM-5's native `a2a` with Q replication is not timed
([issue #28](https://github.com/Skazza94/vllm-decaf/issues/28)).

On CPU the simulator preserves vLLM's DCP cache bookkeeping and runs no
attention kernels: the scheduler's blocks span DCP times the local pages.

| Model family | Scope |
| --- | --- |
| Dense MLA (e.g. DeepSeek-V3) | Simulated; can also use real NIXL transfers (see [Real KV Transfers](real_kv_transfers.md#decode-context-parallelism)). |
| Sparse MLA (DeepSeek-V3.2, GLM-5) | One simulated CPU instance, with the packed FlashMLA cache (`fp8_ds_mla`) and the default FP8 indexer. FlashMLA requires fewer than 32 local query heads, the same padded head count before and after gathering, and HiSparse off. |
| GQA (Qwen3-30B-A3B, Qwen3-235B-A22B) | Standalone CPU simulation at TP8/DCP2 (the only layout vLLM allows for their four KV heads at TP up to 8), H100 or H200, BF16, KV interleave 1, 16 or 32, the merge requirements above. |

Notes:

- **GLM-5 on B200** with DCP represents an explicit packed-cache deployment:
  native `auto` instead uses standard FP8 with FlashInfer.
- **GQA**: the native reference is FlashAttention 3; AISim uses its
  default attention selector and vLLM 0.24.0 tables. It models prefill over
  cached context with a KV gather, where native FlashAttention gathers Q and
  merges the ranks' partial outputs. KV page size determines cache geometry;
  changing it did not change the estimates in the qualified queries, and
  AISim takes no interleave (the estimates are the same at each).
- **Not qualified**: GQA models whose layouts need more than eight GPUs
  (Qwen3-8B, MiniMax-M2.5), FP8 KV cache and B200 for GQA; GQA DCP with KV
  transfers, compressed MLA, hybrids, and pipeline, prefill context, data and
  expert parallelism.

### Speculative decoding with DCP

With DCP greater than one, speculative decoding is limited to standalone CPU
simulation of the native single-module MTP of DeepSeek-V3, DeepSeek-V3.2 and
GLM-5, at depth 1 to 5 and AISim vLLM 0.24.0 tables, at the layouts
AISim was qualified for:

| Model | GPU | Weights / cache | Pages | TP/DCP |
| --- | --- | --- | --- | --- |
| DeepSeek-V3 | H200 | Published block-FP8, BF16 model and KV | 16 tokens | 2/2, 4/2, 4/4, 8/2, 8/4, 8/8 |
| DeepSeek-V3.2 | H200 | Published block-FP8, packed `fp8_ds_mla` cache | 64 tokens | TP8 with DCP 2 or 4 |
| GLM-5 | B200 | Published NVFP4, packed cache, explicit `ag_rs`, Q replication off | 64 tokens | TP4 with DCP 2 or 4 |

- **DeepSeek-V3.2 at TP8 with DCP 8** stays refused: AISim times it, but
  on the qualified FP8 FlashMLA path vLLM refuses it natively, since the
  backend requires the local (16) and the DCP-gathered (128) query heads to
  pad to the same decode kernel envelope (64 against 128).
- **KV interleave**: DeepSeek-V3 requires interleave 1 (its CPU MLA backend
  does not declare MTP support above it, and vLLM refuses it). The sparse
  models take any interleave vLLM allows for their 64-token page (1, 2, 4, 8,
  16, 32 or 64), the sparse backend declaring MTP support at them; the
  deprecated `dcp_kv_cache_interleave_size` counts only as 1 or equal to
  `cp_kv_cache_interleave_size`.
- The interleave decides which DCP rank holds which token inside a scheduler
  block (the page size times DCP); the pages, capacity and lookahead do not
  depend on it. AISim takes no interleave: the estimates above interleave
  1 are the interleave-1 estimates with the decode-width classification. The
  sparse backend's batch metadata classifies verified rows differently above
  interleave 1; its interleave-dependent GPU costs are not modeled, and the
  timing assumptions record this.
- **Prefix caching** is qualified for these cells with the default MTP
  lookahead hashes: a hit lands on a scheduler block, and its hash also proves
  the next token. Divergence exactly at a boundary can invalidate the
  preceding block. The single module's prefill/hash lookahead is one token;
  depth k still reserves k slots beyond the target's scheduled tokens. Cached
  context steps keep the plain-MLA timing approximation below; no pricing
  correction is applied. Overrides that disable lookahead hashes or EAGLE
  block dropping remain unqualified.
- **KV offloading and KV transfers** stay outside this combined scope. Over
  NIXL with DCP, vLLM resolves the interleave to the block size at startup,
  and its compatibility check refuses MTP above interleave 1 on the simulated
  dense CPU MLA backend, which does not declare support for it (the sparse
  backend does; sparse MTP with DCP and transfers is outside the qualified
  scope). An instance with DCP above 1 refuses speculative transfers when its
  configuration is checked; an instance at DCP 1 refuses at the handshake,
  once it learns the peer's DCP, before any READ.
- The single draft layer's sharded KV (a sparse model's latent and indexer
  caches) belongs to the target's cache groups and shares their scheduler
  block span; each depth's lookahead enters the scheduler's block demand. Its
  cache contributes to the engine's virtual memory plan even though the timing
  model omits draft KV.

## Speculative decoding

Speculative decoding is timed for the native single-module MTP of
DeepSeek-V3, DeepSeek-V3.2, GLM-5, DeepSeek-V4, Qwen3.5 and Nemotron-H only
(`method: mtp`, the checkpoint's one MTP layer, 1 to 5 tokens, the draft at the
target's parallelism and precision), as AISim's MTP depth:

- a decode step verifies k + 1 tokens a request and runs k target-shaped draft
  layers, the full width even when a step drafted fewer;
- draft weights and KV, prefill draft work and host glue are omitted;
- acceptance comes from `spec_acceptance_rates` and does not change a step's
  cost.

## Known approximations

!!! warning "Plain-MLA cached-prefix steps"
    The pinned AISim version has a cached-prefix timing limitation for
    plain-MLA context steps (not sparse MLA): in DeepSeek-V3 and Kimi-K3
    probes, estimates decrease over some already-positive prefix intervals with
    the new-token count fixed. Prefix-cache hits, later prefill chunks and
    decode's resumed context step take this path; other plain-MLA models
    (Kimi-K2) likely share it, unmeasured. No timing correction is applied
    ([issue #21](https://github.com/Skazza94/vllm-decaf/issues/21)).

The `timing` record lists such approximations, and names the target, the
resolved estimator configuration and its data's vLLM version.
