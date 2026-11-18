# Real KV Transfers

Prefill and decode instances run as simulated servers behind any proxy. With a
KV connector, compute stays simulated but the KV cache is real: CPU tensors
that a native connector registers and moves between the instances, so transfers
take their actual time over the actual network, and every byte can be checked.

See [Simulated Execution](../simulated_execution.md) for the simulator itself.

## Prerequisites

- A native connector: `NixlConnector`, `NixlPullConnector` or
  `MooncakeConnector`.
- An explicit `--kv-cache-memory-bytes`: every rank's KV buffers are allocated
  in host memory.
- The CPU image (see
  [Running without a GPU](../simulated_execution.md#running-without-a-gpu)),
  with the NIXL plugins and Mooncake's transfer engine it ships.

## Basic usage

### Producer (prefill)

```bash
vllm serve Qwen/Qwen3-8B --simulate-forward \
  --kv-cache-memory-bytes 4294967296 \
  --simulation-config '{"timing": "zero", "kv_pattern_fill": true,
                        "log_path": "/tmp/prefill-steps.jsonl"}' \
  --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_producer",
                         "kv_buffer_device": "cpu",
                         "kv_connector_extra_config": {"backends": ["UCX"],
                                                       "rank_addresses": ["10.0.0.1"]}}'
```

### Consumer (decode)

```bash
vllm serve Qwen/Qwen3-8B --simulate-forward \
  --kv-cache-memory-bytes 4294967296 \
  --simulation-config '{"timing": "zero", "kv_pattern_check": true,
                        "log_path": "/tmp/decode-steps.jsonl"}' \
  --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_consumer",
                         "kv_buffer_device": "cpu",
                         "kv_connector_extra_config": {"backends": ["UCX"],
                                                       "rank_addresses": ["10.0.0.2"]}}'
```

Requests then go through the usual P/D proxy (for example
`tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`, or
`examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py` for
`MooncakeConnector`). Both instances must use the same simulation and
prefix-caching settings for the byte checks.

## Supported models

| Model family | NIXL | Mooncake | Notes |
| --- | --- | --- | --- |
| Full attention, dense or MoE (experts hold no KV) | yes | yes | |
| Dense MLA (one latent per token and layer) | yes | yes | |
| Sliding-window attention, alone or beside full attention | yes | yes | Each window's last blocks only ([details](#sliding-window-attention)) |
| Hybrid: Qwen's gated delta-net (Qwen3.5, as its language model) | yes | yes | [Hybrid models](#hybrid-models) |
| Hybrid: Mamba2 (`MambaMixer2`: Nemotron-H, Nemotron-3) | yes | yes | [Hybrid models](#hybrid-models) |
| Hybrid: Kimi-K3's delta attention (KDA beside MLA, as its language model) | yes | yes | [Hybrid models](#hybrid-models) |
| Sparse MLA (DSA: DeepSeek-V3.2, GLM-5) | yes | yes | [Sparse MLA](#sparse-mla) |
| DeepSeek-V4 | yes | no | [DeepSeek-V4](#deepseek-v4) |

Not supported: other recurrent layers (Mamba, Kimi-Linear's or OLMo's
delta-nets), a mixer beside attention (Falcon-H1, Zamba2), ReplaySSM or
RecoverSSM, attention-free models, other sparse-MLA models, chunked local or
RSWA attention.

### Sliding-window attention

Each layer kind is a KV cache group; groups share memory, a block being one
group's at a time, of different block sizes when their pages are equal
(Gemma4's and MiMo-V2-Flash's full and sliding layers at most TP sizes). Either
connector moves each window's last blocks only, and says where they start among
the request's blocks (NIXL: `local_first_block_indices`).

Over NIXL a model of several groups is checked through NIXL's own selection: a
group's block in each layer slot NIXL selects for it, the slots of its own
layers only (NIXL pairs regions by layer).

### Hybrid models

- **State per request**: each request's state is one block of its recurrent
  groups (`mamba_cache_mode` none), padded to the attention page. Prefill
  computes the prompt but its last token, which decode computes.
- **With prefix caching** vLLM keeps a state a block (`mamba_cache_mode`
  align): the state after the tokens up to the block's end, which a later
  prompt sharing them reuses, earlier states' places null. The connector moves
  the request's running state.
- The native runner copies a state from block to block around the forward; a
  simulated step computes none and copies none.
- Each decoder layer holds one recurrent mixer at most, without attention
  beside it.
- Kimi-K3's SiTU experts and Nemotron-3's non-gated relu^2 experts are built by
  the CPU platform only for a simulation.

| Connector | How a state moves |
| --- | --- |
| `MooncakeConnector` | The padded state pages. |
| NIXL | Each state as NIXL selects it: the conv sub-projections and the SSM state of each state region of the group's layers, not the page's padding; the pattern checks follow that selection. Needs the DS conv-state layout (`VLLM_SSM_CONV_STATE_LAYOUT=DS`, set before startup). |

### Sparse MLA

DeepSeek-V3.2's and GLM-5's sparse MLA (DSA), over either connector:

- the latent cache in BF16 (`auto` or `bfloat16`) or FlashMLA's packed FP8
  (`fp8`, `fp8_e4m3` or `fp8_ds_mla`: 656 bytes a token);
- each attention layer's latent (576 wide in BF16) and, on the layers holding
  an indexer (GLM-5's every few, by its config), the indexer's FP8 keys and
  scales (132 bytes a token);
- pages of two sizes in one group, each layer its own region; the pattern
  checks cover each layer's page.

### DeepSeek-V4

Over `NixlConnector` (or `NixlPullConnector`), on one rank, with tensor
parallelism, alone or inside data-parallel attention with expert parallelism,
with or without prefix caching:

- each TP rank holds and moves the whole caches;
- the cache is the packed FP8 one (`fp8_ds_mla`): its sliding windows,
  compressed latents, indexer keys and compressor states are KV cache groups
  packed into one allocation, each block all of one group's layers; NIXL moves
  whole blocks;
- decode loads all N prompt tokens and computes the last one again to sample.
  With bounded window replay on (a cache whose SWA spec sets `bounded_replay`,
  DeepSeek-V4.1's), NIXL loads N - 1 and decode computes the last.

## Speculative decoding

Only the native single-module MTP (`method: mtp`, the checkpoint's one MTP
layer, 1 to 5 tokens, the draft at the target's parallelism and precision).
Both instances must run the same speculative configuration.

| Models | Connectors |
| --- | --- |
| DeepSeek-V3, DeepSeek-V3.2, GLM-5 | NIXL or `MooncakeConnector` |
| DeepSeek-V4, Qwen3.5, Nemotron-H | NIXL only |

- **NIXL's handshake** checks the instances' transfer compatibility, the draft
  among it: a peer without the MTP draft is refused.
- **`MooncakeConnector` has no such check**: it refuses a producer layer the
  consumer lacks, not a consumer-only draft or another depth.
- The MTP layer's KV is a layer of the target's KV cache groups and moves with
  them. Qwen3.5's and Nemotron-H's MTP stay NIXL's: Mooncake would also move
  each state's k scratch slots, which NIXL leaves.
- Drafts are not computed; their acceptance follows `spec_acceptance_rates`.

## Parallel layouts

### Tensor parallelism of the same size

Each rank moves its own KV heads to its peer, or the whole MLA latent, which
every rank holds.

### Tensor parallelism of different sizes

The two sizes must divide one another.

| Model | Connectors | Condition |
| --- | --- | --- |
| Plain attention, sliding-window layers included | both | The connector re-shards the KV heads either way; each head's pattern is its own. |
| Plain MLA (DeepSeek-V3, Kimi-K2) | both | The same page on every rank. |
| Sparse MLA (DeepSeek-V3.2, GLM-5), BF16 or packed FP8 | both | The latent's and the indexer's pages, each the same on every rank. |
| Layers of different KV head counts (Gemma4) | both | Every layer's count at least the larger size and divisible by both (Gemma4's 8 and 2: TP 1 with 2). |
| Hybrids of Mamba2 or gated delta-net states | NIXL | The state splits over both sizes and the KV heads split or replicate evenly on each side ([details](#hybrids-at-unequal-tp)). |
| Kimi-K3 | NIXL | [Kimi-K3 pairs](#kimi-k3-pairs). |
| MLA with native MTP (DeepSeek-V3, DeepSeek-V3.2, GLM-5-NVFP4) | NIXL | [MLA with MTP](#mla-with-mtp-at-unequal-tp). |
| DeepSeek-V4 | NIXL | [DeepSeek-V4 at unequal TP](#deepseek-v4-at-unequal-tp). |

Refused: another MLA hybrid, Kimi-K3 at other TP sizes, other forms of
speculative decoding, and unequal-TP MTP over Mooncake (its native layout
exchange does not qualify both endpoints' speculative configuration). A NIXL
decode instance refuses, as NIXL does, a larger prefill instance whose KV heads
are replicated (more ranks than KV heads); MLA's latent is exempt. A refused
pair fails at the handshake.

#### Hybrids at unequal TP

- With Qwen3.5's and Nemotron-H's native MTP too: the conv state is widened to
  `conv_kernel - 1 + k` rows, as vLLM allocates it, and the draft's attention
  layer is read with the target's. Decode refuses a producer of another depth
  before any READ, its state sizes differing.
- Where the two sizes give different attention block sizes (Nemotron-3-Nano:
  1056 tokens at TP 4, 2112 at TP 2; Qwen3.5-27B: 800 at TP 1, 2 and 4, 416 at
  TP 8, a requested 400 kept at TP 8 and aligned to 800 below it; 416 and 800
  pair with neither), only a decode block size that is a multiple of the
  prefill's, as NIXL supports: each decode block is read as several prefill
  blocks, the states one to one, each attention page read whole from one
  producer, including pages holding several KV heads.
- Coverage combines that producer's head intervals only when they own the
  complete page exactly once, before comparing the native wire chunks. NIXL's
  refusal of head-sharded reads from several producers at unequal kernel block
  sizes stays in place.
- With prefix caching, byte checks at unequal attention block sizes need
  `kv_pattern_grain` (see [Byte checks](#byte-checks)). NIXL pairs the two
  sides' sub-blocks from the handoff's last token backwards (decode blocks
  split into producer-sized sub-blocks, each side's past that token dropped;
  it needs the producer's `remote_num_tokens`, which a proxy forwards with the
  rest of its `kv_transfer_params`) and zeroes the part of a decode block it
  did not write. The byte check derives that part from the handoff and the
  producer's retained blocks, not from NIXL's lists.
- Native NIXL refuses hybrid pairs with different ratios when prefix caching
  is enabled.

#### Kimi-K3 pairs

| Pair | Scope |
| --- | --- |
| Prefill TP 8 → decode TP 4, no prefix caching | Its delta-attention states (Q, K and V conv components, the recurrent state FP32 unless `--mamba-ssm-cache-dtype` says otherwise) split as the states above, each decode rank's half of every component from its two producers; its MLA latent, the same on every rank, from one of them, each decode attention block (twice the prefill's tokens) read as two prefill blocks, the uncovered half of a request's last block zero, as NIXL leaves it. |
| Prefill TP 8 → decode TP 4, prefix caching | The published text-only BF16-KV Kimi-K3 with AISimulate's B200 target, whole 768-token prefill and 1,536-token decode pages, align-mode KDA, producer fill, decode byte checks and `kv_pattern_grain=768` on both sides. Before a READ, decode checks each actual producer's advertised scope, registration, and the layers of its regions and their cache groups. Native NIXL pairing by token from the handoff's last token is kept; the byte check verifies it with a decode prefix hit and an incomplete final page. |
| Prefill TP 4 → decode TP 8 | Only the published text-only MXFP4 model with BF16 KV, FP32 recurrent state, CPU NIXL pull, AISimulate's B200 target, prefix caching off, whole 1,536-token prefill and 768-token decode pages, KDA cache mode `none` on both sides, producer pattern fill and decode pattern checks. Decode rank `r` reads from producer `r // 2`. The checker verifies the sources and destinations of the native READ against the original producer block IDs independently of the byte check; the raw registration carries geometry, not the producer's fill or prefix settings. |
| Other cold pairs within TP 1, 2, 4 and 8 | The same published Kimi-K3 and checked CPU NIXL scope; each side uses its whole attention page sized for its TP-sharded state. Native NIXL pairs attention rows in the greatest common unit and keeps each producer's original state-page stride. The checker validates every source separately, with KDA spans matched by global columns and heads, and distinguishes copied patterned attention padding from any uncovered decode rows NIXL zeroes. Native transport limits stay per READ. |

Reverse TP 4 → TP 8 with prefix caching stays refused: NIXL needs matching
physical blocks per logical page for a prefix-cached hybrid.

#### MLA with MTP at unequal TP

On the simulated CPU MLA path, NIXL pull pairs the native single-module MTP of
DeepSeek-V3, DeepSeek-V3.2 and GLM-5-NVFP4 at depths 1 to 5, with unequal TP
sizes from 1, 2, 4 and 8 in either direction:

- Both instances must agree on depth, prefix caching, logical page size, KV
  layout and precision; DCP and other parallel axes remain 1.
- Dense V3 uses whole 16-, 32-, 64- or 128-token BF16 pages. Sparse V3.2 uses
  BF16 latents and GLM-5-NVFP4 packed FP8 latents, each with byte indexer
  keys: logical pages hold 64 or 128 tokens, the latter two native 64-token
  kernel pages.
- Both actual registrations must agree before a READ, including without byte
  checks. Each decode rank reads every target and draft region from the
  producer selected by native replicated-MLA mapping.
- Byte checks cover every moved byte and exact destination intervals, producer
  identity and completion, and report draft latent/indexer totals separately.
  They do not independently audit remote source addresses. Prefix hits are
  checked without rewriting them, including pages completed by accepted
  drafts.
- Timing remains the explicit published AISimulate deployment, with its
  documented MTP approximations. Compute-small fixtures preserving published KV
  geometry are checked as that KV layout, not timed as a smaller model.
- Unequal logical page sizes, DCP and custom kernel declarations stay outside
  this scope; native checks and pairings remain unchanged.

#### DeepSeek-V4 at unequal TP

The published V4-Flash on AISimulate's B200 target, packed FP8 cache, NIXL
pull, tensor parallelism of 1, 2, 4 or 8 alone, prefix caching on or off on
both sides, no MTP or native MTP of any depth from 1 to 5 on each side (equal
in presence), pairs one size dividing the other:

- Every cache layer is one replicated shard, so each decode rank reads whole
  packed rows from producer rank `rank * producer TP // decode TP`; the
  producer ranks it does not read still complete their sends.
- Each endpoint states its qualification as one scope entry (role, prefix
  mode, MTP presence, cache format, geometry digest, GPU target and TP size;
  never the MTP depth). Before registering a peer of another TP size, the
  decode side requires the producer's entry, its own and the peer's registered
  geometry to agree, with byte checks on or off. NIXL's own compatibility hash
  still applies (MTP on one side only is refused by it).
- Each layer's page is checked as the global head's pattern, so any producer
  rank gives the same bytes. The READ itself is checked apart: exactly one
  completed record of the handoff's producer engine and the owner rank,
  delivering each expected row once, and the draft's payload, alignment and
  the row's tail reported apart from the target's, the whole row still
  compared.

### Data parallelism

An MoE model's data-parallel attention with expert parallelism
(`--data-parallel-size N --enable-expert-parallel`):

- Each DP rank holds its own requests' KV. A request's prefill DP rank
  (pinned by Mooncake's proxy, else prefill's load balancer) names itself in
  its transfer parameters (NIXL: its engine id and side-channel port), and
  decode's load balancer picks its own, so the instances' DP sizes may differ.
- A DP rank may be tensor-parallel itself (equal TP sizes on both instances):
  each of its TP ranks moves its own KV.
- With `--data-parallel-multi-port-external-lb` each rank has an API server of
  its own (the port plus its rank) and an external router picks the rank on
  both instances. This needs NIXL, as `MooncakeConnector`'s bootstrap server is
  one an instance.

Not supported: pipeline or prefill context parallelism, nor data parallelism of
dense models (independent engines).

### Decode context parallelism

For a dense MLA model over NIXL's pull connector, each instance's DCP 1 or its
TP size, the TP sizes one dividing the other.

- **Which producer**: a decode rank reads the producer ranks NIXL maps to its
  DCP slice: a replicated producer's one rank, or each sharded rank whose pages
  fall in its slice (one, several, or all of them for a replicated consumer); a
  mapped rank holding none of its pages is only notified.
- **Pages**: a DCP rank's blocks are the global pages vLLM gives it (rank r
  every DCP-th page from r, the interleave NIXL resolves to the page), and are
  filled and checked as those pages. The pages under the producer's handoff are
  the payload; a later page (allocation padding) is moved by NIXL or left
  untouched.
- **Which blocks moved** is taken from the READs NIXL posts, not from their
  contents: each block a READ wrote must hold its page's pattern, and every
  other block the sentinel. The check counts each kind apart
  (`payload_blocks`, `padding_blocks`, `untouched_blocks`). A rank with an
  empty native READ selection takes NIXL's notify-only path and reads nothing;
  a rank whose blocks all lie past the handoff may still read allocation
  padding. Each rank's `kv_layout` record holds its resolved DCP geometry.
- **Prefix caching**: a hit is credited per scheduler block (DCP pages) and a
  decode rank pulls only its pages past its hit, NIXL offsetting its slice by
  it. Pages are then checked by their content, and a page past the prompt
  (padding) by its place and the content it follows; the byte checks need the
  same prefix-caching setting on both instances.
- **Handshake**: NIXL checks the peer's DCP, TP and page size. With DCP on
  either side, transfers require matching P/D logical block sizes even when a
  logical block holds several kernel blocks; the simulator checks logical sizes
  during the handshake, as upstream vLLM's NIXL connector does, and native NIXL
  still checks kernel sizes and region layout. When either side has DCP, the
  simulator checks both instances' scope, including the side of DCP 1: this
  instance's from its config, the producer's from its NIXL metadata
  (`simulated_scope`).
- **Refused with DCP**: Mooncake's connector (which slices no page by DCP
  rank), NIXL's push connector, speculative decoding, pipeline, prefill
  context or data parallelism, expert parallelism and TP sizes neither of which
  divides the other, on either side.

## Block sizes and kernel geometry

Blocks take any multiple of 16 tokens, as on a GPU; MLA blocks are not forced
to the CPU MLA kernel's 16.

For CPU simulations with NIXL pull and AISimulate,
`simulation_config.kv_kernel_backend` names a built-in GPU attention backend
whose kernel block sizes are taken from vLLM's backend source:

- The selected rule, its source and the registered logical and kernel sizes
  are in `kv_layout`. The CPU backend and AISimulate's timing profile stay as
  configured.
- Unset, the previous CPU block rules apply (one kernel block a block), and
  such an instance refuses at the NIXL handshake, before any READ, a peer whose
  blocks hold several kernel blocks.
- For dense MLA, if either peer subdivides its blocks, the two peers must have
  the same logical block size, even without decode context parallelism or byte
  checking; their kernel block sizes can still differ where native NIXL allows
  it.
- The initial rule reader covers plain BF16 KV without drafts: FlashInfer's
  Hopper rule, Triton attention, and the qualified dense MLA rules from
  FlashMLA, FlashInfer MLA, FlashAttention MLA and Triton MLA.

## Byte checks

| Option | Instance | Effect |
| --- | --- | --- |
| `kv_pattern_fill` | prefill (`kv_producer`) | Writes a pattern into each request's KV blocks, derived from the request, the TP rank, the layer and the block's place among the request's blocks of its KV cache group. |
| `kv_pattern_check` | decode (`kv_consumer`) | Checks every byte it receives and logs one `kv_check` record per request, each group's result nested in it. Needs `log_path`. |
| `kv_pattern_grain` | both | With prefix caching at unequal attention block sizes: the prefill's block size, on both instances. Attention content is keyed in pieces of that many tokens and a state by its endpoint, so a decode block holds the same bytes whether pulled, computed or hit. |

### With prefix caching

A block may hold KV computed for another request, so its pattern is its
content's instead: the tokens up to the block's end, as prefix caching keys a
full block. Prefix hits are then compared too, read-only, on both instances
(`kind` hit, apart from the transferred blocks).

- Such a run takes token prompts only: no LoRA adapter, `cache_salt`, prompt
  embeddings or multimodal inputs.
- With speculative decoding, decode writes a block's content once its tokens
  are known and their KV computed, a draft's only once accepted, at the end of
  the step that accepted it (each such write a `kv_write` record, apart from
  the checks).
- A hybrid's state is its block's content too: the tokens it has absorbed,
  which stop one short of the prompt on a prefill instance (decode expects
  that). With a hybrid's MTP (over NIXL only) each state is keyed by its
  endpoint alone, so a pulled state is the producer's at the handoff whichever
  slot the read names: vLLM's scheduler moves it to the running slot when it
  admits the request, and relocates speculative state blocks as drafts are
  accepted or rejected. A state group the pull fills has no hit of decode's
  own.
- MTP drops the last matched block of a prefix hit, so a shared prefix reaches
  a hit only past its first block. A hit ending inside a block is reported as
  outside the checks.
- Decode reuses a cached prefix of its own only where its pulls left one: not
  a sliding window's (the connector moves a window's last blocks only:
  DeepSeek-V4, Gemma4), and for a hybrid its attention blocks over NIXL,
  nothing over Mooncake.

### What a check covers

A group's blocks are its layers' only, each block the page the connector moves
(a state's padding included; in DeepSeek-V4's packed blocks, each layer's real
page, its padding unchecked). The null block that pads a window's head is
neither filled nor checked. The `kv_layout` record describes each group (its
page, the state's real bytes, the stride between blocks) and counts each
storage once, as the connector registers it.

## Transports

### Per-rank addresses

`"rank_addresses": [...]` in the connector's extra config gives one IPv4 KV
address per rank, so each rank can use a NIC of its own.

| Connector / backend | What binds to the rank's address |
| --- | --- |
| `MooncakeConnector` | Its handshakes, TENT's RPC server and `hp_tcp`. |
| `NixlConnector` over NIXL's Mooncake backend alone (`"backends": ["Mooncake"]`) | Each rank's backend segment and `hp_tcp`; NIXL's handshake stays on the instance's side channel (`VLLM_NIXL_SIDE_CHANNEL_HOST`). |
| `NixlConnector` over `["UCX"]` or `["LIBFABRIC"]` | The backend's transport (required, see below). |

- The list holds the ranks of an engine, by TP rank. An MoE model's DP engines
  on one host share it, host-local DP rank by TP rank (`local_kv_rank`), and so
  must be launched together: by one server, or by one multi-port supervisor,
  which tells each server it launches its place among the host's ranks
  (`VLLM_DP_SUPERVISED_RANK_LOCAL`, `VLLM_DP_SUPERVISED_SIZE_LOCAL`; not to be
  set by hand).
- Each rank's transfer engine serves TENT's metrics on the base port
  (`MC_TENT_CONF`'s `metrics/http_port`, else `TENT_METRICS_HTTP_PORT`, else
  9100) plus its host-local DP rank times the TP size plus its TP rank.

### TCP

| NIXL backend | Transport | Requirements |
| --- | --- | --- |
| `["Mooncake"]` | TENT's `hp_tcp` | `rank_addresses` optional. |
| `["UCX"]` | `tcp,self` on the address's interface | `rank_addresses`, even for TP1. |
| `["LIBFABRIC"]` | The direct `tcp` provider, IPv4 only | `rank_addresses`, even for TP1; the interface UP and RUNNING. |

- For UCX and LIBFABRIC, the selected IPv4 address must identify one
  non-loopback interface without IPv4 aliases. Use the CPU image built with the
  corresponding plugins; the harness in `tools/e2e` selects them with
  `server.nixl_backend: ucx` or `libfabric`, keeping `mooncake` as its default.
- **Managed LIBFABRIC** sets provider and interface policy before loading the
  native library. It requires `/opt/libfabric/etc/libfabric.conf` to be absent
  or empty, and refuses configuration or provider overrides and rebinding a
  process to another interface (`FI_LOG_LEVEL`, `FI_LOG_PROV` and
  `FI_LOG_SUBSYS` are allowed for diagnostics). Its agent config uses four
  posting threads and background progress; these configured values do not
  independently observe live threads or provider/rail selection. Backend
  options and policy are logged separately.
- **Accounting**: TENT accounting applies to Mooncake, while UCX and LIBFABRIC
  use NIXL telemetry, byte checks and network counters. For both, the logged
  binding is the environment set before the agent is created; the native
  transport and device selection is not observed
  (`native_transport_observed: false` for UCX).
- **Managed TCP keeps TENT's RDMA transport disabled** (TENT enables it on any
  RDMA NIC it sees) and refuses an `MC_TENT_CONF` that enables it.

### RDMA

| Extra config key | Value |
| --- | --- |
| `kv_transport` | `"tcp"` (default) or `"rdma"`. |
| `rank_rdma_devices` | One `device[:port]` per rank (port 1 by default, 1-255), ordered like `rank_addresses`. Ranks may share a device. |
| `rank_addresses` | Still required: the handshakes and TENT's RPC stay on each rank's address. |

Each binding is set before the native engine is created, and settings that
would move it are refused rather than overridden:

| Path | Binding | Refused |
| --- | --- | --- |
| NIXL `["UCX"]` | `rc,self` on the rank's `device:port` (RC connects its endpoints over UD, which UCX adds for wireup). No TCP transport, so a missing RDMA path fails instead of falling back. | The overrides refused for managed UCX over TCP. |
| NIXL `["Mooncake"]` and `MooncakeConnector` | TENT (`MC_USE_TENT`): its RDMA transport alone, on the bare device name (`topology/rdma_whitelist`) and port (`transports/rdma/device/port`), its TCP, `hp_tcp`, SHM and io_uring transports disabled, in `MC_TENT_CONF` (inline or a file; other settings kept). | `MC_IB_PORT`, `MC_TE_FILTERS`, `MC_TE_FILTERS_EXCLUDE`, `MC_FORCE_TCP`, a different `MOONCAKE_LOCAL_HOSTNAME`, `MC_TENT_CONF` values contradicting the binding; for `MooncakeConnector`, a `mooncake_protocol` other than `rdma` (its default) or a `device_name`. |
| NIXL `["LIBFABRIC"]` | Refused over RDMA: the pinned managed path has no verbs support (NIXL selects libfabric's CXI, EFA, TCP or sockets providers). | |

!!! note
    The image's RDMA support (verbs providers, `librdmacm`, UCX built with
    verbs, rdmacm and mlx5) is used only when a run selects RDMA and its
    containers are given the RDMA devices, `IPC_LOCK` and an unlimited memlock
    limit.

### Troubleshooting: a blocked handshake

A managed UCX or LIBFABRIC `add_remote_agent` can block indefinitely when the
peer's side channel answers but its KV address is unreachable (a wrong
`rank_addresses` entry, a missing route, a dead peer). The call cannot be
interrupted and holds NIXL's single handshake thread: later handshakes wait
behind it and the process does not exit until it is killed. After 30 seconds an
error names the backend and the likely cause; check the peer's KV address and
routes, then restart both instances.

## Logs

Every rank of the last pipeline stage logs its own KV into
`<log_path>[.dp<i>][.tp<r>]`, for its global DP rank `i` and TP rank `r` when
not 0.

| Record | Content |
| --- | --- |
| `kv_layout` | Each group's page, the state's real bytes, the stride between blocks; the rank's DP and TP ranks and host-local DP rank; its transport binding (`nixl_transport`, or `mooncake_transport` for `MooncakeConnector` over RDMA); DCP geometry and kernel block rules where they apply. |
| `kv_transfers` | Each transfer's peer rank and its plan (for a NIXL read, the descriptors selected and the count NIXL merges them into). |
| `kv_check` | One per checked request, each group's result nested in it. |
| `kv_write` | With prefix caching and speculative decoding: each block decode writes once its content is known. |
| `kv_events` | Block allocation; with several KV cache groups or a sliding window, `allocated_groups` lists each request's new blocks by group (the group, the place of the first block among the request's blocks of that group, and the blocks, null ones in place), the connector's pulls included. |
| `spec_acceptance` | With speculative decoding: the rank's requests' drafts and accepted drafts each step. |
| `threads` | The rank's threads. |

Under MoE DP the timing record and the joint steps stay in DP rank 0's logs,
also when a multi-port supervisor launched each rank's server.
