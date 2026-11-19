# KV profiler (experimental)

This library is the core of the KV network profiler. The NIXL pull connector
and the frontend attach to it at level L1 (see "NIXL adapter" below); nothing
else does. Without `kv_connector_extra_config["network_profiler"]` nothing is
imported, started or recorded.

The native recorder requires Linux tmpfs and GIL-enabled CPython. GPU/RDMA
execution, real-node overhead and exact socket/QP attribution are not qualified
by the core tests.

## Explicit library lifecycle

Build the `_kv_profiler` extension through the normal vLLM build, or build just
this standalone target during development:

```console
cmake -S csrc/kv_profiler -B build/kv-profiler \
  -DPython_EXECUTABLE=.venv/bin/python \
  -DCMAKE_INSTALL_PREFIX=.venv/lib/python3.12/site-packages
cmake --build build/kv-profiler
cmake --install build/kv-profiler
```

The connector-extra namespace is reserved for the later adapter.
`NetworkProfilerConfig.from_extra_config` parses it without modifying the
connector configuration. The default `level="off"` starts no helper and
imports no native recorder.

```python
from uuid import uuid4
from vllm.kv_profiling.config import NetworkProfilerConfig
from vllm.kv_profiling.core import open_recorder
from vllm.kv_profiling.schema import EventKind

config = NetworkProfilerConfig(
    level="l1",
    run_id=str(uuid4()),  # Supply the same run ID to all participating endpoints.
    output_dir="/profiles",
    context={"role": "producer", "tp_rank": 0},
)
recorder = open_recorder(config)
assert recorder is not None
# Bind this native callable once. All sixteen fields are positional.
record = recorder.record
record(
    EventKind.REQUEST_BIND, 0, "request", None, None, None,
    None, None, None, None, None, None, None, 0, 0, 0,
)
# Detach and stop producers before closing.
result = recorder.close()
print(result.saved, result.complete, result.output_dir)
```

Record fields, in order:

1. Event kind and numeric context.
2. Request, handoff and peer identities (`str`, `bytes`, or `None`).
3. Optional local handle.
4. Seven optional unsigned values: bytes, selected descriptors, merged
   descriptors, two durations in ns, and two event-specific auxiliary values.
5. Transport (unknown/TCP/RDMA/mixed), status and numeric subject.

`None` is unknown/not supplied; a valid zero remains zero. A transfer bind
allocates a generation for a local handle. Explicit retirement frees the
handle-map slot, even if its retirement observation is lost to a full ring.
A later bind of the same handle gets a different generation. Losing a bind or
retirement to contention invalidates earlier handle associations conservatively;
new binds can establish fresh generations. A conflicting bind invalidates the
old association instead of allowing later events to reuse it. Missing binds
remain explicitly unbound. None of this operates a transport.

The C ABI in `csrc/kv_profiler/recorder.h` accepts the same native event inputs
without Python allocation. The mapping and input buffers must remain alive
through a call; only the creating process may publish. Stop all producers
before closing the mapping. This is not yet an observer/proxy API.

## Publication and loss

Each accepted event reserves space for all first-seen identity fragments and
the event before changing any dictionary. One release-store publishes that
whole group. A reader cannot observe half a committed dictionary.

Published sequence numbers are record commit order, not caller-entry order.
Dropped attempts do not reserve sequence numbers; exact attempt/drop counters,
not sequence gaps alone, establish capture loss. Timestamps are taken inside
the producer section. The producer makes one try-acquire attempt and drops
instead of waiting when contended.

The identity arena and its IDs last until recorder shutdown. They are never
reclaimed or reused after a drain. Input identities are bounded to 512 encoded
bytes, comparisons include the complete bytes and encoding, and map probing is
bounded. Full arenas/maps report loss rather than truncate or alias identities.
The reader preserves dictionary fragments in canonical JSONL; missing or
conflicting fragments stay unresolved during offline export.

The default ring holds 65,536 records of 128 bytes. ID storage, identity slots,
handle slots and batch size are independently bounded; the native mapping
rejects total sizes above 256 MiB. Startup reserves/prefaults the storage.
Memory does not grow with the number of recorded transfers.

## Sink and shutdown

`flush="end"` starts an idle helper and drains only at close.
`flush="background"` drains in a separate process. The consumer owns a kernel
reader lease, never the producer guard. Its death releases that lease.
Readers copy a bounded batch, write and flush it, and only then acknowledge it.
Every row, including metadata and the footer, uses complete-write handling.
A failed write leaves an unacknowledged batch available for recovery; a failed
footer cannot publish successful completion. Consumer death can
cause identical sequence replays; the exporter deduplicates them and rejects
conflicting replays.

Helpers launch their own lightweight module; they do not re-import the serving
entry point or import Torch through distributed-package initializers. They still
pay the installed vLLM package's baseline initialization cost, which can include
Torch imports; the core does not remove or qualify that cost.
All potentially blocking output belongs to the helper. Close signals the ring,
then applies bounded join/terminate/kill waits. When possible an initially
incomplete manifest remains after a stopped/dead/failed helper. The returned
`CloseResult` reports saving and completeness separately; loss is not success.
Failed captures retain their fixed-size shared-memory directory for explicit
recovery. A hard kill can still lose uncommitted records. Do not infer zero
loss from a missing end record.

There is no serving-process JSON writer thread and no recording-path
formatting, I/O, resizing or descriptor traversal. Native allocation tests
cover cached and first-seen IDs; they do not qualify a future adapter.

## Clocks

`ClockServer` is a separately started helper and `measure_peer` performs
bounded exchanges over its own TCP socket. They do not touch a connector
protocol. There is no automatic discovery/connection hook in this checkpoint.

Frames are network byte order: 56-byte PROBE and 88-byte REPLY. Seven samples
by default retain t1/t2/t3/t4 and both clock generations. Select minimum
network RTT; retain raw RTT as sample uncertainty. Persist
`ClockMapping.from_samples(start, end).to_json()` including all samples.
Use the recorder metadata's `clock_id` as the local generation when starting
its `ClockServer` and measuring peers. Missing end calibration means drift
unknown. A generation change is rejected. Failed exchanges produce no mapping;
they must remain unaligned. Helpers exit when their parent disappears.

Projection uses integer differences and rational arithmetic before converting
to trace microseconds. Linear drift interpolation is an estimate, not a bound
on arbitrary mid-run clock changes or network asymmetry.

## Offline Perfetto export

```console
.venv/bin/python -m vllm.kv_profiling.perfetto \
  /profiles/run/process/events.jsonl --output-dir /profiles/perfetto
```

Without calibration, clocks are exported separately. Supply
`--calibration calibration.json` for saved clock mappings and optionally
`--reference-clock UUID`. Aligned processes share **one origin after clock
mapping**; individually zeroing processes would erase real delays.

The exporter retains raw ns, generations, calibration samples, uncertainty,
unknown IDs and losses in args/metadata. Per-process capture status and exact
footer counters remain in metadata even when every event was dropped. The
exporter reconciles the final sequence with the footer's published head. Request observations and transfer
spans are per-process/rank; a DONE span includes polling observation delay.
It does not infer an automatic notification's send instant or an exact
socket/QP. Qualified cross-process handoff flows await the adapter.

`import_modeled_steps` converts an already collected step log offline.
Modeled device/host costs use a separate document and elapsed clock. The importer
retains the timing-provider record and refuses combined logs from restarts;
split those generations first. This elapsed track sums device costs, not wall
clock idle gaps or pipeline overlap.
L1 does not enable a synchronous step logger or add per-token records to
manufacture a missing compute track. The format follows Chrome Trace Event
conventions used by vllm-vcr, not its inferred request-prefill durations.

## Offline analysis summary

```console
.venv/bin/python -m vllm.kv_profiling.summary \
  /profiles/run --output summary.json [--calibration mapping.json ...]
```

`--output` is required. Directories are searched for `events.jsonl`; identical
copies of one process capture fold into one source, conflicting copies are
rejected. The summary reports, per process
and per transfer, the handshake, prepared-to-posted and posted-to-completed
durations (with transfer throughput and NIXL's own integer microseconds),
per-request handoffs, releases and failures, and per-backend/per-transport
percentiles; TCP and RDMA are always separate keys. A request completes only
when every one of its transfers completed and no required failure was
recorded: receive-finished and senders-done are terminal notifications,
recorded with their times but never establishing success, and only an
identified cleanup or notification failure (a late notification, a failed
notification or release) after the completed movement stays cleanup.
Failure codes outside that set, and any failure before the movement
boundary, keep the request failed. Durations stay inside one process's
recorded clock; cross-process handoff-to-first-post spans require saved
calibrations and keep the projection's uncertainty. Every event that
could not be paired is listed in `problems`, and a metric the records do not
carry is listed in `not_available`, never invented: the block-pool return
(`BLOCKS_RELEASED`) is recorded by no adapter, so `release_requested_time_ns`
is the scheduler's permission only. A missing end record means completeness
is unknown, not that the capture is empty: rows flushed before the stop are
summarized, rows left in a ring that was never closed are lost.

## Qualification boundaries

The focused core tests use real recorder mappings, native publishers, helper
processes, files and loopback sockets. They cover publication, loss,
contention, wrap, retirement, failure/recovery and clock/export semantics.

They do not validate connector attachment, forwarding through vectorcall and
ordinary calls, native stats serialization under an observer, data transfer,
or production overhead. Keep the separately pinned timing/image baseline for
later serving A/B tests. No connector behavior or pairing is changed here.

## NIXL adapter (L1)

Enable it on both the prefill and the decode instance with the same run ID:

```json
{"kv_connector_extra_config": {"network_profiler": {
  "level": "l1", "run_id": "<UUID shared by P and D>", "output_dir": "/profiles",
  "transport": "tcp", "clock_port_offset": 1000}}}
```

`transport` is the configured transport (`tcp`, `rdma` or `unknown`) and the
backend list comes from the connector's `backends`; both are recorded as
configured, never as observed. Invalid options, and the push connector, are
refused when the connector is built. The key is not part of the NIXL compatibility
hash, so a profiled instance pairs with an unprofiled one.

Each scheduler, worker and frontend process owns one recorder and writes its own
`events.jsonl`. One worker and one scheduler record per rank; the frontend
profiler is owned by the `AsyncLLM` or `LLMEngine` that builds the input
processor and is closed with it.

Recorded at control points that already run once per request or transfer (the
polling of unfinished transfers is untouched): the prefill handoff and its
block count, the handshake (begin, per-peer end, each waiting request, the
listener's served reply and NIXL's own signed clock offset in nanoseconds as a
cross-check), each READ (`transfer_bind` with the descriptors vLLM selected,
`post_return`, `complete_observed` with NIXL's bytes, merged descriptors and its
own post/transfer microseconds, `transfer_retire`, one per handle including the
legs of mixed-memory reads), every notification sent or observed, all-consumers
done, the scheduler's aggregated receive-finished and release permission, and
every failure (request-scoped from the connector's `failure_type` constants,
transfer-scoped with the handle). Native durations (`aux0`/`aux1` of
`complete_observed`) are NIXL's; the interval between `post_return` and
`complete_observed` timestamps is the first poll that saw DONE, not a NIXL
timestamp.

Not recorded: socket or queue-pair identity, the return of blocks to the pool
(only the scheduler's permission to free them), push-mode transfers, requests of
endpoints other than completions and chat (their router ID is unknown).

Request identity is bound where it exists: the frontend records the exact
`X-Request-Id` header (or whether the body supplied the ID or it was generated),
the API parent and item IDs, the engine ID with its random suffix, and the child
engine ID of a `n > 1` request; the connectors record the P and D engine IDs and
the handoff between them. Joins are exact equality of recorded IDs, never a parse
of an ID's text, and repeated IDs stay separate occurrences.

Clocks: a producer's scheduler starts a clock server at its NIXL side-channel
port plus `clock_port_offset`; a consumer's scheduler calibrates each new peer
(first seen in a handoff) on one background thread, with a bounded peer table
(`clock_max_peers`), and takes end anchors at shutdown under one budget
(`clock_end_budget_s`, capped by `shutdown_timeout_s`). Peers that cannot be
reached or finish after the budget are listed with their status in
`calibration.json`. The exporter treats processes with the same boot ID, time
namespace and clock source as one clock; endpoints spanning hosts stay unaligned.

Hooks never raise into the connector: a hook error disables the profiler and is
counted (`hook_errors`, `first_error` and any `shutdown_failures` in
`profiler.json`; no log handler is called from a hook or from shutdown); a full ring
is counted loss.
Hooks build at most a few small transient integers (a length, a sum) and
retain nothing; formatting, I/O and waiting stay out of them.

Engine shutdown: the scheduler and worker recorders live in the engine-core
process and close when it tears down. With the default `--shutdown-timeout 0`
the API server signals the engine-core process and force-kills it at once, before
its recorders close: their manifests stay `open`, no `profiler.json` is written and
the end flush is lost (the frontend recorders, in the API-server process, still
complete). When the request and process timeouts are both zero, an explicitly set
`VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` gives the engine-core process that much
cleanup grace; set it above the profiler's close deadlines (two recorders,
`shutdown_timeout_s` each, plus the clock budget). A positive `--shutdown-timeout`
is a shared shutdown budget that in-flight requests can consume, so it does not by
itself guarantee that the recorders close.

## Endpoint observations at L3

L3 includes the L1 connector instrumentation (NIXL pull and the Mooncake
adapter below) and declares an external observer. It is off by default. Transport
metrics come from the separate L2 collector (below); eBPF is an opt-in collector
of this observer (below), unqualified until live validation. Sampling, JSON formatting
and kernel queries run in a separate endpoint process, never in a transfer
callback. The native L1 ring ABI and transfer metrics are unchanged.

Use `network_profiler.level: l3`, the same run UUID as L1, and a `probes` object.
`probes` accepts `interval_s` (default 1), `slow_interval_s` (default 5), `tcp`,
`interfaces`, `driver_statistics`, `rdma_ports`, `qp` and `required`. TCP needs
explicit `local_ips` and `peer_ips`; optional `local_ports`, `peer_ports` and
`exclude_ports` narrow discovery. Both address families have kernel and user
filters. Interface names and RDMA `{device, port}` entries are explicit endpoint
allowlists. No intermediate-network observation or packet modification occurs.
`required` names enabled sources (`tcp`, `netdev`, `driver`, `rdma`, `qp`).
Declare requirements before a capture: unsupported optional sources stay
visible as unavailable; failures of required sources invalidate that capture.

For example, the probe settings for a TCP endpoint might be:

```json
{
  "interval_s": 1,
  "slow_interval_s": 5,
  "tcp": {
    "local_ips": ["192.0.2.1"],
    "peer_ips": ["192.0.2.2"],
    "exclude_ports": [5559, 6559]
  },
  "interfaces": ["eth0"],
  "required": ["tcp", "netdev"]
}
```

Replace the documentation addresses, interface and exclusions with the actual
endpoint configuration. An IP match alone can include control traffic.

### Deployment and lifecycle

Locate the installed `vllm/kv_profiling/probe.py` with `importlib.metadata` or
the installation manifest. Run it by absolute file path to avoid importing
vLLM/Torch in the observer:

```console
.venv/bin/python /installed/vllm/kv_profiling/probe.py --config endpoint.json
```

`endpoint.json` has the following runtime fields in addition to `probes`:

| Field | Meaning |
|---|---|
| `run_id`, `endpoint`, `role` | Shared UUID, endpoint ID, `prefill` or `decode` |
| `owner_pid`, `owner_start` | Shared-namespace server PID and proc start ticks |
| `cpus` | Nonempty observer CPU list, disjoint from serving CPUs |
| `irq_policy` | `operator_verified` or `virtual_not_applicable` |
| `input_dir`, `output_dir` | Read-only L1 root and separate probe output |
| `stop_file`, `endpoint_complete_file` | Stop and child-closure paths |

Share the endpoint network and PID namespaces. Check physical IRQ affinities
externally; the observer never retunes them. Existing L1 opening metadata makes
worker discovery possible without adding transfer-path I/O. PID start ticks,
namespace identities, engine and rank are validated before optional fd ownership
reads. Permission denial remains unavailable; do not add privileges to hide it.

Start the observer before admitting workload and wait for its opening manifest.
After child/L1 completion, the controller writes the endpoint completion file:
`{"run_id": "<UUID>", "endpoint": "<ID>", "children_closed": true}`, then
creates `stop_file`. The observer independently checks the L1 manifests, samples
once more and flushes. Unconfirmed owner exit, signal termination, incomplete
child captures,
required-source failures, queue loss and missed ticks cannot yield a complete
capture. Owner exit is not proof of worker/helper closure: the observer allows
a bounded two-second controller notification interval, then marks a timeout
incomplete. The controller must
also collect observer exit/timeout status; kill only its own observer after the
shutdown deadline. Kernel driver I/O can outlast user-space deadlines. Observer
failure never signals or stops the server.

The harness's `profiler.level: l3` additionally requires per-node `probes` and
`observer_cpus` mappings. It launches endpoint-only observer containers with
shared network/PID namespaces, a read-only profile mount and separate output,
then collects manifests and logs. `observer.stdout.log` and
`observer.stderr.log` preserve each Docker log stream; `observer.log` labels them
without implying an order across streams. `endpoint-status.json` records log
collection failures separately from the observer exit/timeout and cleanup.
The generic CLI contains all probe logic.
Current-image availability and real serving shutdown are separate qualification
steps; installing this code does not establish L3 overhead acceptance.

### Data and interpretation

Sidecar JSONL uses `schema: 1, probe_schema: 1`; unknown probe versions are
refused. The Python-only endpoint/status/resource kinds are 22–24, outside the
native ring. Sample begin/end times use `CLOCK_MONOTONIC`, with boot and time
namespace identities for existing clock mapping. Socket, interface and port
observations occupy separate Perfetto lanes; raw values retain their units and
meaning. Configured backend hints never overwrite observed transport provenance.

TCP uses socket cookies and five-tuples. TCP_INFO tail fields absent on an older
kernel are unavailable, not zero. Queue bytes, retransmission gauges/counters,
RTT/variance, cwnd and rate sentinels remain distinct. Polling cannot observe every
short-lived connection; empty snapshots and disappearances are explicit. Dump
interruption, byte/socket limits and timeouts are not complete empty results.
Counter decreases and identity changes break observed continuity. Observation
generations cannot prove that no reuse happened between polls.

Interface and driver statistics describe their whole endpoint link, not a KV
request. Driver units/types stay unknown unless documented. Standard RDMA data
counters retain four-octet units and an explicit byte conversion. Hardware names
are preserved; absent retry/RNR/CNP/WR/CQE fields are unavailable, not fabricated.
Refresh `lifespan` is configuration, never written or counted as traffic.
Optional QP identities are snapshots: QPN reuse between samples is not excluded.
Physical NIC/RDMA counter correctness requires separate real-node qualification.

`probe_join.candidates` makes offline local engine/rank ownership and interval
matches. It validates run/endpoint/namespace/PID identities and projects different
clock domains before comparing times. Unique temporal candidates remain
candidates, never exact READ-to-FD/QP attribution. Shared socket/interface/port
counters are never divided among requests. TCP bootstrap flows are not RDMA data.

The observer bounds sockets (1,024), interfaces/ports (eight each), driver
statistics (4,096), netlink dumps (4 MiB), queued rows (512 × 16 KiB), output size
and polling rates. Large counter sets are chunked. It skips missed ticks rather
than catching up, records gaps/loss and never fabricates completeness. Final
manifests retain observer CPU/RSS accounting separately from serving and L1
helpers. Off-vs-L3 serving overhead, complete capture closure and real RDMA-node
validation must pass their frozen acceptance protocol before qualification.

### Capture inventory and deployment snapshot

The observer runtime config requires `inventory`: the frozen identities of
**every** expected frontend, scheduler and worker capture on that endpoint.
`--inventory <profiles-root> --run-id <UUID>` emits the observed opening
metadata; the deployment must also check those counts against its topology.
The harness checks one frontend, one scheduler per local engine and one worker
per local rank. Unknown, extra, missing or identity-mismatched captures cannot
close successfully. Both the native event stream's terminal/loss state and its
matching manifest are required; a manifest alone is insufficient.

CPU isolation is mandatory for all probe configurations, including netdev-only
and RDMA-only. Every live serving process and thread in the inventory is checked
by PID generation and affinity at each sample. Overlap or unverified placement
invalidates the capture independently of optional fd-ownership permissions.
These are placement snapshots; transient activity between polls is not proven.

The harness makes a deterministic, hash-recorded `observer.pyz` containing only
the stdlib probe package. It mounts that immutable file read-only into the
endpoint sidecar; it does not rely on inherited source mounts or another image
build. The generic CLI also runs from this zipapp. Its package/config hash and
inventory are recorded with the capture. The harness acknowledges each endpoint
as soon as that endpoint's own shutdown completes, before shutting down the next
one, preserving decode-before-prefill clock shutdown.

Offline validation checks the full core inventory, raw observer event file,
terminal records, loss/isolation status, matching run/endpoint/config identities,
package hash and saved observer container exit/timeout/cleanup status. Malformed
or missing evidence fails the verdict rather than being accepted from flags.
Partial input streams qualify candidate associations as `partial_coverage`,
with reasons; an observed unique candidate never conceals incomplete coverage.

For required RDMA observation, `rdma_required_families` explicitly names required
counter directories: `counters` (default), `hw_counters`, or both. At least one
actual counter must be readable in each required family. A `lifespan` value is
refresh configuration and cannot satisfy counter coverage. Optional vendor
families remain reported as unavailable without invalidating another family.

For Docker deployment, the profile directory must have a daemon-exposed backing
path: an existing bind/volume (preferred), or a classic overlay driver's merged
root. Other drivers without an exposed backing path are refused explicitly.
Procfs process-root magic links are not used as Docker bind sources. The helper
uses `docker create` then `docker start`, records the created container ID and
cleans up that ID even if startup fails; it never removes a name collision.

### Opt-in eBPF collector (unqualified until live validation)

The stdlib observer accepts an optional `probes.ebpf` mapping with `enabled`,
`privileged_opt_in`, `sets`, `artifacts` and `kernel_audit`. Off is the default:
there is no native child, eBPF import, capability syscall or automatic privileged
launch. Off configurations keep their previous configuration identity. Existing
saved stdlib captures still validate. A requested source can be made mandatory
with `required: [ebpf]`; unavailable or incomplete required observations prevent
a complete capture.

The implementation provides native NIXL post attempts and endpoint TCP events.
It uses only explicitly positive process-scoped uprobe-multi links and disables
skeleton autoattach. Missing support has no fallback. `provider_wr` is explicitly
unavailable: the current image lacks an audited provider. Owned fixture parsers
exercise list and extended-builder contracts, not real-provider coverage.

Artifacts are compiled separately with `tools/kv-profiler-ebpf/build.sh` in the
accepted tooling image and mounted read-only. The manifest fixes hashes, image,
limits and target kernel/BTF identity. The compiler/object inspector runs without
loading or attaching the object. Starting observation additionally requires
explicit opt-in, effective privilege, controller launch/namespace evidence and
an audited target-kernel allocation/accounting contract. A raw map-size estimate
or a claimed privilege flag alone cannot enable it. No SDK, provider or transport
library is patched or replaced.

The frozen limits include 256 processes, 128 resident thread records per process,
2,048 total, 4,096 phased uprobe slots, a 16 MiB kernel-map cap, call depth four,
64 WRs/list and eight SGEs/WR. All limits are joint refusal bounds; overflow,
missing returns, corrupt metadata and generation changes are explicit failures.
API attempts have observed epochs, not L1 transfer generations. They never gain
request/transfer bindings from a shared thread or matching times.

TCP records distinguish event counts and RX skb buffer lengths from wire or KV
bytes. Zero socket cookies remain namespace-level unknown identities, not guessed
flows. Snapshot records retain their native read spans. API kernel timestamps
and endpoint timestamps remain separate; conversion arithmetic does not establish
runtime clock qualification. The existing probe envelope records delivery time.

The native channel bounds each JSON record to 256 bytes and queues to 4,096
records. The existing common probe envelope keeps its separate 16 KiB record and
512-row writer limits. Kernel epoch retirement waits for acknowledgement after
that writer flushes; live counters are never read-and-zeroed. The observer adds
the exact owned child's final `wait4` CPU and RSS HWM to its helper accounting.
An exit reaped outside this path is explicit missing accounting, never replaced
with cumulative usage from unrelated children. Measured map allocations above
the audited bound or the frozen cap are failures with observed bytes retained,
even for optional eBPF sources; missing accounting remains unavailable.
Shutdown drains buffered output and stderr independently of stdin delivery or
child exit. An abnormal exit or incomplete drain cannot pass as an optional
source; owned-child accounting retains its separate completeness status. Missing
normal-load loss visibility, target-kernel quiescence or privileged cleanup proof
cannot become a qualified capture. The current native terminal explicitly stays
incomplete until those live gates are validated; unprivileged tests do not lift
this restriction. No serving or overhead qualification is implied.

## Mooncake adapter (L1)

The same `network_profiler` namespace in the Mooncake connector's
`kv_connector_extra_config` enables it, on the prefill (producer) and decode
(consumer) instances with one run ID. The enabled-level rule is the shared
configuration's (`NetworkProfilerConfig.enabled`), never a literal level in the
connector: an absent namespace or `off` attaches nothing (no recorder, helper or
clock thread), an invalid one is refused when the connector is built.

Mooncake's producer writes KV into the consumer, so the unit is a **write batch**:
the requests one pull found ready for one consumer rank. Each batch has its own
generation, in the frame that owns it (`send_kv_to_decode`), and the sequence is

```
transfer_bind -> prepared -> post_begin -> post_return
    -> complete_observed | failed -> batch_member / request failures -> transfer_retire
```

`transfer_bind` is preparation (peer: the consumer's `host:port`; subject: its TP
rank; `aux0`: the members). `prepared` is the executor thread picking the batch up
(queue time), `post_begin` and `post_return` bracket the synchronous
`batch_transfer_sync_write` call, and `complete_observed` carries the connector's
own totals (bytes and descriptors posted) and its own timer around the call
(`duration0_ns`, read as `mooncake_write_duration_ns`; an observation interval, not
engine telemetry, and never rendered under a NIXL name). There is no asynchronous
in-flight interval for a blocking write. A write that returns non-zero, raises or is
cancelled has a `failed` (never a fabricated `post_return`); a batch that moves
nothing (a replica of a replicated producer) has a `complete_observed` with explicit
zero bytes and descriptors and status 2, and a batch whose every member was rejected
a transfer-scoped `failed`; neither has `post_begin`/`post_return`. `transfer_retire`
runs in the batch's `finally`, once, after every record that needs the generation.
Each member's outcome is a `request_bind` row with subject `batch_member` (its
`transfer` is the batch; it is not a request occurrence), a rejected member also a
request-scoped `failed`. A member is never attributed bytes of its own: descriptors
are the batch's.

Request-scoped events outside a batch: the producer answering a pull with an error or
for an expired lease, the readiness timeout (`ready_timeout`, a missing producer, not
a transport failure; recorded before the response is sent), the producer worker's
pairing of its request with the shared transfer ID (`request_bind`, subject
`handoff_identity`, never a canonical handoff), the lease expiry, and
`senders_done`. The scheduler records the producer's handoff (after the checks that
make the request deferred; a decode request cancelled before its pull is not one) and
the consumer's, joined by the shared transfer ID, `request_bound`, the aggregation of
finished receives (`recv_finished_all`; not proof of success, a failed request also
reports finished, status 1 when its failure was reported by request) and the release
permission. The consumer's worker records the bootstrap query (`handshake_begin`/
`handshake_end`), each request's rank-local receive completion only when every peer
answered and none failed (the failure flag is sticky: a later peer's success does not
heal an earlier peer's failure), and `failed(receive_failed)`. A recovered
`LAYOUT_MISS` retry is not observed; only a terminal failure is.

The adapter keeps a bounded registry of open generations (`inflight_cap`). If the
profiler closes while a generation is open it records `failed(interrupted)` and the
retire for it first, and marks the capture **incomplete observation** in
`profiler.json` (`observation_complete: false`, `interrupted_generations`) even when
no record was dropped; a later call for that generation records nothing, so a late
success cannot turn an interrupted observation into a complete one. An overflow of the
registry (`inflight_overflow`) is the same. The offline reader reports such a capture
with `loss: true` and names the interruption (`failure_class:
observation_interrupted`), distinct from a connector-reported failure. Closing does
not wait for a transfer and serving shutdown is unchanged.

Clocks: the producer's scheduler binds an explicit `clock_bind` (`host:port`) and the
consumer's maps each remote engine ID to the producer's advertised clock endpoint in
`clock_peers`; no wire message changes and no side-channel offset is added. A peer
without a mapping stays unaligned and is listed in `profiler.json`
(`clock.unmapped_engines`). Capabilities are by process role and KV role
(`capabilities` in the metadata): a worker or scheduler without the producer role does
not claim write batches or handoff blocks.

## Transport metrics collector (L2)

`python <vllm>/kv_profiling/collector.py --run-id <UUID> --output-dir <dir> --manifest
<targets.json>` (the file's path in the source tree or the installed package; it
imports nothing outside the standard library, so it runs without Torch or the rest of
vLLM, and `python -m vllm.kv_profiling.collector` is not the supported form because it
imports the whole `vllm` package first) scrapes each target's Prometheus endpoint (TENT's `/metrics`) from
outside the serving processes, with the standard library only, and writes
`<dir>/<run id>/<process id>/events.jsonl` and `profiler.json`. Two record kinds are
reserved for collector JSONL (native recorder kinds, its ABI and the endpoint-probe
kinds 22-24 are unchanged): 25 `transport_metrics` (by `transport`: bytes written and
read, write and read request counts, the registered-buffer gauge) and 26
`transport_metrics_failures` (transport attempts and attempt failures by `transport`
and `operation`, task failures by `transport` and `reason`). They are different label
partitions of the source: every record carries its own `labels` and typed `fields`
with units, a field the page did not serve is `null` and listed in `missing`
(unavailable, never zero), and no transport-wide value is copied into another
partition. The two kinds of one scrape share the sample ID, the target (engine ID,
role, rank, endpoint, process ID and pid where the manifest gives them), the reset
generation and the scrape's begin and end; every record has its own sequence. A page
that fails or parses incompletely gives missing rows for that half only, with
`partial: true` and the reason; a counter that decreases is a discontinuity and starts
a new reset generation (not a claim that the engine restarted).

Bounds are fixed and validated: response bytes, series examined, labels and label
size, targets, retained output bytes (records past the cap are dropped and counted
as loss, the terminal record always lands), the counter history kept between scrapes
(`max_history` series, least recently seen out first: a series seen again after an
eviction has `baseline: "unavailable"` and the capture says `history_evicted`), a total
deadline per scrape that covers connecting, the headers and every piece of the body
(a peer sending a byte at a time cannot extend it), the startup budget and the
shutdown budget. The shutdown budget starts when stop is seen and covers a scrape in
progress and the final pass together. All records of one sample, of both kinds,
carry one reset generation; a discontinuity in any series moves it once, and each
record keeps its own `discontinuity` flag. In a trace each target, label set and
reset generation is its own counter track (numeric fields only), with the sample's
identity and completeness on an instant event beside it.
Endpoints are IP literals: a hostname makes the target unavailable with that reason, because a name lookup cannot be interrupted and would leave work behind every scrape. A target without an explicit endpoint is resolved from its own recorded
configuration (`MC_TENT_CONF`, nested `metrics/http_port` before the flat key, then
`TENT_METRICS_HTTP_PORT`, then the default), never from the collector's environment;
unreadable or ambiguous ownership makes the target unavailable with a reason. The
collector takes an initial scrape at start and, on SIGTERM or a stop file, one final
scrape before it exits, so the orchestrator must stop it after the workload and
before the engine's metrics server goes away; the terminal record says whether every
scrape was complete (exit status 3 when it was not). Its CPU and the target's
`/metrics` cost are measured, not assumed zero.
