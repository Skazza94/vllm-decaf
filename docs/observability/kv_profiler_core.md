# KV profiler (experimental)

This library is the core of the KV network profiler. The NIXL pull connector,
the Mooncake connector and the frontend attach to it (see the adapters below).
Without `kv_connector_extra_config["network_profiler"]` the serving processes
start no profiler helpers and record no events.

The native recorder requires Linux tmpfs and GIL-enabled CPython, and its
extension, `vllm._kv_profiler`, built into the installed `vllm` package. The
normal build compiles it. A `VLLM_USE_PRECOMPILED=1` install does not, because
the upstream wheel has none: build it as below. An enabled profiler without it
fails at startup with an error that gives the build commands for that
installation. Each recording
process (every scheduler, worker and API server) allocates its ring in
`shared_dir` (`/dev/shm` by default) when the profiler starts: 17.25 MiB at the
default sizes. Docker's default `/dev/shm` (64 MB) holds three, fewer than a
tensor-parallel server opens, so give containers a larger one
(`docker run --shm-size`) or lower `ring_records`. Without room, the server fails
at startup with an error that names the directory, its free space and the size
each recorder needs.

Overhead was measured with two-node serving A/Bs of one workload, profiler off
against on: the NIXL pull connector at concurrency one, with a CPU-simulated
model. Each check compares a 95% bound on latency or throughput with a fixed
limit.

- **L1, default `flush="background"`, UCX/TCP:** p99 time to first token, mean
  time per output token, and mean and p99 inter-token latency were within their
  limits. The bounds for throughput, mean time to first token, and mean and p99
  end-to-end latency were not, though all four point estimates favored L1;
  variation among the profiler-off runs accounted for most of that uncertainty.
- **`flush="end"`, UCX/TCP:** L2 was within every limit. L1 (with the request
  timelines) and L3 were within the throughput and mean-latency limits, not the
  p99 time to first token and p99 end-to-end ones.
- **`flush="end"`, UCX/RDMA:** L1, L2 and L3 together were within every limit.
- **eBPF**, L3 in both arms over UCX/TCP, with the earlier address-based event
  selection: throughput, every mean latency and p99 inter-token latency were
  within their limits; p99 time to first token and p99 end-to-end latency were
  not. Host-wide tracing also exceeded the p99 end-to-end limit. The owned-socket
  selection that replaced the address lists was not measured.

## Limitations

- **Overhead** is measured only for the workload above: not for the Mooncake
  connector, other concurrencies or GPU serving. GPU execution is untested.
- **No per-transfer network attribution:** TCP sockets and RDMA queue pairs are
  never matched exactly to individual transfers; joins with them are candidates.
- **eBPF** covers TCP tracepoints only. There is no per-transfer or per-RDMA-work
  request tracing; RDMA is observed through port counters and queue-pair
  snapshots.
- **No network emulation:** the profiler observes the network, never shapes it.
- **The e2e harness** runs and profiles TCP transports only.

## Explicit library lifecycle

Build the `_kv_profiler` extension through the normal vLLM build, or build just
this standalone target, for example after a `VLLM_USE_PRECOMPILED=1` install.
It installs into `<prefix>/vllm`, so the prefix is the directory that holds the
`vllm` package the service imports: the source tree for an editable install,
`site-packages` otherwise. The startup error of an enabled profiler without the
extension prints these commands for its own interpreter and prefix. To work them
out yourself, run the service's interpreter in the service's environment from
the source tree; `-P` keeps the current directory off the import path, so the
source tree's `vllm` is not picked up by mistake.

```console
python="$PWD/.venv/bin/python"  # the service's interpreter, absolute (not resolved)
prefix=$("$python" -P -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent.parent)')
cmake -S csrc/kv_profiler -B build/kv-profiler \
  -DPython_EXECUTABLE="$python" -DCMAKE_INSTALL_PREFIX="$prefix"
cmake --build build/kv-profiler
cmake --install build/kv-profiler
```

The connector adapters (below) read their settings from the connector-extra
namespace, `kv_connector_extra_config["network_profiler"]`.
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
5. Transport (unknown/TCP/RDMA/mixed), status and numeric subject (at most
   2³²−1; a wider subject is refused).

The recorder adds the recording thread's kernel ID to every event (decoded as
`tid`; word 8 holds the subject in its low 32 bits and the thread in the high
32, binary layout version 2). It costs one `gettid` per record and no
allocation.

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
before closing the mapping. It is not an observer or proxy API.

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

`flush="background"` (the default) drains the ring while the run serves, in a
separate process, so a capture is not bounded by the ring's size.
`flush="end"` starts an idle helper and drains only at close: a capture then
holds at most `ring_records` records per process, and later ones are counted as
`drop_full`. The consumer owns a kernel
reader lease, never the producer guard. Its death releases that lease.
Readers copy a bounded batch, write and flush it, and only then acknowledge it.
Every row, including metadata and the footer, uses complete-write handling.
A failed write leaves an unacknowledged batch available for recovery; a failed
footer cannot publish successful completion. Consumer death can
cause identical sequence replays; the exporter deduplicates them and rejects
conflicting replays.

Each recorder's metadata (the metadata row and the manifest) also carries
`config`, a configuration snapshot taken once when the recorder opens, for
every enabled role (frontend, scheduler, worker): the profiler's own settings
(level, flush, sizes), vLLM's parallel, cache and KV-transfer settings, the
transport environment (variables prefixed `UCX_`, `NIXL_`, `MC_`, `TENT_`,
`MOONCAKE_`, `FI_`, `VLLM_NIXL_`, `VLLM_MOONCAKE_`), TENT's `MC_TENT_CONF`
settings as TENT reads them (file or inline, with its source), library
versions, and, for a NIXL worker, the agent's initialized backends with the
options NIXL reports beside what was requested. Credentials are redacted at any
depth (secret-named keys, URL user information, secret-named query
parameters). The snapshot is at most 4 KiB: its largest entries are dropped
first and named under `dropped`; if the metadata row would still exceed the
reader's line limit, `config` is an explicit `unavailable` entry.

Thread names are best effort and read off the recording path: the helper
reads `/proc/<pid>/task/*/comm` of the recording process every second and at
the end. Before the end record it writes the inventory of every `tid` it
decoded as `threads` rows (`{"type": "threads", "threads": {tid: name}}`, each
under 6 KiB so any number of threads stays readable). A thread that exited
before the helper ever saw it has a `null` name. The end record gives the
count (`thread_count`) and the manifest the whole map (`threads`). A helper
restarted after a crash seeds its inventory from the rows already written.
The offline summary reports each transfer's posting thread (`thread`,
`thread_name`), from its post record, and `null` for a transfer never posted.

Helpers launch their own lightweight module; they do not re-import the serving
entry point or import Torch through distributed-package initializers. They still
pay the installed vLLM package's baseline initialization cost, which can include
Torch imports; the core does not remove or qualify that cost.
All potentially blocking output belongs to the helper. Close signals the ring,
then applies bounded join/terminate/kill waits. When possible an initially
incomplete manifest remains after a stopped/dead/failed helper. The returned
`CloseResult` reports saving and completeness separately; loss is not success.
Failed captures retain their fixed-size shared-memory directory for explicit
recovery. When the recording process dies, its helper saves what the ring holds
and, once saved, removes the ring's directory itself. A hard kill can still lose uncommitted records. Do not infer zero
loss from a missing end record.

There is no serving-process JSON writer thread and no recording-path
formatting, I/O, resizing or descriptor traversal. Native allocation tests
cover cached and first-seen IDs; the adapters' hooks are covered by their own
tests.

## Clocks

`ClockServer` is a separately started helper and `measure_peer` performs
bounded exchanges over its own TCP socket. They do not touch a connector
protocol. There is no automatic discovery/connection hook in this checkpoint.

Frames are network byte order: 56-byte PROBE and 104-byte REPLY (version 2). Seven samples
by default retain t1/t2/t3/t4 and both clock generations. Select minimum
network RTT; retain raw RTT as sample uncertainty. Persist
`ClockMapping.from_samples(start, end).to_json()` including all samples.
Use the recorder metadata's `clock_id` as the local generation when starting
its `ClockServer` and measuring peers. Missing end calibration means drift
unknown. A generation change is rejected. Failed exchanges produce no mapping;
they must remain unaligned. Helpers exit when their parent disappears.

A clock server listens on the family its host resolves to (an IPv6 literal
listens on IPv6 only). A wildcard host listens dual-stack where the host
supports it, and on IPv4 otherwise. A server that cannot listen or start (an
occupied port, an address the host does not have) leaves only the clock
unavailable: the recorder runs, and `profiler.json` reports `clock.server`
with its `state`, its `family` (`ipv4`, `ipv6` or `dual`), the `fallback`
reason of a wildcard that listens on IPv4 only, and the error of an
unavailable one. A port past 65535 (a side-channel port plus the offset, for
example) is refused with its value on both ends, never wrapped; port 0 asks for
an ephemeral one. A reply names its server's engine (a digest of its ID), and a
consumer accepts only the engine it asked for: a peer whose server is
unavailable, even when another server of the run answers on its port, is
recorded as failed with the reason in `calibration.json` (`detail`) and stays
unaligned.

Projection uses integer differences and rational arithmetic before converting
to trace microseconds. Linear drift interpolation is an estimate, not a bound
on arbitrary mid-run clock changes or network asymmetry.

## Offline Perfetto export

```console
.venv/bin/python -m vllm.kv_profiling.perfetto \
  /profiles/run/process/events.jsonl --output-dir /profiles/perfetto
```

Without calibration, clocks are exported separately. Supply each saved clock
mapping with `--calibration`, repeated: the `calibration-N.json` files the
consumer's scheduler saves (its `calibration.json` is the peer list naming them),
and optionally `--reference-clock UUID`. Aligned processes share **one origin after clock
mapping**; individually zeroing processes would erase real delays.

The exporter retains raw ns, generations, calibration samples, uncertainty,
unknown IDs and losses in args/metadata. Per-process capture status and exact
footer counters remain in metadata even when every event was dropped. The
exporter reconciles the final sequence with the footer's published head. Request observations and transfer
spans are per-process/rank; a DONE span includes polling observation delay.
It does not infer an automatic notification's send instant or an exact
socket/QP. A cross-process handoff arrow joins a consumer's read to the
producer's notification only when both are recorded with the same producer and
request, one to one (`cross_process_flows: exact_recorded_handoff_ids`); repeated
or fan-in occurrences draw none.

`import_modeled_steps` converts an already collected step log offline.
Modeled device/host costs use a separate document and elapsed clock. The importer
retains the timing-provider record and refuses combined logs from restarts;
split those generations first. This elapsed track sums device costs, not wall
clock idle gaps or pipeline overlap.
L1 does not enable a synchronous step logger or add per-token records to
manufacture a missing compute track. The format follows the Chrome Trace Event
conventions.

## Offline analysis summary

```console
.venv/bin/python -m vllm.kv_profiling.summary \
  /profiles/run --output summary.json [--calibration calibration-0.json ...]
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

## Request timeline and decode stall attribution

Two native record kinds carry the wall-clock compute timeline of a request at L1. They
are recorded by both connector adapters (NIXL and Mooncake) and by the frontend, at
points that already run once per request or per prefill chunk; nothing is recorded per
token or per descriptor. The modeled step track of `import_modeled_steps` (kind 20)
is a different, simulated clock: the summary never adds it to or compares it with these
wall-clock records.

**`request_phase`** (kind 29), one row per transition. The `subject` is the phase and
the value words are named per phase (`step`, `computed_tokens`, `scheduled_tokens`,
`prefill_tokens`, `output_tokens`, `external_tokens`, `prompt_tokens`):

| phase | recorded by | meaning |
|---|---|---|
| `queued` | scheduler | the engine core accepted the request |
| `load_prepared` | scheduler | the connector was told blocks are allocated for an external load. This is the connector's submission preparation, not the scheduler's park: it follows the park when the load is deferred and precedes it on the immediate path |
| `admitted` | scheduler | the request is in `scheduled_new_reqs`: its first schedule, a decode node's admission once the KV is in, or a re-send after preemption (`status` 1). `prefill_tokens` is the effective prefill length, which includes generated tokens on a resume |
| `prefill_chunk`, `prefill_done` | scheduler | a middle chunk; the step that schedules the last token of the effective prefill |
| `preempted`, `finished` | scheduler | preemption; `request_finished` (`status` 0 done, 1 aborted, 2 error) |
| `first_token_output` | frontend | the first output with a non-empty token list of this request state (once per child; streaming input records its first chunk's first token only) |
| `output_finished` | frontend | the final output was processed (`status` 0), the request was aborted (1: aborted locally, or the engine delivered an abort) or failed (2: an error finish, or rejected on arrival); frontend processing, not delivery to the client |

**`step_span`** (kind 30). The scheduler marks, in the connector metadata, the steps that
carry a phase transition (`profiler_step`, `profiler_scope`, `profiler_flags`: its step
ordinal, its generation, and whether the step is marked or carried no tokens). A worker
records one `step` row per marked step with the hook's own clock readings, taken when the
connector received the step's metadata and when the runner called `finish_forward`
(`begin_ns`, `end_ns`; the row's timestamp is its publication time and is never a step
bound). Every worker step that carries the scheduler's metadata, marked or not, also
feeds a rollup window (`step_window_s`, 1 s by default, up to 60, 0 disables), and every
scheduler step feeds the scheduler's:

- a **worker window** (`step_span` subject 2) counts its steps (`marked_steps` among
  them), their busy (summed) and maximum time, and the zero-token polls apart
  (`empty_steps`, whose time is not busy);
- a **scheduler window** (subject 3) counts its steps with tokens, their
  `scheduled_tokens`, the most requests in one step (`max_requests`), and its empty and
  marked steps.

A window opens at its first step and is published by the first step at least
`step_window_s` after that, or at close: windows are not on a fixed grid, and none is
written while idle (driven by the step hooks; no thread or timer). A worker step without
that metadata feeds no window and is counted (`timeline_steps_without_marker`). The row count scales with request transitions and
prefill chunks, never with decode tokens.

A step's end is the runner's `finish_forward`: the host-side end of the forward, which on
asynchronous device execution is the dispatch return rather than device completion, and in a
simulated worker excludes the device deadline wait that follows. The summary therefore calls
the span `step_host_span`, shared by every request in the step's batch, and never compute.

The summary's `timelines` has one entry per scheduler-side request occurrence, joined from
the scheduler, the workers of the same engine (by the marker's scheduler generation, so a
restart that reuses step numbers never joins the old one) and the frontend (by the existing
identity edges). Its fields:

- `api`: the serving layer's entry (`arrival_ns`, not the HTTP accept), `first_token_output_ns`,
  `output_finished_ns` and their token counts;
- `scheduler`: the phase times, every admission (`later_admissions` lists those after the
  first token, which stay out of the first-token stall), preemptions;
- `tokens`: `tokens_remaining_at_admission` is `prefill_tokens - computed_at_admission`; it is
  not the number of tokens a failed load invalidated, which no hook sees
  (`tokens_attributed_to_failure` stays `None`);
- `kv.outcome`: `received`, `load_failed` (a failure was reported), `load_failed_recomputed` (a
  failure and an admission after the load that starts below the whole prefix a successful load
  leaves cached, which is the evidence of recomputation), `local_hit`, `none` or `unknown`.
  `recomputation_shown` is true or false from that rule (false when no admission followed the
  load) and `null` when the records cannot establish it. The rule counts the local prefix: the
  connector's external token count is incremental to the tokens the local cache held when it was
  asked (`local_cached_at_load`, recorded at `load_prepared`); a successful load resumes at
  `min(local + external, prefill_length - 1)`, the last prompt token staying with the scheduler,
  and an admission below that is the evidence. A replay policy moves that position, so with one
  the result is `null`. Receives, failures, frontend rows and handoffs
  belong to the occurrence they fall in: by sequence inside the scheduler, and in other processes
  by time. Occurrences of one request ID in captures of the same KV role are related: one that
  certainly ended (its scheduler finish) before another was queued is sequential (a restart, the
  next use of a completed ID); anything else is concurrent (independent engines that overlap, a
  missing finish, clocks that cannot be compared). A row of another process belongs to an
  occurrence from its queueing on, the next sequential occurrence taking it from its own queueing
  (the boundary row is the successor's); with a concurrent occurrence of the same ID that may
  have started, and within the clocks' uncertainty of a boundary, it belongs to neither. The API
  entry is matched through the call the prompt belongs to, not through the external ID alone: a
  call records its entry, then one parent-to-item bind per prompt, and the engine binds of its
  prompts come after all of its item binds. The events of one parent are cut into blocks, each
  open until its engine binds balance its item binds and every entry in it has its first prompt.
  A block holding exactly one entry and no item ID twice belongs to that call, so all its engine
  binds have that entry; a block with several entries could give any prompt to any of them
  (ambiguous), and one with none (an entry that was not recorded) or that never closes assigns
  nothing. With an item stage the engine binds name the prompt IDs, without one the parent; a client ID can equal another call's generated prompt ID (clients `batch` and `batch-0`), so the two are never mixed, and a parent whose ID is also some call's prompt ID and that some call used directly (an entry with no first-prompt bind, judged per entry: a later indexed call reusing the ID does not make the earlier direct one indexed) cannot tell its engine binds from that call's (ambiguous), and neither can the parent that owns the colliding prompt: its engine binds cannot be counted either, so none of that parent's prompts is assigned. Calls whose events do not overlap another's are exact; nothing is guessed beyond the recorded order (an entry belongs to the first call whose prompts follow it: which of two calls an entry belongs to cannot be told when an entry-less call's events fall between that entry and its own prompts), and the
  unavailable arrival falls back to the scheduler-queue basis. A parent used both ways (prompts declared under it and engine binds directly under its ID beyond any prompts of that ID) and a prompt ID some engine bind of which comes before the prompt declaration it would need (the k-th bind before the k-th declaration: a direct call's) assign nothing; the comparison is by time, not by totals, because a declaration that belongs to a call whose entry was not recorded, or a bind the capture ended before, would make the totals look balanced. A parent with no item stage pairs each engine bind with the entry before it when exactly one is waiting (none waiting: an entry-less call's, nothing assigned; several waiting: ambiguous, each ambiguous bind consuming one unknown waiting entry). A call whose engine bind never
  arrives (for example one rejected before the engine) keeps later calls with the same client ID
  from being matched. Of the entries paired with an engine ID that was itself reused,
  the occurrence's must precede its queueing, follow the start of a sequential predecessor, and
  not be possibly a concurrent occurrence's.
  Worker failures are assigned to the request only from workers that recorded the scheduler
  generation's marker; others are listed apart (`failures_unlinked`). A failed load is paired
  with its first admission after the load, whichever later admission finishes the prefill.
  An explicit abort or error keeps
  its terminal `state` even before admission;
- `sequences`: the record sequences of the scheduler capture this occurrence owns, from
  its first row to the next occurrence's first (`end` `null`: to the end). A handoff of
  the request lies in exactly one occurrence's range; each join carries its incoming and
  outgoing handoff sequences (`consumer_handoff_sequence`, `producer_handoff_sequence`);
- `kv_writes` (Mooncake): the producers' write batches, each listed once from the producer's own
  records with its recorded call and completion times (`times`), bytes, descriptors and members
  (decode request, transfer ID, share), whether or not the consumer capture is present; each member
  lists the consumer occurrences linked to it through the producer engine, the transfer ID of that
  occurrence's incoming handoff and the member row, only when exactly one incoming handoff names that
  (producer engine, transfer ID, request): a reused identity leaves the write in the inventory
  without a link. `kv.writes` on a consumer timeline references the
  batch and the request's share, so the write and its bytes are counted once in aggregate;
- `steps`: per step the per-rank spans, the expected and present ranks (from the scheduler's
  recorded parallel layout), `complete`, and the shared `host_span_ns` and skews, withheld with
  the reason when a rank's capture or row is missing, the layout is unknown or the clocks differ;
- `intervals_ns` and `stall`: buckets over the basis start (`api_arrival`, or `scheduler_queued`
  when the frontend rows are missing) to the first token output: `pre_engine`, `queue`,
  `waiting_for_kv`, `earlier_admissions` (the work of admissions before the one that finished the
  prefill, up to their preemption), `preempted` (the wait from each preemption to the next
  admission), `step_host_span`, `between_steps`, `after_step_to_first_token` (sampling, output
  transport and frontend processing; never charged to compute) and `other`. Every interval between
  two recorded boundaries is counted in one bucket; a negative remainder is a problem, never
  clipped. `first_step_host_span` and `prefill_done_step_host_span` are different steps when the
  prefill ran in chunks, and each step carries its own `scheduler_to_step_begin_ns`;
- `decode_window`: exact frontend times and, only for a normal finish with both token counts,
  `mean_time_per_token_after_first_ns`; no inter-output mean is derived. `node_cadence` is the
  workers' rollup windows lying entirely inside the window, node-level and shared by the batch;
- `cross_node`: the producer join (NIXL on producer engine and request; Mooncake on producer
  engine and transfer ID), the signed `consumer_queued_minus_producer_handoff_ns` (a concurrently
  launched consumer can queue before the producer's handoff), `ordering` and `producer_pending_ns`
  with the calibration's uncertainty; without a calibration or a shared clock domain they are
  unavailable.

Every interval is computed inside one clock domain or between calibrated clocks; whatever the
records cannot carry goes to the timeline's `unavailable` and the summary's `not_available`,
unmatched rows to `problems`. A figure that sums rows or depends on a row's absence is withheld
when any capture it depends on (the scheduler, the frontend used, the workers of the engine) has
loss, a missing or incomplete end record or an unhealthy adapter: the timeline is `observed_only`
with that capture listed under `quality`, its totals are reported under `*_observed` names, and
each step keeps every worker observation (`rank_observations`) with the shared span only observed.
Only the workers that recorded the current scheduler generation's marker count for a step.
A step is complete only with exactly one row for each expected rank (a rank recorded twice is
ambiguous) from healthy captures on one clock domain. A cross-node block and a decode window list
the quality of the producer or frontend capture they use.

## The request explorer (viewer)

```console
.venv/bin/python tools/kv-profiler-viewer/viewer.py RUN_DIR [RUN_DIR ...]
  [--port N] [--host H]
```

It serves on `127.0.0.1:8321` by default, with no authentication: binding
`--host` to another address exposes the runs' captures to whoever can reach it.
It reads UCX protocol tables from `ucx-protocols/**/*.json` under each run
directory (the parser's output, as the e2e harness saves it); without them the
page says UCX protocol selection was not captured.

With data parallelism each scheduler of an instance calibrates its own clock
against every peer. Schedulers of one clock domain read one clock, so any of
them maps a peer. A mapping is considered only when its samples carry its
capture's run and clock and the peer its manifest names (otherwise it is
reported and unused); per peer the viewer uses one mapping (complete anchors first,
then the reference scheduler's own), relabelled to the reference clock, for the
summary and the page alike, and marks the others as not used. Calibrations from
different clock domains are refused. Engine labels carry the recorded DP rank
(`P0·dp1`), and thread lanes are grouped by engine.

The accepted page (`viewer.html`, beside the server) over one run directory's
captures: per-request rows, the request × time lanes, the record detail and a
run switch when several RUN_DIRs are given, each labelled by the transport its
records carry. The server is standard library only and imports no vLLM or
torch: the pairing is the summary's own, loaded by path (as with
`ucx_protocols.py`), and `vllm/__init__.py` never executes. It is a repository
tool, not part of the installed package: it reads the profiler's modules from
the `vllm/kv_profiling` of the same checkout.

Captures are found by their own metadata row, never by directory names: L1
recorder captures feed `summarize` (naming the calibration mappings from the
captures' `calibration.json` manifests, never raw manifest files); the L2
collector's kinds 25–28 and the L3 observer's `probe_schema` sidecars are
decoded and deduplicated by the server, which never mixes them into the
summary. Identical copies of one capture fold; conflicting rows for one
process generation are named problems. A saved `summary.json` is used only
when its `viewer_input_fingerprints` (per L1 capture content hashes plus the
hashes of every calibration manifest, its resolved named mapping file and
each L1 `profiler.json`, published at `/api/run`) still match; without
fingerprints the run is rebuilt. The inventory reports a capture aligned
when a projection from its clock actually succeeds.

Time lanes project onto one reference clock (the consumer scheduler's when
calibrated): same boot, time namespace and clock source are one clock; other
clocks project through the manifest-named mappings only, and a start-only
mapping means offset projection with uncertainty and unknown drift. A clock
with no evidence of alignment contributes no lanes — absent, never shifted by
a guess.

Every value the page shows is served from the records or an explicit
unavailable/absent state. The metrics follow what the records measure:

- A transfer's interval is the connector's own: acceptance to terminal for an
  asynchronous NIXL READ, call entry to return for a synchronous Mooncake
  WRITE with its terminal status qualifying its bytes. Peak concurrency
  sweeps those intervals — over the run and within each bin (two sequential
  transfers in one bin peak at one); per-pair goodput spreads completed
  payload bytes uniformly over the measured spans, labelled an estimate;
  NIXL's own `xfer_us` is used for per-transfer throughput where the
  connector recorded it.
- Cumulative rates (thread CPU, collector counters) divide the delta between
  consecutive readings of the same series by the sampling interval (reading
  start to reading start when both are recorded, end to end otherwise),
  never by the time spent reading the source; the tick rate comes from the
  enclosing observation, both reading endpoints are projected, and a strip
  draws over the interval its rate covers — from the previous reading's
  end to this reading's end. A decrease, the writer's
  `discontinuity`/`partial` flags or a `reset_generation` change start a new
  generation; rates keep each field's own unit, and the registered-buffer
  gauge is carried as its sampled byte value, never differenced into a
  rate.
- NIC IRQ and softirq rates difference each entity per CPU independently
  (the writer's parallel `cpus`/`counts` arrays read as recorded); a CPU
  absent from the prior reading never credits its earlier count.
- Thread strips join an L1 process only by boot id, namespaces, pid, process
  start ticks and the thread's own lifetime (a strip rides the connecting
  transport's own worker); utilization is labelled shared thread activity
  over its sampled interval. Socket RTT is a per-five-tuple,
  per-incarnation series read from the writer's `transport.tcp`, never a
  pair; requests get RTT and retransmit values only through a unique
  probe-join candidate whose samples overlap the transfer's own measured
  window (retransmits counted by their event type), labelled in the table.
- eBPF events are placed only with a qualified time conversion, enablement
  is read from the writer's own `config.ebpf`, and the run's coverage is
  reported disabled, partial or complete (zero events included).
- Frontend arrival stays separate from the first observed record; a logical
  request is the consumer scheduler's request keyed by its process
  generation, request id and occurrence (two generations reusing one id
  stay two rows); transports attach to their own request's row only (an
  unrelated request and a reused id never share), a Mooncake batch attaches
  once to each member request's row with that member's own share of the
  batch's measured totals; the frontend's recorded parent edges fold its
  client, composed and engine ids into one row, and entries attaching to
  nothing are listed separately with the reason.
- The request timeline columns come from the summary's `timelines` (see
  "Request timeline and decode stall attribution"), never estimated:
    - from the decode node's own timeline: KV wait (`stall.waiting_for_kv_ns`),
      decode admission (`intervals_ns.queued_to_admitted`), the decode window
      (first token output to the finish), and that node's TTFT (its API entry to
      its first token output, only with an `api_arrival` basis);
    - the prefill window (`intervals_ns.prefill_window`), measured on the
      producer's own clock, only through the row's one confirmed handoff join
      (`joins`) whose producer identity is not reused: the producer scheduler it
      names (`producer_sources`, a `producer` or `both` scheduler), and the producer
      occurrence whose record sequences contain the outgoing handoff
      (`producer_handoff_sequence`).

  A row is a connector occurrence. Its scheduler occurrence is the timeline
  whose `sequences` contain the row's own incoming handoff (the record
  sequence the summary's request entry lists, so no join and no producer
  capture is needed): the two numberings can differ, for example when a local
  run of the same ID precedes the pull. A row without a handoff is placed only
  when the scheduler ran that ID once. A saved `summary.json` without the
  timeline `sequences` or the joins' `producer_handoff_sequence` is rebuilt
  from the captures.

  The detail lane draws the decode node's phases: waiting for KV (load prepared
  to KV received, on the scheduler's clock), to the first token, and decode
  (first token to finish, on the frontend's clock, `api.source`). Each point is
  projected onto the reference clock; one that does not project is not drawn,
  and the plotted window covers the phases and the transfers. A missing,
  repeated or ambiguous timeline, join or value leaves that value unavailable,
  with its own reason. The run header counts, per field, the displayed rows
  that carry it (a measured zero counts) and lists a field no displayed row
  carries as unavailable.

## UCX protocol selection

With `UCX_PROTO_INFO=y`, UCX logs a table the first time it selects protocols
for an endpoint configuration and an operation: for each message-size range,
the protocol (for example `short`, `zero-copy`, or `software emulation` for a
remote read over a transport without one) and the lanes it uses, with the share
of the bytes each lane carries. `UCX_PROTO_INFO=used` logs the same tables with
per-range selection counts when a worker is destroyed. Set `UCX_LOG_FILE=stderr`
as well: stderr is unbuffered, so each table is written whole as it is
selected. On a block-buffered stdout redirected to a file, tables can be cut by
other processes' output or lost when a process exits.

```console
.venv/bin/python -m vllm.kv_profiling.ucx_protocols server.log \
  [--profiles /profiles] [--run-id RUN] [--output ucx.json]
```

The viewer shows a run's tables when their output is saved under the run directory's
`ucx-protocols/` (the e2e harness writes `ucx-protocols/<role>/<node>.json`).

The parser groups each table by its log prefix (time, host, PID and UCX thread,
kept as printed, e.g. `a` for UCX's async thread), so several processes can share
one log. A table is recognized by its frame, whatever its title says, so a
damaged table is counted and loses only itself: the next one is still parsed.
It reports the tables per process, UCX context, locality (`self`, `intra-node`
or `inter-node`) and endpoint configuration (`cfg#N`), the fields UCX prints.
UCX numbers `cfg#N` per UCP worker and does not print the worker, so worker
identity stays unresolved: a grouping may hold several workers' tables.
`worker_collision_detected` marks a grouping where that is visible (one label
with two localities, or one operation selected twice); its absence does not
mean one worker.
Each range has its `start` and `end` in bytes (`null` for unbounded), the
protocol `description` (`estimated` when UCX marks it `(?)`), the raw `config`,
and the parsed `lanes` (`transport`, `device`, `path`, `share_percent`).
`--profiles` joins each PID to the recorder that process ran (role, rank,
engine), restricted to one profiler run with `--run-id`. A PID that matches
more than one recorder (a PID reused across runs) stays unjoined, with
`recorder_candidates` counting the matches. The log and the profiles must come
from the same node. Unreadable or malformed recorder metadata is skipped and
listed (`recorders_skipped`), never fatal. `UCX_PROTO_INFO=used` counts are printed as workers are
destroyed, so parse a log copied after the servers stopped.

UCX names no peer in these tables. Endpoints to peers reached the same way
share one configuration, so a configuration stands for every such peer. The
tables reflect what UCX selected, not how many bytes each protocol moved. A
table that does not parse is counted under `unparsed`, with its reason, never
dropped silently.

## Qualification boundaries

The focused core tests use real recorder mappings, native publishers, helper
processes, files and loopback sockets. They cover publication, loss,
contention, wrap, retirement, failure/recovery and clock/export semantics.

They do not validate connector attachment, forwarding through vectorcall and
ordinary calls, native stats serialization under an observer, data transfer,
or overhead: the connector and harness tests cover attachment and transfers, and
overhead is covered only by the serving A/B above, for its tested workload. No
connector behavior or pairing is changed here.

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

Each `transfer_bind` also carries the **post pass**: a per-process counter of the worker's
`start_load_kv` calls (one call per engine step, `load_pass`, which records nothing), in the
bind's first value word, which that kind leaves otherwise unset (bytes belong to the completion).
The sink decodes it as `post_pass` and leaves `bytes` unset on that row; the exporter shows it
as `post_pass`, never as bytes; the capture's `capabilities.post_pass` advertises it. It is what
makes the **`post_groups`** view possible: no existing record marks a pass (ring adjacency merges
adjacent passes and splits one by records from inside it), so a capture without it has no groups
(`post_group unavailable` in `not_available`) and no heuristic is applied. A `post_group` is the
independent reads one worker posted in one pass, keyed by the capture and the pass; it is a derived
view over the reads, not a transport call and not an additive duration (the reads overlap). It
lists the prepared attempts with their outcomes (a failed post or an unused mixed-memory leg has
no accepted post), the sums of bytes and descriptors (whole only when every read completed), the
posting span (first prepared to last accepted post) and the completion span (**first accepted post
to last observed completion**), and the sums of the reads' own NIXL microseconds labelled as sums.
It is `exact` only if the capture shows no loss and its last pass is not unfinished (a pass
counter alone does not prove the last pass finished) and every prepared read has a known fate;
otherwise it is `observed_only`: whole-pass totals and spans are withheld and what was seen is
reported under `*_observed` names.

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

With data parallelism vLLM runs an API server process for each DP rank (its
`--api-server-count` default), each with its own frontend recorder. At
`--shutdown-timeout 0` the serving process force-kills them right after SIGTERM,
before they close. With `flush="end"` their recorders keep only their metadata
row, which records the timeout in effect (`shutdown_timeout_s`,
`close_at_shutdown`); with `flush="background"` the rows already written survive,
and the capture is still incomplete. Every
frontend records its process name as `api_server`: `MainProcess` with one API
server, `ApiServer_<i>` with several, `APIServer_DPRank_<r>` for a DP rank's own
server. Give such deployments a positive `--shutdown-timeout`, above
the recorders' close deadlines; the e2e harness passes the profiler's shutdown
grace for every profiled server with data parallelism, and fails a run unless
each server node has exactly the frontends of its launch's API server slots, a
scheduler per DP rank and a worker per (DP rank, TP rank) on its rank's engine,
all of the side's KV role, each whole and closed (the observer package's
`--check-core <profiles directory> --run-id <UUID>`: stream, terminal row and
manifest; it exits 2 when one is not).

## Endpoint observations at L3

L3 includes the L1 connector instrumentation (NIXL pull and the Mooncake
adapter below) and declares an external observer. It is off by default. Transport
metrics come from the separate L2 collector (below); eBPF is an opt-in collector
of this observer (below), validated live; its overhead is described above. Sampling, JSON formatting
and kernel queries run in a separate endpoint process, never in a transfer
callback. The native L1 ring ABI and transfer metrics are unchanged.

Use `network_profiler.level: l3`, the same run UUID as L1, and a `probes` object.
Interface names and RDMA `{device, port}` entries are explicit endpoint
allowlists. An unknown key is refused.

| Key | Default | Accepted |
| --- | --- | --- |
| `interval_s`, `slow_interval_s` | 1, 5 | seconds, 0.1..3600 |
| `tcp` | `{}` (off) | `owned_sockets: true`, optional `local_ports`, `peer_ports`, `exclude_ports` (each at most 64 ports, 1..65535) |
| `discovery_interval_s` | 0.25 | seconds, 0.05..10 |
| `interfaces` | `[]` | at most eight names of `[A-Za-z0-9_.-]` (not `.` or `..`) |
| `driver_statistics` | false | needs `interfaces` |
| `rdma_ports` | `[]` | at most eight `{device, port}`, port 1..255 |
| `rdma_required_families` | `["counters"]` | a non-empty subset of `counters`, `hw_counters` |
| `qp` | false | needs `rdma_ports` |
| `threads`, `interrupts` | false | booleans |
| `max_threads` | 256 | 1..1024 |
| `max_sockets` | 1024 | 1..1024 (it can only be lowered) |
| `queue_rows` | 512 | 1..512 (it can only be lowered) |
| `output_bytes` | 256 MiB | 4096 bytes..1 GiB |
| `ebpf` | `{}` (off) | see the eBPF section; disabled, only `enabled: false` |
| `required` | `[]` | enabled sources among `ebpf`, `tcp`, `netdev`, `driver`, `rdma`, `qp`, `threads`, `interrupts` |

TCP sampling (`tcp: {"owned_sockets": true}`; `tcp: {}` is off) takes the
`TCP_INFO` of the sockets the endpoint's connector processes own, never an
address: there are no address selectors, and optional `local_ports`,
`peer_ports` and `exclude_ports` only narrow it. A socket is identified by its
network namespace and kernel cookie, unique per boot and never reused, so a
frontend reusing a worker's address tuple is another socket. Every
`discovery_interval_s` (whenever TCP sampling or eBPF is enabled) the observer
reads the fd tables of the inventory's processes that host a scheduler or
worker, once per process whatever number of recorders it hosts, each recorder
identity verified on its own against its frozen start ticks and namespaces;
takes one bounded `sock_diag` dump of the namespace's TCP sockets (with
`TCP_INFO`, which sampling reads from); and reads those fd tables again. A
socket is owned when a process held it in both reads with the same verified
scheduler or worker identity in both. A frontend process owns none, so its HTTP connections are never
selected, even between the same addresses; a process that hosts a frontend as
well as a scheduler or worker owns its frontend's sockets too, which cannot be
told apart (`frontend_colocated` in the record). Other processes holding a
socket are recorded, read last on a best-effort budget; a socket held by more
than one process is `shared`. Owned sockets span every interface of the
namespace, loopback included (`loopback` in each record): the selection is by
owner, never by interface. Each change is a `socket_ownership` record:
`admitted` (an admission ID, the socket, its owners' recorder identities and
its other holders, as lists that split like any long row), `removed` (gone,
its owner unreadable, in other namespaces, or exited), `reversioned` (its
verified owner identities, in full, or its holders changed, so a new process
generation under the same PID is another owner: a new admission with
`supersedes` takes over, and the old record stays as it was for late events)
and `suspended`. An admission never changes once
recorded. A scan whose owner reads, dump or record construction did not finish
within its bounds admits nothing and suspends every admission until a complete
scan confirms it again (under a new admission). An owner confirmed to have
exited (gone, a zombie, or another process under its PID), before or during an
fd read, is retired, as in an orderly shutdown. One with its frozen start ticks
but other namespaces, for now, holds nothing until they match again (it is not
retired); it, one that is alive but unreadable, and an incomplete scan fail the
TCP source. Every scan is bounded (/proc entries and processes, fds per
process, all fds, sockets, and time) and reported (`discovery`), and discovery
runs on a fixed schedule whose skipped periods are counted. Each sampled socket
row is one observation, timed by its dump's own interval, and carries
`selection`: its cookie, admission (whose record names the owners), owner roles
and count. A socket that opens and closes between two scans is never observed. No intermediate-network observation or packet modification occurs.
`required` names enabled sources (`ebpf`, `tcp`, `netdev`, `driver`, `rdma`, `qp`,
`threads`, `interrupts`). The kernel reports request sockets (`SYN_RECV`,
timer 1) and time-wait sockets (`FIN_WAIT2` or `TIME_WAIT` after close, timer 3)
as compact records: inode 0 and no `TCP_INFO` (at most `INET_DIAG_MARK`, for a
reader with `CAP_NET_ADMIN`). They
are recorded as `record: request` or `time_wait`, with `tcp_info_expected:
false`. A required TCP source fails only when a full socket lacks `TCP_INFO`.
Declare requirements before a capture: unsupported optional sources stay
visible as unavailable; failures of required sources invalidate that capture.

`threads: true` samples every thread of the endpoint's serving processes (each
process in the recorder inventory, and the owner) from `/proc/<pid>/task/*`: the
thread name, the CPU it last ran on, user and system CPU ticks, the scheduler's
time on CPU and time runnable but waiting for a CPU (`schedstat`), timeslices,
and voluntary and involuntary context switches, all cumulative kernel counters
(rates are the reader's). Thread IDs match the `tid` of L1 records, so a transfer
can be read beside the CPU and scheduling of the thread that posted it. A kernel
without schedstats reports those fields as `null`, never zero, and names the
file under the thread's `unavailable`. Each process and thread is read through
its own `/proc` directory, so one sample never mixes two processes or two
threads that reused an ID. A serving process that exited (or whose PID now
belongs to another process) is reported as `exited`, not as a failure; threads
that exit mid-sample are counted in `vanished_between_reads`. An empty task
listing is confirmed through the pinned process: an exit is `exited`. Threads
that cannot be read (permissions) are counted in `unreadable`, and thread files
that exist but cannot be read in `unreadable_files`; both are source failures
(`coverage_problem`), while the threads that were read are kept. At most
`max_threads` threads per process are read; `truncated` says when more exist.

`interrupts: true` samples, per CPU ID (as the file headers name them, so
offline CPUs do not shift counts), the interrupts of the configured interfaces
and RDMA devices (their MSI interrupts listed in sysfs, read from
`/proc/interrupts`, busiest CPU first) and the host's `NET_RX` and `NET_TX`
softirqs (`/proc/softirqs`). Both files are host-wide, not per namespace. Some
container runtimes mask `/proc/interrupts` (an empty file): the interrupts are
then read from `/sys/kernel/irq/<irq>/per_cpu_count` (one count per CPU ID) and
`source` says which was read; an empty `/proc/softirqs`, or a masked
`/proc/interrupts` without `/sys/kernel/irq`, is a source failure that says so. An
interface with no MSI list, or an empty one (virtual or not PCI), is listed as
unavailable. Missing coverage is a source failure, beside the counters that were
read: a configured device that does not exist (`absent`), an interrupt sysfs
lists that `/proc/interrupts` lacks (at most 32 in `missing`, with the exact
`missing_count` and `missing_truncated`), an MSI list that exists but cannot be
read (`unreadable`), a missing network softirq, or a row that does not match its
CPU header.

An observation too large for one probe row (16 KiB) is split into several rows:
the items of its lists, taken as one sequence, are halved as often as needed.
Each row's `slices` gives, per list, the `[start, end, total]` range it carries
(possibly empty); every item is in exactly one row.

For example, the probe settings for a TCP endpoint might be:

```json
{
  "interval_s": 1,
  "slow_interval_s": 5,
  "tcp": {"owned_sockets": true},
  "interfaces": ["eth0"],
  "required": ["tcp", "netdev"]
}
```

Replace the interface with the actual endpoint's. The owned sockets are all of
the scheduler's and workers': the KV transport, its side channels and, on a
Mooncake prefill engine's TP0 worker, its bootstrap server and clock.

### Deployment and lifecycle

Locate the installed `vllm/kv_profiling/probe.py` with `importlib.metadata` or
the installation manifest. Run it by absolute file path to avoid importing
vLLM/Torch in the observer:

```console
.venv/bin/python /installed/vllm/kv_profiling/probe.py --config config.json
```

The config (the harness and the sidecar write it as `config.json` in the
observer's directory, which `--check-capture` reads back) has the following
runtime fields in addition to `probes`. There is one launch form: the launcher
names how many recorders of each role the endpoint has, and the observer finds
them and the process to follow itself. A config that supplies `inventory`,
`owner_pid` or `owner_start` is refused, ending the observer with an error
(exit 1). Once started, it exits 0 for a complete capture and 2 for an
incomplete one.

| Field | Meaning |
|---|---|
| `run_id`, `endpoint`, `role` | Shared run UUID, endpoint ID, `prefill` or `decode` |
| `expected` | Recorders per role, `{"frontend": n, "scheduler": n, "worker": n}`: one of each per DP rank and TP rank as the server runs them, at least one frontend |
| `cpus` | Nonempty observer CPU list: within the observer's own affinity, disjoint from the serving CPUs |
| `irq_policy` | `operator_verified` or `virtual_not_applicable` |
| `input_dir`, `output_dir` | Read-only L1 root and separate probe output (not inside the input) |
| `stop_file`, `endpoint_complete_file` | Stop and child-closure paths |
| `owner_exit_grace_s` | Optional: how long after the owner's exit the controller's completion notice is awaited, (0, 600] seconds, default 2 |
| `package_sha256` | Optional: the SHA-256 of the observer package (`observer.pyz`) run; a mismatch refuses to start, and `--check-capture` then checks the package |

Share the endpoint network and PID namespaces. Check physical IRQ affinities
externally; the observer never retunes them. The observer discovers its
endpoint from the L1 recorders' opening metadata in `input_dir`, without
transfer-path I/O:

- the recorders must be exactly the `expected` ones, each carrying a KV role of
  the configured side (`kv_producer` for prefill, `kv_consumer` for decode,
  `kv_both` for either);
- the owner is the single frontend's process, or, with several API servers, their
  one common parent, each read from the observer's own `/proc`.

It freezes the inventory, the owner, the evidence for the owner (each frontend's
parent as read) and the digest of the observer code that ran (`code_sha256`) in
its capture before it is ready. `--check-capture` checks the owner against that
evidence and the code against its own; `--code-digest` prints the digest of the
observer code it runs, as captures record it. PID start ticks, namespace identities,
engine and rank are validated before optional fd ownership reads. If the kernel
refuses a process's namespace links (it holds capabilities the observer lacks, or
is not dumpable), the error names the path, the errno and both causes.

Start the observer before admitting workload and wait for its manifest's `ready`.

`--sidecar SETTINGS --run-id <UUID> --endpoint <ID>` runs the launcher itself
beside the server: a container of the server's pod, or any process that shares
its namespaces. `SETTINGS` holds `probes`, `role`, `expected`, `profiles_dir`,
`observer_root`, `irq_policy` and, optionally, `cpus` (the sidecar's own affinity
by default), `startup_timeout_s` (default 900) and `stop_grace_s` (default 25). Each start is a fresh
attempt in its own directory under `observer_root`, so markers from an earlier
attempt never apply. The sidecar:

1. **Startup, within one deadline (`startup_timeout_s`):** it finds the serving
   generation whose expected recorders are all running, without holding the
   server back. That is `profiles_dir` itself, or one directory per serving
   start beneath it, earlier ones left as they are. It then starts the observer
   from the same entry point, and writes `observer_root/ready` (the attempt's
   path) once the observer is ready, for a readiness probe. An observer that
   fails, or is not ready by the deadline, makes the sidecar exit nonzero.
2. **Shutdown, on SIGTERM or the server's exit:** it removes `ready` at once.
   Then, within `stop_grace_s`, which it also gives the observer as its
   owner-exit wait (`owner_exit_grace_s`), it waits for every frozen serving
   process to exit and its L1 capture to close. The processes include the owner
   (with several API servers, their parent), and a process whose status cannot
   be read is never taken as exited. A quarter of the budget, at most 5 s, is
   kept for stopping the observer. It writes the completion file (`children_closed` only if both
   held), creates the stop file, reaps the observer by the deadline or kills it
   (`timed_out`), and publishes its exit record. Termination during startup
   cancels the attempt; it is not a timeout.

The attempt directory is then checked with `--check-capture <attempt> --core-dir
<generation>` (its `config.json` names the generation); the check exits 0 for a
complete capture and 2 otherwise. `deploy/llm-d/profiling` runs it as a Kubernetes native
sidecar.
After child/L1 completion, the controller writes the endpoint completion file:
`{"run_id": "<UUID>", "endpoint": "<ID>", "children_closed": true}`, then
creates `stop_file`. The observer independently checks the L1 manifests, samples
once more and flushes. Unconfirmed owner exit, signal termination, incomplete
child captures,
required-source failures, queue loss and missed ticks cannot yield a complete
capture. Owner exit is not proof of worker/helper closure: the observer allows a
bounded controller notification interval (`owner_exit_grace_s`, 2 s by default),
then marks a timeout incomplete. The controller must
also collect observer exit/timeout status; kill only its own observer after the
shutdown deadline. Kernel driver I/O can outlast user-space deadlines. Observer
failure never signals or stops the server.

The harness's `profiler.level: l3` additionally requires per-node `probes` and
`observer_cpus` mappings. It launches endpoint-only observer containers with
shared network/PID namespaces, a read-only profile mount and separate output,
then collects manifests and logs. `observer.stdout.log` and
`observer.stderr.log` preserve each Docker log stream; `observer.log` labels them
without implying an order across streams; the launcher's own evidence (the
wait result, the inspected container, its removal, log collection) goes into the
exit record's `evidence`.
The generic CLI contains all probe logic.

Whatever launched the observer and reaped it publishes its exit record,
`exit.json`, in the observer's directory: with `--record-exit DIRECTORY` and the
fields as JSON on stdin, or the same function from the package. The fields are
`schema` (1), `run_id`, `endpoint`, `process_id` (the observer capture's own, so
a record from an earlier attempt never certifies a later one; `null` when no
capture exists), `exit_code`, `signal`, `killed` (`null`, `oom` or `signal`),
`timed_out`, `launcher`, `problems` (what the launcher could not reconcile, each
failing the check) and `evidence` (launcher detail, never checked). It holds no
timestamp. `schema` may be omitted on stdin (it is written); a mistyped or
missing field is refused before anything is written. For example:

```console
echo '{"run_id": "<UUID>", "endpoint": "decode0", "process_id": "<capture process ID>",
       "exit_code": 0, "launcher": "my-launcher"}' |
  python observer.pyz --record-exit <observer directory>
```

`signal`, `killed`, `timed_out`, `problems` and `evidence` default to a clean exit
(`null`, `null`, `false`, `[]`, `{}`); `evidence: null` is written as `{}` as well,
and the exit checks never read it. `--check-capture` requires a well-formed record of a clean exit of
this attempt: exit 0, no signal, not killed, not timed out, no problems. It
never reads a container runtime's own records. A capture carries
`capture_format: 2` from its creation; an earlier capture is refused, to be
rerun.
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
On real nodes the RDMA port's data counters were checked against the KV bytes the
connectors moved over RDMA, and stayed near zero for KV moved over TCP.

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
helpers. Off-vs-L3 serving overhead was measured only for the workload described
at the top of this page, with `flush="end"`, where its p99 latency bounds exceeded
their limits.

### Capture inventory and deployment snapshot

The observer freezes the identities of **every** expected frontend, scheduler
and worker capture on that endpoint when it starts, and refuses any other set.
`--inventory <profiles-root> --run-id <UUID>` prints what it would find. The
harness expects one frontend per local DP rank, one scheduler per local engine
and one worker per local rank. Unknown, extra, missing or identity-mismatched
captures cannot close successfully. Both the native event stream's terminal/loss state and its
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

### Opt-in TCP eBPF events

The stdlib observer accepts `probes.ebpf` with `enabled: true`,
`privileged_opt_in: true`, `mode: tcp_tracepoints`, and an absolute `artifacts`
directory. Without `privileged_opt_in` (or without the effective privilege) the
source is unavailable at runtime (`missing_explicit_effective_privilege`), not a
configuration error. It records the events of the sockets the endpoint's scheduler and
worker processes own, the same owned sockets TCP sampling selects (discovery,
by kernel cookie), never by address: there are no address lists. Optional
`local_ports` and `peer_ports` lists (at most 32 unique ports each) narrow it.
In the kernel, after the network namespace, protocol, family and ports, each
event's socket cookie (`bpf_get_socket_cookie`, assigned on first use, the
value `sock_diag` reports) is looked up in an `allowed` map of the admissions in
force; only an owned socket's event reaches the ring, carrying the admission it
matched. The observer reconciles that map to discovery's latest decision for
each cookie over the collector's stdin, one command per cookie in flight:
sequenced commands (`A`dmit a cookie under an admission, replacing any entry;
`D`elete it only if still under that admission, so a stale delete never
removes a newer one; `R`ead it back), each acknowledged with when it was
applied. The collector's output is read on a thread of its own, so discovery
scans and sampling never pause it: the thread only reads, queueing complete
rows in order (at most 16 MiB and 65,536 rows per collector) and writing the
diagnostics to `native.stderr`. When the queue is full it stops reading, so the
collector's queue and the ring absorb a backlog, and what they cannot is
counted there as eBPF loss. The observer's main thread handles the queued rows
in order, in batches of 2,048, so its other work still runs under load; every
decision (acknowledgements, ownership, verification) stays on it. It takes no
more rows than its capture writer can accept at that moment, leaving a quarter
of the writer's queue to the other sources: a burst waits in the reader's queue
until the writer has written what came before, and is never dropped by the
writer. A command is unacknowledged only once the observer has handled every row
the collector wrote before some time past its second (the reader takes the time,
then the pipe's fill, so the stream position holding everything written by then
is known even when every read is a full one): a reply waiting in the queue is
never declared lost. An unacknowledged command, a failed update or delete, or an entry
that could not be read makes the cookie's kernel state unknown from the
command until a readback establishes it; the readback is followed by the
command that still moves the entry to the decision (a retired cookie's stale
entry is deleted again), within a bound of four attempts. Discovery's
decisions take effect at the time of the scan that made them (an admission's
`admitted_ns`, a removal's or suspension's `ended_ns`). Each update is
atomic per element, and each event is judged on its own cookie. An event is
`verified` when, at its time, discovery had decided its admission, the kernel
held it as acknowledged, no change of the cookie was still unresolved, and its
kernel state was not unknown; a late event of a removed admission, timed
within its interval, stays verified, and an event after a cookie's retirement
never is. A cookie whose entry does not match the decision at readiness fails
the eBPF source; one at the end (a command unsent, unacknowledged or given up)
makes the collector's run unclean (`ownership_unresolved`); a cookie given
up is judged by event time, through its unknown intervals, so its events from
before its retirement stay verified. Every eligible event the ownership filter
rejects is counted once in an exact total (`excluded_unowned`), and best
effort per cookie in an LRU map whose entry keeps the count and the earliest
event time (an atomic minimum, marked unknown under contention, set before the
count). Each admit, after its update, and each delete that removes the entry,
before it, harvests the cookie's entry (read and deleted at once); the dump at
the end reads each entry until two reads agree (`consistent`, since hooks may
still be running) and is `counters_complete` when the whole map was read. The
per-cookie counts are lower bounds: a hook can finish updating an entry after
it was harvested or evicted, an increment the total keeps and no cookie
shows (unattributed). A harvest holds rejections that began while the cookie
was out of the map: `before_or_between_admissions` (the discovery gap, an
event in flight across an admit included). What remains at the end is
`after_retirement` only if the cookie's last harvest was a confirmed delete,
the dump's read of it was consistent, and its earliest event began after both
that delete began and discovery's latest retirement of the cookie (an
admission decided meanwhile, installed or not, was ownership); anything the
evidence cannot place (kernel state unknown, a failed harvest, an
inconsistent read, an earliest time unknown or too early) is `unresolved`. Cookies never admitted are `never_owned`, and the
rest (evictions, `rejected_untracked`) unattributed. A rejected event's content is not recoverable. Off is the default: there is no native
child, capability check, BPF syscall or automatic privileged launch. Off identities
and old stdlib captures remain unchanged. Uprobe `sets` and `kernel_audit` options
are removed; old enabled configurations need an explicit migration.

Four `tp_btf` hooks emit timestamped rare events: `tcp_retransmit_skb`,
`tcp_send_reset`, `tcp_receive_reset`, and `inet_sock_set_state` (old/new state).
There is no `tcp_probe`, per-segment hook, native NIXL post or provider WR probe.
The events complement the sampled TCP_INFO gauges instead of repeating byte or
segment counters. RDMA remains out-of-process hardware/port observation.

Each record has kernel CLOCK_MONOTONIC time (`bpf_ktime_get_ns`), network namespace,
family, local/peer address and port, its cookie and admission, and CO-RE snapshots of
`srtt_us`, `snd_cwnd`, `snd_ssthresh`, `packets_out`, `retrans_out`, `lost_out`,
`total_retrans`, and `mss_cache`. Missing fields are unavailable, never zero;
failed reads are distinct. `srtt_us` retains the kernel's raw usec-shift-3 encoding.
The fields are concurrent reads, not an atomic snapshot. Live time-namespace
conversion is explicit; an unknown conversion leaves raw time without a join.
There is no implied cross-host clock alignment. The channel tolerates missing
optional clock metadata, but the enclosing observer still requires its own
time-namespace metadata at startup.

Only endpoint-network-namespace matches (and optional port matches) reserve
records. Hooks still execute the early filter on unrelated events globally.
Null, TIME_WAIT and request sockets are excluded before full-socket reads and
reported as global unknown exclusions, not endpoint losses. Ordinary namespace,
protocol, family and port rejections are filter exclusions. Reservation/read/output
losses have separate counters and scope. In the offline join to L1 a
verified event is its admission's: the capture's admission record names the
owners, so it is a candidate of those owners' READs only (with the usual clock
and READ-window checks), with or without TCP samples, when the event, its
selection and the admission's record name one cookie and one admission; an
unverified or inconsistent event is no one's. Only events of the schema
before owned-socket selection (2) fall back to tuple/cookie inventory
candidates, which offer possible owners without proving ownership at the
event. Ambiguous or absent evidence stays explicit. Aggregate bytes are never
apportioned and unique candidates are never causal attribution.

Build the artifacts separately, inside the eBPF tooling image
(`tools/kv-profiler-ebpf-image`, which provides the SDK under
`/opt/kv-profiler-tools`; `KVP_TOOLS_PREFIX` points elsewhere), from a source
checkout into a new directory that must not exist yet:

```console
bash tools/kv-profiler-ebpf/build.sh <source root> <new artifact directory>
python tools/kv-profiler-ebpf/check-unprivileged.py <artifact directory> <source root>
```

The directory receives the collector, the BPF object `observe.bpf.o`, its
`limits.json` and `manifest.json` (the files' hashes, which the observer checks),
with inspection records beside them (`build-inputs.json`,
`object-inspection.jsonl`, `observe.skel.h`, `sections.txt`, `disassembly.txt`).
The unprivileged check (run without BPF privileges, with the `vllm` package
importable) verifies the package, its hooks and maps and instruction bounds, and
that an unprivileged arm is refused before anything loads. Mount the
finished directory read-only.

The observer runs the collector as `collector --arm <plan> <observe.bpf.o>`
(`collector --inspect <observe.bpf.o>` only lists the object's maps and
programs). The plan file names the observer process, its network namespace and
the port lists; the ownership commands go over the collector's stdin, one per
line (`A <seq> <cookie> <admission>`, `D <seq> <cookie> <admission>`,
`R <seq> <cookie>`, and `stop`). It reads commands only while its output queue has
room for every acknowledgement they can produce, and its periodic counter report
waits until all its rows fit (the counters are cumulative), so a slow reader of
its output holds them back rather than failing one. Exit 2 has two meanings, told
apart by the rows:
a run that started (a `ready` row) and ended normally, its coverage partial as
always, with its `end` row; and a refusal to start (an invalid plan, missing
privilege or a dead parent, an object that does not open or load), with a
`status` row saying which. Exit 3 is an output or collector failure, a malformed
row included. The observer itself exits 0 for a complete capture and 2 for an
incomplete one once started; a launch config it refuses ends it with an error
(exit 1). One CO-RE BPF
object targets BTF kernels from 5.15 upward, consuming positional argument prefixes
and ignoring appended arguments. This is the compatibility contract, not a claim
that an untested kernel has passed validation. The tooling image and dependency
pins are BUILD reproducibility only. Runtime verifies files against their own
manifest, never an image, kernel release, BTF digest or allocation-audit identity.
Missing privilege/BTF/hook, kernel verifier/load/attach refusal is unavailable with
its diagnostic; there is no fallback, automatic capability grant or policy change.
All four hooks must attach for this explicit mode. Required-source behavior remains
`required: [ebpf]`; optional unavailability leaves other observations unchanged.

The observer waits, bounded (10 s), for the collector's first answer (ready, or a
status that it is unavailable) and marks its capture `ready` in its manifest, the
signal a controller waits for, after that answer or when the bound expires. A
collector still starting at the bound is `not_ready`: a failure when eBPF is
required, even if it answers later. The endpoint capture can close cleanly with
partial eBPF coverage. Only owned sockets' events are captured; an observer
whose inventory has no scheduler or worker owns none, and a discovery gap is a
failure of a required eBPF source as of a TCP one.
Coverage is never provably complete, so a
required eBPF source is judged on a clean run (`clean` in the stop record): ready
and still available, no failure, the collector's own unfailed terminal record, a
normal exit and its final resource accounting. Known event loss (endpoint ring or
read losses, counter wraps, the collector's output drops) marks the capture
`ebpf_degraded`, with the counts in `ebpf_event_loss`: it stands, but its events
cannot certify agreement with other sources. A malformed row is no loss but a
collector bug: the collector exits with an error (status 3), so the run is not
clean and a required eBPF source fails; the count stays in its terminal record.
A row the observer cannot frame or decode is the same: the rows read before it
are still handled, and the error stands to the end of the run, which is not
clean. The
sustained event rate the observer keeps up with depends on its own work per row
on its main thread (decoding the collector's row, annotating it and encoding it
for the capture), on the rest of its work, and on the capture's filesystem:
above it, the backlog fills the reader's queue, then the collector's and the
ring, and the excess is counted as eBPF loss (the source degraded) rather than
dropped by the writer, whose drops (`drop_probe_rows`) would leave the whole
capture incomplete. A
writer that has failed holds nothing back: its rows are dropped and counted, its
capture already incomplete. Namespace,
protocol, family and port exclusions, ownership rejections (counted apart) and global
unknown exclusions are not loss.

In the e2e harness an endpoint opts in with `probes.ebpf: {enabled: true,
privileged_opt_in: true, mode: tcp_tracepoints, artifacts: <built directory>}`,
both flags literal booleans. Its observer container then adds `CAP_BPF` and
`CAP_PERFMON` (never `CAP_SYS_ADMIN`) and mounts the artifact directory
read-only at the same absolute path. The run is refused before anything starts if that directory is not a complete built package: its manifest, collector, BPF object and limits, each matching its manifest hash, and an executable collector. A package of another schema (an earlier build, whose collector reads another plan) is refused with a request to rebuild it with `tools/kv-profiler-ebpf/build.sh`; the observer checks the same before launching it. Without the
opt-in the observer adds neither. Live attachment has been validated on Linux 5.15
with Docker's default seccomp profile. With those two capabilities, all four hooks
attached. Without them, the missing effective privileges were reported before
loading BPF; kernel load or attach failures carry their diagnostic. The final eBPF
retransmitted-segment counter matched the final TCP_INFO sample on each
retransmitting socket. TCP_INFO was polled intermittently. Reset totals agreed
between the sending and receiving endpoints. L3 polled the current socket state,
while eBPF recorded individual transitions.

The ring has fixed 1 MiB data capacity, plus two one-entry ARRAY values: 512 bytes
of control (the namespace and port lists) and 256 bytes of loss counters. That is
1,049,344 bytes of ring/value capacity. The ownership map (`allowed`, a hash of at
most 65,536 cookies, 8-byte keys and values) and the rejection counters
(`rejected`, an LRU hash of at most 4,096 cookies, 8-byte keys, 24-byte values)
are additional. None of this is a
physical kernel memory charge. Kernel bookkeeping and JIT memory are additional.
Map fdinfo is diagnostic only; missing or large reported values are not a gate.
No per-host qualification or per-kernel audit is required. Four programs/links,
256-byte binary events, 2 KiB native JSON records and a 4,096-row native output
queue are bounded. Ring reservation never waits; full ring/output queues count
drops and mark coverage partial. Storms can lose events; rarity is not guaranteed.
The existing common envelope keeps separate 16 KiB/512-row writer limits.

Shutdown disables emission, detaches, drains the ring and both pipes within the
total deadline, then closes owned FDs. Nothing is pinned. Parent-death cleanup
and owned-child `wait4` resource accounting are retained. Abnormal exit, missing
failure evidence and incomplete output cannot pass as an optional capture.
Detachment and an empty userspace queue alone do not establish that every kernel
producer finished: terminal coverage remains partial with quiescence/miss visibility
unverified. Unprivileged fixtures and object inspection do not establish verifier
acceptance, live kernel cleanup or overhead. Those require separate validation;
no per-deployment kernel identity gate is introduced.

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

```text
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
request-scoped `failed`. A member row carries its **share** of the batch as the connector built it
(`bytes` and `selected_descriptors` of the row, read as `member_bytes` and `member_descriptors`): the
build keeps each request's descriptors as a contiguous slice of the flat lists (coalescing never
crosses requests), so the shares partition the batch's measured totals exactly. The share is the
connector's posted descriptors after its own coalescing, not the engine's: TENT's merging of
adjacent descriptors inside the engine across members is counted at transport level and never
attributed to a member; a member has no duration of its own (one call). A share is *posted* when
the batch's write succeeded, *planned* when it failed (kept, never confirmed), and **absent, never
zero,** when the build had not finished (an exception or cancellation before it); a rejected
member, a member with no blocks and a replica's members are explicit zeros. The shares are only
recorded for an observed batch and cost one boundary note per request and one slice sum per member
there; nothing for an unobserved one.

The summary's **`write_batches`** view (one entry per batch: one transport call) lists the members
and checks the partition: `exact` only if the declared member count equals the distinct recorded
identities (a missing zero-byte member is detected, not only a wrong sum), every share is known and
the sums equal the posted totals (and the completion's); `mismatch`, `member_mismatch`,
`shares_unavailable` and `not_posted` say otherwise. `observed_only` means the sums match
but cannot certify the partition: the capture has loss (a member row may be missing) or the
adapter's own record (`profiler.json`) reports a problem, each named in `partition.reasons`;
such a batch is never `exact`.
A request's entry lists its batch memberships (associations, never request occurrences) with
posted, planned-unconfirmed and unknown shares kept apart.

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
heal an earlier peer's failure), and `failed(receive_failed)`. That completion names
the request and its transfer ID, never a generation (the write's generation is the
producer's): the summary keeps it in the request's row, not as a transfer event,
with every such receipt in `receive_completions` (transfer ID and time; a reused
request ID keeps each) and `receive_completed_time_ns`/`receive_transfer_id` when
there is exactly one. The rank moves no transfer of its own, so its receive
completion is the request row's proof of movement (`completed`), under the same
failure rules: a required failure before it keeps the request failed. It is
recognized only where the capture declares `receive_completion` (in its metadata, or
else its context); elsewhere a worker completion without a generation is still an
orphan. A recovered
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

Clocks: nothing is configured. Each producer engine's clock (with data parallelism
each DP rank is an engine of its own, vLLM's `<engine ID>_dp<rank>`) is served by its
TP0/PP0 worker, on the host the worker serves KV from (the address it registers) and an
ephemeral port, and advertised to the instance's bootstrap server under one more route,
`/kv_profiler/clocks`: its run, engine ID, registration DP rank, endpoint and the worker
recorder's `clock_id`. The embedded bootstrap server gets the route when the profiler is
on; the bootstrap server's own `/register` and `/query` are unchanged. The worker retries
in the background (the registry may start later) for `clock_discovery_s`, and
`profiler.json` reports `clock.server` and `clock.advertised`: `ok`, or why not (an
external or older registry without the route answers HTTP 404; only a recognized
acknowledgement counts). The clocks themselves order registrations, never a timestamp:
the registry accepts one only if its endpoint answers as its engine with its generation,
another generation of the same engine replaces the stored one only if the stored clock
no longer answers, and two live clocks of one engine are a conflict. A late retry of a
retired worker therefore never displaces its successor; an identical retry changes
nothing.

The consumer's scheduler looks each new remote engine up at the `remote_bootstrap_addr`
of the request that names it, on its clock thread (never in a hook), with its lookups
and exchanges on a few I/O workers: one lookup is in flight per bootstrap server and
serves all its engines, bounded by the time left in their windows, so a slow registry
delays no other one. A missing engine is looked up again with backoff until
`clock_discovery_s` ends; an exchange that fails within it (a moved endpoint, another
generation) refreshes the listing and retries. The first and last exchanges require the
advertised engine and generation, and an entry without a generation UUID is never
measured. An engine never advertised, an unusable endpoint (malformed, or a port past
65535), an unreachable clock or a registry without the route fails that peer with its
`detail` and endpoint in `calibration.json` (bootstrap URLs and details with their
credentials redacted); local recording is unaffected. A handoff without a
bootstrap address is listed in `profiler.json` (`clock.unaddressed_engines`). Calibrating
an engine's worker aligns every capture of its clock domain (boot ID, time namespace,
clock source), its scheduler on that host included; a process in another domain stays
unaligned. The clock probes travel on the worker's KV address and a few kilobytes of
them count in that interface's bytes. Capabilities are by process role and KV role
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

In the end-to-end harness (`tools/e2e`) a run's optional `collector` key starts one
collector on each server node, inside the node's container (TENT's metrics server
listens on the node's loopback), with a manifest of the node's ranks and the
repository's `collector.py` copied in as it is. It starts after every server and the
router answer and before the harness's own first TENT scrape, and is stopped (its stop
file) after the workload and the harness's last scrape and before any server: its final
scrape needs the metrics servers. Its capture is copied to `collector/<node>/` and
checked: the terminal record and the summary say complete, the collector finished on
its own, every rank has a first and a last sample, no counter went backwards, and each
counter's change between them equals the harness's own scrape of the same rank (both
record kinds). Any miss is a problem of the run's verdict. The collector is not a
service of the run: the harness's server filters leave it alone and its CPU is counted
as a helper of its own kind (`collector_s`).

The same harness turns the Mooncake adapter on with the run's `profiler` key (L1 or L3, any data-parallel size;
at L3 each endpoint's observer as under NIXL):
both servers get one run ID and nothing else: the clocks are served and advertised as above. The run is
refused before any server starts when the prefill engines would be more clock peers than the profiler's
default `clock_max_peers` (64). The verdict works from the captures alone: every prefill engine a decode
scheduler handed off from (its recorded handoffs) calibrates once, `ok`, against that engine's TP0/PP0
worker clock (that capture's `clock_id`, served and advertised), with a mapping file of the same run and
clock pair; a calibration of an engine it never handed off from, or a handoff whose peer
its capture's dictionary does not resolve, fails too. The servers are stopped as under NIXL (decode first, so the clock servers are alive for the
end calibration) before the profile directories are copied. With `collector` as well, a producer rank's
batches and its TENT counters are both recorded, so they can be compared: the batches' bytes should equal the
collector's write bytes, and TENT's requests (it merges adjacent descriptors of a batch) be at most the batches'
descriptors. No tool makes this comparison; the harness checks the TENT counters against the bytes it moved.

### NIXL's built-in telemetry (L2 source `nixl`)

A collector target with `source: "nixl"` reads NIXL's own telemetry through its multi-process Prometheus exporter
(`prometheus_mp`, a beta exporter in the NIXL release the harness was built against): one aggregated endpoint a node serves every NIXL agent of that node. The servers are
given `NIXL_TELEMETRY_EXPORTER=prometheus_mp`, a private `NIXL_TELEMETRY_MULTIPROC_DIR` (tmpfs), a port on the node's loopback
and nothing else: vLLM's NIXL connector already requests telemetry from every agent (a discarding sink without an exporter),
so the exporter only adds accumulation, an endpoint and the scrapes. Two further collector-only record kinds (the native
recorder and kinds 22-26 are unchanged): 27 `nixl_telemetry` (per exporting process: `tx_bytes`, `rx_bytes`, `tx_requests`,
`rx_requests`, `xfer_time_us`, `xfer_post_time_us`, `memory_registered_bytes`, `memory_deregistered_bytes`; cumulative) and 28
`nixl_telemetry_events` (`events_dropped` at NIXL's producer-side staging queue, and `errors` by `status`: the errors NIXL
reports, not every failure, since some early returns bypass its error telemetry). Every record keeps the full series labels
(`hostname`, `agent_name`, `pid`, `agent_instance`, `local_rank` when exported); `agent_name` is a UUID in vLLM and encodes no
rank. A field the page did not serve is `null`; the last-operation gauges, the histogram buckets and the HTTP server's own
families are not recorded. NIXL counts only what an initiator completes: a prefill worker reports registration and no
transfers, and nothing is attributed per peer, backend or request.

NIXL exports asynchronously (a flush every 100 ms), so a successful scrape does not show that every completed transfer has
reached its counters. The collector therefore answers **boundary requests**: an orchestrator drops `<id>.json`
(`{"name": "baseline"}` or `"final"`) in `--boundary-dir`; the collector scrapes a pass that began after it noticed the file and
appends a `boundary` record (never changing an earlier one) naming the pass's sample IDs, its begin and end instants and each
target's completeness. The harness (`collector: {nixl_telemetry: true}`, any NIXL backend and data-parallel size; `tent: false`
for a backend without TENT) uses them as follows. *Inventory:* from the node's process titles, parents and start ticks it
expects one agent for each (DP rank, TP rank): the engine process when TP x DP is 1 (vLLM's uniprocess executor), otherwise
each `EngineCore_DP<d>` engine's worker children, one per TP rank, a worker joining its DP rank through its parent. Ranks are
numbered on the node as DP rank by TP rank, the order of the step logs. *Baseline:* the first boundary sample whose agents equal the inventory
exactly, with every field and every error series served and a positive registration (at least the rank's allocated KV bytes),
equal to the harness's own scrape taken right after; bounded, and a failure aborts the run. *Final:* with the servers and the
collector alive, after the counters have been unchanged for 0.5 s over samples that began after the workload, and still in
agreement with the inventory and the step logs, one more boundary sample equal to the stable stretch and to the harness's
own scrape, with the same processes (pid and start ticks) as at the baseline; a timeout or a mismatch is recorded and fails
the validation. The checks against what moved are per decode rank, over the two boundary samples, with zero tolerance: receive
requests and bytes equal the step log's transfers, the transfer and post times equal the sums of the logged durations each
converted back to integer microseconds, transmit zero, nothing dropped, no served error moved, registration unchanged; a prefill
rank's counters do not move. Equal values during quiescence establish a common accounting boundary, not simultaneous reads; both
reads' instants are kept (`collector/<node>/boundaries.json`). An agent is identified by its whole label set (hostname,
`agent_name`, `pid`, `agent_instance`, `local_rank` when present) and the node's own hostname is part of the inventory; a saved run is
judged again from the raw capture, never from a stored outcome: the request, the pass, the sample and the harness's own read in order,
the final stretch made of real consecutive complete samples (at least three, spanning 0.5 s, begun after the workload's last step in the
step logs) with unchanged counters equal to the selected final sample's, at the saved-run entry point (`harness recheck`) as well, which
reads TENT's saved evidence only where TENT applies.

For a latency comparison of NIXL telemetry configurations the harness has `server.nixl_telemetry_arm` (NIXL connector; the exporter arm is
`collector.nixl_telemetry`). `default` is vLLM's own configuration (the collect-only sink) and `emissions_disabled` sets
`NIXL_TELEMETRY_ENABLED_METRICS` to a token that matches none of NIXL's event names, so no event is emitted or enqueued while the timers, the
per-transfer statistics and the idle flush infrastructure remain (`NIXL_TELEMETRY_ENABLE=0` is not an option: without telemetry the connector's
per-transfer statistics raise and every completed transfer is handled as a failure). Each arm sets what it needs and clears every other
`NIXL_TELEMETRY_*` variable; the run's verdict checks the servers' environment, that no endpoint answers, that the per-transfer statistics are intact
and that no server log shows a telemetry failure, and reports the dropped-event count as unavailable (the collect-only sink discards that record too):
only the exporter arm measures it.

Latency comparisons use two workload options of the harness. `workload.percentiles` reports the benchmark's p50/p95/p99 of TTFT, TPOT, ITL and end-to-end
latency without `measure`'s per-node CPU samplers, so that nothing extra runs in any arm. `workload.monitor` runs a small contamination monitor around the
client's whole benchmark command: one stdlib process on a CPU outside the measured CPUs (15; the measured ones are 0-13) reads, once a second, the per-CPU
lines of `/proc/stat` and the `cpu.stat` of the topology's own containers' cgroups, each reading with its own begin and end instants; it never scans
`/proc` per process. The *foreign busy* time of an interval is the user, nice and system time of the measured CPUs minus the owned cgroups' user and system
time (interrupt and softirq time are reported, never counted: the run's own KV traffic produces it and the cgroups do not account it). Because the two
readings are taken at slightly different instants and `/proc/stat` is quantized in ticks, every interval carries a bound (the owned capacity times the
reading mismatch at both ends, plus the ticks), and a threshold is *exceeded* only if the estimate minus the bound is over it, *clear* only if the estimate
plus the bound is under it, and otherwise *unknown*, which never makes a run clean; accounting that is inconsistent beyond its own bound, a reset or missing
counter, a measured CPU that is absent, a gap above 3 s, many slow reads, samples that do not bracket both ends of the client's benchmark, owned cgroups that
are not disjoint, whose identities changed between the two ends or whose effective CPU sets reach outside 0-13 are unknown as well. The judgement (mean foreign
busy above 0.20 CPU, any 1 s sample above 0.75 CPU, a throttled owned cgroup) is saved with its raw samples (`monitor/`, `bench-window.json`) and re-derived
from them by readers, never taken from the stored status. The evidence is admitted strictly: a record that cannot be parsed or is truncated, a missing,
duplicate or out-of-order sequence entry, invalid or backwards read instants (nothing is sorted or dropped to repair a recording), a counter of any single CPU
or cgroup that falls between two readings (checked one counter at a time before anything is summed, so a fall offset by another counter's rise is found), an
ownership snapshot that is invalid on its own (each snapshot is validated separately: non-empty, disjoint cgroups, non-empty effective CPU sets inside 0-13) or
that differs from the other (a changed id, start instant, cgroup path or effective CPU assignment) are all *unknown*. The sampler's shutdown is evidence too
(`monitor/sampler.json`: started, exit status, whether it had to be killed, whether it was reaped): a non-zero exit, a forced kill or no record is *unknown*, and a
sampler whose starter is gone exits by itself. The window, both ownership snapshots, the sampler's termination and the judgement are saved even when the benchmark
raises or Docker stops answering.
