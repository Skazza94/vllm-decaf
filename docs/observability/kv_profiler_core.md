# KV profiler core (experimental, unattached)

This library is the core of the KV network profiler. It does not yet attach to
a connector, frontend or scheduler. Setting connector extra configuration alone
does **not** start capture. The adapter checkpoint follows separately.

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

## Qualification boundaries

The focused core tests use real recorder mappings, native publishers, helper
processes, files and loopback sockets. They cover publication, loss,
contention, wrap, retirement, failure/recovery and clock/export semantics.

They do not validate connector attachment, forwarding through vectorcall and
ordinary calls, native stats serialization under an observer, data transfer,
or production overhead. Keep the separately pinned timing/image baseline for
later serving A/B tests. No connector behavior or pairing is changed here.
