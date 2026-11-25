# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""First-peer admission and clock calibration, off the request path.

Admission is a preallocated table: four slot lists built at attach, a published
count and a last-hit index, all small cached ints. It takes no lock, creates no
tuple, set, iterator or thread, and never waits. One long-lived thread measures
published peers, each on its own schedule: an advertised peer still being looked
up never holds up another. Only the closer writes files and only the closer
touches the recorder.
"""

import queue
import threading
import time
from pathlib import Path
from uuid import UUID

from .clock import ClockMapping, measure_peer
from .clock_registry import NoClockRoute, endpoint_of, lookup
from .sink import atomic_json
from .snapshot import sanitize

PEER_OK = "ok"
PEER_FAILED = "failed"
PEER_PENDING = "pending_at_close"
PEER_ABANDONED = "abandoned"

IO_WORKERS = 4  # lookups and exchanges in flight at once
IDLE_S = 0.1  # the longest wait between scheduling passes
_NEVER = float("inf")
REFRESH_S = 0.2  # a listing missing an engine is fetched again after this


def _detail(error: BaseException) -> str:
    return sanitize(f"{type(error).__name__}: {error}")


def _advertised(entry) -> tuple[tuple[str, int], str]:
    """An advertised entry's endpoint and generation.

    Raises:
        ValueError: not an entry, a malformed endpoint, or no generation UUID.

    """
    if not isinstance(entry, dict):
        raise ValueError(f"not an entry: {entry!r}")
    generation = entry.get("generation")
    if type(generation) is not str:
        raise ValueError(f"no clock generation: {generation!r}")
    UUID(generation)
    return endpoint_of(entry.get("endpoint")), generation


class _Expired(Exception):
    """A task taken from the queue after its deadline, or after close began."""


class _Task:
    def __init__(self, function, args, deadline):
        self.function, self.args, self.deadline = function, args, deadline
        self.done, self.result, self.error, self.finished = False, None, None, None


class _Pool:
    """A few daemon workers for the clock thread's I/O, started on first use
    and stopped at close. A task runs only if it is claimed before its
    absolute deadline and before close began; ``stop`` cancels every queued
    one, and a running one finishes within its own bound. Stopping,
    registering a worker and claiming a task take one lock (never held across
    I/O), so close sees every worker and every claimed task. A finished task
    wakes the scheduler."""

    def __init__(self, workers: int, wake: threading.Event):
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._wake = wake
        self._workers = workers
        self._threads: list[threading.Thread] = []
        self._stopped = False
        self.running = 0  # claimed tasks not finished yet
        self._lock = threading.Lock()

    def submit(self, function, *args, deadline: float | None = None) -> _Task:
        """Run ``function(*args, deadline)`` on a worker; ``deadline`` (absolute,
        monotonic) or None."""
        task = _Task(function, args, deadline)
        with self._lock:
            if self._stopped:
                task.error, task.finished, task.done = _Expired("closed"), 0.0, True
                return task
            if len(self._threads) < self._workers:
                thread = threading.Thread(
                    target=self._work,
                    name=f"kvprof-clock-io-{len(self._threads)}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()
            self._queue.put(task)
        return task

    def stop(self) -> None:
        """Cancel the queued tasks and let every worker exit."""
        with self._lock:
            self._stopped = True
            for _ in self._threads:
                self._queue.put(None)

    def _work(self) -> None:
        while (task := self._queue.get()) is not None:
            with self._lock:  # a claim is atomic with stop
                if self._stopped:
                    task.error = _Expired("closed")
                elif task.deadline is not None and time.monotonic() >= task.deadline:
                    task.error = _Expired("taken after its deadline")
                else:
                    self.running += 1
            if task.error is None:
                try:
                    task.result = task.function(*task.args, task.deadline)
                except Exception as exc:
                    task.error = exc
                with self._lock:
                    self.running -= 1
            task.finished = time.monotonic()
            task.done = True
            self._wake.set()


def _bounded(deadline: float | None, cap: float = 1.0) -> float:
    """A blocking call's timeout: ``cap``, or the time left before ``deadline``."""
    if deadline is None:
        return cap
    return max(0.001, min(cap, deadline - time.monotonic()))


def _lookup(base: str, run_id: str, deadline: float | None) -> dict:
    return lookup(base, run_id, _bounded(deadline))


class PeerClocks:
    def __init__(
        self, config, run_id: str, local_generation: str, *, offset: int | None = None
    ):
        """``offset`` is added to a directly admitted peer's port: the NIXL
        side-channel port plus ``clock_port_offset`` by default. Peers admitted
        by their bootstrap address (Mooncake) use the endpoint they advertise."""
        self.config = config
        self.offset = config.clock_port_offset if offset is None else offset
        self.run_id = run_id
        self.local_generation = local_generation
        cap = self.cap = config.clock_max_peers
        self.engine = [None] * cap
        self.host = [None] * cap
        self.port = [None] * cap
        self.bootstrap = [None] * cap
        self.tail = 0
        self.last = 0
        self.closed = False
        # by slot, written by the measuring thread only
        self.results: dict[int, dict] = {}
        self.waiting: dict[int, str] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()  # set by a finished I/O task, or close
        self._pool = _Pool(IO_WORKERS, self._wake)
        self.thread = threading.Thread(
            target=self._run, name="kvprof-clock", daemon=True
        )
        self.thread.start()

    def admit(self, engine_id, host, port) -> int:
        """Admit a peer at its side-channel ``host``/``port`` (plus the offset).

        Return 1 for a new stored peer, 0 for a known one, -1 when full.
        """
        return self._admit(engine_id, host, port, None)

    def admit_advertised(self, engine_id, bootstrap) -> int:
        """Admit a peer whose clock endpoint ``bootstrap``'s registry advertises
        (looked up by the measuring thread, never here)."""
        return self._admit(engine_id, None, None, bootstrap)

    def _admit(self, engine_id, host, port, bootstrap) -> int:
        """Publishing order: every slot field is stored before the count moves, so
        the measuring thread never reads a half-written slot.
        """
        tail = self.tail
        if tail and self.engine[self.last] == engine_id:
            return 0
        i = 0
        while i < tail:
            if self.engine[i] == engine_id:
                self.last = i
                return 0
            i += 1
        if tail == self.cap:
            return -1
        self.engine[tail] = engine_id
        self.host[tail] = host
        self.port[tail] = port
        self.bootstrap[tail] = bootstrap
        self.last = tail
        self.tail = tail + 1
        return 1

    def _run(self) -> None:
        """Schedule every peer on its own clock, its I/O on a few daemon
        workers: a direct peer is measured once; an advertised one waits for
        its registry's listing (one lookup in flight per registry, bounded by
        the time left in its peers' windows), and is measured with the
        advertised engine and generation; a failed check within its window
        refreshes the listing and retries. No lookup or exchange of one peer
        or registry waits for another's."""
        pool = self._pool
        slots: dict[int, dict] = {}
        listings: dict[str, dict] = {}
        seen = 0
        while not self.closed:
            now = time.monotonic()
            while seen < self.tail:
                advertised = self.bootstrap[seen] is not None
                slots[seen] = {
                    "due": now,
                    # the discovery window is Mooncake's: a direct (NIXL) peer
                    # is measured once, under the exchange's own bound
                    "deadline": (
                        now + self.config.clock_discovery_s if advertised else _NEVER
                    ),
                    "delay": 0.25,
                    "task": None,
                    "after": 0.0,  # a listing must be newer than this
                    "last": None,
                }
                seen += 1
            for base, listing in listings.items():
                task = listing.get("task")
                if task is not None and task.done:
                    listing["task"] = None
                    listing["fetched"] = task.finished
                    if task.error is None:
                        listing.update(entries=task.result, error=None, final=None)
                    elif isinstance(task.error, NoClockRoute):
                        listing["final"] = sanitize(str(task.error))
                    else:
                        listing["error"] = _detail(task.error)
                    for index, slot in slots.items():  # its peers act on it now
                        if self.bootstrap[index] == base and slot["task"] is None:
                            slot["due"] = now
            for index in list(slots):
                if self._step(index, slots[index], listings, pool, now):
                    del slots[index]
            wake = min(
                (
                    min(s["deadline"], s["due"] if s["task"] is None else _NEVER)
                    for s in slots.values()
                ),
                default=now + IDLE_S,
            )
            self._wake.wait(min(IDLE_S, max(0.0, wake - time.monotonic())))
            self._wake.clear()

    def _step(self, index, slot, listings, pool, now) -> bool:
        """Advance one peer; True once it has a result."""
        engine, base = self.engine[index], self.bootstrap[index]
        task = slot["task"]
        if task is not None and (not task.done or task.finished > slot["deadline"]):
            if now < slot["deadline"]:
                return False
            # the window ended with its exchange queued or running: whatever
            # it returns later does not count
            self._fail(
                index,
                "no clock answer within the discovery window",
                address=slot.get("address"),
            )
            return True
        if task is not None:
            slot["task"] = None
            entry = task.result
            if task.error is not None:  # taken too late (or at close)
                entry = {"engine_id": engine, "address": slot.get("address"),
                         "status": PEER_FAILED, "error": "Expired",
                         "detail": str(task.error)}  # fmt: skip
            if entry["status"] == PEER_OK or base is None or now >= slot["deadline"]:
                self._finish(index, entry)
                return True
            # within the window: the endpoint or generation may have moved
            slot.update(last=entry.get("detail"), after=task.finished, entry=entry)
            slot["due"] = now + slot["delay"]
            slot["delay"] = min(slot["delay"] * 2, 2.0)
            self.waiting[index] = sanitize(f"retrying: {entry.get('detail')}")
            return False
        remaining = slot["deadline"] - now
        if now < slot["due"] and remaining > 0:
            return False
        listing = (
            listings.setdefault(base, {"fetched": None, "entries": {}})
            if base is not None
            else {}
        )
        if remaining <= 0 and slot.get("entry") is not None:
            # its exchanges kept failing: the last one is its record
            self._finish(index, slot["entry"])
            return True
        if remaining <= 0:
            last = listing.get("error") or slot["last"]
            self._fail(
                index,
                f"{engine} not "
                + ("advertised by " + base if base else "calibrated")
                + f" within {self.config.clock_discovery_s:g} s"
                + (f" (last: {last})" if last else ""),
            )
            return True
        if base is None:
            address = (self.host[index], self.port[index] + self.offset)
            slot["address"] = list(address)
            slot["task"] = pool.submit(self._exchange, index, address, None)
            return False
        if listing.get("final"):
            self._fail(index, listing["final"])
            return True
        fresh = listing["fetched"] is not None and listing["fetched"] > slot["after"]
        advertised = listing["entries"].get(engine) if fresh else None
        if advertised is not None:
            try:
                address, generation = _advertised(advertised)
            except ValueError as exc:
                self._fail(index, f"{engine}'s advertised clock is unusable: {exc}")
                return True
            slot["address"] = list(address)
            slot["task"] = pool.submit(
                self._exchange, index, address, generation, deadline=slot["deadline"]
            )
            return False
        stale = not fresh or now - listing["fetched"] >= REFRESH_S
        if listing.get("task") is None and stale:
            # one lookup per registry, shared by its peers, bounded by this
            # peer's window
            listing["task"] = pool.submit(
                _lookup, base, self.run_id, deadline=slot["deadline"]
            )
        self.waiting[index] = sanitize(
            listing.get("error") or f"not yet advertised by {base}"
        )
        slot["due"] = now + slot["delay"]
        slot["delay"] = min(slot["delay"] * 2, 2.0)
        return False

    def _exchange(
        self, index: int, address, generation: str | None, deadline: float | None
    ) -> dict:
        """The start anchors, bounded by 1 s and the window's ``deadline``
        (None for a direct peer); ``generation`` is the advertised one, which
        the first exchange already requires (as the last does)."""
        entry = {"engine_id": self.engine[index], "address": list(address)}
        if generation is not None:
            entry["advertised_generation"] = generation
        try:
            samples = measure_peer(
                address,
                self.run_id,
                self.local_generation,
                expected_engine=self.engine[index],
                expected_remote=generation,
                samples=self.config.clock_samples,
                timeout_s=_bounded(deadline),
            )
            entry.update(
                status=PEER_OK,
                remote_generation=samples[0].remote_generation,
                start=samples,
            )
        except Exception as exc:
            entry.update(
                status=PEER_FAILED, error=type(exc).__name__, detail=sanitize(str(exc))
            )
        return entry

    def _finish(self, index: int, entry: dict) -> None:
        with self._lock:
            self.results[index] = entry
            self.waiting.pop(index, None)

    def _fail(self, index: int, detail: str, address=None) -> None:
        self._finish(
            index,
            {
                "engine_id": self.engine[index],
                "address": address,
                "status": PEER_FAILED,
                "error": "NotDiscovered",
                "detail": sanitize(detail),
            },
        )

    def _address(self, index: int):
        """A direct peer's endpoint; an advertised one's is known once found."""
        if self.bootstrap[index] is not None:
            return None
        return self.host[index], self.port[index] + self.offset

    def close(self, output_dir: Path, join_s: float) -> dict:
        """Stop, take the end anchors under one budget and write the manifest.

        Called after every hook has stopped. A thread still alive after the
        bounded join is abandoned: its list is only snapshotted, never awaited,
        and nothing it does afterwards reaches a file or the recorder.
        """
        self.closed = True
        self._pool.stop()  # nothing queued starts from now on
        self._wake.set()
        output_dir.mkdir(parents=True, exist_ok=True)
        until = time.monotonic() + join_s
        self.thread.join(join_s)
        # an exchange or lookup still in flight is waited for within the same
        # budget, then abandoned with its thread
        while self._pool.running and time.monotonic() < until:
            time.sleep(0.01)
        abandoned = self.thread.is_alive() or bool(self._pool.running)
        with self._lock:  # one copy, even of an abandoned thread
            results = dict(self.results)
            waiting = dict(self.waiting)
        deadline = time.monotonic() + min(
            self.config.clock_end_budget_s, self.config.shutdown_timeout_s
        )
        peers = []
        files = []
        for index in range(self.tail):
            entry = results.get(index)
            direct = self._address(index)
            record = {
                "engine_id": self.engine[index],
                "address": (
                    entry["address"]
                    if entry is not None
                    else (list(direct) if direct else None)
                ),
            }
            if self.bootstrap[index] is not None:
                record["bootstrap"] = sanitize(self.bootstrap[index])
            if entry is None:
                record["status"] = PEER_ABANDONED if abandoned else PEER_PENDING
                if index in waiting:
                    record["detail"] = waiting[index]
                peers.append(record)
                continue
            record["status"] = entry["status"]
            if "advertised_generation" in entry:
                record["advertised_generation"] = entry["advertised_generation"]
            if entry["status"] != PEER_OK:
                record["error"] = entry["error"]
                record["detail"] = sanitize(entry.get("detail"))
                peers.append(record)
                continue
            record["remote_generation"] = entry["remote_generation"]
            end = []
            remaining = deadline - time.monotonic()
            if remaining > 0:
                try:
                    end = measure_peer(
                        tuple(record["address"]),
                        self.run_id,
                        self.local_generation,
                        expected_engine=record["engine_id"],
                        expected_remote=entry["remote_generation"],
                        samples=self.config.clock_samples,
                        timeout_s=min(1, remaining),
                    )
                except Exception as exc:
                    record["end_error"] = type(exc).__name__
            record["end_missing"] = not end
            try:
                mapping = ClockMapping.from_samples(entry["start"], end)
            except ValueError as exc:
                record["status"] = PEER_FAILED
                record["error"] = str(exc)
                peers.append(record)
                continue
            name = f"calibration-{len(files)}.json"
            atomic_json(output_dir / name, mapping.to_json())
            record["calibration_file"] = name
            files.append(name)
            peers.append(record)
        manifest = {
            "run_id": self.run_id,
            "local_generation": self.local_generation,
            "clock_thread": PEER_ABANDONED if abandoned else "joined",
            "peers": peers,
        }
        atomic_json(output_dir / "calibration.json", manifest)
        return manifest
