# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The endpoint's TCP sockets owned by its connector processes, by identity.

A socket is identified by its network namespace and kernel cookie (unique per
boot, never reused; ``sock_diag`` and the eBPF tracing helper use the same
generator), never by its address tuple, which another process may reuse. A
scan reads the fd tables of the frozen inventory's processes that host a
scheduler or worker, each verified against its frozen start ticks and
namespaces; takes one ``sock_diag`` dump of the namespace's TCP sockets (with
TCP_INFO, so the TCP source samples from the same dump, timed by the dump's
own interval); and reads those fd tables again. A socket is owned when one of
those processes, in its frozen generation, held its inode in both reads.
Every other process holding it is recorded on a best-effort read made last,
with the budget left. Each read is bounded, and the scan as a whole too;
nothing is admitted from a scan whose owner reads, dump or matching did not
complete. A process that hosts a frontend too (several roles in one process)
owns its frontend's sockets as well: such records say so.
"""

import errno
import ipaddress
import json
import os
import socket
import struct
import time

from .identity import _stat_fields, process_identity
from .netlink import ObservationError, query
from .tcp import decode_socket

OWNER_ROLES = ("scheduler", "worker")
ALL_STATES = 0xFFF  # every TCP state, LISTEN included
TCP_INFO_EXTENSION = 1 << (2 - 1)  # INET_DIAG_INFO
LISTEN = 10
FROZEN = ("start_ticks", "net_namespace", "pid_namespace")


class ScanBound(ObservationError):
    """One of the scan's own bounds (work or time): the scan is incomplete."""


class Limits:
    """Bounds on one scan's work: the /proc entries and processes read, each
    one's fds, all fds, the sockets dumped, and the whole scan's time."""

    def __init__(
        self,
        processes=4096,
        fds_per_process=16384,
        total_fds=131072,
        sockets=65536,
        deadline_s=0.5,
    ):
        self.processes, self.fds_per_process = processes, fds_per_process
        self.total_fds, self.sockets, self.deadline_s = total_fds, sockets, deadline_s


class Budget:
    """The work left in one scan."""

    def __init__(self, limits: Limits):
        self.limits = limits
        self.deadline = time.monotonic() + limits.deadline_s
        self.fds = limits.total_fds
        self.processes = limits.processes

    def expired(self) -> bool:
        return time.monotonic() >= self.deadline

    def check(self) -> None:
        if self.expired():
            raise ScanBound("discovery scan deadline")


def _inodes(pid: int, budget: Budget) -> set[int]:
    """The socket inodes ``pid`` holds now.

    Raises:
        OSError: its fd table cannot be read.
        ScanBound: a bound was reached.

    """
    found = set()
    with os.scandir(f"/proc/{pid}/fd") as entries:
        for count, entry in enumerate(entries):
            if count >= budget.limits.fds_per_process:
                raise ScanBound(f"process {pid} exceeds the per-process fd bound")
            budget.fds -= 1
            if budget.fds < 0:
                raise ScanBound("discovery scan fd bound")
            if count % 256 == 0:
                budget.check()
            try:
                target = os.readlink(entry.path)
            except FileNotFoundError:
                continue  # closed meanwhile
            if target.startswith("socket:["):
                found.add(int(target[8:-1]))
    return found


class Exited(Exception):
    """A process confirmed to have exited: gone, a zombie, or another process
    (or namespace) now under its PID."""


def _live(pid: int) -> dict:
    """``pid``'s live identity.

    Raises:
        Exited: it has exited (gone or a zombie).
        OSError, ObservationError: its identity cannot be read.

    """
    try:
        state = _stat_fields(pid)[0]
    except FileNotFoundError:
        raise Exited(f"process {pid} is gone") from None
    except ProcessLookupError:
        raise Exited(f"process {pid} is gone") from None
    if state in ("Z", "X"):
        raise Exited(f"process {pid} is a zombie")
    try:
        return process_identity(pid)
    except (FileNotFoundError, ProcessLookupError):
        raise Exited(f"process {pid} is gone") from None


class Mismatch(Exception):
    """A live process of a frozen generation (same start ticks) whose
    namespaces are not its frozen ones now: it holds nothing until they match
    again, and is not retired."""


def _matching(identities: list, live: dict) -> list:
    """The recorder identities that are this live process (start ticks and
    namespaces), each judged on its own."""
    return [
        identity
        for identity in identities
        if all(identity.get(key) in (None, live[key]) for key in FROZEN)
    ]


def _owner_identities(pid: int, identities: list) -> list:
    """The verified scheduler or worker identities (and any others) that
    ``pid`` is now.

    Raises:
        Exited: it has exited, or another process is under its PID.
        Mismatch: one of its generations, in other namespaces.
        OSError, ObservationError: its identity cannot be read.

    """
    live = _live(pid)
    matched = _matching(identities, live)
    if any(identity["role"] in OWNER_ROLES for identity in matched):
        return matched
    if any(
        identity["role"] in OWNER_ROLES
        and identity.get("start_ticks") in (None, live["start_ticks"])
        for identity in identities
    ):
        raise Mismatch(f"process {pid} is not in its frozen namespaces")
    raise Exited(
        f"process {pid} is not the scheduler or worker it was frozen as "
        "(another process under its PID)"
    )


def _dump(budget: Budget, unavailable: list) -> list[dict]:
    """Every TCP socket of this network namespace, with TCP_INFO. A family
    the kernel reports unsupported (``EAFNOSUPPORT``) is skipped and named in
    ``unavailable``.

    Raises:
        ObservationError, OSError: an incomplete dump.

    """
    rows = []
    for family in (socket.AF_INET, socket.AF_INET6):
        sockid = bytes(40) + struct.pack("=II", 0xFFFFFFFF, 0xFFFFFFFF)
        payload = (
            struct.pack("=BBBBI", family, 6, TCP_INFO_EXTENSION, 0, ALL_STATES) + sockid
        )
        try:
            for body in query(4, 20, payload, deadline=budget.deadline):
                rows.append(decode_socket(body))
                if len(rows) > budget.limits.sockets:
                    raise ScanBound("discovery socket bound")
        except OSError as error:
            if family == socket.AF_INET6 and error.errno == errno.EAFNOSUPPORT:
                unavailable.append("ipv6")
                continue
            raise
    return rows


def cookie_of(row) -> int | None:
    """The kernel cookie as one integer (``sock_diag`` gives two halves)."""
    if row.get("cookie") is None:
        return None
    low, high = row["cookie"]
    return low | high << 32


def _loopback(address: str) -> bool:
    value = ipaddress.ip_address(address)
    value = getattr(value, "ipv4_mapped", None) or value
    return value.is_loopback


def _record(row: dict, owning: list, verified: dict) -> dict:
    """One owned socket: its tuple, the owning processes' verified recorder
    identities (every role each hosts), and the dumped row; other holders are
    attached after, best effort."""
    owners = [dict(identity) for pid in owning for identity in verified[pid]]
    roles = {owner["role"] for owner in owners}
    return {
        "cookie": cookie_of(row),
        "inode": row["inode"],
        "family": row["family"],
        "local_ip": row["local_ip"],
        "local_port": row["local_port"],
        "peer_ip": row["peer_ip"],
        "peer_port": row["peer_port"],
        "state": row["state"],
        "listening": row["state"] == LISTEN,
        "loopback": _loopback(row["local_ip"]),
        "owners": owners,
        "owner_pids": sorted(owning),
        # a process hosting a frontend as well: its frontend's sockets are
        # owned too, and cannot be told apart
        "frontend_colocated": "frontend" in roles,
        "holders": [],
        "shared": len(owning) > 1,
        "row": row,
    }


class Scan:
    """One scan's outcome: ``complete``, or why not; the owned sockets by
    cookie; the dump's own interval (its TCP_INFO's acquisition time); the
    owners that could not be read; best-effort holder failures."""

    def __init__(self, started_ns):
        self.started_ns = started_ns
        self.finished_ns = None
        self.dump_started_ns = self.dump_finished_ns = None
        self.complete = False
        self.reason = None
        self.owned: dict[int, dict] = {}
        self.unreadable: dict[int, str] = {}  # live owners that cannot be read
        self.exited: dict[int, str] = {}  # owners confirmed to have exited
        # live owners in other namespaces than their frozen ones, for now
        self.mismatched: dict[int, str] = {}
        self.families_unavailable: list[str] = []
        self.holder_failures = 0
        self.holders_truncated = None  # why other holders were not all read
        self.dumped = 0


class Discovery:
    """Scans the endpoint's TCP sockets for those its connector processes own.

    ``owners`` are the frozen inventory's recorder identities, ``{pid,
    start_ticks, role}`` with their engine, rank and namespaces; a process
    owns sockets when it hosts a scheduler or worker, and is read once per
    pass whatever number of roles it hosts."""

    def __init__(self, owners, limits: Limits | None = None):
        identities: dict[int, list] = {}
        for owner in owners:
            identities.setdefault(owner["pid"], []).append(dict(owner))
        self.identities = {
            pid: found
            for pid, found in identities.items()
            if any(identity["role"] in OWNER_ROLES for identity in found)
        }
        self.retired: dict[int, str] = {}  # owners that exited, never read again
        self.limits = limits or Limits()

    def scan(self) -> Scan:
        result = Scan(time.monotonic_ns())
        budget = Budget(self.limits)
        try:
            # the owners first, around the dump; other holders last, best
            # effort, with whatever budget is left
            first = self._owner_inodes(result, budget)
            result.dump_started_ns = time.monotonic_ns()
            rows = _dump(budget, result.families_unavailable)
            result.dump_finished_ns = time.monotonic_ns()
            result.dumped = len(rows)
            second = self._owner_inodes(result, budget)
            # an identity counts only if verified in both reads, and a process
            # owns only with a scheduler or worker identity among those
            verified = {}
            for pid in first:
                if pid not in second:
                    continue
                both = [i for i in first[pid][1] if i in second[pid][1]]
                if any(i["role"] in OWNER_ROLES for i in both):
                    verified[pid] = both
                else:
                    result.mismatched[pid] = (
                        f"process {pid}'s verified generation changed during the scan"
                    )
            owned = {}
            for index, row in enumerate(rows):
                if index % 256 == 0:
                    budget.check()
                cookie = cookie_of(row)
                if row["record"] != "full" or not row["inode"] or not cookie:
                    continue  # time-wait and request socks have no owner
                owning = [
                    pid
                    for pid in verified
                    if row["inode"] in first[pid][0] and row["inode"] in second[pid][0]
                ]
                if owning:
                    owned[cookie] = _record(row, owning, verified)
                    if len(owned) % 256 == 0:
                        budget.check()
            budget.check()  # the mandatory records are built within the deadline
        except (OSError, ObservationError) as error:
            result.reason = f"{type(error).__name__}: {error}"
            result.finished_ns = time.monotonic_ns()
            return result
        holders = self._other_holders(result, budget)
        for record in owned.values():
            record["holders"] = holders.get(record["inode"], [])
            record["shared"] = bool(record["holders"]) or len(record["owner_pids"]) > 1
        result.owned = owned
        result.complete = True
        result.finished_ns = time.monotonic_ns()
        return result

    def _owner_inodes(self, result: Scan, budget: Budget) -> dict[int, tuple]:
        """Each owning process's socket inodes and the recorder identities it
        was verified as (start ticks and namespaces, each identity on its
        own). A process confirmed to have exited, before or during its read,
        is retired and never read again; one in other namespaces (a mismatch)
        or that cannot be read holds nothing for this scan; none affects
        another. A scan bound fails the whole scan."""
        found = {}
        for pid, identities in list(self.identities.items()):
            budget.check()
            try:
                before = _owner_identities(pid, identities)
                try:
                    inodes = _inodes(pid, budget)
                except (FileNotFoundError, ProcessLookupError):
                    # gone during the read: confirmed only by its status
                    _owner_identities(pid, identities)
                    raise
                after = _owner_identities(pid, identities)
                found[pid] = (inodes, [i for i in before if i in after])
            except ScanBound:
                raise
            except Exited as error:
                self.retired[pid] = str(error)
                del self.identities[pid]
                result.exited[pid] = str(error)
            except Mismatch as error:
                result.mismatched[pid] = str(error)
            except (OSError, ObservationError) as error:
                result.unreadable[pid] = f"{type(error).__name__}: {error}"
        return found

    def _other_holders(self, result: Scan, budget: Budget) -> dict[int, list]:
        """Best effort: every other visible process holding a socket inode,
        each read checked against its own generation. /proc is streamed, and
        every entry counts against the process bound."""
        holders: dict[int, list] = {}
        try:
            entries = os.scandir("/proc")
        except OSError:
            result.holders_truncated = "proc unreadable"
            return holders
        with entries:
            for entry in entries:
                budget.processes -= 1
                if budget.processes < 0 or budget.fds <= 0:
                    result.holders_truncated = "discovery bound"
                    break
                if budget.expired():
                    result.holders_truncated = "discovery scan deadline"
                    break
                if not entry.name.isdigit() or int(entry.name) in self.identities:
                    continue
                pid = int(entry.name)
                try:
                    start = process_identity(pid)["start_ticks"]
                    inodes = _inodes(pid, budget)
                    if process_identity(pid)["start_ticks"] != start:
                        raise ProcessLookupError(pid)
                except (OSError, ObservationError):
                    result.holder_failures += 1
                    continue
                for inode in inodes:
                    holders.setdefault(inode, []).append(
                        {"pid": pid, "start_ticks": start}
                    )
        return holders


class Tracker:
    """The owned sockets in force, by cookie, each under an admission ID.

    An admission is immutable: a socket whose owning processes or observed
    holders change gets a new admission (the old one ``reversioned``), and
    every admission is kept to the end, so a late event resolves to the
    record that was in force when it matched. A complete scan admits new
    cookies and removes those gone or no longer held by a verified owner; an
    incomplete scan suspends every admission at once, none in force until a
    complete scan confirms it again, under a new admission."""

    def __init__(self):
        self.active: dict[int, int] = {}  # cookie -> admission ID
        self.admissions: dict[int, dict] = {}  # admission ID -> record
        self.next_id = 1

    @staticmethod
    def _version(record) -> tuple:
        """What an admission stands for: its verified owner identities in full
        (a new generation under the same PID is another owner) and its
        observed holders."""
        return (
            tuple(
                sorted(json.dumps(owner, sort_keys=True) for owner in record["owners"])
            ),
            tuple(sorted((h["pid"], h["start_ticks"]) for h in record["holders"])),
        )

    def apply(self, scan: Scan) -> list[dict]:
        changes = []
        if not scan.complete:
            for cookie, admission in sorted(self.active.items()):
                changes.append(
                    self._end("suspended", cookie, admission, scan, scan.reason)
                )
            self.active.clear()
            return changes
        for cookie, admission in sorted(self.active.items()):
            if cookie not in scan.owned:
                held = self.admissions[admission]["owner_pids"]
                reason = (
                    "owner exited"
                    if any(pid in scan.exited for pid in held)
                    else "owner identity mismatch"
                    if any(pid in scan.mismatched for pid in held)
                    else "owner unreadable"
                    if any(pid in scan.unreadable for pid in held)
                    else "gone"
                )
                changes.append(self._end("removed", cookie, admission, scan, reason))
                del self.active[cookie]
        for cookie, record in sorted(scan.owned.items()):
            previous = self.active.get(cookie)
            if previous is not None:
                if self._version(self.admissions[previous]) == self._version(record):
                    continue
                changes.append(
                    self._end(
                        "reversioned",
                        cookie,
                        previous,
                        scan,
                        "owners or holders changed",
                    )  # fmt: skip
                )
            admission = self._admit(cookie, record, scan)
            stored = self.admissions[admission]
            # the lists at the top level, where an over-long record splits
            change = {
                "change": "admitted",
                "cookie": cookie,
                "admission": admission,
                "socket": {
                    k: v
                    for k, v in stored.items()
                    if k not in ("owners", "holders", "owner_pids")
                },
                "owners": stored["owners"],
                "holders": stored["holders"],
            }
            if previous is not None:
                change["supersedes"] = previous
            changes.append(change)
        return changes

    def _admit(self, cookie, record, scan) -> int:
        admission = self.next_id
        self.next_id += 1
        self.admissions[admission] = {
            **{k: v for k, v in record.items() if k != "row"},
            "admission": admission,
            "admitted_ns": scan.finished_ns,
            "scan_started_ns": scan.started_ns,
        }
        self.active[cookie] = admission
        return admission

    @staticmethod
    def _end(kind, cookie, admission, scan, reason) -> dict:
        # the admission's record stays exactly as admitted
        return {
            "change": kind,
            "cookie": cookie,
            "admission": admission,
            "reason": reason,
            "ended_ns": scan.finished_ns,
        }

    def owned_rows(self, scan: Scan) -> list[tuple]:
        """(admission, record) of the sockets in force in ``scan``."""
        return [
            (admission, scan.owned[cookie])
            for cookie, admission in self.active.items()
            if cookie in scan.owned
        ]


class Cadence:
    """A fixed schedule of starts every ``interval_s``: a start later than
    its slot counts the whole periods it skipped as missed."""

    def __init__(self, interval_s: float, start: float):
        self.interval_s, self.next = interval_s, start
        self.missed = 0

    def due(self, now: float) -> tuple[bool, int]:
        """(whether a start is due, the periods missed before it)."""
        if now < self.next:
            return False, 0
        skipped = int((now - self.next) / self.interval_s)
        self.missed += skipped
        self.next += (skipped + 1) * self.interval_s
        return True, skipped
