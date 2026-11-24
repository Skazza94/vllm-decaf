# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent bounded control exchange; no NIXL or serving protocol changes."""

import contextlib
import errno
import hashlib
import os
import select
import socket
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from uuid import UUID, uuid4

PROBE = struct.Struct("!4sBBH16s16sQQ")
# The reply names its server's engine (a digest of its ID), so a consumer never
# takes another engine's clock, as a same-run server on a colliding port, for
# the peer it asked for.
REPLY = struct.Struct("!4sBBH16s16s16s16sQQQQ")
MAGIC = b"KVP1"
VERSION = 2


def engine_digest(engine_id: str | None) -> bytes:
    """16 bytes naming ``engine_id`` (zeros for a server that names none)."""
    if engine_id is None:
        return bytes(16)
    return hashlib.blake2b(engine_id.encode(), digest_size=16).digest()


def _check_port(port, low: int) -> int:
    if type(port) is not int or not low <= port <= 65535:
        raise ValueError(f"clock port {port!r} is out of range {low}-65535")
    return port


@dataclass(frozen=True)
class ClockSample:
    run_id: str
    local_generation: str
    remote_generation: str
    t1: int
    t2: int
    t3: int
    t4: int

    def __post_init__(self):
        for value in (self.run_id, self.local_generation, self.remote_generation):
            UUID(value)
        if any(
            type(v) is not int or not 0 <= v < 1 << 64
            for v in (self.t1, self.t2, self.t3, self.t4)
        ):
            raise ValueError("timestamps must be unsigned integer nanoseconds")
        if self.t4 < self.t1 or self.t3 < self.t2 or self.network_rtt_ns < 0:
            raise ValueError("invalid four-timestamp sample")

    @property
    def network_rtt_ns(self):
        return self.t4 - self.t1 - (self.t3 - self.t2)

    @property
    def raw_rtt_ns(self):
        return self.t4 - self.t1

    @property
    def offset(self):
        return Fraction((self.t2 - self.t1) + (self.t3 - self.t4), 2)

    @property
    def anchor(self):
        return Fraction(self.t1 + self.t4, 2)

    def to_json(self):
        return asdict(self)


def select_min_rtt(samples: list[ClockSample]) -> ClockSample:
    if not samples:
        raise ValueError("no valid clock samples")
    identities = {(s.run_id, s.local_generation, s.remote_generation) for s in samples}
    if len(identities) != 1:
        raise ValueError("clock generation changed")
    return min(samples, key=lambda s: (s.network_rtt_ns, s.raw_rtt_ns))


@dataclass(frozen=True)
class Projection:
    time_ns: Fraction
    uncertainty_ns: int
    drift_known: bool
    extrapolated: bool


@dataclass(frozen=True)
class ClockMapping:
    start: ClockSample
    end: ClockSample | None = None
    raw_start: tuple[ClockSample, ...] = ()
    raw_end: tuple[ClockSample, ...] = ()

    def __post_init__(self):
        if self.raw_start and select_min_rtt(list(self.raw_start)) != self.start:
            raise ValueError("start anchor is not minimum RTT")
        if self.raw_end and select_min_rtt(list(self.raw_end)) != self.end:
            raise ValueError("end anchor is not minimum RTT")
        if self.end is not None:
            select_min_rtt([self.start, self.end])
            if self.end.anchor <= self.start.anchor:
                raise ValueError("end calibration must follow start")
            if 1 + self.slope <= 0:
                raise ValueError("non-monotonic clock mapping")

    @classmethod
    def from_samples(cls, start, end=None):
        return cls(
            select_min_rtt(start),
            select_min_rtt(end) if end else None,
            tuple(start),
            tuple(end or ()),
        )

    def to_json(self):
        return {
            "start": [s.to_json() for s in self.raw_start or (self.start,)],
            "end": [
                s.to_json() for s in self.raw_end or ((self.end,) if self.end else ())
            ],
            "model": "linear_estimate_not_a_bound",
        }

    @classmethod
    def from_json(cls, value):
        return cls.from_samples(
            [ClockSample(**s) for s in value["start"]],
            [ClockSample(**s) for s in value["end"]],
        )

    @property
    def slope(self):
        if self.end is None:
            return Fraction(0)
        return (self.end.offset - self.start.offset) / (
            self.end.anchor - self.start.anchor
        )

    def project(self, remote_ns: int) -> Projection:
        local = (remote_ns - self.start.offset + self.slope * self.start.anchor) / (
            1 + self.slope
        )
        return Projection(
            local,
            max(self.start.raw_rtt_ns, self.end.raw_rtt_ns if self.end else 0),
            self.end is not None,
            (local != self.start.anchor)
            if self.end is None
            else (local < self.start.anchor or local > self.end.anchor),
        )


class ClockGroup:
    """One reference epoch after alignment, never one zero per process."""

    def __init__(self, reference: str, mappings: dict[str, ClockMapping] | None = None):
        UUID(reference)
        self.reference = reference
        self.mappings = dict(mappings or {})
        runs = {m.start.run_id for m in self.mappings.values()}
        if len(runs) > 1:
            raise ValueError("clock group mixes runs")
        self.run_id = next(iter(runs), None)
        for clock, mapping in self.mappings.items():
            if (
                mapping.start.local_generation != reference
                or mapping.start.remote_generation != clock
            ):
                raise ValueError("mapping does not belong to this clock group")

    def project(self, clock: str, timestamp: int) -> Projection | None:
        if clock == self.reference:
            return Projection(Fraction(timestamp), 0, True, False)
        mapping = self.mappings.get(clock)
        return mapping.project(timestamp) if mapping is not None else None


def _read(sock: socket.socket, length: int, deadline: float) -> bytes:
    data = bytearray()
    while len(data) < length:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("clock exchange deadline")
        sock.settimeout(remaining)
        part = sock.recv(length - len(data))
        if not part:
            raise EOFError("short clock frame")
        data.extend(part)
    return bytes(data)


def _server(sock: socket.socket, run: str, generation: str, engine, ready, parent):
    run_bytes, generation_bytes = UUID(run).bytes, UUID(generation).bytes
    engine_bytes = bytes.fromhex(engine)
    sock.settimeout(0.1)
    os.write(ready, b"ready")
    os.close(ready)
    while os.getppid() == parent:
        try:
            connection, _ = sock.accept()
        except TimeoutError:
            continue
        with connection:
            try:
                while True:
                    deadline = time.monotonic() + 1
                    prefix = _read(connection, 8, deadline)
                    magic, version, kind, length = struct.unpack("!4sBBH", prefix)
                    if (magic, version, kind, length) != (
                        MAGIC,
                        VERSION,
                        1,
                        PROBE.size,
                    ):
                        break
                    request = PROBE.unpack(
                        prefix + _read(connection, PROBE.size - 8, deadline)
                    )
                    t2 = time.monotonic_ns()
                    if request[4] != run_bytes:
                        break
                    _, _, _, _, _, initiator, nonce, t1 = request
                    t3 = time.monotonic_ns()
                    reply = REPLY.pack(
                        MAGIC,
                        VERSION,
                        2,
                        REPLY.size,
                        run_bytes,
                        initiator,
                        generation_bytes,
                        engine_bytes,
                        nonce,
                        t1,
                        t2,
                        t3,
                    )
                    connection.settimeout(max(0.001, deadline - time.monotonic()))
                    connection.sendall(reply)
            except (OSError, EOFError):
                pass


# A wildcard dual-stack bind failing with these means the host has no usable
# IPv6; any other error (an occupied port) is the bind's own.
_NO_IPV6 = (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL, errno.EPROTONOSUPPORT)


def listen(host: str | None, port: int) -> tuple[socket.socket, str, str | None]:
    """A listening socket, its family (``ipv4``, ``ipv6`` or ``dual``) and why a
    wildcard fell back to IPv4 (None otherwise).

    ``host`` None is the wildcard: dual-stack where the host supports it, else
    IPv4. Otherwise the family is the one ``host`` resolves to first (an IPv6
    literal binds IPv6 only), and its bind error is raised as is.
    """
    _check_port(port, 0)  # never let the resolver wrap it (0 is ephemeral)
    if host is None:
        fallback = "dual-stack IPv6 not supported"
        if socket.has_dualstack_ipv6():
            try:
                sock = socket.create_server(
                    ("::", port), family=socket.AF_INET6, backlog=8, dualstack_ipv6=True
                )
                return sock, "dual", None
            except OSError as error:
                if error.errno not in _NO_IPV6:
                    raise
                fallback = f"IPv6 unavailable: {os.strerror(error.errno)}"
        sock = socket.create_server(("0.0.0.0", port), family=socket.AF_INET, backlog=8)
        return sock, "ipv4", fallback
    family, _, _, _, address = socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )[0]
    sock = socket.create_server(address, family=family, backlog=8)
    return sock, "ipv6" if family == socket.AF_INET6 else "ipv4", None


class ClockServer:
    """A separate helper process in the caller's time namespace.

    Integration with first connection/end-of-run waits for the adapter stage.
    No listener is created just by importing this module.
    """

    def __init__(
        self,
        run_id: str,
        *,
        generation: str | None = None,
        engine_id: str | None = None,
        host: str | None = "127.0.0.1",
        port: int = 0,
    ):
        self.run_id = str(UUID(run_id))
        self.generation = str(UUID(generation)) if generation else str(uuid4())
        self.engine_id = engine_id
        self.socket, self.family, self.fallback = listen(host, port)
        self.address = self.socket.getsockname()[:2]
        receive = send = None
        try:
            receive, send = os.pipe()
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "vllm.kv_profiling.clock",
                    str(self.socket.fileno()),
                    self.run_id,
                    self.generation,
                    engine_digest(engine_id).hex(),
                    str(send),
                    str(os.getpid()),
                ],
                pass_fds=(self.socket.fileno(), send),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except BaseException:
            # nothing of a failed start stays open: the pipe and the listener
            for fd in (receive, send):
                if fd is not None:
                    os.close(fd)
            self.socket.close()
            raise
        os.close(send)
        try:
            # poll, not select: the pipe may be any descriptor number
            ready = select.poll()
            ready.register(receive, select.POLLIN)
            if not ready.poll(5000) or os.read(receive, 5) != b"ready":
                raise TimeoutError("clock helper did not start")
        except BaseException:
            # a helper that may already listen is stopped and reaped, and the
            # listener closed, so the port is free again
            self.close()
            raise
        finally:
            os.close(receive)

    def close(self, timeout=0.5):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.process.wait(timeout)
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def start_server(
    run_id: str, generation: str, engine_id: str, host: str | None, port: int
):
    """``(server, report)``: a producer's clock server, or None when it cannot
    listen or start, so only the clock is unavailable (never the recorder). The
    report goes into ``profiler.json``."""
    try:
        server = ClockServer(
            run_id, generation=generation, engine_id=engine_id, host=host, port=port
        )
    except (OSError, ValueError) as error:  # out of range, the bind, the helper
        return None, {
            "state": "unavailable",
            "host": host,
            "port": port,
            "error": f"{type(error).__name__}: {error}",
        }
    return server, {
        "state": "available",
        "family": server.family,
        "fallback": server.fallback,
        "address": list(server.address),
    }


def measure_peer(
    address: tuple[str, int],
    run_id: str,
    local_generation: str,
    *,
    expected_engine: str | None = None,
    expected_remote: str | None = None,
    samples: int = 7,
    timeout_s: float = 1,
) -> list[ClockSample]:
    """Keep all raw samples; the caller persists start/end separately.

    Raises:
        ValueError: an out-of-range port, a reply of another run, origin or
            exchange, of another engine than ``expected_engine`` (when given),
            or a changed generation.

    """
    if not 1 <= samples <= 32 or not 0 < timeout_s <= 10:
        raise ValueError("invalid clock sample count or deadline")
    _check_port(address[1], 1)
    engine = None if expected_engine is None else engine_digest(expected_engine)
    run, local = UUID(run_id).bytes, UUID(local_generation).bytes
    deadline = time.monotonic() + timeout_s
    result = []
    with socket.create_connection(address, timeout=timeout_s) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for nonce in range(samples):
            t1 = time.monotonic_ns()
            request = PROBE.pack(MAGIC, VERSION, 1, PROBE.size, run, local, nonce, t1)
            sock.settimeout(max(0.001, deadline - time.monotonic()))
            sock.sendall(request)
            raw = _read(sock, REPLY.size, deadline)
            t4 = time.monotonic_ns()
            (m, version, kind, length, r, origin, remote, answering, n, echoed, t2,
             t3) = REPLY.unpack(raw)  # fmt: skip
            if (
                (m, version, kind, length) != (MAGIC, VERSION, 2, REPLY.size)
                or r != run
                or origin != local
                or n != nonce
                or echoed != t1
            ):
                raise ValueError("clock reply mismatch")
            if engine is not None and answering != engine:
                raise ValueError("clock answered by another engine")
            generation = str(UUID(bytes=remote))
            if expected_remote is not None and generation != expected_remote:
                raise ValueError("clock generation changed")
            result.append(
                ClockSample(
                    run_id,
                    local_generation,
                    generation,
                    t1,
                    t2,
                    t3,
                    t4,
                )
            )
    select_min_rtt(result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("socket_fd", type=int)
    parser.add_argument("run_id")
    parser.add_argument("generation")
    parser.add_argument("engine_digest")
    parser.add_argument("ready_fd", type=int)
    parser.add_argument("parent_pid", type=int)
    args = parser.parse_args()
    with socket.socket(fileno=args.socket_fd) as listener:
        _server(
            listener,
            args.run_id,
            args.generation,
            args.engine_digest,
            args.ready_fd,
            args.parent_pid,
        )
