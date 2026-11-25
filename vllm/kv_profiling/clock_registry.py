# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Advertised clock endpoints of Mooncake producer engines.

A producer engine's clock server runs in its TP0/PP0 worker, on the worker's own
KV host and an ephemeral port, and is advertised to the instance's bootstrap
server under one more route (``ROUTE``). A consumer finds it from the
``remote_bootstrap_addr`` its requests carry. The bootstrap server's own
``/register`` and ``/query`` are unchanged. Standard library only, apart from
the route itself (FastAPI, which the bootstrap server already runs).
"""

import http.client
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from uuid import UUID

from .snapshot import sanitize

ROUTE = "/kv_profiler/clocks"
MAX_ENGINES = 4096
FIELDS = ("run_id", "engine_id", "dp_rank", "endpoint", "generation")
ACKNOWLEDGED = ("registered", "unchanged", "replaced")


def endpoint_of(text) -> tuple[str, int]:
    """``host:port`` (an IPv6 host in brackets) as ``(host, port)``.

    Raises:
        ValueError: not ``host:port``, or a port outside 1-65535.

    """
    if type(text) is not str or ":" not in text:
        raise ValueError(f"clock endpoint {text!r} is not host:port")
    host, _, port = text.rpartition(":")
    host = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    if not host or any(c.isspace() for c in host) or not port.isdigit():
        raise ValueError(f"clock endpoint {text!r} is not host:port")
    if not 1 <= int(port) <= 65535:
        raise ValueError(f"clock port {int(port)} is out of range 1-65535")
    return host, int(port)


def endpoint_text(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def validated(record) -> dict:
    """A registration with every field of its type.

    Raises:
        ValueError: a missing, extra or mistyped field.

    """
    if not isinstance(record, dict) or set(record) != set(FIELDS):
        raise ValueError(f"a clock registration has exactly the fields {FIELDS}")
    UUID(record["run_id"])
    UUID(record["generation"])
    if type(record["engine_id"]) is not str or not record["engine_id"]:
        raise ValueError("engine_id must be a nonempty string")
    if type(record["dp_rank"]) is not int or record["dp_rank"] < 0:
        raise ValueError("dp_rank must be a nonnegative integer")
    endpoint_of(record["endpoint"])
    return dict(record)


def probe(record: dict) -> str | None:
    """Why ``record``'s endpoint does not answer as its engine with its
    generation (one exchange), or None when it does."""
    from uuid import uuid4

    from .clock import measure_peer

    try:
        measure_peer(
            endpoint_of(record["endpoint"]),
            record["run_id"],
            str(uuid4()),
            expected_engine=record["engine_id"],
            expected_remote=record["generation"],
            samples=1,
            timeout_s=1,
        )
    except (OSError, ValueError) as error:
        return f"{type(error).__name__}: {error}"
    return None


class ClockRegistry:
    """The registrations one bootstrap server holds, by run and engine.

    The clocks themselves order a replacement, never a timestamp: a
    registration is accepted only if its endpoint answers as its engine with
    its generation; another generation replaces the stored one only if the
    stored clock no longer does, and two live clocks of one engine are a
    conflict (their order cannot be established). A late retry of a retired
    worker therefore never displaces its successor. Identical retries change
    nothing. Registrations are serialized; each verifies at most two clocks.
    """

    def __init__(self, verify=probe):
        self._verify = verify
        self._lock = threading.Lock()  # the records: never held across I/O
        self._serial = threading.Lock()  # one registration at a time
        self._records: dict[tuple[str, str], dict] = {}

    def register(self, record) -> tuple[int, dict]:
        try:
            record = validated(record)
        except (ValueError, TypeError, AttributeError) as error:
            return 400, {"status": "invalid", "detail": str(error)}
        key = (record["run_id"], record["engine_id"])
        with self._serial:
            with self._lock:
                known = self._records.get(key)
                full = known is None and len(self._records) >= MAX_ENGINES
            if known == record:
                return 200, {"status": "unchanged"}
            if full:
                return 503, {"status": "full", "detail": f"{MAX_ENGINES} engines"}
            # the clocks are checked with no lock a listing needs
            if (why := self._verify(record)) is not None:
                return 422, {
                    "status": "unverified",
                    "detail": f"the clock at {record['endpoint']} does not answer "
                    f"as {record['engine_id']} with clock {record['generation']}: "
                    f"{why}",
                }
            if known is not None and self._verify(known) is None:
                return 409, {
                    "status": "conflict",
                    "detail": f"engine {record['engine_id']}'s clock "
                    f"{known['generation']} at {known['endpoint']} still answers: "
                    f"the order of it and {record['generation']} cannot be "
                    "established",
                }
            with self._lock:
                self._records[key] = record
            return (
                (201, {"status": "registered"})
                if known is None
                else (
                    200,
                    {"status": "replaced"},
                )
            )

    def listing(self, run_id: str) -> dict:
        with self._lock:
            return {
                engine: {k: v for k, v in record.items() if k != "run_id"}
                for (run, engine), record in self._records.items()
                if run == run_id
            }


def add_routes(app, registry: ClockRegistry) -> None:
    """Add ``ROUTE`` (POST registers, GET lists one run) to a FastAPI app."""
    from fastapi import Body
    from fastapi.responses import JSONResponse

    # a plain function: FastAPI runs it off the event loop, as its clock
    # verification blocks for up to two exchanges
    def register(record=Body(...)):  # noqa: B008
        code, body = registry.register(record)
        return JSONResponse(body, code)

    async def listing(run_id: str):
        return JSONResponse(registry.listing(run_id))

    app.add_api_route(ROUTE, register, methods=["POST"])
    app.add_api_route(ROUTE, listing, methods=["GET"])


def _no_route(url: str) -> str:
    return (
        f"the bootstrap registry at {url} has no clock route (HTTP 404): an "
        "external or older registry"
    )


def _summary(body: bytes) -> str:
    """A reply body for a saved diagnostic: its JSON with secret-named fields
    and URL credentials redacted, or only its size."""
    try:
        value = json.loads(body)
    except ValueError:
        return f"a {len(body)}-byte non-JSON body"
    return json.dumps(sanitize(value))[:200]


def advertise(base: str, record: dict, deadline_s: float, stop) -> dict:
    """POST ``record`` to ``base`` + ``ROUTE`` until it is accepted, refused,
    ``deadline_s`` passes or ``stop`` (an Event) is set; the report for
    ``profiler.json``. Connection errors and 5xx answers are retried with
    backoff (the registry may start later than this worker); every attempt
    is bounded by the time left, and an answer after the deadline does not
    count."""
    url = base.rstrip("/") + ROUTE
    shown = sanitize(url)
    data = json.dumps(record).encode()
    deadline = time.monotonic() + deadline_s
    delay, attempts, last = 0.1, 0, None
    received = None  # the last HTTP status received, carried into the report

    def report(state, **fields):
        fields.setdefault("status", received)
        return {"state": state, "registry": shown, "attempts": attempts, **fields}

    while (remaining := deadline - time.monotonic()) > 0:
        attempts += 1
        request = urllib.request.Request(
            url, data, {"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=remaining) as response:
                code = received = response.status
                try:
                    body = response.read()
                except (OSError, http.client.HTTPException) as error:
                    return report(
                        "failed",
                        status=code,
                        detail=f"HTTP {code}: the reply was interrupted "
                        f"({type(error).__name__})",
                    )
            if time.monotonic() > deadline:
                last = f"answered after the {deadline_s:g} s window"
                break
            try:
                answer = json.loads(body)
            except ValueError:
                answer = None
            status = answer.get("status") if isinstance(answer, dict) else None
            if status not in ACKNOWLEDGED:
                return report(
                    "failed",
                    status=code,
                    detail=f"HTTP {code}: unrecognized registry reply: "
                    f"{_summary(body)}",
                )
            return report("ok", status=code, answer=status)
        except urllib.error.HTTPError as error:
            received = error.code
            if error.code == 404:
                return report("failed", status=404, detail=sanitize(_no_route(url)))
            if error.code < 500:
                try:
                    body = error.read()
                except (OSError, http.client.HTTPException) as interrupted:
                    return report(
                        "failed",
                        status=error.code,
                        detail=f"HTTP {error.code}: the reply was interrupted "
                        f"({type(interrupted).__name__})",
                    )
                try:
                    answer = json.loads(body)
                except ValueError:
                    answer = None
                detail = answer.get("detail") if isinstance(answer, dict) else None
                why = (
                    sanitize(detail)
                    if isinstance(detail, str)
                    else f"unrecognized reply: {_summary(body)}"
                )
                return report(
                    "failed", status=error.code, detail=f"HTTP {error.code}: {why}"
                )
            last = f"HTTP {error.code}"
        except (OSError, ValueError, http.client.HTTPException) as error:
            # no complete status line: retried like a connection error
            last = f"{type(error).__name__}: {error}"
        if stop.wait(min(delay, max(0.0, deadline - time.monotonic()))):
            return report("pending_at_close", detail=sanitize(last))
        delay = min(delay * 2, 2.0)
    return report(
        "failed", detail=sanitize(f"not accepted within {deadline_s:g} s ({last})")
    )


class NoClockRoute(Exception):
    """A lookup that cannot succeed by retrying (no clock route)."""


def lookup(base: str, run_id: str, timeout_s: float = 1) -> dict:
    """The engines ``base``'s registry advertises for ``run_id``.

    Raises:
        NoClockRoute: the registry has no clock route.
        OSError, ValueError: a transient or malformed answer (retried by the
            caller).

    """
    url = base.rstrip("/") + ROUTE + "?" + urllib.parse.urlencode({"run_id": run_id})
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            listing = json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise NoClockRoute(_no_route(base.rstrip("/") + ROUTE)) from None
        raise
    if not isinstance(listing, dict):
        raise ValueError("a clock listing is a JSON object")
    return listing
