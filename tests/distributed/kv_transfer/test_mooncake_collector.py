# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The out-of-process transport-metrics collector: records, bounds, lifecycle."""

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from vllm.kv_profiling import collector
from vllm.kv_profiling.collector import Collector, Limits, Target
from vllm.kv_profiling.perfetto import _load, export_traces

PAGE = """\
# HELP tent_write_bytes_total Total bytes written via TENT
# TYPE tent_write_bytes_total counter
tent_write_bytes_total{transport="tcp"} 1000
tent_write_bytes_total{transport="rdma"} 500
# TYPE tent_write_requests_total counter
tent_write_requests_total{transport="tcp"} 10
tent_read_bytes_total{transport="tcp"} 20
tent_read_requests_total{transport="tcp"} 2
# TYPE tent_registered_buffer_bytes gauge
tent_registered_buffer_bytes{transport="tcp"} 4096
tent_transport_attempts_total{transport="tcp",operation="write"} 12
tent_transport_attempt_failures_total{transport="tcp",operation="write"} 1
tent_task_failures_total{transport="tcp",reason="timeout"} 1
tent_write_latency_us_bucket{transport="tcp",le="100"} 3
"""


class Target_(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        server = self.server
        server.hits += 1
        if server.delay:
            time.sleep(server.delay)
        if server.status != 200:
            self.send_response(server.status)
            self.end_headers()
            return
        body = server.page().encode() if callable(server.page) else server.page.encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body) + server.short))
        self.end_headers()
        with contextlib.suppress(OSError):
            self.wfile.write(body)

    def log_message(self, *args):
        pass


class Page:
    def __init__(self, page=PAGE, delay=0.0, status=200, short=0):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Target_)
        self.server.page, self.server.delay, self.server.status = page, delay, status
        self.server.short, self.server.hits = short, 0
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def serve():
    servers = []

    def make(*args, **kwargs):
        servers.append(Page(*args, **kwargs))
        return servers[-1]

    yield make
    for server in servers:
        server.close()


def target(server, tid="prefill-0", **extra):
    return Target(
        id=tid,
        role="prefill",
        engine_id="engine-p",
        rank={"tp": 0},
        endpoint=f"127.0.0.1:{server.port}",
        process_id="proc-1",
        pid=4242,
        **extra,
    )


def limits(**changes):
    return Limits(
        **{
            "interval_s": 0.05,
            "scrape_timeout_s": 1,
            "startup_timeout_s": 5,
            "shutdown_timeout_s": 5,
            **changes,
        }
    )


def one_scrape(tmp_path, targets, **changes):
    """Initial scrape, immediate stop, final scrape: the capture's rows and end."""
    stop = threading.Event()
    stop.set()
    summary = collector.run(limits(**changes), targets, tmp_path, str(uuid4()), stop)
    return summary, read(summary_path(tmp_path))


def summary_path(tmp_path):
    (path,) = list(Path(tmp_path).glob("*/*/events.jsonl"))
    return path


def read(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {
        "meta": rows[0],
        "events": [r for r in rows if r["type"] == "event"],
        "end": rows[-1],
    }


def by(events, kind, **labels):
    return [
        e
        for e in events
        if e["kind"] == kind and all(e["labels"].get(k) == v for k, v in labels.items())
    ]


# ------------------------------------------------------------------- the records
def test_label_scopes_are_preserved_and_typed_not_summed(serve, tmp_path):
    summary, cap = one_scrape(tmp_path, [target(serve())])
    events = cap["events"]
    tcp, rdma = (
        by(events, "transport_metrics", transport="tcp")[0],
        by(events, "transport_metrics", transport="rdma")[0],
    )
    assert tcp["fields"] == {
        "write_bytes": 1000.0,
        "write_requests": 10.0,
        "read_bytes": 20.0,
        "read_requests": 2.0,
        "registered_buffer_bytes": 4096.0,
    }
    assert tcp["units"]["registered_buffer_bytes"] == "bytes"
    # Another transport's partition carries only what the page served for it, and no
    # transport-wide value is copied across partitions.
    assert rdma["fields"] == {"write_bytes": 500.0}
    assert sorted(rdma["missing"]) == [
        "read_bytes",
        "read_requests",
        "registered_buffer_bytes",
        "write_requests",
    ]
    attempts = by(
        events, "transport_metrics_failures", transport="tcp", operation="write"
    )[0]
    assert (
        attempts["fields"] == {"attempts": 12.0, "attempt_failures": 1.0}
        and attempts["missing"] == []
    )
    tasks = by(events, "transport_metrics_failures", transport="tcp", reason="timeout")[
        0
    ]
    assert tasks["fields"] == {"task_failures": 1.0} and tasks["labels"] == {
        "transport": "tcp",
        "reason": "timeout",
    }
    assert tasks["missing"] == []
    assert not any("tent_write_latency" in json.dumps(e) for e in events)


def test_both_kinds_of_a_scrape_share_sample_target_and_generation_not_sequence(
    serve, tmp_path
):
    _, cap = one_scrape(tmp_path, [target(serve())])
    samples = {}
    for e in cap["events"]:
        samples.setdefault(e["sample_id"], []).append(e)
    assert len(samples) == 2, "an initial and a final scrape"
    for rows in samples.values():
        assert {e["kind"] for e in rows} == {
            "transport_metrics",
            "transport_metrics_failures",
        }
        assert (
            len(
                {
                    (e["target"]["id"], e["reset_generation"], e["scrape"]["begin_ns"])
                    for e in rows
                }
            )
            == 1
        )
        assert (
            rows[0]["target"]["process_id"] == "proc-1"
            and rows[0]["target"]["pid"] == 4242
        )
        assert rows[0]["scrape"]["end_ns"] >= rows[0]["scrape"]["begin_ns"]
    sequences = [e["sequence"] for e in cap["events"]]
    assert sequences == list(range(1, len(sequences) + 1))
    assert [
        e["kind_id"] for e in cap["events"] if e["kind"] == "transport_metrics"
    ] == [25] * len(by(cap["events"], "transport_metrics"))
    assert {e["kind_id"] for e in by(cap["events"], "transport_metrics_failures")} == {
        26
    }


def test_the_existing_loader_and_exporter_consume_the_capture(serve, tmp_path):
    _, cap = one_scrape(tmp_path, [target(serve())])
    path = summary_path(tmp_path)
    meta, events, decoder, rows, missing = _load(path)
    assert not missing and len(events) == len(cap["events"])
    documents = export_traces([path], str(tmp_path / "trace"))
    (document,) = documents.values()
    trace = json.loads(document.read_text())
    counters = [e for e in trace["traceEvents"] if e.get("ph") == "C"]
    instants = [
        e for e in trace["traceEvents"] if e.get("name", "").startswith("sample")
    ]
    assert len(instants) == len(cap["events"])
    assert any(e["args"].get("write_bytes") == 1000.0 for e in counters)
    assert all(
        isinstance(v, int | float) and not isinstance(v, bool)
        for e in counters
        for v in e["args"].values()
    )
    (capture,) = trace["metadata"]["captures"]
    assert (
        capture["loss"] is False
        and capture["observation"]["observation_complete"] is True
    )


def test_the_terminal_record_says_whether_every_scrape_was_complete(serve, tmp_path):
    summary, cap = one_scrape(tmp_path, [target(serve())])
    assert cap["end"]["complete"] is True and summary["observation_complete"] is True
    assert (
        cap["end"]["stats"]["head"] == cap["end"]["stats"]["tail"] == len(cap["events"])
    )
    assert (
        cap["end"]["stats"]["samples"] == 2
        and cap["end"]["stats"]["partial_samples"] == 0
    )
    assert (
        cap["meta"]["context"]["role"] == "collector"
        and cap["meta"]["capabilities"]["transport_metrics"]
    )


# ------------------------------------------------------- missing, partial and empty
def test_an_empty_page_is_missing_rows_for_both_halves_never_complete_records(
    serve, tmp_path
):
    summary, cap = one_scrape(tmp_path, [target(serve(page=""))])
    assert {e["missing_reason"] for e in cap["events"]} == {"empty_page"}
    assert all(
        e["partial"] and e["fields"] == {} and e["missing"] for e in cap["events"]
    )
    assert cap["end"]["complete"] is False and summary["observation_complete"] is False


def test_a_page_without_one_half_is_unavailable_not_zero_and_not_partial(
    serve, tmp_path
):
    page = "\n".join(
        line
        for line in PAGE.splitlines()
        if "tent_transport" not in line and "task_failures" not in line
    )
    _, cap = one_scrape(tmp_path, [target(serve(page=page))])
    failures = by(cap["events"], "transport_metrics_failures")
    assert len(failures) == 2 and all(
        e["fields"] == {} and e["missing_reason"] == "no_series_served"
        for e in failures
    )
    assert not any(e["partial"] for e in failures)
    assert (
        by(cap["events"], "transport_metrics", transport="tcp")[0]["fields"][
            "write_bytes"
        ]
        == 1000.0
    )


def test_a_page_cut_short_is_a_missing_half_not_a_parsed_prefix(serve, tmp_path):
    summary, cap = one_scrape(tmp_path, [target(serve(short=500))])
    assert {e["missing_reason"] for e in cap["events"]} == {"truncated_response"}
    assert all(e["partial"] and e["fields"] == {} for e in cap["events"])
    assert summary["observation_complete"] is False


def test_a_failing_endpoint_marks_only_its_sample_partial(serve, tmp_path):
    good, bad = serve(), serve(status=503)
    summary, cap = one_scrape(tmp_path, [target(good, "good"), target(bad, "bad")])
    reasons = {
        (e["target"]["id"], e["missing_reason"]) for e in cap["events"] if e["partial"]
    }
    assert reasons == {("bad", "http_503")}
    assert not any(e["partial"] for e in cap["events"] if e["target"]["id"] == "good")
    assert (
        summary["observation_complete"] is False
        and summary["status"]["good"] == "ok"
        and summary["status"]["bad"] == "http_503"
    )


def test_series_with_other_labels_than_the_source_documents_are_not_merged(
    serve, tmp_path
):
    page = (
        PAGE
        + 'tent_write_bytes_total{transport="tcp",operation="write"} 7\n'
        + 'tent_transport_attempts_total{transport="tcp"} 9\n'
    )
    summary, cap = one_scrape(tmp_path, [target(serve(page=page))])
    tcp = by(cap["events"], "transport_metrics", transport="tcp")[0]
    assert tcp["fields"]["write_bytes"] == 1000.0
    assert all(
        e["fields"].get("attempts") != 9.0
        for e in by(cap["events"], "transport_metrics_failures")
    )
    assert (
        "unexpected_labels" in cap["end"]["reasons"]
        and summary["observation_complete"] is False
    )


def test_malformed_and_nonfinite_lines_make_the_sample_partial(serve, tmp_path):
    page = (
        PAGE
        + 'tent_write_bytes_total{transport="tcp" 5\n'
        + 'tent_read_bytes_total{transport="x"} NaN\n'
    )
    summary, cap = one_scrape(tmp_path, [target(serve(page=page))])
    assert "malformed" in cap["end"]["reasons"] and summary["partial_samples"] == 2
    assert all(
        e["partial"] for e in by(cap["events"], "transport_metrics", transport="tcp")
    )


# --------------------------------------------------------------------- reset
def test_a_counter_decrease_is_a_discontinuity_and_a_new_generation_not_a_restart_claim(
    serve, tmp_path
):
    values = iter([1000, 40, 90])
    page = lambda: PAGE.replace(
        'tent_write_bytes_total{transport="tcp"} 1000',
        f'tent_write_bytes_total{{transport="tcp"}} {next(values)}',
    )  # noqa: E731
    server = serve(page=page)
    c = Collector(
        limits(),
        [target(server)],
        collector.Capture(tmp_path, str(uuid4()), limits(), [target(server)]),
    )
    for _ in range(3):
        c.scrape(c.targets[0], time.monotonic() + 2)
    c.capture.end(True, [], {})
    rows = by(
        read(summary_path(tmp_path))["events"], "transport_metrics", transport="tcp"
    )
    assert [
        (r["fields"]["write_bytes"], r["reset_generation"], r["discontinuity"])
        for r in rows
    ] == [(1000.0, 0, False), (40.0, 1, True), (90.0, 1, False)]
    gauges = {r["fields"]["registered_buffer_bytes"] for r in rows}
    assert gauges == {4096.0}
    assert "restart" not in json.dumps(rows)


def test_a_gauge_decrease_is_not_a_reset(serve, tmp_path):
    values = iter([4096, 1024])
    page = lambda: PAGE.replace(
        'tent_registered_buffer_bytes{transport="tcp"} 4096',
        f'tent_registered_buffer_bytes{{transport="tcp"}} {next(values)}',
    )  # noqa: E731
    server = serve(page=page)
    capture = collector.Capture(tmp_path, str(uuid4()), limits(), [target(server)])
    c = Collector(limits(), [target(server)], capture)
    c.scrape(c.targets[0], time.monotonic() + 2)
    c.scrape(c.targets[0], time.monotonic() + 2)
    capture.end(True, [], {})
    rows = by(
        read(summary_path(tmp_path))["events"], "transport_metrics", transport="tcp"
    )
    assert [r["discontinuity"] for r in rows] == [False, False] and [
        r["reset_generation"] for r in rows
    ] == [0, 0]


# -------------------------------------------------------------------- the bounds
def test_the_response_byte_cap_makes_the_half_missing(serve, tmp_path):
    big = PAGE + "".join(f"# filler {i}\n" for i in range(5000))
    summary, cap = one_scrape(
        tmp_path, [target(serve(page=big))], max_response_bytes=2048
    )
    assert {e["missing_reason"] for e in cap["events"]} == {"response_too_large"}
    assert summary["observation_complete"] is False


def test_the_series_and_label_caps_are_counted_never_silent(serve, tmp_path):
    many = PAGE + "".join(f'other_metric{{i="{i}"}} 1\n' for i in range(200))
    summary, cap = one_scrape(tmp_path, [target(serve(page=many))], max_series=50)
    assert (
        "series_cap" in cap["end"]["reasons"]
        and summary["observation_complete"] is False
    )
    long_label = PAGE + 'tent_write_bytes_total{transport="' + "x" * 300 + '"} 1\n'
    _, cap = one_scrape(
        tmp_path / "b", [target(serve(page=long_label))], max_label_bytes=64
    )
    assert "label_cap" in cap["end"]["reasons"]


def test_the_target_cap_refuses_a_manifest(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {"targets": [{"id": f"t{i}", "endpoint": "127.0.0.1:1"} for i in range(5)]}
        )
    )
    with pytest.raises(ValueError, match="max_targets"):
        collector.load_manifest(manifest, limits(max_targets=4))
    assert len(collector.load_manifest(manifest, limits(max_targets=5))) == 5
    manifest.write_text(json.dumps({"targets": [{"id": "a"}, {"id": "a"}]}))
    with pytest.raises(ValueError, match="unique"):
        collector.load_manifest(manifest, limits())


def test_invalid_limits_are_refused():
    for change in (
        {"interval_s": 0},
        {"scrape_timeout_s": float("nan")},
        {"max_series": 0},
        {"max_response_bytes": 10},
        {"max_output_bytes": 100},
        {"shutdown_timeout_s": 10_000},
    ):
        with pytest.raises(ValueError):
            Limits(**change)


def test_retained_output_is_capped_and_the_terminal_record_still_lands(serve, tmp_path):
    stop = threading.Event()
    threading.Timer(0.6, stop.set).start()
    server = serve()
    heavy = Target(
        id="prefill-0",
        role="prefill",
        engine_id="e" * 2000,
        endpoint=f"127.0.0.1:{server.port}",
    )
    summary = collector.run(
        limits(max_output_bytes=1 << 16, interval_s=0.05),
        [heavy],
        tmp_path,
        str(uuid4()),
        stop,
    )
    path = summary_path(tmp_path)
    cap = read(path)
    assert cap["end"]["complete"] is False and "output_cap" in cap["end"]["reasons"]
    assert cap["end"]["stats"]["drop_output_cap"] == summary["dropped_records"] > 0
    assert path.stat().st_size <= 1 << 16 and summary["observation_complete"] is False
    # The loader sees the dropped records as loss, not as a clean capture.
    document = next(iter(export_traces([path], str(tmp_path / "t")).values()))
    assert json.loads(document.read_text())["metadata"]["captures"][0]["loss"] is True


def test_a_slow_target_is_bounded_by_the_scrape_deadline(serve, tmp_path):
    started = time.monotonic()
    summary, cap = one_scrape(
        tmp_path,
        [target(serve(delay=3))],
        scrape_timeout_s=0.3,
        startup_timeout_s=1,
        shutdown_timeout_s=1,
    )
    assert time.monotonic() - started < 3
    assert {e["missing_reason"] for e in cap["events"]} <= {
        "timeout",
        "scrape_deadline",
        "TimeoutError",
        "deadline_exhausted",
    }
    assert summary["observation_complete"] is False


def test_the_final_scrape_shares_one_shutdown_budget_across_targets(serve, tmp_path):
    slow = [serve(delay=2) for _ in range(4)]
    started = time.monotonic()
    summary, cap = one_scrape(
        tmp_path,
        [target(s, f"t{i}") for i, s in enumerate(slow)],
        scrape_timeout_s=1,
        startup_timeout_s=0.8,
        shutdown_timeout_s=0.8,
    )
    # Each pass stays inside its own budget, not 4 targets x timeout each.
    assert time.monotonic() - started < 4
    assert summary["observation_complete"] is False
    assert (
        "pass_budget_exhausted" in cap["end"]["reasons"]
        or "final_scrape_incomplete" in cap["end"]["reasons"]
    )


# ------------------------------------------------- deadlines against slow peers
def trickle_server(head, body, head_gap, body_gap):
    """A raw TCP peer answering any request one byte at a time (the head, then the
    body), each byte ``*_gap`` seconds after the last."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    stop = threading.Event()

    def serve_one(connection):
        with contextlib.suppress(OSError):
            connection.recv(4096)
            for data, gap in ((head, head_gap), (body, body_gap)):
                for byte in data:
                    if stop.is_set():
                        return
                    connection.sendall(bytes([byte]))
                    time.sleep(gap)
        connection.close()

    def accept():
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except OSError:
                return
            threading.Thread(target=serve_one, args=(connection,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()

    def close():
        stop.set()
        listener.close()

    return listener.getsockname()[1], close


@pytest.mark.parametrize("where", ["headers", "body"])
def test_a_peer_trickling_bytes_cannot_hold_a_scrape_past_its_deadline(where):
    head = b"HTTP/1.1 200 OK\r\nContent-Length: 4000\r\n\r\n"
    port, close = trickle_server(
        head, b"x" * 4000, 0.025 if where == "headers" else 0, 0.025
    )
    try:
        started = time.monotonic()
        with pytest.raises(collector.ScrapeError) as error:
            collector.fetch(
                "127.0.0.1", port, "/metrics", time.monotonic() + 0.08, Limits()
            )
        assert error.value.reason == "scrape_deadline"
        assert time.monotonic() - started < 0.3
    finally:
        close()


def test_stopping_during_a_scrape_bounds_it_by_the_shutdown_budget(tmp_path):
    port, close = trickle_server(
        b"HTTP/1.1 200 OK\r\nContent-Length: 4000\r\n\r\n", b"x" * 4000, 0, 0.02
    )
    try:
        stop = threading.Event()
        started = {}

        def stopper():
            time.sleep(0.15)
            started["stop"] = time.monotonic()
            stop.set()

        slow = Target(id="slow", endpoint=f"127.0.0.1:{port}")
        threading.Thread(target=stopper, daemon=True).start()
        summary = collector.run(
            limits(scrape_timeout_s=30, interval_s=0.05, shutdown_timeout_s=0.5),
            [slow],
            tmp_path,
            str(uuid4()),
            stop,
        )
        after_stop = time.monotonic() - started["stop"]
    finally:
        close()
    # The scrape in progress and the final pass share one 0.5 s budget from the stop.
    assert after_stop < 1.0, after_stop
    assert summary["observation_complete"] is False
    cap = read(summary_path(tmp_path))
    assert {e["missing_reason"] for e in cap["events"]} == {"scrape_deadline"}


def test_the_final_pass_gets_what_a_scrape_in_progress_left_of_the_budget(
    serve, tmp_path
):
    server = serve(delay=0.4)
    stop = threading.Event()
    threading.Thread(target=lambda: (time.sleep(0.1), stop.set()), daemon=True).start()
    started = time.monotonic()
    summary = collector.run(
        limits(scrape_timeout_s=5, interval_s=0.05, shutdown_timeout_s=0.7),
        [target(server)],
        tmp_path,
        str(uuid4()),
        stop,
    )
    # The initial scrape (0.4 s) is in progress at the stop; the final one cannot
    # take another 0.4 s inside the 0.7 s from the stop, so it is incomplete.
    assert time.monotonic() - started < 1.5
    assert summary["observation_complete"] is False
    assert "final_scrape_incomplete" in read(summary_path(tmp_path))["end"]["reasons"]


# ------------------------------------- connect, send and resolve inside the deadline
@contextlib.contextmanager
def stalled_listener():
    """A loopback listener with a full accept queue: connects never complete."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(0)
    port = listener.getsockname()[1]
    fillers = []
    try:
        for _ in range(8):
            filler = socket.socket()
            filler.setblocking(False)
            filler.connect_ex(("127.0.0.1", port))
            fillers.append(filler)
        time.sleep(0.1)
        probe = socket.socket()
        probe.settimeout(0.3)
        try:
            probe.connect(("127.0.0.1", port))
        except OSError:
            pass
        else:
            pytest.skip("this kernel completes connects to a full accept queue")
        finally:
            probe.close()
        yield port
    finally:
        for filler in fillers:
            filler.close()
        listener.close()


def test_a_connection_that_never_completes_is_bounded_by_the_deadline():
    with stalled_listener() as port:
        started = time.monotonic()
        with pytest.raises(collector.ScrapeError) as error:
            collector.fetch(
                "127.0.0.1", port, "/metrics", time.monotonic() + 0.15, Limits()
            )
        assert error.value.reason == "scrape_deadline"
        assert time.monotonic() - started < 0.4


def test_stopping_during_a_stalled_connect_ends_inside_the_shutdown_budget(tmp_path):
    with stalled_listener() as port:
        stop = threading.Event()
        stopped = {}

        def stopper():
            time.sleep(0.03)
            stopped["at"] = time.monotonic()
            stop.set()

        threading.Thread(target=stopper, daemon=True).start()
        summary = collector.run(
            limits(scrape_timeout_s=0.65, interval_s=0.05, shutdown_timeout_s=0.08),
            [Target(id="stalled", endpoint=f"127.0.0.1:{port}")],
            tmp_path,
            str(uuid4()),
            stop,
        )
        after_stop = time.monotonic() - stopped["at"]
    # The 0.65 s scrape deadline would have run to 0.65 s: stop moves it to the 80 ms
    # shutdown budget (plus polling slices), and the final pass has nothing left.
    assert after_stop < 0.35, after_stop
    assert summary["observation_complete"] is False
    cap = read(summary_path(tmp_path))
    assert {e["missing_reason"] for e in cap["events"]} <= {"scrape_deadline"}


def test_sending_a_request_to_a_peer_that_never_reads_is_bounded():
    a, b = socket.socketpair()
    try:
        for end in (a, b):
            end.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        sock = collector._DeadlineSocket(fileno=a.detach())
        started = time.monotonic()
        sock.deadline = lambda: started + 0.2
        with pytest.raises(TimeoutError):
            sock.sendall(b"x" * (64 << 20))
        assert time.monotonic() - started < 0.6
        sock.close()
    finally:
        b.close()


def test_names_are_never_resolved_so_no_resolver_work_can_outlive_scrapes(tmp_path):
    started_threads = threading.active_count()

    def forbidden(*args, **kwargs):
        raise AssertionError("no name may be resolved")

    with patch.object(socket, "getaddrinfo", forbidden):
        # Directly: refused at once, no helper thread.
        with pytest.raises(collector.ScrapeError) as error:
            collector.fetch(
                "metrics.example.invalid",
                9100,
                "/metrics",
                time.monotonic() + 5,
                Limits(),
            )
        assert error.value.reason == "hostname_unsupported"
        # Through the collector: the target is unavailable with the reason, once per
        # scrape, and repeated scrapes add no thread, however many.
        named = Target(id="named", endpoint="metrics.example.invalid:9100")
        c = Collector(
            limits(),
            [named],
            collector.Capture(tmp_path, str(uuid4()), limits(), [named]),
        )
        assert "IP literal" in c.resolved["named"][2]
        for _ in range(50):
            assert c.scrape(named, time.monotonic() + 1) is False
            assert threading.active_count() <= started_threads
    c.capture.end(False, c.reasons, {})
    cap = read(summary_path(tmp_path))
    assert {e["missing_reason"] for e in cap["events"]} == {"unavailable"}
    assert "IP literal" in c.unavailable["named"]
    # IP literals, v4 and v6, are addressed without resolution.
    assert collector._addresses("10.1.2.3", 9100) == [
        (socket.AF_INET, ("10.1.2.3", 9100))
    ]
    assert collector._addresses("[::1]", 9100)[0][0] == socket.AF_INET6
    # Hostnames from a target's own configuration are unavailable the same way.
    conf = Target(
        id="c",
        mc_tent_conf=json.dumps(
            {"metrics": {"http_port": 9201, "http_host": "tent.local"}}
        ),
    )
    assert collector.resolve_targets([conf])["c"][0] is None


# ------------------------------------------------- one generation per target sample
def test_a_reset_across_kinds_and_transports_has_one_generation_per_sample(
    serve, tmp_path
):
    pages = iter(
        [
            PAGE.replace('{transport="rdma"} 500', '{transport="rdma"} 900'),
            # Everything restarted: bytes of both transports, attempts, task failures.
            PAGE.replace('{transport="tcp"} 1000', '{transport="tcp"} 5')
            .replace('{transport="rdma"} 500', '{transport="rdma"} 7')
            .replace('operation="write"} 12', 'operation="write"} 1')
            .replace('reason="timeout"} 1', 'reason="timeout"} 0'),
            PAGE.replace('{transport="rdma"} 500', '{transport="rdma"} 9'),
        ]
    )
    server = serve(page=lambda: next(pages))
    c = Collector(
        limits(),
        [target(server)],
        collector.Capture(tmp_path, str(uuid4()), limits(), [target(server)]),
    )
    for _ in range(3):
        c.scrape(c.targets[0], time.monotonic() + 2)
    c.capture.end(True, [], {})
    events = read(summary_path(tmp_path))["events"]
    samples = {}
    for e in events:
        samples.setdefault(e["sample_id"], []).append(e)
    generations = [{e["reset_generation"] for e in rows} for rows in samples.values()]
    assert generations == [{0}, {1}, {1}], generations
    second = list(samples.values())[1]
    flagged = {
        (e["kind"], tuple(sorted(e["labels"].items())))
        for e in second
        if e["discontinuity"]
    }
    assert ("transport_metrics", (("transport", "tcp"),)) in flagged
    assert ("transport_metrics", (("transport", "rdma"),)) in flagged
    assert (
        "transport_metrics_failures",
        (("operation", "write"), ("transport", "tcp")),
    ) in flagged


# ------------------------------------------------------------- bounded history
def test_counter_history_is_bounded_and_a_reappearing_series_has_no_baseline(
    serve, tmp_path
):
    counter = iter(range(1, 1000))

    def page():
        n = next(counter)
        return (
            f'tent_task_failures_total{{transport="tcp",reason="r{n}"}} {n}\n'
            'tent_write_bytes_total{transport="tcp"} 10\n'
        )

    server = serve(page=page)
    bound = limits(max_history=6, max_output_bytes=1 << 16)
    c = Collector(
        bound,
        [target(server)],
        collector.Capture(tmp_path, str(uuid4()), bound, [target(server)]),
    )
    for _ in range(300):
        c.scrape(c.targets[0], time.monotonic() + 2)
        assert len(c.last) <= 6
    assert c.capture.dropped > 0, "the output cap was reached; growth must stop anyway"
    assert c.evicted_total > 250 and "history_evicted" in c.reasons
    c.capture.end(False, c.reasons, {})
    # The oldest series was evicted long ago: seen again it has no assumed baseline.
    rows = c._partitions(
        "prefill-0",
        {
            ("tent_task_failures_total", (("reason", "r1"), ("transport", "tcp"))): 1.0,
        },
        collector._FAILURE_FAMILIES,
    )
    assert rows[0][5] == "unavailable"


def test_a_series_kept_in_history_stays_an_established_baseline(serve, tmp_path):
    server = serve()
    c = Collector(
        limits(max_history=64),
        [target(server)],
        collector.Capture(tmp_path, str(uuid4()), limits(), [target(server)]),
    )
    for _ in range(2):
        c.scrape(c.targets[0], time.monotonic() + 2)
    c.capture.end(True, [], {})
    rows = read(summary_path(tmp_path))["events"]
    first, second = rows[: len(rows) // 2], rows[len(rows) // 2 :]
    assert {e["baseline"] for e in first if e["fields"]} == {"first"}
    assert {e["baseline"] for e in second if e["fields"]} == {"established"}
    assert not c.evicted_total


# ------------------------------------------------------- counter tracks in a trace
def test_two_endpoints_with_identical_labels_have_separate_counter_tracks(
    serve, tmp_path
):
    one = serve(page='tent_write_bytes_total{transport="tcp"} 100\n')
    two = serve(page='tent_write_bytes_total{transport="tcp"} 900\n')
    one_scrape(tmp_path, [target(one, "a"), target(two, "b")])
    (document,) = export_traces(
        [summary_path(tmp_path)], str(tmp_path / "trace")
    ).values()
    trace = json.loads(document.read_text())
    counters = [
        e
        for e in trace["traceEvents"]
        if e.get("ph") == "C" and "write_bytes" in e["args"]
    ]
    tracks = {}
    for e in counters:
        tracks.setdefault(e["name"], set()).add(e["args"]["write_bytes"])
    assert sorted(tracks.values(), key=min) == [{100.0}, {900.0}], tracks
    assert all(
        isinstance(v, int | float) and not isinstance(v, bool)
        for e in counters
        for v in e["args"].values()
    )
    instant = next(
        e for e in trace["traceEvents"] if e.get("name", "").startswith("sample")
    )
    assert instant["args"]["target"]["id"] in ("a", "b")
    assert "sample_id" in instant["args"]


def test_a_counter_reset_starts_a_new_counter_track_in_the_trace(serve, tmp_path):
    values = iter([1000, 40])
    page = lambda: PAGE.replace(  # noqa: E731
        'tent_write_bytes_total{transport="tcp"} 1000',
        f'tent_write_bytes_total{{transport="tcp"}} {next(values)}',
    )
    one_scrape(tmp_path, [target(serve(page=page))])
    (document,) = export_traces(
        [summary_path(tmp_path)], str(tmp_path / "trace")
    ).values()
    trace = json.loads(document.read_text())
    names = {
        e["name"]
        for e in trace["traceEvents"]
        if e.get("ph") == "C" and "write_bytes" in e["args"] and '"tcp"' in e["name"]
    }
    assert len(names) == 2 and any(n.endswith("reset 1") for n in names)


# ------------------------------------------------------------------- lifecycle
def test_start_periodic_final_scrape_then_the_terminal_record(serve, tmp_path):
    server = serve()
    stop = threading.Event()
    done = []
    thread = threading.Thread(
        target=lambda: done.append(
            collector.run(
                limits(interval_s=0.1), [target(server)], tmp_path, str(uuid4()), stop
            )
        )
    )
    thread.start()
    time.sleep(0.5)
    hits_before_stop = server.server.hits
    assert hits_before_stop >= 3, "the initial scrape and periodic ones"
    stop.set()
    thread.join(10)
    assert done and server.server.hits > hits_before_stop, (
        "a final scrape after the stop"
    )
    cap = read(summary_path(tmp_path))
    ends = [e["scrape"]["end_ns"] for e in cap["events"]]
    assert ends == sorted(ends) and cap["end"]["complete"] is True
    last_sample = cap["events"][-1]["sample_id"]
    assert int(last_sample.rsplit("-", 1)[1]) == cap["end"]["stats"]["samples"]


def test_the_final_scrape_needs_the_endpoint_to_outlive_the_collector(serve, tmp_path):
    server = serve()
    stop = threading.Event()
    summaries = []
    thread = threading.Thread(
        target=lambda: summaries.append(
            collector.run(
                limits(interval_s=0.1), [target(server)], tmp_path, str(uuid4()), stop
            )
        )
    )
    thread.start()
    time.sleep(0.4)
    server.close()  # the engine's metrics server went away before the final scrape
    stop.set()
    thread.join(10)
    cap = read(summary_path(tmp_path))
    assert (
        summaries[0]["observation_complete"] is False
        and cap["end"]["complete"] is False
    )
    assert cap["events"][-1]["partial"] is True


def test_without_a_final_scrape_the_capture_says_so(serve, tmp_path):
    stop = threading.Event()
    stop.set()
    summary = collector.run(
        limits(), [target(serve())], tmp_path, str(uuid4()), stop, final_scrape=False
    )
    cap = read(summary_path(tmp_path))
    assert (
        "no_final_scrape" in cap["end"]["reasons"]
        and summary["observation_complete"] is False
    )


def test_the_collector_cli_stops_on_a_file_or_a_signal_and_exits_by_completeness(
    serve, tmp_path
):
    server = serve()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "id": "p0",
                        "role": "prefill",
                        "endpoint": f"127.0.0.1:{server.port}",
                    }
                ]
            }
        )
    )
    stop_file = tmp_path / "stop"
    run_id = str(uuid4())
    base = [
        sys.executable,
        "-I",
        collector.__file__,
        "--run-id",
        run_id,
        "--output-dir",
        str(tmp_path / "out"),
        "--manifest",
        str(manifest),
        "--interval-s",
        "0.1",
        "--shutdown-timeout-s",
        "3",
    ]
    proc = subprocess.Popen(
        [*base, "--stop-file", str(stop_file)], stdout=subprocess.PIPE, text=True
    )
    time.sleep(0.6)
    stop_file.touch()
    out, _ = proc.communicate(timeout=20)
    assert proc.returncode == 0 and json.loads(out)["observation_complete"] is True
    # A SIGTERM is the same stop; a target that never answers makes the exit code 3.
    dead = tmp_path / "dead.json"
    dead.write_text(
        json.dumps({"targets": [{"id": "gone", "endpoint": "127.0.0.1:1"}]})
    )
    proc = subprocess.Popen(
        [
            *base[:-4],
            "--manifest",
            str(dead),
            "--interval-s",
            "0.1",
            "--startup-timeout-s",
            "1",
            "--shutdown-timeout-s",
            "1",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.8)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=20)
    assert proc.returncode == 3 and json.loads(out)["observation_complete"] is False


def test_the_supported_launch_path_imports_nothing_but_the_standard_library(
    serve, tmp_path
):
    """The file run by path in an isolated interpreter without site-packages (no vLLM,
    no Torch, no package stubs), once for ``--help`` and once for a real scrape."""
    help_ = subprocess.run(
        [sys.executable, "-I", "-S", collector.__file__, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert help_.returncode == 0 and "--manifest" in help_.stdout, help_.stderr
    server = serve()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"targets": [{"id": "p0", "endpoint": f"127.0.0.1:{server.port}"}]})
    )
    stop_file = tmp_path / "stop"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            collector.__file__,
            "--run-id",
            str(uuid4()),
            "--output-dir",
            str(tmp_path / "out"),
            "--manifest",
            str(manifest),
            "--interval-s",
            "0.05",
            "--shutdown-timeout-s",
            "3",
            "--stop-file",
            str(stop_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.5)
    stop_file.touch()
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 0, err
    summary = json.loads(out)
    assert summary["observation_complete"] is True and summary["samples"] >= 2


# ----------------------------------------------------------------- endpoints
def test_endpoint_precedence_manifest_then_the_targets_conf_then_its_env_then_default(
    monkeypatch,
):
    monkeypatch.setenv(
        "TENT_METRICS_HTTP_PORT", "1111"
    )  # the collector's own: never read
    monkeypatch.setenv("MC_TENT_CONF", json.dumps({"metrics": {"http_port": 2222}}))
    resolve = collector.resolve_endpoint
    assert resolve(Target(id="a", endpoint="10.0.0.5:7000"))[:3] == (
        "10.0.0.5",
        7000,
        "manifest",
    )
    nested = {
        "metrics": {"http_port": 9201, "http_host": "10.0.0.9"},
        "metrics/http_port": 9202,
    }
    assert resolve(Target(id="a", mc_tent_conf=json.dumps(nested)))[:3] == (
        "10.0.0.9",
        9201,
        "target MC_TENT_CONF",
    )
    assert (
        resolve(Target(id="a", mc_tent_conf=json.dumps({"metrics/http_port": 9202})))[1]
        == 9202
    )
    assert (
        resolve(
            Target(
                id="a",
                env={"MC_TENT_CONF": json.dumps({"metrics": {"http_port": 9203}})},
            )
        )[1]
        == 9203
    )
    assert resolve(Target(id="a", env={"TENT_METRICS_HTTP_PORT": "9300"}))[:3] == (
        "127.0.0.1",
        9300,
        "target environment",
    )
    assert resolve(Target(id="a"))[:3] == ("127.0.0.1", 9100, "default")
    assert (
        resolve(Target(id="a", env={"TENT_METRICS_HTTP_PORT": "9300"}, port_offset=3))[
            1
        ]
        == 9303
    )


def test_unreadable_or_invalid_ownership_is_unavailable_never_a_guess(tmp_path):
    resolve = collector.resolve_endpoint
    for target_ in (
        Target(id="a", mc_tent_conf="{not json"),
        Target(id="a", mc_tent_conf=str(tmp_path / "missing.json")),
        Target(id="a", mc_tent_conf="[1, 2]"),
        Target(id="a", env={"TENT_METRICS_HTTP_PORT": "x"}),
        Target(id="a", endpoint="nohost"),
        Target(id="a", endpoint="h:99999"),
        Target(id="a", env={"TENT_METRICS_HTTP_PORT": "65535"}, port_offset=5),
    ):
        host, port, reason = resolve(target_)
        assert host is None and port is None and reason
    both = collector.resolve_targets(
        [
            Target(id="x", endpoint="127.0.0.1:9100"),
            Target(id="y", endpoint="127.0.0.1:9100"),
            Target(id="z", endpoint="127.0.0.1:9101"),
        ]
    )
    assert (
        both["x"][0] is None
        and "ambiguous" in both["x"][2]
        and both["y"][0] is None
        and both["z"][1] == 9101
    )


def test_an_unavailable_target_is_a_gap_with_a_reason_in_the_capture(serve, tmp_path):
    summary, cap = one_scrape(tmp_path, [Target(id="orphan", mc_tent_conf="{broken")])
    assert {e["missing_reason"] for e in cap["events"]} == {"unavailable"}
    assert summary["unavailable"] == {
        "orphan": "the target's MC_TENT_CONF is unreadable"
    }
    assert (
        cap["events"][0]["target"]["endpoint"] is None
        and cap["end"]["complete"] is False
    )


def test_the_conf_lookup_equals_the_connectors_own(tmp_path):
    pytest.importorskip("torch")
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import tent_config

    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    confs = [
        {"metrics": {"http_port": 1}},
        {"metrics/http_port": 2},
        {"metrics": {"http_port": 1}, "metrics/http_port": 2},
        {"metrics": {"http_port": "x"}, "metrics/http_port": 3},
        {"metrics": {"http_port": True}},
        {"metrics": 5},
        {},
    ]
    for conf in confs:
        found = tent_config.lookup(conf, "metrics/http_port", number)
        mine = collector._lookup(conf, "metrics/http_port", collector._port_value)
        assert (None if found is None else int(found[1])) == (
            None if mine is None else int(mine)
        ), conf


def test_the_collector_imports_only_the_standard_library():
    source = Path(collector.__file__).read_text()
    imports = {
        line.split()[1].split(".")[0]
        for line in source.splitlines()
        if line.startswith(("import ", "from "))
    }
    allowed = (
        set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else imports
    )
    assert imports <= allowed, imports - allowed
    assert os.path.basename(collector.__file__) == "collector.py"


def test_a_kernel_without_time_namespaces_keeps_the_boot_id(tmp_path, monkeypatch):
    """A missing /proc/self/ns/time is None on its own, never a lost boot ID."""
    real_readlink = os.readlink

    def readlink(path, *args, **kwargs):
        if str(path) == "/proc/self/ns/time":
            raise FileNotFoundError("time namespaces not configured")
        return real_readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", readlink)
    capture = collector.Capture(tmp_path, str(uuid4()), limits(), [])
    capture.end(True, [], {})
    meta = read(summary_path(tmp_path))["meta"]
    assert meta["time_namespace"] is None
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    assert meta["boot_id"] == boot_id
