# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract: UCX's protocol selection tables in a server log become per-process,
per-endpoint-configuration records of the protocol and lanes for each message
size, and nothing in them is silently lost.

Failures guarded: tables of processes that share a log mixed together, lanes
or their shares misread, a table that does not parse dropped without a trace.
The fixtures are tables as UCX 1.22 renders them (``UCX_PROTO_INFO=y`` and
``=used``)."""

import itertools
import json

from vllm.kv_profiling import ucx_protocols


def _table(pid, time, title, operation, header, rows, thread=0):
    prefix = f"[{time}] [node0:{pid:<5}:{thread}]   "
    width = 70
    lines = [
        "+" + "-" * width + "+",
        f"| {title} |",
        f"| {operation} |",
        "+" + "-" * width + "+",
        "| " + " | ".join(header) + " |",
        "+" + "-" * width + "+",
        *("| " + " | ".join(row) + " |" for row in rows),
        "+" + "-" * width + "+",
    ]
    return [prefix + line for line in lines]


GET = "remote memory read by ucp_get*(multi) into host memory from host"
PUT = "remote memory write by ucp_put*(multi) from host memory to host"


def test_tables_of_interleaved_processes_are_kept_apart():
    """Two processes writing one log: each table is grouped by its own prefix."""
    first = _table(7, "1700000000.000001", "vllm inter-node cfg#0", GET,
                   ["Range", "Description", "Config"],
                   [["0..inf", "zero-copy", "tcp/kv0"]])  # fmt: skip
    second = _table(9, "1700000000.000001", "vllm inter-node cfg#2", PUT,
                    ["Range", "Description", "Config"],
                    [["0..8K", "short", "tcp/kv1"],
                     ["8K..inf", "zero-copy", "tcp/kv1"]])  # fmt: skip
    pairs = itertools.zip_longest(first, second)
    mixed = [line for pair in pairs for line in pair if line is not None]
    mixed.insert(3, "INFO 11-20 an unrelated vLLM log line")
    # Another writer's partial line, flushed just before a table's first line.
    mixed[0] = "INFO 11-20 a partial line" + mixed[0]
    result = ucx_protocols.parse_lines(mixed)
    assert result["unparsed"] == 0
    by_pid = {table["pid"]: table for table in result["tables"]}
    assert (by_pid[7]["ep_config"], by_pid[7]["api"]) == (0, "ucp_get*")
    assert by_pid[9]["context"] == "vllm" and by_pid[9]["locality"] == "inter-node"
    assert [(r["start"], r["end"], r["description"]) for r in by_pid[9]["ranges"]] == [
        (0, 8192, "short"),
        (8192, None, "zero-copy"),
    ]


def test_lanes_and_their_shares_are_read_as_ucx_prints_them():
    assert ucx_protocols.lanes("tcp/eth0") == [
        {"transport": "tcp", "device": "eth0", "path": None, "share_percent": 100}
    ]
    assert ucx_protocols.lanes("30% on rc/d0:1, 40% on rc/d1:1 and 30% on tcp/e0") == [
        {"transport": "rc", "device": "d0:1", "path": None, "share_percent": 30},
        {"transport": "rc", "device": "d1:1", "path": None, "share_percent": 40},
        {"transport": "tcp", "device": "e0", "path": None, "share_percent": 30},
    ]
    # Lanes of one device on several paths: the device once, then each path.
    assert ucx_protocols.lanes("rc/d0:1 50% on path0 and 50% on path1") == [
        {"transport": "rc", "device": "d0:1", "path": 0, "share_percent": 50},
        {"transport": "rc", "device": "d0:1", "path": 1, "share_percent": 50},
    ]
    assert ucx_protocols.lanes("rc/d0:1/path1")[0]["path"] == 1
    assert ucx_protocols.lanes("no lane named") == []


def test_selection_counts_estimates_and_two_tables_under_one_prefix():
    """``UCX_PROTO_INFO=used`` adds a Count column; ``(?)`` marks an estimate;
    two tables written in the same microsecond by one thread are both kept."""
    used = _table(5, "1700000000.5", "inter-node cfg#1", GET,
                  ["Count", "Range", "Description", "Config"],
                  [["305", "0..4K", "(?) copy-in", "tcp/kv0"],
                   ["9", "4097..inf", "zero-copy",
                    "50% on tcp/kv0 and 50% on tcp/kv1"]])  # fmt: skip
    again = _table(5, "1700000000.5", "inter-node cfg#1", PUT,
                   ["Count", "Range", "Description", "Config"],
                   [["1", "0..inf", "zero-copy", "tcp/kv0"]])  # fmt: skip
    result = ucx_protocols.parse_lines(used + again)
    assert result["unparsed"] == 0 and len(result["tables"]) == 2
    table = result["tables"][0]
    assert table["context"] is None and table["selection_counts"] is True
    copy_in, zero_copy = table["ranges"]
    assert (copy_in["selections"], copy_in["estimated"], copy_in["end"]) == (
        305, True, 4096)  # fmt: skip
    assert copy_in["description"] == "copy-in"
    assert [lane["device"] for lane in zero_copy["lanes"]] == ["kv0", "kv1"]
    assert result["tables"][1]["api"] == "ucp_put*"


def test_a_table_that_does_not_parse_is_counted_and_the_others_kept():
    broken = _table(3, "1700000000.1", "vllm inter-node cfg#0", GET,
                    ["Range", "Protocol", "Lanes"],
                    [["0..inf", "zero-copy", "tcp/kv0"]])  # fmt: skip
    good = _table(3, "1700000000.2", "vllm inter-node cfg#0", PUT,
                  ["Range", "Description", "Config"],
                  [["0..inf", "zero-copy", "tcp/kv0"]])  # fmt: skip
    result = ucx_protocols.parse_lines(broken + good[:-1] + good[-1:])
    assert len(result["tables"]) == 1 and result["unparsed"] == 1
    assert "header" in result["unparsed_examples"][0]["reason"]
    truncated = ucx_protocols.parse_lines(good[:-1])  # the closing line missing
    assert truncated["tables"] == [] and truncated["unparsed"] == 1


def test_the_cli_joins_each_process_to_its_recorder(tmp_path):
    log = tmp_path / "decode.log"
    log.write_text(
        "\n".join(
            _table(42, "1700000000.000001", "vllm inter-node cfg#3", GET,
                   ["Range", "Description", "Config"],
                   [["0..inf", "zero-copy", "tcp/kv0"]])  # fmt: skip
        )
        + "\n"
    )
    recorder = tmp_path / "profiles" / "run" / "proc-1"
    recorder.mkdir(parents=True)
    (recorder / "events.jsonl").write_text(
        json.dumps(
            {"pid": 42, "process_id": "proc-1",
             "context": {"role": "worker", "rank": 1, "engine_id": "e"}}  # fmt: skip
        )
        + "\n"
    )
    out = tmp_path / "ucx.json"
    assert ucx_protocols.main(
        [str(log), "--profiles", str(tmp_path / "profiles"), "--output", str(out)]
    ) == 0  # fmt: skip
    result = json.loads(out.read_text())
    (endpoint,) = result["endpoint_configurations"]
    assert endpoint["recorder"] == {"run_id": None, "process_id": "proc-1",
                                    "role": "worker", "rank": 1,
                                    "engine_id": "e"}  # fmt: skip
    assert (endpoint["ep_config"], endpoint["pid"]) == (3, 42)
    assert result["tables"] == 1 and result["unparsed"] == 0



def _write_recorder(root, run, process, pid, rank):
    directory = root / run / process
    directory.mkdir(parents=True)
    (directory / "events.jsonl").write_text(
        json.dumps({"pid": pid, "run_id": run, "process_id": process,
                    "context": {"role": "worker", "rank": rank}}) + "\n"  # fmt: skip
    )


def _write_log(path, *tables):
    path.write_text("\n".join(line for table in tables for line in table) + "\n")


HEADER = ["Range", "Description", "Config"]


def test_a_damaged_table_is_counted_once_and_the_next_one_is_kept():
    """Under one prefix (same microsecond), a damaged count or a missing closing
    border loses only that table: the following one is parsed."""
    get = _table(4, "1700000000.3", "vllm inter-node cfg#1", GET, HEADER,
                 [["0..inf", "zero-copy", "tcp/kv0"]])  # fmt: skip
    damaged = _table(4, "1700000000.3", "vllm inter-node cfg#1", PUT,
                     ["Count", *HEADER],
                     [["x9", "0..inf", "zero-copy", "tcp/kv0"]])  # fmt: skip
    unclosed = _table(4, "1700000000.3", "vllm inter-node cfg#1", PUT, HEADER,
                      [["0..inf", "zero-copy", "tcp/kv0"]])[:-1]  # fmt: skip
    for first in (damaged, unclosed):
        result = ucx_protocols.parse_lines(first + get)
        assert result["unparsed"] == 1, result
        assert [t["api"] for t in result["tables"]] == ["ucp_get*"]


def test_every_printed_thread_and_worker_label_is_kept(tmp_path):
    """UCX prints its async thread as ``a``; cfg#N is numbered per UCP worker,
    so one process with two workers prints cfg#0 twice: the locality and the
    thread of each table are kept, and such groupings are marked ambiguous."""
    asynchronous = _table(6, "1700000000.4", "vllm inter-node cfg#0", GET, HEADER,
                          [["0..inf", "zero-copy", "tcp/kv0"]], thread="a")  # fmt: skip
    assert ucx_protocols.parse_lines(asynchronous)["tables"][0]["ucs_thread"] == "a"
    own = _table(6, "1700000000.5", "vllm self cfg#0", PUT, HEADER,
                 [["0..inf", "zero-copy", "self/memory"]])  # fmt: skip
    _write_log(tmp_path / "two-workers.log", asynchronous, own)
    result = ucx_protocols.summarize([tmp_path / "two-workers.log"])
    groups = {e["locality"]: e for e in result["endpoint_configurations"]}
    assert groups["inter-node"]["operations"][0]["api"] == "ucp_get*"
    assert groups["self"]["operations"][0]["api"] == "ucp_put*"
    assert all(e["worker_collision_detected"] for e in groups.values())
    assert result["worker_identity"].startswith("unresolved")
    assert groups["inter-node"]["operations"][0]["ucs_thread"] == "a"
    # One operation selected twice under one label: two workers, ambiguous.
    twice = _table(6, "1700000000.6", "vllm inter-node cfg#0", GET, HEADER,
                   [["0..inf", "zero-copy", "tcp/kv1"]])  # fmt: skip
    _write_log(tmp_path / "twice.log", asynchronous, twice)
    (group,) = ucx_protocols.summarize([tmp_path / "twice.log"])[
        "endpoint_configurations"
    ]
    assert group["worker_collision_detected"] and len(group["operations"]) == 2
    _write_log(tmp_path / "one.log", asynchronous)
    (single,) = ucx_protocols.summarize([tmp_path / "one.log"])[
        "endpoint_configurations"
    ]
    assert single["worker_collision_detected"] is False


def test_a_reused_pid_joins_only_the_selected_run(tmp_path):
    """Two runs on one node reused PID 82: without a run the match is
    ambiguous and stays unjoined; selecting the run, or passing its own
    directory, joins the right recorder."""
    profiles = tmp_path / "profiles"
    _write_recorder(profiles, "older", "proc-old", 82, 1)
    _write_recorder(profiles, "later", "proc-new", 82, 0)
    log = tmp_path / "decode.log"
    _write_log(log, _table(82, "1700000000.7", "vllm inter-node cfg#1", GET,
                           HEADER, [["0..inf", "zero-copy", "tcp/kv1"]]))  # fmt: skip

    def joined(**kwargs):
        (entry,) = ucx_protocols.summarize([log], **kwargs)["endpoint_configurations"]
        return entry["recorder"], entry["recorder_candidates"]

    assert joined(profiles=profiles) == (None, 2)
    recorder, candidates = joined(profiles=profiles, run_id="older")
    assert (recorder["process_id"], recorder["rank"], candidates) == ("proc-old", 1, 1)
    recorder, _ = joined(profiles=profiles / "older")
    assert recorder["process_id"] == "proc-old"



def test_a_table_frame_is_counted_even_when_its_title_does_not_parse():
    """A complete table with an unrecognized title is one error, alone or next
    to a valid table under the same prefix, in either order."""
    bad = _table(4, "1700000000.8", "vllm inter-node cfg#X", PUT, HEADER,
                 [["0..inf", "zero-copy", "tcp/kv0"]])  # fmt: skip
    get = _table(4, "1700000000.8", "vllm inter-node cfg#1", GET, HEADER,
                 [["0..inf", "zero-copy", "tcp/kv0"]])  # fmt: skip
    assert ucx_protocols.parse_lines(bad)["unparsed"] == 1
    for lines in (bad + get, get + bad):
        result = ucx_protocols.parse_lines(lines)
        assert result["unparsed"] == 1
        assert [t["api"] for t in result["tables"]] == ["ucp_get*"]


def test_a_thread_name_with_spaces_is_kept_whole():
    named = _table(6, "1700000000.9", "vllm inter-node cfg#0", GET, HEADER,
                   [["0..inf", "zero-copy", "tcp/kv0"]],
                   thread="progress lane")  # fmt: skip
    (table,) = ucx_protocols.parse_lines(named)["tables"]
    assert table["ucs_thread"] == "progress lane"


def test_tables_that_cannot_show_a_collision_leave_worker_identity_unresolved(
    tmp_path,
):
    """A PUT and a GET with every printed field equal may come from one worker
    or two: no collision is detected, and worker identity stays unresolved."""
    put = _table(6, "1700000001.0", "vllm inter-node cfg#0", PUT, HEADER,
                 [["0..inf", "zero-copy", "tcp/kv0"]])  # fmt: skip
    get = _table(6, "1700000001.1", "vllm inter-node cfg#0", GET, HEADER,
                 [["0..inf", "zero-copy", "tcp/kv0"]])  # fmt: skip
    _write_log(tmp_path / "put-get.log", put, get)
    result = ucx_protocols.summarize([tmp_path / "put-get.log"])
    (group,) = result["endpoint_configurations"]
    assert group["worker_collision_detected"] is False
    assert result["worker_identity"] == "unresolved: UCX does not print the UCP worker"


def test_a_malformed_recorder_entry_is_skipped_and_the_joins_kept(tmp_path):
    profiles = tmp_path / "profiles"
    _write_recorder(profiles, "run", "proc-1", 82, 1)
    bad = profiles / "run" / "proc-bad"
    bad.mkdir()
    (bad / "events.jsonl").write_text(json.dumps({"pid": 99, "run_id": "run",
                                                  "context": 7}) + "\n")  # fmt: skip
    log = tmp_path / "decode.log"
    _write_log(log, _table(82, "1700000001.2", "vllm inter-node cfg#1", GET,
                           HEADER, [["0..inf", "zero-copy", "tcp/kv1"]]))  # fmt: skip
    result = ucx_protocols.summarize([log], profiles, "run")
    assert result["tables"] == 1 and result["recorders_skipped"] == 1
    assert result["endpoint_configurations"][0]["recorder"]["process_id"] == "proc-1"


def test_a_table_cut_off_at_the_end_of_the_log_is_counted():
    """After a complete GET under one prefix, a second table's opening border
    and title with nothing after them (the log ended) is one error."""
    get = _table(4, "1700000001.3", "vllm inter-node cfg#1", GET, HEADER,
                 [["0..inf", "zero-copy", "tcp/kv0"]])  # fmt: skip
    cut = _table(4, "1700000001.3", "vllm inter-node cfg#2", PUT, HEADER,
                 [["0..inf", "zero-copy", "tcp/kv0"]])[:2]  # fmt: skip
    result = ucx_protocols.parse_lines(get + cut)
    assert result["unparsed"] == 1
    assert [t["api"] for t in result["tables"]] == ["ucp_get*"]


def test_a_recorder_without_a_context_is_skipped_not_a_join_candidate(tmp_path):
    """A second metadata entry for the same PID and run, missing its context,
    is skipped: the valid recorder still joins."""
    profiles = tmp_path / "profiles"
    _write_recorder(profiles, "run", "proc-1", 82, 1)
    bad = profiles / "run" / "proc-copy"
    bad.mkdir()
    (bad / "events.jsonl").write_text(
        json.dumps({"pid": 82, "run_id": "run", "process_id": "proc-copy"}) + "\n"
    )
    log = tmp_path / "decode.log"
    _write_log(log, _table(82, "1700000001.4", "vllm inter-node cfg#1", GET,
                           HEADER, [["0..inf", "zero-copy", "tcp/kv1"]]))  # fmt: skip
    result = ucx_protocols.summarize([log], profiles, "run")
    (entry,) = result["endpoint_configurations"]
    assert result["recorders_skipped"] == 1
    assert (entry["recorder"]["process_id"], entry["recorder_candidates"]) == (
        "proc-1", 1)  # fmt: skip
