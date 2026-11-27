# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The offline parser of NIXL's LIBFABRIC rail records: one record per agent
only where its rail sequence lines up, a diagnostic otherwise."""

from vllm.kv_profiling.libfabric_providers import parse_lines, summarize


def version(tid):
    return f"I1127 21:26:39.400000 {tid:>7} nixl_log.cpp:80] NIXL version: 1.5.2"


def opened(pid, fabric, provider="tcp"):
    prefix = f"libfabric:{pid}:1795814799::core:core:fi_fabric_()"
    using = f"Using {provider} provider 2.7, path:/opt/libfabric/lib/libfabric.so.1"
    return [f"{prefix}:1659<info> Opened fabric: {fabric}",
            f"{prefix}:1666<info> {using}"]  # fmt: skip


def rail(tid, fabric):
    return (f"I1127 21:26:39.428507 {tid:>7} libfabric_rail.cpp:509] "
            f"fabric_attr->name {fabric}")  # fmt: skip


def manager(tid, rails, provider="tcp"):
    return (f"I1127 21:26:39.433469 {tid:>7} libfabric_rail_manager.cpp:232] "
            f"Created {rails} rails using provider={provider}")  # fmt: skip


def agent(tid, name, rails):
    return (f"I1127 21:26:39.433642 {tid:>7} libfabric_backend.cpp:569] "
            f"Created self-connection for agent: {name} on {rails} rails")  # fmt: skip


def effective(pid, value):
    prefix = f"libfabric:{pid}:1795814799::core:core:fi_param_get_()"
    if value is None:
        return f"{prefix}:372<info> variable provider=<not set>"
    return f"{prefix}:381<info> read string var provider={value}"


def init(tid, name, fabrics, provider="tcp", logged=True, filter_value="tcp"):
    lines = [version(tid)] + ([effective(tid, filter_value)] if logged else [])
    for fabric in fabrics:
        lines += (opened(tid, fabric, provider) if logged else []) + [rail(tid, fabric)]
    return lines + [manager(tid, len(fabrics)), agent(tid, name, len(fabrics))]


def test_an_agents_rails_and_opened_providers_are_one_record():
    records, problems = parse_lines(init(81, "agent-a", ["10.0.0.0/24", "10.1.0.0/24"]))
    (record,) = records
    assert (record["agent"], record["manager_provider"], record["rails"]) == (
        "agent-a", "tcp", 2)  # fmt: skip
    assert [o["fabric"] for o in record["opened"]] == ["10.0.0.0/24", "10.1.0.0/24"]
    assert [o["provider"] for o in record["opened"]] == ["tcp", "tcp"]
    assert problems == []


def test_processes_on_distinct_threads_interleave_safely():
    a, b = init(81, "agent-a", ["n0"]), init(82, "agent-b", ["n1"], "verbs")
    lines = [line for pair in zip(a, b) for line in pair]
    records, problems = parse_lines(lines)
    assert {(r["agent"], r["opened"][0]["provider"]) for r in records} == {
        ("agent-a", "tcp"), ("agent-b", "verbs")}  # fmt: skip
    assert problems == []


def test_two_processes_sharing_a_thread_id_pair_nothing():
    """Two containers' logs merged, both initializing on thread ID 1: their
    lines interleave, so neither agent is paired with a provider."""
    a, b = init(1, "agent-a", ["n0"]), init(1, "agent-b", ["n1"], "verbs")
    lines = [line for pair in zip(a, b) for line in pair]
    records, problems = parse_lines(lines)
    assert records == [] and problems


def test_a_failed_rail_creation_ends_its_sequence():
    """A failure ends the pending sequence (reported); the retry pairs alone."""
    failed = [version(81), *opened(81, "n0"), rail(81, "n0"),
              ("E1127 21:26:39.430000      81 libfabric_rail_manager.cpp:364] "
               "Failed to create rails: fi_enable failed for rail 0")]  # fmt: skip
    records, problems = parse_lines(failed + init(81, "agent-a", ["n0"])[1:])
    assert [r["agent"] for r in records] == ["agent-a"]
    assert problems[0]["reason"].startswith("rail creation failed")


def test_sequences_that_do_not_line_up_are_diagnostics():
    mismatch = init(81, "agent-a", ["n0", "n1"])
    mismatch[-1] = agent(81, "agent-a", 1)
    other_fabric = [version(82), *opened(82, "n9"), rail(82, "n0"),
                    manager(82, 1), agent(82, "agent-b", 1)]  # fmt: skip
    unfinished = init(83, "agent-c", ["n0"])[:-1]
    records, problems = parse_lines(mismatch + other_fabric + unfinished)
    assert records == []
    reasons = [p["reason"] for p in problems]
    assert any(r.startswith("rail counts disagree") for r in reasons)
    assert "the opened fabric is not the rail's" in reasons
    assert "the log ends before the agent's self-connection" in reasons


def test_without_libfabric_logging_the_record_has_no_open(tmp_path):
    """FI_LOG_LEVEL unset: the manager's choice is recorded, no open."""
    (record,) = parse_lines(init(81, "agent-a", ["n0"], logged=False))[0]
    assert record["manager_provider"] == "tcp" and record["opened"] == []


def test_each_log_is_parsed_on_its_own_and_conflicts_are_listed(tmp_path):
    """A sequence split across two inputs never borrows the other input's
    lines: the first ends unfinished, the second has no process start. One
    agent named by two records is a conflict."""
    whole = init(81, "agent-a", ["n0"])
    first, second = tmp_path / "a.log", tmp_path / "b.log"
    first.write_text("\n".join(whole[:3]) + "\n")
    second.write_text("\n".join(whole[3:]) + "\n")
    split = summarize([first, second])
    assert split["records"] == [] and split["unparsed"] >= 2
    twice = tmp_path / "twice.log"
    twice.write_text("\n".join(whole + init(82, "agent-a", ["n0"])) + "\n")
    assert summarize([twice])["conflicts"] == ["agent-a"]


def test_two_valid_startups_on_one_thread_id_are_ambiguous():
    """Two process lifetimes on one thread ID in one input (a restart, or two
    containers' logs merged), each valid alone: neither record is kept."""
    a, b = init(1, "agent-a", ["n0"]), init(1, "agent-b", ["n1"], "sockets")
    for lines in (a + b, [line for pair in zip(a, b) for line in pair]):
        records, problems = parse_lines(lines)
        assert records == [] and problems


def test_the_rail_sequence_order_is_enforced():
    """An open after the rail manager, or an open no NIXL rail took, breaks
    the sequence."""
    late = init(81, "agent-a", ["n0"])
    late.insert(-1, opened(81, "n9")[0])
    dangling = init(82, "agent-b", ["n0"])
    dangling[-2:-2] = opened(82, "n9")
    records, problems = parse_lines(late + dangling)
    assert records == []
    reasons = [p["reason"] for p in problems]
    assert "a libfabric open after the rail manager" in reasons
    assert "a libfabric open with no NIXL rail" in reasons


def test_a_rail_failure_is_kept_even_without_info_logging():
    """At NIXL's default level only the error is logged: it is still reported."""
    failure = ("E1127 21:26:39.430000      81 libfabric_rail_manager.cpp:364] "
               "Failed to create rails: fi_enable failed for rail 0")  # fmt: skip
    records, problems = parse_lines([failure])
    assert records == [] and problems[0]["reason"].startswith("rail creation failed")


def test_the_effective_filter_is_libfabrics_own():
    (record,) = parse_lines(init(81, "agent-a", ["n0"], filter_value="tcp"))[0]
    assert record["effective_filter"] == "tcp"
    (record,) = parse_lines(init(81, "agent-a", ["n0"], filter_value=None))[0]
    assert record["effective_filter"] is None
    (record,) = parse_lines(init(81, "agent-a", ["n0"], logged=False))[0]
    assert record["effective_filter"] == "unknown"


def test_only_the_cores_info_records_are_opens_or_filters():
    """A provider's own or a WARN-level message is no open, and a provider-
    local parameter line is not the core's filter."""
    lines = init(81, "agent-a", ["n0"])
    position = lines.index(opened(81, "n0")[0])
    lines[position:position + 2] = [
        line.replace("::core:core:", "::tcp:core:") for line in opened(81, "n0")]
    records, problems = parse_lines(lines)
    assert records == [] and problems  # the rail has no core open
    warned = init(82, "agent-b", ["n0"])
    position = warned.index(opened(82, "n0")[0])
    warned[position:position + 2] = [
        line.replace("<info>", "<warn>") for line in opened(82, "n0")]
    assert parse_lines(warned)[0] == []
    local = init(83, "agent-c", ["n0"])
    local.insert(2, effective(83, "verbs").replace("::core:core:", "::tcp:core:"))
    (record,) = parse_lines(local)[0]
    assert record["effective_filter"] == "tcp"


def test_each_rail_takes_its_open_before_the_next_open():
    lines = [version(81), effective(81, "tcp"), *opened(81, "n0"), *opened(81, "n1"),
             rail(81, "n0"), rail(81, "n1"), manager(81, 2), agent(81, "agent-a", 2)]
    records, problems = parse_lines(lines)
    assert records == []
    assert "a second open before NIXL's rail took the first" in [
        p["reason"] for p in problems]


def test_only_libfabrics_startup_errors_are_rail_failures():
    """Another backend's error, or a transfer's error after startup, is no
    LIBFABRIC rail failure."""
    ucx = ("E1127 21:26:39.430000      81 backend_plugin.h:102] Failed to create "
           "engine: UCX worker creation failed")  # fmt: skip
    later = ("E1127 21:26:45.000000      81 libfabric_rail.cpp:1200] fi_read "
             "failed for rail 0: Connection refused")  # fmt: skip
    records, problems = parse_lines([ucx, *init(81, "agent-a", ["n0"]), later])
    assert [r["agent"] for r in records] == ["agent-a"]
    assert not [p for p in problems if p["reason"].startswith("rail creation failed")]
