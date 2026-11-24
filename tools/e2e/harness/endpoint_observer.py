"""L3 lifecycle only. The installed generic CLI owns all sampling and correlation."""

import contextlib
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from . import shell


@dataclass
class EndpointObserver:
    name: str
    output: Path
    run_id: str
    endpoint: str
    container_id: str | None = None

    @classmethod
    def start(
        cls,
        container,
        *,
        output,
        run_id,
        endpoint,
        role,
        cpus,
        probes,
        expected_workers,
        expected_schedulers,
        expected_frontends=1,
    ):
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        package_hash = bundle(output / "observer.pyz")
        # the observer finds, verifies and freezes these recorders and their
        # owner itself (endpoint_probes.identity.discover): no docker exec here
        expected = {"frontend": expected_frontends, "scheduler": expected_schedulers, "worker": expected_workers}
        description = json.loads(shell.run(["docker", "inspect", container]))[0]
        input_path, input_strategy = profile_backing(description)
        (output / "profile-input.json").write_text(
            json.dumps({"path": input_path, "strategy": input_strategy})
        )
        name = f"kv-observer-{run_id[:8]}-{endpoint}"
        config = {
            "probes": probes,
            "package_sha256": package_hash,
            "run_id": run_id,
            "endpoint": endpoint,
            "role": role,
            "expected": expected,
            "cpus": cpus,
            "input_dir": "/input",
            "output_dir": "/output/captures",
            "irq_policy": "virtual_not_applicable",
            "stop_file": "/output/stop",
            "endpoint_complete_file": "/output/children.json",
        }
        (output / "config.json").write_text(json.dumps(config))
        # Share only the endpoint's live capture directory, read-only. Never mount
        # endpoint source trees or make the observer depend on harness algorithms.
        if uninherited := uninherited_capabilities(description):
            raise ValueError(f"{container} added capabilities {uninherited} the observer does not take: it could not "
                             f"read the endpoint's namespaces (only {list(INHERITED_CAPABILITIES)} are inherited)")
        command = observer_command(name, container, description, cpus, input_path, output, probes)
        (output / "launch.json").write_text(json.dumps(command))
        observer = cls(name, output, run_id, endpoint)
        try:
            observer.container_id = shell.run(command).strip()
            start_command = ["docker", "start", observer.container_id]
            (output / "start.json").write_text(json.dumps(start_command))
            shell.run(start_command)
            # an eBPF collector loads and attaches first (the observer's
            # EBPF_ANSWER_TIMEOUT_S, 10 s) before the capture says ready
            ebpf = (probes or {}).get("ebpf") or {}
            deadline = time.monotonic() + READY_TIMEOUT_S + (
                EBPF_ANSWER_TIMEOUT_S if ebpf.get("enabled") is True else 0)
            while time.monotonic() < deadline:
                ready = list((output / "captures").glob(f"{run_id}/*/manifest.json"))
                if len(ready) == 1 and json.loads(ready[0].read_text()).get("ready"):
                    return observer
                state = json.loads(
                    shell.run(["docker", "inspect", observer.container_id])
                )[0]["State"]
                if not state["Running"]:
                    logs = subprocess.run(["docker", "logs", observer.container_id], capture_output=True,
                                          text=True, timeout=10, check=False)
                    raise RuntimeError("endpoint observer exited before readiness: "
                                       + (logs.stdout + logs.stderr)[-2000:])
                time.sleep(0.05)
            raise TimeoutError("endpoint observer readiness timeout")
        except BaseException:
            if observer.container_id is not None:
                observer.stop(False)
            raise

    def stop(self, children_closed):
        """A controller assertion does not override the CLI's L1 manifest check."""
        if self.container_id is None:
            return
        notice = self.output / "children.tmp"
        notice.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "endpoint": self.endpoint,
                    "children_closed": children_closed,
                }
            )
        )
        notice.replace(self.output / "children.json")
        (self.output / "stop").touch()
        # The Docker launcher's evidence, reconciled into the deployment-neutral
        # exit record (endpoint_probes.exit_record) once the observer is reaped.
        status = {
            "container_id": self.container_id,
            "timed_out": False,
            "exit_code": None,
            "removed": False,
        }
        try:
            status["exit_code"] = int(
                shell.run(["docker", "wait", self.container_id], timeout=3).strip()
            )
        except Exception as error:
            status.update(timed_out=True, error=str(error))
            shell.run(["docker", "kill", self.container_id], check=False)
        finally:
            stdout = stderr = b""
            try:
                logs = subprocess.run(
                    ["docker", "logs", self.container_id],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                stdout, stderr = logs.stdout, logs.stderr
                status["logs_exit_code"] = logs.returncode
            except (OSError, subprocess.SubprocessError) as error:
                status["logs_error"] = str(error)
                if isinstance(error, subprocess.TimeoutExpired):
                    stdout, stderr = error.stdout or b"", error.stderr or b""
            (self.output / "observer.stdout.log").write_bytes(stdout)
            (self.output / "observer.stderr.log").write_bytes(stderr)
            (self.output / "observer.log").write_bytes(
                b"=== stdout ===\n"
                + stdout
                + b"\n=== stderr (cross-stream order unavailable) ===\n"
                + stderr
            )
            inspect = shell.run(["docker", "inspect", self.container_id], check=False)
            (self.output / "container.json").write_text(inspect)  # diagnostics only
            try:
                shell.run(["docker", "rm", "-f", self.container_id])
                status["removed"] = True
            except Exception as error:
                status["remove_error"] = str(error)
            finally:
                self._publish_exit(status, inspect)

    def _publish_exit(self, status, inspect):
        """The exit record from Docker's evidence: success only when the wait
        result, the inspected container (this one, stopped, its exit code the
        waited one, not OOM-killed) and its removal agree."""
        problems, killed = [], None
        try:
            container = json.loads(inspect)[0]
        except (ValueError, IndexError, KeyError, TypeError):
            container = None
        state = container.get("State") if isinstance(container, dict) else None
        if not isinstance(state, dict):
            problems.append("the container could not be inspected")
            state = {}
        elif container.get("Id") != self.container_id:
            problems.append("the inspected container is not the launched one")
        # each fact must be stated with its type: unknown evidence is never clean
        # positive evidence only: an empty or partial State certifies nothing
        oom = state.get("OOMKilled")
        if type(oom) is not bool:
            problems.append(f"the container's OOM state is unknown ({oom!r})")
        elif oom is True:
            killed = "oom"
        if state.get("Running") is not False:
            problems.append(f"the container is not shown stopped (Running {state.get('Running')!r})")
        code = state.get("ExitCode")
        if type(code) is not int or code != status["exit_code"]:
            problems.append(f"inspected exit code {code!r} is not the waited {status['exit_code']!r}")
        if not status["removed"]:
            problems.append(f"the container was not removed: {status.get('remove_error')}")
        captures = list((self.output / "captures").glob(f"{self.run_id}/*/manifest.json"))
        process_id = None
        if len(captures) == 1:
            with contextlib.suppress(OSError, ValueError, AttributeError):
                process_id = json.loads(captures[0].read_text()).get("process_id")
        record = {"run_id": self.run_id, "endpoint": self.endpoint, "process_id": process_id,
                  "exit_code": status["exit_code"], "killed": killed, "timed_out": status["timed_out"],
                  "launcher": "docker", "problems": problems, "evidence": status}
        # written by the frozen observer package itself, as check_captures reads it
        subprocess.run([sys.executable, str(self.output / "observer.pyz"), "--record-exit", str(self.output)],
                       input=json.dumps(record), text=True, check=True, timeout=10)


def profile_backing(description):
    """Use a Docker-exposed backing path; procfs magic links cannot be bind sources."""
    target = PurePosixPath("/e2e/profiles")
    candidates = []
    for mount in description.get("Mounts", []):
        destination = PurePosixPath(mount["Destination"])
        if (
            mount.get("Type") in ("bind", "volume")
            and mount.get("Source")
            and target.is_relative_to(destination)
        ):
            source = PurePosixPath(mount["Source"])
            if source.is_absolute():
                candidates.append(
                    (
                        len(destination.parts),
                        str(source / target.relative_to(destination)),
                    )
                )
    if candidates:
        return max(candidates)[1], "existing_mount"
    merged = description.get("GraphDriver", {}).get("Data", {}).get("MergedDir")
    if merged and PurePosixPath(merged).is_absolute():
        return str(PurePosixPath(merged) / target.relative_to("/")), "overlay_merged"
    raise ValueError("L3 needs an exposed profile bind/volume or overlay merged root")


# How long the controller waits for an observer's capture to be ready, plus, when
# the endpoint opts into eBPF, the observer's own bound for the collector's first
# answer (runner.EBPF_ANSWER_TIMEOUT_S).
READY_TIMEOUT_S = 5.0
EBPF_ANSWER_TIMEOUT_S = 10.0

# The capabilities an observer takes from its endpoint when the endpoint's container added them: the kernel lets
# a process read another's /proc/<pid>/ns only when the target's capabilities are a subset of its own (without
# CAP_SYS_PTRACE), and an RDMA server adds IPC_LOCK. No other added privilege is copied.
INHERITED_CAPABILITIES = ("IPC_LOCK",)


def _capabilities(values) -> set[str]:
    return {str(v).upper().removeprefix("CAP_") for v in values or []}


def uninherited_capabilities(description) -> list[str]:
    """What the endpoint's container added beyond ``INHERITED_CAPABILITIES``: with any of them its processes hold
    a capability the observer lacks, so their namespaces are unreadable to it."""
    host = description.get("HostConfig") or {}
    return sorted(_capabilities(host.get("CapAdd")) - set(INHERITED_CAPABILITIES) - _capabilities(host.get("CapDrop")))


# What the opt-in eBPF collector needs, and nothing broader (never CAP_SYS_ADMIN): loading BPF programs and
# attaching them to tracepoints.
EBPF_CAPABILITIES = ("BPF", "PERFMON")


def observer_command(name, container, description, cpus, input_path, output, probes=None):
    """The observer's ``docker create``: in the endpoint container's network and PID namespaces, on its own CPUs,
    with the endpoint image. It drops what the endpoint's container dropped (``HostConfig.CapDrop``) and adds,
    of what it added (``CapAdd``), only ``INHERITED_CAPABILITIES``. With the endpoint's ``probes.ebpf`` enabled
    and explicitly opted into privilege, it also holds ``EBPF_CAPABILITIES`` and the built artifact directory,
    read-only at the same absolute path."""
    # literal booleans only (pd.ebpf_opt_in, the run's preflight): a string such as "false" is never consent
    from .pd import ebpf_opt_in

    privileged = ebpf_opt_in("endpoint", probes)
    artifacts = probes["ebpf"]["artifacts"] if privileged else None
    host = description.get("HostConfig") or {}
    added = sorted(_capabilities(host.get("CapAdd")) & set(INHERITED_CAPABILITIES))
    dropped = sorted(_capabilities(host.get("CapDrop")))
    return [
        "docker",
        "create",
        "--pull",
        "never",
        "--name",
        name,
        "--network",
        f"container:{container}",
        "--pid",
        f"container:{container}",
        *(arg for capability in dropped for arg in ("--cap-drop", capability)),
        *(arg for capability in added if capability not in dropped for arg in ("--cap-add", capability)),
        *(arg for capability in (EBPF_CAPABILITIES if privileged else ()) for arg in ("--cap-add", capability)),
        *(("--mount", f"type=bind,src={artifacts},dst={artifacts},readonly") if privileged else ()),
        "--cpuset-cpus",
        ",".join(map(str, cpus)),
        "--memory",
        "128m",
        "--mount",
        f"type=bind,src={input_path},dst=/input,readonly",
        "--mount",
        f"type=bind,src={output},dst=/output",
        "--mount",
        f"type=bind,src={output / 'observer.pyz'},dst=/observer.pyz,readonly",
        "--entrypoint",
        "/opt/venv/bin/python",
        description["Image"],
        "/observer.pyz",
        "--config",
        "/output/config.json",
    ]


def bundle(path, *, source_root=None):
    """Snapshot only the stdlib observer, not the endpoint's mutable source mount."""
    source_root = (
        Path(source_root) if source_root else Path(__file__).resolve().parents[3]
    )
    root = source_root / "vllm/kv_profiling"
    files = {"__main__.py": (root / "probe.py").read_bytes()}
    files.update(
        {
            "endpoint_probes/" + source.name: source.read_bytes()
            for source in sorted((root / "endpoint_probes").glob("*.py"))
        }
    )
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as archive:
        for name, contents in files.items():
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = 0o444 << 16
            archive.writestr(entry, contents)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    Path(path).with_suffix(".json").write_text(
        json.dumps(
            {
                "sha256": digest,
                "files": {
                    name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()
                },
            },
            indent=2,
        )
    )
    return digest


def check_captures(output, run, verdict):
    """Use the frozen generic observer's validator, not a second harness algorithm."""
    output, summaries = Path(output), {}
    for role, side in (("prefill", run.prefill), ("decode", run.decode)):
        for node in side.nodes:
            directory = output / "endpoint-probes" / node
            try:
                if run.profiler is None:
                    raise ValueError("L3 config missing")
                config = json.loads((directory / "config.json").read_text())
                profile = json.loads((output / "profiler.json").read_text())
                if (
                    config["run_id"] != profile["run_id"]
                    or config["endpoint"] != node
                    or config["role"] != role
                    or config["probes"] != run.profiler.probes[node]
                    or config["cpus"] != run.profiler.observer_cpus[node]
                ):
                    raise ValueError("swapped/stale endpoint config")
                command = [
                    sys.executable,
                    str(directory / "observer.pyz"),
                    "--check-capture",
                    str(directory),
                    "--core-dir",
                    str(output / "profiles" / role / node),
                ]
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=10
                )
                summary = json.loads(result.stdout)
                if result.returncode != 0 or summary.get("complete") is not True:
                    raise ValueError("incomplete endpoint capture: " + str(summary))
                summaries[node] = summary
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                subprocess.TimeoutExpired,
                AttributeError,
                IndexError,
            ) as error:
                verdict["problems"].append(
                    f"L3 {node}: incomplete endpoint capture: {error}"
                )
    verdict["endpoint_probes"] = summaries
