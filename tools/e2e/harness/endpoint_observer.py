"""L3 lifecycle only. The installed generic CLI owns all sampling and correlation."""

import hashlib
import json
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from . import shell, topology


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
    ):
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        package_hash = bundle(output / "observer.pyz")
        remote_package = "/e2e/observer.pyz"
        shell.run(
            [
                "docker",
                "cp",
                str(output / "observer.pyz"),
                f"{container}:{remote_package}",
            ]
        )
        inventory = json.loads(
            topology.exec_in(
                container,
                [
                    "/opt/venv/bin/python",
                    remote_package,
                    "--inventory",
                    "/e2e/profiles",
                    "--run-id",
                    run_id,
                ],
            )
        )
        counts = {
            role: sum(m["context"]["role"] == role for m in inventory)
            for role in ("frontend", "scheduler", "worker")
        }
        if counts != {
            "frontend": 1,
            "scheduler": expected_schedulers,
            "worker": expected_workers,
        }:
            raise ValueError("endpoint capture inventory count mismatch")
        owner = next(m for m in inventory if m["context"]["role"] == "frontend")
        description = json.loads(shell.run(["docker", "inspect", container]))[0]
        input_path, input_strategy = profile_backing(description)
        (output / "profile-input.json").write_text(
            json.dumps({"path": input_path, "strategy": input_strategy})
        )
        name = f"kv-observer-{run_id[:8]}-{endpoint}"
        config = {
            "probes": probes,
            "inventory": inventory,
            "package_sha256": package_hash,
            "run_id": run_id,
            "endpoint": endpoint,
            "role": role,
            "owner_pid": owner["pid"],
            "owner_start": owner["start_ticks"],
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
        command = [
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
        (output / "launch.json").write_text(json.dumps(command))
        observer = cls(name, output, run_id, endpoint)
        try:
            observer.container_id = shell.run(command).strip()
            start_command = ["docker", "start", observer.container_id]
            (output / "start.json").write_text(json.dumps(start_command))
            shell.run(start_command)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                ready = list((output / "captures").glob(f"{run_id}/*/manifest.json"))
                if len(ready) == 1 and json.loads(ready[0].read_text()).get("ready"):
                    return observer
                state = json.loads(
                    shell.run(["docker", "inspect", observer.container_id])
                )[0]["State"]
                if not state["Running"]:
                    raise RuntimeError("endpoint observer exited before readiness")
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
        status = {
            "run_id": self.run_id,
            "endpoint": self.endpoint,
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
            (self.output / "container.json").write_text(inspect)
            try:
                shell.run(["docker", "rm", "-f", self.container_id])
                status["removed"] = True
            finally:
                (self.output / "endpoint-status.json").write_text(json.dumps(status))


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
