"""Reproduce one frozen S23 cell from isolated snapshots with a full-call timeout."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def stop_child(child):
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def run_worker(command, timeout, log_path):
    child = None
    try:
        with log_path.open("w") as stream:
            child = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                raise TimeoutError("Worker exceeded the frozen full-call limit") from None
            if code:
                raise RuntimeError(f"Worker returned {code}")
    finally:
        stop_child(child)


def interrupted(_signal, _frame):
    raise KeyboardInterrupt("Termination requested; preserve partial evidence")


def main():
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("configs/spin-glass/s23-panel.json"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--cell")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    panel = json.loads(args.panel.read_text())
    definitions = {d["id"]: d for d in panel["cells"]}
    if args.list:
        print(
            json.dumps(
                [
                    {k: d[k] for k in ("id", "phase", "length", "family", "seed")}
                    for d in definitions.values()
                ],
                indent=2,
            )
        )
        return
    if args.cell not in definitions or args.output is None:
        parser.error("Select an existing cell and provide --output")
    root = Path(__file__).resolve().parents[2]
    provenance = json.loads((root / "provenance/spin-glass/release-v1.json").read_text())
    if digest(args.panel) != provenance["public_panel_sha256"]:
        raise ValueError("The frozen panel changed")
    runtime = provenance["numerical_files_byte_identical_to_executed_source"]
    for row in runtime:
        if digest(root / row["path"]) != row["sha256"]:
            raise ValueError("Measured runtime source changed")
    definition = definitions[args.cell]
    instance = root / panel["instance_directory"] / definition["instance_file"]
    manifest = json.loads((instance.parent.parent / "manifest.json").read_text())
    if digest(instance) != manifest["files"]["instances/" + instance.name]["sha256"]:
        raise ValueError("Frozen coupling file changed")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "definition.json", definition)
    shutil.copyfile(__file__, output / "launcher.py")
    status, started = "failed", None
    try:
        for row in runtime:
            target = output / "runtime-source" / Path(row["path"]).relative_to("src")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / row["path"], target)
            if digest(target) != row["sha256"]:
                raise ValueError("Restored runtime source differs")
        shutil.copyfile(instance, output / "instance.json")
        environment = output / "environment"
        environment.mkdir()
        for name in ("pyproject.toml", "uv.lock", ".python-version"):
            shutil.copyfile(root / name, environment / name)
        config = output / "prepared-config.json"
        write(config, {"instance_file": "instance.json", "solver": definition["solver"]})
        code = (
            "import sys;sys.path.insert(0,sys.argv[1]);"
            "from cubic_spin_glass.cli import main;main(sys.argv[2:])"
        )
        command = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            code,
            str(output / "runtime-source"),
            "--config",
            str(config),
            "--output",
            str(output / "result"),
            "--environment-project",
            str(environment),
            "--threads",
            "1",
        ]
        write(output / "command.json", command)
        started = time.perf_counter()
        run_worker(command, panel["full_call_limit_seconds"], output / "worker.log")
        status = "completed"
    except BaseException as error:
        write(output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        write(
            output / "timing.json",
            {
                "status": status,
                "complete_call_seconds": time.perf_counter() - started
                if started is not None
                else None,
                "new_independent_coupling_instances": 0,
            },
        )
        write(
            output / "manifest.json",
            {
                "schema": "cubic-spin-glass-reproduction-v1",
                "status": status,
                "files": {
                    str(p.relative_to(output)): {"sha256": digest(p), "bytes": p.stat().st_size}
                    for p in sorted(output.rglob("*"))
                    if p.is_file()
                },
            },
        )
    print(output)


if __name__ == "__main__":
    main()
