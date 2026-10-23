"""Explicit preparation, execution, streaming verification and replay commands."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .solver.publication_io import canonical_digest, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-run", help="Freeze a run without using a GPU")
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--name", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--project", type=Path, default=Path.cwd())
    prepare.add_argument("--arm", choices=("physical", "selected_root_integer"), default="physical")
    run = commands.add_parser("run", help="Execute one prepared run from its frozen source")
    run.add_argument("--run-dir", type=Path, required=True)
    verify = commands.add_parser("verify", help="Incrementally verify sealed chunks and witnesses")
    verify.add_argument("--stream", type=Path, required=True)
    verify.add_argument("--receipts", type=Path, required=True)
    verify.add_argument("--fresh", action="store_true", help="Require a new receipt directory")
    pull = commands.add_parser(
        "pull", help="Download and verify completed chunks during computation"
    )
    pull.add_argument("--remote", required=True)
    pull.add_argument("--staging", type=Path, required=True)
    pull.add_argument("--output", type=Path, required=True)
    pull.add_argument("--receipts", type=Path, required=True)
    pull.add_argument("--ssh-command", default="ssh")
    pull.add_argument("--watch", action="store_true")
    pull.add_argument("--interval", type=float, default=5)
    replay = commands.add_parser(
        "replay", help="Independently replay all received trajectory chunks"
    )
    replay.add_argument("--stream", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--watch", action="store_true")
    replay.add_argument("--max-seconds", type=float, required=True)
    report = commands.add_parser("report", help="Summarize all runs against the frozen denominator")
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--runs", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    classical = commands.add_parser("classical", help="Run one frozen YAFU CPU control")
    classical.add_argument("--plan", type=Path, required=True)
    classical.add_argument("--name", required=True)
    classical.add_argument("--binary", type=Path, required=True)
    classical.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-run":
        from .repro import prepare

        plan = read_json(args.plan)
        config = dict(
            next(
                c
                for c in plan["physical_runs"] + plan.get("cost_runs", [])
                if c["run_id"] == args.name
            )
        )
        config["plan_sha256"] = canonical_digest(plan)
        if args.arm != "physical":
            if args.name not in plan["controls"]["selected_root_integer"]["paired_runs"]:
                raise ValueError("This control is not in the frozen plan")
            config["paired_run"] = config["run_id"]
            config["run_id"] += "-integer"
            config["arm"] = args.arm
        config.setdefault("task_overhead_seconds", plan["budget"]["task_overhead_cap_seconds"])
        result = prepare(args.project.resolve(), config, args.output, plan)
    elif args.command == "run":
        from .repro import execute

        raise SystemExit(execute(args.run_dir))
    elif args.command == "verify":
        from .runner import certificate_verifier
        from .stream import IncrementalVerifier

        if args.fresh and args.receipts.exists():
            raise FileExistsError("Fresh audit needs a new receipt directory")
        identity = read_json(args.stream / "identity.json")
        certify = None if identity.get("arm") else certificate_verifier(identity["config"])
        result = IncrementalVerifier(args.stream, args.receipts, certify).advance()
    elif args.command == "pull":
        result = follow_pull(args)
    elif args.command == "replay":
        from .replay import replay_stream

        result = replay_stream(args.stream, args.output, args.max_seconds, watch=args.watch)
    elif args.command == "classical":
        from .classical import run_yafu

        plan = read_json(args.plan)
        settings = plan["controls"]["classical"]
        config = next(c for c in settings["runs"] if c["run_id"] == args.name)
        result = run_yafu(
            args.binary,
            config["target"],
            args.output,
            seed=config["seed"],
            max_seconds=settings["wall_cap_seconds_per_run"],
        )
    else:
        from .report import build_report

        result = build_report(args.plan, args.runs, args.output)
    print(json.dumps(result, indent=2))


def follow_pull(args):
    from concurrent.futures import ThreadPoolExecutor

    from .runner import certificate_verifier
    from .stream import IncrementalVerifier, pull_ssh

    if not 1 <= args.interval <= 60:
        raise ValueError("Polling interval must be between one and 60 seconds")
    verifier = None
    pending = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpu-certificate") as pool:
        while True:
            result = pull_ssh(
                args.remote, args.staging, args.output, args.receipts, args.ssh_command
            )
            if verifier is None and (args.output / "identity.json").exists():
                identity = read_json(args.output / "identity.json")
                if not identity.get("arm"):
                    verifier = IncrementalVerifier(
                        args.output,
                        args.receipts / "certificates",
                        certificate_verifier(identity["config"]),
                    )
            if pending is not None and pending.done():
                pending.result()
                pending = None
            if verifier is not None and pending is None:
                pending = pool.submit(verifier.advance)
            print(json.dumps(result), flush=True)
            if not args.watch or result.get("terminal"):
                if pending is not None:
                    pending.result()
                if verifier is not None:
                    verifier.advance()
                return result
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
