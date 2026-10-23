"""Prepare, execute or verify one isolated job-shop experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from combinatorial_solvers.core.evidence import verify_manifest
from combinatorial_solvers.jssp.runner import registry
from combinatorial_solvers.workflows.execution import execute_experiment, prepare_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "verify"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.config is None or args.run_id is None:
            parser.error("prepare requires --config and --run-id")
        print(
            prepare_experiment(
                registry().get("jssp"), args.config, repository=Path.cwd(), run_id=args.run_id
            )
        )
    elif args.run_dir is None:
        parser.error("run and verify require --run-dir")
    elif args.action == "run":
        raise SystemExit(execute_experiment(args.run_dir))
    else:
        print(json.dumps(verify_manifest(args.run_dir)))


if __name__ == "__main__":
    main()
