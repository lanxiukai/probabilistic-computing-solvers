"""Internal subprocess entry for the archived execution workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from combinatorial_solvers.workflows.execution import execute_worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    execute_worker(args.run_dir, args.source_root)


if __name__ == "__main__":
    main()
