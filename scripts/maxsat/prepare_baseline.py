"""Build USW-LS with isolated stopping and certificate-output instrumentation."""

from __future__ import annotations

import argparse
import difflib
import json
import platform
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from combinatorial_solvers.core.evidence import file_identity

MAIN = r"""#include <signal.h>
volatile sig_atomic_t benchmark_stop = 0;
long long benchmark_step_limit = -1;
bool benchmark_ready = false;
#include "basis_pms.h"
#include "build.h"
#include "pms.h"
#include "heuristic.h"

USW s;
int seed = 1;
long long best_known = 0;
long long total_step = 0;
long long consecutive_better_soft = 0;
char *file_name = NULL;
void interrupt(int) { benchmark_stop = 1; }

int main(int argc, char *argv[]) {
    if (argc < 3) return 2;
    start_timing();
    signal(SIGTERM, interrupt);
    sscanf(argv[2], "%d", &seed);
    if (argc > 3) benchmark_step_limit = strtoll(argv[3], NULL, 10);
    srand(seed);
    s.build_instance(argv[1]);
    s.settings();
    s.parse_parameters2(argc, argv);
    s.local_search_with_decimation(argv[1]);
    cout << "c benchmark-flips " << total_step << endl;
    s.print_best_solution();
    return 0;
}
"""


def replace_once(text, original, updated):
    if text.count(original) != 1:
        raise ValueError("Upstream instrumentation anchor is not unique")
    return text.replace(original, updated, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != "1b1d216f91e7da81460a21f5cfffab5eed40d3c9":
        raise ValueError("Unexpected frozen reference commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=source,
        text=True,
    ).strip():
        raise ValueError("Build from a clean pinned upstream checkout")
    with (output / "upstream.tar.gz").open("xb") as handle:
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar.gz",
                "HEAD",
                "source-code",
                "README.md",
                "starexec_usw-ls-runsolver.sh",
            ],
            cwd=source,
            stdout=handle,
            check=True,
        )
    with tempfile.TemporaryDirectory(prefix="maxsat-usw-build-") as temporary:
        build = Path(temporary) / "source-code"
        shutil.copytree(source / "source-code", build)
        old = {name: (build / name).read_text() for name in ("pms.cpp", "pms.h", "heuristic.h")}
        (build / "pms.cpp").write_text(MAIN)
        header = replace_once(
            old["pms.h"],
            "if (best_soln_feasible == 0)",
            "if (opt_unsat_weight == __LONG_LONG_MAX__)",
        )
        header = replace_once(
            header,
            'printf("v ");',
            'if (!verify_sol()) { cerr << "Invalid final witness" << endl; exit(2); }'
            '\n\tprintf("v ");',
        )
        (build / "pms.h").write_text(header)
        heuristic = replace_once(
            old["heuristic.h"],
            "            int flipvar = pick_var();",
            "            if (benchmark_stop || (benchmark_step_limit >= 0 && "
            "total_step >= benchmark_step_limit)) return;\n"
            "            if (!benchmark_ready) {\n"
            "                benchmark_ready = true;\n"
            '                cout << "c benchmark-search-ready" << endl;\n'
            "                char resume;\n"
            '                if (scanf(" %c", &resume) != 1) return;\n'
            "            }\n"
            "            int flipvar = pick_var();",
        )
        (build / "heuristic.h").write_text(heuristic)
        changes = []
        for name, original in old.items():
            changes.extend(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    (build / name).read_text().splitlines(keepends=True),
                    fromfile=f"upstream/{name}",
                    tofile=f"instrumented/{name}",
                )
            )
        (output / "instrumentation.patch").write_text("".join(changes))
        with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
            for path in sorted(build.iterdir()):
                if path.suffix in {".cpp", ".h"} or path.name == "Makefile":
                    archive.add(path, arcname=f"source-code/{path.name}")
            archive.add(source / "README.md", arcname="README.md")
        command = ["g++", "-O3", "-std=c++17", "pms.cpp", "-o", str(output / "USW-LS")]
        metadata = {
            "url": "https://github.com/shaowei-cai-group/USW-LS",
            "commit": commit,
            "compiler": subprocess.check_output(["g++", "--version"], text=True),
            "libc": platform.libc_ver(),
            "build_command": command,
            "license_observation": (
                "No repository LICENSE file observed; no redistribution license inferred"
            ),
            "instrumentation": (
                "Stop flag checked at iteration boundary; one initial-state timing handshake; "
                "optional fixed-flip replay; verify and print final witness. "
                "Search score, flip, and weight-update routines unchanged."
            ),
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with (output / "build.log").open("x") as handle:
            handle.write(json.dumps(command) + "\n")
            handle.flush()
            subprocess.run(
                command, cwd=build, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=120
            )
    metadata["files"] = {
        name: file_identity(output / name)
        for name in (
            "USW-LS",
            "source.tar.gz",
            "upstream.tar.gz",
            "instrumentation.patch",
            "build.log",
        )
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"commit": commit, "output": str(output)}))


if __name__ == "__main__":
    main()
