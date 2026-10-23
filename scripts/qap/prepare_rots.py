"""Fetch and instrument Taillard's external non-commercial QAP reference.

Only this measurement/build adapter belongs to this project. The downloaded and
instrumented upstream algorithm retains its original conditions and must not be
redistributed as part of this project's permissively licensed source package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

URL = "https://mistic.iict-heig-vd.ch/taillard/codes.dir/tabou_qap2.c"
SHA256 = "d9f0c81a7cca847bbf4f9388313af695c1dab4e3185930280eca240af45a734b"
CONDITIONS = (
    "External source: E. Taillard, standard C version dated 2006-03-14. "
    "The source permits non-commercial use and requires acknowledgement of "
    "E. Taillard for use or modification. It is not licensed under this project's "
    "MIT license. Original and instrumented upstream source are downloaded/build "
    "artifacts, not files for inclusion in the public source distribution."
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instrument(raw):
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Official Ro-TS source differs from the pinned original")
    source = raw.decode("ascii").replace("\r\n", "\n")
    pattern = r"(double rando\(\)\s*\{\s*)static (int x10[\s\S]+?x22\s*=\s*43210;)"
    source, count = re.subn(
        pattern,
        lambda match: match[2] + "\n\n" + match[1] + "++rots_rng_draws;\n",
        source,
        count=1,
    )
    if count != 1:
        raise ValueError("Pinned source RNG instrumentation anchor changed")
    anchor = "  /******************** main tabu search loop ********************/"
    if source.count(anchor) != 1:
        raise ValueError("Pinned source initialization anchor changed")
    source = source.replace(
        anchor,
        "  rots_observe(0, n, a, b, p, current_cost, best_sol, *best_cost, "
        "tabu_list, delta);\n" + anchor,
    )
    anchor = "current_iteration <= nr_iterations && *best_cost > opt;"
    if source.count(anchor) != 1:
        raise ValueError("Pinned source iteration boundary changed")
    source = source.replace(
        anchor,
        "current_iteration <= nr_iterations && *best_cost > opt "
        "&& rots_before_iteration(current_iteration);",
    )
    anchor = "   }; \n  /* free memory*/"
    if source.count(anchor) != 1:
        raise ValueError("Pinned source completed-iteration anchor changed")
    source = source.replace(
        anchor,
        "    rots_observe(current_iteration, n, a, b, p, current_cost, "
        "best_sol, *best_cost, tabu_list, delta);\n   };\n"
        "  rots_finish(n, p, current_cost, best_sol, *best_cost, tabu_list);\n"
        "  /* free memory*/",
    )
    return source


def prepare(output, original=None, compiler="cc"):
    if output.exists():
        raise ValueError("Use a new external reference build directory")
    output.mkdir(parents=True)
    raw = (
        original.read_bytes()
        if original is not None
        else urllib.request.urlopen(URL, timeout=60).read()
    )
    instrumented = instrument(raw)
    (output / "upstream.c").write_bytes(raw)
    (output / "instrumented.c").write_text(instrumented)
    (output / "UPSTREAM-CONDITIONS.txt").write_text(CONDITIONS + "\n" + URL + "\n")
    source_directory = Path(__file__).resolve().parent
    for name in ("rots_adapter.c", "rots_probe.c"):
        shutil.copyfile(source_directory / name, output / name)
    commands = []
    with (output / "build.log").open("w") as log:
        subprocess.run([compiler, "--version"], stdout=log, stderr=log, check=True)
        for name, executable in (("rots_adapter.c", "rots"), ("rots_probe.c", "rots-probe")):
            command = [compiler, "-O3", "-std=c99", name, "-o", executable]
            commands.append(command)
            subprocess.run(command, cwd=output, stdout=log, stderr=log, check=True)
    metadata = {
        "upstream_url": URL,
        "upstream_sha256": SHA256,
        "upstream_conditions": CONDITIONS,
        "measurement_adapter": (
            "Exposed six original RNG words and counted draws; added completed-iteration "
            "observation and stop hooks. Move selection, age aspiration, nonuniform tabu "
            "durations and delta-cache updates remain upstream code. Reference early "
            "termination is disabled by opt=-1. Verbose upstream prints are replaced by "
            "structured witnesses."
        ),
        "commands": commands,
        "files": {
            name: digest(output / name)
            for name in (
                "upstream.c",
                "instrumented.c",
                "rots_adapter.c",
                "rots_probe.c",
                "rots",
                "rots-probe",
                "build.log",
                "UPSTREAM-CONDITIONS.txt",
            )
        },
    }
    (output / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--compiler", default="cc")
    parser.add_argument("--accept-noncommercial-conditions", action="store_true")
    args = parser.parse_args()
    if not args.accept_noncommercial_conditions:
        parser.error(
            "Read the external source conditions and pass --accept-noncommercial-conditions"
        )
    print(json.dumps(prepare(args.output, args.original, args.compiler), indent=2))
