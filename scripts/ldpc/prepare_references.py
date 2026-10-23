"""Preserve and build the pinned MIT decoding reference and WiFi matrices."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from combinatorial_solvers.core.evidence import file_identity, write_new_json
from combinatorial_solvers.ldpc.model import Code

DECODER_COMMIT = "d3429964cd4ffe1abfc041c6ec8b8425cb174f40"
WIFI_COMMIT = "05ee7f4af36ed5dacf52861315af68b8a17e71e0"


def extract(archive_path, destination, prefix):
    with tarfile.open(archive_path) as archive:
        if any(name.split("/")[0] != prefix for name in archive.getnames()):
            raise ValueError("Reference archive root differs from its pinned commit")
        archive.extractall(destination, filter="data")
    return destination / prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-archive", type=Path, required=True)
    parser.add_argument("--decoder-commit", type=Path, required=True)
    parser.add_argument("--wifi-archive", type=Path, required=True)
    parser.add_argument("--wifi-commit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        json.loads(args.decoder_commit.read_text())["sha"] != DECODER_COMMIT
        or json.loads(args.wifi_commit.read_text())["sha"] != WIFI_COMMIT
    ):
        raise ValueError("Unexpected upstream version")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for source, name in (
        (args.decoder_archive, "decoder-upstream.tar.gz"),
        (args.decoder_commit, "decoder-commit.json"),
        (args.wifi_archive, "wifi-upstream.tar.gz"),
        (args.wifi_commit, "wifi-commit.json"),
        (Path("scripts/ldpc/reference_driver.cpp"), "reference_driver.cpp"),
    ):
        shutil.copy2(source, output / name)
    with tempfile.TemporaryDirectory(prefix="ldpc-reference-build-") as temporary:
        work = Path(temporary)
        decoder = extract(args.decoder_archive, work, f"quantumgizmos-ldpc-{DECODER_COMMIT[:7]}")
        wifi = extract(args.wifi_archive, work, f"tavildar-LDPC-{WIFI_COMMIT[:7]}")
        shutil.copy2(decoder / "LICENSE", output / "decoder-LICENSE")
        shutil.copy2(wifi / "LICENSE", output / "wifi-LICENSE")
        shutil.copy2(wifi / "LdpcC/WiFiLDPC.h", output / "WiFiLDPC.h")
        shutil.copy2(wifi / "LdpcC/LdpcCode.cpp", output / "wifi-expansion.cpp")
        command = [
            "c++",
            "-std=c++17",
            "-O3",
            "-fopenmp",
            f"-I{decoder / 'src_cpp'}",
            f"-I{decoder / 'include/robin_map'}",
            str(output / "reference_driver.cpp"),
            "-o",
            str(output / "reference"),
        ]
        with (output / "build.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        text = (output / "WiFiLDPC.h").read_text()
        matrices = []
        for size, lift in ((648, 27), (1944, 81)):
            match = re.search(rf"int H_{size}_1_2\[12\]\[24\]\s*=\s*\{{(.*?)\}};", text, re.S)
            if match is None:
                raise ValueError("Expected WiFi matrix is absent")
            values = np.asarray([int(x) for x in re.findall(r"-?\d+", match[1])])
            base = values.reshape(12, 24)
            code = Code.wifi(base, lift)
            # Independently check every circulant against the author's row/column convention.
            dense = np.zeros((12 * lift, 24 * lift), dtype=np.uint8)
            for a in range(12):
                for b in range(24):
                    if base[a, b] >= 0:
                        block = np.roll(np.eye(lift, dtype=np.uint8), int(base[a, b]), axis=1)
                        dense[a * lift : (a + 1) * lift, b * lift : (b + 1) * lift] = block
            actual = np.zeros_like(dense)
            for a, row in enumerate(code.rows):
                actual[a, row] = 1
            if not np.array_equal(actual, dense):
                raise AssertionError("WiFi circulant expansion differs")
            np.savez_compressed(
                output / f"wifi-{size}-rate-half.npz",
                base=base,
                lift=lift,
                size=code.size,
                pointers=code.pointers,
                columns=code.columns,
            )
            matrices.append(
                {
                    "size": size,
                    "checks": code.checks,
                    "edges": len(code.columns),
                    "lift": lift,
                    "rate": "1/2",
                }
            )
    write_new_json(
        output / "metadata.json",
        {
            "decoder_url": "https://github.com/quantumgizmos/ldpc",
            "decoder_commit": DECODER_COMMIT,
            "wifi_url": "https://github.com/tavildar/LDPC",
            "wifi_commit": WIFI_COMMIT,
            "upstream_source_changes": [],
            "osd_scoring_adapter": (
                "BP uses logistic(-LLR); OSD's -log(p) scoring receives exp(-LLR) "
                "to rank the original independent-Bernoulli log-odds objective"
            ),
            "osd_policy": "Always postprocess nonzero syndromes, then retain the lower valid cost",
            "lsd_policy": "Order-zero fallback when BP has not converged",
            "maximum_osd_variables": 2048,
            "matrices": matrices,
            "build_command": command,
            "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
            "files": {p.name: file_identity(p) for p in sorted(output.iterdir()) if p.is_file()},
        },
    )
    print(json.dumps({"status": "built", "output": str(output), "matrices": matrices}))


if __name__ == "__main__":
    main()
