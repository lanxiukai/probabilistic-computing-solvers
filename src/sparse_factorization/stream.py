"""Seal, transfer and incrementally verify completed trajectory segments."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .solver.publication_io import atomic_json, atomic_npz, canonical_digest, digest, read_json

SCHEMA = "factor-trajectory-chunk-v1"
FILES = {"checkpoint.npz", "metrics.npz", "first_hits.npz", "witnesses.npz"}


def arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class ChunkWriter:
    """One compression worker with at most two owned host snapshots in flight.

    CUDA copies occur in the sampling thread at completed event boundaries. The
    background worker only receives owned NumPy arrays and never touches the GPU.
    A directory becomes downloadable only after its complete manifest is sealed.
    """

    def __init__(self, root: Path, identity: dict, initial: dict, initial_files=None):
        self.root, self.identity = root, identity
        root.mkdir(parents=True, exist_ok=False)
        (root / "chunks").mkdir()
        atomic_json(root / "identity.json", identity)
        self.previous = None
        self.last_event = 0
        self.submitted = 0
        self.slots = threading.Semaphore(2)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chunk-compress")
        self.pending = []
        self._seal("initial", 0, 0, {"checkpoint.npz": initial, **(initial_files or {})})

    def _seal(self, name: str, start: int, end: int, files: dict):
        parent = self.root if name == "initial" else self.root / "chunks"
        temporary = parent / ("." + name + ".partial")
        temporary.mkdir()
        for filename, payload in files.items():
            if filename not in FILES:
                raise ValueError("Unknown trajectory file")
            atomic_npz(temporary / filename, **payload)
        entries = [
            {"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(temporary.iterdir())
        ]
        manifest = {
            "schema": SCHEMA,
            "identity_sha256": canonical_digest(self.identity),
            "name": name,
            "start_event_exclusive": start,
            "end_event_inclusive": end,
            "previous_manifest_sha256": self.previous,
            "files": entries,
        }
        atomic_json(temporary / "manifest.json", manifest)
        self.previous = digest(temporary / "manifest.json")
        temporary.rename(parent / name)
        return manifest

    def submit(self, end: int, files: dict):
        if end <= self.last_event or end > self.identity["maximum_events"]:
            raise ValueError("Segments must advance the frozen event horizon")
        for future in self.pending:
            if future.done():
                future.result()
        self.pending = [future for future in self.pending if not future.done()]
        self.slots.acquire()
        start = self.last_event
        self.last_event = end
        self.submitted += 1
        name = f"chunk-{self.submitted:06d}"

        def write():
            try:
                return self._seal(name, start, end, files)
            finally:
                self.slots.release()

        self.pending.append(self.executor.submit(write))

    def drain(self):
        for future in self.pending:
            future.result()

    def close(self, status: str, summary: dict | None = None):
        try:
            self.drain()
        finally:
            self.executor.shutdown(wait=True)
        extra = {}
        if summary is not None:
            atomic_json(self.root / "summary.json", summary)
            extra["summary_sha256"] = digest(self.root / "summary.json")
        atomic_json(
            self.root / "terminal.json",
            {
                "status": status,
                "events": self.last_event,
                "chunks": self.submitted,
                "last_manifest_sha256": self.previous,
                **extra,
            },
        )


def validate_segment(path: Path, identity: dict, previous: str | None, start: int) -> dict:
    manifest = read_json(path / "manifest.json")
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("identity_sha256") != canonical_digest(identity)
        or manifest.get("previous_manifest_sha256") != previous
        or manifest.get("start_event_exclusive") != start
        or manifest.get("name") != path.name
    ):
        raise ValueError("Segment identity, order or checkpoint chain differs")
    listed = [row["path"] for row in manifest["files"]]
    if len(set(listed)) != len(listed) or not set(listed) <= FILES:
        raise ValueError("Invalid segment file inventory")
    actual = {p.name for p in path.iterdir() if p.name != "manifest.json"}
    if actual != set(listed) or "checkpoint.npz" not in actual:
        raise ValueError("Incomplete or unexpected segment files")
    for row in manifest["files"]:
        file = path / row["path"]
        if file.is_symlink() or not file.is_file():
            raise ValueError("Segment members must be regular files")
        if file.stat().st_size != row["bytes"] or digest(file) != row["sha256"]:
            raise ValueError("Segment checksum mismatch")
    checkpoint = arrays(path / "checkpoint.npz")
    end = manifest["end_event_inclusive"]
    if not isinstance(end, int) or not start <= end <= identity["maximum_events"]:
        raise ValueError("Invalid segment event range")
    if int(checkpoint["circuit_events"]) != end:
        raise ValueError("Checkpoint event differs from segment boundary")
    shape = (identity["replicas"], identity["free_spins"])
    for key in ("states", "best_states"):
        state = checkpoint[key]
        if state.shape != shape or not np.isin(state, (-1, 1)).all():
            raise ValueError("Invalid bipolar checkpoint state")
    rng = checkpoint["circuit_rng_state"]
    if rng.dtype != np.uint8 or rng.ndim != 1 or not len(rng):
        raise ValueError("Missing or invalid RNG checkpoint")
    if path.name == "initial":
        if start != 0 or end != 0 or "metrics.npz" in actual:
            raise ValueError("Invalid initial checkpoint")
    else:
        if end <= start or not {"metrics.npz", "first_hits.npz"} <= actual:
            raise ValueError("Empty or incomplete trajectory segment")
        metrics = arrays(path / "metrics.npz")
        counts, wall = metrics["cumulative_by_position"], metrics["event_wall"]
        columns = identity.get("metric_columns", 19)
        if counts.shape != (end - start, 32, columns) or counts.dtype != np.int64:
            raise ValueError("Wrong event coverage or counter layout")
        if (counts < 0).any() or (np.diff(counts, axis=0) < 0).any():
            raise ValueError("Cumulative counters decreased")
        if wall.shape != (end - start, 3) or not np.isfinite(wall).all():
            raise ValueError("Invalid event timing coverage")
        if (np.diff(wall, axis=1) < 0).any() or (np.diff(wall[:, 2]) < 0).any():
            raise ValueError("Nonmonotone event timestamps")
        expected = checkpoint["arithmetic_counts"].reshape(-1, 32, columns).sum(axis=0)
        if not np.array_equal(counts[-1], expected):
            raise ValueError("Last counters differ from checkpoint")
    return manifest


class IncrementalVerifier:
    """Check each newly sealed segment once and retain explicit local receipts."""

    def __init__(self, root: Path, receipts: Path, certify=None):
        self.root, self.receipts, self.certify = root, receipts, certify
        receipts.mkdir(parents=True, exist_ok=True)
        self.identity = read_json(root / "identity.json")
        self.previous = None
        self.event = 0
        self.last_counts = None
        self.names = []
        self.initial_checked = False
        # Receipts are local derived state. Raw files remain separately immutable.
        for receipt in sorted(receipts.glob("*.json")):
            row = read_json(receipt)
            if row["identity_sha256"] != canonical_digest(self.identity):
                raise ValueError("Receipt belongs to a different run")
            relative = row["segment"]
            if relative != "initial" and relative != f"chunks/chunk-{len(self.names):06d}":
                raise ValueError("Receipt sequence is incomplete")
            if digest(root / relative / "manifest.json") != row["manifest_sha256"]:
                raise ValueError("Previously verified segment manifest changed")
            self.previous, self.event = row["manifest_sha256"], row["events"]
            self.names.append(relative)
        if self.names:
            self.initial_checked = True
            checkpoint = arrays(root / self.names[-1] / "checkpoint.npz")
            self.last_counts = (
                checkpoint["arithmetic_counts"]
                .reshape(-1, 32, self.identity.get("metric_columns", 19))
                .sum(0)
            )

    def advance(self) -> dict:
        candidates = [] if self.initial_checked else [self.root / "initial"]
        candidates += sorted((self.root / "chunks").glob("chunk-*"))
        for path in candidates:
            if not (path / "manifest.json").is_file():
                break
            relative = path.relative_to(self.root).as_posix()
            if relative in self.names:
                continue
            expected = "initial" if not self.names else f"chunks/chunk-{len(self.names):06d}"
            if relative != expected:
                raise ValueError("Segment number has a gap or duplication")
            manifest = validate_segment(path, self.identity, self.previous, self.event)
            checkpoint = arrays(path / "checkpoint.npz")
            if path.name != "initial":
                counts = arrays(path / "metrics.npz")["cumulative_by_position"]
                if self.last_counts is not None and (counts[0] < self.last_counts).any():
                    raise ValueError("Counters decreased across a segment boundary")
            certificates = None
            if (path / "witnesses.npz").exists() and self.certify is not None:
                certificates = self.certify(arrays(path / "witnesses.npz"))
            self.previous = digest(path / "manifest.json")
            self.event = manifest["end_event_inclusive"]
            self.last_counts = (
                checkpoint["arithmetic_counts"]
                .reshape(-1, 32, self.identity.get("metric_columns", 19))
                .sum(0)
            )
            atomic_json(
                self.receipts / f"{len(self.names):06d}.json",
                {
                    "identity_sha256": canonical_digest(self.identity),
                    "segment": relative,
                    "manifest_sha256": self.previous,
                    "events": self.event,
                    "certificates": certificates,
                    "full_origin_replay": False,
                },
            )
            self.names.append(relative)
            self.initial_checked = True
        terminal = self.root / "terminal.json"
        if terminal.exists():
            final = read_json(terminal)
            if (
                "summary_sha256" in final
                and digest(self.root / "summary.json") != final["summary_sha256"]
            ):
                raise ValueError("Final certification summary differs")
            if len(list((self.root / "chunks").glob("chunk-*"))) > len(self.names) - 1:
                return self.advance()
            if (
                final["events"] != self.event
                or final["chunks"] != len(self.names) - 1
                or final["last_manifest_sha256"] != self.previous
            ):
                raise ValueError("Terminal totals differ from the verified chain")
        return {
            "verified_events": self.event,
            "verified_chunks": max(0, len(self.names) - 1),
            "terminal": terminal.exists(),
            "full_origin_replay": False,
        }


def mirror_local(source: Path, destination: Path, receipts: Path, certify=None) -> dict:
    """Copy only immutable completed directories; useful for local and transport tests."""
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "chunks").mkdir(exist_ok=True)
    identity = read_json(source / "identity.json")
    if (destination / "identity.json").exists():
        if read_json(destination / "identity.json") != identity:
            raise ValueError("Destination belongs to a different run")
    else:
        atomic_json(destination / "identity.json", identity)
    verifier = IncrementalVerifier(destination, receipts, certify)
    candidates = [source / "initial", *sorted((source / "chunks").glob("chunk-*"))]
    for path in candidates:
        if not (path / "manifest.json").is_file():
            break
        relative = path.relative_to(source)
        target = destination / relative
        if target.exists():
            if digest(target / "manifest.json") != digest(path / "manifest.json"):
                raise ValueError("Refusing to overwrite a completed local segment")
            continue
        with tempfile.TemporaryDirectory(prefix=".incoming-", dir=destination) as temporary:
            staged = Path(temporary) / path.name
            shutil.copytree(path, staged, symlinks=True)
            validate_segment(staged, identity, verifier.previous, verifier.event)
            staged.rename(target)
        verifier.advance()
    if (source / "terminal.json").exists():
        terminal = read_json(source / "terminal.json")
        # A producer may seal another chunk after the directory listing above.
        # Defer the terminal marker until the local chain has caught up.
        if terminal["events"] != verifier.event or terminal["chunks"] != len(verifier.names) - 1:
            return verifier.advance()
        if "summary_sha256" in terminal:
            if digest(source / "summary.json") != terminal["summary_sha256"]:
                raise ValueError("Source final summary differs")
            if not (destination / "summary.json").exists():
                atomic_json(destination / "summary.json", read_json(source / "summary.json"))
        if (destination / "terminal.json").exists():
            if read_json(source / "terminal.json") != read_json(destination / "terminal.json"):
                raise ValueError("Terminal record changed")
        else:
            atomic_json(destination / "terminal.json", read_json(source / "terminal.json"))
    return verifier.advance()


def pull_ssh(
    remote: str,
    staging: Path,
    destination: Path,
    receipts: Path,
    ssh_command: str = "ssh",
    certify=None,
) -> dict:
    """Fetch sealed data with rsync, excluding already verified immutable segments.

    ``remote`` is an rsync endpoint such as ``gpu:/workspace/run/stream``. SSH
    identity and ports can be supplied via the user's normal SSH alias or -e.
    No remote shell pipeline, credentials or instance lifecycle actions are used.
    """
    if ":" not in remote or remote.startswith("-"):
        raise ValueError("Require an explicit SSH endpoint")
    staging.mkdir(parents=True, exist_ok=True)
    excludes = []
    if (destination / "initial").exists():
        excludes.append("/initial/***")
    for path in sorted((destination / "chunks").glob("chunk-*")):
        excludes.append(f"/chunks/{path.name}/***")
    command = [
        "rsync",
        "-a",
        "--protect-args",
        "--delay-updates",
        "--partial-dir=.partial",
        "-e",
        ssh_command,
    ]
    for pattern in excludes:
        command += ["--exclude", pattern]
    command += [
        "--include=/identity.json",
        "--include=/initial/***",
        "--include=/chunks/",
        "--include=/chunks/chunk-*/***",
        "--include=/terminal.json",
        "--include=/summary.json",
        "--exclude=*",
        "--",
        remote.rstrip("/") + "/",
        str(staging) + "/",
    ]
    subprocess.run(command, check=True)
    # Previously fetched directories remain in staging; already verified ones are
    # never downloaded again. Mirrors verify only new segments, not whole archives.
    return mirror_local(staging, destination, receipts, certify)
