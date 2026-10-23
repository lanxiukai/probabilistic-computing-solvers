"""Wall-limited USW-LS wrapper with a search-ready handshake and saved witness."""

from __future__ import annotations

import json
import os
import re
import selectors
import signal
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np


def run(
    instance,
    config,
    output,
    *,
    input_path=None,
    absolute_deadline=None,
    barrier=None,
    stop_event=None,
):
    binary = Path(config["executable"]).resolve(strict=True)
    seconds = float(config["max_seconds"])
    seed = int(config["seed"])
    if not np.isfinite(seconds) or not 0 < seconds <= 600 or not 0 <= seed < 2**31:
        raise ValueError("Invalid external baseline budget or seed")
    filename = input_path or output / "instance.wcnf"
    if input_path is None:
        instance.write_wcnf(filename)
    command = [
        str(binary),
        str(filename),
        str(seed),
        str(int(config.get("fixed_flips", -1))),
        "-best",
        str(int(config.get("target_cost", 0))),
    ]
    (output / "baseline-command.json").write_text(
        json.dumps({"command": command, "search_seconds": seconds}, indent=2) + "\n"
    )
    started, ready, stop_sent = perf_counter(), None, None
    tail = b""
    with (
        (output / "stdout.txt").open("xb", buffering=0) as stdout,
        (output / "stderr.txt").open("xb", buffering=0) as stderr,
    ):
        with subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr
        ) as process:
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while selector.get_map():
                        now = perf_counter()
                        deadline = (ready or started) + (seconds if ready else 30)
                        if absolute_deadline is not None:
                            deadline = min(deadline, absolute_deadline)
                        if stop_sent is None and (
                            now >= deadline or (stop_event is not None and stop_event.is_set())
                        ):
                            process.send_signal(signal.SIGTERM)
                            stop_sent = now
                        if stop_sent is not None and now - stop_sent > 20:
                            raise TimeoutError("USW-LS did not finish after its stop request")
                        wait = (
                            0.05 if stop_sent is not None else min(0.05, max(0.0, deadline - now))
                        )
                        for event, _ in selector.select(timeout=wait):
                            chunk = os.read(event.fd, 65536)
                            if not chunk:
                                selector.unregister(event.fileobj)
                                continue
                            stdout.write(chunk)
                            tail += chunk
                            if ready is None and b"c benchmark-search-ready\n" in tail:
                                if barrier is not None:
                                    barrier.wait(timeout=30)
                                ready = perf_counter()
                                process.stdin.write(b"g\n")
                                process.stdin.flush()
                            tail = tail[-128:]
                    returncode = process.wait(timeout=5)
            finally:
                if process.poll() is None:
                    process.stdin.close()
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    ended = perf_counter()
    text = (output / "stdout.txt").read_text()
    models = re.findall(r"^v ([01]+)$", text, re.M)
    costs = re.findall(r"^o (\d+)$", text, re.M)
    flips = re.findall(r"^c benchmark-flips (\d+)$", text, re.M)
    if returncode != 0 or not models or not costs or not flips:
        raise RuntimeError(f"USW-LS produced no complete certificate: exit={returncode}")
    bits = np.asarray([int(x) for x in models[-1]], dtype=np.uint8)
    cost = int(instance.cost(bits))
    if cost != int(costs[-1]):
        raise AssertionError("USW-LS witness and reported objective disagree")
    if ready is None and barrier is not None:
        barrier.wait(timeout=30)
    return bits, {
        "method": "usw-ls",
        "best_cost": cost,
        "completed_flips": int(flips[-1]),
        "process_seconds": ended - started,
        "preparation_seconds": (ready - started) if ready else ended - started,
        "search_seconds": ended - ready if ready else 0.0,
        "requested_search_seconds": seconds,
        "stopped_at_iteration_boundary": stop_sent is not None,
        "timing_scope": (
            "Preparation to first incumbent is separate; wall clock starts at the handshake "
            "and stops at a complete iteration with final output included"
        ),
    }


def portfolio(instance, config, output, *, absolute_deadline=None, on_certificate=None):
    """Run independent CPU workers; a certified target can cancel the other workers."""
    workers = int(config["workers"])
    if not 2 <= workers <= 16:
        raise ValueError("Portfolio supports 2 through 16 workers")
    filename = output / "instance.wcnf"
    instance.write_wcnf(filename)
    barrier, stop = threading.Barrier(workers), threading.Event()
    fixed = config.get("fixed_flips_by_worker")
    if fixed is not None and len(fixed) != workers:
        raise ValueError("Replay flip counts must cover every worker")
    results = [None] * workers
    started = perf_counter()

    def worker(index):
        folder = output / f"worker-{index:02d}"
        folder.mkdir()
        local = {**config, "seed": (int(config["seed"]) + 1000003 * index) % (2**31 - 1)}
        if fixed is not None:
            local["fixed_flips"] = int(fixed[index])
        try:
            return run(
                instance,
                local,
                folder,
                input_path=filename,
                absolute_deadline=absolute_deadline,
                barrier=barrier,
                stop_event=stop,
            )
        except BaseException:
            stop.set()
            barrier.abort()
            raise

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, i): i for i in range(workers)}
        for future in as_completed(futures):
            index = futures[future]
            bits, summary = future.result()
            results[index] = (bits, summary)
            packet = output / f"worker-{index:02d}" / "states.npz"
            np.savez(packet, best_bits=bits)
            if on_certificate is not None:
                on_certificate(summary["best_cost"], packet, None)
            if fixed is None and summary["best_cost"] <= config.get("target_cost", 0):
                stop.set()
    bits = np.stack([item[0] for item in results])
    costs = np.asarray([item[1]["best_cost"] for item in results], dtype=np.int64)
    winner = int(costs.argmin())
    return {"best_bits": bits[winner], "portfolio_bits": bits, "portfolio_costs": costs}, {
        "method": "usw-portfolio",
        "workers": workers,
        "best_cost": int(costs[winner]),
        "winner": winner,
        "worker_summaries": [item[1] for item in results],
        "portfolio_seconds": perf_counter() - started,
        "cancel_on_first_target": fixed is None,
    }
