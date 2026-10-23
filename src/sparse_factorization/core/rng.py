"""Non-consuming audits for per-batch PyTorch random-number streams."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class RngStreamAudit:
    """Generator key, opaque state fingerprints, and logical request counts."""

    stream_name: str
    seed: int
    device: str
    engine: str
    counter_based: bool
    offset_unit: str | None
    start_offset: int | None
    end_offset: int | None
    start_state_sha256: str
    end_state_sha256: str
    state_bytes: int
    random_calls_by_operation: dict[str, int]
    logical_values_by_operation: dict[str, int]
    exact_counter_consumption_known: bool
    note: str


def _state_sha256(state: torch.Tensor) -> str:
    values = state.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(values.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(values.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _counter_state_sha256(engine: str, seed: int, offset: int) -> str:
    payload = f"{engine}\0{int(seed)}\0{int(offset)}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _engine(device: torch.device) -> tuple[str, bool, str | None]:
    if device.type == "cuda":
        return "pytorch_cuda_philox", True, "opaque_philox_offset"
    if device.type == "mps":
        return "pytorch_mps_philox", True, "opaque_philox_offset"
    if device.type == "cpu":
        return "pytorch_cpu_mt19937", False, None
    return f"pytorch_{device.type}_generator", False, None


def generator_offset(generator: torch.Generator) -> int | None:
    """Return a Philox-style offset when the backend exposes one."""

    try:
        return int(generator.get_offset())
    except RuntimeError:
        return None


class RngAuditRecorder:
    """Record logical requests without drawing from or mutating the generator."""

    def __init__(
        self,
        generator: torch.Generator,
        *,
        seed: int,
        stream_name: str,
    ) -> None:
        if not stream_name:
            raise ValueError("stream_name must be non-empty")
        self._generator = generator
        self._seed = int(seed)
        self._stream_name = stream_name
        self._device = torch.device(generator.device)
        self._start_state = generator.get_state()
        self._start_offset = generator_offset(generator)
        self._calls: dict[str, int] = {}
        self._values: dict[str, int] = {}

    def record(self, operation: str, logical_values: int) -> None:
        if not operation or logical_values < 0:
            raise ValueError("RNG audit operations need a name and nonnegative value count")
        self._calls[operation] = self._calls.get(operation, 0) + 1
        self._values[operation] = self._values.get(operation, 0) + int(logical_values)

    def finish(self) -> RngStreamAudit:
        end_state = self._generator.get_state()
        end_offset = generator_offset(self._generator)
        engine, counter_based, offset_unit = _engine(self._device)
        return RngStreamAudit(
            stream_name=self._stream_name,
            seed=self._seed,
            device=str(self._device),
            engine=engine,
            counter_based=counter_based,
            offset_unit=offset_unit,
            start_offset=self._start_offset,
            end_offset=end_offset,
            start_state_sha256=_state_sha256(self._start_state),
            end_state_sha256=_state_sha256(end_state),
            state_bytes=int(self._start_state.numel() * self._start_state.element_size()),
            random_calls_by_operation=dict(sorted(self._calls.items())),
            logical_values_by_operation=dict(sorted(self._values.items())),
            exact_counter_consumption_known=False,
            note=(
                "CUDA/MPS offsets are recorded as opaque backend state. Logical value counts "
                "do not imply an offset unit because vectorized operators may reserve counters "
                "in implementation-dependent blocks. Reproducibility is scoped to the recorded "
                "PyTorch version, device class, and execution path."
            ),
        )


class CounterRngAuditRecorder:
    """Reserve exact non-overlapping logical offsets for a versioned counter RNG."""

    engine = "triton_philox_4x32_10_counter_v1"
    offset_unit = "logical_spin_offset_u64"

    def __init__(
        self,
        *,
        seed: int,
        device: str | torch.device,
        stream_name: str,
        start_offset: int = 0,
    ) -> None:
        if not stream_name:
            raise ValueError("stream_name must be non-empty")
        if seed < 0 or seed >= 1 << 63:
            raise ValueError("counter RNG seed must fit a nonnegative signed 64-bit integer")
        if start_offset < 0 or start_offset >= 1 << 63:
            raise ValueError("counter RNG offset must fit a nonnegative signed 64-bit integer")
        self._seed = int(seed)
        self._device = torch.device(device)
        self._stream_name = stream_name
        self._start_offset = int(start_offset)
        self._offset = int(start_offset)
        self._calls: dict[str, int] = {}
        self._values: dict[str, int] = {}

    @property
    def next_offset(self) -> int:
        return self._offset

    def reserve(self, operation: str, logical_values: int) -> int:
        """Return the first offset and advance by exactly ``logical_values``."""

        if not operation or logical_values < 0:
            raise ValueError("Counter RNG reservations need a name and nonnegative size")
        end = self._offset + int(logical_values)
        if end >= 1 << 63:
            raise OverflowError("Counter RNG logical offset exceeds signed 64-bit range")
        start = self._offset
        self._offset = end
        self._calls[operation] = self._calls.get(operation, 0) + 1
        self._values[operation] = self._values.get(operation, 0) + int(logical_values)
        return start

    def finish(self) -> RngStreamAudit:
        return RngStreamAudit(
            stream_name=self._stream_name,
            seed=self._seed,
            device=str(self._device),
            engine=self.engine,
            counter_based=True,
            offset_unit=self.offset_unit,
            start_offset=self._start_offset,
            end_offset=self._offset,
            start_state_sha256=_counter_state_sha256(
                self.engine,
                self._seed,
                self._start_offset,
            ),
            end_state_sha256=_counter_state_sha256(
                self.engine,
                self._seed,
                self._offset,
            ),
            state_bytes=16,
            random_calls_by_operation=dict(sorted(self._calls.items())),
            logical_values_by_operation=dict(sorted(self._values.items())),
            exact_counter_consumption_known=True,
            note=(
                "Versioned Triton Philox protocol: one 10-round 4x32 counter invocation per "
                "logical spin, addressed by an explicit contiguous unsigned logical offset."
            ),
        )


def derive_stream_seed(base_seed: int, namespace: str) -> int:
    """Derive a stable nonnegative 63-bit seed for a named auxiliary stream."""

    if not namespace:
        raise ValueError("RNG stream namespace must be non-empty")
    payload = f"ising-solvers-rng-v1\0{int(base_seed)}\0{namespace}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def audit_stream_partition(audits: Sequence[RngStreamAudit]) -> dict[str, object]:
    """Summarize generator-key reuse within one N-by-M experiment bundle."""

    if not audits:
        raise ValueError("At least one RNG stream audit is required")
    keys = [(audit.engine, audit.device, audit.seed) for audit in audits]
    unique = len(keys) == len(set(keys))
    duplicate_keys = sorted(
        {
            f"{engine}|{device}|{seed}"
            for engine, device, seed in keys
            if keys.count((engine, device, seed)) > 1
        }
    )
    counter_ranges_valid = all(
        not audit.counter_based
        or (
            audit.start_offset is not None
            and audit.end_offset is not None
            and audit.end_offset >= audit.start_offset
        )
        for audit in audits
    )
    if not counter_ranges_valid:
        raise ValueError("A counter-based stream has an invalid or unavailable offset range")
    overlapping_counter_ranges = []
    for left_index, left in enumerate(audits):
        if not left.counter_based or left.start_offset is None or left.end_offset is None:
            continue
        for right in audits[left_index + 1 :]:
            if (left.engine, left.device, left.seed) != (right.engine, right.device, right.seed):
                continue
            if right.start_offset is None or right.end_offset is None:
                continue
            if max(left.start_offset, right.start_offset) < min(left.end_offset, right.end_offset):
                overlapping_counter_ranges.append(
                    {
                        "counter_key": f"{left.engine}|{left.device}|{left.seed}",
                        "left_stream": left.stream_name,
                        "right_stream": right.stream_name,
                    }
                )
    return {
        "schema_version": 1,
        "strategy": "one_explicit_torch_generator_per_batch",
        "stream_key": ["engine", "device", "seed"],
        "unique_counter_keys": unique,
        "duplicate_counter_keys": duplicate_keys,
        "all_streams_counter_based": all(audit.counter_based for audit in audits),
        "counter_ranges_valid": counter_ranges_valid,
        "overlapping_counter_ranges": overlapping_counter_ranges,
        "governance_passed": unique and not overlapping_counter_ranges,
        "statistical_independence_proven": False,
        "cross_release_reproducibility_claimed": False,
        "batches": [asdict(audit) for audit in audits],
        "limitations": (
            "Distinct keys and valid state transitions prevent accidental seed reuse inside the "
            "bundle; they do not constitute a statistical independence proof. PyTorch does not "
            "guarantee identical random sequences across releases or CPU/CUDA platforms."
        ),
    }


def validate_rng_audit_payload(payload: Mapping[str, object]) -> None:
    """Reject malformed persisted audit payloads used by benchmark comparisons."""

    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported or invalid RNG audit payload")
    batches = payload.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("RNG audit payload has no batches")
    identities = []
    for raw in batches:
        if not isinstance(raw, Mapping):
            raise ValueError("RNG audit batch entries must be mappings")
        identities.append(
            (raw.get("stream_name"), raw.get("engine"), raw.get("device"), raw.get("seed"))
        )
    if len(identities) != len(set(identities)):
        raise ValueError("Persisted RNG audit contains duplicate stream identities")
