"""Gate-level netlists for unsigned array multipliers."""

from __future__ import annotations

from dataclasses import dataclass

from .gates import PenaltyProfile, get_gate


@dataclass(frozen=True)
class GateInstance:
    """One gate instance with logical signals in template port order."""

    name: str
    gate: str
    signals: tuple[str, ...]

    def __post_init__(self) -> None:
        template = get_gate(self.gate)
        if len(self.signals) != template.size:
            raise ValueError(f"{self.name} does not match the {self.gate} port count")
        if len(set(self.signals)) != len(self.signals):
            raise ValueError(f"{self.name} connects two gate ports to the same signal")


@dataclass(frozen=True)
class MultiplierNetlist:
    """A topologically ordered unsigned multiplier netlist."""

    factor_bits: int
    profile: PenaltyProfile
    gates: tuple[GateInstance, ...]
    factor_a: tuple[str, ...]
    factor_b: tuple[str, ...]
    product: tuple[str, ...]
    zero_signals: tuple[str, ...]
    signal_names: tuple[str, ...]

    @property
    def product_bits(self) -> int:
        return 2 * self.factor_bits

    def forward_assignment(self, factor_a: int, factor_b: int) -> dict[str, int]:
        """Evaluate the acyclic gate netlist for one pair of unsigned factors."""

        limit = 1 << self.factor_bits
        if not 0 <= factor_a < limit or not 0 <= factor_b < limit:
            raise ValueError("Factors do not fit the multiplier width")

        values: dict[str, int] = {}
        for index, signal in enumerate(self.factor_a):
            values[signal] = (factor_a >> index) & 1
        for index, signal in enumerate(self.factor_b):
            values[signal] = (factor_b >> index) & 1
        for signal in self.zero_signals:
            values[signal] = 0

        for instance in self.gates:
            if instance.gate == "and":
                left, right, output = instance.signals
                values[output] = values[left] & values[right]
            elif instance.gate == "full_adder":
                left, right, carry_in, sum_signal, carry_out = instance.signals
                total = values[left] + values[right] + values[carry_in]
                values[sum_signal] = total & 1
                values[carry_out] = (total >> 1) & 1
            else:
                raise ValueError(f"Unsupported multiplier gate: {instance.gate}")

        if set(values) != set(self.signal_names):
            missing = sorted(set(self.signal_names) - set(values))
            raise RuntimeError(f"Multiplier evaluation left signals undefined: {missing}")
        if self.decode_product(values) != factor_a * factor_b:
            raise RuntimeError("Multiplier netlist produced an incorrect product")
        return values

    def decode_product(self, assignment: dict[str, int]) -> int:
        return sum(int(assignment[signal]) << index for index, signal in enumerate(self.product))


def _ordered_unique(groups: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            if value not in seen:
                result.append(value)
                seen.add(value)
    return tuple(result)


def build_unsigned_array_multiplier(
    factor_bits: int,
    *,
    profile: PenaltyProfile = "aadit",
) -> MultiplierNetlist:
    """Build the regular AND/FA array used by the Aadit factorizer.

    The construction uses ``m**2`` AND gates, ``m*(m-1)`` full adders, and
    ``m`` distinct zero pins. The latter keep the published ``3*m**2 + m``
    fused-node count while making every arithmetic boundary condition explicit.
    """

    if factor_bits < 2:
        raise ValueError("factor_bits must be at least two")
    if profile not in ("canonical", "aadit"):
        raise ValueError(f"Unknown gate penalty profile: {profile}")

    width = int(factor_bits)
    factor_a = tuple(f"factor_a[{index}]" for index in range(width))
    factor_b = tuple(f"factor_b[{index}]" for index in range(width))
    zero_signals = tuple(f"zero[{index}]" for index in range(width))
    partial = tuple(
        tuple(f"partial[{left},{right}]" for right in range(width)) for left in range(width)
    )

    gates: list[GateInstance] = []
    for left in range(width):
        for right in range(width):
            gates.append(
                GateInstance(
                    name=f"and[{left},{right}]",
                    gate="and",
                    signals=(factor_a[left], factor_b[right], partial[left][right]),
                )
            )

    product: list[str | None] = [None] * (2 * width)
    product[0] = partial[0][0]
    accumulator = [partial[index][0] for index in range(1, width)] + [zero_signals[0]]

    for row in range(1, width):
        carry = zero_signals[row]
        sums: list[str] = []
        for column in range(width):
            sum_signal = f"sum[{row},{column}]"
            carry_out = f"carry[{row},{column}]"
            gates.append(
                GateInstance(
                    name=f"full_adder[{row},{column}]",
                    gate="full_adder",
                    signals=(
                        accumulator[column],
                        partial[column][row],
                        carry,
                        sum_signal,
                        carry_out,
                    ),
                )
            )
            sums.append(sum_signal)
            carry = carry_out

        product[row] = sums[0]
        if row == width - 1:
            for column in range(1, width):
                product[row + column] = sums[column]
            product[2 * width - 1] = carry
        else:
            accumulator = [*sums[1:], carry]

    if any(signal is None for signal in product):
        raise RuntimeError("Array multiplier did not define every product bit")
    product_signals = tuple(str(signal) for signal in product)
    gate_signals = tuple(signal for gate in gates for signal in gate.signals)
    signal_names = _ordered_unique(
        (factor_a, factor_b, product_signals, zero_signals, gate_signals)
    )
    expected_signals = 3 * width * width + width
    if len(signal_names) != expected_signals:
        raise RuntimeError(
            f"Unexpected fused signal count: {len(signal_names)} != {expected_signals}"
        )

    return MultiplierNetlist(
        factor_bits=width,
        profile=profile,
        gates=tuple(gates),
        factor_a=factor_a,
        factor_b=factor_b,
        product=product_signals,
        zero_signals=zero_signals,
        signal_names=signal_names,
    )
