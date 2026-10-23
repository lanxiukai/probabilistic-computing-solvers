"""Compact native-event recording for the standalone publication runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sparse_factorization.solver.publication_io import atomic_npz as _atomic_npz


def witness_packet(contexts: list[dict], witnesses: list[dict]) -> dict[str, np.ndarray]:
    """Serialize boundary contexts once even when several groups hit in one transition."""

    return {
        **{f"context_{key}": np.stack([row[key] for row in contexts]) for key in contexts[0]},
        **{f"witness_{key}": np.stack([row[key] for row in witnesses]) for key in witnesses[0]},
    }


SELECTED_CONTEXT_LAYOUT = "selected_slots_v1"


def compact_witness_packet(contexts: list[dict], witnesses: list[dict]) -> dict[str, np.ndarray]:
    """Serialize only representative slots, retaining their original population indices."""
    counts = np.array([len(row["slot_indices"]) for row in contexts], dtype=np.int64)
    return {
        "context_layout": np.array(SELECTED_CONTEXT_LAYOUT),
        "context_state_offsets": np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts))),
        "context_slot_indices": np.concatenate([row["slot_indices"] for row in contexts]),
        "context_states": np.concatenate([row["states"] for row in contexts], axis=0),
        **{
            f"context_{key}": np.stack([row[key] for row in contexts])
            for key in contexts[0]
            if key not in {"slot_indices", "states"}
        },
        **{f"witness_{key}": np.stack([row[key] for row in witnesses]) for key in witnesses[0]},
    }


def compact_full_packet(packet: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Create a compact audit copy of an existing full-population witness packet."""
    if "context_layout" in packet:
        raise ValueError("The conversion input must use the original full-population layout")
    contexts = []
    for index in range(len(packet["context_states"])):
        rows = np.flatnonzero(packet["witness_context"] == index)
        slots = np.unique(packet["witness_slot"][rows])
        if not len(slots):
            raise ValueError("Every saved context must support an actual witness")
        context = {
            key.removeprefix("context_"): value[index]
            for key, value in packet.items()
            if key.startswith("context_") and key != "context_states"
        }
        context["slot_indices"] = slots
        context["states"] = packet["context_states"][index, slots].copy()
        contexts.append(context)
    witnesses = [
        {
            key.removeprefix("witness_"): value[index]
            for key, value in packet.items()
            if key.startswith("witness_")
        }
        for index in range(len(packet["witness_slot"]))
    ]
    return compact_witness_packet(contexts, witnesses)


def selected_context(
    packet: dict[str, np.ndarray], index: int, population: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return original slot identities and pre-transition physical states."""
    low, high = packet["context_state_offsets"][index : index + 2]
    slots = packet["context_slot_indices"][low:high]
    if len(np.unique(slots)) != len(slots) or np.any((slots < 0) | (slots >= population)):
        raise ValueError("Recorded slots must be unique and inside the original population")
    return slots, packet["context_states"][low:high]


class NativeEventRecorder:
    """Keep every slot's first passage and replay evidence for its group's first hit."""

    def __init__(
        self,
        engine,
        betas: torch.Tensor,
        auxiliary,
        group_size: int,
        *,
        auxiliary_kind: str = "circuit",
        context_layout: str = "full_population_v1",
    ) -> None:
        if engine.replicas % group_size:
            raise ValueError("Native event groups must divide the replica population")
        if context_layout not in {"full_population_v1", SELECTED_CONTEXT_LAYOUT}:
            raise ValueError("Unknown native witness context layout")
        self.context_layout = context_layout
        self.engine, self.betas, self.auxiliary = engine, betas, auxiliary
        self.fixed_betas_host = (
            betas.cpu().numpy()
            if getattr(auxiliary, "history_backend", None) == "triton_device_v1"
            else None
        )
        self.group_size = group_size
        self.auxiliary_kind = auxiliary_kind
        self.seen = np.zeros(engine.replicas, dtype=bool)
        self.group_seen = np.zeros(engine.replicas // group_size, dtype=bool)
        self.lower = np.full(engine.replicas, np.nan)
        self.upper = np.full(engine.replicas, np.nan)
        self.kinds = np.full(engine.replicas, "", dtype="U16")
        self.events = np.full(engine.replicas, -1, dtype=np.int64)
        self.transitions = np.full(engine.replicas, -1, dtype=np.int64)
        self.contexts: list[dict] = []
        self.witnesses: list[dict] = []

    def context(self) -> dict:
        return {
            "states": self.engine.states.clone(),
            "raw_sweep": self.engine.global_sweep,
            "gibbs_offset": self.engine.counter_offset,
            "betas": self.betas.cpu().numpy()
            if self.fixed_betas_host is None
            else self.fixed_betas_host.copy(),
            "auxiliary_events": self.auxiliary.events if self.auxiliary else 0,
            "auxiliary_rng_state": self.auxiliary.generator.get_state().numpy()
            if self.auxiliary
            else np.empty(0, dtype=np.uint8),
        }

    def observe(self, before: dict, kind: str, lower: float, upper: float, transition: int) -> int:
        cached = getattr(self.auxiliary, "first_hit_sweeps_host", None)
        first = (
            cached
            if kind == self.auxiliary_kind and cached is not None
            else self.engine.first_hit_sweeps.cpu().numpy()
        )
        fresh = np.flatnonzero((first >= 0) & ~self.seen)
        if not len(fresh):
            return 0
        event = self.auxiliary.events if kind == self.auxiliary_kind else -1
        self.seen[fresh] = True
        self.lower[fresh], self.upper[fresh] = lower, upper
        self.kinds[fresh], self.events[fresh], self.transitions[fresh] = kind, event, transition
        chosen = []
        for group in np.unique(fresh // self.group_size):
            if self.group_seen[group]:
                continue
            slots = fresh[fresh // self.group_size == group]
            slot = min(slots, key=lambda index: (first[index], index))
            chosen.append((int(group), int(slot)))
            self.group_seen[group] = True
        if chosen:
            if self.context_layout == SELECTED_CONTEXT_LAYOUT:
                slots = np.array([slot for _, slot in chosen], dtype=np.int64)
                before["slot_indices"] = slots
                before["states"] = before["states"][slots.tolist()].cpu().numpy()
            else:
                before["states"] = before["states"].cpu().numpy()
            context_index = len(self.contexts)
            self.contexts.append(before)
            for group, slot in chosen:
                self.witnesses.append(
                    {
                        "context": context_index,
                        "group": group,
                        "slot": slot,
                        "raw_sweep": int(first[slot]),
                        "kind": kind,
                        "auxiliary_event": event,
                        "transition": transition,
                        "state": self.engine.first_hit_states[slot].cpu().numpy().copy(),
                        "wall_lower": lower,
                        "wall_upper": upper,
                    }
                )
        return len(chosen)

    def packet(self) -> dict[str, np.ndarray]:
        if self.context_layout == SELECTED_CONTEXT_LAYOUT:
            return compact_witness_packet(self.contexts, self.witnesses)
        return witness_packet(self.contexts, self.witnesses)

    def write(self, output: Path) -> None:
        _atomic_npz(
            output / "first_hits.npz",
            first_hit_sweeps=self.engine.first_hit_sweeps.cpu().numpy(),
            first_hit_kind=self.kinds,
            first_hit_auxiliary_event=self.events,
            first_hit_transition=self.transitions,
            first_hit_wall_lower=self.lower,
            first_hit_wall_upper=self.upper,
        )
        if self.witnesses:
            _atomic_npz(
                output / "first_hit_certificates.npz",
                **self.packet(),
            )

    def within_budget(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        slots = self.seen & (self.upper <= seconds)
        return slots, slots.reshape(-1, self.group_size).any(axis=1)
