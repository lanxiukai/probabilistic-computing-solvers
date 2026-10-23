"""A small, lazy registry for problem-specific configuration runners."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast


class ConfigRunner(Protocol):
    """A runner validates its own problem, solves it, and returns an artifact directory."""

    def __call__(
        self,
        config: Mapping[str, Any],
        *,
        output_override: Path | None = None,
        overwrite: bool = False,
        repository: Path | None = None,
    ) -> Path: ...


@dataclass(frozen=True)
class SolverSpec:
    """Describe a runner without importing its numerical implementation."""

    name: str
    entry_point: str
    description: str
    input_fields: tuple[str, ...] = ()
    working_directory_fields: tuple[str, ...] = ()

    def load(self) -> ConfigRunner:
        module, attribute = self.entry_point.split(":", 1)
        runner = getattr(import_module(module), attribute)
        if not callable(runner):
            raise TypeError(f"Solver entry point is not callable: {self.entry_point}")
        return cast(ConfigRunner, runner)


class SolverRegistry:
    """Register additional problems without editing the common CLI or existing solvers."""

    def __init__(self, specifications: Iterable[SolverSpec] = ()) -> None:
        self._specifications: dict[str, SolverSpec] = {}
        for specification in specifications:
            self.register(specification)

    def register(self, specification: SolverSpec) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9-]*", specification.name):
            raise ValueError("Solver names must be lowercase slugs")
        if specification.name in self._specifications:
            raise ValueError(f"Solver is already registered: {specification.name}")
        if not re.fullmatch(
            r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", specification.entry_point
        ):
            raise ValueError("Solver entry points use 'module:function' syntax")
        fields = (*specification.input_fields, *specification.working_directory_fields)
        if len(set(fields)) != len(fields) or any(
            not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", field) for field in fields
        ):
            raise ValueError("File and working-directory fields must be distinct dotted keys")
        self._specifications[specification.name] = specification

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specifications))

    def get(self, name: str) -> SolverSpec:
        try:
            return self._specifications[name]
        except KeyError as error:
            raise ValueError(
                f"Unknown solver '{name}'; choose from {', '.join(self.names)}"
            ) from error


def default_registry() -> SolverRegistry:
    """Return only solvers present in this source distribution."""
    from combinatorial_solvers.public_api import registry

    return registry()
