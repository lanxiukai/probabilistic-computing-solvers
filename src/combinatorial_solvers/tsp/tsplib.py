"""A strict TSPLIB95 parser for the symmetric instances used by this project."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .instance import TSPInstance

_SECTION_NAMES = {
    "NODE_COORD_SECTION",
    "EDGE_WEIGHT_SECTION",
    "DISPLAY_DATA_SECTION",
    "FIXED_EDGES_SECTION",
    "TOUR_SECTION",
}


def _split_tsplib(text: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    headers: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    active_section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper == "EOF":
            break
        if upper in _SECTION_NAMES:
            active_section = upper
            sections.setdefault(active_section, [])
            continue
        if active_section is not None:
            sections[active_section].append(line)
            continue
        if ":" in line:
            key, value = line.split(":", maxsplit=1)
        else:
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"Invalid TSPLIB header line: {line}")
            key, value = fields
        headers[key.strip().upper()] = value.strip()
    return headers, sections


def _parse_coordinates(lines: list[str], dimension: int) -> np.ndarray:
    indexed: dict[int, tuple[float, ...]] = {}
    coordinate_size: int | None = None
    for line in lines:
        fields = line.split()
        if len(fields) not in (3, 4):
            raise ValueError(f"Invalid TSPLIB coordinate line: {line}")
        node = int(fields[0])
        values = tuple(float(value) for value in fields[1:])
        if node in indexed:
            raise ValueError(f"Duplicate TSPLIB node identifier: {node}")
        coordinate_size = coordinate_size or len(values)
        if len(values) != coordinate_size or not all(math.isfinite(value) for value in values):
            raise ValueError("TSPLIB coordinates must have a consistent finite dimension")
        indexed[node] = values
    expected = set(range(1, dimension + 1))
    if set(indexed) != expected:
        raise ValueError("TSPLIB node identifiers must be exactly 1 through DIMENSION")
    return np.asarray([indexed[node] for node in range(1, dimension + 1)], dtype=np.float64)


def _nearest_integer(value: float) -> int:
    return int(math.floor(value + 0.5))


def _geographical_radians(value: float) -> float:
    degrees = math.trunc(value)
    minutes = value - degrees
    return math.pi * (degrees + 5.0 * minutes / 3.0) / 180.0


def _coordinate_distances(coordinates: np.ndarray, edge_weight_type: str) -> np.ndarray:
    dimension = coordinates.shape[0]
    distance = np.zeros((dimension, dimension), dtype=np.int64)
    for left in range(dimension):
        for right in range(left + 1, dimension):
            delta = coordinates[left] - coordinates[right]
            if edge_weight_type == "EUC_2D":
                value = _nearest_integer(math.hypot(float(delta[0]), float(delta[1])))
            elif edge_weight_type == "ATT":
                pseudo = math.sqrt(float(delta[0] ** 2 + delta[1] ** 2) / 10.0)
                rounded = _nearest_integer(pseudo)
                value = rounded + 1 if rounded < pseudo else rounded
            elif edge_weight_type == "GEO":
                latitude_left = _geographical_radians(float(coordinates[left, 0]))
                longitude_left = _geographical_radians(float(coordinates[left, 1]))
                latitude_right = _geographical_radians(float(coordinates[right, 0]))
                longitude_right = _geographical_radians(float(coordinates[right, 1]))
                q1 = math.cos(longitude_left - longitude_right)
                q2 = math.cos(latitude_left - latitude_right)
                q3 = math.cos(latitude_left + latitude_right)
                argument = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
                central_angle = math.acos(max(-1.0, min(1.0, argument)))
                value = int(6378.388 * central_angle + 1.0)
            else:
                raise ValueError(f"Unsupported coordinate edge weight type: {edge_weight_type}")
            distance[left, right] = distance[right, left] = value
    return distance


def _explicit_positions(dimension: int, edge_weight_format: str) -> list[tuple[int, int]]:
    if edge_weight_format == "FULL_MATRIX":
        return [(row, column) for row in range(dimension) for column in range(dimension)]
    if edge_weight_format == "UPPER_ROW":
        return [(row, column) for row in range(dimension) for column in range(row + 1, dimension)]
    if edge_weight_format == "LOWER_ROW":
        return [(row, column) for row in range(dimension) for column in range(row)]
    if edge_weight_format == "UPPER_DIAG_ROW":
        return [(row, column) for row in range(dimension) for column in range(row, dimension)]
    if edge_weight_format == "LOWER_DIAG_ROW":
        return [(row, column) for row in range(dimension) for column in range(row + 1)]
    if edge_weight_format == "UPPER_COL":
        return [(row, column) for column in range(1, dimension) for row in range(column)]
    if edge_weight_format == "LOWER_COL":
        return [
            (row, column) for column in range(dimension - 1) for row in range(column + 1, dimension)
        ]
    if edge_weight_format == "UPPER_DIAG_COL":
        return [(row, column) for column in range(dimension) for row in range(column + 1)]
    if edge_weight_format == "LOWER_DIAG_COL":
        return [(row, column) for column in range(dimension) for row in range(column, dimension)]
    raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}")


def _explicit_distances(lines: list[str], dimension: int, edge_weight_format: str) -> np.ndarray:
    try:
        values = [int(value) for line in lines for value in line.split()]
    except ValueError as error:
        raise ValueError("EXPLICIT edge weights must be integers") from error
    positions = _explicit_positions(dimension, edge_weight_format)
    if len(values) != len(positions):
        raise ValueError(
            f"EDGE_WEIGHT_SECTION contains {len(values)} values; expected {len(positions)}"
        )
    distance = np.zeros((dimension, dimension), dtype=np.int64)
    full_matrix = edge_weight_format == "FULL_MATRIX"
    for (row, column), value in zip(positions, values, strict=True):
        distance[row, column] = value
        if not full_matrix:
            distance[column, row] = value
    return distance


def parse_tsplib(text: str, *, optimum: int | None = None) -> TSPInstance:
    """Parse one symmetric TSPLIB95 problem from text."""

    headers, sections = _split_tsplib(text)
    try:
        name = headers["NAME"]
        problem_type = headers["TYPE"].upper()
        dimension = int(headers["DIMENSION"])
        edge_weight_type = headers["EDGE_WEIGHT_TYPE"].upper()
    except KeyError as error:
        raise ValueError(f"Missing required TSPLIB header: {error.args[0]}") from error
    if problem_type != "TSP":
        raise ValueError("Only symmetric TYPE=TSP instances are supported")
    if dimension < 2:
        raise ValueError("TSPLIB DIMENSION must be at least two")

    coordinates = None
    if edge_weight_type in {"EUC_2D", "GEO", "ATT"}:
        if "NODE_COORD_SECTION" not in sections:
            raise ValueError(f"{edge_weight_type} requires NODE_COORD_SECTION")
        coordinates = _parse_coordinates(sections["NODE_COORD_SECTION"], dimension)
        if coordinates.shape[1] != 2:
            raise ValueError(f"{edge_weight_type} requires two-dimensional coordinates")
        distance = _coordinate_distances(coordinates, edge_weight_type)
    elif edge_weight_type == "EXPLICIT":
        if "EDGE_WEIGHT_SECTION" not in sections:
            raise ValueError("EXPLICIT requires EDGE_WEIGHT_SECTION")
        edge_weight_format = headers.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX").upper()
        distance = _explicit_distances(
            sections["EDGE_WEIGHT_SECTION"], dimension, edge_weight_format
        )
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}")

    metadata = {key: value for key, value in headers.items() if key not in {"NAME"}}
    return TSPInstance(
        name=name,
        distance=distance,
        edge_weight_type=edge_weight_type,
        coordinates=coordinates,
        optimum=optimum,
        metadata=metadata,
    )


def load_tsplib(path: str | Path, *, optimum: int | None = None) -> TSPInstance:
    source = Path(path)
    return parse_tsplib(source.read_text(encoding="utf-8"), optimum=optimum)


def parse_tsplib_tour(text: str, *, expected_dimension: int | None = None) -> tuple[int, ...]:
    """Parse the first TOUR_SECTION and convert one-based identifiers to zero-based."""

    headers, sections = _split_tsplib(text)
    values: list[int] = []
    for line in sections.get("TOUR_SECTION", []):
        for token in line.split():
            city = int(token)
            if city == -1:
                break
            values.append(city)
        else:
            continue
        break
    if not values:
        raise ValueError("TOUR_SECTION does not contain a tour")
    dimension_value = headers.get("DIMENSION")
    declared_dimension = int(dimension_value) if dimension_value is not None else None
    dimension = expected_dimension if expected_dimension is not None else declared_dimension
    if dimension is not None and len(values) != dimension:
        raise ValueError("TSPLIB tour length does not match DIMENSION")
    if dimension is not None and set(values) != set(range(1, dimension + 1)):
        raise ValueError("TSPLIB tour must contain identifiers 1 through DIMENSION")
    return tuple(city - 1 for city in values)


def load_tsplib_tour(
    path: str | Path,
    *,
    expected_dimension: int | None = None,
) -> tuple[int, ...]:
    source = Path(path)
    return parse_tsplib_tour(
        source.read_text(encoding="utf-8"), expected_dimension=expected_dimension
    )
