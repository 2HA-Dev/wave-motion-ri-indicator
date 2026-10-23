from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_ALLOWED_AGGREGATIONS = {"mean", "sum", "max"}


@dataclass(frozen=True)
class StructuredCellMaterialImageResult:
    cell_centers: np.ndarray
    cell_values: np.ndarray
    aggregation: str
    tolerance: float
    nx: int
    ny: int
    num_nodes: int
    num_cells: int
    min_nodes_per_cell: int
    local_min: float
    local_max: float
    local_sum: float
    local_l2_norm: float

    def as_dict(self, *, include_values: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": "structured_cell_material_image",
            "aggregation": self.aggregation,
            "tolerance": self.tolerance,
            "nx": self.nx,
            "ny": self.ny,
            "num_nodes": self.num_nodes,
            "num_cells": self.num_cells,
            "min_nodes_per_cell": self.min_nodes_per_cell,
            "local_min": self.local_min,
            "local_max": self.local_max,
            "local_sum": self.local_sum,
            "local_l2_norm": self.local_l2_norm,
        }

        if include_values:
            data["cell_centers"] = self.cell_centers.tolist()
            data["cell_values"] = self.cell_values.tolist()

        return data


def _validate_inputs(
    *,
    node_coordinates: np.ndarray,
    node_values: np.ndarray,
    nx: int,
    ny: int,
    aggregation: str,
    tolerance: float,
    min_nodes_per_cell: int,
) -> tuple[np.ndarray, np.ndarray]:
    if aggregation not in _ALLOWED_AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation '{aggregation}'. "
            f"Expected one of {sorted(_ALLOWED_AGGREGATIONS)}."
        )

    nx = int(nx)
    ny = int(ny)

    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be strictly positive.")

    tolerance = float(tolerance)

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be strictly positive and finite.")

    min_nodes_per_cell = int(min_nodes_per_cell)

    if min_nodes_per_cell < 1 or min_nodes_per_cell > 4:
        raise ValueError("min_nodes_per_cell must be between 1 and 4.")

    coordinates = np.asarray(node_coordinates, dtype=float)
    values = np.asarray(node_values, dtype=float)

    if coordinates.ndim != 2:
        raise ValueError("node_coordinates must be two-dimensional.")

    if coordinates.shape[1] != 2:
        raise ValueError("node_coordinates must have shape (n_nodes, 2).")

    if values.ndim != 1:
        raise ValueError("node_values must be one-dimensional.")

    if coordinates.shape[0] != values.size:
        raise ValueError("node_coordinates and node_values size mismatch.")

    if not np.all(np.isfinite(coordinates)):
        raise ValueError("node_coordinates contains non-finite entries.")

    if not np.all(np.isfinite(values)):
        raise ValueError("node_values contains non-finite entries.")

    return coordinates, values


def aggregate_nodal_image_to_structured_cells(
    *,
    node_coordinates: np.ndarray,
    node_values: np.ndarray,
    nx: int,
    ny: int,
    aggregation: str = "mean",
    tolerance: float = 1.0e-12,
    min_nodes_per_cell: int = 1,
) -> StructuredCellMaterialImageResult:
    coordinates, values = _validate_inputs(
        node_coordinates=node_coordinates,
        node_values=node_values,
        nx=nx,
        ny=ny,
        aggregation=aggregation,
        tolerance=tolerance,
        min_nodes_per_cell=min_nodes_per_cell,
    )

    nx = int(nx)
    ny = int(ny)
    min_nodes_per_cell = int(min_nodes_per_cell)

    if values.size == 0:
        return StructuredCellMaterialImageResult(
            cell_centers=np.empty((0, 2), dtype=np.float64),
            cell_values=np.empty((0,), dtype=np.float64),
            aggregation=aggregation,
            tolerance=float(tolerance),
            nx=nx,
            ny=ny,
            num_nodes=0,
            num_cells=0,
            min_nodes_per_cell=min_nodes_per_cell,
            local_min=0.0,
            local_max=0.0,
            local_sum=0.0,
            local_l2_norm=0.0,
        )

    rounded_keys = {
        (int(round(x / tolerance)), int(round(y / tolerance))): float(value)
        for (x, y), value in zip(coordinates, values, strict=True)
    }

    cell_centers: list[tuple[float, float]] = []
    cell_values: list[float] = []

    for j in range(ny):
        y0 = j / ny
        y1 = (j + 1) / ny

        for i in range(nx):
            x0 = i / nx
            x1 = (i + 1) / nx

            corner_coordinates = (
                (x0, y0),
                (x1, y0),
                (x0, y1),
                (x1, y1),
            )

            corner_values: list[float] = []

            for x, y in corner_coordinates:
                key = (int(round(x / tolerance)), int(round(y / tolerance)))

                if key in rounded_keys:
                    corner_values.append(rounded_keys[key])

            if len(corner_values) < min_nodes_per_cell:
                continue

            array = np.asarray(corner_values, dtype=float)

            if aggregation == "mean":
                value = float(np.mean(array))
            elif aggregation == "sum":
                value = float(np.sum(array))
            elif aggregation == "max":
                value = float(np.max(array))
            else:
                raise AssertionError("unreachable aggregation branch")

            cell_centers.append(((x0 + x1) * 0.5, (y0 + y1) * 0.5))
            cell_values.append(value)

    centers = np.asarray(cell_centers, dtype=np.float64).reshape((-1, 2))
    cell_array = np.asarray(cell_values, dtype=np.float64)

    local_min = float(np.min(cell_array)) if cell_array.size else 0.0
    local_max = float(np.max(cell_array)) if cell_array.size else 0.0
    local_sum = float(np.sum(cell_array))
    local_l2_norm = float(np.linalg.norm(cell_array))

    return StructuredCellMaterialImageResult(
        cell_centers=np.ascontiguousarray(centers, dtype=np.float64),
        cell_values=np.ascontiguousarray(cell_array, dtype=np.float64),
        aggregation=aggregation,
        tolerance=float(tolerance),
        nx=nx,
        ny=ny,
        num_nodes=int(values.size),
        num_cells=int(cell_array.size),
        min_nodes_per_cell=min_nodes_per_cell,
        local_min=local_min,
        local_max=local_max,
        local_sum=local_sum,
        local_l2_norm=local_l2_norm,
    )
