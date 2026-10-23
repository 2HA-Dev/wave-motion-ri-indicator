from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_ALLOWED_AGGREGATIONS = {"mean", "sum", "max"}


@dataclass(frozen=True)
class NodalMaterialImageResult:
    node_coordinates: np.ndarray
    node_values: np.ndarray
    aggregation: str
    tolerance: float
    num_dofs: int
    num_nodes: int
    local_min: float
    local_max: float
    local_sum: float
    local_l2_norm: float

    def as_dict(self, *, include_values: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": "nodal_material_image",
            "aggregation": self.aggregation,
            "tolerance": self.tolerance,
            "num_dofs": self.num_dofs,
            "num_nodes": self.num_nodes,
            "local_min": self.local_min,
            "local_max": self.local_max,
            "local_sum": self.local_sum,
            "local_l2_norm": self.local_l2_norm,
        }

        if include_values:
            data["node_coordinates"] = self.node_coordinates.tolist()
            data["node_values"] = self.node_values.tolist()

        return data


def aggregate_dof_image_to_nodes(
    *,
    local_coordinates: np.ndarray,
    local_values: np.ndarray,
    aggregation: str = "mean",
    tolerance: float = 1.0e-12,
) -> NodalMaterialImageResult:
    if aggregation not in _ALLOWED_AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation '{aggregation}'. "
            f"Expected one of {sorted(_ALLOWED_AGGREGATIONS)}."
        )

    tolerance = float(tolerance)

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be strictly positive and finite.")

    coordinates = np.asarray(local_coordinates, dtype=float)
    values = np.asarray(local_values, dtype=float)

    if coordinates.ndim != 2:
        raise ValueError("local_coordinates must be two-dimensional.")

    if coordinates.shape[1] != 2:
        raise ValueError("local_coordinates must have shape (n_dofs, 2).")

    if values.ndim != 1:
        raise ValueError("local_values must be one-dimensional.")

    if coordinates.shape[0] != values.size:
        raise ValueError("local_coordinates and local_values size mismatch.")

    if not np.all(np.isfinite(coordinates)):
        raise ValueError("local_coordinates contains non-finite entries.")

    if not np.all(np.isfinite(values)):
        raise ValueError("local_values contains non-finite entries.")

    if values.size == 0:
        return NodalMaterialImageResult(
            node_coordinates=np.empty((0, 2), dtype=np.float64),
            node_values=np.empty((0,), dtype=np.float64),
            aggregation=aggregation,
            tolerance=tolerance,
            num_dofs=0,
            num_nodes=0,
            local_min=0.0,
            local_max=0.0,
            local_sum=0.0,
            local_l2_norm=0.0,
        )

    scaled = np.rint(coordinates / tolerance).astype(np.int64)

    unique_keys, inverse = np.unique(
        scaled,
        axis=0,
        return_inverse=True,
    )

    num_nodes = int(unique_keys.shape[0])

    node_coordinates = np.zeros((num_nodes, 2), dtype=np.float64)
    node_values = np.zeros(num_nodes, dtype=np.float64)
    counts = np.zeros(num_nodes, dtype=np.int64)

    if aggregation == "max":
        node_values.fill(-np.inf)

    for dof_index, node_index in enumerate(inverse):
        node_coordinates[node_index] += coordinates[dof_index]
        counts[node_index] += 1

        if aggregation in {"mean", "sum"}:
            node_values[node_index] += values[dof_index]
        elif aggregation == "max":
            node_values[node_index] = max(node_values[node_index], values[dof_index])

    node_coordinates = node_coordinates / counts[:, None]

    if aggregation == "mean":
        node_values = node_values / counts

    local_min = float(np.min(node_values)) if node_values.size else 0.0
    local_max = float(np.max(node_values)) if node_values.size else 0.0
    local_sum = float(np.sum(node_values))
    local_l2_norm = float(np.linalg.norm(node_values))

    return NodalMaterialImageResult(
        node_coordinates=np.ascontiguousarray(node_coordinates, dtype=np.float64),
        node_values=np.ascontiguousarray(node_values, dtype=np.float64),
        aggregation=aggregation,
        tolerance=tolerance,
        num_dofs=int(values.size),
        num_nodes=num_nodes,
        local_min=local_min,
        local_max=local_max,
        local_sum=local_sum,
        local_l2_norm=local_l2_norm,
    )
