from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CellDefectLocalizationMetrics:
    defect_center: tuple[float, float]
    defect_radius: float
    num_cells: int
    num_inside_cells: int
    num_outside_cells: int
    max_value: float
    max_center: tuple[float, float]
    max_distance_to_defect_center: float
    max_inside_defect: bool
    total_indicator_mass: float
    inside_indicator_mass: float
    outside_indicator_mass: float
    inside_mass_fraction: float
    inside_mean: float
    outside_mean: float
    inside_outside_mean_ratio: float
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "cell_defect_localization_metrics",
            "status": self.status,
            "defect_center": list(self.defect_center),
            "defect_radius": self.defect_radius,
            "num_cells": self.num_cells,
            "num_inside_cells": self.num_inside_cells,
            "num_outside_cells": self.num_outside_cells,
            "max_value": self.max_value,
            "max_center": list(self.max_center),
            "max_distance_to_defect_center": self.max_distance_to_defect_center,
            "max_inside_defect": self.max_inside_defect,
            "total_indicator_mass": self.total_indicator_mass,
            "inside_indicator_mass": self.inside_indicator_mass,
            "outside_indicator_mass": self.outside_indicator_mass,
            "inside_mass_fraction": self.inside_mass_fraction,
            "inside_mean": self.inside_mean,
            "outside_mean": self.outside_mean,
            "inside_outside_mean_ratio": self.inside_outside_mean_ratio,
        }


def compute_cell_defect_localization_metrics(
    *,
    cell_centers: np.ndarray,
    cell_values: np.ndarray,
    defect_center_x: float,
    defect_center_y: float,
    defect_radius: float,
) -> CellDefectLocalizationMetrics:
    centers = np.asarray(cell_centers, dtype=float)
    values = np.asarray(cell_values, dtype=float)

    defect_center_x = float(defect_center_x)
    defect_center_y = float(defect_center_y)
    defect_radius = float(defect_radius)

    if centers.ndim != 2:
        raise ValueError("cell_centers must be two-dimensional.")

    if centers.shape[1] != 2:
        raise ValueError("cell_centers must have shape (n_cells, 2).")

    if values.ndim != 1:
        raise ValueError("cell_values must be one-dimensional.")

    if centers.shape[0] != values.size:
        raise ValueError("cell_centers and cell_values size mismatch.")

    if not np.all(np.isfinite(centers)):
        raise ValueError("cell_centers contains non-finite entries.")

    if not np.all(np.isfinite(values)):
        raise ValueError("cell_values contains non-finite entries.")

    if not np.isfinite(defect_center_x) or not np.isfinite(defect_center_y):
        raise ValueError("defect center must be finite.")

    if not np.isfinite(defect_radius) or defect_radius <= 0.0:
        raise ValueError("defect_radius must be strictly positive and finite.")

    if values.size == 0:
        return CellDefectLocalizationMetrics(
            defect_center=(defect_center_x, defect_center_y),
            defect_radius=defect_radius,
            num_cells=0,
            num_inside_cells=0,
            num_outside_cells=0,
            max_value=0.0,
            max_center=(float("nan"), float("nan")),
            max_distance_to_defect_center=float("nan"),
            max_inside_defect=False,
            total_indicator_mass=0.0,
            inside_indicator_mass=0.0,
            outside_indicator_mass=0.0,
            inside_mass_fraction=0.0,
            inside_mean=0.0,
            outside_mean=0.0,
            inside_outside_mean_ratio=float("inf"),
            status="empty",
        )

    distances = np.sqrt(
        (centers[:, 0] - defect_center_x) ** 2
        + (centers[:, 1] - defect_center_y) ** 2
    )

    inside_mask = distances <= defect_radius
    outside_mask = ~inside_mask

    max_index = int(np.argmax(values))
    max_value = float(values[max_index])
    max_center = (
        float(centers[max_index, 0]),
        float(centers[max_index, 1]),
    )
    max_distance = float(distances[max_index])
    max_inside_defect = bool(inside_mask[max_index])

    total_mass = float(np.sum(values))
    inside_mass = float(np.sum(values[inside_mask])) if np.any(inside_mask) else 0.0
    outside_mass = float(np.sum(values[outside_mask])) if np.any(outside_mask) else 0.0

    inside_fraction = inside_mass / total_mass if total_mass > 0.0 else 0.0

    inside_mean = float(np.mean(values[inside_mask])) if np.any(inside_mask) else 0.0
    outside_mean = float(np.mean(values[outside_mask])) if np.any(outside_mask) else 0.0

    if outside_mean > 0.0:
        ratio = inside_mean / outside_mean
    elif inside_mean > 0.0:
        ratio = float("inf")
    else:
        ratio = 0.0

    return CellDefectLocalizationMetrics(
        defect_center=(defect_center_x, defect_center_y),
        defect_radius=defect_radius,
        num_cells=int(values.size),
        num_inside_cells=int(np.count_nonzero(inside_mask)),
        num_outside_cells=int(np.count_nonzero(outside_mask)),
        max_value=max_value,
        max_center=max_center,
        max_distance_to_defect_center=max_distance,
        max_inside_defect=max_inside_defect,
        total_indicator_mass=total_mass,
        inside_indicator_mass=inside_mass,
        outside_indicator_mass=outside_mass,
        inside_mass_fraction=float(inside_fraction),
        inside_mean=inside_mean,
        outside_mean=outside_mean,
        inside_outside_mean_ratio=float(ratio),
        status="ok",
    )
