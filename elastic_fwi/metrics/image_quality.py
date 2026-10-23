from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ImageQualityMetrics:
    method: str
    field_name: str
    status: str
    max_x: float
    max_y: float
    localization_error: float
    max_value: float
    max_value_in_defect: float
    max_value_outside_defect: float
    peak_to_artifact_ratio: float
    energy_in_defect: float
    energy_outside_defect: float
    false_maxima_count: int
    half_max_area: float
    half_max_fraction: float
    num_samples: int
    num_inside_defect: int
    num_outside_defect: int
    defect_center_x: float
    defect_center_y: float
    defect_radius: float
    threshold_false_max: float
    false_max_min_distance: float
    value_mode: str
    extra: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        data = {
            "method": self.method,
            "field_name": self.field_name,
            "status": self.status,
            "max_x": self.max_x,
            "max_y": self.max_y,
            "localization_error": self.localization_error,
            "max_value": self.max_value,
            "max_value_in_defect": self.max_value_in_defect,
            "max_value_outside_defect": self.max_value_outside_defect,
            "peak_to_artifact_ratio": self.peak_to_artifact_ratio,
            "energy_in_defect": self.energy_in_defect,
            "energy_outside_defect": self.energy_outside_defect,
            "false_maxima_count": self.false_maxima_count,
            "half_max_area": self.half_max_area,
            "half_max_fraction": self.half_max_fraction,
            "num_samples": self.num_samples,
            "num_inside_defect": self.num_inside_defect,
            "num_outside_defect": self.num_outside_defect,
            "defect_center_x": self.defect_center_x,
            "defect_center_y": self.defect_center_y,
            "defect_radius": self.defect_radius,
            "threshold_false_max": self.threshold_false_max,
            "false_max_min_distance": self.false_max_min_distance,
            "value_mode": self.value_mode,
        }
        data.update(self.extra)
        return data


def _as_2d_centers(cell_centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(cell_centers, dtype=float)
    if centers.ndim != 2:
        raise ValueError("cell_centers must be a two-dimensional array.")
    if centers.shape[1] < 2:
        raise ValueError("cell_centers must contain at least x and y coordinates.")
    centers = centers[:, :2]
    if not np.all(np.isfinite(centers)):
        raise ValueError("cell_centers contains non-finite entries.")
    return centers


def _as_scalar_values(values: np.ndarray, *, value_mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim == 2 and arr.shape[1] > 1:
        if value_mode == "norm":
            arr = np.linalg.norm(arr, axis=1)
        else:
            raise ValueError(
                "image_values is vector-valued. Use value_mode='norm' or provide scalar values."
            )
    elif arr.ndim != 1:
        raise ValueError("image_values must be one-dimensional or shape (n, 1).")
    if not np.all(np.isfinite(arr)):
        raise ValueError("image_values contains non-finite entries.")
    return np.asarray(arr, dtype=float)


def _as_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size != n:
        raise ValueError("cell_weights must be one-dimensional with the same length as values.")
    if not np.all(np.isfinite(w)):
        raise ValueError("cell_weights contains non-finite entries.")
    if np.any(w < 0.0):
        raise ValueError("cell_weights must be non-negative.")
    return w


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator > 0.0:
        return float(numerator / denominator)
    if numerator > 0.0:
        return float("inf")
    return 0.0


def _count_separated_false_maxima(
    *,
    centers: np.ndarray,
    values: np.ndarray,
    outside_mask: np.ndarray,
    threshold_value: float,
    min_distance: float,
) -> int:
    """Count separated false maxima outside the defect region.

    This bounded implementation avoids quadratic blow-up when a dense image has
    many above-threshold candidates.
    """
    candidate_indices = np.flatnonzero(outside_mask & (values >= threshold_value))
    if candidate_indices.size == 0:
        return 0

    max_candidates = 5000

    if candidate_indices.size > max_candidates:
        strongest_local = np.argsort(values[candidate_indices])[-max_candidates:]
        candidate_indices = candidate_indices[strongest_local]

    ordered = candidate_indices[np.argsort(values[candidate_indices])[::-1]]

    selected_points: list[np.ndarray] = []
    min_distance = max(float(min_distance), 0.0)

    for idx in ordered:
        point = centers[int(idx)]

        if min_distance == 0.0:
            selected_points.append(point)
            continue

        if not selected_points:
            selected_points.append(point)
            continue

        previous = np.vstack(selected_points)
        distances = np.linalg.norm(previous - point[None, :], axis=1)

        if bool(np.all(distances >= min_distance)):
            selected_points.append(point)

    return int(len(selected_points))


def compute_image_quality_metrics(
    *,
    cell_centers: np.ndarray,
    image_values: np.ndarray,
    defect_center_x: float,
    defect_center_y: float,
    defect_radius: float,
    method: str,
    field_name: str,
    cell_weights: np.ndarray | None = None,
    threshold_false_max: float = 0.5,
    half_max_fraction: float = 0.5,
    false_max_min_distance: float | None = None,
    value_mode: str = "abs",
    extra: Mapping[str, Any] | None = None,
) -> ImageQualityMetrics:
    centers = _as_2d_centers(cell_centers)
    raw_values = _as_scalar_values(
        image_values,
        value_mode="norm" if value_mode == "norm" else "abs",
    )

    if centers.shape[0] != raw_values.size:
        raise ValueError("cell_centers and image_values size mismatch.")

    defect_center_x = float(defect_center_x)
    defect_center_y = float(defect_center_y)
    defect_radius = float(defect_radius)
    threshold_false_max = float(threshold_false_max)
    half_max_fraction = float(half_max_fraction)

    if not isfinite(defect_center_x) or not isfinite(defect_center_y):
        raise ValueError("defect center must be finite.")
    if not isfinite(defect_radius) or defect_radius <= 0.0:
        raise ValueError("defect_radius must be strictly positive and finite.")
    if not isfinite(threshold_false_max) or threshold_false_max < 0.0:
        raise ValueError("threshold_false_max must be finite and non-negative.")
    if not isfinite(half_max_fraction) or half_max_fraction < 0.0:
        raise ValueError("half_max_fraction must be finite and non-negative.")

    n = int(raw_values.size)
    weights = _as_weights(cell_weights, n)
    extra_dict = dict(extra or {})

    if n == 0:
        return ImageQualityMetrics(
            method=str(method),
            field_name=str(field_name),
            status="empty",
            max_x=float("nan"),
            max_y=float("nan"),
            localization_error=float("nan"),
            max_value=0.0,
            max_value_in_defect=0.0,
            max_value_outside_defect=0.0,
            peak_to_artifact_ratio=0.0,
            energy_in_defect=0.0,
            energy_outside_defect=0.0,
            false_maxima_count=0,
            half_max_area=0.0,
            half_max_fraction=half_max_fraction,
            num_samples=0,
            num_inside_defect=0,
            num_outside_defect=0,
            defect_center_x=defect_center_x,
            defect_center_y=defect_center_y,
            defect_radius=defect_radius,
            threshold_false_max=threshold_false_max,
            false_max_min_distance=0.0 if false_max_min_distance is None else float(false_max_min_distance),
            value_mode=str(value_mode),
            extra=extra_dict,
        )

    if value_mode == "signed":
        values = raw_values.copy()
        ranking_values = np.abs(raw_values)
    elif value_mode == "norm":
        values = raw_values.copy()
        ranking_values = raw_values.copy()
    elif value_mode == "abs":
        values = np.abs(raw_values)
        ranking_values = values.copy()
    else:
        raise ValueError("value_mode must be one of 'abs', 'signed', or 'norm'.")

    distances = np.sqrt(
        (centers[:, 0] - defect_center_x) ** 2
        + (centers[:, 1] - defect_center_y) ** 2
    )
    inside_mask = distances <= defect_radius
    outside_mask = ~inside_mask

    max_index = int(np.argmax(ranking_values))
    max_value = float(values[max_index])
    max_x = float(centers[max_index, 0])
    max_y = float(centers[max_index, 1])
    localization_error = float(distances[max_index])

    max_inside = float(np.max(ranking_values[inside_mask])) if np.any(inside_mask) else 0.0
    max_outside = float(np.max(ranking_values[outside_mask])) if np.any(outside_mask) else 0.0
    peak_to_artifact = _safe_ratio(max_inside, max_outside)

    energy_values = ranking_values ** 2
    energy_inside = float(np.sum(weights[inside_mask] * energy_values[inside_mask])) if np.any(inside_mask) else 0.0
    energy_outside = float(np.sum(weights[outside_mask] * energy_values[outside_mask])) if np.any(outside_mask) else 0.0

    global_peak = float(np.max(ranking_values)) if ranking_values.size else 0.0
    threshold_value = threshold_false_max * global_peak
    if false_max_min_distance is None:
        false_max_min_distance_value = 0.5 * defect_radius
    else:
        false_max_min_distance_value = float(false_max_min_distance)

    if global_peak > 0.0:
        false_count = _count_separated_false_maxima(
            centers=centers,
            values=ranking_values,
            outside_mask=outside_mask,
            threshold_value=threshold_value,
            min_distance=false_max_min_distance_value,
        )
    else:
        false_count = 0

    half_threshold = half_max_fraction * global_peak
    half_max_area = float(np.sum(weights[ranking_values >= half_threshold])) if global_peak > 0.0 else 0.0

    if not np.any(inside_mask):
        status = "no_inside_samples"
    elif not np.any(outside_mask):
        status = "no_outside_samples"
    elif global_peak <= 0.0:
        status = "zero_image"
    else:
        status = "ok"

    return ImageQualityMetrics(
        method=str(method),
        field_name=str(field_name),
        status=status,
        max_x=max_x,
        max_y=max_y,
        localization_error=localization_error,
        max_value=max_value,
        max_value_in_defect=max_inside,
        max_value_outside_defect=max_outside,
        peak_to_artifact_ratio=peak_to_artifact,
        energy_in_defect=energy_inside,
        energy_outside_defect=energy_outside,
        false_maxima_count=false_count,
        half_max_area=half_max_area,
        half_max_fraction=half_max_fraction,
        num_samples=n,
        num_inside_defect=int(np.count_nonzero(inside_mask)),
        num_outside_defect=int(np.count_nonzero(outside_mask)),
        defect_center_x=defect_center_x,
        defect_center_y=defect_center_y,
        defect_radius=defect_radius,
        threshold_false_max=threshold_false_max,
        false_max_min_distance=false_max_min_distance_value,
        value_mode=str(value_mode),
        extra=extra_dict,
    )
