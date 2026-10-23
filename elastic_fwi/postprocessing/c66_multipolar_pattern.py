from __future__ import annotations

"""Quantitative multipolar-pattern analysis for signed normalized C66 fields.

The actual article-ready outputs are DG0 cell fields stored as PVD/PVTU/VTU.
All spatial statistics therefore use cell areas as quadrature weights.  NPZ and
CSV inputs are also supported for portable synthetic or converted datasets,
provided coherent cell measures are available or can be inferred from a
regular Cartesian grid.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from elastic_fwi.postprocessing.signed_sensitivity_fields import (
    AnalysisConfig,
    DiscoveredSignedField,
    discover_signed_fields,
    load_defect_circles,
    load_vtk_cell_field,
)


FORMAT_VERSION = "c66_multipolar_metrics_v2"
SUPPORTED_VTK_SUFFIXES = {".pvd", ".pvtu", ".vtu"}
DEFAULT_OBSERVATION_MODES = ("classical_full", "residual_interaction")


class C66MultipolarError(RuntimeError):
    """Raised when a C66 multipolar analysis cannot be performed safely."""


@dataclass(frozen=True)
class DefectGeometry:
    center_x: float
    center_y: float
    radius: float

    def validate(self) -> None:
        values = (self.center_x, self.center_y, self.radius)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Defect center and radius must be finite.")
        if self.radius <= 0.0:
            raise ValueError("Defect radius must be strictly positive.")


@dataclass(frozen=True)
class C66Field:
    centers: np.ndarray
    values: np.ndarray
    cell_measures: np.ndarray
    source_path: Path
    source_format: str
    field_name: str
    normalization_factor: float
    values_were_normalized: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def num_values(self) -> int:
        return int(np.asarray(self.values).size)


@dataclass(frozen=True)
class CaseMetadata:
    case_name: str
    observation_mode: str
    modality: str
    aperture: str
    campaign_id: str = ""
    campaign_dir: str = ""
    manifest_path: str = ""
    final_time: float | None = None


@dataclass(frozen=True)
class MultipolarConfig:
    num_radial_bins: int = 80
    num_angular_bins: int = 180
    radial_bin_values: tuple[int, ...] = (60, 80, 100)
    annulus_width_factor: float = 0.15
    annulus_width_values: tuple[float, ...] = (0.10, 0.15, 0.20)
    annulus_radius_factors: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
    max_radius_factor: float = 8.0
    primary_radius_min_factor: float = 0.75
    primary_radius_max_factor: float = 3.0
    near_defect_radius_factor: float = 2.0
    max_fourier_mode: int = 8
    minimum_mode_power_fraction: float = 0.05
    minimum_angular_coverage: float = 0.75
    center_search_extent_factor: float = 0.5
    center_search_steps: int = 7
    orientation_stability_degrees: float = 10.0
    symmetry_stability_tolerance: float = 0.15
    peak_radius_stability_factor: float = 0.25
    dominant_mode_agreement_threshold: float = 0.75
    minimum_valid_robustness_configurations: int = 2
    minimum_valid_robustness_fraction: float = 0.5
    center_proximity_factor: float = 0.5
    max_single_cell_energy_fraction: float = 0.05
    compare_discretizations: bool = True

    def validate(self) -> None:
        if self.num_radial_bins < 8:
            raise ValueError("num_radial_bins must be at least 8.")
        if self.num_angular_bins < 16 or self.num_angular_bins % 2 != 0:
            raise ValueError("num_angular_bins must be even and at least 16.")
        if self.max_fourier_mode < 1:
            raise ValueError("max_fourier_mode must be positive.")
        if self.max_fourier_mode >= self.num_angular_bins // 2:
            raise ValueError("max_fourier_mode violates the angular Nyquist limit.")
        if self.max_radius_factor <= 0.0:
            raise ValueError("max_radius_factor must be positive.")
        if not 0.0 < self.primary_radius_min_factor < self.primary_radius_max_factor:
            raise ValueError("Invalid primary-radius search interval.")
        if self.primary_radius_max_factor > self.max_radius_factor:
            raise ValueError("Primary-radius interval must lie inside the analysis ROI.")
        if self.annulus_width_factor <= 0.0:
            raise ValueError("annulus_width_factor must be positive.")
        if self.center_search_steps < 1 or self.center_search_steps % 2 == 0:
            raise ValueError("center_search_steps must be a positive odd integer.")
        if self.center_search_extent_factor < 0.0:
            raise ValueError("center_search_extent_factor must be non-negative.")
        for value in self.radial_bin_values:
            if int(value) < 8:
                raise ValueError("All radial-bin robustness values must be at least 8.")
        for value in self.annulus_width_values:
            if float(value) <= 0.0:
                raise ValueError("All annulus-width robustness values must be positive.")
        for value in self.annulus_radius_factors:
            if float(value) <= 0.0:
                raise ValueError("All annulus radii must be positive.")
        if self.minimum_valid_robustness_configurations < 1:
            raise ValueError("minimum_valid_robustness_configurations must be positive.")
        if not 0.0 < self.minimum_valid_robustness_fraction <= 1.0:
            raise ValueError("minimum_valid_robustness_fraction must lie in (0, 1].")


@dataclass(frozen=True)
class CaseInput:
    field: C66Field
    defect: DefectGeometry
    metadata: CaseMetadata


@dataclass(frozen=True)
class CaseAnalysis:
    summary: dict[str, Any]
    radial_profile: tuple[dict[str, Any], ...]
    annuli: tuple[dict[str, Any], ...]
    fourier_modes: tuple[dict[str, Any], ...]
    robustness: tuple[dict[str, Any], ...]
    center_search: tuple[dict[str, Any], ...]
    figure_png: Path | None = None
    figure_pdf: Path | None = None


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_identifier(value: str) -> str:
    chars = [character if character.isalnum() or character in "._-" else "_" for character in str(value)]
    text = "".join(chars).strip("_")
    return text or "case"


def _as_float_array(values: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return array


def _validate_field_arrays(
    centers: np.ndarray,
    values: np.ndarray,
    cell_measures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = _as_float_array(centers, name="centers", ndim=2)
    field_values = _as_float_array(values, name="values").reshape(-1)
    measures = _as_float_array(cell_measures, name="cell_measures").reshape(-1)
    if xy.shape[1] < 2:
        raise ValueError("centers must have at least two columns.")
    xy = np.ascontiguousarray(xy[:, :2])
    if xy.shape[0] != field_values.size or field_values.size != measures.size:
        raise ValueError(
            "centers, values, and cell_measures have incompatible lengths: "
            f"{xy.shape[0]}, {field_values.size}, {measures.size}."
        )
    if field_values.size == 0:
        raise ValueError("The field is empty.")
    if np.any(measures <= 0.0):
        raise ValueError("cell_measures must be strictly positive.")
    return xy, np.ascontiguousarray(field_values), np.ascontiguousarray(measures)


def _normalize_signed(values: np.ndarray) -> tuple[np.ndarray, float, bool]:
    array = _as_float_array(values, name="values").reshape(-1)
    maximum = float(np.max(np.abs(array))) if array.size else 0.0
    if maximum <= 0.0:
        raise C66MultipolarError("The C66 field is identically zero; no multipolar pattern is defined.")
    normalized = np.ascontiguousarray(array / maximum)
    already_normalized = bool(math.isclose(maximum, 1.0, rel_tol=1.0e-10, abs_tol=1.0e-12))
    return normalized, maximum, already_normalized


def _infer_regular_cell_measure(x: np.ndarray, y: np.ndarray) -> float:
    unique_x = np.unique(np.asarray(x, dtype=np.float64))
    unique_y = np.unique(np.asarray(y, dtype=np.float64))
    if unique_x.size < 2 or unique_y.size < 2:
        raise C66MultipolarError(
            "Cannot infer cell measures from fewer than two distinct x and y coordinates."
        )
    if np.asarray(x).size != unique_x.size * unique_y.size:
        raise C66MultipolarError(
            "Cell measures can only be inferred when the CSV/coordinate arrays form "
            "a complete regular Cartesian grid."
        )
    dx = np.diff(unique_x)
    dy = np.diff(unique_y)
    if not np.allclose(dx, dx[0], rtol=1.0e-8, atol=1.0e-12):
        raise C66MultipolarError("x coordinates are not a regular Cartesian grid.")
    if not np.allclose(dy, dy[0], rtol=1.0e-8, atol=1.0e-12):
        raise C66MultipolarError("y coordinates are not a regular Cartesian grid.")
    return float(abs(dx[0] * dy[0]))


def _first_existing_key(payload: Mapping[str, Any], names: Sequence[str]) -> str | None:
    for name in names:
        if name in payload:
            return name
    return None


def _load_npz(path: Path, field_name: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as payload:
        keys = set(payload.files)
        selected = field_name if field_name in keys else None
        if selected is None:
            selected = _first_existing_key(
                {key: None for key in keys},
                ("values", "C66", "c66", "gradient_C66", "residual_interaction_C66"),
            )
        if selected is None:
            raise C66MultipolarError(
                f"No C66 value array found in {path}; available keys are {sorted(keys)}."
            )
        values = np.asarray(payload[selected], dtype=np.float64)

        if "centers" in keys:
            centers = np.asarray(payload["centers"], dtype=np.float64)
            measures_key = _first_existing_key(
                {key: None for key in keys},
                ("cell_measures", "cell_areas", "weights", "areas"),
            )
            if measures_key is None:
                raise C66MultipolarError(
                    "NPZ inputs with an unstructured 'centers' array must also provide "
                    "cell_measures/cell_areas/weights."
                )
            measures = np.asarray(payload[measures_key], dtype=np.float64)
            return centers, values.reshape(-1), measures.reshape(-1), str(selected)

        if "x" not in keys or "y" not in keys:
            raise C66MultipolarError("NPZ input must provide either centers or x/y coordinates.")
        x = np.asarray(payload["x"], dtype=np.float64)
        y = np.asarray(payload["y"], dtype=np.float64)
        if x.ndim == 1 and y.ndim == 1 and values.ndim == 2:
            xx, yy = np.meshgrid(x, y, indexing="xy")
        elif x.shape == y.shape == values.shape:
            xx, yy = x, y
        else:
            raise C66MultipolarError(
                "Regular NPZ data require 1-D x/y axes with a (ny,nx) field, or "
                "x, y, and values arrays of identical shape."
            )
        measure = _infer_regular_cell_measure(xx.reshape(-1), yy.reshape(-1))
        centers = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
        measures = np.full(values.size, measure, dtype=np.float64)
        return centers, values.reshape(-1), measures, str(selected)


def _load_csv(path: Path, field_name: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames or []
    if not rows:
        raise C66MultipolarError(f"CSV input is empty: {path}")
    x_name = _first_existing_key({name: None for name in columns}, ("x", "center_x", "cell_center_x"))
    y_name = _first_existing_key({name: None for name in columns}, ("y", "center_y", "cell_center_y"))
    value_name = field_name if field_name in columns else _first_existing_key(
        {name: None for name in columns},
        ("value", "values", "C66", "c66", "gradient_C66", "residual_interaction_C66"),
    )
    if x_name is None or y_name is None or value_name is None:
        raise C66MultipolarError(
            f"CSV requires x, y, and C66 value columns; available columns are {columns}."
        )
    x = np.asarray([float(row[x_name]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[y_name]) for row in rows], dtype=np.float64)
    values = np.asarray([float(row[value_name]) for row in rows], dtype=np.float64)
    measure_name = _first_existing_key(
        {name: None for name in columns},
        ("cell_measure", "cell_area", "weight", "area"),
    )
    if measure_name is None:
        measure = _infer_regular_cell_measure(x, y)
        measures = np.full(values.size, measure, dtype=np.float64)
    else:
        measures = np.asarray([float(row[measure_name]) for row in rows], dtype=np.float64)
    return np.column_stack((x, y)), values, measures, str(value_name)


def load_c66_field(
    path: str | Path,
    *,
    field_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> C66Field:
    """Load and sign-normalize a C66 field without modifying the source file."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(str(source))
    suffix = source.suffix.lower()
    selected = field_name
    if suffix in SUPPORTED_VTK_SUFFIXES:
        candidates = tuple(
            dict.fromkeys(
                name
                for name in (
                    field_name,
                    "gradient_C66",
                    "residual_interaction_C66",
                    "C66",
                )
                if name
            )
        )
        errors: list[str] = []
        loaded = None
        for candidate in candidates:
            try:
                loaded = load_vtk_cell_field(source, candidate)
                selected = loaded.selected_field_name
                break
            except Exception as exc:  # candidate probing is intentional
                errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
        if loaded is None:
            raise C66MultipolarError(
                f"Could not load a C66 cell field from {source}. " + " | ".join(errors)
            )
        centers, raw_values, measures = _validate_field_arrays(
            loaded.centers,
            loaded.values,
            loaded.cell_measures,
        )
        source_format = "pvd_pvtu_vtu_dg0"
        extra = {
            "piece_count": len(loaded.piece_paths),
            "cell_types": list(loaded.cell_types),
            "num_ghost_cells_removed": loaded.num_ghost_cells_removed,
        }
    elif suffix == ".npz":
        centers0, raw_values0, measures0, selected = _load_npz(source, field_name)
        centers, raw_values, measures = _validate_field_arrays(centers0, raw_values0, measures0)
        source_format = "npz"
        extra = {}
    elif suffix == ".csv":
        centers0, raw_values0, measures0, selected = _load_csv(source, field_name)
        centers, raw_values, measures = _validate_field_arrays(centers0, raw_values0, measures0)
        source_format = "csv"
        extra = {}
    elif suffix == ".xdmf":
        raise C66MultipolarError(
            "XDMF is detected but not accepted here because this analysis requires "
            "cell-area weights. Convert the field to PVD/PVTU/VTU or NPZ with "
            "centers, values, and cell_measures."
        )
    elif suffix == ".npy":
        raise C66MultipolarError(
            "A standalone NPY array does not contain coordinates and cell measures. "
            "Use NPZ with centers/values/cell_measures or a regular x/y grid."
        )
    else:
        raise C66MultipolarError(f"Unsupported C66 input format: {source.suffix}")

    normalized, factor, already = _normalize_signed(raw_values)
    merged_metadata = dict(metadata or {})
    merged_metadata.update(extra)
    return C66Field(
        centers=centers,
        values=normalized,
        cell_measures=measures,
        source_path=source,
        source_format=source_format,
        field_name=str(selected or "C66"),
        normalization_factor=factor,
        values_were_normalized=already,
        metadata=merged_metadata,
    )


def compute_polar_coordinates(
    centers: np.ndarray,
    center_x: float,
    center_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return radii and angles relative to a prescribed center."""

    xy = _as_float_array(centers, name="centers", ndim=2)
    dx = xy[:, 0] - float(center_x)
    dy = xy[:, 1] - float(center_y)
    return np.hypot(dx, dy), np.arctan2(dy, dx)


def _binned_sums(indices: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    valid = (indices >= 0) & (indices < size)
    return np.bincount(indices[valid], weights=values[valid], minlength=size).astype(np.float64)


def compute_radial_profiles(
    radii: np.ndarray,
    values: np.ndarray,
    cell_measures: np.ndarray,
    *,
    defect_radius: float,
    num_bins: int,
    max_radius_factor: float,
) -> tuple[dict[str, np.ndarray], ...]:
    """Compute area-weighted signed, absolute, maximum, and energy profiles."""

    r = _as_float_array(radii, name="radii").reshape(-1)
    f = _as_float_array(values, name="values").reshape(-1)
    w = _as_float_array(cell_measures, name="cell_measures").reshape(-1)
    if not (r.size == f.size == w.size):
        raise ValueError("radii, values, and cell_measures must have equal lengths.")
    edges = np.linspace(0.0, float(max_radius_factor), int(num_bins) + 1)
    scaled = r / float(defect_radius)
    indices = np.searchsorted(edges, scaled, side="right") - 1
    indices[np.isclose(scaled, edges[-1])] = int(num_bins) - 1
    area = _binned_sums(indices, w, int(num_bins))
    signed = _binned_sums(indices, w * f, int(num_bins))
    absolute = _binned_sums(indices, w * np.abs(f), int(num_bins))
    energy = _binned_sums(indices, w * f * f, int(num_bins))
    counts = _binned_sums(indices, np.ones_like(f), int(num_bins)).astype(np.int64)
    maximum = np.full(int(num_bins), np.nan, dtype=np.float64)
    maximum_signed = np.full(int(num_bins), np.nan, dtype=np.float64)
    minimum_signed = np.full(int(num_bins), np.nan, dtype=np.float64)
    valid = (indices >= 0) & (indices < int(num_bins))
    if np.any(valid):
        order = np.argsort(indices[valid], kind="stable")
        sorted_indices = indices[valid][order]
        sorted_signed_values = f[valid][order]
        sorted_absolute_values = np.abs(sorted_signed_values)
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_indices)) + 1]
        populated_bins = sorted_indices[starts]
        maximum[populated_bins] = np.maximum.reduceat(sorted_absolute_values, starts)
        maximum_signed[populated_bins] = np.maximum.reduceat(sorted_signed_values, starts)
        minimum_signed[populated_bins] = np.minimum.reduceat(sorted_signed_values, starts)
    with np.errstate(divide="ignore", invalid="ignore"):
        signed_mean = np.divide(signed, area, out=np.full_like(signed, np.nan), where=area > 0.0)
        absolute_mean = np.divide(absolute, area, out=np.full_like(absolute, np.nan), where=area > 0.0)
        energy_density = np.divide(energy, area, out=np.full_like(energy, np.nan), where=area > 0.0)
    total_energy = float(np.sum(energy))
    energy_fraction = energy / total_energy if total_energy > 0.0 else np.zeros_like(energy)
    centers = 0.5 * (edges[:-1] + edges[1:])
    result: list[dict[str, np.ndarray]] = []
    for index in range(int(num_bins)):
        result.append(
            {
                "bin_index": int(index),
                "radius_min_over_R": float(edges[index]),
                "radius_max_over_R": float(edges[index + 1]),
                "radius_center_over_R": float(centers[index]),
                "cell_count": int(counts[index]),
                "area": float(area[index]),
                "signed_mean": float(signed_mean[index]),
                "absolute_mean": float(absolute_mean[index]),
                "maximum_absolute": float(maximum[index]),
                "maximum_signed": float(maximum_signed[index]),
                "minimum_signed": float(minimum_signed[index]),
                "energy": float(energy[index]),
                "energy_density": float(energy_density[index]),
                "energy_fraction": float(energy_fraction[index]),
            }
        )
    return tuple(result)


def _primary_radius_from_profile(
    profile: Sequence[Mapping[str, Any]],
    *,
    minimum: float,
    maximum: float,
) -> float:
    candidates = [
        row
        for row in profile
        if minimum <= float(row["radius_center_over_R"]) <= maximum
        and math.isfinite(float(row["energy_density"]))
        and int(row["cell_count"]) > 0
    ]
    if not candidates:
        raise C66MultipolarError("No populated radial bin lies in the primary annulus search interval.")
    selected = max(candidates, key=lambda row: float(row["energy_density"]))
    return float(selected["radius_center_over_R"])


def _angular_profile(
    theta: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    num_bins: int,
) -> dict[str, np.ndarray | float]:
    edges = np.linspace(-math.pi, math.pi, int(num_bins) + 1)
    wrapped = (np.asarray(theta, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi
    indices = np.searchsorted(edges, wrapped, side="right") - 1
    indices[indices == int(num_bins)] = int(num_bins) - 1
    area = _binned_sums(indices, np.asarray(weights, dtype=np.float64), int(num_bins))
    weighted = _binned_sums(indices, np.asarray(weights) * np.asarray(values), int(num_bins))
    means = np.divide(
        weighted,
        area,
        out=np.full(int(num_bins), np.nan, dtype=np.float64),
        where=area > 0.0,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "angles": centers,
        "values": means,
        "areas": area,
        "coverage": float(np.count_nonzero(area > 0.0) / int(num_bins)),
    }


def compute_180_symmetry(
    angular_values: np.ndarray,
    angular_weights: np.ndarray | None = None,
) -> float:
    """Return the signed correlation between an angular profile and its pi rotation."""

    values = np.asarray(angular_values, dtype=np.float64).reshape(-1)
    if values.size < 2 or values.size % 2 != 0:
        raise ValueError("The angular profile must contain an even number of bins.")
    weights = (
        np.ones_like(values)
        if angular_weights is None
        else np.asarray(angular_weights, dtype=np.float64).reshape(-1)
    )
    if weights.shape != values.shape:
        raise ValueError("angular_weights has an incompatible shape.")
    shifted = np.roll(values, values.size // 2)
    shifted_weights = np.roll(weights, values.size // 2)
    valid = np.isfinite(values) & np.isfinite(shifted) & (weights > 0.0) & (shifted_weights > 0.0)
    if np.count_nonzero(valid) < 4:
        return math.nan
    pair_weights = np.sqrt(weights[valid] * shifted_weights[valid])
    left = values[valid]
    right = shifted[valid]
    numerator = float(np.sum(pair_weights * left * right, dtype=np.float64))
    denominator = math.sqrt(
        float(np.sum(pair_weights * left * left, dtype=np.float64))
        * float(np.sum(pair_weights * right * right, dtype=np.float64))
    )
    return numerator / denominator if denominator > 0.0 else math.nan


def compute_global_180_symmetry(
    radii: np.ndarray,
    theta: np.ndarray,
    values: np.ndarray,
    cell_measures: np.ndarray,
    *,
    defect_radius: float,
    max_radius_factor: float,
    num_radial_bins: int,
    num_angular_bins: int,
) -> dict[str, float]:
    """Compare the field with its pi rotation at matched polar bins.

    Radial and angular bins are area averaged first.  Each populated
    ``(r, theta)`` bin is then paired with ``(r, theta + pi)``, so the global
    score does not mix distinct radii.
    """

    if int(num_angular_bins) % 2 != 0:
        raise ValueError("num_angular_bins must be even for pi-rotation pairing.")
    r = _as_float_array(radii, name="radii").reshape(-1) / float(defect_radius)
    angles = _as_float_array(theta, name="theta").reshape(-1)
    field_values = _as_float_array(values, name="values").reshape(-1)
    weights = _as_float_array(cell_measures, name="cell_measures").reshape(-1)
    if not (r.size == angles.size == field_values.size == weights.size):
        raise ValueError("Global symmetry arrays must have equal lengths.")

    radial_edges = np.linspace(0.0, float(max_radius_factor), int(num_radial_bins) + 1)
    angular_edges = np.linspace(-math.pi, math.pi, int(num_angular_bins) + 1)
    wrapped = (angles + math.pi) % (2.0 * math.pi) - math.pi
    radial_index = np.searchsorted(radial_edges, r, side="right") - 1
    angular_index = np.searchsorted(angular_edges, wrapped, side="right") - 1
    radial_index[np.isclose(r, radial_edges[-1])] = int(num_radial_bins) - 1
    angular_index[angular_index == int(num_angular_bins)] = int(num_angular_bins) - 1
    valid = (
        (radial_index >= 0)
        & (radial_index < int(num_radial_bins))
        & (angular_index >= 0)
        & (angular_index < int(num_angular_bins))
    )
    flat_index = radial_index[valid] * int(num_angular_bins) + angular_index[valid]
    size = int(num_radial_bins) * int(num_angular_bins)
    area = np.bincount(flat_index, weights=weights[valid], minlength=size).astype(np.float64)
    weighted = np.bincount(
        flat_index,
        weights=weights[valid] * field_values[valid],
        minlength=size,
    ).astype(np.float64)
    means = np.divide(
        weighted,
        area,
        out=np.full(size, np.nan, dtype=np.float64),
        where=area > 0.0,
    ).reshape(int(num_radial_bins), int(num_angular_bins))
    areas = area.reshape(int(num_radial_bins), int(num_angular_bins))
    shifted_values = np.roll(means, int(num_angular_bins) // 2, axis=1)
    shifted_areas = np.roll(areas, int(num_angular_bins) // 2, axis=1)
    paired = (
        np.isfinite(means)
        & np.isfinite(shifted_values)
        & (areas > 0.0)
        & (shifted_areas > 0.0)
    )
    if np.count_nonzero(paired) < 8:
        return {
            "signed_symmetry_180": math.nan,
            "absolute_symmetry_180": math.nan,
            "paired_bin_fraction": 0.0,
            "num_paired_bins": int(np.count_nonzero(paired)),
        }
    pair_weights = np.sqrt(areas[paired] * shifted_areas[paired])

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        numerator = float(np.sum(pair_weights * left * right, dtype=np.float64))
        denominator = math.sqrt(
            float(np.sum(pair_weights * left * left, dtype=np.float64))
            * float(np.sum(pair_weights * right * right, dtype=np.float64))
        )
        return numerator / denominator if denominator > 0.0 else math.nan

    left = means[paired]
    right = shifted_values[paired]
    populated = int(np.count_nonzero(areas > 0.0))
    return {
        "signed_symmetry_180": correlation(left, right),
        "absolute_symmetry_180": correlation(np.abs(left), np.abs(right)),
        "paired_bin_fraction": (
            float(np.count_nonzero(paired) / populated) if populated > 0 else 0.0
        ),
        "num_paired_bins": int(np.count_nonzero(paired)),
    }


def estimate_lobe_orientation(coefficient: complex, mode: int) -> tuple[float, float]:
    """Return the positive-lobe orientation modulo 2*pi/mode."""

    if int(mode) <= 0:
        return math.nan, math.nan
    if not (math.isfinite(coefficient.real) and math.isfinite(coefficient.imag)):
        return math.nan, math.nan
    period = 2.0 * math.pi / int(mode)
    angle = (-math.atan2(coefficient.imag, coefficient.real) / int(mode)) % period
    return angle, math.degrees(angle)


def compute_angular_fourier(
    theta: np.ndarray,
    values: np.ndarray,
    cell_measures: np.ndarray,
    *,
    max_mode: int,
    num_angular_bins: int,
    minimum_mode_power_fraction: float = 0.05,
    minimum_angular_coverage: float = 0.75,
) -> dict[str, Any]:
    """Compute area-weighted angular Fourier modes and pi-rotation symmetry."""

    angles = _as_float_array(theta, name="theta").reshape(-1)
    field_values = _as_float_array(values, name="values").reshape(-1)
    weights = _as_float_array(cell_measures, name="cell_measures").reshape(-1)
    if not (angles.size == field_values.size == weights.size):
        raise ValueError("theta, values, and cell_measures must have equal lengths.")
    if angles.size == 0 or float(np.sum(weights)) <= 0.0:
        raise C66MultipolarError("The angular annulus is empty.")
    weight_sum = float(np.sum(weights, dtype=np.float64))
    mean_square = float(np.sum(weights * field_values * field_values, dtype=np.float64) / weight_sum)
    coefficients: list[complex] = []
    rows: list[dict[str, Any]] = []
    for mode in range(int(max_mode) + 1):
        coefficient = complex(
            np.sum(weights * field_values * np.exp(-1j * mode * angles), dtype=np.complex128)
            / weight_sum
        )
        coefficients.append(coefficient)
        amplitude = abs(coefficient) if mode == 0 else 2.0 * abs(coefficient)
        mode_power = abs(coefficient) ** 2 if mode == 0 else 2.0 * abs(coefficient) ** 2
        power_fraction = mode_power / mean_square if mean_square > 0.0 else math.nan
        orientation_rad, orientation_deg = estimate_lobe_orientation(coefficient, mode)
        rows.append(
            {
                "mode": mode,
                "coefficient_real": float(coefficient.real),
                "coefficient_imag": float(coefficient.imag),
                "coefficient_magnitude": float(abs(coefficient)),
                "reconstructed_amplitude": float(amplitude),
                "mode_power": float(mode_power),
                "mode_power_fraction_of_rms": float(power_fraction),
                "phase_radians": float(math.atan2(coefficient.imag, coefficient.real)),
                "phase_degrees": float(math.degrees(math.atan2(coefficient.imag, coefficient.real))),
                "orientation_radians": orientation_rad,
                "orientation_degrees": orientation_deg,
            }
        )
    nonzero = rows[1:]
    dominant = max(nonzero, key=lambda row: float(row["mode_power_fraction_of_rms"]))
    dominant_mode = int(dominant["mode"])
    dominant_fraction = float(dominant["mode_power_fraction_of_rms"])

    angular = _angular_profile(
        angles,
        field_values,
        weights,
        num_bins=int(num_angular_bins),
    )
    signed_symmetry = compute_180_symmetry(
        np.asarray(angular["values"]),
        np.asarray(angular["areas"]),
    )
    absolute_symmetry = compute_180_symmetry(
        np.abs(np.asarray(angular["values"])),
        np.asarray(angular["areas"]),
    )
    coverage = float(angular["coverage"])
    coverage_sufficient = bool(coverage >= float(minimum_angular_coverage))
    mode_power_sufficient = bool(
        math.isfinite(dominant_fraction)
        and dominant_fraction >= float(minimum_mode_power_fraction)
    )
    interpretable = bool(coverage_sufficient and mode_power_sufficient)
    if not coverage_sufficient:
        interpretation_reason = "INSUFFICIENT_ANGULAR_COVERAGE"
    elif not mode_power_sufficient:
        interpretation_reason = "WEAK_ANGULAR_MODE_POWER"
    else:
        interpretation_reason = "OK"
    return {
        "modes": tuple(rows),
        "coefficients": tuple(coefficients),
        "dominant_mode": dominant_mode if interpretable else 0,
        "dominant_mode_raw": dominant_mode,
        "dominant_mode_power_fraction": dominant_fraction,
        "dominant_orientation_radians": (
            float(dominant["orientation_radians"]) if interpretable else math.nan
        ),
        "dominant_orientation_degrees": (
            float(dominant["orientation_degrees"]) if interpretable else math.nan
        ),
        "orientation_period_degrees": (360.0 / dominant_mode if interpretable else math.nan),
        "signed_symmetry_180": float(signed_symmetry),
        "absolute_symmetry_180": float(absolute_symmetry),
        "angular_coverage": coverage,
        "angular_coverage_sufficient": coverage_sufficient,
        "mode_power_sufficient": mode_power_sufficient,
        "mean_square": mean_square,
        "interpretation_status": "OK" if interpretable else "WEAK_OR_INCOMPLETE_ANGULAR_CONTENT",
        "interpretation_reason": interpretation_reason,
        "angular_profile_angles": np.asarray(angular["angles"]),
        "angular_profile_values": np.asarray(angular["values"]),
        "angular_profile_areas": np.asarray(angular["areas"]),
    }


def _annulus_mask(
    radii_over_radius: np.ndarray,
    center_factor: float,
    width_factor: float,
) -> np.ndarray:
    return np.abs(np.asarray(radii_over_radius) - float(center_factor)) <= float(width_factor)


def compute_annular_energy(
    radii: np.ndarray,
    values: np.ndarray,
    cell_measures: np.ndarray,
    *,
    defect_radius: float,
    annulus_center_factor: float,
    annulus_width_factor: float,
    roi_max_factor: float,
) -> dict[str, Any]:
    """Compute raw, area-normalized, and fractional energy in one annulus."""

    scaled = np.asarray(radii, dtype=np.float64) / float(defect_radius)
    annulus = _annulus_mask(scaled, annulus_center_factor, annulus_width_factor)
    roi = scaled <= float(roi_max_factor)
    f = np.asarray(values, dtype=np.float64)
    w = np.asarray(cell_measures, dtype=np.float64)
    annulus_area = float(np.sum(w[annulus], dtype=np.float64))
    energy = float(np.sum(w[annulus] * f[annulus] ** 2, dtype=np.float64))
    roi_energy = float(np.sum(w[roi] * f[roi] ** 2, dtype=np.float64))
    return {
        "annulus_center_over_R": float(annulus_center_factor),
        "annulus_width_over_R": float(annulus_width_factor),
        "radius_min_over_R": float(annulus_center_factor - annulus_width_factor),
        "radius_max_over_R": float(annulus_center_factor + annulus_width_factor),
        "cell_count": int(np.count_nonzero(annulus)),
        "annulus_area": annulus_area,
        "annular_energy": energy,
        "annular_energy_density": energy / annulus_area if annulus_area > 0.0 else math.nan,
        "annular_energy_fraction": energy / roi_energy if roi_energy > 0.0 else math.nan,
        "mask": annulus,
    }


def _analyze_annulus(
    radii: np.ndarray,
    theta: np.ndarray,
    values: np.ndarray,
    measures: np.ndarray,
    *,
    defect_radius: float,
    center_factor: float,
    width_factor: float,
    config: MultipolarConfig,
) -> dict[str, Any]:
    energy = compute_annular_energy(
        radii,
        values,
        measures,
        defect_radius=defect_radius,
        annulus_center_factor=center_factor,
        annulus_width_factor=width_factor,
        roi_max_factor=config.max_radius_factor,
    )
    mask = np.asarray(energy.pop("mask"), dtype=bool)
    if np.count_nonzero(mask) < max(16, config.max_fourier_mode * 4):
        return {
            **energy,
            "status": "INSUFFICIENT_CELLS",
            "dominant_mode": 0,
            "dominant_mode_power_fraction": math.nan,
            "dominant_orientation_radians": math.nan,
            "dominant_orientation_degrees": math.nan,
            "orientation_period_degrees": math.nan,
            "signed_symmetry_180": math.nan,
            "absolute_symmetry_180": math.nan,
            "angular_coverage": 0.0,
            "angular_coverage_sufficient": False,
            "mode_power_sufficient": False,
            "interpretation_reason": "INSUFFICIENT_CELLS",
            "fourier": None,
        }
    fourier = compute_angular_fourier(
        theta[mask],
        values[mask],
        measures[mask],
        max_mode=config.max_fourier_mode,
        num_angular_bins=config.num_angular_bins,
        minimum_mode_power_fraction=config.minimum_mode_power_fraction,
        minimum_angular_coverage=config.minimum_angular_coverage,
    )
    return {
        **energy,
        "status": fourier["interpretation_status"],
        "dominant_mode": fourier["dominant_mode"],
        "dominant_mode_raw": fourier["dominant_mode_raw"],
        "dominant_mode_power_fraction": fourier["dominant_mode_power_fraction"],
        "dominant_orientation_radians": fourier["dominant_orientation_radians"],
        "dominant_orientation_degrees": fourier["dominant_orientation_degrees"],
        "orientation_period_degrees": fourier["orientation_period_degrees"],
        "signed_symmetry_180": fourier["signed_symmetry_180"],
        "absolute_symmetry_180": fourier["absolute_symmetry_180"],
        "angular_coverage": fourier["angular_coverage"],
        "angular_coverage_sufficient": fourier["angular_coverage_sufficient"],
        "mode_power_sufficient": fourier["mode_power_sufficient"],
        "interpretation_reason": fourier["interpretation_reason"],
        "fourier": fourier,
    }


def _periodic_angle_difference_degrees(left: float, right: float, period: float) -> float:
    if not all(math.isfinite(value) for value in (left, right, period)) or period <= 0.0:
        return math.nan
    delta = (left - right + 0.5 * period) % period - 0.5 * period
    return abs(delta)


def _orientation_dispersion_degrees(angles_degrees: Sequence[float], mode: int) -> float:
    valid = np.asarray([value for value in angles_degrees if math.isfinite(value)], dtype=np.float64)
    if valid.size < 2 or int(mode) <= 0:
        return 0.0 if valid.size == 1 else math.nan
    angles = np.deg2rad(valid)
    resultant = abs(np.mean(np.exp(1j * int(mode) * angles)))
    resultant = min(max(float(resultant), np.finfo(float).tiny), 1.0)
    return math.degrees(math.sqrt(max(0.0, -2.0 * math.log(resultant))) / int(mode))


def estimate_symmetry_center(
    centers: np.ndarray,
    values: np.ndarray,
    cell_measures: np.ndarray,
    defect: DefectGeometry,
    *,
    annulus_center_factor: float,
    expected_mode: int,
    config: MultipolarConfig,
) -> dict[str, Any]:
    """Search locally for the center matching the dominant mode parity.

    A pure angular mode ``n`` changes under a pi rotation by ``(-1)**n``.
    The objective therefore favors pair symmetry for even modes and
    antisymmetry for odd modes, avoiding false centers with the wrong parity.
    """

    defect.validate()
    steps = int(config.center_search_steps)
    extent = float(config.center_search_extent_factor) * defect.radius
    offsets = np.linspace(-extent, extent, steps) if steps > 1 else np.asarray([0.0])
    base_radii, _ = compute_polar_coordinates(centers, defect.center_x, defect.center_y)
    local_limit = (
        float(annulus_center_factor)
        + float(config.annulus_width_factor)
        + float(config.center_search_extent_factor)
    ) * defect.radius
    local = base_radii <= local_limit
    local_centers = np.asarray(centers)[local]
    local_values = np.asarray(values)[local]
    local_measures = np.asarray(cell_measures)[local]
    if local_values.size < 16:
        raise C66MultipolarError("Too few cells are available for symmetry-center search.")
    records: list[dict[str, Any]] = []
    for dx in offsets:
        for dy in offsets:
            candidate_x = defect.center_x + float(dx)
            candidate_y = defect.center_y + float(dy)
            radii, theta = compute_polar_coordinates(local_centers, candidate_x, candidate_y)
            annulus = _analyze_annulus(
                radii,
                theta,
                local_values,
                local_measures,
                defect_radius=defect.radius,
                center_factor=annulus_center_factor,
                width_factor=config.annulus_width_factor,
                config=config,
            )
            symmetry = float(annulus.get("signed_symmetry_180", math.nan))
            records.append(
                {
                    "center_x": candidate_x,
                    "center_y": candidate_y,
                    "offset_x_over_R": float(dx / defect.radius),
                    "offset_y_over_R": float(dy / defect.radius),
                    "distance_to_defect_center_over_R": float(math.hypot(dx, dy) / defect.radius),
                    "signed_symmetry_180": symmetry,
                    "absolute_symmetry_180": float(annulus.get("absolute_symmetry_180", math.nan)),
                    "angular_coverage": float(annulus.get("angular_coverage", 0.0)),
                    "annular_energy": float(annulus.get("annular_energy", 0.0)),
                    "cell_count": int(annulus.get("cell_count", 0)),
                }
            )
    max_energy = max((row["annular_energy"] for row in records), default=0.0)
    expected_parity = (-1.0) ** int(expected_mode) if int(expected_mode) > 0 else None
    for row in records:
        symmetry = row["signed_symmetry_180"]
        normalized_energy = row["annular_energy"] / max_energy if max_energy > 0.0 else 0.0
        parity_score = (
            expected_parity * symmetry
            if expected_parity is not None
            else abs(symmetry)
        )
        row["expected_mode"] = int(expected_mode)
        row["expected_pi_parity"] = expected_parity
        row["parity_score"] = parity_score if math.isfinite(symmetry) else math.nan
        row["objective"] = (
            max(0.0, parity_score)
            * row["angular_coverage"]
            * math.sqrt(normalized_energy)
            if math.isfinite(symmetry)
            else -math.inf
        )
    finite = [row for row in records if math.isfinite(float(row["objective"]))]
    if not finite:
        raise C66MultipolarError("No valid candidate center was found.")
    best = max(
        finite,
        key=lambda row: (
            float(row["objective"]),
            -float(row["distance_to_defect_center_over_R"]),
        ),
    )
    maximum_objective = float(best["objective"])
    plateau = [
        row
        for row in finite
        if float(row["objective"]) >= 0.95 * maximum_objective
    ]
    plateau_radius = max(
        (
            math.hypot(row["center_x"] - best["center_x"], row["center_y"] - best["center_y"])
            / defect.radius
            for row in plateau
        ),
        default=0.0,
    )
    return {
        "best": dict(best),
        "records": tuple(records),
        "plateau_radius_over_R": float(plateau_radius),
        "num_candidates": len(records),
    }


def _energy_concentration_metrics(
    radii: np.ndarray,
    values: np.ndarray,
    measures: np.ndarray,
    *,
    defect_radius: float,
    max_radius_factor: float,
) -> dict[str, float]:
    roi = np.asarray(radii) <= float(max_radius_factor) * float(defect_radius)
    energy_cells = np.asarray(measures)[roi] * np.asarray(values)[roi] ** 2
    total = float(np.sum(energy_cells, dtype=np.float64))
    if total <= 0.0 or energy_cells.size == 0:
        return {
            "max_cell_energy_fraction": math.nan,
            "top_one_percent_energy_fraction": math.nan,
        }
    maximum = float(np.max(energy_cells) / total)
    count = max(1, int(math.ceil(0.01 * energy_cells.size)))
    if count >= energy_cells.size:
        top_fraction = 1.0
    else:
        split = np.partition(energy_cells, energy_cells.size - count)
        top_fraction = float(np.sum(split[-count:], dtype=np.float64) / total)
    return {
        "max_cell_energy_fraction": maximum,
        "top_one_percent_energy_fraction": top_fraction,
    }


def _robustness_analysis(
    radii: np.ndarray,
    theta: np.ndarray,
    values: np.ndarray,
    measures: np.ndarray,
    defect: DefectGeometry,
    config: MultipolarConfig,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    radial_values = tuple(dict.fromkeys((config.num_radial_bins,) + tuple(config.radial_bin_values)))
    width_values = tuple(dict.fromkeys((config.annulus_width_factor,) + tuple(config.annulus_width_values)))
    if not config.compare_discretizations:
        radial_values = (config.num_radial_bins,)
        width_values = (config.annulus_width_factor,)
    records: list[dict[str, Any]] = []
    for num_bins in radial_values:
        profile = compute_radial_profiles(
            radii,
            values,
            measures,
            defect_radius=defect.radius,
            num_bins=int(num_bins),
            max_radius_factor=config.max_radius_factor,
        )
        peak = _primary_radius_from_profile(
            profile,
            minimum=config.primary_radius_min_factor,
            maximum=config.primary_radius_max_factor,
        )
        for width in width_values:
            annulus = _analyze_annulus(
                radii,
                theta,
                values,
                measures,
                defect_radius=defect.radius,
                center_factor=peak,
                width_factor=float(width),
                config=config,
            )
            records.append(
                {
                    "num_radial_bins": int(num_bins),
                    "annulus_width_over_R": float(width),
                    "peak_radius_over_R": peak,
                    "dominant_mode": int(annulus.get("dominant_mode", 0)),
                    "dominant_mode_raw": int(annulus.get("dominant_mode_raw", 0)),
                    "dominant_mode_power_fraction": float(
                        annulus.get("dominant_mode_power_fraction", math.nan)
                    ),
                    "orientation_degrees": float(
                        annulus.get("dominant_orientation_degrees", math.nan)
                    ),
                    "orientation_period_degrees": float(
                        annulus.get("orientation_period_degrees", math.nan)
                    ),
                    "signed_symmetry_180": float(annulus.get("signed_symmetry_180", math.nan)),
                    "angular_coverage": float(annulus.get("angular_coverage", 0.0)),
                    "angular_coverage_sufficient": bool(
                        annulus.get("angular_coverage_sufficient", False)
                    ),
                    "mode_power_sufficient": bool(
                        annulus.get("mode_power_sufficient", False)
                    ),
                    "interpretation_reason": str(
                        annulus.get("interpretation_reason", "UNKNOWN")
                    ),
                    "status": str(annulus.get("status", "ERROR")),
                }
            )
    return tuple(records), _summarize_robustness_records(records, config)


def _summarize_robustness_records(
    records: Sequence[Mapping[str, Any]],
    config: MultipolarConfig,
) -> dict[str, Any]:
    """Summarize robustness without treating under-sampled annuli as mode changes."""

    rows = [dict(record) for record in records]
    valid = [
        row
        for row in rows
        if str(row.get("status", "")) == "OK"
        and int(row.get("dominant_mode", 0)) > 0
    ]
    total = len(rows)
    num_valid = len(valid)
    required = max(
        int(config.minimum_valid_robustness_configurations),
        int(math.ceil(config.minimum_valid_robustness_fraction * total)),
    ) if total else int(config.minimum_valid_robustness_configurations)

    coverage_limited_rows = [
        row
        for row in rows
        if str(row.get("interpretation_reason", ""))
        == "INSUFFICIENT_ANGULAR_COVERAGE"
    ]
    weak_mode_rows = [
        row
        for row in rows
        if str(row.get("interpretation_reason", "")) == "WEAK_ANGULAR_MODE_POWER"
    ]
    insufficient_cell_rows = [
        row
        for row in rows
        if str(row.get("interpretation_reason", "")) == "INSUFFICIENT_CELLS"
        or str(row.get("status", "")) == "INSUFFICIENT_CELLS"
    ]
    excluded = [
        row
        for row in rows
        if not (
            str(row.get("status", "")) == "OK"
            and int(row.get("dominant_mode", 0)) > 0
        )
    ]

    valid_modes = [int(row["dominant_mode"]) for row in valid]
    if valid_modes:
        counts = {mode: valid_modes.count(mode) for mode in set(valid_modes)}
        consensus_mode = max(counts, key=lambda mode: (counts[mode], -mode))
        consensus_count = counts[consensus_mode]
        agreement_valid = consensus_count / num_valid
        agreement_all = consensus_count / total if total else 0.0
    else:
        consensus_mode = 0
        consensus_count = 0
        agreement_valid = 0.0
        agreement_all = 0.0

    consensus_rows = [
        row for row in valid if int(row.get("dominant_mode", 0)) == consensus_mode
    ]
    orientations = [float(row.get("orientation_degrees", math.nan)) for row in consensus_rows]
    orientation_std = _orientation_dispersion_degrees(orientations, consensus_mode)
    symmetries = np.asarray(
        [
            float(row.get("signed_symmetry_180", math.nan))
            for row in valid
            if math.isfinite(float(row.get("signed_symmetry_180", math.nan)))
        ],
        dtype=np.float64,
    )
    peak_by_bins: dict[int, float] = {}
    for row in rows:
        value = float(row.get("peak_radius_over_R", math.nan))
        if math.isfinite(value):
            peak_by_bins[int(row.get("num_radial_bins", len(peak_by_bins)))] = value
    peaks = np.asarray(list(peak_by_bins.values()), dtype=np.float64)
    symmetry_range = float(np.ptp(symmetries)) if symmetries.size else math.nan
    peak_range = float(np.ptp(peaks)) if peaks.size else math.nan

    enough_valid = num_valid >= required
    checks = {
        "dominant_mode_stable": bool(
            enough_valid
            and agreement_valid >= config.dominant_mode_agreement_threshold
        ),
        "orientation_stable": bool(
            enough_valid
            and math.isfinite(orientation_std)
            and orientation_std <= config.orientation_stability_degrees
        ),
        "symmetry_stable": bool(
            enough_valid
            and math.isfinite(symmetry_range)
            and symmetry_range <= config.symmetry_stability_tolerance
        ),
        "peak_radius_stable": bool(
            math.isfinite(peak_range)
            and peak_range <= config.peak_radius_stability_factor
        ),
    }

    failed_secondary = [
        name
        for name in ("orientation_stable", "symmetry_stable", "peak_radius_stable")
        if not checks[name]
    ]
    only_coverage_exclusions = bool(excluded) and len(coverage_limited_rows) == len(excluded)
    if not enough_valid:
        robustness_status = "INSUFFICIENT_VALID_CONFIGURATIONS"
        robustness_accepted = False
    elif not checks["dominant_mode_stable"]:
        robustness_status = "MODE_UNSTABLE"
        robustness_accepted = False
    elif failed_secondary:
        robustness_status = "MODE_STABLE_WITH_SECONDARY_VARIATION"
        robustness_accepted = False
    elif only_coverage_exclusions:
        robustness_status = "ROBUST_WITH_COVERAGE_LIMITATION"
        robustness_accepted = True
    elif excluded:
        robustness_status = "ROBUST_WITH_SAMPLING_LIMITATION"
        robustness_accepted = True
    else:
        robustness_status = "OK"
        robustness_accepted = True

    notes: list[str] = []
    if coverage_limited_rows:
        notes.append(
            f"{len(coverage_limited_rows)} configuration(s) excluded from the "
            "mode consensus because angular coverage was insufficient."
        )
    if weak_mode_rows:
        notes.append(
            f"{len(weak_mode_rows)} configuration(s) excluded because the "
            "dominant nonzero angular mode was too weak."
        )
    if insufficient_cell_rows:
        notes.append(
            f"{len(insufficient_cell_rows)} configuration(s) excluded because "
            "the annulus contained too few cells."
        )

    return {
        "num_configurations": total,
        "num_valid_configurations": num_valid,
        "num_excluded_configurations": len(excluded),
        "num_insufficient_coverage_configurations": len(coverage_limited_rows),
        "num_weak_mode_configurations": len(weak_mode_rows),
        "num_insufficient_cell_configurations": len(insufficient_cell_rows),
        "minimum_required_valid_configurations": required,
        "valid_configuration_fraction": num_valid / total if total else 0.0,
        "coverage_limited": bool(coverage_limited_rows),
        "consensus_dominant_mode": consensus_mode,
        "dominant_mode_consensus_count": consensus_count,
        "dominant_mode_agreement_fraction": float(agreement_valid),
        "valid_mode_agreement_fraction": float(agreement_valid),
        "dominant_mode_agreement_fraction_all_configurations": float(agreement_all),
        "orientation_dispersion_degrees": float(orientation_std),
        "signed_symmetry_range": symmetry_range,
        "peak_radius_range_over_R": peak_range,
        **checks,
        "robustness_accepted": robustness_accepted,
        "robustness_status": robustness_status,
        "robustness_failed_secondary_checks": failed_secondary,
        "robustness_notes": notes,
    }



def analyze_c66_case(
    case: CaseInput,
    *,
    config: MultipolarConfig,
) -> CaseAnalysis:
    """Analyze one signed normalized C66 field and return testable metrics."""

    config.validate()
    case.defect.validate()
    field = case.field
    centers, values, measures = _validate_field_arrays(
        field.centers,
        field.values,
        field.cell_measures,
    )
    maximum = float(np.max(np.abs(values)))
    if not math.isclose(maximum, 1.0, rel_tol=1.0e-10, abs_tol=1.0e-12):
        values, _, _ = _normalize_signed(values)
    radii, theta = compute_polar_coordinates(
        centers,
        case.defect.center_x,
        case.defect.center_y,
    )
    scaled_radii = radii / case.defect.radius
    roi = scaled_radii <= config.max_radius_factor
    if np.count_nonzero(roi) < 32:
        raise C66MultipolarError("The analysis ROI contains fewer than 32 cells.")

    radial_profile = compute_radial_profiles(
        radii,
        values,
        measures,
        defect_radius=case.defect.radius,
        num_bins=config.num_radial_bins,
        max_radius_factor=config.max_radius_factor,
    )
    primary_radius = _primary_radius_from_profile(
        radial_profile,
        minimum=config.primary_radius_min_factor,
        maximum=config.primary_radius_max_factor,
    )

    annulus_centers = tuple(
        dict.fromkeys((primary_radius,) + tuple(config.annulus_radius_factors))
    )
    annulus_results: list[dict[str, Any]] = []
    primary_annulus: dict[str, Any] | None = None
    fourier_rows: list[dict[str, Any]] = []
    for center_factor in annulus_centers:
        annulus = _analyze_annulus(
            radii,
            theta,
            values,
            measures,
            defect_radius=case.defect.radius,
            center_factor=float(center_factor),
            width_factor=config.annulus_width_factor,
            config=config,
        )
        if math.isclose(center_factor, primary_radius, rel_tol=0.0, abs_tol=1.0e-14):
            primary_annulus = annulus
        record = {key: value for key, value in annulus.items() if key != "fourier"}
        record["is_primary"] = bool(
            math.isclose(center_factor, primary_radius, rel_tol=0.0, abs_tol=1.0e-14)
        )
        annulus_results.append(record)
        fourier = annulus.get("fourier")
        if isinstance(fourier, Mapping):
            for mode_row in fourier["modes"]:
                fourier_rows.append(
                    {
                        "annulus_center_over_R": float(center_factor),
                        "annulus_width_over_R": config.annulus_width_factor,
                        "is_primary": record["is_primary"],
                        **dict(mode_row),
                    }
                )
    if primary_annulus is None:
        raise C66MultipolarError("Internal error: primary annulus was not analyzed.")

    near = scaled_radii <= config.near_defect_radius_factor
    roi_energy = float(np.sum(measures[roi] * values[roi] ** 2, dtype=np.float64))
    near_energy = float(np.sum(measures[near] * values[near] ** 2, dtype=np.float64))
    global_symmetry = compute_global_180_symmetry(
        radii,
        theta,
        values,
        measures,
        defect_radius=case.defect.radius,
        max_radius_factor=config.max_radius_factor,
        num_radial_bins=config.num_radial_bins,
        num_angular_bins=config.num_angular_bins,
    )
    concentration = _energy_concentration_metrics(
        radii,
        values,
        measures,
        defect_radius=case.defect.radius,
        max_radius_factor=config.max_radius_factor,
    )
    center = estimate_symmetry_center(
        centers,
        values,
        measures,
        case.defect,
        annulus_center_factor=primary_radius,
        expected_mode=int(primary_annulus.get("dominant_mode", 0)),
        config=config,
    )
    best_center = center["best"]
    center_distance = math.hypot(
        float(best_center["center_x"]) - case.defect.center_x,
        float(best_center["center_y"]) - case.defect.center_y,
    )

    robustness_rows, robustness_summary = _robustness_analysis(
        radii,
        theta,
        values,
        measures,
        case.defect,
        config,
    )
    center_close = center_distance / case.defect.radius <= config.center_proximity_factor
    concentration_ok = (
        math.isfinite(concentration["max_cell_energy_fraction"])
        and concentration["max_cell_energy_fraction"]
        <= config.max_single_cell_energy_fraction
    )
    warnings: list[str] = []
    if not center_close:
        warnings.append("Estimated symmetry center is not close to the defect center.")
    if not concentration_ok:
        warnings.append("The ROI energy depends excessively on one cell.")
    robustness_notes = list(robustness_summary.get("robustness_notes", ()))
    if not bool(robustness_summary.get("robustness_accepted", False)):
        robustness_status = str(robustness_summary.get("robustness_status", "UNKNOWN"))
        if robustness_status == "MODE_UNSTABLE":
            warnings.append(
                "The dominant angular mode is not stable across the interpretable "
                "robustness configurations."
            )
        elif robustness_status == "MODE_STABLE_WITH_SECONDARY_VARIATION":
            failed = ", ".join(
                str(value)
                for value in robustness_summary.get(
                    "robustness_failed_secondary_checks", ()
                )
            )
            warnings.append(
                "The dominant angular mode is stable, but secondary robustness "
                f"metrics vary beyond tolerance: {failed}."
            )
        elif robustness_status == "INSUFFICIENT_VALID_CONFIGURATIONS":
            warnings.append(
                "Too few interpretable annular configurations are available for "
                "a robust multipolar consensus."
            )
        else:
            warnings.append(
                f"Multipolar robustness validation failed with status {robustness_status}."
            )
    if primary_annulus["status"] != "OK":
        warnings.append("The primary angular pattern is weak or incompletely sampled.")
    status = "OK" if not warnings else "VALIDATION_WARNING"

    summary: dict[str, Any] = {
        "analysis_date_utc": _now_utc(),
        "format_version": FORMAT_VERSION,
        "status": status,
        "warnings": warnings,
        "robustness_notes": robustness_notes,
        "case_name": case.metadata.case_name,
        "campaign_id": case.metadata.campaign_id,
        "campaign_dir": case.metadata.campaign_dir,
        "manifest_path": case.metadata.manifest_path,
        "observation_mode": case.metadata.observation_mode,
        "modality": case.metadata.modality,
        "aperture": case.metadata.aperture,
        "final_time": case.metadata.final_time,
        "source_path": str(field.source_path),
        "source_format": field.source_format,
        "field_name": field.field_name,
        "num_values": field.num_values,
        "normalization_factor": field.normalization_factor,
        "input_was_already_normalized": field.values_were_normalized,
        "defect_center_x": case.defect.center_x,
        "defect_center_y": case.defect.center_y,
        "defect_radius": case.defect.radius,
        "analysis_max_radius_over_R": config.max_radius_factor,
        "roi_cell_count": int(np.count_nonzero(roi)),
        "roi_area": float(np.sum(measures[roi], dtype=np.float64)),
        "roi_energy": roi_energy,
        "near_defect_radius_over_R": config.near_defect_radius_factor,
        "near_defect_energy_fraction": near_energy / roi_energy if roi_energy > 0.0 else math.nan,
        "primary_radius_energy_peak_over_R": primary_radius,
        "primary_annulus_width_over_R": config.annulus_width_factor,
        "primary_annular_energy_fraction": primary_annulus["annular_energy_fraction"],
        "primary_annular_energy_density": primary_annulus["annular_energy_density"],
        "signed_symmetry_180": primary_annulus["signed_symmetry_180"],
        "absolute_symmetry_180": primary_annulus["absolute_symmetry_180"],
        "global_signed_symmetry_180": global_symmetry["signed_symmetry_180"],
        "global_absolute_symmetry_180": global_symmetry["absolute_symmetry_180"],
        "global_symmetry_paired_bin_fraction": global_symmetry["paired_bin_fraction"],
        "global_symmetry_num_paired_bins": global_symmetry["num_paired_bins"],
        "dominant_mode": primary_annulus["dominant_mode"],
        "dominant_mode_raw": primary_annulus.get("dominant_mode_raw", 0),
        "dominant_mode_power_fraction": primary_annulus["dominant_mode_power_fraction"],
        "dominant_orientation_radians": primary_annulus["dominant_orientation_radians"],
        "dominant_orientation_degrees": primary_annulus["dominant_orientation_degrees"],
        "orientation_period_degrees": primary_annulus["orientation_period_degrees"],
        "angular_coverage": primary_annulus["angular_coverage"],
        "symmetry_center_x": best_center["center_x"],
        "symmetry_center_y": best_center["center_y"],
        "symmetry_center_distance": center_distance,
        "symmetry_center_distance_over_R": center_distance / case.defect.radius,
        "symmetry_center_objective": best_center["objective"],
        "symmetry_center_plateau_radius_over_R": center["plateau_radius_over_R"],
        "symmetry_center_close_to_defect": center_close,
        **concentration,
        "single_cell_concentration_ok": concentration_ok,
        **robustness_summary,
        "configuration": asdict(config),
        "field_metadata": dict(field.metadata),
    }
    return CaseAnalysis(
        summary=summary,
        radial_profile=tuple(radial_profile),
        annuli=tuple(annulus_results),
        fourier_modes=tuple(fourier_rows),
        robustness=tuple(robustness_rows),
        center_search=tuple(center["records"]),
    )


def _rasterize_local_field(
    centers: np.ndarray,
    values: np.ndarray,
    measures: np.ndarray,
    defect: DefectGeometry,
    *,
    max_radius_factor: float,
    resolution: int = 420,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    half = max_radius_factor * defect.radius
    xmin, xmax = defect.center_x - half, defect.center_x + half
    ymin, ymax = defect.center_y - half, defect.center_y + half
    x = np.asarray(centers)[:, 0]
    y = np.asarray(centers)[:, 1]
    inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    xedges = np.linspace(xmin, xmax, int(resolution) + 1)
    yedges = np.linspace(ymin, ymax, int(resolution) + 1)
    weighted, _, _ = np.histogram2d(
        x[inside], y[inside], bins=(xedges, yedges), weights=np.asarray(measures)[inside] * np.asarray(values)[inside]
    )
    area, _, _ = np.histogram2d(
        x[inside], y[inside], bins=(xedges, yedges), weights=np.asarray(measures)[inside]
    )
    image = np.divide(weighted, area, out=np.full_like(weighted, np.nan), where=area > 0.0)
    return image.T, (xmin, xmax, ymin, ymax)


def make_composite_figure(
    case: CaseInput,
    analysis: CaseAnalysis,
    *,
    output_png: Path,
    output_pdf: Path | None,
    config: MultipolarConfig,
) -> None:
    """Create the six-panel multipolar diagnostic figure."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    image, extent = _rasterize_local_field(
        case.field.centers,
        case.field.values,
        case.field.cell_measures,
        case.defect,
        max_radius_factor=config.max_radius_factor,
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)

    ax = axes[0, 0]
    artist = ax.imshow(
        image,
        extent=extent,
        origin="lower",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        rasterized=True,
    )
    ax.add_patch(
        Circle(
            (case.defect.center_x, case.defect.center_y),
            case.defect.radius,
            fill=False,
            edgecolor="black",
            linestyle="--",
            linewidth=1.2,
        )
    )
    ax.plot(case.defect.center_x, case.defect.center_y, "k+", markersize=8, label="defect center")
    ax.plot(
        analysis.summary["symmetry_center_x"],
        analysis.summary["symmetry_center_y"],
        marker="x",
        linestyle="none",
        markersize=7,
        label="estimated symmetry center",
    )
    orientation = float(analysis.summary["dominant_orientation_radians"])
    if math.isfinite(orientation):
        length = 1.8 * case.defect.radius
        ax.plot(
            [case.defect.center_x - length * math.cos(orientation), case.defect.center_x + length * math.cos(orientation)],
            [case.defect.center_y - length * math.sin(orientation), case.defect.center_y + length * math.sin(orientation)],
            linewidth=1.4,
            label="dominant signed-lobe axis",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Signed normalized C66")
    ax.legend(fontsize=7, loc="upper right")
    fig.colorbar(artist, ax=ax, label=r"$C_{66}/\max|C_{66}|$")

    radial = analysis.radial_profile
    radius = np.asarray([row["radius_center_over_R"] for row in radial])
    ax = axes[0, 1]
    ax.plot(radius, [row["signed_mean"] for row in radial], label="signed mean")
    ax.plot(radius, [row["absolute_mean"] for row in radial], label="absolute mean")
    ax.plot(radius, [row["maximum_absolute"] for row in radial], label="maximum absolute")
    ax.axvline(1.0, linestyle="--", linewidth=1.0, label="defect boundary")
    ax.axvline(analysis.summary["primary_radius_energy_peak_over_R"], linestyle=":", linewidth=1.2, label="energy peak")
    ax.set_xlabel(r"$r/R_d$")
    ax.set_ylabel("normalized amplitude")
    ax.set_title("Radial profiles")
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    ax.plot(radius, [row["energy_fraction"] for row in radial], label="energy fraction/bin")
    ax.plot(radius, [row["energy_density"] for row in radial], label="energy/area")
    ax.axvline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$r/R_d$")
    ax.set_title("Annular energy")
    ax.legend(fontsize=7)

    primary_modes = [row for row in analysis.fourier_modes if row["is_primary"] and int(row["mode"]) >= 1]
    ax = axes[1, 0]
    ax.bar(
        [int(row["mode"]) for row in primary_modes],
        [float(row["mode_power_fraction_of_rms"]) for row in primary_modes],
    )
    ax.set_xlabel("angular Fourier mode n")
    ax.set_ylabel("mode power / angular RMS power")
    ax.set_title("Primary-annulus Fourier spectrum")

    ax = axes[1, 1]
    valid_annuli = [row for row in analysis.annuli if math.isfinite(float(row["signed_symmetry_180"]))]
    ax.plot(
        [row["annulus_center_over_R"] for row in valid_annuli],
        [row["signed_symmetry_180"] for row in valid_annuli],
        marker="o",
        label="signed S180",
    )
    ax.plot(
        [row["annulus_center_over_R"] for row in valid_annuli],
        [row["absolute_symmetry_180"] for row in valid_annuli],
        marker="s",
        label="absolute S180",
    )
    ax.axhline(0.0, linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"annulus center $r/R_d$")
    ax.set_ylabel(r"$S_{180}$")
    ax.set_title("Pi-rotation symmetry")
    ax.legend(fontsize=7)

    ax = axes[1, 2]
    primary_annulus = next(row for row in analysis.annuli if row["is_primary"])
    primary_fourier = next(
        (
            row
            for row in analysis.fourier_modes
            if row["is_primary"] and int(row["mode"]) == int(analysis.summary["dominant_mode"])
        ),
        None,
    )
    radii, theta = compute_polar_coordinates(
        case.field.centers,
        case.defect.center_x,
        case.defect.center_y,
    )
    mask = _annulus_mask(
        radii / case.defect.radius,
        float(primary_annulus["annulus_center_over_R"]),
        config.annulus_width_factor,
    )
    angular = _angular_profile(
        theta[mask], case.field.values[mask], case.field.cell_measures[mask], num_bins=config.num_angular_bins
    )
    angle_deg = np.rad2deg(np.asarray(angular["angles"]))
    profile_values = np.asarray(angular["values"])
    ax.plot(angle_deg, profile_values, label="angular profile")
    if primary_fourier is not None:
        mode = int(primary_fourier["mode"])
        coefficient = complex(primary_fourier["coefficient_real"], primary_fourier["coefficient_imag"])
        fitted = 2.0 * np.real(coefficient * np.exp(1j * mode * np.asarray(angular["angles"])))
        ax.plot(angle_deg, fitted, label=f"mode n={mode}")
    ax.set_xlabel("angle (degrees)")
    ax.set_ylabel("signed normalized C66")
    ax.set_title("Primary-annulus angular profile")
    ax.legend(fontsize=7)

    fig.suptitle(
        f"C66 multipolar pattern — {case.metadata.case_name} — "
        f"status={analysis.summary['status']}",
        fontsize=13,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf)
    plt.close(fig)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    if not columns:
        columns = ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _summary_csv_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value)
        for key, value in summary.items()
    }


def _replace_case_row(path: Path, row: Mapping[str, Any]) -> None:
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    case_name = str(row.get("case_name", ""))
    existing = [item for item in existing if str(item.get("case_name", "")) != case_name]
    existing.append(dict(row))
    _write_rows(path, existing)


def _orientation_difference_between_cases(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_mode = int(left.get("dominant_mode", 0))
    right_mode = int(right.get("dominant_mode", 0))
    if left_mode <= 0 or left_mode != right_mode:
        return math.nan
    period = 360.0 / left_mode
    return _periodic_angle_difference_degrees(
        float(left.get("dominant_orientation_degrees", math.nan)),
        float(right.get("dominant_orientation_degrees", math.nan)),
        period,
    )


def build_case_comparisons(summaries: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for index, left in enumerate(summaries):
        for right in summaries[index + 1 :]:
            rows.append(
                {
                    "analysis_mode": (
                        left.get("analysis_mode")
                        if left.get("analysis_mode") == right.get("analysis_mode")
                        else "mixed"
                    ),
                    "case_left": left.get("case_name"),
                    "case_right": right.get("case_name"),
                    "modality_left": left.get("modality"),
                    "modality_right": right.get("modality"),
                    "aperture_left": left.get("aperture"),
                    "aperture_right": right.get("aperture"),
                    "dominant_mode_left": left.get("dominant_mode"),
                    "dominant_mode_right": right.get("dominant_mode"),
                    "dominant_mode_equal": left.get("dominant_mode") == right.get("dominant_mode"),
                    "orientation_difference_degrees": _orientation_difference_between_cases(left, right),
                    "signed_symmetry_difference": abs(
                        float(left.get("signed_symmetry_180", math.nan))
                        - float(right.get("signed_symmetry_180", math.nan))
                    ),
                    "energy_peak_radius_difference_over_R": abs(
                        float(left.get("primary_radius_energy_peak_over_R", math.nan))
                        - float(right.get("primary_radius_energy_peak_over_R", math.nan))
                    ),
                    "symmetry_center_distance_difference_over_R": abs(
                        float(left.get("symmetry_center_distance_over_R", math.nan))
                        - float(right.get("symmetry_center_distance_over_R", math.nan))
                    ),
                }
            )
    return tuple(rows)


def _json_safe(value: Any) -> Any:
    """Convert NumPy and non-finite scalar values to strict JSON data."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_metrics(
    output_dir: Path,
    analyses: Sequence[CaseAnalysis],
    cases: Sequence[CaseInput],
    *,
    config: MultipolarConfig,
    missing_cases: Sequence[Mapping[str, Any]] = (),
    analysis_mode: str = "auto",
    save_json: bool = True,
    save_csv: bool = True,
) -> dict[str, Path]:
    """Write aggregate and case-level CSV/JSON outputs."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    summaries = [analysis.summary for analysis in analyses]
    if save_csv:
        summary_path = output / "C66_multipolar_summary.csv"
        for summary in summaries:
            _replace_case_row(summary_path, _summary_csv_row(summary))
        for missing in missing_cases:
            _replace_case_row(summary_path, _summary_csv_row(missing))
        paths["summary_csv"] = summary_path
        comparison_path = output / "C66_multipolar_comparisons.csv"
        _write_rows(comparison_path, build_case_comparisons(summaries))
        paths["comparisons_csv"] = comparison_path
        for analysis in analyses:
            identifier = _safe_identifier(str(analysis.summary["case_name"]))
            for label, rows in (
                ("radial_profiles", analysis.radial_profile),
                ("annuli", analysis.annuli),
                ("fourier", analysis.fourier_modes),
                ("robustness", analysis.robustness),
                ("center_search", analysis.center_search),
            ):
                path = output / f"C66_multipolar_{identifier}_{label}.csv"
                _write_rows(path, rows)
                paths[f"{identifier}_{label}_csv"] = path
    if save_json:
        json_path = output / "C66_multipolar_metrics.json"
        payload = {
            "kind": "C66_multipolar_pattern_analysis",
            "format_version": FORMAT_VERSION,
            "analysis_date_utc": _now_utc(),
            "status": (
                "ok"
                if not missing_cases
                and all(analysis.summary["status"] == "OK" for analysis in analyses)
                else "validation_warning"
            ),
            "analysis_mode": analysis_mode,
            "scope": {
                "analysis_mode": analysis_mode,
                "parameter": "C66",
                "signed_normalization": "g/max(abs(g))",
                "priority": 7,
                "solver_modified": False,
                "raw_outputs_modified": False,
            },
            "configuration": asdict(config),
            "cases": [
                {
                    "summary": analysis.summary,
                    "radial_profile": list(analysis.radial_profile),
                    "annuli": list(analysis.annuli),
                    "fourier_modes": list(analysis.fourier_modes),
                    "robustness": list(analysis.robustness),
                    "center_search": list(analysis.center_search),
                }
                for analysis in analyses
            ],
            "comparisons": list(build_case_comparisons(summaries)),
            "missing_cases": [dict(item) for item in missing_cases],
        }
        json_path.write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        paths["metrics_json"] = json_path
    return paths


def _observation_mode_alias(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "classical": "classical_full",
        "classical_full": "classical_full",
        "interaction": "residual_interaction",
        "residual_interaction": "residual_interaction",
        "reflection": "standard_reflection_only",
        "standard_reflection_only": "standard_reflection_only",
        "transmission": "standard_transmission_only",
        "standard_transmission_only": "standard_transmission_only",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported observation mode: {value!r}") from exc


def _modality_label(mode: str) -> str:
    return {
        "classical_full": "classical",
        "residual_interaction": "interaction",
        "standard_reflection_only": "reflection",
        "standard_transmission_only": "transmission",
    }.get(mode, mode)


def _aperture_from_item(item: DiscoveredSignedField) -> str:
    geometry = str(item.configuration.get("receiver_array_geometry", item.geometry_id)).lower()
    return "box" if "box" in geometry else "line" if "line" in geometry else geometry


def _final_time(configuration: Mapping[str, Any]) -> float | None:
    try:
        return float(configuration["dt"]) * int(configuration["num_steps"])
    except (KeyError, TypeError, ValueError):
        return None


def discover_c66_cases_from_campaign(
    campaign_dir: str | Path,
    *,
    observation_modes: Sequence[str] = DEFAULT_OBSERVATION_MODES,
    require_final_time: float | None = 20.0,
    final_time_tolerance: float = 1.0e-10,
) -> tuple[CaseInput, ...]:
    """Discover signed C66 cases from a validated gradient campaign manifest."""

    campaign = Path(campaign_dir).expanduser().resolve()
    manifest = campaign / "gradient_combination_campaign_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(str(manifest))
    discovery = discover_signed_fields(
        AnalysisConfig(
            input_dir=campaign,
            output_dir=campaign / ".unused_c66_discovery",
            manifest=manifest,
            field_patterns=("*C66*",),
        )
    )
    requested = {_observation_mode_alias(mode) for mode in observation_modes}
    selected = [
        item
        for item in discovery.fields
        if item.parameter_name == "C66" and item.observation_mode in requested
    ]
    missing = requested - {item.observation_mode for item in selected}
    if missing:
        raise C66MultipolarError(
            f"Missing C66 observation modes in {manifest}: {sorted(missing)}"
        )
    cases: list[CaseInput] = []
    for item in selected:
        time_value = _final_time(item.configuration)
        if require_final_time is not None:
            if time_value is None or not math.isclose(
                time_value,
                float(require_final_time),
                rel_tol=0.0,
                abs_tol=float(final_time_tolerance),
            ):
                raise C66MultipolarError(
                    f"Campaign {campaign} has T_final={time_value}, expected {require_final_time}. "
                    "No T26 campaign is accepted by this T20 analysis."
                )
        defects = load_defect_circles(item)
        if len(defects) != 1:
            raise C66MultipolarError(
                f"Expected one defect for the T20 L73 analysis, found {len(defects)}."
            )
        defect = DefectGeometry(
            center_x=defects[0].center_x,
            center_y=defects[0].center_y,
            radius=defects[0].radius,
        )
        field_name = "residual_interaction_C66" if item.observation_mode == "residual_interaction" else "gradient_C66"
        loaded = load_c66_field(
            item.input_path,
            field_name=field_name,
            metadata={
                "campaign_id": item.campaign_id,
                "geometry_id": item.geometry_id,
                "observation_mode": item.observation_mode,
                "manifest_path": str(manifest),
            },
        )
        aperture = _aperture_from_item(item)
        case_name = _safe_identifier(
            f"{aperture}_T20_{_modality_label(item.observation_mode)}_C66"
        )
        cases.append(
            CaseInput(
                field=loaded,
                defect=defect,
                metadata=CaseMetadata(
                    case_name=case_name,
                    observation_mode=item.observation_mode,
                    modality=_modality_label(item.observation_mode),
                    aperture=aperture,
                    campaign_id=item.campaign_id,
                    campaign_dir=str(campaign),
                    manifest_path=str(manifest),
                    final_time=time_value,
                ),
            )
        )
    return tuple(sorted(cases, key=lambda case: case.metadata.case_name))
