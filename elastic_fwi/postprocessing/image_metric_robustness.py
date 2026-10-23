from __future__ import annotations

"""Robustness analysis for qualitative, normalized image metrics.

The module deliberately treats input maps as qualitative indicators.  Signed
fields are converted to ``abs(field) / max(abs(field))`` before image-quality
metrics are evaluated.  Campaign PVD/PVTU/VTU fields keep their exact DG0 cell
areas through the validated VTK reader used elsewhere in the project.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from elastic_fwi.postprocessing.signed_sensitivity_fields import (
    DefectCircle,
    load_vtk_cell_field,
)


FORMAT_VERSION = "image_metric_robustness_v2"
DEFAULT_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50)
DEFAULT_CIRCLE_RADIUS_FACTORS = (0.80, 1.00, 1.20)
DEFAULT_RANKING_METRICS = (
    "active_target_energy_ratio",
    "target_energy_ratio",
    "peak_to_artifact_ratio",
    "target_active_fraction",
)

_METRIC_ALIASES = {
    "peak_to_artifact": "peak_to_artifact_ratio",
    "peak_to_artifact_ratio": "peak_to_artifact_ratio",
    "target_energy_ratio": "target_energy_ratio",
    "active_target_energy_ratio": "active_target_energy_ratio",
    "thresholded_target_energy_ratio": "active_target_energy_ratio",
    "target_energy_fraction": "target_energy_fraction",
    "target_mass_ratio": "target_mass_ratio",
    "active_target_mass_ratio": "active_target_mass_ratio",
    "thresholded_target_mass_ratio": "active_target_mass_ratio",
    "target_active_fraction": "target_active_fraction",
    "outside_active_fraction": "outside_active_fraction",
    "max_in_target": "target_peak",
    "target_peak": "target_peak",
    "contrast": "contrast_score",
    "contrast_score": "contrast_score",
    "peak_distance_to_target": "peak_distance_to_target",
    "false_maxima_count": "false_maxima_count",
}

_METRIC_DIRECTIONS = {
    "peak_to_artifact_ratio": "descending",
    "target_energy_ratio": "descending",
    "active_target_energy_ratio": "descending",
    "target_energy_fraction": "descending",
    "target_mass_ratio": "descending",
    "active_target_mass_ratio": "descending",
    "target_active_fraction": "descending",
    "outside_active_fraction": "ascending",
    "target_peak": "descending",
    "contrast_score": "descending",
    "peak_distance_to_target": "ascending",
    "false_maxima_count": "ascending",
}

# A ranking metric must declare which perturbations can change its value.
# Non-thresholded mass/energy/peak metrics are deliberately evaluated only
# once per target zone; repeating them for every threshold would over-count
# identical ranking events.
_METRIC_DEPENDENCIES = {
    "peak_to_artifact_ratio": {"threshold": False, "zone": True},
    "target_energy_ratio": {"threshold": False, "zone": True},
    "active_target_energy_ratio": {"threshold": True, "zone": True},
    "target_energy_fraction": {"threshold": False, "zone": True},
    "target_mass_ratio": {"threshold": False, "zone": True},
    "active_target_mass_ratio": {"threshold": True, "zone": True},
    "target_active_fraction": {"threshold": True, "zone": True},
    "outside_active_fraction": {"threshold": True, "zone": True},
    "target_peak": {"threshold": False, "zone": True},
    "contrast_score": {"threshold": False, "zone": True},
    "peak_distance_to_target": {"threshold": False, "zone": True},
    "false_maxima_count": {"threshold": True, "zone": True},
}

_DETAIL_COLUMNS = [
    "analysis_date_utc",
    "format_version",
    "campaign_id",
    "geometry_kind",
    "ranking_group",
    "map_name",
    "parameter_name",
    "modality",
    "field_name",
    "source_path",
    "source_format",
    "normalization_factor",
    "num_samples",
    "domain_measure",
    "threshold",
    "target_zone_id",
    "target_zone_kind",
    "target_zone_scale",
    "target_measure",
    "outside_measure",
    "num_target_samples",
    "num_outside_samples",
    "num_target_active_samples",
    "num_outside_active_samples",
    "target_active_measure",
    "outside_active_measure",
    "target_active_fraction",
    "outside_active_fraction",
    "target_active_sample_fraction",
    "outside_active_sample_fraction",
    "target_mass",
    "outside_mass",
    "target_mass_ratio",
    "target_mass_fraction",
    "active_target_mass",
    "active_outside_mass",
    "active_target_mass_ratio",
    "target_energy",
    "outside_energy",
    "target_energy_ratio",
    "target_energy_fraction",
    "target_energy_density",
    "outside_energy_density",
    "target_energy_density_ratio",
    "active_target_energy",
    "active_outside_energy",
    "active_target_energy_ratio",
    "target_mean",
    "outside_mean",
    "contrast_score",
    "global_peak",
    "global_peak_x",
    "global_peak_y",
    "target_peak",
    "outside_peak",
    "peak_to_artifact_ratio",
    "peak_inside_target",
    "peak_distance_to_target_center",
    "peak_distance_to_target",
    "peak_distance_to_target_normalized",
    "false_maximum_present",
    "outside_peak_above_threshold",
    "false_maxima_count",
    "status",
    "message",
]

_RANKING_COLUMNS = [
    "ranking_group",
    "campaign_id",
    "geometry_kind",
    "parameter_name",
    "metric",
    "metric_depends_on_threshold",
    "metric_depends_on_zone",
    "analysis_axis",
    "threshold",
    "target_zone_id",
    "map_name",
    "modality",
    "score",
    "rank",
    "is_reference_condition",
    "reference_threshold",
    "reference_zone_id",
    "top_set_size",
    "distinct_rank_count",
]

_INVERSION_COLUMNS = [
    "ranking_group",
    "campaign_id",
    "geometry_kind",
    "parameter_name",
    "metric",
    "analysis_axis",
    "map_a",
    "map_b",
    "occurrence_count",
    "conditions",
    "top_changed_any",
    "complete_reversal_any",
]

_TIE_CHANGE_COLUMNS = [
    "ranking_group",
    "campaign_id",
    "geometry_kind",
    "parameter_name",
    "metric",
    "analysis_axis",
    "map_a",
    "map_b",
    "occurrence_count",
    "conditions",
    "top_changed_any",
]



@dataclass(frozen=True)
class ImageMap:
    map_name: str
    parameter_name: str
    modality: str
    centers: np.ndarray
    values: np.ndarray
    weights: np.ndarray
    source_path: str
    source_format: str
    field_name: str = ""
    campaign_id: str = "standalone"
    geometry_kind: str = "unknown"
    grid_shape: tuple[int, int] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CampaignMapSource:
    campaign_id: str
    geometry_kind: str
    map_name: str
    parameter_name: str
    modality: str
    field_name: str
    source_path: Path
    manifest_path: Path
    defects: tuple[DefectCircle, ...]
    configuration: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetZone:
    zone_id: str
    kind: str
    mask: np.ndarray
    scale: float
    center_x: float
    center_y: float
    characteristic_radius: float
    circles: tuple[DefectCircle, ...] = ()


@dataclass(frozen=True)
class RobustnessConfig:
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    ranking_metrics: tuple[str, ...] = DEFAULT_RANKING_METRICS
    reference_threshold: float = 0.30
    reference_zone_id: str = "exact"
    false_max_threshold_fraction: float = 0.50
    false_max_min_distance_factor: float = 0.50
    fail_top_change_fraction: float = 0.25
    score_sensitivity_warning: float = 0.50
    ranking_tie_tolerance: float = 1.0e-12
    minimum_maps_per_group: int = 2


@dataclass(frozen=True)
class RobustnessResult:
    detail_rows: tuple[dict[str, Any], ...]
    ranking_rows: tuple[dict[str, Any], ...]
    inversion_rows: tuple[dict[str, Any], ...]
    tie_change_rows: tuple[dict[str, Any], ...]
    group_summaries: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_identifier(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_") or "item"


def normalize_metric_names(names: Sequence[str]) -> tuple[str, ...]:
    output: list[str] = []
    for raw in names:
        key = str(raw).strip()
        if not key:
            continue
        canonical = _METRIC_ALIASES.get(key)
        if canonical is None:
            raise ValueError(
                f"Unsupported ranking metric {key!r}. Supported names: "
                f"{sorted(_METRIC_ALIASES)}"
            )
        if canonical not in output:
            output.append(canonical)
    if not output:
        raise ValueError("At least one ranking metric is required.")
    return tuple(output)


def metric_dependencies(metric: str) -> dict[str, bool]:
    """Return whether a ranking metric depends on threshold and target zone."""

    canonical = normalize_metric_names((metric,))[0]
    return dict(_METRIC_DEPENDENCIES[canonical])


def validate_thresholds(values: Sequence[float]) -> tuple[float, ...]:
    thresholds = tuple(sorted({float(value) for value in values}))
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("Thresholds must be finite and lie in [0, 1].")
    return thresholds


def normalize_qualitative_image(values: np.ndarray) -> tuple[np.ndarray, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("Image map is empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Image map contains NaN or Inf values.")
    magnitude = np.abs(array)
    scale = float(np.max(magnitude))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Image map is identically zero and cannot be normalized.")
    return np.ascontiguousarray(magnitude / scale), scale


def validate_image_map(image: ImageMap) -> None:
    centers = np.asarray(image.centers, dtype=float)
    values = np.asarray(image.values, dtype=float).reshape(-1)
    weights = np.asarray(image.weights, dtype=float).reshape(-1)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("Image centers must have shape (n_samples, 2).")
    if centers.shape[0] != values.size or values.size != weights.size:
        raise ValueError("Image centers, values, and weights have incompatible sizes.")
    if not np.all(np.isfinite(centers)):
        raise ValueError("Image centers contain NaN or Inf values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Image values contain NaN or Inf values.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Image weights must be finite and strictly positive.")
    if image.grid_shape is not None and int(np.prod(image.grid_shape)) != values.size:
        raise ValueError("grid_shape does not match the number of image samples.")


def _safe_ratio(numerator: float, denominator: float) -> float:
    eps = np.finfo(np.float64).tiny
    if denominator > eps:
        return float(numerator / denominator)
    if numerator > eps:
        return float("inf")
    return 0.0


def _parameter_name(field_name: str) -> str:
    for prefix in ("residual_interaction_", "gradient_", "kernel_"):
        if field_name.startswith(prefix):
            return field_name[len(prefix) :]
    return field_name


def _resolve_manifest_path(raw: str | Path, *, manifest_path: Path, campaign_dir: Path) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    probes = [
        Path.cwd() / candidate,
        manifest_path.parent / candidate,
        campaign_dir / candidate,
    ]
    parts = candidate.parts
    if "outputs" in parts:
        suffix = Path(*parts[parts.index("outputs") :])
        probes.extend((Path.cwd() / suffix, campaign_dir.parent / suffix))
    for probe in probes:
        if probe.exists():
            return probe.resolve()
    return probes[0].resolve()


def _load_defects_from_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    campaign_dir: Path,
    override_center: tuple[float, float] | None = None,
    override_radius: float | None = None,
) -> tuple[DefectCircle, ...]:
    if override_center is not None or override_radius is not None:
        if override_center is None or override_radius is None:
            raise ValueError("Defect center and radius overrides must be provided together.")
        radius = float(override_radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("Defect radius override must be finite and positive.")
        return (
            DefectCircle(
                center_x=float(override_center[0]),
                center_y=float(override_center[1]),
                radius=radius,
                name="defect",
            ),
        )

    config = payload.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    direct_keys = ("defect_center_x", "defect_center_y", "defect_radius")
    if all(key in config for key in direct_keys):
        radius = float(config["defect_radius"])
        if radius <= 0.0:
            raise ValueError("Campaign defect_radius must be positive.")
        return (
            DefectCircle(
                center_x=float(config["defect_center_x"]),
                center_y=float(config["defect_center_y"]),
                radius=radius,
                name="defect",
            ),
        )

    raw = config.get("defects_json")
    if raw:
        path = _resolve_manifest_path(raw, manifest_path=manifest_path, campaign_dir=campaign_dir)
        if path.exists():
            defect_payload = json.loads(path.read_text(encoding="utf-8"))
            circles: list[DefectCircle] = []
            for index, defect in enumerate(defect_payload.get("defects", [])):
                if not isinstance(defect, Mapping):
                    continue
                circles.append(
                    DefectCircle(
                        center_x=float(defect["center_x"]),
                        center_y=float(defect["center_y"]),
                        radius=float(defect["radius"]),
                        name=str(defect.get("name", f"defect_{index + 1}")),
                    )
                )
            if circles:
                return tuple(circles)
    raise ValueError(
        f"No usable defect geometry was found in campaign manifest {manifest_path}. "
        "Provide --defect-center-x, --defect-center-y, and --defect-radius."
    )


def discover_campaign_maps(
    campaign_dir: str | Path,
    *,
    modalities: Sequence[str] = (),
    parameters: Sequence[str] = (),
    defect_center: tuple[float, float] | None = None,
    defect_radius: float | None = None,
    require_final_time: float | None = None,
) -> tuple[CampaignMapSource, ...]:
    root = Path(campaign_dir).expanduser().resolve()
    manifest_path = root / "gradient_combination_campaign_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing campaign manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    if require_final_time is not None:
        dt = float(config.get("dt", float("nan")))
        num_steps = int(config.get("num_steps", -1))
        final_time = dt * num_steps
        if not math.isfinite(final_time) or not math.isclose(
            final_time, float(require_final_time), rel_tol=0.0, abs_tol=1.0e-10
        ):
            raise ValueError(
                f"Campaign {root} has dt*num_steps={final_time}, expected {require_final_time}."
            )
    defects = _load_defects_from_manifest(
        payload,
        manifest_path=manifest_path,
        campaign_dir=root,
        override_center=defect_center,
        override_radius=defect_radius,
    )
    validation = payload.get("validation", {})
    found = validation.get("found", {}) if isinstance(validation, Mapping) else {}
    if not isinstance(found, Mapping) or not found:
        raise ValueError(f"Missing validation.found map inventory in {manifest_path}.")

    selected_modalities = {str(value) for value in modalities if str(value)}
    selected_parameters = {str(value) for value in parameters if str(value)}
    campaign_id = sanitize_identifier(str(payload.get("output_dir", root.name)).split("/")[-1])
    geometry_kind = str(config.get("receiver_array_geometry", "unknown"))
    sources: list[CampaignMapSource] = []
    for modality, fields in sorted(found.items()):
        if selected_modalities and modality not in selected_modalities:
            continue
        if not isinstance(fields, Mapping):
            continue
        for field_name, raw_path in sorted(fields.items()):
            parameter = _parameter_name(str(field_name))
            if selected_parameters and parameter not in selected_parameters:
                continue
            path = _resolve_manifest_path(
                str(raw_path), manifest_path=manifest_path, campaign_dir=root
            )
            sources.append(
                CampaignMapSource(
                    campaign_id=campaign_id,
                    geometry_kind=geometry_kind,
                    map_name=sanitize_identifier(f"{modality}_{parameter}"),
                    parameter_name=parameter,
                    modality=str(modality),
                    field_name=str(field_name),
                    source_path=path,
                    manifest_path=manifest_path,
                    defects=defects,
                    configuration=dict(config),
                )
            )
    if not sources:
        raise ValueError(f"No campaign maps matched the requested filters in {manifest_path}.")
    return tuple(sources)


def load_campaign_map(source: CampaignMapSource) -> ImageMap:
    loaded = load_vtk_cell_field(source.source_path, source.field_name)
    values, factor = normalize_qualitative_image(loaded.values)
    return ImageMap(
        map_name=source.map_name,
        parameter_name=source.parameter_name,
        modality=source.modality,
        centers=loaded.centers,
        values=values,
        weights=loaded.cell_measures,
        source_path=str(source.source_path),
        source_format="pvd_pvtu_vtu_dg0",
        field_name=loaded.selected_field_name,
        campaign_id=source.campaign_id,
        geometry_kind=source.geometry_kind,
        metadata={
            "normalization_factor": factor,
            "manifest_path": str(source.manifest_path),
            "num_ghost_cells_removed": loaded.num_ghost_cells_removed,
            "cell_types": list(loaded.cell_types),
        },
    )


def _regular_centers(shape: tuple[int, int], x: np.ndarray | None, y: np.ndarray | None) -> np.ndarray:
    ny, nx = shape
    if x is None:
        x_values = np.arange(nx, dtype=float)
    else:
        x_values = np.asarray(x, dtype=float).reshape(-1)
        if x_values.size != nx:
            raise ValueError("x coordinate length does not match map width.")
    if y is None:
        y_values = np.arange(ny, dtype=float)
    else:
        y_values = np.asarray(y, dtype=float).reshape(-1)
        if y_values.size != ny:
            raise ValueError("y coordinate length does not match map height.")
    xx, yy = np.meshgrid(x_values, y_values)
    return np.column_stack((xx.reshape(-1), yy.reshape(-1)))


def _regular_weights(shape: tuple[int, int], x: np.ndarray | None, y: np.ndarray | None) -> np.ndarray:
    ny, nx = shape
    dx = 1.0
    dy = 1.0
    if x is not None:
        values = np.asarray(x, dtype=float).reshape(-1)
        if values.size > 1:
            dx = float(np.median(np.abs(np.diff(values))))
    if y is not None:
        values = np.asarray(y, dtype=float).reshape(-1)
        if values.size > 1:
            dy = float(np.median(np.abs(np.diff(values))))
    return np.full(ny * nx, dx * dy, dtype=float)


def load_standalone_maps(path: str | Path) -> tuple[ImageMap, ...]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Maps input does not exist: {source}")
    if source.is_dir():
        maps: list[ImageMap] = []
        for child in sorted(source.iterdir()):
            if child.suffix.lower() in {".npy", ".npz", ".csv"}:
                maps.extend(load_standalone_maps(child))
        if not maps:
            raise ValueError(f"No supported map files found in directory {source}.")
        return tuple(maps)

    suffix = source.suffix.lower()
    if suffix == ".npy":
        raw = np.load(source, allow_pickle=False)
        if raw.ndim != 2:
            raise ValueError("Standalone NPY map must be a two-dimensional array.")
        shape = tuple(int(value) for value in raw.shape)
        values, factor = normalize_qualitative_image(raw)
        return (
            ImageMap(
                map_name=sanitize_identifier(source.stem),
                parameter_name=sanitize_identifier(source.stem),
                modality="standalone",
                centers=_regular_centers(shape, None, None),
                values=values,
                weights=np.ones(values.size, dtype=float),
                source_path=str(source),
                source_format="npy_regular",
                grid_shape=shape,
                metadata={"normalization_factor": factor},
            ),
        )

    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as data:
            keys = set(data.files)
            x = np.asarray(data["x"], dtype=float) if "x" in keys else None
            y = np.asarray(data["y"], dtype=float) if "y" in keys else None
            if "maps" in keys:
                stack = np.asarray(data["maps"], dtype=float)
                if stack.ndim == 2:
                    stack = stack[None, ...]
                if stack.ndim != 3:
                    raise ValueError("NPZ key 'maps' must have shape (n_maps, ny, nx).")
                names = (
                    [str(value) for value in np.asarray(data["map_names"]).tolist()]
                    if "map_names" in keys
                    else [f"map_{index}" for index in range(stack.shape[0])]
                )
                if len(names) != stack.shape[0]:
                    raise ValueError("map_names length does not match NPZ map stack.")
                shape = (int(stack.shape[1]), int(stack.shape[2]))
                centers = _regular_centers(shape, x, y)
                if "weights" in keys:
                    weights = np.asarray(data["weights"], dtype=float).reshape(-1)
                else:
                    weights = _regular_weights(shape, x, y)
                output = []
                for index, name in enumerate(names):
                    values, factor = normalize_qualitative_image(stack[index])
                    output.append(
                        ImageMap(
                            map_name=sanitize_identifier(name),
                            parameter_name=sanitize_identifier(name),
                            modality="standalone",
                            centers=centers,
                            values=values,
                            weights=weights,
                            source_path=str(source),
                            source_format="npz_regular_stack",
                            grid_shape=shape,
                            metadata={"normalization_factor": factor},
                        )
                    )
                return tuple(output)

            if "centers" in keys:
                centers = np.asarray(data["centers"], dtype=float)
                if centers.ndim != 2 or centers.shape[1] != 2:
                    raise ValueError("NPZ centers must have shape (n_samples, 2).")
                if "weights" in keys:
                    weights = np.asarray(data["weights"], dtype=float).reshape(-1)
                elif "cell_measures" in keys:
                    weights = np.asarray(data["cell_measures"], dtype=float).reshape(-1)
                else:
                    weights = np.ones(centers.shape[0], dtype=float)
                reserved = {"centers", "weights", "cell_measures", "mask", "x", "y"}
                map_keys = [
                    key
                    for key in data.files
                    if key not in reserved and np.asarray(data[key]).ndim == 1
                    and np.asarray(data[key]).size == centers.shape[0]
                ]
                if not map_keys:
                    raise ValueError("Point-cloud NPZ contains no one-dimensional map arrays.")
                output = []
                for key in map_keys:
                    values, factor = normalize_qualitative_image(np.asarray(data[key]))
                    output.append(
                        ImageMap(
                            map_name=sanitize_identifier(key),
                            parameter_name=sanitize_identifier(key),
                            modality="standalone",
                            centers=centers,
                            values=values,
                            weights=weights,
                            source_path=str(source),
                            source_format="npz_point_cloud",
                            metadata={"normalization_factor": factor},
                        )
                    )
                return tuple(output)

            arrays = {
                key: np.asarray(data[key], dtype=float)
                for key in data.files
                if key not in {"mask", "x", "y", "weights"}
                and np.asarray(data[key]).ndim == 2
            }
            if not arrays:
                raise ValueError("Unsupported NPZ map schema.")
            shapes = {array.shape for array in arrays.values()}
            if len(shapes) != 1:
                raise ValueError("All two-dimensional NPZ maps must have the same shape.")
            shape = tuple(int(value) for value in next(iter(shapes)))
            centers = _regular_centers(shape, x, y)
            weights = (
                np.asarray(data["weights"], dtype=float).reshape(-1)
                if "weights" in keys
                else _regular_weights(shape, x, y)
            )
            output = []
            for key, raw in arrays.items():
                values, factor = normalize_qualitative_image(raw)
                output.append(
                    ImageMap(
                        map_name=sanitize_identifier(key),
                        parameter_name=sanitize_identifier(key),
                        modality="standalone",
                        centers=centers,
                        values=values,
                        weights=weights,
                        source_path=str(source),
                        source_format="npz_regular_named_arrays",
                        grid_shape=shape,
                        metadata={"normalization_factor": factor},
                    )
                )
            return tuple(output)

    if suffix == ".csv":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pandas is required to read standalone CSV maps.") from exc
        frame = pd.read_csv(source)
        required = {"x", "y", "value"}
        if not required.issubset(frame.columns):
            raise ValueError("CSV maps require columns x, y, and value.")
        if "map_name" not in frame.columns:
            frame["map_name"] = source.stem
        output = []
        for name, group in frame.groupby("map_name", sort=True):
            centers = group[["x", "y"]].to_numpy(dtype=float)
            raw = group["value"].to_numpy(dtype=float)
            weights = (
                group["weight"].to_numpy(dtype=float)
                if "weight" in group.columns
                else np.ones(raw.size, dtype=float)
            )
            values, factor = normalize_qualitative_image(raw)
            parameter = str(group["parameter_name"].iloc[0]) if "parameter_name" in group.columns else str(name)
            modality = str(group["modality"].iloc[0]) if "modality" in group.columns else "standalone"
            output.append(
                ImageMap(
                    map_name=sanitize_identifier(str(name)),
                    parameter_name=sanitize_identifier(parameter),
                    modality=modality,
                    centers=centers,
                    values=values,
                    weights=weights,
                    source_path=str(source),
                    source_format="csv_point_cloud",
                    metadata={"normalization_factor": factor},
                )
            )
        return tuple(output)

    raise ValueError(
        f"Unsupported maps format {source.suffix!r}; expected NPY, NPZ, CSV, or a directory."
    )


def load_explicit_mask(path: str | Path, *, expected_size: int) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Mask input does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".npy":
        raw = np.load(source, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as data:
            if "mask" in data.files:
                raw = np.asarray(data["mask"])
            elif len(data.files) == 1:
                raw = np.asarray(data[data.files[0]])
            else:
                raise ValueError("Mask NPZ must contain a 'mask' key or a single array.")
    elif suffix == ".csv":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pandas is required to read a CSV mask.") from exc
        frame = pd.read_csv(source)
        if "mask" not in frame.columns:
            raise ValueError("CSV mask requires a 'mask' column.")
        raw = frame["mask"].to_numpy()
    else:
        raise ValueError("Mask format must be NPY, NPZ, or CSV.")
    mask = np.asarray(raw).astype(bool).reshape(-1)
    if mask.size != int(expected_size):
        raise ValueError(
            f"Mask has {mask.size} samples, expected {expected_size}."
        )
    if not np.any(mask):
        raise ValueError("Target mask is empty.")
    if np.all(mask):
        raise ValueError("Target mask covers the complete domain; no outside zone remains.")
    return mask


def _shift_boolean(array: np.ndarray, dy: int, dx: int, *, fill: bool) -> np.ndarray:
    output = np.full_like(array, fill, dtype=bool)
    y_src_start = max(0, -dy)
    y_src_end = array.shape[0] - max(0, dy)
    x_src_start = max(0, -dx)
    x_src_end = array.shape[1] - max(0, dx)
    y_dst_start = max(0, dy)
    y_dst_end = y_dst_start + (y_src_end - y_src_start)
    x_dst_start = max(0, dx)
    x_dst_end = x_dst_start + (x_src_end - x_src_start)
    if y_src_end > y_src_start and x_src_end > x_src_start:
        output[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = array[
            y_src_start:y_src_end, x_src_start:x_src_end
        ]
    return output


def binary_dilation(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if result.ndim != 2:
        raise ValueError("Binary dilation requires a two-dimensional mask.")
    for _ in range(int(iterations)):
        neighbours = [
            _shift_boolean(result, dy, dx, fill=False)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
        result = np.logical_or.reduce(neighbours)
    return result


def binary_erosion(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if result.ndim != 2:
        raise ValueError("Binary erosion requires a two-dimensional mask.")
    for _ in range(int(iterations)):
        neighbours = [
            _shift_boolean(result, dy, dx, fill=False)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
        result = np.logical_and.reduce(neighbours)
    return result


def make_explicit_mask_zones(
    image: ImageMap,
    base_mask: np.ndarray,
    *,
    modes: Sequence[str] = ("exact", "dilated", "eroded"),
    dilation_pixels: Sequence[int] = (1, 2, 3),
    erosion_pixels: Sequence[int] = (1, 2, 3),
) -> tuple[TargetZone, ...]:
    validate_image_map(image)
    mask = np.asarray(base_mask, dtype=bool).reshape(-1)
    if mask.size != image.values.size:
        raise ValueError("Mask and image sizes differ.")
    coordinates = np.asarray(image.centers, dtype=float)
    center = np.average(coordinates[mask], axis=0, weights=np.asarray(image.weights)[mask])
    zones: list[TargetZone] = []
    requested = {str(mode) for mode in modes}
    if "exact" in requested:
        zones.append(
            TargetZone(
                zone_id="exact",
                kind="explicit_mask",
                mask=mask.copy(),
                scale=0.0,
                center_x=float(center[0]),
                center_y=float(center[1]),
                characteristic_radius=float("nan"),
            )
        )
    if requested.intersection({"dilated", "eroded"}) and image.grid_shape is None:
        raise ValueError(
            "Dilated/eroded pixel masks require regular-grid maps with grid_shape metadata."
        )
    if image.grid_shape is not None:
        grid = mask.reshape(image.grid_shape)
        if "dilated" in requested:
            for pixels in sorted({int(value) for value in dilation_pixels if int(value) > 0}):
                candidate = binary_dilation(grid, pixels).reshape(-1)
                if np.any(candidate) and not np.all(candidate):
                    zones.append(
                        TargetZone(
                            zone_id=f"dilated_{pixels}px",
                            kind="explicit_mask_dilated",
                            mask=candidate,
                            scale=float(pixels),
                            center_x=float(center[0]),
                            center_y=float(center[1]),
                            characteristic_radius=float("nan"),
                        )
                    )
        if "eroded" in requested:
            for pixels in sorted({int(value) for value in erosion_pixels if int(value) > 0}):
                candidate = binary_erosion(grid, pixels).reshape(-1)
                if np.any(candidate) and not np.all(candidate):
                    zones.append(
                        TargetZone(
                            zone_id=f"eroded_{pixels}px",
                            kind="explicit_mask_eroded",
                            mask=candidate,
                            scale=float(pixels),
                            center_x=float(center[0]),
                            center_y=float(center[1]),
                            characteristic_radius=float("nan"),
                        )
                    )
    if not zones:
        raise ValueError("No valid target zones could be constructed from the mask.")
    return tuple(zones)


def make_circle_zones(
    centers: np.ndarray,
    defects: Sequence[DefectCircle],
    *,
    radius_factors: Sequence[float] = DEFAULT_CIRCLE_RADIUS_FACTORS,
) -> tuple[TargetZone, ...]:
    coordinates = np.asarray(centers, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("centers must have shape (n_samples, 2).")
    circles = tuple(defects)
    if not circles:
        raise ValueError("At least one defect circle is required.")
    factors = tuple(sorted({float(value) for value in radius_factors}))
    if any(not math.isfinite(value) or value <= 0.0 for value in factors):
        raise ValueError("Circle radius factors must be finite and positive.")
    total_area = sum(math.pi * circle.radius**2 for circle in circles)
    weighted_center_x = sum(
        math.pi * circle.radius**2 * circle.center_x for circle in circles
    ) / total_area
    weighted_center_y = sum(
        math.pi * circle.radius**2 * circle.center_y for circle in circles
    ) / total_area
    reference_radius = max(circle.radius for circle in circles)
    zones: list[TargetZone] = []
    for factor in factors:
        mask = np.zeros(coordinates.shape[0], dtype=bool)
        scaled_circles: list[DefectCircle] = []
        for circle in circles:
            radius = factor * circle.radius
            distance = np.linalg.norm(
                coordinates - np.asarray([circle.center_x, circle.center_y])[None, :],
                axis=1,
            )
            mask |= distance <= radius
            scaled_circles.append(
                DefectCircle(
                    center_x=circle.center_x,
                    center_y=circle.center_y,
                    radius=radius,
                    name=circle.name,
                )
            )
        if not np.any(mask):
            raise ValueError(f"Target zone factor {factor} contains no image samples.")
        if np.all(mask):
            raise ValueError(f"Target zone factor {factor} covers the complete image domain.")
        if math.isclose(factor, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            zone_id = "exact"
            kind = "circle_exact"
        elif factor > 1.0:
            zone_id = f"dilated_{factor:.3g}R"
            kind = "circle_dilated"
        else:
            zone_id = f"eroded_{factor:.3g}R"
            kind = "circle_eroded"
        zones.append(
            TargetZone(
                zone_id=zone_id,
                kind=kind,
                mask=mask,
                scale=factor,
                center_x=float(weighted_center_x),
                center_y=float(weighted_center_y),
                characteristic_radius=float(reference_radius * factor),
                circles=tuple(scaled_circles),
            )
        )
    return tuple(zones)


def _peak_distance_to_zone(point: np.ndarray, zone: TargetZone, centers: np.ndarray) -> tuple[float, float, float]:
    target_center = np.asarray([zone.center_x, zone.center_y], dtype=float)
    center_distance = float(np.linalg.norm(point - target_center))
    if zone.circles:
        distances_to_centers = []
        distances_to_zones = []
        normalized = []
        for circle in zone.circles:
            distance = float(
                np.linalg.norm(point - np.asarray([circle.center_x, circle.center_y]))
            )
            distances_to_centers.append(distance)
            distances_to_zones.append(max(0.0, distance - circle.radius))
            normalized.append(max(0.0, distance - circle.radius) / circle.radius)
        return min(distances_to_centers), min(distances_to_zones), min(normalized)
    target_points = np.asarray(centers, dtype=float)[zone.mask]
    if target_points.size == 0:
        return center_distance, float("nan"), float("nan")
    distance = float(np.min(np.linalg.norm(target_points - point[None, :], axis=1)))
    return center_distance, distance, float("nan")


def _count_separated_false_maxima(
    centers: np.ndarray,
    values: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    min_distance: float,
    max_candidates: int = 5000,
) -> int:
    indices = np.flatnonzero(candidate_mask)
    if indices.size == 0:
        return 0
    if indices.size > max_candidates:
        strongest = np.argpartition(values[indices], -max_candidates)[-max_candidates:]
        indices = indices[strongest]
    ordered = indices[np.argsort(values[indices])[::-1]]
    selected: list[np.ndarray] = []
    for index in ordered:
        point = centers[int(index)]
        if not selected:
            selected.append(point)
            continue
        previous = np.vstack(selected)
        if np.all(np.linalg.norm(previous - point[None, :], axis=1) >= min_distance):
            selected.append(point)
    return len(selected)


def compute_threshold_metrics(
    image: ImageMap,
    zone: TargetZone,
    threshold: float,
    *,
    false_max_threshold_fraction: float = 0.50,
    false_max_min_distance_factor: float = 0.50,
) -> dict[str, Any]:
    validate_image_map(image)
    values = np.asarray(image.values, dtype=np.float64).reshape(-1)
    weights = np.asarray(image.weights, dtype=np.float64).reshape(-1)
    centers = np.asarray(image.centers, dtype=np.float64)
    mask = np.asarray(zone.mask, dtype=bool).reshape(-1)
    if mask.size != values.size:
        raise ValueError("Target-zone mask and image have incompatible sizes.")
    if not np.any(mask) or np.all(mask):
        raise ValueError("Target zone must contain samples and leave outside samples.")
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must lie in [0, 1].")

    outside = ~mask
    active = values >= float(threshold)
    target_measure = float(np.sum(weights[mask], dtype=np.float64))
    outside_measure = float(np.sum(weights[outside], dtype=np.float64))
    num_target_samples = int(np.count_nonzero(mask))
    num_outside_samples = int(np.count_nonzero(outside))
    num_target_active_samples = int(np.count_nonzero(mask & active))
    num_outside_active_samples = int(np.count_nonzero(outside & active))
    target_active_measure = float(np.sum(weights[mask & active], dtype=np.float64))
    outside_active_measure = float(np.sum(weights[outside & active], dtype=np.float64))

    target_mass = float(np.sum(weights[mask] * values[mask], dtype=np.float64))
    outside_mass = float(np.sum(weights[outside] * values[outside], dtype=np.float64))
    active_target_mass = float(
        np.sum(weights[mask & active] * values[mask & active], dtype=np.float64)
    )
    active_outside_mass = float(
        np.sum(weights[outside & active] * values[outside & active], dtype=np.float64)
    )
    target_energy = float(
        np.sum(weights[mask] * values[mask] ** 2, dtype=np.float64)
    )
    outside_energy = float(
        np.sum(weights[outside] * values[outside] ** 2, dtype=np.float64)
    )
    active_target_energy = float(
        np.sum(weights[mask & active] * values[mask & active] ** 2, dtype=np.float64)
    )
    active_outside_energy = float(
        np.sum(weights[outside & active] * values[outside & active] ** 2, dtype=np.float64)
    )

    global_peak_index = int(np.argmax(values))
    global_peak = float(values[global_peak_index])
    target_peak = float(np.max(values[mask]))
    outside_peak = float(np.max(values[outside]))
    peak_point = centers[global_peak_index]
    center_distance, target_distance, normalized_distance = _peak_distance_to_zone(
        peak_point, zone, centers
    )

    if math.isfinite(zone.characteristic_radius) and zone.characteristic_radius > 0.0:
        min_distance = false_max_min_distance_factor * zone.characteristic_radius
    elif image.grid_shape is not None:
        min_distance = max(false_max_min_distance_factor, 1.0)
    else:
        extent = np.ptp(centers, axis=0)
        min_distance = false_max_min_distance_factor * max(float(np.min(extent)), 1.0)
    false_threshold = max(float(threshold), false_max_threshold_fraction * global_peak)
    false_count = _count_separated_false_maxima(
        centers,
        values,
        outside & (values >= false_threshold),
        min_distance=float(min_distance),
    )

    total_mass = target_mass + outside_mass
    total_energy = target_energy + outside_energy
    normalization_factor = float(image.metadata.get("normalization_factor", 1.0))
    return {
        "analysis_date_utc": now_utc(),
        "format_version": FORMAT_VERSION,
        "campaign_id": image.campaign_id,
        "geometry_kind": image.geometry_kind,
        "map_name": image.map_name,
        "parameter_name": image.parameter_name,
        "modality": image.modality,
        "field_name": image.field_name,
        "source_path": image.source_path,
        "source_format": image.source_format,
        "normalization_factor": normalization_factor,
        "num_samples": int(values.size),
        "domain_measure": float(np.sum(weights, dtype=np.float64)),
        "threshold": float(threshold),
        "target_zone_id": zone.zone_id,
        "target_zone_kind": zone.kind,
        "target_zone_scale": float(zone.scale),
        "target_measure": target_measure,
        "outside_measure": outside_measure,
        "num_target_samples": num_target_samples,
        "num_outside_samples": num_outside_samples,
        "num_target_active_samples": num_target_active_samples,
        "num_outside_active_samples": num_outside_active_samples,
        "target_active_measure": target_active_measure,
        "outside_active_measure": outside_active_measure,
        "target_active_fraction": _safe_ratio(target_active_measure, target_measure),
        "outside_active_fraction": _safe_ratio(outside_active_measure, outside_measure),
        "target_active_sample_fraction": _safe_ratio(
            num_target_active_samples, num_target_samples
        ),
        "outside_active_sample_fraction": _safe_ratio(
            num_outside_active_samples, num_outside_samples
        ),
        "target_mass": target_mass,
        "outside_mass": outside_mass,
        "target_mass_ratio": _safe_ratio(target_mass, outside_mass),
        "target_mass_fraction": _safe_ratio(target_mass, total_mass),
        "active_target_mass": active_target_mass,
        "active_outside_mass": active_outside_mass,
        "active_target_mass_ratio": _safe_ratio(active_target_mass, active_outside_mass),
        "target_energy": target_energy,
        "outside_energy": outside_energy,
        "target_energy_ratio": _safe_ratio(target_energy, outside_energy),
        "target_energy_fraction": _safe_ratio(target_energy, total_energy),
        "target_energy_density": _safe_ratio(target_energy, target_measure),
        "outside_energy_density": _safe_ratio(outside_energy, outside_measure),
        "target_energy_density_ratio": _safe_ratio(
            _safe_ratio(target_energy, target_measure),
            _safe_ratio(outside_energy, outside_measure),
        ),
        "active_target_energy": active_target_energy,
        "active_outside_energy": active_outside_energy,
        "active_target_energy_ratio": _safe_ratio(
            active_target_energy, active_outside_energy
        ),
        "target_mean": _safe_ratio(target_mass, target_measure),
        "outside_mean": _safe_ratio(outside_mass, outside_measure),
        "contrast_score": _safe_ratio(
            _safe_ratio(target_mass, target_measure),
            _safe_ratio(outside_mass, outside_measure),
        ),
        "global_peak": global_peak,
        "global_peak_x": float(peak_point[0]),
        "global_peak_y": float(peak_point[1]),
        "target_peak": target_peak,
        "outside_peak": outside_peak,
        "peak_to_artifact_ratio": _safe_ratio(target_peak, outside_peak),
        "peak_inside_target": bool(mask[global_peak_index]),
        "peak_distance_to_target_center": center_distance,
        "peak_distance_to_target": target_distance,
        "peak_distance_to_target_normalized": normalized_distance,
        "false_maximum_present": bool(not mask[global_peak_index]),
        "outside_peak_above_threshold": bool(outside_peak >= threshold),
        "false_maxima_count": int(false_count),
        "status": "ok",
        "message": "",
    }


def analyze_image_map(
    image: ImageMap,
    zones: Sequence[TargetZone],
    config: RobustnessConfig,
    *,
    ranking_group: str,
) -> list[dict[str, Any]]:
    thresholds = validate_thresholds(config.thresholds)
    rows: list[dict[str, Any]] = []
    for zone in zones:
        for threshold in thresholds:
            row = compute_threshold_metrics(
                image,
                zone,
                threshold,
                false_max_threshold_fraction=config.false_max_threshold_fraction,
                false_max_min_distance_factor=config.false_max_min_distance_factor,
            )
            row["ranking_group"] = ranking_group
            rows.append(row)
    return rows


def _metric_score(row: Mapping[str, Any], metric: str) -> float:
    value = float(row[metric])
    if math.isnan(value):
        return -math.inf if _METRIC_DIRECTIONS[metric] == "descending" else math.inf
    return value


def compute_ranking(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    tie_tolerance: float = 1.0e-12,
) -> list[dict[str, Any]]:
    canonical = normalize_metric_names((metric,))[0]
    direction = _METRIC_DIRECTIONS[canonical]
    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        name = str(row["map_name"])
        if name in unique:
            raise ValueError(f"Duplicate map {name!r} in one ranking condition.")
        unique[name] = row
    if direction == "descending":
        ordered = sorted(
            unique.values(),
            key=lambda row: (-_metric_score(row, canonical), str(row["map_name"])),
        )
    else:
        ordered = sorted(
            unique.values(),
            key=lambda row: (_metric_score(row, canonical), str(row["map_name"])),
        )
    ranking: list[dict[str, Any]] = []
    previous_score: float | None = None
    dense_rank = 0
    for index, row in enumerate(ordered, start=1):
        score = float(row[canonical])
        if previous_score is None or not math.isclose(
            score, previous_score, rel_tol=tie_tolerance, abs_tol=tie_tolerance
        ):
            dense_rank = index
        ranking.append(
            {
                "map_name": str(row["map_name"]),
                "score": score,
                "rank": dense_rank,
            }
        )
        previous_score = score
    return ranking


def _ranking_order(ranking: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(str(item["map_name"]) for item in ranking)


def _ranking_tie_groups(
    ranking: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], ...]:
    by_rank: dict[int, list[str]] = {}
    for item in ranking:
        by_rank.setdefault(int(item["rank"]), []).append(str(item["map_name"]))
    return tuple(
        tuple(sorted(by_rank[rank])) for rank in sorted(by_rank)
    )


def compare_rankings(
    reference: Sequence[Mapping[str, Any]],
    alternative: Sequence[Mapping[str, Any]],
    *,
    tie_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    del tie_tolerance  # Ranks already encode score ties.
    ref_ranks = {str(item["map_name"]): int(item["rank"]) for item in reference}
    alt_ranks = {str(item["map_name"]): int(item["rank"]) for item in alternative}
    if set(ref_ranks) != set(alt_ranks):
        raise ValueError("Reference and alternative rankings contain different maps.")
    names = sorted(ref_ranks)
    inversions: list[tuple[str, str]] = []
    tie_changes: list[tuple[str, str]] = []
    comparable_pairs = 0
    agreements = 0
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            ref_difference = ref_ranks[first] - ref_ranks[second]
            alt_difference = alt_ranks[first] - alt_ranks[second]
            ref_tie = ref_difference == 0
            alt_tie = alt_difference == 0
            if ref_tie or alt_tie:
                if ref_tie != alt_tie:
                    tie_changes.append((first, second))
                continue
            comparable_pairs += 1
            if ref_difference * alt_difference < 0:
                inversions.append((first, second))
            else:
                agreements += 1

    reference_order = _ranking_order(reference)
    alternative_order = _ranking_order(alternative)
    reference_tie_groups = _ranking_tie_groups(reference)
    alternative_tie_groups = _ranking_tie_groups(alternative)
    ref_best_rank = min(ref_ranks.values()) if ref_ranks else 0
    alt_best_rank = min(alt_ranks.values()) if alt_ranks else 0
    reference_top = {name for name, rank in ref_ranks.items() if rank == ref_best_rank}
    alternative_top = {name for name, rank in alt_ranks.items() if rank == alt_best_rank}
    top_changed = reference_top != alternative_top
    strict_top_changed = bool(top_changed and reference_top.isdisjoint(alternative_top))
    no_reference_ties = len(reference_tie_groups) == len(names)
    no_alternative_ties = len(alternative_tie_groups) == len(names)
    complete_reversal = bool(
        len(reference_order) > 1
        and no_reference_ties
        and no_alternative_ties
        and alternative_order == tuple(reversed(reference_order))
    )
    agreement_fraction = agreements / comparable_pairs if comparable_pairs else 1.0
    return {
        "reference_order": list(reference_order),
        "alternative_order": list(alternative_order),
        "reference_tie_groups": [list(group) for group in reference_tie_groups],
        "alternative_tie_groups": [list(group) for group in alternative_tie_groups],
        "reference_top": sorted(reference_top),
        "alternative_top": sorted(alternative_top),
        "top_changed": bool(top_changed),
        "strict_top_changed": strict_top_changed,
        "complete_reversal": complete_reversal,
        "num_inversions": len(inversions),
        "inversions": [list(pair) for pair in inversions],
        "num_tie_changes": len(tie_changes),
        "tie_changes": [list(pair) for pair in tie_changes],
        "pairwise_agreement_fraction": float(agreement_fraction),
        "reference_num_distinct_ranks": len(reference_tie_groups),
        "alternative_num_distinct_ranks": len(alternative_tie_groups),
        "reference_has_ties": len(reference_tie_groups) < len(names),
        "alternative_has_ties": len(alternative_tie_groups) < len(names),
    }


def _nearest_threshold(thresholds: Sequence[float], requested: float) -> float:
    return min(thresholds, key=lambda value: abs(float(value) - float(requested)))


def _relative_score_range(
    condition_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    reference_scores: Mapping[str, float],
) -> float:
    values_by_map: dict[str, list[float]] = {}
    for row in condition_rows:
        value = float(row[metric])
        if math.isfinite(value):
            values_by_map.setdefault(str(row["map_name"]), []).append(value)
    ranges: list[float] = []
    for name, values in values_by_map.items():
        if not values:
            continue
        reference = abs(float(reference_scores.get(name, 0.0)))
        scale = max(reference, max(abs(value) for value in values), np.finfo(np.float64).eps)
        ranges.append((max(values) - min(values)) / scale)
    return max(ranges, default=0.0)


def _status_severity(status: str) -> int:
    return {
        "not_applicable": 0,
        "not_tested": 0,
        "ok": 0,
        "warning": 1,
        "failed": 2,
    }.get(str(status), 2)


def _worst_status(*statuses: str) -> str:
    meaningful = [status for status in statuses if status not in {"not_applicable", "not_tested"}]
    if not meaningful:
        return "not_tested"
    return max(meaningful, key=_status_severity)


def _axis_summary(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    axis: str,
    applicable: bool,
    is_primary: bool,
    fail_top_change_fraction: float,
) -> dict[str, Any]:
    local = [item for item in comparisons if str(item["analysis_axis"]) == axis]
    if not applicable:
        return {
            "axis": axis,
            "status": "not_applicable",
            "num_conditions": 0,
            "num_conditions_with_inversion": 0,
            "num_conditions_with_tie_change": 0,
            "top_change_fraction": 0.0,
            "strict_top_change_fraction": 0.0,
            "inversion_condition_fraction": 0.0,
            "tie_change_condition_fraction": 0.0,
            "minimum_pairwise_agreement_fraction": 1.0,
            "complete_reversal_detected": False,
        }
    if not local:
        return {
            "axis": axis,
            "status": "not_tested",
            "num_conditions": 0,
            "num_conditions_with_inversion": 0,
            "num_conditions_with_tie_change": 0,
            "top_change_fraction": 0.0,
            "strict_top_change_fraction": 0.0,
            "inversion_condition_fraction": 0.0,
            "tie_change_condition_fraction": 0.0,
            "minimum_pairwise_agreement_fraction": 1.0,
            "complete_reversal_detected": False,
        }
    count = len(local)
    top_change_fraction = sum(bool(item["top_changed"]) for item in local) / count
    strict_top_change_fraction = sum(bool(item["strict_top_changed"]) for item in local) / count
    inversion_condition_fraction = sum(int(item["num_inversions"]) > 0 for item in local) / count
    tie_change_condition_fraction = sum(int(item["num_tie_changes"]) > 0 for item in local) / count
    complete_reversal = any(bool(item["complete_reversal"]) for item in local)
    minimum_agreement = min(float(item["pairwise_agreement_fraction"]) for item in local)
    if is_primary and (
        complete_reversal or strict_top_change_fraction > fail_top_change_fraction
    ):
        status = "failed"
    elif (
        inversion_condition_fraction > 0.0
        or top_change_fraction > 0.0
        or tie_change_condition_fraction > 0.0
    ):
        status = "warning"
    else:
        status = "ok"
    return {
        "axis": axis,
        "status": status,
        "num_conditions": count,
        "num_conditions_with_inversion": sum(int(item["num_inversions"]) > 0 for item in local),
        "num_conditions_with_tie_change": sum(int(item["num_tie_changes"]) > 0 for item in local),
        "top_change_fraction": float(top_change_fraction),
        "strict_top_change_fraction": float(strict_top_change_fraction),
        "inversion_condition_fraction": float(inversion_condition_fraction),
        "tie_change_condition_fraction": float(tie_change_condition_fraction),
        "minimum_pairwise_agreement_fraction": float(minimum_agreement),
        "complete_reversal_detected": bool(complete_reversal),
    }


def _aggregate_pair_events(
    raw_events: Sequence[Mapping[str, Any]],
    *,
    event_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    for event in raw_events:
        pair = tuple(sorted(str(value) for value in event[event_key]))
        key = (
            str(event["ranking_group"]),
            str(event["metric"]),
            str(event["analysis_axis"]),
            pair[0],
            pair[1],
        )
        payload = grouped.setdefault(
            key,
            {
                "ranking_group": str(event["ranking_group"]),
                "campaign_id": str(event.get("campaign_id", "")),
                "geometry_kind": str(event.get("geometry_kind", "")),
                "parameter_name": str(event.get("parameter_name", "")),
                "metric": str(event["metric"]),
                "analysis_axis": str(event["analysis_axis"]),
                "map_a": pair[0],
                "map_b": pair[1],
                "occurrence_count": 0,
                "conditions": [],
                "top_changed_any": False,
                "complete_reversal_any": False,
            },
        )
        payload["occurrence_count"] += 1
        payload["conditions"].append(
            {
                "threshold": float(event["threshold"]),
                "target_zone_id": str(event["target_zone_id"]),
            }
        )
        payload["top_changed_any"] = bool(
            payload["top_changed_any"] or event.get("top_changed", False)
        )
        payload["complete_reversal_any"] = bool(
            payload["complete_reversal_any"] or event.get("complete_reversal", False)
        )
    return [grouped[key] for key in sorted(grouped)]


def analyze_ranking_robustness(
    detail_rows: Sequence[Mapping[str, Any]],
    config: RobustnessConfig,
    *,
    external_reference_rankings: Mapping[str, Sequence[str]] | None = None,
) -> RobustnessResult:
    metrics = normalize_metric_names(config.ranking_metrics)
    thresholds = validate_thresholds(config.thresholds)
    reference_threshold = _nearest_threshold(thresholds, config.reference_threshold)
    rows = [dict(row) for row in detail_rows]
    if not rows:
        summary = {
            "format_version": FORMAT_VERSION,
            "analysis_date_utc": now_utc(),
            "status": "failed",
            "message": "No image-metric rows were supplied.",
            "num_maps": 0,
        }
        return RobustnessResult((), (), (), (), (), summary)

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["ranking_group"]), []).append(row)

    ranking_rows: list[dict[str, Any]] = []
    raw_inversion_events: list[dict[str, Any]] = []
    raw_tie_events: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    any_warning = False
    any_fail = False

    for group_name, group_rows in sorted(groups.items()):
        map_names = sorted({str(row["map_name"]) for row in group_rows})
        group_campaign_id = str(group_rows[0].get("campaign_id", ""))
        group_geometry = str(group_rows[0].get("geometry_kind", ""))
        group_parameter = str(group_rows[0].get("parameter_name", ""))
        zones = sorted({str(row["target_zone_id"]) for row in group_rows})
        available_thresholds = sorted({float(row["threshold"]) for row in group_rows})
        if len(map_names) < config.minimum_maps_per_group:
            any_fail = True
            group_summaries.append(
                {
                    "ranking_group": group_name,
                    "campaign_id": group_campaign_id,
                    "geometry_kind": group_geometry,
                    "parameter_name": group_parameter,
                    "status": "failed",
                    "message": (
                        f"Only {len(map_names)} map(s) are available; at least "
                        f"{config.minimum_maps_per_group} are required for ranking."
                    ),
                    "num_maps": len(map_names),
                }
            )
            continue

        reference_rows = [
            row
            for row in group_rows
            if math.isclose(
                float(row["threshold"]), reference_threshold, rel_tol=0.0, abs_tol=1.0e-12
            )
            and str(row["target_zone_id"]) == config.reference_zone_id
        ]
        if len(reference_rows) != len(map_names):
            any_fail = True
            group_summaries.append(
                {
                    "ranking_group": group_name,
                    "campaign_id": group_campaign_id,
                    "geometry_kind": group_geometry,
                    "parameter_name": group_parameter,
                    "status": "failed",
                    "message": (
                        f"Reference condition threshold={reference_threshold}, "
                        f"zone={config.reference_zone_id!r} does not contain every map."
                    ),
                    "num_maps": len(map_names),
                }
            )
            continue

        map_metadata = {
            str(row["map_name"]): {
                "modality": str(row.get("modality", "")),
            }
            for row in reference_rows
        }
        group_metric_summaries: list[dict[str, Any]] = []
        group_has_warning = False
        group_has_fail = False

        for metric_index, metric in enumerate(metrics):
            dependency = metric_dependencies(metric)
            is_primary = metric_index == 0
            internal_reference_ranking = compute_ranking(
                reference_rows, metric=metric, tie_tolerance=config.ranking_tie_tolerance
            )
            reference_ranking = internal_reference_ranking
            external = None
            if external_reference_rankings:
                external = external_reference_rankings.get(group_name)
                if external is None:
                    external = external_reference_rankings.get("all")
            if external is not None:
                external_order = tuple(str(name) for name in external)
                if set(external_order) != set(map_names):
                    raise ValueError(
                        f"External reference ranking for {group_name!r} does not match maps."
                    )
                score_lookup = {
                    str(item["map_name"]): float(item["score"])
                    for item in internal_reference_ranking
                }
                reference_ranking = [
                    {"map_name": name, "score": score_lookup[name], "rank": index + 1}
                    for index, name in enumerate(external_order)
                ]

            reference_scores = {
                str(item["map_name"]): float(item["score"])
                for item in reference_ranking
            }
            condition_specs: list[tuple[str, float, str]] = [
                ("reference", reference_threshold, config.reference_zone_id)
            ]
            if dependency["threshold"]:
                condition_specs.extend(
                    ("threshold", threshold, config.reference_zone_id)
                    for threshold in available_thresholds
                    if not math.isclose(threshold, reference_threshold, rel_tol=0.0, abs_tol=1.0e-12)
                )
            condition_specs.extend(
                ("zone", reference_threshold, zone_id)
                for zone_id in zones
                if zone_id != config.reference_zone_id
            )
            if dependency["threshold"]:
                condition_specs.extend(
                    ("joint", threshold, zone_id)
                    for threshold in available_thresholds
                    if not math.isclose(threshold, reference_threshold, rel_tol=0.0, abs_tol=1.0e-12)
                    for zone_id in zones
                    if zone_id != config.reference_zone_id
                )

            comparisons: list[dict[str, Any]] = []
            for axis, threshold, zone_id in condition_specs:
                condition_rows = [
                    row
                    for row in group_rows
                    if math.isclose(
                        float(row["threshold"]), threshold, rel_tol=0.0, abs_tol=1.0e-12
                    )
                    and str(row["target_zone_id"]) == zone_id
                ]
                if len(condition_rows) != len(map_names):
                    continue
                ranking = compute_ranking(
                    condition_rows,
                    metric=metric,
                    tie_tolerance=config.ranking_tie_tolerance,
                )
                comparison = compare_rankings(
                    reference_ranking,
                    ranking,
                    tie_tolerance=config.ranking_tie_tolerance,
                )
                comparison.update(
                    {
                        "ranking_group": group_name,
                        "campaign_id": group_campaign_id,
                        "geometry_kind": group_geometry,
                        "parameter_name": group_parameter,
                        "metric": metric,
                        "analysis_axis": axis,
                        "threshold": threshold,
                        "target_zone_id": zone_id,
                        "is_reference_condition": axis == "reference",
                    }
                )
                comparisons.append(comparison)
                top_set_size = len(comparison["alternative_top"])
                distinct_rank_count = int(comparison["alternative_num_distinct_ranks"])
                for item in ranking:
                    name = str(item["map_name"])
                    ranking_rows.append(
                        {
                            "ranking_group": group_name,
                            "campaign_id": group_campaign_id,
                            "geometry_kind": group_geometry,
                            "parameter_name": group_parameter,
                            "metric": metric,
                            "metric_depends_on_threshold": bool(dependency["threshold"]),
                            "metric_depends_on_zone": bool(dependency["zone"]),
                            "analysis_axis": axis,
                            "threshold": threshold,
                            "target_zone_id": zone_id,
                            "map_name": name,
                            "modality": map_metadata.get(name, {}).get("modality", ""),
                            "score": item["score"],
                            "rank": item["rank"],
                            "is_reference_condition": axis == "reference",
                            "reference_threshold": reference_threshold,
                            "reference_zone_id": config.reference_zone_id,
                            "top_set_size": top_set_size,
                            "distinct_rank_count": distinct_rank_count,
                        }
                    )
                for pair in comparison["inversions"]:
                    raw_inversion_events.append({**comparison, "pair": pair})
                for pair in comparison["tie_changes"]:
                    raw_tie_events.append({**comparison, "pair": pair})

            external_comparison = next(
                (item for item in comparisons if item["analysis_axis"] == "reference"),
                None,
            )
            reference_condition_inversion = bool(
                external is not None
                and external_comparison is not None
                and (
                    int(external_comparison["num_inversions"]) > 0
                    or bool(external_comparison["top_changed"])
                    or int(external_comparison["num_tie_changes"]) > 0
                )
            )
            threshold_summary = _axis_summary(
                comparisons,
                axis="threshold",
                applicable=bool(dependency["threshold"]),
                is_primary=is_primary,
                fail_top_change_fraction=config.fail_top_change_fraction,
            )
            zone_summary = _axis_summary(
                comparisons,
                axis="zone",
                applicable=bool(dependency["zone"]),
                is_primary=is_primary,
                fail_top_change_fraction=config.fail_top_change_fraction,
            )
            joint_summary = _axis_summary(
                comparisons,
                axis="joint",
                applicable=bool(dependency["threshold"] and dependency["zone"]),
                is_primary=is_primary,
                fail_top_change_fraction=config.fail_top_change_fraction,
            )

            reference_tie_groups = _ranking_tie_groups(reference_ranking)
            reference_distinct_rank_count = len(reference_tie_groups)
            reference_top_set_size = len(reference_tie_groups[0]) if reference_tie_groups else 0
            tie_change_count = sum(int(item["num_tie_changes"]) for item in comparisons)
            if reference_distinct_rank_count <= 1:
                tie_stability_status = "failed" if is_primary else "warning"
                tie_message = "The reference metric does not discriminate between the maps."
            elif reference_top_set_size > 1 or tie_change_count > 0:
                tie_stability_status = "warning"
                tie_message = "Reference ties or tie-set changes were detected."
            else:
                tie_stability_status = "ok"
                tie_message = "No ranking tie instability was detected."

            threshold_rows = [
                row
                for row in group_rows
                if str(row["target_zone_id"]) == config.reference_zone_id
            ] if dependency["threshold"] else reference_rows
            zone_rows = [
                row
                for row in group_rows
                if math.isclose(
                    float(row["threshold"]), reference_threshold, rel_tol=0.0, abs_tol=1.0e-12
                )
            ]
            threshold_score_range = (
                _relative_score_range(
                    threshold_rows, metric=metric, reference_scores=reference_scores
                )
                if dependency["threshold"]
                else 0.0
            )
            zone_score_range = _relative_score_range(
                zone_rows, metric=metric, reference_scores=reference_scores
            )
            sensitivity_warning = bool(
                is_primary
                and max(threshold_score_range, zone_score_range)
                > config.score_sensitivity_warning
            )

            metric_status = _worst_status(
                threshold_summary["status"],
                zone_summary["status"],
                joint_summary["status"],
                tie_stability_status,
                "warning" if reference_condition_inversion else "ok",
                "warning" if sensitivity_warning else "ok",
            )
            if metric_status == "failed":
                group_has_fail = True
            elif metric_status == "warning":
                group_has_warning = True

            all_axis_summaries = [threshold_summary, zone_summary, joint_summary]
            top_change_fraction = max(
                (float(item["top_change_fraction"]) for item in all_axis_summaries),
                default=0.0,
            )
            minimum_agreement = min(
                (float(item["minimum_pairwise_agreement_fraction"]) for item in all_axis_summaries),
                default=1.0,
            )
            inversion_condition_fraction = max(
                (float(item["inversion_condition_fraction"]) for item in all_axis_summaries),
                default=0.0,
            )
            group_metric_summaries.append(
                {
                    "metric": metric,
                    "is_primary": is_primary,
                    "status": metric_status,
                    "depends_on_threshold": bool(dependency["threshold"]),
                    "depends_on_zone": bool(dependency["zone"]),
                    "reference_threshold": reference_threshold,
                    "reference_zone_id": config.reference_zone_id,
                    "reference_ranking": list(_ranking_order(reference_ranking)),
                    "reference_tie_groups": [list(group) for group in reference_tie_groups],
                    "reference_top_set": list(reference_tie_groups[0]) if reference_tie_groups else [],
                    "reference_top_set_size": reference_top_set_size,
                    "reference_num_distinct_ranks": reference_distinct_rank_count,
                    "reference_condition_inversion": reference_condition_inversion,
                    "threshold_robustness_status": threshold_summary["status"],
                    "zone_robustness_status": zone_summary["status"],
                    "joint_robustness_status": joint_summary["status"],
                    "tie_stability_status": tie_stability_status,
                    "tie_stability_message": tie_message,
                    "threshold_analysis": threshold_summary,
                    "zone_analysis": zone_summary,
                    "joint_analysis": joint_summary,
                    "top_change_fraction": float(top_change_fraction),
                    "inversion_condition_fraction": float(inversion_condition_fraction),
                    "minimum_pairwise_agreement_fraction": float(minimum_agreement),
                    "maximum_relative_score_range": float(
                        max(threshold_score_range, zone_score_range)
                    ),
                    "threshold_relative_score_range": float(threshold_score_range),
                    "zone_relative_score_range": float(zone_score_range),
                    "score_sensitivity_warning_detected": sensitivity_warning,
                    "num_evaluated_conditions": len(comparisons),
                    "num_raw_inversion_occurrences": sum(
                        int(item["num_inversions"]) for item in comparisons
                    ),
                    "num_raw_tie_change_occurrences": tie_change_count,
                    "alternative_rankings": [
                        {
                            "analysis_axis": str(item["analysis_axis"]),
                            "threshold": float(item["threshold"]),
                            "target_zone_id": str(item["target_zone_id"]),
                            "ranking": list(item["alternative_order"]),
                            "tie_groups": item["alternative_tie_groups"],
                            "top_set": item["alternative_top"],
                            "top_changed": bool(item["top_changed"]),
                            "strict_top_changed": bool(item["strict_top_changed"]),
                            "num_inversions": int(item["num_inversions"]),
                            "num_tie_changes": int(item["num_tie_changes"]),
                            "pairwise_agreement_fraction": float(
                                item["pairwise_agreement_fraction"]
                            ),
                        }
                        for item in comparisons
                        if item["analysis_axis"] != "reference"
                    ],
                }
            )

        if group_has_fail:
            status = "failed"
            message = "A major primary-ranking instability or non-discriminative primary metric was detected."
            any_fail = True
        elif group_has_warning:
            status = "warning"
            message = "Ranking, tie, or score sensitivity was detected."
            any_warning = True
        else:
            status = "ok"
            message = "Threshold and target-zone robustness checks are satisfied."
        primary_summary = group_metric_summaries[0] if group_metric_summaries else {}
        group_summaries.append(
            {
                "ranking_group": group_name,
                "campaign_id": group_campaign_id,
                "geometry_kind": group_geometry,
                "parameter_name": group_parameter,
                "status": status,
                "message": message,
                "num_maps": len(map_names),
                "maps": map_names,
                "primary_metric": primary_summary.get("metric"),
                "reference_ranking": primary_summary.get("reference_ranking", []),
                "threshold_robustness_status": primary_summary.get(
                    "threshold_robustness_status", "not_tested"
                ),
                "zone_robustness_status": primary_summary.get(
                    "zone_robustness_status", "not_tested"
                ),
                "joint_robustness_status": primary_summary.get(
                    "joint_robustness_status", "not_tested"
                ),
                "tie_stability_status": primary_summary.get(
                    "tie_stability_status", "not_tested"
                ),
                "metrics": group_metric_summaries,
            }
        )

    inversion_rows = _aggregate_pair_events(raw_inversion_events, event_key="pair")
    tie_change_rows = _aggregate_pair_events(raw_tie_events, event_key="pair")
    global_status = "failed" if any_fail else "warning" if any_warning else "ok"
    unique_maps = sorted(
        {(str(row["ranking_group"]), str(row["map_name"])) for row in rows}
    )
    major_primary_instability_detected = any(
        group.get("status") == "failed" for group in group_summaries
    )
    major_inversion_detected = any(
        metric.get("is_primary")
        and any(
            metric.get(key) == "failed"
            for key in (
                "threshold_robustness_status",
                "zone_robustness_status",
                "joint_robustness_status",
            )
        )
        for group in group_summaries
        for metric in group.get("metrics", [])
    )
    global_threshold_status = _worst_status(
        *(str(group.get("threshold_robustness_status", "not_tested")) for group in group_summaries)
    )
    global_zone_status = _worst_status(
        *(str(group.get("zone_robustness_status", "not_tested")) for group in group_summaries)
    )
    global_joint_status = _worst_status(
        *(str(group.get("joint_robustness_status", "not_tested")) for group in group_summaries)
    )
    primary_tie_status = _worst_status(
        *(str(group.get("tie_stability_status", "not_tested")) for group in group_summaries)
    )
    all_metric_threshold_status = _worst_status(
        *(
            str(metric.get("threshold_robustness_status", "not_tested"))
            for group in group_summaries
            for metric in group.get("metrics", [])
        )
    )
    all_metric_zone_status = _worst_status(
        *(
            str(metric.get("zone_robustness_status", "not_tested"))
            for group in group_summaries
            for metric in group.get("metrics", [])
        )
    )
    all_metric_tie_status = _worst_status(
        *(
            str(metric.get("tie_stability_status", "not_tested"))
            for group in group_summaries
            for metric in group.get("metrics", [])
        )
    )
    unique_inversion_patterns = {
        (
            str(row["ranking_group"]),
            str(row["metric"]),
            str(row["analysis_axis"]),
            str(row["map_a"]),
            str(row["map_b"]),
        )
        for row in inversion_rows
    }
    summary = {
        "format_version": FORMAT_VERSION,
        "analysis_date_utc": now_utc(),
        "kind": "image_metric_robustness_summary",
        "status": global_status,
        "ranking_inversion_detected": bool(inversion_rows),
        "tie_instability_detected": bool(tie_change_rows),
        "major_ranking_inversion_detected": bool(major_inversion_detected),
        "major_primary_instability_detected": bool(major_primary_instability_detected),
        "threshold_robustness_status": global_threshold_status,
        "zone_robustness_status": global_zone_status,
        "joint_robustness_status": global_joint_status,
        "primary_tie_stability_status": primary_tie_status,
        "tie_stability_status": all_metric_tie_status,
        "all_metrics_threshold_robustness_status": all_metric_threshold_status,
        "all_metrics_zone_robustness_status": all_metric_zone_status,
        "num_maps": len(unique_maps),
        "num_detail_rows": len(rows),
        "num_ranking_rows": len(ranking_rows),
        "num_inversions": len(inversion_rows),
        "num_unique_inversion_patterns": len(unique_inversion_patterns),
        "num_raw_inversion_occurrences": len(raw_inversion_events),
        "num_tie_change_patterns": len(tie_change_rows),
        "num_raw_tie_change_occurrences": len(raw_tie_events),
        "thresholds": list(thresholds),
        "target_zones": sorted({str(row["target_zone_id"]) for row in rows}),
        "ranking_metrics": list(metrics),
        "metric_dependencies": {
            metric: metric_dependencies(metric) for metric in metrics
        },
        "reference_threshold_requested": float(config.reference_threshold),
        "reference_threshold_used": float(reference_threshold),
        "reference_zone_id": config.reference_zone_id,
        "fail_top_change_fraction": float(config.fail_top_change_fraction),
        "score_sensitivity_warning": float(config.score_sensitivity_warning),
        "groups": group_summaries,
        "inversions": inversion_rows,
        "tie_changes": tie_change_rows,
        "conclusion": (
            "robust"
            if global_status == "ok"
            else "ranking_or_tie_sensitivity_detected"
            if global_status == "warning"
            else "major_primary_ranking_instability_or_invalid_input"
        ),
        "interpretation_scope": (
            "Qualitative normalized-image metric robustness only; this diagnostic "
            "does not validate quantitative material reconstruction or gradient scale."
        ),
    }
    return RobustnessResult(
        detail_rows=tuple(rows),
        ranking_rows=tuple(ranking_rows),
        inversion_rows=tuple(inversion_rows),
        tie_change_rows=tuple(tie_change_rows),
        group_summaries=tuple(group_summaries),
        summary=summary,
    )


def load_reference_rankings(path: str | Path | None) -> dict[str, tuple[str, ...]]:
    if path is None:
        return {}
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"all": tuple(str(value) for value in payload)}
    if not isinstance(payload, Mapping):
        raise ValueError("Reference ranking JSON must be a list or an object.")
    output: dict[str, tuple[str, ...]] = {}
    for key, value in payload.items():
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("Each reference-ranking entry must be a list of map names.")
        output[str(key)] = tuple(str(item) for item in value)
    return output


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        columns = ordered
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = {}
            for key in columns:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, sort_keys=True)
                payload[key] = value
            writer.writerow(payload)
    return destination


def make_sensitivity_figure(
    detail_rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
    output_png: Path | None,
    output_pdf: Path | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    canonical_metrics = normalize_metric_names(metrics)
    groups = sorted({str(row["ranking_group"]) for row in detail_rows})
    if not groups:
        return
    figure, axes = plt.subplots(
        len(canonical_metrics),
        len(groups),
        figsize=(6.0 * len(groups), 4.0 * len(canonical_metrics)),
        squeeze=False,
    )
    for metric_index, canonical in enumerate(canonical_metrics):
        dependency = metric_dependencies(canonical)
        for group_index, group in enumerate(groups):
            axis = axes[metric_index, group_index]
            local = [row for row in detail_rows if str(row["ranking_group"]) == group]
            if dependency["threshold"]:
                for map_name in sorted({str(row["map_name"]) for row in local}):
                    for zone_id in sorted({str(row["target_zone_id"]) for row in local}):
                        subset = [
                            row
                            for row in local
                            if str(row["map_name"]) == map_name
                            and str(row["target_zone_id"]) == zone_id
                        ]
                        subset.sort(key=lambda row: float(row["threshold"]))
                        axis.plot(
                            [float(row["threshold"]) for row in subset],
                            [float(row[canonical]) for row in subset],
                            marker="o",
                            linewidth=1.0,
                            markersize=3.0,
                            label=f"{map_name} | {zone_id}",
                        )
                axis.set_xlabel("Normalized threshold")
                subtitle = "threshold and zone sensitivity"
            else:
                zone_order = sorted(
                    {str(row["target_zone_id"]) for row in local},
                    key=lambda zone_id: min(
                        float(row.get("target_zone_scale", 0.0))
                        for row in local
                        if str(row["target_zone_id"]) == zone_id
                    ),
                )
                positions = np.arange(len(zone_order), dtype=float)
                for map_name in sorted({str(row["map_name"]) for row in local}):
                    values: list[float] = []
                    for zone_id in zone_order:
                        candidates = [
                            row
                            for row in local
                            if str(row["map_name"]) == map_name
                            and str(row["target_zone_id"]) == zone_id
                        ]
                        candidates.sort(key=lambda row: float(row["threshold"]))
                        values.append(float(candidates[0][canonical]))
                    axis.plot(
                        positions,
                        values,
                        marker="o",
                        linewidth=1.0,
                        markersize=3.0,
                        label=map_name,
                    )
                axis.set_xticks(positions)
                axis.set_xticklabels(zone_order, rotation=30, ha="right")
                axis.set_xlabel("Target-zone variant")
                subtitle = "zone sensitivity; metric is threshold-independent"
            finite_values = [
                float(row[canonical])
                for row in local
                if math.isfinite(float(row[canonical])) and float(row[canonical]) > 0.0
            ]
            if finite_values and max(finite_values) / min(finite_values) > 100.0:
                axis.set_yscale("log")
            axis.set_title(f"{group}\n{canonical}: {subtitle}")
            axis.set_ylabel(canonical)
            axis.grid(True, alpha=0.25)
            if metric_index == 0 and group_index == len(groups) - 1 and len(local) <= 300:
                axis.legend(fontsize=6, ncol=2, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    figure.suptitle(
        "Image-metric robustness by declared metric dependency\n"
        "Qualitative maps normalized by max(abs(field))",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 0.95, 0.96))
    destination = output_png or output_pdf
    if destination is None:
        plt.close(figure)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if output_png is not None:
        figure.savefig(output_png, dpi=180, bbox_inches="tight")
    if output_pdf is not None:
        figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def make_ranking_stability_figure(
    group_summaries: Sequence[Mapping[str, Any]],
    *,
    output_png: Path | None,
    output_pdf: Path | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    labels: list[str] = []
    threshold_scores: list[float] = []
    zone_scores: list[float] = []
    tie_scores: list[float] = []
    for group in group_summaries:
        for metric in group.get("metrics", []):
            labels.append(f"{group['ranking_group']}\n{metric['metric']}")
            threshold = metric.get("threshold_analysis", {})
            zone = metric.get("zone_analysis", {})
            threshold_scores.append(
                1.0
                if threshold.get("status") in {"not_applicable", "not_tested"}
                else float(threshold.get("minimum_pairwise_agreement_fraction", 1.0))
            )
            zone_scores.append(
                1.0
                if zone.get("status") in {"not_applicable", "not_tested"}
                else float(zone.get("minimum_pairwise_agreement_fraction", 1.0))
            )
            tie_scores.append(
                {"ok": 1.0, "warning": 0.5, "failed": 0.0}.get(
                    str(metric.get("tie_stability_status", "ok")), 1.0
                )
            )
    if not labels:
        return
    positions = np.arange(len(labels), dtype=float)
    width = 0.26
    figure, axis = plt.subplots(figsize=(max(8.0, 0.9 * len(labels)), 5.2))
    axis.bar(positions - width, threshold_scores, width=width, label="Threshold agreement")
    axis.bar(positions, zone_scores, width=width, label="Target-zone agreement")
    axis.bar(positions + width, tie_scores, width=width, label="Tie stability")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Stability score")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    axis.set_title("Separate threshold, target-zone, and tie robustness")
    figure.tight_layout()
    destination = output_png or output_pdf
    if destination is None:
        plt.close(figure)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if output_png is not None:
        figure.savefig(output_png, dpi=180, bbox_inches="tight")
    if output_pdf is not None:
        figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def write_outputs(
    result: RobustnessResult,
    *,
    output_dir: str | Path,
    csv_name: str = "image_metric_robustness.csv",
    summary_name: str = "image_metric_robustness_summary.json",
    figure_formats: Sequence[str] = ("png", "pdf"),
) -> dict[str, Path]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    detail_path = write_csv(root / csv_name, result.detail_rows, _DETAIL_COLUMNS)
    rankings_path = write_csv(
        root / "image_metric_robustness_rankings.csv",
        result.ranking_rows,
        _RANKING_COLUMNS,
    )
    inversions_path = write_csv(
        root / "image_metric_robustness_inversions.csv",
        result.inversion_rows,
        _INVERSION_COLUMNS,
    )
    tie_changes_path = write_csv(
        root / "image_metric_robustness_tie_changes.csv",
        result.tie_change_rows,
        _TIE_CHANGE_COLUMNS,
    )

    formats = {str(value).lower() for value in figure_formats if str(value).strip()}
    unsupported = formats.difference({"png", "pdf"})
    if unsupported:
        raise ValueError(f"Unsupported figure formats: {sorted(unsupported)}")
    sensitivity_png = (
        root / "image_metric_robustness_sensitivity.png" if "png" in formats else None
    )
    sensitivity_pdf = (
        root / "image_metric_robustness_sensitivity.pdf" if "pdf" in formats else None
    )
    ranking_png = (
        root / "image_metric_robustness_ranking_stability.png" if "png" in formats else None
    )
    ranking_pdf = (
        root / "image_metric_robustness_ranking_stability.pdf" if "pdf" in formats else None
    )
    if formats:
        primary_metric = str(
            result.summary.get("ranking_metrics", ["target_energy_ratio"])[0]
        )
        sensitivity_metrics = [primary_metric]
        if primary_metric != "target_active_fraction":
            sensitivity_metrics.append("target_active_fraction")
        make_sensitivity_figure(
            result.detail_rows,
            metrics=sensitivity_metrics,
            output_png=sensitivity_png,
            output_pdf=sensitivity_pdf,
        )
        make_ranking_stability_figure(
            result.group_summaries,
            output_png=ranking_png,
            output_pdf=ranking_pdf,
        )

    output_paths: dict[str, Path] = {
        "detail_csv": detail_path,
        "rankings_csv": rankings_path,
        "inversions_csv": inversions_path,
        "tie_changes_csv": tie_changes_path,
    }
    if sensitivity_png is not None and sensitivity_png.exists():
        output_paths["sensitivity_png"] = sensitivity_png
    if sensitivity_pdf is not None and sensitivity_pdf.exists():
        output_paths["sensitivity_pdf"] = sensitivity_pdf
    if ranking_png is not None and ranking_png.exists():
        output_paths["ranking_png"] = ranking_png
    if ranking_pdf is not None and ranking_pdf.exists():
        output_paths["ranking_pdf"] = ranking_pdf

    summary_path = root / summary_name
    output_paths["summary_json"] = summary_path
    payload = dict(result.summary)
    payload["outputs"] = {
        key: str(path) for key, path in output_paths.items()
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_paths

