from __future__ import annotations

"""Signed DG0 sensitivity-field analysis for article-ready FWI campaigns.

The module keeps the numerical logic independent from the command-line entry
point.  VTK/PVD discovery is manifest-driven when possible and falls back to
the validated PVD/PVTU/VTU reader already used by the project.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
from itertools import zip_longest
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from elastic_fwi.postprocessing.vtk_cell_fields import (
    CellFieldChunk,
    VTKCellFieldError,
    iter_vtk_cell_field_chunks,
    resolve_vtk_dataset,
)


FORMAT_VERSION = "signed_sensitivity_metrics_v1"
INTEGRATION_METHOD = "cell_volume_weighted"
BARYCENTER_METHOD = "dg0_cell_center_area_weighted"
MOMENT_METHOD = "dg0_cell_center_centered_second_moment"


CSV_COLUMNS = [
    "analysis_date_utc",
    "format_version",
    "geometry_id",
    "campaign_id",
    "field_id",
    "parameter_name",
    "observation_mode",
    "input_path",
    "article_safe_path",
    "manifest_path",
    "status",
    "metric_status",
    "article_safe_status",
    "message",
    "source_format",
    "integration_method",
    "barycenter_method",
    "moment_method",
    "configuration",
    "input_sha256",
    "article_safe_sha256",
    "num_values",
    "num_positive",
    "num_negative",
    "num_zero",
    "num_ghost_cells_removed",
    "fraction_positive_values",
    "fraction_negative_values",
    "fraction_zero_values",
    "domain_measure",
    "cell_measure_min",
    "cell_measure_max",
    "min_value",
    "max_value",
    "max_abs_value",
    "signed_integral",
    "positive_integral",
    "negative_magnitude_integral",
    "absolute_integral",
    "positive_mass_fraction",
    "negative_mass_fraction",
    "signed_balance_index",
    "positive_barycenter_x",
    "positive_barycenter_y",
    "negative_barycenter_x",
    "negative_barycenter_y",
    "absolute_barycenter_x",
    "absolute_barycenter_y",
    "positive_negative_barycenter_distance",
    "positive_moment_xx",
    "positive_moment_yy",
    "positive_moment_xy",
    "positive_moment_trace",
    "positive_moment_determinant",
    "positive_moment_eigenvalue_major",
    "positive_moment_eigenvalue_minor",
    "positive_principal_direction_x",
    "positive_principal_direction_y",
    "positive_principal_direction_angle_degrees",
    "positive_principal_std",
    "positive_secondary_std",
    "positive_spatial_anisotropy_ratio",
    "positive_moment_status",
    "negative_moment_xx",
    "negative_moment_yy",
    "negative_moment_xy",
    "negative_moment_trace",
    "negative_moment_determinant",
    "negative_moment_eigenvalue_major",
    "negative_moment_eigenvalue_minor",
    "negative_principal_direction_x",
    "negative_principal_direction_y",
    "negative_principal_direction_angle_degrees",
    "negative_principal_std",
    "negative_secondary_std",
    "negative_spatial_anisotropy_ratio",
    "negative_moment_status",
    "absolute_moment_xx",
    "absolute_moment_yy",
    "absolute_moment_xy",
    "absolute_moment_trace",
    "absolute_moment_determinant",
    "absolute_moment_eigenvalue_major",
    "absolute_moment_eigenvalue_minor",
    "absolute_principal_direction_x",
    "absolute_principal_direction_y",
    "absolute_principal_direction_angle_degrees",
    "absolute_principal_std",
    "absolute_secondary_std",
    "absolute_spatial_anisotropy_ratio",
    "absolute_moment_status",
    "article_safe_shape_compatible",
    "article_safe_mesh_compatible",
    "article_safe_values_finite",
    "article_safe_max_abs_error",
    "article_safe_l2_error",
    "article_safe_relative_l2_error",
    "article_safe_allclose",
    "signed_figure_png",
    "signed_figure_pdf",
    "reconstructed_article_safe_png",
    "reconstructed_article_safe_pdf",
]


@dataclass(frozen=True)
class DefectCircle:
    center_x: float
    center_y: float
    radius: float
    name: str = "defect"


@dataclass(frozen=True)
class DiscoveredSignedField:
    geometry_id: str
    campaign_id: str
    field_id: str
    parameter_name: str
    observation_mode: str
    input_path: Path
    article_safe_candidates: tuple[Path, ...]
    manifest_path: Path | None
    configuration: Mapping[str, Any] = field(default_factory=dict)
    defect_config_path: Path | None = None


@dataclass(frozen=True)
class DiscoveryResult:
    fields: tuple[DiscoveredSignedField, ...]
    manifests: tuple[Path, ...]
    ignored: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class LoadedCellField:
    requested_path: Path
    selected_field_name: str
    centers: np.ndarray
    values: np.ndarray
    cell_measures: np.ndarray
    cell_types: tuple[str, ...]
    piece_paths: tuple[Path, ...]
    num_ghost_cells_removed: int

    @property
    def num_values(self) -> int:
        return int(self.values.size)


@dataclass(frozen=True)
class AnalysisConfig:
    input_dir: Path
    output_dir: Path
    analysis_mode: str = "auto"
    geometry_patterns: tuple[str, ...] = ()
    field_patterns: tuple[str, ...] = ("*",)
    article_safe_dir: Path | None = None
    manifest: Path | None = None
    rtol: float = 1.0e-10
    atol: float = 1.0e-12
    zero_tolerance: float = 1.0e-14
    overwrite: bool = False
    quiet: bool = False


@dataclass(frozen=True)
class AnalysisResult:
    rows: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    csv_path: Path
    manifest_path: Path
    exit_code: int


def _sanitize_identifier(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_") or "field"


def _json_compact(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _sha256_file(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_manifest_path(
    raw: str | Path,
    *,
    manifest_path: Path,
    input_dir: Path,
) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    probes: list[Path] = [
        Path.cwd() / candidate,
        manifest_path.parent / candidate,
        input_dir / candidate,
    ]
    parts = candidate.parts
    for marker in ("inverse", "gradient_combinations_raw_signed", "gradient_combinations"):
        if marker in parts:
            index = parts.index(marker)
            suffix = Path(*parts[index:])
            probes.extend((manifest_path.parent / suffix, input_dir / suffix))

    for probe in probes:
        if probe.exists():
            return probe.resolve()
    return probes[0].resolve()


def _parameter_name(field_name: str) -> str:
    for prefix in ("residual_interaction_", "gradient_", "kernel_"):
        if field_name.startswith(prefix):
            return field_name[len(prefix) :]
    return field_name


def _campaign_id(payload: Mapping[str, Any], manifest_path: Path) -> str:
    for key in ("campaign_name", "scenario_id", "name"):
        value = payload.get(key)
        if value:
            return _sanitize_identifier(str(value))
    output_dir = payload.get("output_dir") or payload.get("inverse_output_dir")
    if output_dir:
        path = Path(str(output_dir))
        if path.name == "inverse" and path.parent.name:
            return _sanitize_identifier(path.parent.name)
        if path.name:
            return _sanitize_identifier(path.name)
    return _sanitize_identifier(manifest_path.parent.name)


def _geometry_id(payload: Mapping[str, Any], campaign_id: str) -> str:
    config = payload.get("config", {})
    if isinstance(config, Mapping):
        for key in ("geometry_id", "receiver_array_geometry", "geometry"):
            value = config.get(key)
            if value:
                return _sanitize_identifier(str(value))
    for token in ("line", "box"):
        if re.search(rf"(^|[_-]){token}([_-]|$)", campaign_id, flags=re.IGNORECASE):
            return token
    return campaign_id


def _defect_config_path(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    input_dir: Path,
) -> Path | None:
    config = payload.get("config", {})
    if not isinstance(config, Mapping):
        return None
    raw = config.get("defects_json")
    if not raw:
        return None
    path = _resolve_manifest_path(raw, manifest_path=manifest_path, input_dir=input_dir)
    return path if path.exists() else None


def _vtk_path_from_export(payload: Any) -> str | None:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, Mapping):
        return None
    if isinstance(payload.get("pvd"), str):
        return str(payload["pvd"])
    vtk = payload.get("vtk")
    if isinstance(vtk, Mapping) and isinstance(vtk.get("pvd"), str):
        return str(vtk["pvd"])
    if isinstance(payload.get("paraview_entrypoint"), str):
        return str(payload["paraview_entrypoint"])
    return None


def _found_map_from_campaign_section(section: Any) -> dict[str, dict[str, str]]:
    if not isinstance(section, Mapping):
        return {}
    found = section.get("found", {})
    if not isinstance(found, Mapping):
        return {}
    result: dict[str, dict[str, str]] = {}
    for combination, fields in found.items():
        if not isinstance(fields, Mapping):
            continue
        result[str(combination)] = {
            str(name): str(path)
            for name, path in fields.items()
            if isinstance(path, (str, Path))
        }
    return result


def _found_map_from_inverse_exports(section: Any) -> dict[str, dict[str, str]]:
    if not isinstance(section, Mapping):
        return {}
    result: dict[str, dict[str, str]] = {}
    for combination, payload in section.items():
        if not isinstance(payload, Mapping):
            continue
        vtk = payload.get("vtk", {})
        fields: dict[str, str] = {}
        if isinstance(vtk, Mapping):
            for name, item in vtk.items():
                path = _vtk_path_from_export(item)
                if path:
                    fields[str(name)] = path
        if fields:
            result[str(combination)] = fields
    return result


def _article_safe_override(
    raw_path: Path,
    *,
    article_safe_dir: Path | None,
) -> Path | None:
    if article_safe_dir is None:
        return None
    parts = raw_path.parts
    if "gradient_combinations_raw_signed" in parts:
        index = parts.index("gradient_combinations_raw_signed")
        relative = Path(*parts[index + 1 :])
        return (article_safe_dir / relative).resolve()
    return (article_safe_dir / raw_path.name).resolve()


def _field_matches(
    item: DiscoveredSignedField,
    *,
    geometry_patterns: Sequence[str],
    field_patterns: Sequence[str],
) -> bool:
    if geometry_patterns and not any(
        fnmatch(item.geometry_id, pattern) or fnmatch(item.campaign_id, pattern)
        for pattern in geometry_patterns
    ):
        return False
    haystacks = (
        item.field_id,
        item.parameter_name,
        item.observation_mode,
        item.input_path.name,
        str(item.input_path),
    )
    return any(fnmatch(value, pattern) for pattern in field_patterns for value in haystacks)


def _discover_from_manifest(
    manifest_path: Path,
    *,
    input_dir: Path,
    article_safe_dir: Path | None,
) -> tuple[list[DiscoveredSignedField], list[str]]:
    payload = _read_json(manifest_path)
    kind = str(payload.get("kind", ""))
    if kind == "gradient_combination_campaign" or "raw_signed_validation" in payload:
        raw_map = _found_map_from_campaign_section(payload.get("raw_signed_validation"))
        safe_map = _found_map_from_campaign_section(payload.get("validation"))
    else:
        exports = payload.get("exports", {})
        if not isinstance(exports, Mapping):
            return [], [f"No exports mapping in manifest {manifest_path}."]
        raw_map = _found_map_from_inverse_exports(
            exports.get("gradient_combinations_raw_signed")
        )
        safe_map = _found_map_from_inverse_exports(exports.get("gradient_combinations"))

    campaign_id = _campaign_id(payload, manifest_path)
    geometry_id = _geometry_id(payload, campaign_id)
    configuration = payload.get("config", {})
    if not isinstance(configuration, Mapping):
        configuration = {}
    defect_path = _defect_config_path(
        payload,
        manifest_path=manifest_path,
        input_dir=input_dir,
    )

    fields: list[DiscoveredSignedField] = []
    warnings: list[str] = []
    for combination, raw_fields in raw_map.items():
        for field_name, raw_text in raw_fields.items():
            raw_path = _resolve_manifest_path(
                raw_text,
                manifest_path=manifest_path,
                input_dir=input_dir,
            )
            candidates: list[Path] = []
            safe_text = safe_map.get(combination, {}).get(field_name)
            if safe_text:
                candidates.append(
                    _resolve_manifest_path(
                        safe_text,
                        manifest_path=manifest_path,
                        input_dir=input_dir,
                    )
                )
            override = _article_safe_override(
                raw_path,
                article_safe_dir=article_safe_dir,
            )
            if override is not None:
                candidates.insert(0, override)
            candidates = list(dict.fromkeys(path.resolve() for path in candidates))
            parameter = _parameter_name(field_name)
            field_id = _sanitize_identifier(
                f"{campaign_id}__{combination}__{parameter}"
            )
            fields.append(
                DiscoveredSignedField(
                    geometry_id=geometry_id,
                    campaign_id=campaign_id,
                    field_id=field_id,
                    parameter_name=parameter,
                    observation_mode=str(combination),
                    input_path=raw_path,
                    article_safe_candidates=tuple(candidates),
                    manifest_path=manifest_path.resolve(),
                    configuration=dict(configuration),
                    defect_config_path=defect_path,
                )
            )
    if not raw_map:
        warnings.append(f"No raw-signed gradient mapping found in {manifest_path}.")
    return fields, warnings


def _discover_from_directory(
    input_dir: Path,
    *,
    article_safe_dir: Path | None,
) -> list[DiscoveredSignedField]:
    fields: list[DiscoveredSignedField] = []
    for path in sorted(input_dir.rglob("*.pvd")):
        parts = path.parts
        if "gradient_combinations_raw_signed" not in parts:
            continue
        index = parts.index("gradient_combinations_raw_signed")
        suffix = Path(*parts[index + 1 :])
        combination = suffix.parts[0] if len(suffix.parts) >= 2 else path.parent.parent.name
        filename = path.stem
        field_name = filename
        prefix = "material_parameter_gradients_"
        if field_name.startswith(prefix):
            field_name = field_name[len(prefix) :]
        parameter = _parameter_name(field_name)
        campaign_root = Path(*parts[:index]).resolve()
        campaign_id = _sanitize_identifier(
            campaign_root.parent.name if campaign_root.name == "inverse" else campaign_root.name
        )
        geometry_id = next(
            (
                token
                for token in ("line", "box")
                if re.search(
                    rf"(^|[_-]){token}([_-]|$)",
                    campaign_id,
                    flags=re.IGNORECASE,
                )
            ),
            campaign_id,
        )
        override = _article_safe_override(path, article_safe_dir=article_safe_dir)
        if override is not None:
            safe_candidates = (override,)
        else:
            safe_parts = list(parts)
            safe_parts[index] = "gradient_combinations"
            safe_candidates = (Path(*safe_parts).resolve(),)
        fields.append(
            DiscoveredSignedField(
                geometry_id=geometry_id,
                campaign_id=campaign_id,
                field_id=_sanitize_identifier(
                    f"{campaign_id}__{combination}__{parameter}"
                ),
                parameter_name=parameter,
                observation_mode=combination,
                input_path=path.resolve(),
                article_safe_candidates=safe_candidates,
                manifest_path=None,
            )
        )
    return fields


def discover_signed_fields(config: AnalysisConfig) -> DiscoveryResult:
    input_dir = config.input_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if config.manifest is not None:
        manifests = [config.manifest.expanduser().resolve()]
    else:
        manifests = sorted(input_dir.rglob("gradient_combination_campaign_manifest.json"))
        if not manifests:
            manifests = sorted(input_dir.rglob("inverse_time_series_manifest*.json"))

    discovered: list[DiscoveredSignedField] = []
    warnings: list[str] = []
    for manifest_path in manifests:
        if not manifest_path.exists():
            warnings.append(f"Manifest does not exist: {manifest_path}")
            continue
        fields, local_warnings = _discover_from_manifest(
            manifest_path,
            input_dir=input_dir,
            article_safe_dir=config.article_safe_dir,
        )
        discovered.extend(fields)
        warnings.extend(local_warnings)

    discovered.extend(
        _discover_from_directory(
            input_dir,
            article_safe_dir=config.article_safe_dir,
        )
    )

    merged: dict[tuple[str, str, str, str], DiscoveredSignedField] = {}
    ignored: list[dict[str, Any]] = []
    for item in discovered:
        key = (
            str(item.input_path.resolve()),
            item.observation_mode,
            item.parameter_name,
            item.campaign_id,
        )
        previous = merged.get(key)
        if previous is None:
            merged[key] = item
            continue
        candidates = tuple(
            dict.fromkeys(previous.article_safe_candidates + item.article_safe_candidates)
        )
        manifest_path = previous.manifest_path or item.manifest_path
        defect_path = previous.defect_config_path or item.defect_config_path
        configuration = previous.configuration or item.configuration
        merged[key] = DiscoveredSignedField(
            geometry_id=previous.geometry_id,
            campaign_id=previous.campaign_id,
            field_id=previous.field_id,
            parameter_name=previous.parameter_name,
            observation_mode=previous.observation_mode,
            input_path=previous.input_path,
            article_safe_candidates=candidates,
            manifest_path=manifest_path,
            configuration=configuration,
            defect_config_path=defect_path,
        )

    selected: list[DiscoveredSignedField] = []
    for item in sorted(
        merged.values(),
        key=lambda value: (
            value.geometry_id,
            value.campaign_id,
            value.observation_mode,
            value.parameter_name,
            str(value.input_path),
        ),
    ):
        if _field_matches(
            item,
            geometry_patterns=config.geometry_patterns,
            field_patterns=config.field_patterns,
        ):
            selected.append(item)
        else:
            ignored.append(
                {
                    "input_path": str(item.input_path),
                    "reason": "filtered_by_geometry_or_field_pattern",
                }
            )

    return DiscoveryResult(
        fields=tuple(selected),
        manifests=tuple(path.resolve() for path in manifests),
        ignored=tuple(ignored),
        warnings=tuple(warnings),
    )


def split_signed_field(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    return np.maximum(array, 0.0), np.maximum(-array, 0.0)


def compute_weighted_barycenter(
    centers: np.ndarray,
    nonnegative_density: np.ndarray,
    cell_measures: np.ndarray,
    *,
    mass_tolerance: float = 0.0,
) -> tuple[np.ndarray, float, str]:
    coordinates = np.asarray(centers, dtype=np.float64)
    density = np.asarray(nonnegative_density, dtype=np.float64)
    measures = np.asarray(cell_measures, dtype=np.float64)
    weights = density * measures
    mass = float(np.sum(weights, dtype=np.float64))
    if not math.isfinite(mass) or mass <= float(mass_tolerance):
        return np.asarray([math.nan, math.nan]), mass, "UNDEFINED_ZERO_MASS"
    barycenter = np.sum(
        coordinates * weights[:, None],
        axis=0,
        dtype=np.float64,
    ) / mass
    return np.asarray(barycenter, dtype=np.float64), mass, "OK"


def compute_centered_second_moment(
    centers: np.ndarray,
    nonnegative_density: np.ndarray,
    cell_measures: np.ndarray,
    *,
    mass_tolerance: float = 0.0,
) -> dict[str, Any]:
    barycenter, mass, status = compute_weighted_barycenter(
        centers,
        nonnegative_density,
        cell_measures,
        mass_tolerance=mass_tolerance,
    )
    if status != "OK":
        return {
            "barycenter": barycenter,
            "mass": mass,
            "moment": np.full((2, 2), math.nan),
            "trace": math.nan,
            "determinant": math.nan,
            "eigenvalues": np.asarray([math.nan, math.nan]),
            "principal_direction": np.asarray([math.nan, math.nan]),
            "principal_angle_degrees": math.nan,
            "principal_std": math.nan,
            "secondary_std": math.nan,
            "anisotropy_ratio": math.nan,
            "status": status,
        }

    coordinates = np.asarray(centers, dtype=np.float64)
    density = np.asarray(nonnegative_density, dtype=np.float64)
    measures = np.asarray(cell_measures, dtype=np.float64)
    weights = density * measures
    delta = coordinates - barycenter[None, :]
    moment = (delta.T * weights) @ delta / mass
    moment = 0.5 * (moment + moment.T)
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    principal = eigenvectors[:, order[0]]
    if principal[0] < 0.0 or (principal[0] == 0.0 and principal[1] < 0.0):
        principal = -principal
    major, minor = float(eigenvalues[0]), float(eigenvalues[1])
    scale_tolerance = np.finfo(np.float64).eps * max(major + minor, 1.0)
    if major <= scale_tolerance:
        principal = np.asarray([math.nan, math.nan], dtype=np.float64)
        principal_angle = math.nan
        anisotropy = 1.0
        moment_status = "ZERO_SPREAD"
    else:
        principal_angle = float(
            math.degrees(math.atan2(float(principal[1]), float(principal[0])))
        )
        if minor <= scale_tolerance:
            anisotropy = math.inf
            moment_status = "RANK_ONE"
        else:
            anisotropy = math.sqrt(major / minor)
            moment_status = "OK"
    return {
        "barycenter": barycenter,
        "mass": mass,
        "moment": np.asarray(moment, dtype=np.float64),
        "trace": major + minor,
        "determinant": major * minor,
        "eigenvalues": eigenvalues,
        "principal_direction": np.asarray(principal, dtype=np.float64),
        "principal_angle_degrees": principal_angle,
        "principal_std": math.sqrt(major),
        "secondary_std": math.sqrt(minor),
        "anisotropy_ratio": anisotropy,
        "status": moment_status,
    }


def _moment_columns(prefix: str, result: Mapping[str, Any]) -> dict[str, Any]:
    barycenter = np.asarray(result["barycenter"], dtype=float)
    moment = np.asarray(result["moment"], dtype=float)
    eigenvalues = np.asarray(result["eigenvalues"], dtype=float)
    direction = np.asarray(result["principal_direction"], dtype=float)
    return {
        f"{prefix}_barycenter_x": float(barycenter[0]),
        f"{prefix}_barycenter_y": float(barycenter[1]),
        f"{prefix}_moment_xx": float(moment[0, 0]),
        f"{prefix}_moment_yy": float(moment[1, 1]),
        f"{prefix}_moment_xy": float(moment[0, 1]),
        f"{prefix}_moment_trace": float(result["trace"]),
        f"{prefix}_moment_determinant": float(result["determinant"]),
        f"{prefix}_moment_eigenvalue_major": float(eigenvalues[0]),
        f"{prefix}_moment_eigenvalue_minor": float(eigenvalues[1]),
        f"{prefix}_principal_direction_x": float(direction[0]),
        f"{prefix}_principal_direction_y": float(direction[1]),
        f"{prefix}_principal_direction_angle_degrees": float(
            result["principal_angle_degrees"]
        ),
        f"{prefix}_principal_std": float(result["principal_std"]),
        f"{prefix}_secondary_std": float(result["secondary_std"]),
        f"{prefix}_spatial_anisotropy_ratio": float(result["anisotropy_ratio"]),
        f"{prefix}_moment_status": str(result["status"]),
    }


def compute_signed_field_metrics(
    values: np.ndarray,
    centers: np.ndarray,
    cell_measures: np.ndarray,
    *,
    zero_tolerance: float = 1.0e-14,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    coordinates = np.asarray(centers, dtype=np.float64)
    measures = np.asarray(cell_measures, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if coordinates.shape != (array.size, 2):
        raise ValueError("centers must have shape (num_values, 2).")
    if measures.shape != array.shape:
        raise ValueError("cell_measures must have the same shape as values.")
    if array.size == 0:
        raise ValueError("At least one cell value is required.")
    if not np.all(np.isfinite(array)):
        raise ValueError("values contain NaN or Inf.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("centers contain NaN or Inf.")
    if not np.all(np.isfinite(measures)) or np.any(measures <= 0.0):
        raise ValueError("cell_measures must be finite and strictly positive.")
    zero_tolerance = float(zero_tolerance)
    if not math.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and non-negative.")

    positive, negative = split_signed_field(array)
    absolute = np.abs(array)
    domain_measure = float(np.sum(measures, dtype=np.float64))
    mass_tolerance = zero_tolerance * domain_measure
    signed_integral = float(np.sum(array * measures, dtype=np.float64))
    positive_integral = float(np.sum(positive * measures, dtype=np.float64))
    negative_integral = float(np.sum(negative * measures, dtype=np.float64))
    absolute_integral = positive_integral + negative_integral
    max_abs = float(np.max(absolute))

    positive_mask = array > zero_tolerance
    negative_mask = array < -zero_tolerance
    zero_mask = ~(positive_mask | negative_mask)
    num_values = int(array.size)
    num_positive = int(np.count_nonzero(positive_mask))
    num_negative = int(np.count_nonzero(negative_mask))
    num_zero = num_values - num_positive - num_negative

    if absolute_integral > mass_tolerance:
        positive_fraction = positive_integral / absolute_integral
        negative_fraction = negative_integral / absolute_integral
        balance = (positive_integral - negative_integral) / absolute_integral
    else:
        positive_fraction = math.nan
        negative_fraction = math.nan
        balance = math.nan

    moment_results = {
        "positive": compute_centered_second_moment(
            coordinates,
            positive,
            measures,
            mass_tolerance=mass_tolerance,
        ),
        "negative": compute_centered_second_moment(
            coordinates,
            negative,
            measures,
            mass_tolerance=mass_tolerance,
        ),
        "absolute": compute_centered_second_moment(
            coordinates,
            absolute,
            measures,
            mass_tolerance=mass_tolerance,
        ),
    }

    positive_b = np.asarray(moment_results["positive"]["barycenter"], dtype=float)
    negative_b = np.asarray(moment_results["negative"]["barycenter"], dtype=float)
    if np.all(np.isfinite(positive_b)) and np.all(np.isfinite(negative_b)):
        distance = float(np.linalg.norm(positive_b - negative_b))
    else:
        distance = math.nan

    status = (
        "ZERO_FIELD"
        if max_abs <= zero_tolerance or absolute_integral <= mass_tolerance
        else "OK"
    )
    result: dict[str, Any] = {
        "metric_status": status,
        "num_values": num_values,
        "num_positive": num_positive,
        "num_negative": num_negative,
        "num_zero": num_zero,
        "fraction_positive_values": num_positive / num_values,
        "fraction_negative_values": num_negative / num_values,
        "fraction_zero_values": num_zero / num_values,
        "domain_measure": domain_measure,
        "cell_measure_min": float(np.min(measures)),
        "cell_measure_max": float(np.max(measures)),
        "min_value": float(np.min(array)),
        "max_value": float(np.max(array)),
        "max_abs_value": max_abs,
        "signed_integral": signed_integral,
        "positive_integral": positive_integral,
        "negative_magnitude_integral": negative_integral,
        "absolute_integral": absolute_integral,
        "positive_mass_fraction": positive_fraction,
        "negative_mass_fraction": negative_fraction,
        "signed_balance_index": balance,
        "positive_negative_barycenter_distance": distance,
    }
    for prefix, moment_result in moment_results.items():
        result.update(_moment_columns(prefix, moment_result))
    return result


def reconstruct_article_safe(
    signed_values: np.ndarray,
    *,
    zero_tolerance: float = 1.0e-14,
) -> np.ndarray:
    values = np.asarray(signed_values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("signed_values contain NaN or Inf.")
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    if max_abs <= float(zero_tolerance):
        return np.zeros_like(values, dtype=np.float64)
    return np.abs(values) / max_abs


def compare_article_safe(
    reconstructed: np.ndarray,
    reference: np.ndarray,
    cell_measures: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    left = np.asarray(reconstructed, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    measures = np.asarray(cell_measures, dtype=np.float64)
    if left.shape != right.shape or left.shape != measures.shape:
        raise ValueError("reconstructed, reference, and cell_measures shapes differ.")
    finite = bool(np.all(np.isfinite(right)))
    if not finite:
        return {
            "article_safe_values_finite": False,
            "article_safe_max_abs_error": math.nan,
            "article_safe_l2_error": math.nan,
            "article_safe_relative_l2_error": math.nan,
            "article_safe_allclose": False,
        }
    error = left - right
    max_abs_error = float(np.max(np.abs(error))) if error.size else 0.0
    l2_error = math.sqrt(float(np.sum(error * error * measures, dtype=np.float64)))
    reference_l2 = math.sqrt(
        float(np.sum(right * right * measures, dtype=np.float64))
    )
    relative_l2 = l2_error / reference_l2 if reference_l2 > 0.0 else math.nan
    return {
        "article_safe_values_finite": True,
        "article_safe_max_abs_error": max_abs_error,
        "article_safe_l2_error": l2_error,
        "article_safe_relative_l2_error": relative_l2,
        "article_safe_allclose": bool(
            np.allclose(left, right, rtol=float(rtol), atol=float(atol))
        ),
    }


def load_vtk_cell_field(
    path: str | Path,
    field_name: str,
) -> LoadedCellField:
    requested = Path(path).expanduser().resolve()
    dataset = resolve_vtk_dataset(requested, time_index=0)
    center_blocks: list[np.ndarray] = []
    value_blocks: list[np.ndarray] = []
    measure_blocks: list[np.ndarray] = []
    cell_types: list[str] = []
    selected: str | None = None
    num_ghost = 0
    for chunk in iter_vtk_cell_field_chunks(
        requested,
        (field_name,),
        time_index=0,
        remove_ghost_cells=True,
        require_finite=False,
    ):
        selected = chunk.field_name if selected is None else selected
        center_blocks.append(np.asarray(chunk.centers, dtype=np.float64))
        value_blocks.append(np.asarray(chunk.values, dtype=np.float64))
        measure_blocks.append(np.asarray(chunk.cell_measures, dtype=np.float64))
        cell_types.append(chunk.cell_type)
        num_ghost += int(chunk.num_ghost_cells_removed)
    if selected is None or not value_blocks:
        raise VTKCellFieldError(f"No cell values found in {requested}.")
    return LoadedCellField(
        requested_path=requested,
        selected_field_name=selected,
        centers=np.ascontiguousarray(np.concatenate(center_blocks, axis=0)),
        values=np.ascontiguousarray(np.concatenate(value_blocks, axis=0)),
        cell_measures=np.ascontiguousarray(np.concatenate(measure_blocks, axis=0)),
        cell_types=tuple(cell_types),
        piece_paths=dataset.piece_paths,
        num_ghost_cells_removed=num_ghost,
    )


def load_defect_circles(
    item: DiscoveredSignedField,
) -> tuple[DefectCircle, ...]:
    circles: list[DefectCircle] = []
    if item.defect_config_path is not None and item.defect_config_path.exists():
        payload = _read_json(item.defect_config_path)
        defects = payload.get("defects", [])
        if isinstance(defects, Sequence):
            for index, defect in enumerate(defects):
                if not isinstance(defect, Mapping):
                    continue
                try:
                    circles.append(
                        DefectCircle(
                            center_x=float(defect["center_x"]),
                            center_y=float(defect["center_y"]),
                            radius=float(defect["radius"]),
                            name=str(defect.get("name", f"defect_{index + 1}")),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    config = item.configuration
    if not circles and all(
        key in config for key in ("defect_center_x", "defect_center_y", "defect_radius")
    ):
        try:
            circles.append(
                DefectCircle(
                    center_x=float(config["defect_center_x"]),
                    center_y=float(config["defect_center_y"]),
                    radius=float(config["defect_radius"]),
                    name="defect",
                )
            )
        except (TypeError, ValueError):
            pass
    return tuple(circles)


def plot_cell_field(
    *,
    centers: np.ndarray,
    values: np.ndarray,
    output_png: Path,
    output_pdf: Path,
    title: str,
    signed: bool,
    defects: Sequence[DefectCircle] = (),
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    coordinates = np.asarray(centers, dtype=np.float64)
    field_values = np.asarray(values, dtype=np.float64)
    if coordinates.shape != (field_values.size, 2):
        raise ValueError("Plot coordinates and values are incompatible.")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    marker_size = max(0.05, min(12.0, 180000.0 / max(field_values.size, 1)))
    if signed:
        amplitude = float(np.max(np.abs(field_values))) if field_values.size else 0.0
        color_amplitude = amplitude if amplitude > 0.0 else 1.0
        artist = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=field_values,
            s=marker_size,
            marker="s",
            linewidths=0.0,
            cmap="RdBu_r",
            vmin=-color_amplitude,
            vmax=color_amplitude,
            rasterized=True,
        )
        colorbar_label = "Signed sensitivity density"
    else:
        artist = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=field_values,
            s=marker_size,
            marker="s",
            linewidths=0.0,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        colorbar_label = r"$|g| / \max |g|$"

    for defect in defects:
        ax.add_patch(
            Circle(
                (defect.center_x, defect.center_y),
                defect.radius,
                fill=False,
                edgecolor="black",
                linewidth=1.0,
                linestyle="--",
            )
        )

    colorbar = fig.colorbar(artist, ax=ax)
    colorbar.set_label(colorbar_label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.01)
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_pdf)
    plt.close(fig)


def _select_article_safe_candidate(
    candidates: Sequence[Path],
) -> tuple[Path | None, str]:
    unique = tuple(dict.fromkeys(Path(path).resolve() for path in candidates))
    existing = tuple(path for path in unique if path.exists())
    if len(existing) > 1:
        return None, "AMBIGUOUS_MATCH"
    if len(existing) == 1:
        return existing[0], "PRESENT"
    if unique:
        return unique[0], "ARTICLE_SAFE_MISSING"
    return None, "ARTICLE_SAFE_MISSING"


def _empty_article_metrics() -> dict[str, Any]:
    return {
        "article_safe_shape_compatible": False,
        "article_safe_mesh_compatible": False,
        "article_safe_values_finite": False,
        "article_safe_max_abs_error": math.nan,
        "article_safe_l2_error": math.nan,
        "article_safe_relative_l2_error": math.nan,
        "article_safe_allclose": False,
    }


def _overall_status(metric_status: str, article_status: str) -> str:
    if metric_status in {"NONFINITE_VALUES", "ERROR"}:
        return metric_status
    if article_status in {
        "AMBIGUOUS_MATCH",
        "INCOMPATIBLE_SHAPE",
        "INCOMPATIBLE_MESH",
        "NONFINITE_VALUES",
        "ARTICLE_SAFE_MISMATCH",
        "ARTICLE_SAFE_MISSING",
        "ERROR",
    }:
        return (
            "ARTICLE_SAFE_MISMATCH"
            if article_status == "INCOMPATIBLE_MESH"
            else article_status
        )
    if metric_status == "ZERO_FIELD":
        return "ZERO_FIELD"
    return "OK"


def analyze_discovered_field(
    item: DiscoveredSignedField,
    *,
    config: AnalysisConfig,
    analysis_date_utc: str,
) -> dict[str, Any]:
    field_name = (
        f"residual_interaction_{item.parameter_name}"
        if item.observation_mode == "residual_interaction"
        else f"gradient_{item.parameter_name}"
    )
    article_path, initial_article_status = _select_article_safe_candidate(
        item.article_safe_candidates
    )
    row: dict[str, Any] = {
        "analysis_date_utc": analysis_date_utc,
        "format_version": FORMAT_VERSION,
        "geometry_id": item.geometry_id,
        "campaign_id": item.campaign_id,
        "field_id": item.field_id,
        "parameter_name": item.parameter_name,
        "observation_mode": item.observation_mode,
        "input_path": str(item.input_path),
        "article_safe_path": "" if article_path is None else str(article_path),
        "manifest_path": "" if item.manifest_path is None else str(item.manifest_path),
        "source_format": "pvd_pvtu_vtu_dg0",
        "integration_method": INTEGRATION_METHOD,
        "barycenter_method": BARYCENTER_METHOD,
        "moment_method": MOMENT_METHOD,
        "configuration": _json_compact(item.configuration),
        "input_sha256": _sha256_file(item.input_path),
        "article_safe_sha256": _sha256_file(article_path),
        "message": "",
        **_empty_article_metrics(),
    }

    figures_dir = config.output_dir / "figures"
    reconstructed_dir = config.output_dir / "reconstructed_article_safe"
    signed_png = figures_dir / f"{item.field_id}__signed.png"
    signed_pdf = figures_dir / f"{item.field_id}__signed.pdf"
    reconstructed_png = (
        reconstructed_dir / f"{item.field_id}__reconstructed_article_safe.png"
    )
    reconstructed_pdf = (
        reconstructed_dir / f"{item.field_id}__reconstructed_article_safe.pdf"
    )
    row.update(
        {
            "signed_figure_png": str(signed_png),
            "signed_figure_pdf": str(signed_pdf),
            "reconstructed_article_safe_png": str(reconstructed_png),
            "reconstructed_article_safe_pdf": str(reconstructed_pdf),
        }
    )

    try:
        raw = load_vtk_cell_field(item.input_path, field_name)
    except FileNotFoundError as exc:
        row.update(
            {
                "metric_status": "ERROR",
                "article_safe_status": initial_article_status,
                "status": "ERROR",
                "message": str(exc),
            }
        )
        return row
    except Exception as exc:
        row.update(
            {
                "metric_status": "ERROR",
                "article_safe_status": initial_article_status,
                "status": "ERROR",
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
        return row

    row["num_ghost_cells_removed"] = raw.num_ghost_cells_removed
    if not np.all(np.isfinite(raw.values)):
        row.update(
            {
                "metric_status": "NONFINITE_VALUES",
                "article_safe_status": initial_article_status,
                "status": "NONFINITE_VALUES",
                "message": "Signed field contains NaN or Inf.",
            }
        )
        return row

    try:
        metrics = compute_signed_field_metrics(
            raw.values,
            raw.centers,
            raw.cell_measures,
            zero_tolerance=config.zero_tolerance,
        )
        row.update(metrics)
        reconstructed = reconstruct_article_safe(
            raw.values,
            zero_tolerance=config.zero_tolerance,
        )
        defects = load_defect_circles(item)
        plot_cell_field(
            centers=raw.centers,
            values=raw.values,
            output_png=signed_png,
            output_pdf=signed_pdf,
            title=(
                f"{item.geometry_id} | {item.observation_mode} | "
                f"{item.parameter_name} | signed"
            ),
            signed=True,
            defects=defects,
        )
        plot_cell_field(
            centers=raw.centers,
            values=reconstructed,
            output_png=reconstructed_png,
            output_pdf=reconstructed_pdf,
            title=(
                f"{item.geometry_id} | {item.observation_mode} | "
                f"{item.parameter_name} | reconstructed article_safe"
            ),
            signed=False,
            defects=defects,
        )
    except Exception as exc:
        row.update(
            {
                "metric_status": "ERROR",
                "article_safe_status": initial_article_status,
                "status": "ERROR",
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
        return row

    article_status = initial_article_status
    if initial_article_status == "PRESENT" and article_path is not None:
        try:
            article = load_vtk_cell_field(article_path, field_name)
            shape_compatible = article.values.shape == raw.values.shape
            row["article_safe_shape_compatible"] = shape_compatible
            if not shape_compatible:
                article_status = "INCOMPATIBLE_SHAPE"
            else:
                center_compatible = bool(
                    np.allclose(
                        article.centers,
                        raw.centers,
                        rtol=0.0,
                        atol=max(config.atol, 1.0e-12),
                    )
                )
                measure_compatible = bool(
                    np.allclose(
                        article.cell_measures,
                        raw.cell_measures,
                        rtol=config.rtol,
                        atol=config.atol,
                    )
                )
                type_compatible = article.cell_types == raw.cell_types
                mesh_compatible = center_compatible and measure_compatible and type_compatible
                row["article_safe_mesh_compatible"] = mesh_compatible
                if not mesh_compatible:
                    article_status = "INCOMPATIBLE_MESH"
                elif not np.all(np.isfinite(article.values)):
                    article_status = "NONFINITE_VALUES"
                    row["article_safe_values_finite"] = False
                else:
                    comparison = compare_article_safe(
                        reconstructed,
                        article.values,
                        raw.cell_measures,
                        rtol=config.rtol,
                        atol=config.atol,
                    )
                    row.update(comparison)
                    article_status = (
                        "OK"
                        if comparison["article_safe_allclose"]
                        else "ARTICLE_SAFE_MISMATCH"
                    )
        except FileNotFoundError:
            article_status = "ARTICLE_SAFE_MISSING"
        except Exception as exc:
            article_status = "ERROR"
            row["message"] = f"article_safe: {type(exc).__name__}: {exc}"

    row["article_safe_status"] = article_status
    row["status"] = _overall_status(str(row["metric_status"]), article_status)
    if not row["message"] and row["status"] != "OK":
        row["message"] = row["status"]
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in CSV_COLUMNS})


def _manifest_field_record(item: DiscoveredSignedField) -> dict[str, Any]:
    return {
        "geometry_id": item.geometry_id,
        "campaign_id": item.campaign_id,
        "field_id": item.field_id,
        "parameter_name": item.parameter_name,
        "observation_mode": item.observation_mode,
        "input_path": str(item.input_path),
        "article_safe_candidates": [str(path) for path in item.article_safe_candidates],
        "manifest_path": None if item.manifest_path is None else str(item.manifest_path),
        "defect_config_path": (
            None if item.defect_config_path is None else str(item.defect_config_path)
        ),
    }


def _validate_output_directory(config: AnalysisConfig) -> None:
    output_dir = config.output_dir
    protected = (
        output_dir / "signed_sensitivity_metrics.csv",
        output_dir / "signed_sensitivity_manifest.json",
    )
    if not config.overwrite and any(path.exists() for path in protected):
        raise FileExistsError(
            f"Signed-sensitivity outputs already exist in {output_dir}; "
            "use --overwrite to replace them."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "reconstructed_article_safe").mkdir(parents=True, exist_ok=True)



def _resolve_analysis_mode(
    requested: str,
    geometry_ids: Sequence[str],
) -> str:
    allowed = {"auto", "line-only", "line-box"}
    if requested not in allowed:
        raise ValueError(
            "analysis_mode must be one of: auto, line-only, line-box."
        )
    geometries = {str(value).strip().lower() for value in geometry_ids}
    if requested == "auto":
        if geometries == {"line"}:
            return "line-only"
        if geometries == {"box"}:
            return "box-only"
        if geometries == {"line", "box"}:
            return "line-box"
        return "multi-geometry"
    if requested == "line-only":
        if geometries != {"line"}:
            raise ValueError(
                "line-only analysis requires exclusively line fields; "
                f"discovered geometries={sorted(geometries)}."
            )
        return "line-only"
    if not {"line", "box"}.issubset(geometries):
        raise ValueError(
            "line-box analysis requires both line and box fields; "
            f"discovered geometries={sorted(geometries)}."
        )
    unknown = geometries.difference({"line", "box"})
    if unknown:
        raise ValueError(
            "line-box analysis received unsupported additional geometries: "
            f"{sorted(unknown)}."
        )
    return "line-box"


def run_signed_sensitivity_analysis(config: AnalysisConfig) -> AnalysisResult:
    config = AnalysisConfig(
        **{
            **asdict(config),
            "input_dir": Path(config.input_dir).expanduser().resolve(),
            "output_dir": Path(config.output_dir).expanduser().resolve(),
            "article_safe_dir": (
                None
                if config.article_safe_dir is None
                else Path(config.article_safe_dir).expanduser().resolve()
            ),
            "manifest": (
                None
                if config.manifest is None
                else Path(config.manifest).expanduser().resolve()
            ),
        }
    )
    if config.analysis_mode not in {"auto", "line-only", "line-box"}:
        raise ValueError(
            "analysis_mode must be one of: auto, line-only, line-box."
        )
    if config.rtol < 0.0 or config.atol < 0.0 or config.zero_tolerance < 0.0:
        raise ValueError("rtol, atol, and zero_tolerance must be non-negative.")
    _validate_output_directory(config)
    discovery = discover_signed_fields(config)
    if not discovery.fields:
        raise RuntimeError(
            f"No signed sensitivity PVD field was discovered under {config.input_dir}."
        )
    geometry_ids = sorted({item.geometry_id for item in discovery.fields})
    analysis_mode = _resolve_analysis_mode(config.analysis_mode, geometry_ids)

    analysis_date = _now_utc()
    rows: list[dict[str, Any]] = []
    for item in discovery.fields:
        if not config.quiet:
            print(
                f"[SIGNED] {item.geometry_id} / {item.observation_mode} / "
                f"{item.parameter_name}",
                flush=True,
            )
        row = analyze_discovered_field(
            item,
            config=config,
            analysis_date_utc=analysis_date,
        )
        row["analysis_mode"] = analysis_mode
        rows.append(row)

    csv_path = config.output_dir / "signed_sensitivity_metrics.csv"
    manifest_path = config.output_dir / "signed_sensitivity_manifest.json"
    _write_csv(csv_path, rows)

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "ERROR"))
        status_counts[status] = status_counts.get(status, 0) + 1
    successful_statuses = {"OK", "ZERO_FIELD"}
    exit_code = 0 if all(str(row.get("status")) in successful_statuses for row in rows) else 1
    allclose_count = sum(bool(row.get("article_safe_allclose")) for row in rows)
    processed = [
        {
            "field_id": row.get("field_id"),
            "input_path": row.get("input_path"),
            "article_safe_path": row.get("article_safe_path"),
            "status": row.get("status"),
            "metric_status": row.get("metric_status"),
            "article_safe_status": row.get("article_safe_status"),
            "article_safe_allclose": row.get("article_safe_allclose"),
            "message": row.get("message"),
        }
        for row in rows
    ]
    manifest = {
        "kind": "signed_sensitivity_analysis",
        "format_version": FORMAT_VERSION,
        "analysis_date_utc": analysis_date,
        "status": "ok" if exit_code == 0 else "validation_failed",
        "analysis_mode": analysis_mode,
        "geometry_ids": geometry_ids,
        "arguments": {
            "input_dir": str(config.input_dir),
            "output_dir": str(config.output_dir),
            "analysis_mode": config.analysis_mode,
            "geometry_patterns": list(config.geometry_patterns),
            "field_patterns": list(config.field_patterns),
            "article_safe_dir": (
                None if config.article_safe_dir is None else str(config.article_safe_dir)
            ),
            "manifest": None if config.manifest is None else str(config.manifest),
            "rtol": config.rtol,
            "atol": config.atol,
            "zero_tolerance": config.zero_tolerance,
            "overwrite": config.overwrite,
            "quiet": config.quiet,
        },
        "manifests_inspected": [str(path) for path in discovery.manifests],
        "files_discovered": [
            _manifest_field_record(item) for item in discovery.fields
        ],
        "files_processed": processed,
        "files_ignored": list(discovery.ignored),
        "tolerances": {
            "rtol": config.rtol,
            "atol": config.atol,
            "zero_tolerance": config.zero_tolerance,
            "barycenter_mass_tolerance": "zero_tolerance * domain_measure",
        },
        "mathematical_convention": {
            "positive_part": "g_plus=max(g,0)",
            "negative_magnitude": "g_minus=max(-g,0)",
            "signed_field": "g=g_plus-g_minus",
            "absolute_value": "abs(g)=g_plus+g_minus",
            "signed_balance_index": "(M_plus-M_minus)/(M_plus+M_minus)",
            "article_safe_reconstruction": "abs(g)/max(abs(g)); zero field -> zero map",
            "spatial_anisotropy_ratio": (
                "sqrt(lambda_major/lambda_minor); 1 for zero spread and inf for "
                "a rank-one moment"
            ),
            "value_counts": (
                "positive if g>zero_tolerance, negative if g<-zero_tolerance; "
                "integrals use the unthresholded signed field"
            ),
        },
        "integration_method": INTEGRATION_METHOD,
        "barycenter_method": BARYCENTER_METHOD,
        "moment_method": MOMENT_METHOD,
        "field_representation": "DG0 cell data from PVD/PVTU/VTU",
        "association_method": (
            "manifest raw_signed_validation/validation pairing, then mirrored "
            "gradient_combinations_raw_signed -> gradient_combinations fallback"
        ),
        "warnings": list(discovery.warnings),
        "errors": [
            {
                "field_id": row.get("field_id"),
                "status": row.get("status"),
                "message": row.get("message"),
            }
            for row in rows
            if str(row.get("status")) not in successful_statuses
        ],
        "summary": {
            "analysis_mode": analysis_mode,
            "geometry_ids": geometry_ids,
            "num_discovered": len(discovery.fields),
            "num_processed": len(rows),
            "status_counts": status_counts,
            "num_article_safe_allclose": allclose_count,
            "num_article_safe_not_allclose": len(rows) - allclose_count,
            "csv_path": str(csv_path),
            "figures_dir": str(config.output_dir / "figures"),
            "reconstructed_article_safe_dir": str(
                config.output_dir / "reconstructed_article_safe"
            ),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    if not config.quiet:
        print(
            f"[SIGNED] processed={len(rows)} allclose={allclose_count} "
            f"status={manifest['status']}",
            flush=True,
        )
    return AnalysisResult(
        rows=tuple(rows),
        manifest=manifest,
        csv_path=csv_path,
        manifest_path=manifest_path,
        exit_code=exit_code,
    )
