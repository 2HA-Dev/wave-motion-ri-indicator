from __future__ import annotations

"""Receiver-aperture analysis for validated single-defect campaigns.

The module supports a line-only mode and a line/box comparison. It is deliberately independent of FEniCSx, PETSc, and MPI. It
reads long-format residual CSV files, obtains receiver centres from validated
vector-array manifests, validates physical time against the integer step index,
and computes receiver-, side-, and aperture-level metrics.

The original residual CSV files are never modified.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REQUIRED_RESIDUAL_COLUMNS = (
    "time",
    "step",
    "array",
    "role",
    "receiver_id",
    "component",
    "residual",
    "reference",
    "observed",
)

ALLOWED_GEOMETRIES = ("line", "box")
ALLOWED_ANALYSIS_GEOMETRIES = ("line", "line-box")
ALLOWED_TIME_POLICIES = ("strict-physical", "legacy-step-index")
ALLOWED_CORNER_POLICIES = ("shared",)
EXPECTED_COMPONENTS = ("x", "y")
EXPECTED_RECEIVER_COUNTS = {
    ("line", "reflection"): 21,
    ("line", "transmission"): 21,
    ("box", "reflection"): 21,
    ("box", "transmission"): 61,
}

SCHEMA_VERSION = "1.1.0"


class ReceiverApertureError(RuntimeError):
    """Base class for contextual receiver-aperture errors."""


class ResidualCSVError(ReceiverApertureError):
    """Raised when a residual CSV violates the canonical contract."""


class TimeConventionError(ReceiverApertureError):
    """Raised when the original time column is incompatible with the policy."""


class ReceiverGeometryError(ReceiverApertureError):
    """Raised when receiver centres or aperture geometry are inconsistent."""


class ResidualConventionError(ReceiverApertureError):
    """Raised when the residual sign convention cannot be validated."""


@dataclass(frozen=True)
class ApertureComparisonInputs:
    line_residuals: Path
    box_residuals: Path | None = None
    line_manifest: Path | None = None
    box_manifest: Path | None = None

    def resolved(self) -> "ApertureComparisonInputs":
        def _resolved(path: Path | None) -> Path | None:
            return None if path is None else Path(path).expanduser().resolve()

        return ApertureComparisonInputs(
            line_residuals=Path(self.line_residuals).expanduser().resolve(),
            box_residuals=_resolved(self.box_residuals),
            line_manifest=_resolved(self.line_manifest),
            box_manifest=_resolved(self.box_manifest),
        )


@dataclass(frozen=True)
class ApertureComparisonConfig:
    inputs: ApertureComparisonInputs
    output_dir: Path
    dt: float = 0.05
    time_atol: float = 1.0e-12
    time_rtol: float = 1.0e-12
    value_atol: float = 1.0e-14
    value_rtol: float = 1.0e-12
    geometry_atol: float = 1.0e-12
    ratio_epsilon: float = 1.0e-30
    time_column_policy: str = "strict-physical"
    corner_policy: str = "shared"
    overwrite: bool = False
    quiet: bool = False
    geometry: str = "line-box"

    def __post_init__(self) -> None:
        if self.geometry not in ALLOWED_ANALYSIS_GEOMETRIES:
            raise ValueError(
                "geometry must be one of "
                f"{ALLOWED_ANALYSIS_GEOMETRIES}, got {self.geometry!r}."
            )
        if self.inputs.line_manifest is None:
            raise ValueError("line_manifest is required.")
        if self.geometry == "line-box":
            if self.inputs.box_residuals is None or self.inputs.box_manifest is None:
                raise ValueError(
                    "box_residuals and box_manifest are required in line-box mode."
                )
        elif self.inputs.box_residuals is not None or self.inputs.box_manifest is not None:
            raise ValueError(
                "box_residuals and box_manifest must be omitted in line-only mode."
            )
        if not math.isfinite(float(self.dt)) or float(self.dt) <= 0.0:
            raise ValueError("dt must be finite and strictly positive.")
        for name in (
            "time_atol",
            "time_rtol",
            "value_atol",
            "value_rtol",
            "geometry_atol",
            "ratio_epsilon",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.time_column_policy not in ALLOWED_TIME_POLICIES:
            raise ValueError(
                "time_column_policy must be one of "
                f"{ALLOWED_TIME_POLICIES}, got {self.time_column_policy!r}."
            )
        if self.corner_policy not in ALLOWED_CORNER_POLICIES:
            raise ValueError(
                f"corner_policy must be one of {ALLOWED_CORNER_POLICIES}."
            )


@dataclass(frozen=True)
class ReceiverSpec:
    geometry: str
    residual_family: str
    array_name: str
    role: str
    centers: np.ndarray
    manifest_geometry: str
    spacing: float | None
    sigma: float | None
    source_manifest: Path

    @property
    def num_receivers(self) -> int:
        return int(self.centers.shape[0])


@dataclass(frozen=True)
class CampaignInspection:
    geometry: str
    path: Path
    file_size_bytes: int
    sha256: str
    num_rows: int
    columns: tuple[str, ...]
    dtypes: Mapping[str, str]
    arrays: tuple[str, ...]
    roles: tuple[str, ...]
    components: tuple[str, ...]
    num_steps: int
    step_min: int
    step_max: int
    time_original_min: float
    time_original_max: float
    time_status: str
    residual_convention: str
    duplicate_count: int
    missing_value_count: int
    nonfinite_count: int
    receiver_counts: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "geometry": self.geometry,
            "path": str(self.path),
            "file_size_bytes": int(self.file_size_bytes),
            "sha256": self.sha256,
            "num_rows": int(self.num_rows),
            "columns": list(self.columns),
            "dtypes": dict(self.dtypes),
            "arrays": list(self.arrays),
            "roles": list(self.roles),
            "components": list(self.components),
            "num_steps": int(self.num_steps),
            "step_min": int(self.step_min),
            "step_max": int(self.step_max),
            "time_original_min": float(self.time_original_min),
            "time_original_max": float(self.time_original_max),
            "time_status": self.time_status,
            "residual_convention": self.residual_convention,
            "duplicate_count": int(self.duplicate_count),
            "missing_value_count": int(self.missing_value_count),
            "nonfinite_count": int(self.nonfinite_count),
            "receiver_counts": dict(self.receiver_counts),
        }


@dataclass(frozen=True)
class ApertureComparisonResult:
    inspections: tuple[CampaignInspection, ...]
    canonical_residuals: pd.DataFrame
    receiver_geometry: pd.DataFrame
    rms_by_component: pd.DataFrame
    rms_vector: pd.DataFrame
    rms_by_side: pd.DataFrame
    line_box_summary: pd.DataFrame
    time_validation: pd.DataFrame
    common_aperture_consistency: pd.DataFrame
    representative_signals: pd.DataFrame
    summary: dict[str, Any]
    elapsed_seconds: float


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    csv_dir: Path
    figures_dir: Path
    tables_dir: Path
    manifests_dir: Path
    logs_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "OutputPaths":
        root = Path(root).expanduser().resolve()
        return cls(
            root=root,
            csv_dir=root / "csv",
            figures_dir=root / "figures",
            tables_dir=root / "tables",
            manifests_dir=root / "manifests",
            logs_dir=root / "logs",
        )

    def create(self, *, overwrite: bool) -> None:
        if self.root.exists():
            existing = [item for item in self.root.iterdir()]
            if existing and not overwrite:
                raise FileExistsError(
                    f"Output directory is not empty: {self.root}. "
                    "Use --overwrite to replace generated files."
                )
            if overwrite:
                for item in existing:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
        for directory in (
            self.root,
            self.csv_dir,
            self.figures_dir,
            self.tables_dir,
            self.manifests_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _require_file(path: Path, *, role: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing {role}: {path}")
    if not path.is_file():
        raise ValueError(f"{role} must be a file: {path}")
    return path


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(path),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReceiverGeometryError(f"Invalid JSON manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReceiverGeometryError(f"Manifest root must be an object: {path}")
    return payload


def _nested(payload: Mapping[str, Any], path: Sequence[str], *, context: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            joined = ".".join(path)
            raise ReceiverGeometryError(
                f"Missing manifest field {joined!r} in {context}."
            )
        current = current[key]
    return current


def _family_from_role_array(role: str, array_name: str) -> str:
    role = str(role)
    array_name = str(array_name)
    if role == "source_reflection" and array_name == "gamma1":
        return "reflection"
    if role == "transmission" and array_name == "gamma2":
        return "transmission"
    raise ResidualCSVError(
        "Unsupported array/role pair: "
        f"array={array_name!r}, role={role!r}; expected "
        "gamma1/source_reflection or gamma2/transmission."
    )


def _extract_receiver_spec(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    geometry: str,
    residual_family: str,
) -> ReceiverSpec:
    if residual_family == "reflection":
        key = "observed_reflection"
        simulated_key = "simulated_reflection"
    elif residual_family == "transmission":
        key = "observed_transmission"
        simulated_key = "simulated_transmission"
    else:
        raise ValueError(f"Unsupported residual family: {residual_family}")

    base = ["extra", "comparison"]
    observed_spec = _nested(
        manifest,
        [*base, key, "spec", "array"],
        context=str(manifest_path),
    )
    simulated_spec = _nested(
        manifest,
        [*base, simulated_key, "spec", "array"],
        context=str(manifest_path),
    )
    if not isinstance(observed_spec, Mapping) or not isinstance(simulated_spec, Mapping):
        raise ReceiverGeometryError(
            f"Receiver array specifications must be objects in {manifest_path}."
        )

    required = ("centers", "name", "role", "geometry", "num_elements")
    missing = [name for name in required if name not in observed_spec]
    if missing:
        raise ReceiverGeometryError(
            f"Missing receiver fields {missing} in {manifest_path}, family={residual_family}."
        )

    centers = np.asarray(observed_spec["centers"], dtype=np.float64)
    simulated_centers = np.asarray(simulated_spec.get("centers"), dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ReceiverGeometryError(
            f"Receiver centres must have shape (N, 2) in {manifest_path}, "
            f"family={residual_family}; got {centers.shape}."
        )
    if not np.all(np.isfinite(centers)):
        raise ReceiverGeometryError(
            f"Non-finite receiver centres in {manifest_path}, family={residual_family}."
        )
    if int(observed_spec["num_elements"]) != int(centers.shape[0]):
        raise ReceiverGeometryError(
            f"num_elements disagrees with centres in {manifest_path}, "
            f"family={residual_family}."
        )
    if simulated_centers.shape != centers.shape or not np.allclose(
        simulated_centers,
        centers,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise ReceiverGeometryError(
            f"Observed and simulated receiver centres differ in {manifest_path}, "
            f"family={residual_family}."
        )

    unique = np.unique(np.round(centers, decimals=12), axis=0)
    if unique.shape[0] != centers.shape[0]:
        raise ReceiverGeometryError(
            f"Duplicate receiver centres in {manifest_path}, family={residual_family}."
        )

    array_name = str(observed_spec["name"])
    role = str(observed_spec["role"])
    inferred = _family_from_role_array(role, array_name)
    if inferred != residual_family:
        raise ReceiverGeometryError(
            f"Manifest family mismatch in {manifest_path}: expected {residual_family}, "
            f"inferred {inferred}."
        )

    expected_count = EXPECTED_RECEIVER_COUNTS[(geometry, residual_family)]
    if centers.shape[0] != expected_count:
        raise ReceiverGeometryError(
            f"Unexpected receiver count for {geometry}/{residual_family}: "
            f"expected {expected_count}, found {centers.shape[0]} in {manifest_path}."
        )

    spacing_raw = observed_spec.get("spacing")
    sigma_raw = observed_spec.get("sigma")
    spacing = None if spacing_raw is None else float(spacing_raw)
    sigma = None if sigma_raw is None else float(sigma_raw)
    return ReceiverSpec(
        geometry=geometry,
        residual_family=residual_family,
        array_name=array_name,
        role=role,
        centers=np.ascontiguousarray(centers, dtype=np.float64),
        manifest_geometry=str(observed_spec["geometry"]),
        spacing=spacing,
        sigma=sigma,
        source_manifest=manifest_path,
    )


def load_receiver_specs(
    manifest_path: str | Path,
    *,
    geometry: str,
) -> dict[str, ReceiverSpec]:
    if geometry not in ALLOWED_GEOMETRIES:
        raise ValueError(f"Unsupported geometry: {geometry!r}")
    path = _require_file(Path(manifest_path), role=f"{geometry} vector-array manifest")
    payload = _load_json(path)
    return {
        family: _extract_receiver_spec(
            payload,
            manifest_path=path,
            geometry=geometry,
            residual_family=family,
        )
        for family in ("reflection", "transmission")
    }


def _read_residual_csv(path: Path) -> pd.DataFrame:
    dtype = {
        "time": "float64",
        "step": "int64",
        "array": "string",
        "role": "string",
        "receiver_id": "int64",
        "component": "string",
        "residual": "float64",
        "reference": "float64",
        "observed": "float64",
    }
    try:
        frame = pd.read_csv(path, dtype=dtype, usecols=list(REQUIRED_RESIDUAL_COLUMNS))
    except Exception as exc:
        raise ResidualCSVError(f"Could not read residual CSV {path}: {exc}") from exc
    return frame


def _validate_time_column(
    frame: pd.DataFrame,
    *,
    geometry: str,
    dt: float,
    atol: float,
    rtol: float,
    policy: str,
) -> tuple[str, np.ndarray]:
    steps = frame["step"].to_numpy(dtype=np.int64, copy=False)
    original = frame["time"].to_numpy(dtype=np.float64, copy=False)
    expected = steps.astype(np.float64) * float(dt)
    physical_ok = np.allclose(original, expected, rtol=rtol, atol=atol)
    step_alias_ok = np.allclose(
        original,
        steps.astype(np.float64),
        rtol=rtol,
        atol=atol,
    )

    if physical_ok:
        return "physical_time", expected
    if step_alias_ok:
        if policy != "legacy-step-index":
            max_error = float(np.max(np.abs(original - expected)))
            raise TimeConventionError(
                f"{geometry}: CSV column 'time' is an exact alias of integer 'step', "
                "not physical time. The physical convention requires time=step*dt "
                f"with dt={dt}; max discrepancy={max_error:.17g}. "
                "Re-run explicitly with --time-column-policy legacy-step-index "
                "to read the CSV time column as a step index and convert it to physical time."
            )
        return "legacy_step_index_alias", expected

    errors = np.abs(original - expected)
    index = int(np.argmax(errors))
    raise TimeConventionError(
        f"{geometry}: incompatible CSV time column at row {index}: "
        f"time_original={original[index]:.17g}, step={steps[index]}, "
        f"expected_physical={expected[index]:.17g}, "
        f"abs_error={errors[index]:.17g}, atol={atol}, rtol={rtol}."
    )


def _infer_residual_convention(
    frame: pd.DataFrame,
    *,
    geometry: str,
    atol: float,
    rtol: float,
) -> str:
    residual = frame["residual"].to_numpy(dtype=np.float64, copy=False)
    reference = frame["reference"].to_numpy(dtype=np.float64, copy=False)
    observed = frame["observed"].to_numpy(dtype=np.float64, copy=False)
    ref_minus_obs = reference - observed
    obs_minus_ref = observed - reference
    if np.allclose(residual, ref_minus_obs, rtol=rtol, atol=atol):
        return "reference_minus_observed"
    if np.allclose(residual, obs_minus_ref, rtol=rtol, atol=atol):
        return "observed_minus_reference"
    error_ref = float(np.max(np.abs(residual - ref_minus_obs)))
    error_obs = float(np.max(np.abs(residual - obs_minus_ref)))
    raise ResidualConventionError(
        f"{geometry}: residual values match neither reference-observed nor "
        f"observed-reference within atol={atol}, rtol={rtol}. "
        f"max_error_reference_minus_observed={error_ref:.17g}, "
        f"max_error_observed_minus_reference={error_obs:.17g}."
    )


def _validate_completeness(
    frame: pd.DataFrame,
    *,
    geometry: str,
    specs: Mapping[str, ReceiverSpec],
) -> None:
    key = [
        "array_name",
        "role",
        "residual_family",
        "receiver_id",
        "component",
        "step_index",
    ]
    duplicated = frame.duplicated(key, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, key].head(10).to_dict("records")
        raise ResidualCSVError(
            f"{geometry}: duplicate logical residual records for key {key}; "
            f"examples={examples}."
        )

    expected_components = set(EXPECTED_COMPONENTS)
    found_components = set(str(v) for v in frame["component"].unique())
    missing_components = sorted(expected_components - found_components)
    if missing_components:
        raise ResidualCSVError(
            f"{geometry}: required components are missing: {missing_components}; "
            f"found={sorted(found_components)}."
        )

    family_steps: dict[str, np.ndarray] = {}
    for family, spec in specs.items():
        subset = frame.loc[frame["residual_family"] == family]
        if subset.empty:
            raise ResidualCSVError(f"{geometry}: missing residual family {family!r}.")
        arrays = sorted(str(value) for value in subset["array_name"].unique())
        roles = sorted(str(value) for value in subset["role"].unique())
        if arrays != [spec.array_name] or roles != [spec.role]:
            raise ResidualCSVError(
                f"{geometry}/{family}: CSV array/role do not match manifest; "
                f"CSV arrays={arrays}, roles={roles}, manifest array={spec.array_name}, "
                f"role={spec.role}."
            )
        receiver_ids = np.sort(subset["receiver_id"].unique())
        expected_ids = np.arange(spec.num_receivers, dtype=np.int64)
        if not np.array_equal(receiver_ids, expected_ids):
            raise ResidualCSVError(
                f"{geometry}/{family}: receiver IDs must be contiguous 0.."
                f"{spec.num_receivers - 1}; found={receiver_ids.tolist()}."
            )
        steps = np.sort(subset["step_index"].unique())
        if steps.size == 0:
            raise ResidualCSVError(f"{geometry}/{family}: no time steps.")
        expected_steps = np.arange(int(steps[0]), int(steps[-1]) + 1, dtype=np.int64)
        if not np.array_equal(steps, expected_steps):
            missing = np.setdiff1d(expected_steps, steps).tolist()
            raise ResidualCSVError(
                f"{geometry}/{family}: missing or non-contiguous step indices: {missing}."
            )
        family_steps[family] = steps

        counts = subset.groupby(
            ["receiver_id", "component"],
            sort=False,
            observed=True,
        )["step_index"].nunique()
        bad_counts = counts[counts != steps.size]
        if not bad_counts.empty:
            raise ResidualCSVError(
                f"{geometry}/{family}: incomplete receiver/component time series: "
                f"{bad_counts.head(10).to_dict()}."
            )
        expected_rows = spec.num_receivers * len(found_components) * steps.size
        if int(subset.shape[0]) != int(expected_rows):
            raise ResidualCSVError(
                f"{geometry}/{family}: expected {expected_rows} rows from the "
                f"Cartesian contract, found {subset.shape[0]}."
            )

    if not np.array_equal(family_steps["reflection"], family_steps["transmission"]):
        raise ResidualCSVError(
            f"{geometry}: reflection and transmission do not share the same steps."
        )


def load_and_validate_campaign(
    residual_csv: str | Path,
    *,
    geometry: str,
    specs: Mapping[str, ReceiverSpec],
    config: ApertureComparisonConfig,
) -> tuple[pd.DataFrame, CampaignInspection]:
    if geometry not in ALLOWED_GEOMETRIES:
        raise ValueError(f"Unsupported geometry: {geometry!r}")
    path = _require_file(Path(residual_csv), role=f"{geometry} residual CSV")
    frame = _read_residual_csv(path)
    original_dtypes = {name: str(dtype) for name, dtype in frame.dtypes.items()}

    missing_columns = sorted(set(REQUIRED_RESIDUAL_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ResidualCSVError(f"{geometry}: missing columns {missing_columns} in {path}.")
    if frame.empty:
        raise ResidualCSVError(f"{geometry}: residual CSV is empty: {path}")

    missing_values = int(frame.isna().sum().sum())
    if missing_values:
        columns = frame.columns[frame.isna().any()].tolist()
        raise ResidualCSVError(
            f"{geometry}: {missing_values} missing values in columns {columns}: {path}"
        )

    numeric_columns = (
        "time",
        "step",
        "receiver_id",
        "residual",
        "reference",
        "observed",
    )
    nonfinite = 0
    for column in numeric_columns:
        values = frame[column].to_numpy(dtype=np.float64, copy=False)
        count = int(np.count_nonzero(~np.isfinite(values)))
        nonfinite += count
        if count:
            raise ResidualCSVError(
                f"{geometry}: column {column!r} contains {count} non-finite values."
            )

    if (frame["receiver_id"] < 0).any():
        raise ResidualCSVError(f"{geometry}: receiver_id must be non-negative.")
    if (frame["step"] < 0).any():
        raise ResidualCSVError(f"{geometry}: step must be non-negative.")

    frame = frame.rename(
        columns={
            "time": "time_original",
            "step": "step_index",
            "array": "array_name",
        }
    )
    frame.insert(0, "geometry", geometry)
    frame["residual_family"] = [
        _family_from_role_array(role, array_name)
        for role, array_name in zip(
            frame["role"].array,
            frame["array_name"].array,
            strict=True,
        )
    ]

    time_status, physical_time = _validate_time_column(
        frame.rename(columns={"time_original": "time", "step_index": "step"}),
        geometry=geometry,
        dt=config.dt,
        atol=config.time_atol,
        rtol=config.time_rtol,
        policy=config.time_column_policy,
    )
    frame.insert(3, "time_physical", physical_time)

    residual_convention = _infer_residual_convention(
        frame,
        geometry=geometry,
        atol=config.value_atol,
        rtol=config.value_rtol,
    )
    frame["residual_convention"] = residual_convention
    frame["source_csv"] = str(path)

    _validate_completeness(frame, geometry=geometry, specs=specs)

    duplicate_count = int(
        frame.duplicated(
            [
                "geometry",
                "array_name",
                "role",
                "residual_family",
                "receiver_id",
                "component",
                "step_index",
            ]
        ).sum()
    )
    steps = np.sort(frame["step_index"].unique())
    receiver_counts = {
        family: int(
            frame.loc[frame["residual_family"] == family, "receiver_id"].nunique()
        )
        for family in ("reflection", "transmission")
    }
    inspection = CampaignInspection(
        geometry=geometry,
        path=path,
        file_size_bytes=int(path.stat().st_size),
        sha256=_sha256(path),
        num_rows=int(frame.shape[0]),
        columns=tuple(REQUIRED_RESIDUAL_COLUMNS),
        dtypes=original_dtypes,
        arrays=tuple(sorted(str(v) for v in frame["array_name"].unique())),
        roles=tuple(sorted(str(v) for v in frame["role"].unique())),
        components=tuple(sorted(str(v) for v in frame["component"].unique())),
        num_steps=int(steps.size),
        step_min=int(steps[0]),
        step_max=int(steps[-1]),
        time_original_min=float(frame["time_original"].min()),
        time_original_max=float(frame["time_original"].max()),
        time_status=time_status,
        residual_convention=residual_convention,
        duplicate_count=duplicate_count,
        missing_value_count=missing_values,
        nonfinite_count=nonfinite,
        receiver_counts=receiver_counts,
    )

    frame = frame.sort_values(
        [
            "residual_family",
            "step_index",
            "receiver_id",
            "component",
        ],
        kind="mergesort",
        ignore_index=True,
    )
    return frame, inspection


def trapezoidal_point_weights(points: np.ndarray) -> np.ndarray:
    """Return point weights for a one-dimensional polyline trapezoidal rule."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ReceiverGeometryError(
            f"Polyline points must have shape (N, 2), got {points.shape}."
        )
    if points.shape[0] < 2:
        raise ReceiverGeometryError("At least two points are required for quadrature.")
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if not np.all(np.isfinite(distances)) or np.any(distances <= 0.0):
        raise ReceiverGeometryError(
            "Polyline centres must be distinct and consecutive distances positive."
        )
    weights = np.empty(points.shape[0], dtype=np.float64)
    weights[0] = 0.5 * distances[0]
    weights[-1] = 0.5 * distances[-1]
    if points.shape[0] > 2:
        weights[1:-1] = 0.5 * (distances[:-1] + distances[1:])
    return weights


def _curvilinear_coordinate(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(distances, dtype=np.float64)))


def _side_local_rows(
    centers: np.ndarray,
    *,
    geometry_atol: float,
) -> dict[str, np.ndarray]:
    x = centers[:, 0]
    y = centers[:, 1]
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    ymax = float(np.max(y))
    left = np.flatnonzero(np.isclose(x, xmin, rtol=0.0, atol=geometry_atol))
    top = np.flatnonzero(np.isclose(y, ymax, rtol=0.0, atol=geometry_atol))
    right = np.flatnonzero(np.isclose(x, xmax, rtol=0.0, atol=geometry_atol))
    if left.size < 2 or top.size < 2 or right.size < 2:
        raise ReceiverGeometryError(
            "Box transmission centres do not define left, top, and right segments."
        )
    covered = set(left.tolist()) | set(top.tolist()) | set(right.tolist())
    if len(covered) != centers.shape[0]:
        missing = sorted(set(range(centers.shape[0])) - covered)
        raise ReceiverGeometryError(
            f"Box transmission centres outside left/top/right segments: {missing}."
        )
    return {"left": left, "top": top, "right": right}


def build_receiver_geometry(
    specs_by_geometry: Mapping[str, Mapping[str, ReceiverSpec]],
    *,
    geometry_atol: float,
    corner_policy: str,
) -> pd.DataFrame:
    if corner_policy != "shared":
        raise ReceiverGeometryError(
            "Only the physically conservative 'shared' corner policy is implemented."
        )

    rows: list[dict[str, Any]] = []
    unknown = sorted(set(specs_by_geometry) - set(ALLOWED_GEOMETRIES))
    if unknown:
        raise ReceiverGeometryError(f"Unsupported receiver geometries: {unknown}.")
    geometries = [
        geometry for geometry in ALLOWED_GEOMETRIES if geometry in specs_by_geometry
    ]
    if not geometries:
        raise ReceiverGeometryError("No receiver geometry was provided.")
    for geometry in geometries:
        specs = specs_by_geometry[geometry]
        for family in ("reflection", "transmission"):
            spec = specs[family]
            centers = spec.centers
            all_weights = trapezoidal_point_weights(centers)
            all_s = _curvilinear_coordinate(centers)
            side_weights = {
                "left": np.zeros(spec.num_receivers, dtype=np.float64),
                "top": np.zeros(spec.num_receivers, dtype=np.float64),
                "right": np.zeros(spec.num_receivers, dtype=np.float64),
                "line": np.zeros(spec.num_receivers, dtype=np.float64),
            }
            side_s = {
                "left": np.full(spec.num_receivers, np.nan, dtype=np.float64),
                "top": np.full(spec.num_receivers, np.nan, dtype=np.float64),
                "right": np.full(spec.num_receivers, np.nan, dtype=np.float64),
                "line": np.full(spec.num_receivers, np.nan, dtype=np.float64),
            }
            memberships: list[list[str]] = [[] for _ in range(spec.num_receivers)]

            if geometry == "box" and family == "transmission":
                side_indices = _side_local_rows(
                    centers,
                    geometry_atol=geometry_atol,
                )
                for side, indices in side_indices.items():
                    if side == "left":
                        order = indices[np.argsort(centers[indices, 1], kind="mergesort")]
                    elif side == "top":
                        order = indices[np.argsort(centers[indices, 0], kind="mergesort")]
                    else:
                        order = indices[
                            np.argsort(-centers[indices, 1], kind="mergesort")
                        ]
                    local_points = centers[order]
                    local_weights = trapezoidal_point_weights(local_points)
                    local_s = _curvilinear_coordinate(local_points)
                    side_weights[side][order] = local_weights
                    side_s[side][order] = local_s
                    for receiver_id in order.tolist():
                        memberships[receiver_id].append(side)

                decomposed = (
                    side_weights["left"]
                    + side_weights["top"]
                    + side_weights["right"]
                )
                if not np.allclose(
                    decomposed,
                    all_weights,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                ):
                    error = float(np.max(np.abs(decomposed - all_weights)))
                    raise ReceiverGeometryError(
                        "Shared-corner side weights do not reproduce the full U-shaped "
                        f"aperture weights; max error={error:.17g}."
                    )
            else:
                side_weights["line"] = all_weights.copy()
                side_s["line"] = all_s.copy()
                memberships = [["line"] for _ in range(spec.num_receivers)]

            for receiver_id in range(spec.num_receivers):
                member = memberships[receiver_id]
                if not member:
                    raise ReceiverGeometryError(
                        f"No side membership for {geometry}/{family}/receiver={receiver_id}."
                    )
                rows.append(
                    {
                        "geometry": geometry,
                        "residual_family": family,
                        "array_name": spec.array_name,
                        "role": spec.role,
                        "receiver_id": receiver_id,
                        "center_x": float(centers[receiver_id, 0]),
                        "center_y": float(centers[receiver_id, 1]),
                        "curvilinear_s_all": float(all_s[receiver_id]),
                        "weight_all": float(all_weights[receiver_id]),
                        "side_membership": "|".join(member),
                        "is_corner": len(member) > 1,
                        "weight_left": float(side_weights["left"][receiver_id]),
                        "weight_top": float(side_weights["top"][receiver_id]),
                        "weight_right": float(side_weights["right"][receiver_id]),
                        "weight_line": float(side_weights["line"][receiver_id]),
                        "s_left": float(side_s["left"][receiver_id]),
                        "s_top": float(side_s["top"][receiver_id]),
                        "s_right": float(side_s["right"][receiver_id]),
                        "s_line": float(side_s["line"][receiver_id]),
                        "manifest_geometry": spec.manifest_geometry,
                        "nominal_spacing": spec.spacing,
                        "receiver_sigma": spec.sigma,
                        "source_manifest": str(spec.source_manifest),
                        "corner_policy": corner_policy,
                    }
                )

    table = pd.DataFrame(rows)
    key = ["geometry", "residual_family", "receiver_id"]
    if table.duplicated(key).any():
        raise ReceiverGeometryError(f"Duplicate receiver geometry rows for key {key}.")
    return table.sort_values(key, kind="mergesort", ignore_index=True)


def attach_receiver_geometry(
    frame: pd.DataFrame,
    receiver_geometry: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "geometry",
        "residual_family",
        "array_name",
        "role",
        "receiver_id",
        "center_x",
        "center_y",
        "curvilinear_s_all",
        "weight_all",
        "side_membership",
        "is_corner",
        "weight_left",
        "weight_top",
        "weight_right",
        "weight_line",
        "s_left",
        "s_top",
        "s_right",
        "s_line",
        "corner_policy",
    ]
    merged = frame.merge(
        receiver_geometry[columns],
        on=["geometry", "residual_family", "array_name", "role", "receiver_id"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if merged["center_x"].isna().any() or merged["center_y"].isna().any():
        bad = merged.loc[
            merged["center_x"].isna() | merged["center_y"].isna(),
            ["geometry", "residual_family", "receiver_id"],
        ].head(10)
        raise ReceiverGeometryError(
            f"Could not attach receiver centres to residual rows: {bad.to_dict('records')}"
        )
    return merged


def compute_receiver_rms_by_component(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(
        [
            "geometry",
            "residual_family",
            "array_name",
            "role",
            "receiver_id",
            "component",
            "center_x",
            "center_y",
            "curvilinear_s_all",
            "weight_all",
            "side_membership",
            "is_corner",
            "weight_left",
            "weight_top",
            "weight_right",
            "weight_line",
            "s_left",
            "s_top",
            "s_right",
            "s_line",
            "corner_policy",
            "residual_convention",
            "source_csv",
        ],
        sort=True,
        observed=True,
        dropna=False,
    )
    output = grouped.agg(
        num_steps=("residual", "size"),
        step_index_min=("step_index", "min"),
        step_index_max=("step_index", "max"),
        time_original_min=("time_original", "min"),
        time_original_max=("time_original", "max"),
        time_physical_min=("time_physical", "min"),
        time_physical_max=("time_physical", "max"),
        sum_squared=("residual", lambda values: float(np.dot(values.to_numpy(), values.to_numpy()))),
        signed_sum=("residual", "sum"),
        minimum=("residual", "min"),
        maximum=("residual", "max"),
    ).reset_index()
    output["rms"] = np.sqrt(
        output["sum_squared"].to_numpy(dtype=np.float64)
        / output["num_steps"].to_numpy(dtype=np.float64)
    )

    positive = frame.assign(
        positive=np.maximum(frame["residual"].to_numpy(dtype=np.float64), 0.0),
        negative_abs=np.maximum(-frame["residual"].to_numpy(dtype=np.float64), 0.0),
    ).groupby(
        [
            "geometry",
            "residual_family",
            "receiver_id",
            "component",
        ],
        sort=True,
        observed=True,
    )[["positive", "negative_abs"]].sum().reset_index()
    output = output.merge(
        positive,
        on=["geometry", "residual_family", "receiver_id", "component"],
        how="left",
        validate="one_to_one",
    )
    return output.sort_values(
        ["geometry", "residual_family", "receiver_id", "component"],
        kind="mergesort",
        ignore_index=True,
    )


def compute_receiver_rms_vector(
    rms_by_component: pd.DataFrame,
    receiver_geometry: pd.DataFrame,
) -> pd.DataFrame:
    index = [
        "geometry",
        "residual_family",
        "array_name",
        "role",
        "receiver_id",
        "center_x",
        "center_y",
        "residual_convention",
        "source_csv",
    ]
    pivot = rms_by_component.pivot(index=index, columns="component", values="rms")
    pivot.columns = [f"rms_{column}" for column in pivot.columns]
    pivot = pivot.reset_index()
    for component in EXPECTED_COMPONENTS:
        name = f"rms_{component}"
        if name not in pivot.columns:
            raise ResidualCSVError(f"Missing component RMS column {name!r}.")
    component_columns = sorted(
        column for column in pivot.columns if column.startswith("rms_")
    )
    pivot["rms_vector_xy"] = np.sqrt(
        pivot["rms_x"].to_numpy(dtype=np.float64) ** 2
        + pivot["rms_y"].to_numpy(dtype=np.float64) ** 2
    )
    pivot["rms_all_components"] = np.sqrt(
        np.sum(
            np.column_stack(
                [pivot[column].to_numpy(dtype=np.float64) ** 2 for column in component_columns]
            ),
            axis=1,
        )
    )
    pivot["num_components"] = len(component_columns)

    totals = rms_by_component.groupby(
        ["geometry", "residual_family", "receiver_id"],
        sort=True,
        observed=True,
    ).agg(
        num_steps=("num_steps", "max"),
        step_index_min=("step_index_min", "min"),
        step_index_max=("step_index_max", "max"),
        time_physical_min=("time_physical_min", "min"),
        time_physical_max=("time_physical_max", "max"),
        sum_squared=("sum_squared", "sum"),
        signed_sum=("signed_sum", "sum"),
        positive_sum=("positive", "sum"),
        negative_sum_abs=("negative_abs", "sum"),
    ).reset_index()
    pivot = pivot.merge(
        totals,
        on=["geometry", "residual_family", "receiver_id"],
        how="left",
        validate="one_to_one",
    )
    geometry_columns = [
        "geometry",
        "residual_family",
        "receiver_id",
        "curvilinear_s_all",
        "weight_all",
        "side_membership",
        "is_corner",
        "weight_left",
        "weight_top",
        "weight_right",
        "weight_line",
        "s_left",
        "s_top",
        "s_right",
        "s_line",
        "corner_policy",
    ]
    pivot = pivot.merge(
        receiver_geometry[geometry_columns],
        on=["geometry", "residual_family", "receiver_id"],
        how="left",
        validate="one_to_one",
    )
    return pivot.sort_values(
        ["geometry", "residual_family", "receiver_id"],
        kind="mergesort",
        ignore_index=True,
    )


def _safe_ratio(numerator: float, denominator: float, *, epsilon: float) -> tuple[float, str]:
    numerator = float(numerator)
    denominator = float(denominator)
    if abs(denominator) <= epsilon:
        if abs(numerator) <= epsilon:
            return math.nan, "both_near_zero"
        return math.nan, "denominator_near_zero"
    return numerator / denominator, "ok"


def _scopes_for(geometry: str, family: str) -> tuple[str, ...]:
    if geometry == "box" and family == "transmission":
        return ("all", "left", "top", "right")
    return ("all", "line")


def _weight_column(scope: str) -> str:
    return "weight_all" if scope == "all" else f"weight_{scope}"


def compute_aperture_summaries(
    rms_vector: pd.DataFrame,
    *,
    dt: float,
    ratio_epsilon: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (geometry, family), subset in rms_vector.groupby(
        ["geometry", "residual_family"], sort=True, observed=True
    ):
        for scope in _scopes_for(str(geometry), str(family)):
            weight_column = _weight_column(scope)
            scoped = subset.loc[subset[weight_column] > 0.0].copy()
            if scoped.empty:
                continue
            weights = scoped[weight_column].to_numpy(dtype=np.float64)
            rms = scoped["rms_vector_xy"].to_numpy(dtype=np.float64)
            sum_weights = float(np.sum(weights))
            if sum_weights <= 0.0:
                raise ReceiverGeometryError(
                    f"Non-positive aperture length for {geometry}/{family}/{scope}."
                )
            rms_mean = float(np.mean(rms))
            rms_global_sum = float(np.sqrt(np.sum(rms * rms)))
            rms_global_normalized = float(np.sqrt(np.mean(rms * rms)))
            rms_weighted = float(np.sqrt(np.sum(weights * rms * rms) / sum_weights))
            rms_x = scoped["rms_x"].to_numpy(dtype=np.float64)
            rms_y = scoped["rms_y"].to_numpy(dtype=np.float64)
            rms_x_weighted = float(np.sqrt(np.sum(weights * rms_x * rms_x) / sum_weights))
            rms_y_weighted = float(np.sqrt(np.sum(weights * rms_y * rms_y) / sum_weights))
            rms_y_over_x, rms_y_over_x_status = _safe_ratio(
                rms_y_weighted, rms_x_weighted, epsilon=ratio_epsilon
            )
            component_energy_sum = rms_x_weighted**2 + rms_y_weighted**2
            x_fraction = (
                math.nan if component_energy_sum <= ratio_epsilon
                else rms_x_weighted**2 / component_energy_sum
            )
            y_fraction = (
                math.nan if component_energy_sum <= ratio_epsilon
                else rms_y_weighted**2 / component_energy_sum
            )
            energy = float(
                dt
                * np.sum(
                    weights
                    * scoped["sum_squared"].to_numpy(dtype=np.float64)
                )
            )
            discrete_l2_norm = float(np.sqrt(max(energy, 0.0)))
            positive_l1 = float(
                dt
                * np.sum(
                    weights
                    * scoped["positive_sum"].to_numpy(dtype=np.float64)
                )
            )
            negative_l1_abs = float(
                dt
                * np.sum(
                    weights
                    * scoped["negative_sum_abs"].to_numpy(dtype=np.float64)
                )
            )
            signed_integral = float(
                dt
                * np.sum(
                    weights
                    * scoped["signed_sum"].to_numpy(dtype=np.float64)
                )
            )
            signed_denominator = positive_l1 + negative_l1_abs
            signed_balance = (
                math.nan
                if signed_denominator <= ratio_epsilon
                else (positive_l1 - negative_l1_abs) / signed_denominator
            )
            pos_neg_ratio, pos_neg_status = _safe_ratio(
                positive_l1,
                negative_l1_abs,
                epsilon=ratio_epsilon,
            )
            rows.append(
                {
                    "geometry": geometry,
                    "residual_family": family,
                    "role": str(scoped["role"].iloc[0]),
                    "array_name": str(scoped["array_name"].iloc[0]),
                    "scope": scope,
                    "num_receivers": int(scoped.shape[0]),
                    "aperture_length": sum_weights,
                    "num_steps": int(scoped["num_steps"].iloc[0]),
                    "step_index_min": int(scoped["step_index_min"].min()),
                    "step_index_max": int(scoped["step_index_max"].max()),
                    "time_initial": float(scoped["time_physical_min"].min()),
                    "time_final": float(scoped["time_physical_max"].max()),
                    "dt": float(dt),
                    "rms_mean_by_receiver": rms_mean,
                    "rms_global_sum": rms_global_sum,
                    "rms_global_normalized": rms_global_normalized,
                    "rms_spatially_weighted": rms_weighted,
                    "rms_x_spatially_weighted": rms_x_weighted,
                    "rms_y_spatially_weighted": rms_y_weighted,
                    "rms_y_over_x": rms_y_over_x,
                    "rms_y_over_x_status": rms_y_over_x_status,
                    "x_quadratic_fraction": x_fraction,
                    "y_quadratic_fraction": y_fraction,
                    "discrete_energy_dt_space": energy,
                    "discrete_l2_norm_dt_space": discrete_l2_norm,
                    "signed_integral_dt_space": signed_integral,
                    "positive_l1_dt_space": positive_l1,
                    "negative_l1_abs_dt_space": negative_l1_abs,
                    "positive_to_negative_l1_ratio": pos_neg_ratio,
                    "positive_to_negative_l1_ratio_status": pos_neg_status,
                    "signed_balance": signed_balance,
                    "residual_convention": str(scoped["residual_convention"].iloc[0]),
                    "source_csv": str(scoped["source_csv"].iloc[0]),
                    "weighting_method": "polyline_trapezoidal",
                    "corner_policy": str(scoped["corner_policy"].iloc[0]),
                }
            )
    result = pd.DataFrame(rows)
    result["energy_contribution_to_box_all"] = math.nan
    result["energy_contribution_status"] = "not_applicable"

    box_transmission = result.loc[
        (result["geometry"] == "box")
        & (result["residual_family"] == "transmission")
    ]
    if not box_transmission.empty:
        all_rows = box_transmission.loc[box_transmission["scope"] == "all"]
        side_rows = box_transmission.loc[
            box_transmission["scope"].isin(["left", "top", "right"])
        ]
        if len(all_rows) != 1 or len(side_rows) != 3:
            raise ReceiverGeometryError(
                "Expected one box/transmission/all row and three side rows."
            )
        total_energy = float(all_rows.iloc[0]["discrete_energy_dt_space"])
        contributions: dict[str, float] = {}
        for index, row in side_rows.iterrows():
            contribution, status = _safe_ratio(
                float(row["discrete_energy_dt_space"]),
                total_energy,
                epsilon=ratio_epsilon,
            )
            result.loc[index, "energy_contribution_to_box_all"] = contribution
            result.loc[index, "energy_contribution_status"] = status
            contributions[str(row["scope"])] = contribution
        result.loc[all_rows.index, "energy_contribution_to_box_all"] = 1.0
        result.loc[all_rows.index, "energy_contribution_status"] = "reference_total"
        finite_contributions = [
            value for value in contributions.values() if math.isfinite(value)
        ]
        if finite_contributions and not math.isclose(
            sum(finite_contributions), 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ReceiverGeometryError(
                "Box side energy contributions do not sum to one: "
                f"{contributions}."
            )
    return result.sort_values(
        ["residual_family", "geometry", "scope"],
        kind="mergesort",
        ignore_index=True,
    )


def _comparison_ratio(
    numerator: float,
    denominator: float,
    *,
    epsilon: float,
) -> tuple[float, str]:
    return _safe_ratio(numerator, denominator, epsilon=epsilon)


def build_line_box_summary(
    side_summary: pd.DataFrame,
    *,
    ratio_epsilon: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = (
        "rms_mean_by_receiver",
        "rms_global_sum",
        "rms_global_normalized",
        "rms_spatially_weighted",
        "rms_x_spatially_weighted",
        "rms_y_spatially_weighted",
        "rms_y_over_x",
        "discrete_energy_dt_space",
        "discrete_l2_norm_dt_space",
    )
    for family in ("reflection", "transmission"):
        line = side_summary.loc[
            (side_summary["geometry"] == "line")
            & (side_summary["residual_family"] == family)
            & (side_summary["scope"] == "all")
        ]
        box_all = side_summary.loc[
            (side_summary["geometry"] == "box")
            & (side_summary["residual_family"] == family)
            & (side_summary["scope"] == "all")
        ]
        if len(line) != 1 or len(box_all) != 1:
            raise ReceiverApertureError(
                f"Missing line/box all-aperture summaries for family={family}."
            )
        line_row = line.iloc[0]
        box_row = box_all.iloc[0]
        base: dict[str, Any] = {
            "residual_family": family,
            "line_num_receivers": int(line_row["num_receivers"]),
            "box_num_receivers": int(box_row["num_receivers"]),
            "line_aperture_length": float(line_row["aperture_length"]),
            "box_aperture_length": float(box_row["aperture_length"]),
            "residual_convention": str(line_row["residual_convention"]),
            "line_source_csv": str(line_row["source_csv"]),
            "box_source_csv": str(box_row["source_csv"]),
        }
        for metric in metrics:
            line_value = float(line_row[metric])
            box_value = float(box_row[metric])
            ratio, status = _comparison_ratio(
                box_value,
                line_value,
                epsilon=ratio_epsilon,
            )
            base[f"line__{metric}"] = line_value
            base[f"box_all__{metric}"] = box_value
            base[f"box_all_over_line__{metric}"] = ratio
            base[f"box_all_over_line__{metric}__status"] = status

        if family == "transmission":
            for side in ("left", "top", "right"):
                side_match = side_summary.loc[
                    (side_summary["geometry"] == "box")
                    & (side_summary["residual_family"] == family)
                    & (side_summary["scope"] == side)
                ]
                if len(side_match) != 1:
                    raise ReceiverApertureError(
                        f"Missing box transmission side summary: {side}."
                    )
                side_row = side_match.iloc[0]
                base[f"box_{side}__num_receivers"] = int(side_row["num_receivers"])
                base[f"box_{side}__aperture_length"] = float(
                    side_row["aperture_length"]
                )
                base[f"box_{side}__energy_contribution"] = float(
                    side_row["energy_contribution_to_box_all"]
                )
                for metric in metrics:
                    value = float(side_row[metric])
                    base[f"box_{side}__{metric}"] = value
                    ratio, status = _comparison_ratio(
                        value,
                        float(line_row[metric]),
                        epsilon=ratio_epsilon,
                    )
                    base[f"box_{side}_over_line__{metric}"] = ratio
                    base[f"box_{side}_over_line__{metric}__status"] = status
        rows.append(base)
    return pd.DataFrame(rows).sort_values("residual_family", ignore_index=True)


def validate_common_apertures(
    canonical: pd.DataFrame,
    receiver_geometry: pd.DataFrame,
    *,
    atol: float,
    rtol: float,
) -> pd.DataFrame:
    """Compare identical physical apertures across line and box campaigns.

    Reflection compares the complete gamma1 line. Transmission compares the
    line campaign gamma2 line with the top subset of the box gamma2 aperture.
    """
    rows: list[dict[str, Any]] = []
    for family in ("reflection", "transmission"):
        line_geometry = receiver_geometry.loc[
            (receiver_geometry["geometry"] == "line")
            & (receiver_geometry["residual_family"] == family),
            ["receiver_id", "center_x", "center_y"],
        ].copy()
        if family == "reflection":
            box_geometry = receiver_geometry.loc[
                (receiver_geometry["geometry"] == "box")
                & (receiver_geometry["residual_family"] == family),
                ["receiver_id", "center_x", "center_y"],
            ].copy()
        else:
            box_geometry = receiver_geometry.loc[
                (receiver_geometry["geometry"] == "box")
                & (receiver_geometry["residual_family"] == family)
                & (receiver_geometry["weight_top"] > 0.0),
                ["receiver_id", "center_x", "center_y"],
            ].copy()

        line_geometry["coord_key"] = list(
            zip(
                np.round(line_geometry["center_x"], 12),
                np.round(line_geometry["center_y"], 12),
                strict=True,
            )
        )
        box_geometry["coord_key"] = list(
            zip(
                np.round(box_geometry["center_x"], 12),
                np.round(box_geometry["center_y"], 12),
                strict=True,
            )
        )
        if set(line_geometry["coord_key"]) != set(box_geometry["coord_key"]):
            missing_box = sorted(
                set(line_geometry["coord_key"]) - set(box_geometry["coord_key"])
            )
            missing_line = sorted(
                set(box_geometry["coord_key"]) - set(line_geometry["coord_key"])
            )
            raise ReceiverGeometryError(
                f"Common-aperture coordinate mismatch for {family}: "
                f"missing_in_box={missing_box}, missing_in_line={missing_line}."
            )

        line_map = dict(
            zip(line_geometry["receiver_id"], line_geometry["coord_key"], strict=True)
        )
        box_map = dict(
            zip(box_geometry["receiver_id"], box_geometry["coord_key"], strict=True)
        )
        line_data = canonical.loc[
            (canonical["geometry"] == "line")
            & (canonical["residual_family"] == family)
        ].copy()
        box_data = canonical.loc[
            (canonical["geometry"] == "box")
            & (canonical["residual_family"] == family)
            & (canonical["receiver_id"].isin(box_geometry["receiver_id"]))
        ].copy()
        line_data["coord_key"] = line_data["receiver_id"].map(line_map)
        box_data["coord_key"] = box_data["receiver_id"].map(box_map)
        merge_key = ["coord_key", "component", "step_index"]
        merged = line_data.merge(
            box_data,
            on=merge_key,
            how="outer",
            validate="one_to_one",
            suffixes=("_line", "_box"),
            indicator=True,
        )
        if not (merged["_merge"] == "both").all():
            counts = merged["_merge"].value_counts().to_dict()
            raise ResidualCSVError(
                f"Common-aperture record mismatch for {family}: {counts}."
            )
        for quantity in ("residual", "reference", "observed"):
            line_values = merged[f"{quantity}_line"].to_numpy(dtype=np.float64)
            box_values = merged[f"{quantity}_box"].to_numpy(dtype=np.float64)
            differences = np.abs(line_values - box_values)
            tolerance = atol + rtol * np.abs(box_values)
            pass_mask = differences <= tolerance
            absolute = float(np.max(differences))
            denominator = np.maximum(np.abs(box_values), atol)
            relative = float(np.max(differences / denominator))
            rows.append(
                {
                    "residual_family": family,
                    "common_aperture": "gamma1_full"
                    if family == "reflection"
                    else "gamma2_top",
                    "quantity": quantity,
                    "num_compared_values": int(differences.size),
                    "max_abs_difference": absolute,
                    "max_relative_difference": relative,
                    "atol": float(atol),
                    "rtol": float(rtol),
                    "status": "ok" if bool(np.all(pass_mask)) else "failed",
                }
            )
            if not np.all(pass_mask):
                index = int(np.argmax(differences - tolerance))
                record = merged.iloc[index]
                raise ReceiverApertureError(
                    f"Common-aperture consistency failed: family={family}, "
                    f"quantity={quantity}, coord={record['coord_key']}, "
                    f"component={record['component']}, step={record['step_index']}, "
                    f"line={line_values[index]:.17g}, box={box_values[index]:.17g}, "
                    f"abs_diff={differences[index]:.17g}, "
                    f"tolerance={tolerance[index]:.17g}."
                )
    return pd.DataFrame(rows)


def build_time_validation_summary(
    inspections: Sequence[CampaignInspection],
    *,
    dt: float,
    time_atol: float,
    time_rtol: float,
    time_column_policy: str,
) -> pd.DataFrame:
    rows = []
    for inspection in inspections:
        rows.append(
            {
                "geometry": inspection.geometry,
                "num_steps": inspection.num_steps,
                "step_index_min": inspection.step_min,
                "step_index_max": inspection.step_max,
                "time_original_min": inspection.time_original_min,
                "time_original_max": inspection.time_original_max,
                "time_physical_min": inspection.step_min * dt,
                "time_physical_max": inspection.step_max * dt,
                "dt": float(dt),
                "time_column_status": inspection.time_status,
                "time_column_policy": time_column_policy,
                "time_atol": float(time_atol),
                "time_rtol": float(time_rtol),
                "source_csv": str(inspection.path),
            }
        )
    return pd.DataFrame(rows).sort_values("geometry", ignore_index=True)


def select_representative_receivers(
    receiver_geometry: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (geometry, family), subset in receiver_geometry.groupby(
        ["geometry", "residual_family"], sort=True, observed=True
    ):
        subset = subset.sort_values("curvilinear_s_all", kind="mergesort")
        candidates: list[tuple[str, pd.Series]] = []
        candidates.append(("path_start", subset.iloc[0]))
        if geometry == "box" and family == "transmission":
            for side in ("left", "top", "right"):
                side_subset = subset.loc[subset[f"weight_{side}"] > 0.0].sort_values(
                    f"s_{side}", kind="mergesort"
                )
                candidates.append((f"{side}_center", side_subset.iloc[len(side_subset) // 2]))
        else:
            candidates.append(("path_midpoint", subset.iloc[len(subset) // 2]))
        candidates.append(("path_end", subset.iloc[-1]))
        seen: set[int] = set()
        for selection_rule, row in candidates:
            receiver_id = int(row["receiver_id"])
            if receiver_id in seen:
                continue
            seen.add(receiver_id)
            rows.append(
                {
                    "geometry": geometry,
                    "residual_family": family,
                    "receiver_id": receiver_id,
                    "center_x": float(row["center_x"]),
                    "center_y": float(row["center_y"]),
                    "selection_rule": selection_rule,
                    "side_membership": str(row["side_membership"]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["residual_family", "geometry", "receiver_id"],
        kind="mergesort",
        ignore_index=True,
    )


def build_representative_signals(
    canonical: pd.DataFrame,
    representative_receivers: pd.DataFrame,
) -> pd.DataFrame:
    selected = canonical.merge(
        representative_receivers,
        on=["geometry", "residual_family", "receiver_id", "center_x", "center_y"],
        how="inner",
        validate="many_to_one",
    )
    columns = [
        "geometry",
        "residual_family",
        "array_name",
        "role",
        "receiver_id",
        "center_x",
        "center_y",
        "selection_rule",
        "side_membership_y",
        "component",
        "step_index",
        "time_original",
        "time_physical",
        "residual",
        "reference",
        "observed",
        "residual_convention",
        "source_csv",
    ]
    rename = {"side_membership_y": "side_membership"}
    available = [column for column in columns if column in selected.columns]
    selected = selected[available].rename(columns=rename)
    return selected.sort_values(
        [
            "residual_family",
            "geometry",
            "selection_rule",
            "component",
            "step_index",
        ],
        kind="mergesort",
        ignore_index=True,
    )


def _build_scientific_summary(
    inspections: Sequence[CampaignInspection],
    side_summary: pd.DataFrame,
    line_box_summary: pd.DataFrame,
    common: pd.DataFrame,
    *,
    analysis_geometry: str,
) -> dict[str, Any]:
    line_rows = side_summary.loc[
        (side_summary["geometry"] == "line")
        & (side_summary["scope"] == "all")
    ]
    line_apertures: dict[str, Any] = {}
    for _, row in line_rows.iterrows():
        family = str(row["residual_family"])
        line_apertures[family] = {
            "num_receivers": int(row["num_receivers"]),
            "aperture_length": float(row["aperture_length"]),
            "rms_spatially_weighted": float(row["rms_spatially_weighted"]),
            "rms_x_spatially_weighted": float(row["rms_x_spatially_weighted"]),
            "rms_y_spatially_weighted": float(row["rms_y_spatially_weighted"]),
            "rms_y_over_x": float(row["rms_y_over_x"]),
            "x_quadratic_fraction": float(row["x_quadratic_fraction"]),
            "y_quadratic_fraction": float(row["y_quadratic_fraction"]),
            "discrete_energy_dt_space": float(row["discrete_energy_dt_space"]),
            "signed_balance": float(row["signed_balance"]),
        }

    box_transmission = side_summary.loc[
        (side_summary["geometry"] == "box")
        & (side_summary["residual_family"] == "transmission")
    ]
    contributions = {
        str(row["scope"]): float(row["energy_contribution_to_box_all"])
        for _, row in box_transmission.loc[
            box_transmission["scope"].isin(["left", "top", "right"])
        ].iterrows()
    }

    ratios: dict[str, Any] = {}
    for _, row in line_box_summary.iterrows():
        family = str(row["residual_family"])
        ratios[family] = {
            "box_all_over_line_weighted_rms": float(
                row["box_all_over_line__rms_spatially_weighted"]
            ),
            "box_all_over_line_energy": float(
                row["box_all_over_line__discrete_energy_dt_space"]
            ),
        }
        if family == "transmission":
            ratios[family]["box_top_over_line_weighted_rms"] = float(
                row["box_top_over_line__rms_spatially_weighted"]
            )
            ratios[family]["box_left_over_line_weighted_rms"] = float(
                row["box_left_over_line__rms_spatially_weighted"]
            )
            ratios[family]["box_right_over_line_weighted_rms"] = float(
                row["box_right_over_line__rms_spatially_weighted"]
            )

    if common.empty:
        common_summary = {
            "status": "not_applicable",
            "num_checks": 0,
            "max_abs_difference": None,
            "max_relative_difference": None,
        }
    else:
        common_summary = {
            "status": "ok" if (common["status"] == "ok").all() else "failed",
            "num_checks": int(common.shape[0]),
            "max_abs_difference": float(common["max_abs_difference"].max()),
            "max_relative_difference": float(common["max_relative_difference"].max()),
        }

    geometries = [item.geometry for item in inspections]
    return {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "analysis_geometry": analysis_geometry,
        "analysis_mode": "line-only" if analysis_geometry == "line" else "line-box",
        "num_campaigns": len(inspections),
        "num_rows_input": int(sum(item.num_rows for item in inspections)),
        "geometries": geometries,
        "residual_families": ["reflection", "transmission"],
        "components": sorted(set().union(*(set(item.components) for item in inspections))),
        "receiver_counts": {
            item.geometry: dict(item.receiver_counts) for item in inspections
        },
        "time_status": {item.geometry: item.time_status for item in inspections},
        "residual_convention": {
            item.geometry: item.residual_convention for item in inspections
        },
        "line_aperture_metrics": line_apertures,
        "common_aperture_consistency": common_summary,
        "box_transmission_side_energy_contributions": contributions,
        "line_box_ratios": ratios,
        "warnings": [
            "The original CSV time column is a legacy step-index alias. Physical time is reconstructed as step_index*dt."
        ]
        if any(item.time_status == "legacy_step_index_alias" for item in inspections)
        else [],
    }


def run_receiver_aperture_comparison(
    config: ApertureComparisonConfig,
    *,
    logger: logging.Logger | None = None,
) -> ApertureComparisonResult:
    start = time.perf_counter()
    inputs = config.inputs.resolved()
    log = logger or logging.getLogger(__name__)

    if inputs.line_manifest is None:
        raise ValueError("line_manifest is required.")
    line_specs = load_receiver_specs(inputs.line_manifest, geometry="line")
    specs_by_geometry: dict[str, Mapping[str, ReceiverSpec]] = {"line": line_specs}
    log.info("Loaded line receiver centres from vector-array manifest.")

    line_frame, line_inspection = load_and_validate_campaign(
        inputs.line_residuals,
        geometry="line",
        specs=line_specs,
        config=config,
    )
    inspections: tuple[CampaignInspection, ...] = (line_inspection,)
    frames = [line_frame]

    if config.geometry == "line-box":
        if inputs.box_manifest is None or inputs.box_residuals is None:
            raise ValueError(
                "box_manifest and box_residuals are required in line-box mode."
            )
        box_specs = load_receiver_specs(inputs.box_manifest, geometry="box")
        specs_by_geometry["box"] = box_specs
        box_frame, box_inspection = load_and_validate_campaign(
            inputs.box_residuals,
            geometry="box",
            specs=box_specs,
            config=config,
        )
        frames.append(box_frame)
        inspections = (line_inspection, box_inspection)
        log.info(
            "Validated residual CSVs: line=%d rows, box=%d rows.",
            line_frame.shape[0],
            box_frame.shape[0],
        )
    else:
        log.info("Validated line residual CSV: %d rows.", line_frame.shape[0])

    conventions = {item.residual_convention for item in inspections}
    if len(conventions) != 1:
        raise ResidualConventionError(
            f"Residual conventions differ across campaigns: {sorted(conventions)}."
        )

    canonical = pd.concat(frames, ignore_index=True)
    receiver_geometry = build_receiver_geometry(
        specs_by_geometry,
        geometry_atol=config.geometry_atol,
        corner_policy=config.corner_policy,
    )
    canonical = attach_receiver_geometry(canonical, receiver_geometry)
    log.info("Built deterministic receiver geometry and trapezoidal weights.")

    rms_by_component = compute_receiver_rms_by_component(canonical)
    rms_vector = compute_receiver_rms_vector(rms_by_component, receiver_geometry)
    side_summary = compute_aperture_summaries(
        rms_vector,
        dt=config.dt,
        ratio_epsilon=config.ratio_epsilon,
    )

    if config.geometry == "line-box":
        line_box_summary = build_line_box_summary(
            side_summary,
            ratio_epsilon=config.ratio_epsilon,
        )
        common = validate_common_apertures(
            canonical,
            receiver_geometry,
            atol=config.value_atol,
            rtol=config.value_rtol,
        )
    else:
        line_box_summary = pd.DataFrame()
        common = pd.DataFrame()

    time_validation = build_time_validation_summary(
        inspections,
        dt=config.dt,
        time_atol=config.time_atol,
        time_rtol=config.time_rtol,
        time_column_policy=config.time_column_policy,
    )
    representatives = select_representative_receivers(receiver_geometry)
    representative_signals = build_representative_signals(canonical, representatives)
    summary = _build_scientific_summary(
        inspections,
        side_summary,
        line_box_summary,
        common,
        analysis_geometry=config.geometry,
    )
    elapsed = float(time.perf_counter() - start)
    log.info(
        "Computed %s receiver and aperture metrics in %.6f s.",
        "line-only" if config.geometry == "line" else "line/box",
        elapsed,
    )
    return ApertureComparisonResult(
        inspections=inspections,
        canonical_residuals=canonical,
        receiver_geometry=receiver_geometry,
        rms_by_component=rms_by_component,
        rms_vector=rms_vector,
        rms_by_side=side_summary,
        line_box_summary=line_box_summary,
        time_validation=time_validation,
        common_aperture_consistency=common,
        representative_signals=representative_signals,
        summary=summary,
        elapsed_seconds=elapsed,
    )

def _write_dataframe(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "matplotlib"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = None
    return versions


def _write_latex_table(path: Path, summary: pd.DataFrame) -> Path:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison of line and box receiver apertures. The weighted RMS uses polyline trapezoidal quadrature; $E_h$ denotes the discrete time--space squared-residual integral.}",
        r"\label{tab:receiver-aperture-comparison}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Family & $N_{r,L}$ & $N_{r,B}$ & $\mathrm{RMS}_{w,L}$ & $\mathrm{RMS}_{w,B}$ & $\mathrm{RMS}_{w,B}/\mathrm{RMS}_{w,L}$ & $E_{h,B}/E_{h,L}$ \\",
        r"\midrule",
    ]
    labels = {"reflection": "Reflection", "transmission": "Transmission"}
    for _, row in summary.iterrows():
        lines.append(
            "{} & {} & {} & {:.4e} & {:.4e} & {:.4f} & {:.4f} \\\\".format(
                labels.get(str(row["residual_family"]), str(row["residual_family"])),
                int(row["line_num_receivers"]),
                int(row["box_num_receivers"]),
                float(row["line__rms_spatially_weighted"]),
                float(row["box_all__rms_spatially_weighted"]),
                float(row["box_all_over_line__rms_spatially_weighted"]),
                float(row["box_all_over_line__discrete_energy_dt_space"]),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_line_latex_table(path: Path, summary: pd.DataFrame) -> Path:
    line = summary.loc[
        (summary["geometry"] == "line") & (summary["scope"] == "all")
    ].sort_values("residual_family", kind="mergesort")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Line-only receiver-aperture residual metrics. The weighted RMS uses segment trapezoidal quadrature; $E_h$ denotes the discrete time--space squared-residual integral.}",
        r"\label{tab:receiver-aperture-line-only}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Family & $N_r$ & $L$ & $\mathrm{RMS}_{w}$ & $\mathrm{RMS}_{w,x}$ & $\mathrm{RMS}_{w,y}$ & $E_h$ \\",
        r"\midrule",
    ]
    labels = {"reflection": "Reflection", "transmission": "Transmission"}
    for _, row in line.iterrows():
        lines.append(
            "{} & {} & {:.4g} & {:.4e} & {:.4e} & {:.4e} & {:.4e} \\\\".format(
                labels.get(str(row["residual_family"]), str(row["residual_family"])),
                int(row["num_receivers"]),
                float(row["aperture_length"]),
                float(row["rms_spatially_weighted"]),
                float(row["rms_x_spatially_weighted"]),
                float(row["rms_y_spatially_weighted"]),
                float(row["discrete_energy_dt_space"]),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _plot_receiver_rms(
    path: Path,
    rms_vector: pd.DataFrame,
    *,
    family: str,
) -> Path:
    plt = _configure_matplotlib()
    available = set(rms_vector["geometry"].astype(str))
    geometries = [geometry for geometry in ALLOWED_GEOMETRIES if geometry in available]
    if not geometries:
        raise ReceiverApertureError(
            f"No receiver RMS data available for family={family!r}."
        )
    figure, axes = plt.subplots(
        1,
        len(geometries),
        figsize=(6.2 * len(geometries), 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    for column, geometry in enumerate(geometries):
        axis = axes[0, column]
        subset = rms_vector.loc[
            (rms_vector["geometry"] == geometry)
            & (rms_vector["residual_family"] == family)
        ].sort_values("curvilinear_s_all")
        axis.plot(
            subset["curvilinear_s_all"],
            subset["rms_vector_xy"],
            marker="o",
        )
        axis.set_title(f"{geometry} aperture — {family}")
        axis.set_xlabel("Curvilinear position")
        axis.set_ylabel("Vector RMS")
        axis.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path

def _plot_box_sides(path: Path, rms_vector: pd.DataFrame) -> Path:
    plt = _configure_matplotlib()
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    box = rms_vector.loc[
        (rms_vector["geometry"] == "box")
        & (rms_vector["residual_family"] == "transmission")
    ]
    for side in ("left", "top", "right"):
        subset = box.loc[box[f"weight_{side}"] > 0.0].sort_values(f"s_{side}")
        axis.plot(
            subset[f"s_{side}"],
            subset["rms_vector_xy"],
            marker="o",
            label=side,
        )
    axis.set_xlabel("Local side coordinate")
    axis.set_ylabel("Vector RMS")
    axis.set_title("Box transmission residual by receiver side")
    axis.legend()
    axis.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _plot_representative_signals(
    path: Path,
    representative_signals: pd.DataFrame,
    *,
    family: str,
) -> Path:
    plt = _configure_matplotlib()
    subset = representative_signals.loc[
        representative_signals["residual_family"] == family
    ]
    geometries = [g for g in ALLOWED_GEOMETRIES if g in set(subset["geometry"])]
    figure, axes = plt.subplots(
        len(geometries),
        1,
        figsize=(11, 4.2 * len(geometries)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, geometry in enumerate(geometries):
        axis = axes[row_index, 0]
        geometry_data = subset.loc[subset["geometry"] == geometry]
        for (selection_rule, component), series in geometry_data.groupby(
            ["selection_rule", "component"], sort=True, observed=True
        ):
            axis.plot(
                series["time_physical"],
                series["residual"],
                label=f"{selection_rule}/{component}",
            )
        axis.set_title(f"{geometry} — {family}")
        axis.set_xlabel("Physical time")
        axis.set_ylabel("Residual")
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=2, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _plot_summary(path: Path, side_summary: pd.DataFrame) -> Path:
    plt = _configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    scopes = [
        ("line", "all", "line-all"),
        ("box", "all", "box-all"),
        ("box", "left", "box-left"),
        ("box", "top", "box-top"),
        ("box", "right", "box-right"),
    ]
    for axis, family in zip(axes, ("reflection", "transmission"), strict=True):
        labels: list[str] = []
        rms_values: list[float] = []
        for geometry, scope, label in scopes:
            match = side_summary.loc[
                (side_summary["geometry"] == geometry)
                & (side_summary["residual_family"] == family)
                & (side_summary["scope"] == scope)
            ]
            if match.empty:
                continue
            labels.append(label)
            rms_values.append(float(match.iloc[0]["rms_spatially_weighted"]))
        positions = np.arange(len(labels))
        axis.bar(positions, rms_values)
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.set_ylabel("Spatially weighted RMS")
        axis.set_title(family)
        axis.grid(True, axis="y", alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def write_receiver_aperture_outputs(
    result: ApertureComparisonResult,
    config: ApertureComparisonConfig,
    *,
    command: Sequence[str],
    repository_root: Path,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    log = logger or logging.getLogger(__name__)
    paths = OutputPaths.from_root(config.output_dir)
    # The CLI creates output directories before the logger. This method is also
    # public, so preserve a non-empty log directory while replacing data outputs.
    if paths.root.exists() and config.overwrite:
        for directory in (
            paths.csv_dir,
            paths.figures_dir,
            paths.tables_dir,
            paths.manifests_dir,
        ):
            if directory.exists():
                shutil.rmtree(directory)
    for directory in (
        paths.root,
        paths.csv_dir,
        paths.figures_dir,
        paths.tables_dir,
        paths.manifests_dir,
        paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    outputs["receiver_rms_by_component"] = _write_dataframe(
        paths.csv_dir / "receiver_rms_by_component.csv",
        result.rms_by_component,
    )
    outputs["receiver_rms_vector"] = _write_dataframe(
        paths.csv_dir / "receiver_rms_vector.csv",
        result.rms_vector,
    )
    outputs["receiver_rms_by_side"] = _write_dataframe(
        paths.csv_dir / "receiver_rms_by_side.csv",
        result.rms_by_side,
    )
    outputs["time_validation_summary"] = _write_dataframe(
        paths.csv_dir / "time_validation_summary.csv",
        result.time_validation,
    )
    outputs["receiver_geometry_summary"] = _write_dataframe(
        paths.csv_dir / "receiver_geometry_summary.csv",
        result.receiver_geometry,
    )
    outputs["representative_signals"] = _write_dataframe(
        paths.csv_dir / "representative_signals.csv",
        result.representative_signals,
    )

    if config.geometry == "line-box":
        outputs["line_box_summary"] = _write_dataframe(
            paths.csv_dir / "line_box_summary.csv",
            result.line_box_summary,
        )
        outputs["common_aperture_consistency"] = _write_dataframe(
            paths.csv_dir / "common_aperture_consistency.csv",
            result.common_aperture_consistency,
        )
        outputs["latex_table"] = _write_latex_table(
            paths.tables_dir / "line_box_summary_table.tex",
            result.line_box_summary,
        )
    else:
        line_summary = result.rms_by_side.loc[
            (result.rms_by_side["geometry"] == "line")
            & (result.rms_by_side["scope"] == "all")
        ].sort_values("residual_family", kind="mergesort", ignore_index=True)
        outputs["line_aperture_summary"] = _write_dataframe(
            paths.csv_dir / "line_aperture_summary.csv",
            line_summary,
        )
        outputs["latex_table"] = _write_line_latex_table(
            paths.tables_dir / "line_aperture_summary_table.tex",
            result.rms_by_side,
        )

    outputs["figure_rms_reflection"] = _plot_receiver_rms(
        paths.figures_dir / "receiver_rms_reflection.png",
        result.rms_vector,
        family="reflection",
    )
    outputs["figure_rms_transmission"] = _plot_receiver_rms(
        paths.figures_dir / "receiver_rms_transmission.png",
        result.rms_vector,
        family="transmission",
    )
    if config.geometry == "line-box":
        outputs["figure_box_sides"] = _plot_box_sides(
            paths.figures_dir / "box_transmission_rms_by_side.png",
            result.rms_vector,
        )
    outputs["figure_signals_reflection"] = _plot_representative_signals(
        paths.figures_dir / "representative_signals_reflection.png",
        result.representative_signals,
        family="reflection",
    )
    outputs["figure_signals_transmission"] = _plot_representative_signals(
        paths.figures_dir / "representative_signals_transmission.png",
        result.representative_signals,
        family="transmission",
    )
    summary_figure_name = (
        "line_weighted_rms_summary.png"
        if config.geometry == "line"
        else "line_box_weighted_rms_summary.png"
    )
    outputs["figure_summary"] = _plot_summary(
        paths.figures_dir / summary_figure_name,
        result.rms_by_side,
    )

    summary_payload = dict(result.summary)
    summary_payload.update(
        {
            "elapsed_seconds": result.elapsed_seconds,
            "num_receiver_component_rows": int(result.rms_by_component.shape[0]),
            "num_receiver_vector_rows": int(result.rms_vector.shape[0]),
            "num_side_summary_rows": int(result.rms_by_side.shape[0]),
            "num_line_box_summary_rows": int(result.line_box_summary.shape[0]),
            "num_common_aperture_checks": int(
                result.common_aperture_consistency.shape[0]
            ),
        }
    )
    outputs["summary_json"] = _write_json(
        paths.manifests_dir / "receiver_aperture_comparison_summary.json",
        summary_payload,
    )

    inputs = config.inputs.resolved()
    if inputs.line_manifest is None:
        raise ValueError("line_manifest is required.")
    input_payload: dict[str, Any] = {
        "line_residuals": _file_metadata(inputs.line_residuals),
        "line_manifest": _file_metadata(inputs.line_manifest),
    }
    if config.geometry == "line-box":
        if inputs.box_residuals is None or inputs.box_manifest is None:
            raise ValueError(
                "box_residuals and box_manifest are required in line-box mode."
            )
        input_payload.update(
            {
                "box_residuals": _file_metadata(inputs.box_residuals),
                "box_manifest": _file_metadata(inputs.box_manifest),
            }
        )

    corner_contract = (
        "shared side endpoint half-weights; side weights sum to full U-aperture weights"
        if config.geometry == "line-box"
        else "not applicable to line-only segment apertures"
    )
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "receiver_aperture_comparison_manifest",
        "status": "ok",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "analysis_geometry": config.geometry,
        "analysis_mode": "line-only" if config.geometry == "line" else "line-box",
        "inputs": input_payload,
        "parameters": {
            "geometry": config.geometry,
            "dt": config.dt,
            "time_atol": config.time_atol,
            "time_rtol": config.time_rtol,
            "value_atol": config.value_atol,
            "value_rtol": config.value_rtol,
            "geometry_atol": config.geometry_atol,
            "ratio_epsilon": config.ratio_epsilon,
            "time_column_policy": config.time_column_policy,
            "corner_policy": config.corner_policy,
        },
        "contracts": {
            "unique_key": [
                "geometry",
                "array_name",
                "role",
                "residual_family",
                "receiver_id",
                "component",
                "step_index",
            ],
            "time_physical": "step_index * dt",
            "residual_convention": result.inspections[0].residual_convention,
            "receiver_centres_source": "vector_array_manifest.extra.comparison.observed_{reflection,transmission}.spec.array.centers",
            "weighting": "trapezoidal quadrature over ordered manifest centres",
            "corner_policy": corner_contract,
        },
        "metric_definitions": {
            "receiver_component_rms": "sqrt(mean_n(residual[r,c,n]^2))",
            "receiver_vector_rms": "sqrt(rms_x^2 + rms_y^2)",
            "rms_mean_by_receiver": "mean_r(receiver_vector_rms)",
            "rms_global_sum": "sqrt(sum_r(receiver_vector_rms^2))",
            "rms_global_normalized": "sqrt(mean_r(receiver_vector_rms^2))",
            "rms_spatially_weighted": "sqrt(sum_r(w_r*receiver_vector_rms^2)/sum_r(w_r))",
            "discrete_energy_dt_space": "dt*sum_r(w_r*sum_n,c(residual^2))",
            "side_energy_contribution": (
                "side discrete energy / box-all discrete energy"
                if config.geometry == "line-box"
                else "not applicable"
            ),
            "signed_balance": "(positive_L1-negative_L1_abs)/(positive_L1+negative_L1_abs)",
        },
        "validation": {
            "campaign_inspections": [item.as_dict() for item in result.inspections],
            "common_aperture_consistency": (
                result.common_aperture_consistency.to_dict("records")
                if config.geometry == "line-box"
                else []
            ),
            "common_aperture_status": (
                "ok" if config.geometry == "line-box" else "not_applicable"
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": _dependency_versions(),
            "git_commit": _git_commit(repository_root),
        },
        "log_path": str(paths.logs_dir / "receiver_aperture_comparison.log"),
        "outputs": {},
    }

    # Add output hashes after all non-manifest files have been written.
    manifest_payload["outputs"] = {
        name: _file_metadata(path) for name, path in sorted(outputs.items())
    }
    outputs["manifest_json"] = _write_json(
        paths.manifests_dir / "receiver_aperture_comparison_manifest.json",
        manifest_payload,
    )
    log.info("Wrote %d receiver-aperture output files.", len(outputs))
    return outputs
