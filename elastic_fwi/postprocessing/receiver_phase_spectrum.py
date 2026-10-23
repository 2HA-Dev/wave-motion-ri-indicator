from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FORMAT_VERSION = "receiver_phase_spectrum_v2"
SUPPORTED_COMPONENTS = ("x", "y")
SUPPORTED_WINDOWS = ("hann", "none")


class ReceiverTraceFormatError(ValueError):
    """Raised when receiver traces do not follow a supported repository schema."""


class ReceiverTraceAlignmentError(ValueError):
    """Raised when reference and observed traces are not temporally aligned."""


def _scalar_string(value: Any) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    if array.size == 1:
        return str(array.reshape(-1)[0])
    return str(value)


def _finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ReceiverTraceFormatError(
            f"{name} must be {ndim}-dimensional, got shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ReceiverTraceFormatError(f"{name} contains NaN or Inf values.")
    return array


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _axial_angle_difference_radians(observed: float, reference: float) -> float:
    """Return an orientation difference modulo pi, in [-pi/2, pi/2)."""

    return float((observed - reference + 0.5 * np.pi) % np.pi - 0.5 * np.pi)


def _normalise_components(raw: Sequence[Any], count: int) -> tuple[str, ...]:
    values = tuple(str(item) for item in raw)
    if len(values) != count:
        raise ReceiverTraceFormatError(
            f"components length {len(values)} does not match component count {count}."
        )
    if len(set(values)) != len(values):
        raise ReceiverTraceFormatError("components must be unique.")
    return values


@dataclass(frozen=True)
class ReceiverTracePair:
    """Aligned reference/observed receiver traces.

    Values use the repository convention ``(num_steps, num_receivers,
    num_components)``. Receiver coordinates may be unavailable for standalone
    residual NPZ files; in that case ``centers`` contains NaN values.
    """

    case_name: str
    geometry_kind: str
    residual_family: str
    times: np.ndarray
    reference_values: np.ndarray
    observed_values: np.ndarray
    receiver_ids: np.ndarray
    components: tuple[str, ...]
    centers: np.ndarray
    source_paths: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        times = _finite_array(self.times, name="times", ndim=1)
        reference = _finite_array(
            self.reference_values, name="reference_values", ndim=3
        )
        observed = _finite_array(
            self.observed_values, name="observed_values", ndim=3
        )
        receiver_ids = np.asarray(self.receiver_ids, dtype=np.int64)
        centers = np.asarray(self.centers, dtype=np.float64)

        if reference.shape != observed.shape:
            raise ReceiverTraceAlignmentError(
                "reference_values and observed_values must have identical shapes."
            )
        if reference.shape[0] != times.size:
            raise ReceiverTraceAlignmentError(
                "The time axis length does not match the trace length."
            )
        if receiver_ids.shape != (reference.shape[1],):
            raise ReceiverTraceFormatError(
                "receiver_ids must have shape (num_receivers,)."
            )
        if len(np.unique(receiver_ids)) != receiver_ids.size:
            raise ReceiverTraceFormatError("receiver_ids must be unique.")
        if centers.shape != (reference.shape[1], 2):
            raise ReceiverTraceFormatError(
                "centers must have shape (num_receivers, 2)."
            )
        if not np.all(np.isfinite(centers) | np.isnan(centers)):
            raise ReceiverTraceFormatError("centers contains invalid values.")
        if len(self.components) != reference.shape[2]:
            raise ReceiverTraceFormatError(
                "components does not match the last trace dimension."
            )
        if times.size < 4:
            raise ReceiverTraceAlignmentError(
                "At least four time samples are required for spectral analysis."
            )
        increments = np.diff(times)
        if np.any(increments <= 0.0):
            raise ReceiverTraceAlignmentError(
                "The time axis must be strictly increasing."
            )
        median_dt = float(np.median(increments))
        tolerance = max(1.0e-12, 1.0e-10 * abs(median_dt))
        if not np.allclose(increments, median_dt, rtol=1.0e-9, atol=tolerance):
            raise ReceiverTraceAlignmentError(
                "The time axis must be uniformly sampled for FFT analysis."
            )

        object.__setattr__(self, "times", times)
        object.__setattr__(self, "reference_values", reference)
        object.__setattr__(self, "observed_values", observed)
        object.__setattr__(self, "receiver_ids", receiver_ids)
        object.__setattr__(self, "centers", centers)

    @property
    def num_steps(self) -> int:
        return int(self.reference_values.shape[0])

    @property
    def num_receivers(self) -> int:
        return int(self.reference_values.shape[1])

    @property
    def num_components(self) -> int:
        return int(self.reference_values.shape[2])

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.times)))

    @property
    def sampling_frequency(self) -> float:
        return float(1.0 / self.dt)

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0])

    @property
    def final_time(self) -> float:
        return float(self.times[-1])


@dataclass(frozen=True)
class SpectrumConfig:
    """Configuration for receiver-space spectral and phase analysis."""

    expected_source_frequency: float | None = None
    source_frequency_tolerance: float = 0.10
    source_num_periods: float = 1.0
    source_t0: float = 0.0
    source_band_amplitude_threshold: float = 0.05
    phase_amplitude_threshold: float = 0.05
    significant_amplitude_threshold: float = 0.10
    amplitude_ratio_reference_floor_relative: float = 1.0e-4
    amplitude_ratio_reference_floor_absolute: float = 1.0e-30
    window: str = "hann"
    detrend: bool = True
    correlation_epsilon: float = 1.0e-30
    max_lag_seconds: float | None = None
    max_plot_frequency: float | None = None

    def __post_init__(self) -> None:
        if self.window not in SUPPORTED_WINDOWS:
            raise ValueError(
                f"window must be one of {SUPPORTED_WINDOWS}, got {self.window!r}."
            )
        if self.expected_source_frequency is not None:
            if not np.isfinite(self.expected_source_frequency) or self.expected_source_frequency <= 0:
                raise ValueError("expected_source_frequency must be positive and finite.")
        if self.source_frequency_tolerance < 0:
            raise ValueError("source_frequency_tolerance must be non-negative.")
        if not np.isfinite(self.source_num_periods) or self.source_num_periods <= 0.0:
            raise ValueError("source_num_periods must be positive and finite.")
        if not np.isfinite(self.source_t0):
            raise ValueError("source_t0 must be finite.")
        for name, value in (
            ("source_band_amplitude_threshold", self.source_band_amplitude_threshold),
            ("phase_amplitude_threshold", self.phase_amplitude_threshold),
            ("significant_amplitude_threshold", self.significant_amplitude_threshold),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must belong to (0, 1].")
        if (
            not np.isfinite(self.amplitude_ratio_reference_floor_relative)
            or self.amplitude_ratio_reference_floor_relative < 0.0
        ):
            raise ValueError(
                "amplitude_ratio_reference_floor_relative must be finite and non-negative."
            )
        if (
            not np.isfinite(self.amplitude_ratio_reference_floor_absolute)
            or self.amplitude_ratio_reference_floor_absolute <= 0.0
        ):
            raise ValueError(
                "amplitude_ratio_reference_floor_absolute must be positive and finite."
            )
        if self.correlation_epsilon <= 0:
            raise ValueError("correlation_epsilon must be positive.")
        if self.max_lag_seconds is not None and self.max_lag_seconds < 0:
            raise ValueError("max_lag_seconds must be non-negative.")
        if self.max_plot_frequency is not None and self.max_plot_frequency <= 0:
            raise ValueError("max_plot_frequency must be positive.")


@dataclass
class CaseAnalysis:
    pair: ReceiverTracePair
    config: SpectrumConfig
    frequencies: np.ndarray
    reference_amplitudes: np.ndarray
    observed_amplitudes: np.ndarray
    reference_phases: np.ndarray
    observed_phases: np.ndarray
    phase_differences: np.ndarray
    unwrapped_phase_differences: np.ndarray
    phase_reliable: np.ndarray
    source_signal: np.ndarray | None
    source_amplitude: np.ndarray | None
    source_psd: np.ndarray | None
    metrics_rows: list[dict[str, Any]]
    polarization_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def validate_time_axis(
    times: np.ndarray,
    *,
    expected_dt: float | None = None,
    expected_final_time: float | None = None,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-9,
) -> dict[str, Any]:
    """Validate uniform receiver sampling and optional campaign conventions."""

    time_values = _finite_array(times, name="times", ndim=1)
    if time_values.size < 2:
        raise ReceiverTraceAlignmentError("At least two times are required.")
    differences = np.diff(time_values)
    if np.any(differences <= 0.0):
        raise ReceiverTraceAlignmentError("times must be strictly increasing.")
    dt = float(np.median(differences))
    uniform = bool(np.allclose(differences, dt, atol=atol, rtol=rtol))
    if not uniform:
        raise ReceiverTraceAlignmentError("times are not uniformly sampled.")

    dt_matches = (
        None
        if expected_dt is None
        else bool(np.isclose(dt, expected_dt, atol=atol, rtol=rtol))
    )
    final_matches = (
        None
        if expected_final_time is None
        else bool(
            np.isclose(
                float(time_values[-1]),
                expected_final_time,
                atol=atol,
                rtol=rtol,
            )
        )
    )
    return {
        "num_steps": int(time_values.size),
        "first_time": float(time_values[0]),
        "last_time": float(time_values[-1]),
        "dt": dt,
        "sampling_frequency": float(1.0 / dt),
        "nyquist_frequency": float(0.5 / dt),
        "duration": float(time_values[-1] - time_values[0]),
        "uniform_sampling": uniform,
        "expected_dt": expected_dt,
        "dt_matches_expected": dt_matches,
        "expected_final_time": expected_final_time,
        "final_time_matches_expected": final_matches,
        "status": "ok"
        if (dt_matches is not False and final_matches is not False)
        else "validation_warning",
    }


def _load_centers_csv(path: Path, *, array_name: str, role: str) -> dict[int, tuple[float, float]]:
    if not path.is_file():
        return {}
    result: dict[int, tuple[float, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"receiver_id", "x", "y"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            return {}
        for row in reader:
            if "array" in row and array_name and row.get("array") != array_name:
                continue
            if "role" in row and role and row.get("role") != role:
                continue
            receiver_id = int(row["receiver_id"])
            result[receiver_id] = (float(row["x"]), float(row["y"]))
    return result


def load_residual_npz(
    path: str | Path,
    *,
    case_name: str | None = None,
    geometry_kind: str = "unknown",
    centers_csv: str | Path | None = None,
) -> ReceiverTracePair:
    """Load the repository ``vector_array_residual_series`` NPZ format."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing residual NPZ: {source}")

    with np.load(source, allow_pickle=False) as data:
        required = {
            "kind",
            "times",
            "reference_values",
            "observed_values",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ReceiverTraceFormatError(
                f"Missing required NPZ keys in {source}: {missing}."
            )
        kind = _scalar_string(data["kind"])
        if kind != "vector_array_residual_series":
            raise ReceiverTraceFormatError(
                f"Expected vector_array_residual_series in {source}, got {kind!r}."
            )
        times = _finite_array(data["times"], name=f"{source}:times", ndim=1)
        reference = _finite_array(
            data["reference_values"], name=f"{source}:reference_values", ndim=3
        )
        observed = _finite_array(
            data["observed_values"], name=f"{source}:observed_values", ndim=3
        )
        components = _normalise_components(
            data["components"] if "components" in data.files else SUPPORTED_COMPONENTS,
            reference.shape[2],
        )
        label = _scalar_string(data["label"]) if "label" in data.files else source.stem
        array_name = _scalar_string(data["array_name"]) if "array_name" in data.files else ""
        role = _scalar_string(data["role"]) if "role" in data.files else ""
        residual_family = label
        if residual_family not in {"reflection", "transmission"}:
            if "reflection" in source.stem:
                residual_family = "reflection"
            elif "transmission" in source.stem:
                residual_family = "transmission"

    receiver_ids = np.arange(reference.shape[1], dtype=np.int64)
    centers = np.full((reference.shape[1], 2), np.nan, dtype=np.float64)
    if centers_csv is not None:
        coordinate_map = _load_centers_csv(
            Path(centers_csv).expanduser().resolve(),
            array_name=array_name,
            role=role,
        )
        for index, receiver_id in enumerate(receiver_ids):
            if int(receiver_id) in coordinate_map:
                centers[index] = coordinate_map[int(receiver_id)]

    return ReceiverTracePair(
        case_name=case_name or source.stem,
        geometry_kind=str(geometry_kind),
        residual_family=str(residual_family),
        times=times,
        reference_values=reference,
        observed_values=observed,
        receiver_ids=receiver_ids,
        components=components,
        centers=centers,
        source_paths=(str(source),),
        metadata={
            "source_format": "vector_array_residual_series_npz",
            "kind": kind,
            "label": label,
            "array_name": array_name,
            "role": role,
            "centers_csv": None if centers_csv is None else str(centers_csv),
        },
    )


def _load_observation_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"kind", "times", "values"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ReceiverTraceFormatError(
                f"Missing required observation NPZ keys in {path}: {missing}."
            )
        kind = _scalar_string(data["kind"])
        allowed = {
            "vector_array_observation_time_series",
            "direct_receiver_vector_array_time_series",
        }
        if kind not in allowed:
            raise ReceiverTraceFormatError(
                f"Unsupported observation NPZ kind in {path}: {kind!r}."
            )
        times = _finite_array(data["times"], name=f"{path}:times", ndim=1)
        values = _finite_array(data["values"], name=f"{path}:values", ndim=3)
        components = _normalise_components(
            data["components"] if "components" in data.files else SUPPORTED_COMPONENTS,
            values.shape[2],
        )
        centers = (
            _finite_array(data["centers"], name=f"{path}:centers", ndim=2)
            if "centers" in data.files
            else np.full((values.shape[1], 2), np.nan, dtype=np.float64)
        )
        return {
            "kind": kind,
            "times": times,
            "values": values,
            "components": components,
            "centers": centers,
            "label": _scalar_string(data["label"]) if "label" in data.files else path.stem,
            "array_name": _scalar_string(data["array_name"])
            if "array_name" in data.files
            else "",
            "role": _scalar_string(data["role"]) if "role" in data.files else "",
        }


def load_observation_npz_pair(
    reference_path: str | Path,
    observed_path: str | Path,
    *,
    case_name: str,
    geometry_kind: str,
    residual_family: str,
) -> ReceiverTracePair:
    """Load two repository vector-array observation NPZ files."""

    reference_source = Path(reference_path).expanduser().resolve()
    observed_source = Path(observed_path).expanduser().resolve()
    if not reference_source.is_file():
        raise FileNotFoundError(f"Missing reference traces: {reference_source}")
    if not observed_source.is_file():
        raise FileNotFoundError(f"Missing observed traces: {observed_source}")
    reference = _load_observation_npz(reference_source)
    observed = _load_observation_npz(observed_source)
    if reference["values"].shape != observed["values"].shape:
        raise ReceiverTraceAlignmentError(
            "Reference and observed NPZ files have different value shapes."
        )
    if not np.allclose(reference["times"], observed["times"], atol=1e-12, rtol=1e-10):
        raise ReceiverTraceAlignmentError(
            "Reference and observed NPZ files have incompatible time axes."
        )
    if reference["components"] != observed["components"]:
        raise ReceiverTraceAlignmentError(
            "Reference and observed NPZ files have different components."
        )
    if not np.allclose(
        reference["centers"], observed["centers"], equal_nan=True, atol=1e-12, rtol=0
    ):
        raise ReceiverTraceAlignmentError(
            "Reference and observed NPZ files have different receiver centers."
        )
    return ReceiverTracePair(
        case_name=case_name,
        geometry_kind=geometry_kind,
        residual_family=residual_family,
        times=reference["times"],
        reference_values=reference["values"],
        observed_values=observed["values"],
        receiver_ids=np.arange(reference["values"].shape[1], dtype=np.int64),
        components=reference["components"],
        centers=reference["centers"],
        source_paths=(str(reference_source), str(observed_source)),
        metadata={
            "source_format": "paired_vector_array_observation_npz",
            "reference_kind": reference["kind"],
            "observed_kind": observed["kind"],
            "reference_label": reference["label"],
            "observed_label": observed["label"],
            "array_name": reference["array_name"],
            "role": reference["role"],
        },
    )


def load_residual_csv(
    path: str | Path,
    *,
    case_name: str | None = None,
    geometry_kind: str = "unknown",
    residual_family: str | None = None,
    centers_csv: str | Path | None = None,
) -> ReceiverTracePair:
    """Load the repository long vector-array residual CSV format."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("pandas is required to load receiver CSV files.") from exc

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing residual CSV: {source}")
    frame = pd.read_csv(source)
    required = {
        "time",
        "receiver_id",
        "component",
        "reference",
        "observed",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ReceiverTraceFormatError(
            f"Missing required residual CSV columns in {source}: {missing}."
        )
    if residual_family is not None:
        if "role" in frame.columns:
            family_mask = frame["role"].astype(str).str.contains(
                residual_family, case=False, regex=False
            )
            if family_mask.any():
                frame = frame.loc[family_mask].copy()
        elif "array" in frame.columns:
            target = "gamma1" if residual_family == "reflection" else "gamma2"
            frame = frame.loc[frame["array"].astype(str) == target].copy()
    if frame.empty:
        raise ReceiverTraceFormatError(
            f"No rows selected from residual CSV {source}."
        )
    for column in ("time", "reference", "observed"):
        values = frame[column].to_numpy(dtype=np.float64, copy=False)
        if not np.all(np.isfinite(values)):
            raise ReceiverTraceFormatError(f"Column {column} contains NaN or Inf.")

    times = np.sort(frame["time"].unique().astype(np.float64))
    receiver_ids = np.sort(frame["receiver_id"].unique().astype(np.int64))
    components = tuple(sorted(frame["component"].astype(str).unique()))
    shape = (times.size, receiver_ids.size, len(components))
    reference_values = np.full(shape, np.nan, dtype=np.float64)
    observed_values = np.full(shape, np.nan, dtype=np.float64)
    time_index = {float(value): index for index, value in enumerate(times)}
    receiver_index = {int(value): index for index, value in enumerate(receiver_ids)}
    component_index = {value: index for index, value in enumerate(components)}

    keys_seen: set[tuple[int, int, int]] = set()
    for row in frame.itertuples(index=False):
        key = (
            time_index[float(getattr(row, "time"))],
            receiver_index[int(getattr(row, "receiver_id"))],
            component_index[str(getattr(row, "component"))],
        )
        if key in keys_seen:
            raise ReceiverTraceFormatError(
                f"Duplicate time/receiver/component rows in {source}."
            )
        keys_seen.add(key)
        reference_values[key] = float(getattr(row, "reference"))
        observed_values[key] = float(getattr(row, "observed"))
    if not np.all(np.isfinite(reference_values)) or not np.all(np.isfinite(observed_values)):
        raise ReceiverTraceFormatError(
            f"Incomplete receiver/component grid in {source}."
        )

    array_name = str(frame["array"].iloc[0]) if "array" in frame.columns else ""
    role = str(frame["role"].iloc[0]) if "role" in frame.columns else ""
    centers = np.full((receiver_ids.size, 2), np.nan, dtype=np.float64)
    if {"x", "y"}.issubset(frame.columns):
        coordinates = frame.groupby("receiver_id", sort=True)[["x", "y"]].first()
        for index, receiver_id in enumerate(receiver_ids):
            if int(receiver_id) in coordinates.index:
                centers[index] = coordinates.loc[int(receiver_id)].to_numpy(dtype=float)
    elif centers_csv is not None:
        coordinate_map = _load_centers_csv(
            Path(centers_csv).expanduser().resolve(), array_name=array_name, role=role
        )
        for index, receiver_id in enumerate(receiver_ids):
            if int(receiver_id) in coordinate_map:
                centers[index] = coordinate_map[int(receiver_id)]

    family = residual_family or (
        "reflection"
        if "reflection" in role.lower() or array_name == "gamma1"
        else "transmission"
        if "transmission" in role.lower() or array_name == "gamma2"
        else "unknown"
    )
    return ReceiverTracePair(
        case_name=case_name or f"{source.stem}_{family}",
        geometry_kind=geometry_kind,
        residual_family=family,
        times=times,
        reference_values=reference_values,
        observed_values=observed_values,
        receiver_ids=receiver_ids,
        components=components,
        centers=centers,
        source_paths=(str(source),),
        metadata={
            "source_format": "vector_array_residual_long_csv",
            "array_name": array_name,
            "role": role,
            "centers_csv": None if centers_csv is None else str(centers_csv),
        },
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReceiverTraceFormatError(f"Expected a JSON object in {path}.")
    return payload


def _find_inverse_manifest(campaign_dir: Path) -> Path | None:
    inverse_dir = campaign_dir / "inverse"
    candidates = sorted(inverse_dir.glob("inverse_time_series_manifest_rank0000of*.json"))
    if candidates:
        return candidates[0]
    candidate = inverse_dir / "inverse_time_series_manifest.json"
    return candidate if candidate.is_file() else None


def _campaign_context(campaign_dir: Path) -> dict[str, Any]:
    manifest_path = campaign_dir / "gradient_combination_campaign_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing campaign manifest: {manifest_path}")
    payload = _read_json(manifest_path)
    config = payload.get("config", {})
    if not isinstance(config, dict):
        config = {}
    inverse_manifest_path = _find_inverse_manifest(campaign_dir)
    inverse = _read_json(inverse_manifest_path) if inverse_manifest_path else {}
    temporal = inverse.get("temporal_sampling", {})
    if not isinstance(temporal, dict):
        temporal = {}
    dt = float(temporal.get("dt", config.get("dt", np.nan)))
    num_steps = int(temporal.get("num_steps", config.get("num_steps", 0)))
    final_time = float(temporal.get("T_final", dt * num_steps))
    geometry = str(config.get("receiver_array_geometry", "unknown"))
    if "box" in geometry.lower():
        geometry = "box"
    elif geometry not in {"line", "box", "unknown"}:
        geometry = "box" if "box" in campaign_dir.name.lower() else geometry
    return {
        "campaign_manifest": str(manifest_path),
        "inverse_manifest": None
        if inverse_manifest_path is None
        else str(inverse_manifest_path),
        "config": config,
        "dt": dt,
        "num_steps": num_steps,
        "final_time": final_time,
        "source_frequency": config.get("source_frequency"),
        "source_num_periods": config.get("source_num_periods", 1.0),
        "source_t0": config.get("source_t0", 0.0),
        "geometry_kind": geometry,
        "campaign_id": campaign_dir.name,
    }


def load_campaign_trace_pairs(
    campaign_dir: str | Path,
    *,
    residual_families: Sequence[str] = ("reflection", "transmission"),
    require_final_time: float | None = 20.0,
) -> tuple[list[ReceiverTracePair], dict[str, Any]]:
    """Load real vector-array residual NPZ files from one campaign directory."""

    root = Path(campaign_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing campaign directory: {root}")
    context = _campaign_context(root)
    if require_final_time is not None and not np.isclose(
        context["final_time"], require_final_time, atol=1e-12, rtol=0.0
    ):
        raise ReceiverTraceAlignmentError(
            f"Campaign {root.name} has T_final={context['final_time']}, "
            f"expected {require_final_time}."
        )

    vector_dir = root / "inverse" / "vector_array"
    coordinates_csv = vector_dir / "vector_array_coordinates.csv"
    pairs: list[ReceiverTracePair] = []
    missing: list[str] = []
    for family in residual_families:
        path = vector_dir / f"vector_array_{family}_residuals.npz"
        if not path.is_file():
            missing.append(str(path))
            continue
        pair = load_residual_npz(
            path,
            case_name=f"{context['campaign_id']}_{family}",
            geometry_kind=context["geometry_kind"],
            centers_csv=coordinates_csv if coordinates_csv.is_file() else None,
        )
        if np.isfinite(context["dt"]) and not np.isclose(
            pair.dt, context["dt"], atol=1.0e-12, rtol=1.0e-10
        ):
            raise ReceiverTraceAlignmentError(
                f"{path}: trace dt={pair.dt} differs from campaign dt={context['dt']}."
            )
        if context["num_steps"] > 0 and pair.num_steps != context["num_steps"]:
            raise ReceiverTraceAlignmentError(
                f"{path}: trace count={pair.num_steps} differs from campaign "
                f"num_steps={context['num_steps']}."
            )
        if np.isfinite(context["final_time"]) and not np.isclose(
            pair.final_time, context["final_time"], atol=1.0e-12, rtol=1.0e-10
        ):
            raise ReceiverTraceAlignmentError(
                f"{path}: last receiver time={pair.final_time} differs from campaign "
                f"T_final={context['final_time']}."
            )
        metadata = dict(pair.metadata)
        metadata.update(
            {
                "campaign_dir": str(root),
                "campaign_manifest": context["campaign_manifest"],
                "inverse_manifest": context["inverse_manifest"],
                "campaign_dt": context["dt"],
                "campaign_num_steps": context["num_steps"],
                "campaign_final_time": context["final_time"],
                "campaign_source_frequency": context["source_frequency"],
                "campaign_source_num_periods": context["source_num_periods"],
                "campaign_source_t0": context["source_t0"],
            }
        )
        pair = ReceiverTracePair(
            case_name=pair.case_name,
            geometry_kind=pair.geometry_kind,
            residual_family=pair.residual_family,
            times=pair.times,
            reference_values=pair.reference_values,
            observed_values=pair.observed_values,
            receiver_ids=pair.receiver_ids,
            components=pair.components,
            centers=pair.centers,
            source_paths=pair.source_paths,
            metadata=metadata,
        )
        pairs.append(pair)
    context["missing_inputs"] = missing
    context["loaded_families"] = [pair.residual_family for pair in pairs]
    return pairs, context


def _window_values(size: int, name: str) -> np.ndarray:
    if name == "none":
        return np.ones(size, dtype=np.float64)
    if name == "hann":
        return np.hanning(size).astype(np.float64)
    raise ValueError(f"Unsupported window {name!r}.")


def compute_fft_spectrum(
    signal: np.ndarray,
    *,
    dt: float,
    window: str = "hann",
    detrend: bool = True,
    significant_amplitude_threshold: float = 0.10,
) -> dict[str, Any]:
    """Compute a one-sided amplitude spectrum, phase, PSD, and metrics."""

    values = _finite_array(signal, name="signal", ndim=1)
    if values.size < 4:
        raise ValueError("At least four samples are required for an FFT.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite.")
    if not 0.0 < significant_amplitude_threshold <= 1.0:
        raise ValueError("significant_amplitude_threshold must belong to (0, 1].")

    processed = values - float(np.mean(values)) if detrend else values.copy()
    weights = _window_values(values.size, window)
    coherent_gain = float(np.mean(weights))
    if coherent_gain <= 0.0:
        raise ValueError("The spectral window has zero coherent gain.")
    transformed = np.fft.rfft(processed * weights)
    frequencies = np.fft.rfftfreq(values.size, d=float(dt))
    amplitude = np.abs(transformed) / (values.size * coherent_gain)
    if values.size > 1:
        stop = -1 if values.size % 2 == 0 else None
        amplitude[1:stop] *= 2.0
    phase = np.angle(transformed)

    window_energy = float(np.sum(weights * weights))
    psd = np.abs(transformed) ** 2 / (float(1.0 / dt) * window_energy)
    if psd.size > 1:
        stop = -1 if values.size % 2 == 0 else None
        psd[1:stop] *= 2.0
    df = float(frequencies[1] - frequencies[0])
    spectral_energy = float(np.sum(psd) * df)

    candidate_amplitude = amplitude.copy()
    if candidate_amplitude.size > 0:
        candidate_amplitude[0] = 0.0
    dominant_index = int(np.argmax(candidate_amplitude))
    dominant_amplitude = float(candidate_amplitude[dominant_index])
    zero_signal = dominant_amplitude <= np.finfo(np.float64).tiny
    if zero_signal:
        dominant_frequency = None
        dominant_phase = None
        band_min = None
        band_max = None
    else:
        dominant_frequency = float(frequencies[dominant_index])
        dominant_phase = float(phase[dominant_index])
        significant = candidate_amplitude >= (
            significant_amplitude_threshold * dominant_amplitude
        )
        indices = np.flatnonzero(significant)
        band_min = float(frequencies[indices[0]]) if indices.size else None
        band_max = float(frequencies[indices[-1]]) if indices.size else None

    return {
        "frequencies": frequencies,
        "complex_spectrum": transformed,
        "amplitude": amplitude,
        "phase": phase,
        "psd": psd,
        "dominant_index": None if zero_signal else dominant_index,
        "dominant_frequency": dominant_frequency,
        "dominant_amplitude": None if zero_signal else dominant_amplitude,
        "dominant_phase": dominant_phase,
        "significant_band_min": band_min,
        "significant_band_max": band_max,
        "spectral_energy": spectral_energy,
        "frequency_resolution": df,
        "nyquist_frequency": float(frequencies[-1]),
        "window": window,
        "detrended": bool(detrend),
        "status": "zero_signal" if zero_signal else "ok",
    }


def sample_hann_sine_source(
    times: np.ndarray,
    *,
    frequency: float,
    num_periods: float = 1.0,
    t0: float = 0.0,
) -> np.ndarray:
    """Sample the repository Hann-windowed sine source on a time axis."""

    time_values = _finite_array(times, name="source_times", ndim=1)
    frequency = float(frequency)
    num_periods = float(num_periods)
    t0 = float(t0)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency must be positive and finite.")
    if not np.isfinite(num_periods) or num_periods <= 0.0:
        raise ValueError("num_periods must be positive and finite.")
    if not np.isfinite(t0):
        raise ValueError("t0 must be finite.")
    duration = num_periods / frequency
    tau = time_values - t0
    mask = (tau >= 0.0) & (tau <= duration)
    values = np.zeros_like(time_values, dtype=np.float64)
    active_tau = tau[mask]
    values[mask] = (
        np.sin(2.0 * np.pi * frequency * active_tau)
        * np.sin(np.pi * active_tau / duration) ** 2
    )
    return values


def compute_effective_source_spectrum(
    times: np.ndarray,
    *,
    frequency: float,
    num_periods: float = 1.0,
    t0: float = 0.0,
    amplitude_threshold: float = 0.05,
) -> dict[str, Any]:
    """Compute the sampled source spectrum and its effective amplitude band.

    The source pulse is already Hann-windowed in time, so no additional FFT
    window is applied here. The effective band contains bins whose amplitude is
    at least ``amplitude_threshold`` times the source-spectrum peak.
    """

    time_values = _finite_array(times, name="source_times", ndim=1)
    if time_values.size < 4:
        raise ValueError("At least four source samples are required.")
    dt = float(np.median(np.diff(time_values)))
    signal = sample_hann_sine_source(
        time_values,
        frequency=frequency,
        num_periods=num_periods,
        t0=t0,
    )
    spectrum = compute_fft_spectrum(
        signal,
        dt=dt,
        window="none",
        detrend=False,
        significant_amplitude_threshold=amplitude_threshold,
    )
    return {
        **spectrum,
        "signal": signal,
        "source_frequency": float(frequency),
        "source_num_periods": float(num_periods),
        "source_t0": float(t0),
        "source_duration": float(num_periods / frequency),
        "source_band_amplitude_threshold": float(amplitude_threshold),
    }


def classify_frequency_against_source_band(
    frequency: float | None,
    *,
    source_band_min: float | None,
    source_band_max: float | None,
    frequency_resolution: float,
) -> str:
    """Classify a measured dominant frequency against the effective source band."""

    if frequency is None or not np.isfinite(frequency):
        return "missing_frequency"
    if source_band_min is None or source_band_max is None:
        return "missing_source_band"
    tolerance = max(1.0e-12, 0.5 * float(frequency_resolution))
    if frequency < source_band_min - tolerance:
        return "low_frequency_dominated_trace"
    if frequency > source_band_max + tolerance:
        return "high_frequency_dominated_trace"
    return "source_spectrum_compatible"


def _unwrap_masked_phase(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Unwrap contiguous reliable frequency intervals without crossing masked gaps."""

    phase = np.asarray(values, dtype=np.float64)
    reliable = np.asarray(mask, dtype=bool)
    result = np.full(phase.shape, np.nan, dtype=np.float64)
    indices = np.flatnonzero(reliable)
    if indices.size == 0:
        return result
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    for group in np.split(indices, split_points):
        result[group] = np.unwrap(phase[group])
    return result


def compute_phase_difference(
    reference_spectrum: Mapping[str, Any],
    observed_spectrum: Mapping[str, Any],
    *,
    phase_amplitude_threshold: float,
    comparison_frequency: float | None = None,
) -> dict[str, Any]:
    """Compare spectral phases while masking low-amplitude bins."""

    frequencies = np.asarray(reference_spectrum["frequencies"], dtype=np.float64)
    observed_frequencies = np.asarray(
        observed_spectrum["frequencies"], dtype=np.float64
    )
    if frequencies.shape != observed_frequencies.shape or not np.allclose(
        frequencies, observed_frequencies, atol=0.0, rtol=0.0
    ):
        raise ReceiverTraceAlignmentError("Reference and observed FFT axes differ.")
    reference_amplitude = np.asarray(reference_spectrum["amplitude"], dtype=float)
    observed_amplitude = np.asarray(observed_spectrum["amplitude"], dtype=float)
    reference_phase = np.asarray(reference_spectrum["phase"], dtype=float)
    observed_phase = np.asarray(observed_spectrum["phase"], dtype=float)
    reference_peak = float(np.max(reference_amplitude))
    observed_peak = float(np.max(observed_amplitude))
    common_peak = max(reference_peak, observed_peak)
    reliability_floor = phase_amplitude_threshold * common_peak
    reliable = (
        (reference_amplitude >= reliability_floor)
        & (observed_amplitude >= reliability_floor)
        & (reference_amplitude > 0.0)
        & (observed_amplitude > 0.0)
    )
    raw_wrapped = np.angle(np.exp(1j * (observed_phase - reference_phase)))
    wrapped = np.where(reliable, raw_wrapped, np.nan)
    unwrapped = _unwrap_masked_phase(raw_wrapped, reliable)

    if comparison_frequency is None:
        comparison_frequency = reference_spectrum.get("dominant_frequency")
    if comparison_frequency is None:
        return {
            "phase_difference": wrapped,
            "unwrapped_phase_difference": unwrapped,
            "phase_reliable": reliable,
            "comparison_frequency": None,
            "comparison_bin_frequency": None,
            "comparison_index": None,
            "reference_phase_at_comparison": None,
            "reference_phase_degrees_at_comparison": None,
            "observed_phase_at_comparison": None,
            "observed_phase_degrees_at_comparison": None,
            "phase_difference_at_comparison": None,
            "phase_difference_degrees_at_comparison": None,
            "phase_status": "zero_reference_spectrum",
        }
    index = int(np.argmin(np.abs(frequencies - float(comparison_frequency))))
    is_reliable = bool(reliable[index])
    difference = float(wrapped[index]) if is_reliable else None
    reference_value = float(reference_phase[index]) if is_reliable else None
    observed_value = float(observed_phase[index]) if is_reliable else None
    return {
        "phase_difference": wrapped,
        "unwrapped_phase_difference": unwrapped,
        "phase_reliable": reliable,
        "comparison_frequency": float(comparison_frequency),
        "comparison_bin_frequency": float(frequencies[index]),
        "comparison_index": index,
        "reference_phase_at_comparison": reference_value,
        "reference_phase_degrees_at_comparison": None
        if reference_value is None
        else float(np.degrees(reference_value)),
        "observed_phase_at_comparison": observed_value,
        "observed_phase_degrees_at_comparison": None
        if observed_value is None
        else float(np.degrees(observed_value)),
        "phase_difference_at_comparison": difference,
        "phase_difference_degrees_at_comparison": None
        if difference is None
        else float(np.degrees(difference)),
        "phase_status": "ok" if is_reliable else "low_spectral_amplitude",
    }


def compute_cross_correlation(
    reference: np.ndarray,
    observed: np.ndarray,
    *,
    dt: float,
    epsilon: float = 1.0e-30,
    max_lag_seconds: float | None = None,
) -> dict[str, Any]:
    """Estimate the observed-vs-reference delay by normalized correlation.

    Positive lag means that the observed trace is delayed relative to the
    reference trace. No sub-sample interpolation is applied.
    """

    x = _finite_array(reference, name="reference", ndim=1)
    y = _finite_array(observed, name="observed", ndim=1)
    if x.shape != y.shape:
        raise ReceiverTraceAlignmentError(
            "Cross-correlation requires equally shaped signals."
        )
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= epsilon:
        return {
            "cross_correlation_max": None,
            "cross_correlation_lag_steps": None,
            "cross_correlation_lag_time": None,
            "cross_correlation_status": "zero_variance_signal",
        }
    correlation = np.correlate(y, x, mode="full") / denominator
    lags = np.arange(-x.size + 1, x.size, dtype=np.int64)
    if max_lag_seconds is not None:
        max_steps = int(math.floor(max_lag_seconds / dt + 1.0e-12))
        mask = np.abs(lags) <= max_steps
        if not np.any(mask):
            raise ValueError("max_lag_seconds excludes every correlation lag.")
        correlation = correlation[mask]
        lags = lags[mask]
    index = int(np.argmax(correlation))
    lag_steps = int(lags[index])
    return {
        "cross_correlation_max": float(correlation[index]),
        "cross_correlation_lag_steps": lag_steps,
        "cross_correlation_lag_time": float(lag_steps * dt),
        "cross_correlation_status": "ok",
    }


def compute_amplitude_metrics(signal: np.ndarray, *, dt: float) -> dict[str, float]:
    """Compute temporal peak, RMS, and discrete energy metrics."""

    values = _finite_array(signal, name="signal", ndim=1)
    return {
        "temporal_max_abs": float(np.max(np.abs(values))),
        "temporal_rms": float(np.sqrt(np.mean(values * values))),
        "temporal_energy": float(dt * np.dot(values, values)),
        "temporal_mean": float(np.mean(values)),
    }


def compute_polarization(
    values: np.ndarray,
    *,
    components: Sequence[str],
    epsilon: float = 1.0e-30,
) -> dict[str, Any]:
    """Estimate principal 2-D polarization from x/y receiver components."""

    array = _finite_array(values, name="polarization_values", ndim=2)
    component_names = tuple(str(item) for item in components)
    if "x" not in component_names or "y" not in component_names:
        return {
            "polarization_available": False,
            "polarization_status": "missing_vector_component",
            "polarization_angle_radians": None,
            "polarization_angle_degrees": None,
            "polarization_rectilinearity": None,
            "polarization_rms_y_over_x": None,
            "vector_rms": None,
        }
    x_index = component_names.index("x")
    y_index = component_names.index("y")
    xy = array[:, [x_index, y_index]]
    xy = xy - np.mean(xy, axis=0, keepdims=True)
    covariance = xy.T @ xy / float(max(1, xy.shape[0]))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    principal = eigenvectors[:, order[0]]
    total = float(np.sum(eigenvalues))
    if total <= epsilon:
        return {
            "polarization_available": True,
            "polarization_status": "zero_vector_signal",
            "polarization_angle_radians": None,
            "polarization_angle_degrees": None,
            "polarization_rectilinearity": None,
            "polarization_rms_y_over_x": None,
            "vector_rms": 0.0,
        }
    angle = float(np.arctan2(principal[1], principal[0]) % np.pi)
    rectilinearity = float((eigenvalues[0] - eigenvalues[1]) / total)
    rms_x = float(np.sqrt(np.mean(xy[:, 0] ** 2)))
    rms_y = float(np.sqrt(np.mean(xy[:, 1] ** 2)))
    ratio = None if rms_x <= epsilon else float(rms_y / rms_x)
    vector_rms = float(np.sqrt(np.mean(np.sum(xy * xy, axis=1))))
    return {
        "polarization_available": True,
        "polarization_status": "ok",
        "polarization_angle_radians": angle,
        "polarization_angle_degrees": float(np.degrees(angle)),
        "polarization_rectilinearity": rectilinearity,
        "polarization_rms_y_over_x": ratio,
        "vector_rms": vector_rms,
    }


def _safe_ratio(numerator: float, denominator: float, epsilon: float) -> tuple[float | None, str]:
    if abs(denominator) <= epsilon:
        return (None, "both_zero") if abs(numerator) <= epsilon else (None, "zero_reference")
    return float(numerator / denominator), "ok"


def _safe_ratio_with_reference_floor(
    numerator: float,
    denominator: float,
    *,
    reference_scale: float,
    relative_floor: float,
    absolute_floor: float,
) -> tuple[float | None, str, float]:
    """Return a ratio only when the reference denominator is meaningfully nonzero."""

    floor = max(float(absolute_floor), float(relative_floor) * abs(float(reference_scale)))
    if abs(denominator) <= floor:
        status = (
            "both_amplitudes_too_small"
            if abs(numerator) <= floor
            else "reference_amplitude_too_small"
        )
        return None, status, floor
    return float(numerator / denominator), "ok", floor


def analyze_trace_pair(pair: ReceiverTracePair, config: SpectrumConfig) -> CaseAnalysis:
    """Analyze all receivers and components of one aligned trace pair."""

    time_validation = validate_time_axis(pair.times)
    dt = pair.dt
    frequency_axis = np.fft.rfftfreq(pair.num_steps, d=dt)
    shape = (pair.num_receivers, pair.num_components, frequency_axis.size)
    reference_amplitudes = np.empty(shape, dtype=np.float64)
    observed_amplitudes = np.empty(shape, dtype=np.float64)
    reference_phases = np.empty(shape, dtype=np.float64)
    observed_phases = np.empty(shape, dtype=np.float64)
    phase_differences = np.full(shape, np.nan, dtype=np.float64)
    unwrapped_phase_differences = np.full(shape, np.nan, dtype=np.float64)
    phase_reliable = np.zeros(shape, dtype=bool)
    metrics_rows: list[dict[str, Any]] = []
    polarization_rows: list[dict[str, Any]] = []
    source_spectrum = (
        None
        if config.expected_source_frequency is None
        else compute_effective_source_spectrum(
            pair.times,
            frequency=config.expected_source_frequency,
            num_periods=config.source_num_periods,
            t0=config.source_t0,
            amplitude_threshold=config.source_band_amplitude_threshold,
        )
    )

    polarization_by_receiver: dict[int, dict[str, Any]] = {}
    for receiver_index, receiver_id in enumerate(pair.receiver_ids):
        reference_polarization = compute_polarization(
            pair.reference_values[:, receiver_index, :],
            components=pair.components,
            epsilon=config.correlation_epsilon,
        )
        observed_polarization = compute_polarization(
            pair.observed_values[:, receiver_index, :],
            components=pair.components,
            epsilon=config.correlation_epsilon,
        )
        angle_shift = None
        if (
            reference_polarization["polarization_angle_radians"] is not None
            and observed_polarization["polarization_angle_radians"] is not None
        ):
            angle_shift = _axial_angle_difference_radians(
                observed_polarization["polarization_angle_radians"],
                reference_polarization["polarization_angle_radians"],
            )
        reference_vector = np.linalg.norm(
            pair.reference_values[:, receiver_index, :], axis=1
        )
        observed_vector = np.linalg.norm(
            pair.observed_values[:, receiver_index, :], axis=1
        )
        reference_vector_metrics = compute_amplitude_metrics(reference_vector, dt=dt)
        observed_vector_metrics = compute_amplitude_metrics(observed_vector, dt=dt)
        vector_rms_ratio, vector_rms_ratio_status = _safe_ratio(
            observed_vector_metrics["temporal_rms"],
            reference_vector_metrics["temporal_rms"],
            config.correlation_epsilon,
        )
        row = {
            "case_name": pair.case_name,
            "geometry_kind": pair.geometry_kind,
            "residual_family": pair.residual_family,
            "receiver_id": int(receiver_id),
            "x": None if np.isnan(pair.centers[receiver_index, 0]) else float(pair.centers[receiver_index, 0]),
            "y": None if np.isnan(pair.centers[receiver_index, 1]) else float(pair.centers[receiver_index, 1]),
            "reference_polarization_angle_radians": reference_polarization["polarization_angle_radians"],
            "reference_polarization_angle_degrees": reference_polarization["polarization_angle_degrees"],
            "reference_polarization_rectilinearity": reference_polarization["polarization_rectilinearity"],
            "reference_polarization_rms_y_over_x": reference_polarization["polarization_rms_y_over_x"],
            "reference_vector_rms": reference_polarization["vector_rms"],
            "observed_polarization_angle_radians": observed_polarization["polarization_angle_radians"],
            "observed_polarization_angle_degrees": observed_polarization["polarization_angle_degrees"],
            "observed_polarization_rectilinearity": observed_polarization["polarization_rectilinearity"],
            "observed_polarization_rms_y_over_x": observed_polarization["polarization_rms_y_over_x"],
            "observed_vector_rms": observed_polarization["vector_rms"],
            "reference_vector_max_abs": reference_vector_metrics["temporal_max_abs"],
            "observed_vector_max_abs": observed_vector_metrics["temporal_max_abs"],
            "reference_vector_energy": reference_vector_metrics["temporal_energy"],
            "observed_vector_energy": observed_vector_metrics["temporal_energy"],
            "observed_reference_vector_rms_ratio": vector_rms_ratio,
            "observed_reference_vector_rms_ratio_status": vector_rms_ratio_status,
            "polarization_angle_shift_radians": angle_shift,
            "polarization_angle_shift_degrees": None if angle_shift is None else float(np.degrees(angle_shift)),
            "polarization_status": reference_polarization["polarization_status"]
            if reference_polarization["polarization_status"] != "ok"
            else observed_polarization["polarization_status"],
        }
        polarization_rows.append(row)
        polarization_by_receiver[int(receiver_id)] = row

    reliable_phase_count = 0
    total_phase_count = pair.num_receivers * pair.num_components
    for receiver_index, receiver_id in enumerate(pair.receiver_ids):
        for component_index, component in enumerate(pair.components):
            reference_signal = pair.reference_values[:, receiver_index, component_index]
            observed_signal = pair.observed_values[:, receiver_index, component_index]
            reference_spectrum = compute_fft_spectrum(
                reference_signal,
                dt=dt,
                window=config.window,
                detrend=config.detrend,
                significant_amplitude_threshold=config.significant_amplitude_threshold,
            )
            observed_spectrum = compute_fft_spectrum(
                observed_signal,
                dt=dt,
                window=config.window,
                detrend=config.detrend,
                significant_amplitude_threshold=config.significant_amplitude_threshold,
            )
            comparison = compute_phase_difference(
                reference_spectrum,
                observed_spectrum,
                phase_amplitude_threshold=config.phase_amplitude_threshold,
                comparison_frequency=reference_spectrum["dominant_frequency"],
            )
            expected_phase = (
                None
                if config.expected_source_frequency is None
                else compute_phase_difference(
                    reference_spectrum,
                    observed_spectrum,
                    phase_amplitude_threshold=config.phase_amplitude_threshold,
                    comparison_frequency=config.expected_source_frequency,
                )
            )
            correlation = compute_cross_correlation(
                reference_signal,
                observed_signal,
                dt=dt,
                epsilon=config.correlation_epsilon,
                max_lag_seconds=config.max_lag_seconds,
            )
            reference_amplitude_metrics = compute_amplitude_metrics(reference_signal, dt=dt)
            observed_amplitude_metrics = compute_amplitude_metrics(observed_signal, dt=dt)
            polarization = polarization_by_receiver[int(receiver_id)]
            reference_vector_rms = float(polarization["reference_vector_rms"] or 0.0)
            reference_vector_energy = float(polarization["reference_vector_energy"] or 0.0)
            amplitude_ratio, amplitude_ratio_status, amplitude_ratio_floor = (
                _safe_ratio_with_reference_floor(
                    observed_amplitude_metrics["temporal_rms"],
                    reference_amplitude_metrics["temporal_rms"],
                    reference_scale=reference_vector_rms,
                    relative_floor=config.amplitude_ratio_reference_floor_relative,
                    absolute_floor=config.amplitude_ratio_reference_floor_absolute,
                )
            )
            energy_ratio, energy_ratio_status, energy_ratio_floor = (
                _safe_ratio_with_reference_floor(
                    observed_amplitude_metrics["temporal_energy"],
                    reference_amplitude_metrics["temporal_energy"],
                    reference_scale=reference_vector_energy,
                    relative_floor=config.amplitude_ratio_reference_floor_relative**2,
                    absolute_floor=(
                        config.amplitude_ratio_reference_floor_absolute**2
                        * pair.num_steps
                        * dt
                    ),
                )
            )
            reference_amplitudes[receiver_index, component_index] = reference_spectrum["amplitude"]
            observed_amplitudes[receiver_index, component_index] = observed_spectrum["amplitude"]
            reference_phases[receiver_index, component_index] = reference_spectrum["phase"]
            observed_phases[receiver_index, component_index] = observed_spectrum["phase"]
            phase_differences[receiver_index, component_index] = comparison["phase_difference"]
            unwrapped_phase_differences[receiver_index, component_index] = comparison[
                "unwrapped_phase_difference"
            ]
            phase_reliable[receiver_index, component_index] = comparison["phase_reliable"]
            if comparison["phase_status"] == "ok":
                reliable_phase_count += 1
            metrics_rows.append(
                {
                    "case_name": pair.case_name,
                    "geometry_kind": pair.geometry_kind,
                    "residual_family": pair.residual_family,
                    "receiver_id": int(receiver_id),
                    "x": None if np.isnan(pair.centers[receiver_index, 0]) else float(pair.centers[receiver_index, 0]),
                    "y": None if np.isnan(pair.centers[receiver_index, 1]) else float(pair.centers[receiver_index, 1]),
                    "component": str(component),
                    "num_steps": pair.num_steps,
                    "dt": dt,
                    "sampling_frequency": pair.sampling_frequency,
                    "time_start": float(pair.times[0]),
                    "time_end": float(pair.times[-1]),
                    "reference_dominant_frequency": reference_spectrum["dominant_frequency"],
                    "observed_dominant_frequency": observed_spectrum["dominant_frequency"],
                    "dominant_frequency_shift": None
                    if reference_spectrum["dominant_frequency"] is None
                    or observed_spectrum["dominant_frequency"] is None
                    else float(
                        observed_spectrum["dominant_frequency"]
                        - reference_spectrum["dominant_frequency"]
                    ),
                    "reference_dominant_amplitude": reference_spectrum["dominant_amplitude"],
                    "observed_dominant_amplitude": observed_spectrum["dominant_amplitude"],
                    "reference_significant_band_min": reference_spectrum["significant_band_min"],
                    "reference_significant_band_max": reference_spectrum["significant_band_max"],
                    "observed_significant_band_min": observed_spectrum["significant_band_min"],
                    "observed_significant_band_max": observed_spectrum["significant_band_max"],
                    "reference_spectral_energy": reference_spectrum["spectral_energy"],
                    "observed_spectral_energy": observed_spectrum["spectral_energy"],
                    "phase_comparison_frequency": comparison["comparison_bin_frequency"],
                    "reference_phase_radians": comparison["reference_phase_at_comparison"],
                    "reference_phase_degrees": comparison[
                        "reference_phase_degrees_at_comparison"
                    ],
                    "observed_phase_radians": comparison["observed_phase_at_comparison"],
                    "observed_phase_degrees": comparison[
                        "observed_phase_degrees_at_comparison"
                    ],
                    "phase_difference_radians": comparison["phase_difference_at_comparison"],
                    "phase_difference_degrees": comparison["phase_difference_degrees_at_comparison"],
                    "phase_status": comparison["phase_status"],
                    "expected_source_frequency": config.expected_source_frequency,
                    "reference_phase_at_expected_source_radians": None
                    if expected_phase is None
                    else expected_phase["reference_phase_at_comparison"],
                    "reference_phase_at_expected_source_degrees": None
                    if expected_phase is None
                    else expected_phase["reference_phase_degrees_at_comparison"],
                    "observed_phase_at_expected_source_radians": None
                    if expected_phase is None
                    else expected_phase["observed_phase_at_comparison"],
                    "observed_phase_at_expected_source_degrees": None
                    if expected_phase is None
                    else expected_phase["observed_phase_degrees_at_comparison"],
                    "phase_difference_at_expected_source_radians": None
                    if expected_phase is None
                    else expected_phase["phase_difference_at_comparison"],
                    "phase_difference_at_expected_source_degrees": None
                    if expected_phase is None
                    else expected_phase["phase_difference_degrees_at_comparison"],
                    "phase_at_expected_source_status": None
                    if expected_phase is None
                    else expected_phase["phase_status"],
                    **correlation,
                    "reference_temporal_max_abs": reference_amplitude_metrics["temporal_max_abs"],
                    "observed_temporal_max_abs": observed_amplitude_metrics["temporal_max_abs"],
                    "reference_temporal_rms": reference_amplitude_metrics["temporal_rms"],
                    "observed_temporal_rms": observed_amplitude_metrics["temporal_rms"],
                    "amplitude_rms_ratio": amplitude_ratio,
                    "amplitude_rms_ratio_status": amplitude_ratio_status,
                    "amplitude_rms_ratio_reliable": amplitude_ratio_status == "ok",
                    "amplitude_rms_reference_floor": amplitude_ratio_floor,
                    "amplitude_rms_reference_scale": reference_vector_rms,
                    "amplitude_rms_relative_change": None
                    if amplitude_ratio is None
                    else float(amplitude_ratio - 1.0),
                    "reference_temporal_energy": reference_amplitude_metrics["temporal_energy"],
                    "observed_temporal_energy": observed_amplitude_metrics["temporal_energy"],
                    "temporal_energy_ratio": energy_ratio,
                    "temporal_energy_ratio_status": energy_ratio_status,
                    "temporal_energy_ratio_reliable": energy_ratio_status == "ok",
                    "temporal_energy_reference_floor": energy_ratio_floor,
                    "temporal_energy_reference_scale": reference_vector_energy,
                    "reference_polarization_angle_degrees": polarization[
                        "reference_polarization_angle_degrees"
                    ],
                    "observed_polarization_angle_degrees": polarization[
                        "observed_polarization_angle_degrees"
                    ],
                    "polarization_angle_shift_degrees": polarization[
                        "polarization_angle_shift_degrees"
                    ],
                    "reference_polarization_rectilinearity": polarization[
                        "reference_polarization_rectilinearity"
                    ],
                    "observed_polarization_rectilinearity": polarization[
                        "observed_polarization_rectilinearity"
                    ],
                    "polarization_status": polarization["polarization_status"],
                }
            )

    network_reference_psd = np.mean(reference_amplitudes**2, axis=(0, 1))
    network_observed_psd = np.mean(observed_amplitudes**2, axis=(0, 1))
    network_reference_psd[0] = 0.0
    network_observed_psd[0] = 0.0
    network_reference_index = int(np.argmax(network_reference_psd))
    network_observed_index = int(np.argmax(network_observed_psd))
    network_reference_frequency = float(frequency_axis[network_reference_index])
    network_observed_frequency = float(frequency_axis[network_observed_index])
    nominal_source_error = (
        None
        if config.expected_source_frequency is None
        else abs(network_reference_frequency - config.expected_source_frequency)
    )
    observed_nominal_source_error = (
        None
        if config.expected_source_frequency is None
        else abs(network_observed_frequency - config.expected_source_frequency)
    )
    nominal_source_compatible = (
        None
        if nominal_source_error is None
        else bool(nominal_source_error <= config.source_frequency_tolerance)
    )
    observed_nominal_source_compatible = (
        None
        if observed_nominal_source_error is None
        else bool(observed_nominal_source_error <= config.source_frequency_tolerance)
    )
    source_band_min = None if source_spectrum is None else source_spectrum["significant_band_min"]
    source_band_max = None if source_spectrum is None else source_spectrum["significant_band_max"]
    reference_source_spectrum_status = (
        "not_evaluated"
        if source_spectrum is None
        else classify_frequency_against_source_band(
            network_reference_frequency,
            source_band_min=source_band_min,
            source_band_max=source_band_max,
            frequency_resolution=float(frequency_axis[1] - frequency_axis[0]),
        )
    )
    observed_source_spectrum_status = (
        "not_evaluated"
        if source_spectrum is None
        else classify_frequency_against_source_band(
            network_observed_frequency,
            source_band_min=source_band_min,
            source_band_max=source_band_max,
            frequency_resolution=float(frequency_axis[1] - frequency_axis[0]),
        )
    )
    source_compatible = (
        None
        if source_spectrum is None
        else reference_source_spectrum_status == "source_spectrum_compatible"
    )
    observed_source_compatible = (
        None
        if source_spectrum is None
        else observed_source_spectrum_status == "source_spectrum_compatible"
    )
    phase_reliable_fraction = reliable_phase_count / float(total_phase_count)
    unreliable_amplitude_ratio_count = sum(
        row.get("amplitude_rms_ratio_status") != "ok" for row in metrics_rows
    )
    amplitude_ratio_reliable_fraction = 1.0 - (
        unreliable_amplitude_ratio_count / float(max(1, len(metrics_rows)))
    )
    warnings: list[str] = []
    validation_notes: list[str] = []
    source_status_messages = {
        "low_frequency_dominated_trace": (
            "The network dominant frequency lies below the effective sampled-source band."
        ),
        "high_frequency_dominated_trace": (
            "The network dominant frequency lies above the effective sampled-source band."
        ),
        "missing_source_band": "The effective sampled-source band could not be determined.",
        "missing_frequency": "The network dominant frequency could not be determined.",
    }
    if source_compatible is False:
        warnings.append(
            "Reference: " + source_status_messages.get(
                reference_source_spectrum_status,
                "The network dominant frequency is incompatible with the sampled-source spectrum.",
            )
        )
    if observed_source_compatible is False:
        warnings.append(
            "Observed: " + source_status_messages.get(
                observed_source_spectrum_status,
                "The network dominant frequency is incompatible with the sampled-source spectrum.",
            )
        )
    if nominal_source_compatible is False and source_compatible is True:
        validation_notes.append(
            "The reference dominant frequency differs from the nominal carrier but remains inside "
            "the effective spectrum of the finite-duration Hann-sine source."
        )
    if observed_nominal_source_compatible is False and observed_source_compatible is True:
        validation_notes.append(
            "The observed dominant frequency differs from the nominal carrier but remains inside "
            "the effective spectrum of the finite-duration Hann-sine source."
        )
    if unreliable_amplitude_ratio_count:
        validation_notes.append(
            f"{unreliable_amplitude_ratio_count} amplitude ratio(s) were suppressed because "
            "the corresponding reference component was below the configured relative/absolute floor."
        )
    if phase_reliable_fraction < 0.5:
        warnings.append(
            "Phase at the reference-dominant frequency is unreliable for more than half of receiver/component traces."
        )
    if pair.num_components < 2 or not {"x", "y"}.issubset(pair.components):
        warnings.append(
            "Complete two-component polarization is unavailable for this case."
        )

    summary = {
        "format_version": FORMAT_VERSION,
        "case_name": pair.case_name,
        "geometry_kind": pair.geometry_kind,
        "residual_family": pair.residual_family,
        "source_paths": list(pair.source_paths),
        "source_format": pair.metadata.get("source_format"),
        "num_steps": pair.num_steps,
        "num_receivers": pair.num_receivers,
        "num_components": pair.num_components,
        "components": list(pair.components),
        "time_validation": time_validation,
        "dt": dt,
        "duration": pair.duration,
        "first_time": float(pair.times[0]),
        "final_time": float(pair.times[-1]),
        "sampling_frequency": pair.sampling_frequency,
        "frequency_resolution": float(frequency_axis[1] - frequency_axis[0]),
        "network_reference_dominant_frequency": network_reference_frequency,
        "network_observed_dominant_frequency": network_observed_frequency,
        "network_dominant_frequency_shift": float(
            network_observed_frequency - network_reference_frequency
        ),
        "expected_source_frequency": config.expected_source_frequency,
        "source_frequency_absolute_error": nominal_source_error,
        "observed_source_frequency_absolute_error": observed_nominal_source_error,
        "source_frequency_tolerance": config.source_frequency_tolerance,
        "nominal_source_frequency_compatible": nominal_source_compatible,
        "observed_nominal_source_frequency_compatible": observed_nominal_source_compatible,
        "source_frequency_compatible": source_compatible,
        "observed_source_frequency_compatible": observed_source_compatible,
        "source_compatibility_basis": "sampled_hann_sine_effective_band"
        if source_spectrum is not None
        else None,
        "source_spectrum_status": reference_source_spectrum_status,
        "observed_source_spectrum_status": observed_source_spectrum_status,
        "source_spectrum_dominant_frequency": None
        if source_spectrum is None
        else source_spectrum["dominant_frequency"],
        "source_spectrum_band_min": source_band_min,
        "source_spectrum_band_max": source_band_max,
        "source_band_amplitude_threshold": config.source_band_amplitude_threshold,
        "source_num_periods": config.source_num_periods,
        "source_t0": config.source_t0,
        "source_duration": None
        if source_spectrum is None
        else source_spectrum["source_duration"],
        "phase_reliable_trace_fraction": phase_reliable_fraction,
        "num_amplitude_ratios_unreliable": unreliable_amplitude_ratio_count,
        "amplitude_ratio_reliable_fraction": amplitude_ratio_reliable_fraction,
        "amplitude_ratio_reference_floor_relative": config.amplitude_ratio_reference_floor_relative,
        "amplitude_ratio_reference_floor_absolute": config.amplitude_ratio_reference_floor_absolute,
        "polarization_available": bool(
            pair.num_components >= 2 and {"x", "y"}.issubset(pair.components)
        ),
        "validation_notes": validation_notes,
        "warnings": warnings,
        "status": "ok" if not warnings else "validation_warning",
        "metadata": dict(pair.metadata),
    }
    return CaseAnalysis(
        pair=pair,
        config=config,
        frequencies=frequency_axis,
        reference_amplitudes=reference_amplitudes,
        observed_amplitudes=observed_amplitudes,
        reference_phases=reference_phases,
        observed_phases=observed_phases,
        phase_differences=phase_differences,
        unwrapped_phase_differences=unwrapped_phase_differences,
        phase_reliable=phase_reliable,
        source_signal=None if source_spectrum is None else source_spectrum["signal"],
        source_amplitude=None if source_spectrum is None else source_spectrum["amplitude"],
        source_psd=None if source_spectrum is None else source_spectrum["psd"],
        metrics_rows=metrics_rows,
        polarization_rows=polarization_rows,
        summary=summary,
    )


def compare_line_box_cases(analyses: Sequence[CaseAnalysis]) -> list[dict[str, Any]]:
    """Compare temporal and spectral conventions between line and box cases."""

    rows: list[dict[str, Any]] = []
    families = sorted({analysis.pair.residual_family for analysis in analyses})
    for family in families:
        line_cases = [
            analysis
            for analysis in analyses
            if analysis.pair.residual_family == family
            and analysis.pair.geometry_kind == "line"
        ]
        box_cases = [
            analysis
            for analysis in analyses
            if analysis.pair.residual_family == family
            and analysis.pair.geometry_kind == "box"
        ]
        if not line_cases or not box_cases:
            rows.append(
                {
                    "residual_family": family,
                    "status": "missing_input",
                    "line_case": None if not line_cases else line_cases[0].pair.case_name,
                    "box_case": None if not box_cases else box_cases[0].pair.case_name,
                    "message": "Both line and box cases are required for comparison.",
                }
            )
            continue
        line = line_cases[0]
        box = box_cases[0]
        dt_match = bool(np.isclose(line.pair.dt, box.pair.dt, atol=1e-12, rtol=1e-10))
        time_axis_match = bool(
            line.pair.times.shape == box.pair.times.shape
            and np.allclose(line.pair.times, box.pair.times, atol=1e-12, rtol=1e-10)
        )
        source_frequency_difference = abs(
            line.summary["network_reference_dominant_frequency"]
            - box.summary["network_reference_dominant_frequency"]
        )
        def _finite_median(case: CaseAnalysis, key: str) -> float | None:
            values = [row.get(key) for row in case.metrics_rows]
            finite = np.asarray(
                [value for value in values if value is not None and np.isfinite(value)],
                dtype=float,
            )
            return None if finite.size == 0 else float(np.median(finite))

        line_phase_median = _finite_median(
            line, "phase_difference_at_expected_source_degrees"
        )
        box_phase_median = _finite_median(
            box, "phase_difference_at_expected_source_degrees"
        )
        line_lag_median = _finite_median(line, "cross_correlation_lag_time")
        box_lag_median = _finite_median(box, "cross_correlation_lag_time")
        line_correlation_median = _finite_median(line, "cross_correlation_max")
        box_correlation_median = _finite_median(box, "cross_correlation_max")
        status = "ok" if dt_match and time_axis_match else "validation_warning"
        rows.append(
            {
                "residual_family": family,
                "status": status,
                "line_case": line.pair.case_name,
                "box_case": box.pair.case_name,
                "line_dt": line.pair.dt,
                "box_dt": box.pair.dt,
                "dt_match": dt_match,
                "line_num_steps": line.pair.num_steps,
                "box_num_steps": box.pair.num_steps,
                "time_axis_match": time_axis_match,
                "line_network_reference_dominant_frequency": line.summary[
                    "network_reference_dominant_frequency"
                ],
                "box_network_reference_dominant_frequency": box.summary[
                    "network_reference_dominant_frequency"
                ],
                "network_reference_dominant_frequency_difference": source_frequency_difference,
                "line_median_phase_difference_at_expected_source_degrees": line_phase_median,
                "box_median_phase_difference_at_expected_source_degrees": box_phase_median,
                "line_median_cross_correlation_lag_time": line_lag_median,
                "box_median_cross_correlation_lag_time": box_lag_median,
                "line_median_cross_correlation_max": line_correlation_median,
                "box_median_cross_correlation_max": box_correlation_median,
                "line_source_frequency_compatible": line.summary[
                    "source_frequency_compatible"
                ],
                "box_source_frequency_compatible": box.summary[
                    "source_frequency_compatible"
                ],
                "message": "Temporal axes are coherent."
                if status == "ok"
                else "Line and box temporal conventions differ.",
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_jsonable(value), sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else _jsonable(value)
                    for key, value in row.items()
                }
            )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _plot_case_profiles(
    analysis: CaseAnalysis,
    *,
    output_path: Path,
    save_pdf: bool,
) -> list[Path]:
    plt = _configure_matplotlib()
    rows = analysis.metrics_rows
    components = list(analysis.pair.components)
    figure, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    receiver_ids = analysis.pair.receiver_ids
    for component in components:
        component_rows = [row for row in rows if row["component"] == component]
        component_rows.sort(key=lambda item: item["receiver_id"])
        x = np.asarray([row["receiver_id"] for row in component_rows], dtype=float)
        axes[0, 0].plot(
            x,
            [row["reference_dominant_frequency"] for row in component_rows],
            marker="o",
            label=f"reference {component}",
        )
        axes[0, 0].plot(
            x,
            [row["observed_dominant_frequency"] for row in component_rows],
            marker="x",
            linestyle="--",
            label=f"observed {component}",
        )
        phase_values = [
            np.nan if row["phase_difference_degrees"] is None else row["phase_difference_degrees"]
            for row in component_rows
        ]
        axes[0, 1].plot(x, phase_values, marker="o", label=component)
        axes[1, 0].plot(
            x,
            [row["cross_correlation_lag_time"] for row in component_rows],
            marker="o",
            label=component,
        )
        axes[1, 1].plot(
            x,
            [row["cross_correlation_max"] for row in component_rows],
            marker="o",
            label=component,
        )
        axes[2, 0].plot(
            x,
            [row["amplitude_rms_ratio"] for row in component_rows],
            marker="o",
            label=component,
        )
    polarization = sorted(analysis.polarization_rows, key=lambda item: item["receiver_id"])
    if polarization and analysis.summary["polarization_available"]:
        x = np.asarray([row["receiver_id"] for row in polarization], dtype=float)
        axes[2, 1].plot(
            x,
            [row["reference_polarization_angle_degrees"] for row in polarization],
            marker="o",
            label="reference",
        )
        axes[2, 1].plot(
            x,
            [row["observed_polarization_angle_degrees"] for row in polarization],
            marker="x",
            linestyle="--",
            label="observed",
        )
        axes[2, 1].set_ylabel("principal angle (degrees, modulo 180°)")
    else:
        axes[2, 1].text(
            0.5,
            0.5,
            "Complete polarization unavailable",
            ha="center",
            va="center",
            transform=axes[2, 1].transAxes,
        )
    titles = (
        "Dominant frequency",
        "Phase difference at reference-dominant frequency",
        "Cross-correlation delay",
        "Maximum normalized correlation",
        "Observed/reference RMS amplitude",
        "Principal polarization",
    )
    ylabels = (
        "frequency",
        "phase difference (degrees)",
        "delay",
        "correlation",
        "RMS ratio",
        "angle",
    )
    for axis, title, ylabel in zip(axes.flat, titles, ylabels, strict=True):
        axis.set_title(title)
        axis.set_xlabel("receiver id")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize="small")
    figure.suptitle(
        f"{analysis.pair.case_name}: {analysis.pair.residual_family} / {analysis.pair.geometry_kind}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    outputs = [output_path]
    if save_pdf:
        pdf_path = output_path.with_suffix(".pdf")
        figure.savefig(pdf_path)
        outputs.append(pdf_path)
    plt.close(figure)
    return outputs


def _plot_component_spectra(
    analysis: CaseAnalysis,
    *,
    component_index: int,
    output_path: Path,
    save_pdf: bool,
) -> list[Path]:
    plt = _configure_matplotlib()
    frequencies = analysis.frequencies
    mask = np.ones(frequencies.shape, dtype=bool)
    if analysis.config.max_plot_frequency is not None:
        mask = frequencies <= analysis.config.max_plot_frequency
    display_frequencies = frequencies[mask]
    reference = analysis.reference_amplitudes[:, component_index, :][:, mask]
    observed = analysis.observed_amplitudes[:, component_index, :][:, mask]
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    axes[0].plot(display_frequencies, np.median(reference, axis=0), label="reference median")
    axes[0].plot(display_frequencies, np.median(observed, axis=0), label="observed median", linestyle="--")
    source_band_min = analysis.summary.get("source_spectrum_band_min")
    source_band_max = analysis.summary.get("source_spectrum_band_max")
    if source_band_min is not None and source_band_max is not None:
        axes[0].axvspan(
            source_band_min,
            source_band_max,
            alpha=0.12,
            label="effective sampled-source band",
        )
    if analysis.config.expected_source_frequency is not None:
        axes[0].axvline(
            analysis.config.expected_source_frequency,
            linestyle=":",
            label="nominal source carrier",
        )
    axes[0].set_ylabel("amplitude")
    axes[0].set_title("Median amplitude spectrum")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    extent = [
        float(display_frequencies[0]),
        float(display_frequencies[-1]),
        float(analysis.pair.receiver_ids[0]) - 0.5,
        float(analysis.pair.receiver_ids[-1]) + 0.5,
    ]
    image_reference = axes[1].imshow(
        reference,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )
    axes[1].set_title("Reference amplitude spectrum by receiver")
    axes[1].set_ylabel("receiver id")
    figure.colorbar(image_reference, ax=axes[1], label="amplitude")
    image_observed = axes[2].imshow(
        observed,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )
    axes[2].set_title("Observed amplitude spectrum by receiver")
    axes[2].set_ylabel("receiver id")
    axes[2].set_xlabel("frequency")
    figure.colorbar(image_observed, ax=axes[2], label="amplitude")
    figure.suptitle(
        f"{analysis.pair.case_name}: component {analysis.pair.components[component_index]}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    outputs = [output_path]
    if save_pdf:
        pdf_path = output_path.with_suffix(".pdf")
        figure.savefig(pdf_path)
        outputs.append(pdf_path)
    plt.close(figure)
    return outputs


def write_analysis_outputs(
    analyses: Sequence[CaseAnalysis],
    *,
    output_dir: str | Path,
    analysis_mode: str = "auto",
    line_box_comparisons: Sequence[Mapping[str, Any]] | None = None,
    missing_inputs: Sequence[Mapping[str, Any]] = (),
    save_pdf: bool = True,
    save_npz: bool = True,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write deterministic CSV, JSON, NPZ, PNG, and optional PDF outputs."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    all_metrics = [
        {"analysis_mode": analysis_mode, **dict(row)}
        for analysis in analyses
        for row in analysis.metrics_rows
    ]
    all_polarization = [
        {"analysis_mode": analysis_mode, **dict(row)}
        for analysis in analyses
        for row in analysis.polarization_rows
    ]
    summaries = [
        {"analysis_mode": analysis_mode, **dict(analysis.summary)}
        for analysis in analyses
    ]
    if analysis_mode == "line-only":
        comparisons = []
    else:
        comparisons = list(
            compare_line_box_cases(analyses)
            if line_box_comparisons is None
            else line_box_comparisons
        )

    outputs: dict[str, Any] = {}
    outputs["summary_csv"] = str(
        _write_csv(root / "receiver_phase_spectrum_summary.csv", all_metrics)
    )
    outputs["polarization_csv"] = str(
        _write_csv(root / "receiver_polarization_summary.csv", all_polarization)
    )
    outputs["case_summary_csv"] = str(
        _write_csv(root / "receiver_phase_spectrum_case_summary.csv", summaries)
    )
    if analysis_mode == "line-box" or comparisons:
        outputs["line_box_comparison_csv"] = str(
            _write_csv(
                root / "receiver_phase_spectrum_line_box_comparison.csv",
                comparisons,
            )
        )

    case_outputs: dict[str, Any] = {}
    for analysis in analyses:
        case_dir = root / analysis.pair.case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        case_record: dict[str, Any] = {}
        case_record["metrics_csv"] = str(
            _write_csv(
                case_dir / "receiver_metrics.csv",
                [
                    {"analysis_mode": analysis_mode, **dict(row)}
                    for row in analysis.metrics_rows
                ],
            )
        )
        case_record["polarization_csv"] = str(
            _write_csv(
                case_dir / "polarization_metrics.csv",
                [
                    {"analysis_mode": analysis_mode, **dict(row)}
                    for row in analysis.polarization_rows
                ],
            )
        )
        if save_npz:
            npz_path = case_dir / "receiver_spectra.npz"
            np.savez_compressed(
                npz_path,
                kind=np.asarray("receiver_phase_spectrum_arrays", dtype=np.str_),
                format_version=np.asarray(FORMAT_VERSION, dtype=np.str_),
                case_name=np.asarray(analysis.pair.case_name, dtype=np.str_),
                geometry_kind=np.asarray(analysis.pair.geometry_kind, dtype=np.str_),
                residual_family=np.asarray(analysis.pair.residual_family, dtype=np.str_),
                frequencies=analysis.frequencies,
                receiver_ids=analysis.pair.receiver_ids,
                components=np.asarray(analysis.pair.components, dtype=np.str_),
                reference_amplitudes=analysis.reference_amplitudes,
                observed_amplitudes=analysis.observed_amplitudes,
                reference_phases=analysis.reference_phases,
                observed_phases=analysis.observed_phases,
                phase_differences=analysis.phase_differences,
                unwrapped_phase_differences=analysis.unwrapped_phase_differences,
                phase_reliable=analysis.phase_reliable,
                source_signal=np.asarray([], dtype=np.float64)
                if analysis.source_signal is None
                else analysis.source_signal,
                source_amplitude=np.asarray([], dtype=np.float64)
                if analysis.source_amplitude is None
                else analysis.source_amplitude,
                source_psd=np.asarray([], dtype=np.float64)
                if analysis.source_psd is None
                else analysis.source_psd,
            )
            case_record["spectra_npz"] = str(npz_path)
        profile_outputs = _plot_case_profiles(
            analysis,
            output_path=case_dir / f"receiver_phase_spectrum_{analysis.pair.case_name}.png",
            save_pdf=save_pdf,
        )
        case_record["profile_figures"] = [str(path) for path in profile_outputs]
        component_figures: dict[str, list[str]] = {}
        for component_index, component in enumerate(analysis.pair.components):
            component_outputs = _plot_component_spectra(
                analysis,
                component_index=component_index,
                output_path=case_dir
                / f"receiver_spectra_{analysis.pair.case_name}_{component}.png",
                save_pdf=save_pdf,
            )
            component_figures[str(component)] = [str(path) for path in component_outputs]
        case_record["component_spectrum_figures"] = component_figures
        case_outputs[analysis.pair.case_name] = case_record
    outputs["cases"] = case_outputs

    status = "ok"
    if any(summary["status"] != "ok" for summary in summaries):
        status = "validation_warning"
    if not analyses:
        status = "missing_input"
    manifest = {
        "kind": "receiver_phase_spectrum_manifest",
        "format_version": FORMAT_VERSION,
        "analysis_date_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "analysis_mode": analysis_mode,
        "geometry_kinds": sorted(
            {analysis.pair.geometry_kind for analysis in analyses}
        ),
        "num_cases": len(analyses),
        "num_missing_inputs": len(missing_inputs),
        "cases": summaries,
        "line_box_comparison_status": (
            "not_applicable"
            if analysis_mode in {"line-only", "box-only"}
            else "computed" if comparisons
            else "unavailable"
        ),
        "line_box_comparisons": comparisons,
        "missing_inputs": list(missing_inputs),
        "configuration": None
        if not analyses
        else _jsonable(analyses[0].config.__dict__),
        "command": None if command is None else list(command),
        "outputs": outputs,
    }
    manifest_path = root / "receiver_phase_spectrum_manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["manifest_json"] = str(manifest_path)

    lightweight_paths = [
        Path(outputs["summary_csv"]),
        Path(outputs["polarization_csv"]),
        Path(outputs["case_summary_csv"]),
        *(
            [Path(outputs["line_box_comparison_csv"])]
            if "line_box_comparison_csv" in outputs
            else []
        ),
        manifest_path,
    ]
    checksums = {
        str(path): {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in lightweight_paths
        if path.is_file()
    }
    checksum_path = root / "receiver_phase_spectrum_checksums.json"
    checksum_path.write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs["checksums_json"] = str(checksum_path)
    return outputs
