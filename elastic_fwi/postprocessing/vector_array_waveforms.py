from __future__ import annotations

"""Waveform analysis for validated vector-array observations.

The validated campaign CSV stores healthy/reference, defective/observed, and
source residual values in one long-format file.  This module reuses the strict
CSV, time, receiver-count, and receiver-centre validation implemented by
``receiver_aperture_comparison`` and adds per-receiver waveform diagnostics:
analytic envelopes, characteristic times, healthy/defective comparisons,
residual energies, representative traces, figures, and reproducibility
metadata.

No FEniCSx, PETSc, or MPI dependency is required.  Input files are read-only.
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
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .receiver_aperture_comparison import (
    ALLOWED_ANALYSIS_GEOMETRIES,
    ALLOWED_TIME_POLICIES,
    ApertureComparisonConfig,
    ApertureComparisonInputs,
    CampaignInspection,
    ReceiverApertureError,
    ReceiverGeometryError,
    ReceiverSpec,
    attach_receiver_geometry,
    build_receiver_geometry,
    load_and_validate_campaign,
    load_receiver_specs,
    select_representative_receivers,
)


SCHEMA_VERSION = "1.1.0"
ALLOWED_RESIDUAL_CONVENTIONS = (
    "source",
    "healthy-minus-defect",
    "defect-minus-healthy",
)
EXPECTED_COMPONENTS = ("x", "y")
SIGNAL_CASES = ("healthy", "defect", "residual")


class WaveformAnalysisError(RuntimeError):
    """Base class for waveform-analysis failures."""


class WaveformAlignmentError(WaveformAnalysisError):
    """Raised when component or healthy/defective series cannot be aligned."""


class SamplingIntervalError(WaveformAnalysisError):
    """Raised when the validated time grid is inconsistent with the protocol."""


class WaveformMetricError(WaveformAnalysisError):
    """Raised when a signal metric cannot be evaluated safely."""


@dataclass(frozen=True)
class WaveformAnalysisInputs:
    line_residuals: Path
    box_residuals: Path | None = None
    line_manifest: Path | None = None
    box_manifest: Path | None = None

    def resolved(self) -> "WaveformAnalysisInputs":
        def _resolved(path: Path | None) -> Path | None:
            return None if path is None else Path(path).expanduser().resolve()

        return WaveformAnalysisInputs(
            line_residuals=Path(self.line_residuals).expanduser().resolve(),
            box_residuals=_resolved(self.box_residuals),
            line_manifest=_resolved(self.line_manifest),
            box_manifest=_resolved(self.box_manifest),
        )


@dataclass(frozen=True)
class WaveformAnalysisConfig:
    inputs: WaveformAnalysisInputs
    output_dir: Path
    dt: float = 0.05
    final_time: float = 26.0
    arrival_threshold: float = 0.1
    residual_convention: str = "source"
    time_column_policy: str = "strict-physical"
    time_atol: float = 1.0e-12
    time_rtol: float = 1.0e-12
    value_atol: float = 1.0e-14
    value_rtol: float = 1.0e-12
    geometry_atol: float = 1.0e-12
    ratio_epsilon: float = 1.0e-30
    receiver_ids: tuple[int, ...] | None = None
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
        if not math.isfinite(float(self.final_time)) or float(self.final_time) <= 0.0:
            raise ValueError("final_time must be finite and strictly positive.")
        if not math.isfinite(float(self.arrival_threshold)) or not (
            0.0 < float(self.arrival_threshold) <= 1.0
        ):
            raise ValueError("arrival_threshold must lie in (0, 1].")
        if self.residual_convention not in ALLOWED_RESIDUAL_CONVENTIONS:
            raise ValueError(
                "residual_convention must be one of "
                f"{ALLOWED_RESIDUAL_CONVENTIONS}, got {self.residual_convention!r}."
            )
        if self.time_column_policy not in ALLOWED_TIME_POLICIES:
            raise ValueError(
                f"time_column_policy must be one of {ALLOWED_TIME_POLICIES}."
            )
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
        if self.receiver_ids is not None:
            normalized = tuple(sorted(set(int(value) for value in self.receiver_ids)))
            if any(value < 0 for value in normalized):
                raise ValueError("receiver_ids must be non-negative.")
            object.__setattr__(self, "receiver_ids", normalized)


@dataclass(frozen=True)
class WaveformAnalysisResult:
    inspections: tuple[CampaignInspection, ...]
    canonical: pd.DataFrame
    receiver_geometry: pd.DataFrame
    time_validation: pd.DataFrame
    receiver_selection: pd.DataFrame
    selected_traces: pd.DataFrame
    signal_metrics: pd.DataFrame
    comparison_metrics: pd.DataFrame
    network_summary: pd.DataFrame
    summary: dict[str, Any]
    elapsed_seconds: float


@dataclass(frozen=True)
class WaveformOutputPaths:
    root: Path
    csv_dir: Path
    figures_dir: Path
    manifests_dir: Path
    logs_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "WaveformOutputPaths":
        root = Path(root).expanduser().resolve()
        return cls(
            root=root,
            csv_dir=root / "csv",
            figures_dir=root / "figures",
            manifests_dir=root / "manifests",
            logs_dir=root / "logs",
        )

    def create(self, *, overwrite: bool) -> None:
        if self.root.exists() and any(self.root.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Output directory is not empty: {self.root}. Use --overwrite."
                )
            shutil.rmtree(self.root)
        for path in (
            self.root,
            self.csv_dir,
            self.figures_dir,
            self.manifests_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pure numerical utilities
# ---------------------------------------------------------------------------


def analytic_envelope(values: np.ndarray) -> np.ndarray:
    """Return the Hilbert analytic-signal envelope using a NumPy FFT.

    The implementation follows the standard one-sided Fourier multiplier and
    therefore introduces no smoothing.  Input and output are one-dimensional.
    """

    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1:
        raise WaveformMetricError(
            f"analytic_envelope expects a one-dimensional signal, got {signal.shape}."
        )
    if signal.size == 0:
        raise WaveformMetricError("analytic_envelope received an empty signal.")
    if not np.all(np.isfinite(signal)):
        raise WaveformMetricError("analytic_envelope received non-finite values.")
    if signal.size == 1:
        return np.abs(signal)

    spectrum = np.fft.fft(signal)
    multiplier = np.zeros(signal.size, dtype=np.float64)
    multiplier[0] = 1.0
    if signal.size % 2 == 0:
        multiplier[1 : signal.size // 2] = 2.0
        multiplier[signal.size // 2] = 1.0
    else:
        multiplier[1 : (signal.size + 1) // 2] = 2.0
    analytic = np.fft.ifft(spectrum * multiplier)
    return np.abs(analytic).astype(np.float64, copy=False)


def threshold_crossing_index(
    envelope: np.ndarray,
    *,
    fraction: float,
) -> tuple[int | None, float, str]:
    """Return first index crossing ``fraction * max(envelope)``.

    This is a threshold diagnostic, not an automatically interpreted physical
    arrival time.  Zero signals return ``(None, 0, 'zero_signal')``.
    """

    values = np.asarray(envelope, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise WaveformMetricError("Envelope must be a non-empty 1-D array.")
    if not np.all(np.isfinite(values)):
        raise WaveformMetricError("Envelope contains non-finite values.")
    if not math.isfinite(float(fraction)) or not (0.0 < float(fraction) <= 1.0):
        raise ValueError("fraction must lie in (0, 1].")
    maximum = float(np.max(values))
    if maximum <= 0.0:
        return None, 0.0, "zero_signal"
    threshold = float(fraction) * maximum
    indices = np.flatnonzero(values >= threshold)
    if indices.size == 0:
        return None, threshold, "not_crossed"
    return int(indices[0]), threshold, "ok"


def _validate_signal_axis(
    values: np.ndarray,
    steps: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = np.asarray(values, dtype=np.float64)
    step_values = np.asarray(steps, dtype=np.int64)
    time_values = np.asarray(times, dtype=np.float64)
    if signal.ndim != 1:
        raise WaveformMetricError(f"Signal must be 1-D, got {signal.shape}.")
    if signal.size == 0:
        raise WaveformMetricError("Signal is empty.")
    if step_values.shape != signal.shape or time_values.shape != signal.shape:
        raise WaveformMetricError(
            "Signal, step, and time arrays must have identical one-dimensional shapes."
        )
    if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(time_values)):
        raise WaveformMetricError("Signal or physical time contains non-finite values.")
    if np.any(np.diff(step_values) <= 0):
        raise WaveformMetricError("Step indices must be strictly increasing.")
    if np.any(np.diff(time_values) <= 0):
        raise WaveformMetricError("Physical times must be strictly increasing.")
    return signal, step_values, time_values


def signal_characteristics(
    values: np.ndarray,
    *,
    steps: np.ndarray,
    times: np.ndarray,
    dt: float,
    threshold_fraction: float,
) -> dict[str, Any]:
    """Compute signed extrema, envelope metrics, threshold time, and energy."""

    signal, step_values, time_values = _validate_signal_axis(values, steps, times)
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be finite and strictly positive.")

    envelope = analytic_envelope(signal)
    abs_values = np.abs(signal)
    absolute_index = int(np.argmax(abs_values))
    envelope_index = int(np.argmax(envelope))
    crossing_index, threshold_value, crossing_status = threshold_crossing_index(
        envelope,
        fraction=threshold_fraction,
    )
    if crossing_index is None:
        crossing_step: int | None = None
        crossing_time: float | None = None
    else:
        crossing_step = int(step_values[crossing_index])
        crossing_time = float(time_values[crossing_index])

    return {
        "num_steps": int(signal.size),
        "dt": float(dt),
        "time_min": float(time_values[0]),
        "time_max": float(time_values[-1]),
        "signed_max": float(np.max(signal)),
        "signed_min": float(np.min(signal)),
        "absolute_max": float(abs_values[absolute_index]),
        "absolute_peak_step": int(step_values[absolute_index]),
        "absolute_peak_time": float(time_values[absolute_index]),
        "envelope_max": float(envelope[envelope_index]),
        "envelope_peak_step": int(step_values[envelope_index]),
        "envelope_peak_time": float(time_values[envelope_index]),
        "threshold_fraction": float(threshold_fraction),
        "threshold_value": float(threshold_value),
        "threshold_arrival_step": crossing_step,
        "threshold_arrival_time": crossing_time,
        "threshold_status": crossing_status,
        "discrete_energy": float(float(dt) * np.dot(signal, signal)),
        "l2_time_norm": float(np.sqrt(float(dt) * np.dot(signal, signal))),
    }


def normalized_cross_correlation_lag(
    healthy: np.ndarray,
    defect: np.ndarray,
    *,
    dt: float,
    epsilon: float,
) -> dict[str, Any]:
    """Estimate defect-vs-healthy lag by full normalized cross-correlation.

    The signals are demeaned.  Positive lag means that the defective trace is
    delayed relative to the healthy trace.  No interpolation is applied.
    """

    reference = np.asarray(healthy, dtype=np.float64)
    observed = np.asarray(defect, dtype=np.float64)
    if reference.ndim != 1 or observed.ndim != 1 or reference.shape != observed.shape:
        raise WaveformAlignmentError(
            "Cross-correlation requires equally shaped one-dimensional signals."
        )
    if reference.size == 0:
        raise WaveformAlignmentError("Cross-correlation received empty signals.")
    x = reference - float(np.mean(reference))
    y = observed - float(np.mean(observed))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= float(epsilon):
        return {
            "cross_correlation_max": None,
            "cross_correlation_lag_steps": None,
            "cross_correlation_lag_time": None,
            "cross_correlation_status": "zero_variance_signal",
        }
    correlation = np.correlate(y, x, mode="full") / denominator
    lags = np.arange(-reference.size + 1, reference.size, dtype=np.int64)
    index = int(np.argmax(correlation))
    lag_steps = int(lags[index])
    return {
        "cross_correlation_max": float(correlation[index]),
        "cross_correlation_lag_steps": lag_steps,
        "cross_correlation_lag_time": float(lag_steps * float(dt)),
        "cross_correlation_status": "ok",
    }


def safe_ratio(
    numerator: float,
    denominator: float,
    *,
    epsilon: float,
) -> tuple[float | None, str]:
    numerator = float(numerator)
    denominator = float(denominator)
    if abs(denominator) <= float(epsilon):
        if abs(numerator) <= float(epsilon):
            return None, "both_zero"
        return None, "undefined_zero_denominator"
    return numerator / denominator, "ok"


# ---------------------------------------------------------------------------
# Canonical data preparation
# ---------------------------------------------------------------------------


def apply_analysis_residual_convention(
    frame: pd.DataFrame,
    *,
    convention: str,
    atol: float,
    rtol: float,
) -> tuple[pd.DataFrame, str]:
    """Attach ``analysis_residual`` without modifying source columns."""

    if convention not in ALLOWED_RESIDUAL_CONVENTIONS:
        raise ValueError(f"Unsupported residual convention: {convention!r}.")
    source_conventions = sorted(str(v) for v in frame["residual_convention"].unique())
    if len(source_conventions) != 1:
        raise WaveformAlignmentError(
            f"Expected one source residual convention, found {source_conventions}."
        )
    source = source_conventions[0]
    healthy_minus_defect = (
        frame["reference"].to_numpy(dtype=np.float64, copy=False)
        - frame["observed"].to_numpy(dtype=np.float64, copy=False)
    )
    source_values = frame["residual"].to_numpy(dtype=np.float64, copy=False)
    if source == "reference_minus_observed":
        expected_source = healthy_minus_defect
    elif source == "observed_minus_reference":
        expected_source = -healthy_minus_defect
    else:
        raise WaveformAlignmentError(
            f"Unsupported validated source residual convention: {source!r}."
        )
    if not np.allclose(source_values, expected_source, atol=atol, rtol=rtol):
        error = float(np.max(np.abs(source_values - expected_source)))
        raise WaveformAlignmentError(
            "Source residual values no longer match their validated convention; "
            f"max_abs_error={error:.17g}."
        )

    if convention == "source":
        analysis = source_values
        analysis_name = source
    elif convention == "healthy-minus-defect":
        analysis = healthy_minus_defect
        analysis_name = "reference_minus_observed"
    else:
        analysis = -healthy_minus_defect
        analysis_name = "observed_minus_reference"

    result = frame.copy()
    result["healthy"] = result["reference"].to_numpy(dtype=np.float64, copy=False)
    result["defect"] = result["observed"].to_numpy(dtype=np.float64, copy=False)
    result["analysis_residual"] = np.asarray(analysis, dtype=np.float64)
    result["analysis_residual_convention"] = analysis_name
    result["healthy_source_column"] = "reference"
    result["defect_source_column"] = "observed"
    return result, analysis_name


def validate_protocol_time_interval(
    inspections: Sequence[CampaignInspection],
    *,
    dt: float,
    final_time: float,
    atol: float,
    rtol: float,
) -> pd.DataFrame:
    """Validate zero- or one-based sampling and the protocol terminal time.

    Zero-based campaigns store steps ``0..N-1`` and therefore include the
    initial state while excluding the protocol terminal sample.  One-based
    campaigns, such as those of the ``T20`` presets, store steps ``1..N`` and
    therefore start at ``dt`` while including the terminal sample.  Both contracts are explicit and
    reproducible from ``time_physical = step_index * dt``.
    """

    rows: list[dict[str, Any]] = []
    for inspection in inspections:
        physical_min = float(inspection.step_min * dt)
        physical_max = float(inspection.step_max * dt)
        if inspection.step_min == 0:
            initial_status = "initial_state_included"
        elif inspection.step_min == 1:
            initial_status = "initial_state_excluded_first_sample_at_dt"
        else:
            raise SamplingIntervalError(
                f"{inspection.geometry}: step_index_min must be 0 or 1, got "
                f"{inspection.step_min}."
            )
        if math.isclose(physical_max, final_time, abs_tol=atol, rel_tol=rtol):
            status = "terminal_included"
        elif math.isclose(
            physical_max + dt,
            final_time,
            abs_tol=atol,
            rel_tol=rtol,
        ):
            status = "right_open_terminal_excluded"
        else:
            raise SamplingIntervalError(
                f"{inspection.geometry}: last physical sample {physical_max:.17g} is "
                f"neither final_time={final_time:.17g} nor final_time-dt="
                f"{final_time - dt:.17g}."
            )
        rows.append(
            {
                "geometry": inspection.geometry,
                "num_steps": inspection.num_steps,
                "step_index_min": inspection.step_min,
                "step_index_max": inspection.step_max,
                "dt": float(dt),
                "protocol_final_time": float(final_time),
                "time_physical_min": physical_min,
                "time_physical_max": physical_max,
                "initial_sampling_status": initial_status,
                "sampling_status": status,
                "csv_time_status": inspection.time_status,
                "source_csv": str(inspection.path),
            }
        )
    return pd.DataFrame(rows).sort_values("geometry", ignore_index=True)


def _group_metadata_columns() -> list[str]:
    return [
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
        "residual_convention",
        "analysis_residual_convention",
        "source_csv",
    ]


def _sorted_signal_group(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ordered = group.sort_values("step_index", kind="mergesort")
    steps = ordered["step_index"].to_numpy(dtype=np.int64, copy=False)
    times = ordered["time_physical"].to_numpy(dtype=np.float64, copy=False)
    if steps.size == 0:
        raise WaveformAlignmentError("Empty receiver/component group.")
    expected = np.arange(int(steps[0]), int(steps[-1]) + 1, dtype=np.int64)
    if not np.array_equal(steps, expected):
        raise WaveformAlignmentError(
            f"Non-contiguous step indices in waveform group: {steps.tolist()}."
        )
    return steps, times, ordered


def compute_signal_metrics(
    canonical: pd.DataFrame,
    *,
    dt: float,
    threshold_fraction: float,
) -> pd.DataFrame:
    """Return one row per case, receiver, boundary, and x/y component."""

    rows: list[dict[str, Any]] = []
    group_columns = _group_metadata_columns()
    for key, group in canonical.groupby(
        group_columns,
        sort=True,
        observed=True,
        dropna=False,
    ):
        metadata = dict(zip(group_columns, key, strict=True))
        steps, times, ordered = _sorted_signal_group(group)
        for case, column in (
            ("healthy", "healthy"),
            ("defect", "defect"),
            ("residual", "analysis_residual"),
        ):
            values = ordered[column].to_numpy(dtype=np.float64, copy=False)
            characteristics = signal_characteristics(
                values,
                steps=steps,
                times=times,
                dt=dt,
                threshold_fraction=threshold_fraction,
            )
            rows.append(
                {
                    **metadata,
                    "case": case,
                    "signal_type": "component_trace",
                    **characteristics,
                }
            )
    output = pd.DataFrame(rows)
    expected_rows = (
        canonical[["geometry", "residual_family", "receiver_id"]]
        .drop_duplicates()
        .shape[0]
        * len(EXPECTED_COMPONENTS)
        * len(SIGNAL_CASES)
    )
    if output.shape[0] != expected_rows:
        raise WaveformAlignmentError(
            f"Expected {expected_rows} signal-metric rows, found {output.shape[0]}."
        )
    key = [
        "geometry",
        "residual_family",
        "receiver_id",
        "component",
        "case",
    ]
    if output.duplicated(key).any():
        raise WaveformAlignmentError(f"Duplicate signal metrics for key {key}.")
    return output.sort_values(key, kind="mergesort", ignore_index=True)


def _component_series(
    group: pd.DataFrame,
    *,
    column: str,
    component: str,
    expected_steps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = group.loc[group["component"] == component].sort_values(
        "step_index", kind="mergesort"
    )
    if subset.empty:
        raise WaveformAlignmentError(f"Missing component {component!r}.")
    steps = subset["step_index"].to_numpy(dtype=np.int64, copy=False)
    times = subset["time_physical"].to_numpy(dtype=np.float64, copy=False)
    values = subset[column].to_numpy(dtype=np.float64, copy=False)
    if expected_steps is not None and not np.array_equal(steps, expected_steps):
        raise WaveformAlignmentError(
            f"Component {component!r} has incompatible step indices."
        )
    return steps, times, values


def compute_comparison_metrics(
    canonical: pd.DataFrame,
    *,
    dt: float,
    threshold_fraction: float,
    ratio_epsilon: float,
) -> pd.DataFrame:
    """Compare healthy and defective traces per receiver and component."""

    group_columns = [
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
        "residual_convention",
        "analysis_residual_convention",
        "source_csv",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in canonical.groupby(
        group_columns,
        sort=True,
        observed=True,
        dropna=False,
    ):
        metadata = dict(zip(group_columns, key, strict=True))
        x_steps, x_times, _ = _component_series(
            group, column="healthy", component="x"
        )
        y_steps, y_times, _ = _component_series(
            group,
            column="healthy",
            component="y",
            expected_steps=x_steps,
        )
        if not np.allclose(x_times, y_times, atol=0.0, rtol=0.0):
            raise WaveformAlignmentError("x and y components have incompatible times.")

        series: dict[str, dict[str, np.ndarray]] = {
            "x": {},
            "y": {},
            "vector": {},
        }
        for case, column in (
            ("healthy", "healthy"),
            ("defect", "defect"),
            ("residual", "analysis_residual"),
        ):
            _, _, x = _component_series(
                group,
                column=column,
                component="x",
                expected_steps=x_steps,
            )
            _, _, y = _component_series(
                group,
                column=column,
                component="y",
                expected_steps=x_steps,
            )
            series["x"][case] = x
            series["y"][case] = y
            series["vector"][case] = np.sqrt(x * x + y * y)

        for component in ("x", "y", "vector"):
            healthy = series[component]["healthy"]
            defect = series[component]["defect"]
            residual = series[component]["residual"]
            healthy_metrics = signal_characteristics(
                healthy,
                steps=x_steps,
                times=x_times,
                dt=dt,
                threshold_fraction=threshold_fraction,
            )
            defect_metrics = signal_characteristics(
                defect,
                steps=x_steps,
                times=x_times,
                dt=dt,
                threshold_fraction=threshold_fraction,
            )
            residual_metrics = signal_characteristics(
                residual,
                steps=x_steps,
                times=x_times,
                dt=dt,
                threshold_fraction=threshold_fraction,
            )
            ratio, ratio_status = safe_ratio(
                defect_metrics["absolute_max"],
                healthy_metrics["absolute_max"],
                epsilon=ratio_epsilon,
            )
            residual_relative_energy, energy_status = safe_ratio(
                residual_metrics["discrete_energy"],
                healthy_metrics["discrete_energy"],
                epsilon=ratio_epsilon,
            )
            correlation = normalized_cross_correlation_lag(
                healthy,
                defect,
                dt=dt,
                epsilon=ratio_epsilon,
            )
            rows.append(
                {
                    **metadata,
                    "component": component,
                    "num_steps": int(x_steps.size),
                    "dt": float(dt),
                    "time_min": float(x_times[0]),
                    "time_max": float(x_times[-1]),
                    "healthy_peak_amplitude": healthy_metrics["absolute_max"],
                    "defect_peak_amplitude": defect_metrics["absolute_max"],
                    "peak_amplitude_difference": (
                        defect_metrics["absolute_max"]
                        - healthy_metrics["absolute_max"]
                    ),
                    "peak_amplitude_ratio": ratio,
                    "peak_amplitude_ratio_status": ratio_status,
                    "healthy_absolute_peak_step": healthy_metrics[
                        "absolute_peak_step"
                    ],
                    "defect_absolute_peak_step": defect_metrics[
                        "absolute_peak_step"
                    ],
                    "healthy_peak_time": healthy_metrics["absolute_peak_time"],
                    "defect_peak_time": defect_metrics["absolute_peak_time"],
                    "peak_time_shift": (
                        defect_metrics["absolute_peak_time"]
                        - healthy_metrics["absolute_peak_time"]
                    ),
                    "healthy_envelope_peak_time": healthy_metrics[
                        "envelope_peak_time"
                    ],
                    "defect_envelope_peak_time": defect_metrics[
                        "envelope_peak_time"
                    ],
                    "envelope_peak_time_shift": (
                        defect_metrics["envelope_peak_time"]
                        - healthy_metrics["envelope_peak_time"]
                    ),
                    "healthy_threshold_arrival_time": healthy_metrics[
                        "threshold_arrival_time"
                    ],
                    "defect_threshold_arrival_time": defect_metrics[
                        "threshold_arrival_time"
                    ],
                    "threshold_arrival_time_shift": None
                    if healthy_metrics["threshold_arrival_time"] is None
                    or defect_metrics["threshold_arrival_time"] is None
                    else (
                        float(defect_metrics["threshold_arrival_time"])
                        - float(healthy_metrics["threshold_arrival_time"])
                    ),
                    "healthy_energy": healthy_metrics["discrete_energy"],
                    "defect_energy": defect_metrics["discrete_energy"],
                    "residual_energy": residual_metrics["discrete_energy"],
                    "residual_relative_energy": residual_relative_energy,
                    "residual_relative_energy_status": energy_status,
                    **correlation,
                }
            )
    output = pd.DataFrame(rows)
    key = ["geometry", "residual_family", "receiver_id", "component"]
    if output.duplicated(key).any():
        raise WaveformAlignmentError(f"Duplicate comparison metrics for key {key}.")
    return output.sort_values(key, kind="mergesort", ignore_index=True)


def compute_network_summary(signal_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics without discarding per-receiver results."""

    group_columns = [
        "geometry",
        "residual_family",
        "array_name",
        "role",
        "case",
        "component",
        "analysis_residual_convention",
    ]
    work = signal_metrics.copy()
    work["weighted_energy"] = (
        work["weight_all"].to_numpy(dtype=np.float64)
        * work["discrete_energy"].to_numpy(dtype=np.float64)
    )
    work["weighted_absolute_peak"] = (
        work["weight_all"].to_numpy(dtype=np.float64)
        * work["absolute_max"].to_numpy(dtype=np.float64)
    )
    grouped = work.groupby(group_columns, sort=True, observed=True)
    output = grouped.agg(
        num_receivers=("receiver_id", "nunique"),
        aperture_length=("weight_all", "sum"),
        absolute_peak_mean=("absolute_max", "mean"),
        absolute_peak_median=("absolute_max", "median"),
        absolute_peak_max=("absolute_max", "max"),
        envelope_peak_mean=("envelope_max", "mean"),
        threshold_arrival_time_mean=("threshold_arrival_time", "mean"),
        threshold_arrival_time_min=("threshold_arrival_time", "min"),
        threshold_arrival_time_max=("threshold_arrival_time", "max"),
        discrete_energy_sum_receivers=("discrete_energy", "sum"),
        discrete_energy_mean_receiver=("discrete_energy", "mean"),
        weighted_energy_integral=("weighted_energy", "sum"),
        weighted_absolute_peak_integral=("weighted_absolute_peak", "sum"),
    ).reset_index()
    length = output["aperture_length"].to_numpy(dtype=np.float64)
    if np.any(length <= 0.0):
        raise ReceiverGeometryError("Aperture length must be strictly positive.")
    output["weighted_energy_density"] = (
        output["weighted_energy_integral"].to_numpy(dtype=np.float64) / length
    )
    output["weighted_absolute_peak_mean"] = (
        output["weighted_absolute_peak_integral"].to_numpy(dtype=np.float64) / length
    )
    return output.sort_values(group_columns, kind="mergesort", ignore_index=True)


def select_waveform_receivers(
    receiver_geometry: pd.DataFrame,
    *,
    receiver_ids: tuple[int, ...] | None,
) -> pd.DataFrame:
    """Select deterministic representative receivers or explicit IDs."""

    if receiver_ids is None:
        selected = select_representative_receivers(receiver_geometry)
        selected["selection_mode"] = "deterministic_representative"
        return selected

    subset = receiver_geometry.loc[
        receiver_geometry["receiver_id"].isin(receiver_ids),
        [
            "geometry",
            "residual_family",
            "receiver_id",
            "center_x",
            "center_y",
            "side_membership",
        ],
    ].copy()
    found = set(int(v) for v in subset["receiver_id"].unique())
    missing = sorted(set(receiver_ids) - found)
    if missing:
        raise WaveformAlignmentError(
            f"Requested receiver IDs do not exist in any selected array: {missing}."
        )
    subset["selection_rule"] = "explicit_receiver_id"
    subset["selection_mode"] = "explicit"
    return subset.sort_values(
        ["geometry", "residual_family", "receiver_id"],
        kind="mergesort",
        ignore_index=True,
    )


def build_selected_waveform_traces(
    canonical: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["geometry", "residual_family", "receiver_id", "center_x", "center_y"]
    selected = canonical.merge(
        selection,
        on=keys,
        how="inner",
        validate="many_to_one",
        suffixes=("", "_selection"),
    )
    rows: list[pd.DataFrame] = []
    group_columns = [
        "geometry",
        "residual_family",
        "array_name",
        "role",
        "receiver_id",
        "center_x",
        "center_y",
        "side_membership",
        "selection_rule",
        "selection_mode",
        "component",
        "analysis_residual_convention",
        "source_csv",
    ]
    for key, group in selected.groupby(
        group_columns,
        sort=True,
        observed=True,
        dropna=False,
    ):
        metadata = dict(zip(group_columns, key, strict=True))
        ordered = group.sort_values("step_index", kind="mergesort")
        healthy = ordered["healthy"].to_numpy(dtype=np.float64, copy=False)
        defect = ordered["defect"].to_numpy(dtype=np.float64, copy=False)
        residual = ordered["analysis_residual"].to_numpy(dtype=np.float64, copy=False)
        frame = pd.DataFrame(
            {
                **{name: [value] * ordered.shape[0] for name, value in metadata.items()},
                "step_index": ordered["step_index"].to_numpy(dtype=np.int64),
                "time_original": ordered["time_original"].to_numpy(dtype=np.float64),
                "time_physical": ordered["time_physical"].to_numpy(dtype=np.float64),
                "healthy": healthy,
                "defect": defect,
                "residual": residual,
                "healthy_envelope": analytic_envelope(healthy),
                "defect_envelope": analytic_envelope(defect),
                "residual_envelope": analytic_envelope(residual),
            }
        )
        rows.append(frame)
    if not rows:
        raise WaveformAlignmentError("Receiver selection produced no waveform traces.")
    output = pd.concat(rows, ignore_index=True)
    return output.sort_values(
        [
            "geometry",
            "residual_family",
            "receiver_id",
            "component",
            "step_index",
        ],
        kind="mergesort",
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Main reusable pipeline
# ---------------------------------------------------------------------------


def run_vector_array_waveform_analysis(
    config: WaveformAnalysisConfig,
    *,
    logger: logging.Logger | None = None,
) -> WaveformAnalysisResult:
    start = time.perf_counter()
    log = logger or logging.getLogger(__name__)
    inputs = config.inputs.resolved()

    # Reuse the strict reader and geometry contract of receiver_aperture_comparison.
    base_config = ApertureComparisonConfig(
        inputs=ApertureComparisonInputs(
            line_residuals=inputs.line_residuals,
            box_residuals=inputs.box_residuals,
            line_manifest=inputs.line_manifest,
            box_manifest=inputs.box_manifest,
        ),
        output_dir=config.output_dir,
        dt=config.dt,
        time_atol=config.time_atol,
        time_rtol=config.time_rtol,
        value_atol=config.value_atol,
        value_rtol=config.value_rtol,
        geometry_atol=config.geometry_atol,
        ratio_epsilon=config.ratio_epsilon,
        time_column_policy=config.time_column_policy,
        corner_policy="shared",
        overwrite=config.overwrite,
        quiet=config.quiet,
        geometry=config.geometry,
    )

    if inputs.line_manifest is None:
        raise WaveformAnalysisError("line_manifest is required.")
    line_specs = load_receiver_specs(inputs.line_manifest, geometry="line")
    specs_by_geometry: dict[str, Mapping[str, ReceiverSpec]] = {
        "line": line_specs,
    }
    if config.geometry == "line-box":
        if inputs.box_manifest is None:
            raise WaveformAnalysisError(
                "box_manifest is required in line-box mode."
            )
        box_specs = load_receiver_specs(inputs.box_manifest, geometry="box")
        specs_by_geometry["box"] = box_specs
        log.info("Loaded validated receiver centres from line and box manifests.")
    else:
        box_specs = None
        log.info("Loaded validated receiver centres from the line manifest.")

    line_frame, line_inspection = load_and_validate_campaign(
        inputs.line_residuals,
        geometry="line",
        specs=line_specs,
        config=base_config,
    )
    if config.geometry == "line-box":
        if inputs.box_residuals is None or box_specs is None:
            raise WaveformAnalysisError(
                "box_residuals and box receiver specifications are required "
                "in line-box mode."
            )
        box_frame, box_inspection = load_and_validate_campaign(
            inputs.box_residuals,
            geometry="box",
            specs=box_specs,
            config=base_config,
        )
        inspections = (line_inspection, box_inspection)
        canonical = pd.concat([line_frame, box_frame], ignore_index=True)
        log.info(
            "Validated waveform inputs: line=%d rows, box=%d rows.",
            line_frame.shape[0],
            box_frame.shape[0],
        )
    else:
        inspections = (line_inspection,)
        canonical = line_frame.copy()
        log.info("Validated line waveform input: %d rows.", line_frame.shape[0])
    receiver_geometry = build_receiver_geometry(
        specs_by_geometry,
        geometry_atol=config.geometry_atol,
        corner_policy="shared",
    )
    canonical = attach_receiver_geometry(canonical, receiver_geometry)
    canonical, analysis_convention = apply_analysis_residual_convention(
        canonical,
        convention=config.residual_convention,
        atol=config.value_atol,
        rtol=config.value_rtol,
    )
    log.info(
        "Analysis residual convention: %s (source export remains unchanged).",
        analysis_convention,
    )

    time_validation = validate_protocol_time_interval(
        inspections,
        dt=config.dt,
        final_time=config.final_time,
        atol=config.time_atol,
        rtol=config.time_rtol,
    )
    signal_metrics = compute_signal_metrics(
        canonical,
        dt=config.dt,
        threshold_fraction=config.arrival_threshold,
    )
    comparison_metrics = compute_comparison_metrics(
        canonical,
        dt=config.dt,
        threshold_fraction=config.arrival_threshold,
        ratio_epsilon=config.ratio_epsilon,
    )
    network_summary = compute_network_summary(signal_metrics)
    selection = select_waveform_receivers(
        receiver_geometry,
        receiver_ids=config.receiver_ids,
    )
    selected_traces = build_selected_waveform_traces(canonical, selection)

    summary = {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "analysis_mode": (
            "line-only" if config.geometry == "line" else "line-box"
        ),
        "healthy_definition": "reference column; simulated healthy medium",
        "defect_definition": "observed column; defective medium",
        "source_residual_convention": sorted(
            str(v) for v in canonical["residual_convention"].unique()
        )[0],
        "analysis_residual_convention": analysis_convention,
        "dt": float(config.dt),
        "protocol_final_time": float(config.final_time),
        "arrival_threshold_fraction": float(config.arrival_threshold),
        "num_input_rows": int(canonical.shape[0]),
        "num_signal_metric_rows": int(signal_metrics.shape[0]),
        "num_comparison_metric_rows": int(comparison_metrics.shape[0]),
        "num_network_summary_rows": int(network_summary.shape[0]),
        "num_selected_receivers": int(selection.shape[0]),
        "geometries": sorted(str(v) for v in canonical["geometry"].unique()),
        "residual_families": sorted(
            str(v) for v in canonical["residual_family"].unique()
        ),
        "components": sorted(str(v) for v in canonical["component"].unique()),
        "receiver_counts": {
            inspection.geometry: dict(inspection.receiver_counts)
            for inspection in inspections
        },
        "sampling": time_validation.to_dict("records"),
        "threshold_arrival_interpretation": (
            "first sample where the unsmoothed analytic envelope reaches "
            "arrival_threshold_fraction times its own maximum; this is a "
            "diagnostic threshold time, not automatically a physical arrival time"
        ),
        "warnings": [
            "The historical CSV time column is a step-index alias; physical time "
            "is reconstructed as step_index*dt."
        ]
        if any(item.time_status == "legacy_step_index_alias" for item in inspections)
        else [],
    }
    elapsed = float(time.perf_counter() - start)
    log.info("Computed waveform metrics in %.6f s.", elapsed)
    return WaveformAnalysisResult(
        inspections=inspections,
        canonical=canonical,
        receiver_geometry=receiver_geometry,
        time_validation=time_validation,
        receiver_selection=selection,
        selected_traces=selected_traces,
        signal_metrics=signal_metrics,
        comparison_metrics=comparison_metrics,
        network_summary=network_summary,
        summary=summary,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Outputs and plots
# ---------------------------------------------------------------------------


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _write_dataframe(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False),
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
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "numpy": getattr(np, "__version__", None),
        "pandas": getattr(pd, "__version__", None),
    }
    try:
        import matplotlib

        versions["matplotlib"] = getattr(matplotlib, "__version__", None)
    except Exception:
        versions["matplotlib"] = None
    return versions


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _trace_for_component(group: pd.DataFrame, component: str) -> pd.DataFrame:
    subset = group.loc[group["component"] == component].sort_values(
        "step_index", kind="mergesort"
    )
    if subset.empty:
        raise WaveformAlignmentError(f"Missing selected trace component {component!r}.")
    return subset


def _plot_selected_receiver(
    path_png: Path,
    path_pdf: Path,
    group: pd.DataFrame,
    *,
    final_time: float,
    arrival_threshold: float,
) -> tuple[Path, Path]:
    plt = _configure_matplotlib()
    x = _trace_for_component(group, "x")
    y = _trace_for_component(group, "y")
    if not np.array_equal(x["step_index"].to_numpy(), y["step_index"].to_numpy()):
        raise WaveformAlignmentError("Selected x/y traces do not share step indices.")
    time_values = x["time_physical"].to_numpy(dtype=np.float64, copy=False)
    residual_norm = np.sqrt(
        x["residual"].to_numpy(dtype=np.float64, copy=False) ** 2
        + y["residual"].to_numpy(dtype=np.float64, copy=False) ** 2
    )

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 10.0), sharex=True)
    axes = axes.ravel()

    axes[0].plot(time_values, x["healthy"], label="healthy/reference")
    axes[0].plot(time_values, x["defect"], label="defect/observed")
    axes[0].set_title(r"$u_x$: healthy and defect")

    axes[1].plot(time_values, y["healthy"], label="healthy/reference")
    axes[1].plot(time_values, y["defect"], label="defect/observed")
    axes[1].set_title(r"$u_y$: healthy and defect")

    axes[2].plot(time_values, x["residual"], label=r"$r_x$")
    axes[2].plot(time_values, y["residual"], label=r"$r_y$")
    axes[2].plot(time_values, residual_norm, linestyle="--", label="residual vector norm")
    axes[2].set_title("Residual components and vector norm")

    axes[3].plot(time_values, x["healthy_envelope"], label="healthy envelope")
    axes[3].plot(time_values, x["defect_envelope"], label="defect envelope")
    axes[3].set_title(r"$u_x$ analytic envelopes")

    axes[4].plot(time_values, y["healthy_envelope"], label="healthy envelope")
    axes[4].plot(time_values, y["defect_envelope"], label="defect envelope")
    axes[4].set_title(r"$u_y$ analytic envelopes")

    axes[5].plot(time_values, x["residual_envelope"], label=r"$r_x$ envelope")
    axes[5].plot(time_values, y["residual_envelope"], label=r"$r_y$ envelope")
    axes[5].set_title(
        f"Residual envelopes; threshold fraction={arrival_threshold:g}"
    )

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.set_xlim(0.0, float(final_time))
        axis.set_ylabel("Observation amplitude (model units)")
        axis.legend(loc="best", fontsize=8)
    axes[-2].set_xlabel("Physical time")
    axes[-1].set_xlabel("Physical time")

    first = group.iloc[0]
    fig.suptitle(
        "Vector-array waveforms — "
        f"{first['geometry']} / {first['residual_family']} / "
        f"receiver {int(first['receiver_id'])} at "
        f"({float(first['center_x']):.3f}, {float(first['center_y']):.3f}); "
        f"residual={first['analysis_residual_convention']}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_pdf)
    plt.close(fig)
    return path_png, path_pdf


def _plot_network_metric(
    path_png: Path,
    path_pdf: Path,
    comparison: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    yscale: str = "linear",
) -> tuple[Path, Path]:
    plt = _configure_matplotlib()
    data = comparison.loc[comparison["component"].isin(["x", "y"])].copy()
    preferred = (
        ("line", "reflection"),
        ("box", "reflection"),
        ("line", "transmission"),
        ("box", "transmission"),
    )
    available = {
        (str(geometry), str(family))
        for geometry, family in data[
            ["geometry", "residual_family"]
        ].drop_duplicates().itertuples(index=False, name=None)
    }
    panels = [item for item in preferred if item in available]
    if not panels:
        raise WaveformMetricError(
            f"No x/y comparison rows are available for metric {metric!r}."
        )

    if len(panels) <= 2:
        nrows, ncols = 1, len(panels)
        figsize = (6.0 * ncols, 4.5)
    else:
        nrows, ncols = 2, 2
        figsize = (12.0, 8.0)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=False,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for axis, (geometry, family) in zip(flat_axes, panels, strict=False):
        subset = data.loc[
            (data["geometry"] == geometry)
            & (data["residual_family"] == family)
        ].sort_values(["component", "receiver_id"], kind="mergesort")
        for component, component_rows in subset.groupby(
            "component", sort=True, observed=True
        ):
            axis.plot(
                component_rows["receiver_id"],
                component_rows[metric],
                marker="o",
                markersize=2.5,
                label=str(component),
            )
        axis.set_title(f"{geometry} / {family}")
        axis.set_xlabel("Receiver ID")
        axis.set_ylabel(ylabel)
        axis.set_yscale(yscale)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    for axis in flat_axes[len(panels) :]:
        axis.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_pdf)
    plt.close(fig)
    return path_png, path_pdf


def write_waveform_analysis_outputs(
    result: WaveformAnalysisResult,
    config: WaveformAnalysisConfig,
    *,
    command: Sequence[str],
    repository_root: Path,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    log = logger or logging.getLogger(__name__)
    paths = WaveformOutputPaths.from_root(config.output_dir)
    for directory in (
        paths.csv_dir,
        paths.figures_dir,
        paths.manifests_dir,
        paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    outputs["signal_metrics_csv"] = _write_dataframe(
        paths.csv_dir / "waveform_signal_metrics.csv",
        result.signal_metrics,
    )
    outputs["comparison_metrics_csv"] = _write_dataframe(
        paths.csv_dir / "healthy_defect_comparison.csv",
        result.comparison_metrics,
    )
    outputs["network_summary_csv"] = _write_dataframe(
        paths.csv_dir / "waveform_network_summary.csv",
        result.network_summary,
    )
    outputs["time_validation_csv"] = _write_dataframe(
        paths.csv_dir / "waveform_time_validation.csv",
        result.time_validation,
    )
    outputs["receiver_selection_csv"] = _write_dataframe(
        paths.csv_dir / "waveform_receiver_selection.csv",
        result.receiver_selection,
    )
    outputs["selected_traces_csv"] = _write_dataframe(
        paths.csv_dir / "selected_waveform_traces.csv",
        result.selected_traces,
    )

    figure_records: list[dict[str, str]] = []
    grouping = ["geometry", "residual_family", "receiver_id"]
    for key, group in result.selected_traces.groupby(
        grouping,
        sort=True,
        observed=True,
    ):
        geometry, family, receiver_id = key
        folder = paths.figures_dir / str(geometry) / str(family)
        stem = f"receiver_{int(receiver_id):03d}_waveforms"
        png = folder / f"{stem}.png"
        pdf = folder / f"{stem}.pdf"
        _plot_selected_receiver(
            png,
            pdf,
            group,
            final_time=config.final_time,
            arrival_threshold=config.arrival_threshold,
        )
        figure_records.append(
            {
                "geometry": str(geometry),
                "residual_family": str(family),
                "receiver_id": str(int(receiver_id)),
                "png": str(png),
                "pdf": str(pdf),
            }
        )

    summary_figures = (
        (
            "peak_amplitude_ratio",
            "peak_amplitude_ratio",
            "Defect/healthy absolute peak-amplitude ratio",
            "Defect-to-healthy peak-amplitude ratio (logarithmic scale)",
            "log",
        ),
        (
            "peak_time_shift",
            "peak_time_shift",
            "Peak-time shift (defect minus healthy)",
            "Healthy/defective absolute-peak time shifts",
            "linear",
        ),
        (
            "residual_energy",
            "residual_energy",
            r"Residual discrete energy $\Delta t\sum_n r^2$",
            "Residual energy by receiver and component",
            "log",
        ),
    )
    for name, metric, ylabel, title, yscale in summary_figures:
        png = paths.figures_dir / f"waveform_{name}.png"
        pdf = paths.figures_dir / f"waveform_{name}.pdf"
        _plot_network_metric(
            png,
            pdf,
            result.comparison_metrics,
            metric=metric,
            ylabel=ylabel,
            title=title,
            yscale=yscale,
        )
        outputs[f"figure_{name}_png"] = png
        outputs[f"figure_{name}_pdf"] = pdf

    summary_payload = dict(result.summary)
    summary_payload["elapsed_seconds"] = result.elapsed_seconds
    summary_payload["figures"] = figure_records
    summary_path = paths.manifests_dir / "vector_array_waveform_analysis_summary.json"
    outputs["summary_json"] = _write_json(summary_path, summary_payload)

    inputs = config.inputs.resolved()
    input_payload = {
        "line_residuals": _file_metadata(inputs.line_residuals),
        "line_manifest": _file_metadata(inputs.line_manifest),
    }
    if inputs.box_residuals is not None and inputs.box_manifest is not None:
        input_payload.update(
            {
                "box_residuals": _file_metadata(inputs.box_residuals),
                "box_manifest": _file_metadata(inputs.box_manifest),
            }
        )
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "vector_array_waveform_analysis_manifest",
        "status": "ok",
        "analysis_mode": result.summary["analysis_mode"],
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "inputs": input_payload,
        "configuration": {
            "geometry": config.geometry,
            "dt": config.dt,
            "dt_source": "explicit_cli_or_config",
            "final_time": config.final_time,
            "arrival_threshold": config.arrival_threshold,
            "requested_residual_convention": config.residual_convention,
            "analysis_residual_convention": result.summary[
                "analysis_residual_convention"
            ],
            "time_column_policy": config.time_column_policy,
            "time_atol": config.time_atol,
            "time_rtol": config.time_rtol,
            "value_atol": config.value_atol,
            "value_rtol": config.value_rtol,
            "geometry_atol": config.geometry_atol,
            "ratio_epsilon": config.ratio_epsilon,
            "receiver_ids": config.receiver_ids,
        },
        "definitions": {
            "healthy": "reference column (simulated healthy medium)",
            "defect": "observed column (defective medium)",
            "residual": result.summary["analysis_residual_convention"],
            "analytic_envelope": (
                "absolute value of the unsmoothed FFT Hilbert analytic signal"
            ),
            "threshold_arrival_time": (
                "first sample where the signal envelope is >= arrival_threshold "
                "times that envelope's maximum"
            ),
            "discrete_energy": "dt * sum_n signal[n]^2",
            "cross_correlation_lag": (
                "lag maximizing full normalized cross-correlation of demeaned "
                "defect against healthy; positive means defect delayed"
            ),
        },
        "counts": {
            "input_rows": int(result.canonical.shape[0]),
            "signal_metric_rows": int(result.signal_metrics.shape[0]),
            "comparison_metric_rows": int(result.comparison_metrics.shape[0]),
            "network_summary_rows": int(result.network_summary.shape[0]),
            "selected_receiver_records": int(result.receiver_selection.shape[0]),
            "selected_trace_rows": int(result.selected_traces.shape[0]),
        },
        "validations": {
            "sampling": result.time_validation.to_dict("records"),
            "input_inspections": [item.as_dict() for item in result.inspections],
            "status": "ok",
        },
        "software": {
            "dependencies": _dependency_versions(),
            "git_commit": _git_commit(repository_root),
        },
        "produced_files": {
            key: str(value) for key, value in outputs.items()
        },
        "representative_figures": figure_records,
        "elapsed_seconds": result.elapsed_seconds,
    }
    manifest_path = paths.manifests_dir / "vector_array_waveform_analysis_manifest.json"
    outputs["manifest_json"] = _write_json(manifest_path, manifest_payload)

    # Update manifest with the manifest path itself only in the returned mapping;
    # avoid a self-hash recursion.
    log.info("Wrote %d waveform output artifacts (excluding log).", len(outputs))
    return outputs


__all__ = [
    "ALLOWED_RESIDUAL_CONVENTIONS",
    "SCHEMA_VERSION",
    "SamplingIntervalError",
    "WaveformAlignmentError",
    "WaveformAnalysisConfig",
    "WaveformAnalysisError",
    "WaveformAnalysisInputs",
    "WaveformAnalysisResult",
    "WaveformMetricError",
    "WaveformOutputPaths",
    "analytic_envelope",
    "apply_analysis_residual_convention",
    "build_selected_waveform_traces",
    "compute_comparison_metrics",
    "compute_network_summary",
    "compute_signal_metrics",
    "normalized_cross_correlation_lag",
    "run_vector_array_waveform_analysis",
    "safe_ratio",
    "select_waveform_receivers",
    "signal_characteristics",
    "threshold_crossing_index",
    "validate_protocol_time_interval",
    "write_waveform_analysis_outputs",
]
