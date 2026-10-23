from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ResidualSpectrumSummary:
    name: str
    frequencies: np.ndarray
    mean_power_spectrum: np.ndarray
    total_power_spectrum: np.ndarray
    dominant_frequency: float
    effective_bandwidth: float
    peak_power: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "residual_spectrum_summary",
            "name": self.name,
            "num_frequencies": int(self.frequencies.size),
            "dominant_frequency": float(self.dominant_frequency),
            "effective_bandwidth": float(self.effective_bandwidth),
            "peak_power": float(self.peak_power),
        }


def _as_finite_array(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values.")
    return arr


def infer_uniform_dt(times: Any, *, rtol: float = 1.0e-8, atol: float = 1.0e-12) -> float:
    times_arr = _as_finite_array(times, name="times").reshape(-1)
    if times_arr.size < 2:
        raise ValueError("At least two times are required for FFT analysis.")

    dt = np.diff(times_arr)
    if np.any(dt <= 0.0):
        raise ValueError("times must be strictly increasing.")

    if not np.allclose(dt, dt[0], rtol=rtol, atol=atol):
        raise ValueError("times must be uniformly spaced for FFT analysis.")

    return float(dt[0])


def compute_residual_spectrum(
    *,
    times: Any,
    values: Any,
    name: str,
    relative_band_threshold: float = 0.1,
) -> ResidualSpectrumSummary:
    times_arr = _as_finite_array(times, name="times").reshape(-1)
    arr = _as_finite_array(values, name="values")

    if arr.ndim < 1 or arr.shape[0] != times_arr.size:
        raise ValueError("values must have shape (num_steps, ...).")

    threshold = float(relative_band_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("relative_band_threshold must be non-negative and finite.")

    dt = infer_uniform_dt(times_arr)

    transformed = np.fft.rfft(arr, axis=0)
    power = np.abs(transformed) ** 2

    if power.ndim == 1:
        total_power = power
        mean_power = power
    else:
        axes = tuple(range(1, power.ndim))
        total_power = np.sum(power, axis=axes)
        mean_power = np.mean(power, axis=axes)

    frequencies = np.fft.rfftfreq(times_arr.size, d=dt)

    if total_power.size == 0 or float(np.max(total_power)) <= 0.0:
        dominant = 0.0
        peak = 0.0
        bandwidth = 0.0
    else:
        idx = int(np.argmax(total_power))
        dominant = float(frequencies[idx])
        peak = float(total_power[idx])
        active = total_power >= threshold * peak
        if np.any(active):
            bandwidth = float(frequencies[active][-1] - frequencies[active][0])
        else:
            bandwidth = 0.0

    return ResidualSpectrumSummary(
        name=str(name),
        frequencies=frequencies,
        mean_power_spectrum=np.asarray(mean_power, dtype=float),
        total_power_spectrum=np.asarray(total_power, dtype=float),
        dominant_frequency=dominant,
        effective_bandwidth=bandwidth,
        peak_power=peak,
    )
