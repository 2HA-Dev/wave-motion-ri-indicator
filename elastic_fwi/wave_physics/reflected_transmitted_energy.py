from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SignalEnergySummary:
    name: str
    times: np.ndarray
    instantaneous_energy: np.ndarray
    cumulative_energy: np.ndarray
    total_energy: float
    peak_energy: float
    time_of_peak: float
    index_of_peak: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "signal_energy_summary",
            "name": self.name,
            "num_steps": int(self.times.size),
            "total_energy": float(self.total_energy),
            "peak_energy": float(self.peak_energy),
            "time_of_peak": float(self.time_of_peak),
            "index_of_peak": int(self.index_of_peak),
        }


@dataclass(frozen=True)
class ReflectedTransmittedEnergySummary:
    reflection: SignalEnergySummary
    transmission: SignalEnergySummary

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "reflected_transmitted_energy_summary",
            "reflection": self.reflection.as_dict(),
            "transmission": self.transmission.as_dict(),
            "total_reflected_energy": float(self.reflection.total_energy),
            "total_transmitted_energy": float(self.transmission.total_energy),
        }


def _as_finite_array(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def signal_instantaneous_energy(values: Any) -> np.ndarray:
    arr = _as_finite_array(values, name="values")
    if arr.ndim < 1:
        raise ValueError("values must have a leading time dimension.")
    axes = tuple(range(1, arr.ndim))
    if axes:
        return np.sum(arr * arr, axis=axes)
    return arr * arr


def cumulative_time_integral(times: Any, instantaneous: Any) -> np.ndarray:
    times_arr = _as_finite_array(times, name="times").reshape(-1)
    inst = _as_finite_array(instantaneous, name="instantaneous").reshape(-1)

    if times_arr.shape != inst.shape:
        raise ValueError("times and instantaneous arrays must have the same shape.")

    if inst.size == 0:
        return inst.copy()

    if inst.size == 1:
        return np.asarray([0.0], dtype=float)

    dt = np.diff(times_arr)
    if not np.all(dt >= 0.0):
        raise ValueError("times must be non-decreasing.")

    increments = 0.5 * (inst[:-1] + inst[1:]) * dt
    return np.concatenate([[0.0], np.cumsum(increments)])


def summarize_signal_energy(
    *,
    times: Any,
    values: Any,
    name: str,
) -> SignalEnergySummary:
    times_arr = _as_finite_array(times, name="times").reshape(-1)
    inst = signal_instantaneous_energy(values).reshape(-1)

    if inst.shape != times_arr.shape:
        raise ValueError("times length must match values.shape[0].")

    cumulative = cumulative_time_integral(times_arr, inst)

    if inst.size:
        peak_index = int(np.argmax(inst))
        peak_energy = float(inst[peak_index])
        time_peak = float(times_arr[peak_index])
        total = float(cumulative[-1])
    else:
        peak_index = 0
        peak_energy = 0.0
        time_peak = 0.0
        total = 0.0

    return SignalEnergySummary(
        name=str(name),
        times=times_arr,
        instantaneous_energy=inst,
        cumulative_energy=cumulative,
        total_energy=total,
        peak_energy=peak_energy,
        time_of_peak=time_peak,
        index_of_peak=peak_index,
    )


def summarize_reflected_transmitted_energy(
    *,
    times: Any,
    reflection_values: Any,
    transmission_values: Any,
) -> ReflectedTransmittedEnergySummary:
    return ReflectedTransmittedEnergySummary(
        reflection=summarize_signal_energy(
            times=times,
            values=reflection_values,
            name="reflection",
        ),
        transmission=summarize_signal_energy(
            times=times,
            values=transmission_values,
            name="transmission",
        ),
    )
