from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class TimeWindow:
    name: str
    start_time: float
    end_time: float

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.start_time)):
            raise ValueError("start_time must be finite.")
        if not np.isfinite(float(self.end_time)):
            raise ValueError("end_time must be finite.")
        if float(self.end_time) < float(self.start_time):
            raise ValueError("end_time must be greater than or equal to start_time.")

    def mask(self, times: Any) -> np.ndarray:
        times_arr = np.asarray(times, dtype=float).reshape(-1)
        return (times_arr >= float(self.start_time)) & (times_arr <= float(self.end_time))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_time": float(self.start_time),
            "end_time": float(self.end_time),
        }


def default_time_windows(times: Any) -> list[TimeWindow]:
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    if times_arr.size == 0:
        raise ValueError("times must not be empty.")

    t0 = float(times_arr[0])
    t1 = float(times_arr[-1])

    if t1 < t0:
        raise ValueError("times must be non-decreasing.")

    duration = t1 - t0
    q1 = t0 + 0.25 * duration
    q2 = t0 + 0.50 * duration
    q3 = t0 + 0.75 * duration

    return [
        TimeWindow("early", t0, q1),
        TimeWindow("defect_interaction", q1, q2),
        TimeWindow("transmitted_arrival", q2, q3),
        TimeWindow("late_reflections", q3, t1),
        TimeWindow("full", t0, t1),
    ]


def parse_time_window(text: str) -> TimeWindow:
    parts = [part.strip() for part in str(text).split(":")]
    if len(parts) != 3:
        raise ValueError("Time windows must use name:start:end syntax.")
    return TimeWindow(parts[0], float(parts[1]), float(parts[2]))



def trapezoid_integral(values: np.ndarray, times: np.ndarray) -> float:
    """Compatibility wrapper for NumPy >= 2.0 and older NumPy releases."""
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    times_arr = np.asarray(times, dtype=float).reshape(-1)

    if values_arr.shape != times_arr.shape:
        raise ValueError("values and times must have the same shape.")

    if values_arr.size < 2:
        return 0.0

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values_arr, times_arr))

    return float(np.trapz(values_arr, times_arr))


def time_integral_over_mask(times: np.ndarray, values: np.ndarray, mask: np.ndarray) -> float:
    selected_times = np.asarray(times, dtype=float).reshape(-1)[mask]
    selected_values = np.asarray(values, dtype=float).reshape(-1)[mask]

    if selected_times.size == 0:
        return 0.0

    if selected_times.size == 1:
        return 0.0

    return trapezoid_integral(selected_values, selected_times)


def summarize_windowed_energies(
    *,
    times: Any,
    reflection_energy: Any,
    transmission_energy: Any,
    windows: Iterable[TimeWindow],
    epsilon: float = 1.0e-30,
) -> list[dict[str, Any]]:
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    ref = np.asarray(reflection_energy, dtype=float).reshape(-1)
    trans = np.asarray(transmission_energy, dtype=float).reshape(-1)

    if times_arr.shape != ref.shape or times_arr.shape != trans.shape:
        raise ValueError("times, reflection_energy and transmission_energy must match.")

    total_ref = max(trapezoid_integral(ref, times_arr), 0.0)
    total_trans = max(trapezoid_integral(trans, times_arr), 0.0)

    rows: list[dict[str, Any]] = []

    for window in windows:
        mask = window.mask(times_arr)
        ref_e = time_integral_over_mask(times_arr, ref, mask)
        trans_e = time_integral_over_mask(times_arr, trans, mask)

        rows.append(
            {
                "window": window.name,
                "start_time": float(window.start_time),
                "end_time": float(window.end_time),
                "num_samples": int(np.count_nonzero(mask)),
                "reflection_energy": float(ref_e),
                "transmission_energy": float(trans_e),
                "reflection_transmission_ratio": float(ref_e / (trans_e + epsilon)),
                "reflection_fraction": float(ref_e / (total_ref + epsilon)),
                "transmission_fraction": float(trans_e / (total_trans + epsilon)),
            }
        )

    return rows
