from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScatteredFieldSummary:
    name: str
    times: np.ndarray
    values: np.ndarray
    energy: np.ndarray
    norm: np.ndarray
    max_energy: float
    max_norm: float
    time_of_max_energy: float
    index_of_max_energy: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "scattered_field_summary",
            "name": self.name,
            "num_steps": int(self.times.size),
            "shape": list(self.values.shape),
            "max_energy": float(self.max_energy),
            "max_norm": float(self.max_norm),
            "time_of_max_energy": float(self.time_of_max_energy),
            "index_of_max_energy": int(self.index_of_max_energy),
            "total_energy_sum": float(np.sum(self.energy)),
        }


def _as_finite_array(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def compute_scattered_field(*, defect: Any, reference: Any) -> np.ndarray:
    """Return the defect-induced scattered field u_scat = u_defect - u_ref."""
    defect_arr = _as_finite_array(defect, name="defect")
    reference_arr = _as_finite_array(reference, name="reference")
    if defect_arr.shape != reference_arr.shape:
        raise ValueError("defect and reference arrays must have the same shape.")
    return defect_arr - reference_arr


def summarize_scattered_field(
    *,
    times: Any,
    scattered: Any,
    name: str = "scattered_field",
) -> ScatteredFieldSummary:
    times_arr = _as_finite_array(times, name="times").reshape(-1)
    values = _as_finite_array(scattered, name="scattered")

    if values.ndim < 1:
        raise ValueError("scattered must have at least one time dimension.")
    if values.shape[0] != times_arr.size:
        raise ValueError("times length must match scattered.shape[0].")

    reduce_axes = tuple(range(1, values.ndim))
    if reduce_axes:
        energy = np.sum(values * values, axis=reduce_axes)
    else:
        energy = values * values

    energy = np.asarray(energy, dtype=float).reshape(-1)
    norm = np.sqrt(energy)

    if energy.size == 0:
        max_index = 0
        max_energy = 0.0
        time_max = 0.0
    else:
        max_index = int(np.argmax(energy))
        max_energy = float(energy[max_index])
        time_max = float(times_arr[max_index])

    return ScatteredFieldSummary(
        name=str(name),
        times=times_arr,
        values=values,
        energy=energy,
        norm=norm,
        max_energy=max_energy,
        max_norm=float(np.sqrt(max_energy)),
        time_of_max_energy=time_max,
        index_of_max_energy=max_index,
    )


def save_scattered_summary_npz(
    path: str,
    *,
    reflection: ScatteredFieldSummary | None = None,
    transmission: ScatteredFieldSummary | None = None,
) -> str:
    payload: dict[str, Any] = {"kind": np.asarray("scattered_field_summary")}

    if reflection is not None:
        payload.update(
            {
                "reflection_times": reflection.times,
                "reflection_values": reflection.values,
                "reflection_energy": reflection.energy,
                "reflection_norm": reflection.norm,
            }
        )

    if transmission is not None:
        payload.update(
            {
                "transmission_times": transmission.times,
                "transmission_values": transmission.values,
                "transmission_energy": transmission.energy,
                "transmission_norm": transmission.norm,
            }
        )

    np.savez(path, **payload)
    return str(path)
