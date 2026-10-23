from __future__ import annotations

from typing import Any

import numpy as np

from elastic_fwi.wave_physics.modal_proxies import longitudinal_transverse_projection
from elastic_fwi.wave_physics.time_windows import trapezoid_integral

SCIENTIFIC_NOTE = (
    "This is an energy-flux proxy based on vector-component amplitudes and optional "
    "directions. It is not a full mechanical energy flux because stresses and exact "
    "particle velocities are not used."
)


def _as_vector_values(values: Any, *, name: str = "values") -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim < 2 or arr.shape[-1] != 2:
        raise ValueError(f"{name} must have last dimension equal to 2 components.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _as_times(times: Any, *, expected_size: int) -> np.ndarray:
    arr = np.asarray(times, dtype=float).reshape(-1)
    if arr.shape != (int(expected_size),):
        raise ValueError("times length must match values.shape[0].")
    if not np.all(np.isfinite(arr)):
        raise ValueError("times must contain only finite values.")
    if not np.all(np.diff(arr) >= 0.0):
        raise ValueError("times must be non-decreasing.")
    return arr


def _integral(times: np.ndarray | None, values: np.ndarray) -> float:
    flattened = np.asarray(values, dtype=float).reshape(values.shape[0], -1)
    inst = np.sum(flattened, axis=1)
    if times is None:
        return float(np.sum(inst))
    return float(trapezoid_integral(inst, times))


def compute_component_energy_flux_proxy(
    values: Any,
    times: Any | None = None,
) -> dict[str, Any]:
    """Compute component-energy proxy time series and integrated totals."""
    arr = _as_vector_values(values)
    times_arr = None if times is None else _as_times(times, expected_size=arr.shape[0])

    x_energy = arr[..., 0] * arr[..., 0]
    y_energy = arr[..., 1] * arr[..., 1]
    total_energy = x_energy + y_energy

    return {
        "kind": "component_energy_flux_proxy",
        "scientific_note": SCIENTIFIC_NOTE,
        "component_x_energy": x_energy,
        "component_y_energy": y_energy,
        "total_component_energy": total_energy,
        "integrated_component_x_energy": _integral(times_arr, x_energy),
        "integrated_component_y_energy": _integral(times_arr, y_energy),
        "integrated_total_component_energy": _integral(times_arr, total_energy),
    }


def compute_directional_energy_proxy(
    values: Any,
    directions: Any | None = None,
    times: Any | None = None,
) -> dict[str, Any]:
    """Compute longitudinal/transverse directional energy proxies."""
    arr = _as_vector_values(values)
    times_arr = None if times is None else _as_times(times, expected_size=arr.shape[0])
    projection = longitudinal_transverse_projection(arr, directions)
    longitudinal = np.asarray(projection["longitudinal"], dtype=float)
    transverse = np.asarray(projection["transverse"], dtype=float)
    long_energy = longitudinal * longitudinal
    trans_energy = transverse * transverse

    return {
        "kind": "directional_energy_proxy",
        "scientific_note": SCIENTIFIC_NOTE,
        "longitudinal_energy": long_energy,
        "transverse_energy": trans_energy,
        "integrated_longitudinal_energy": _integral(times_arr, long_energy),
        "integrated_transverse_energy": _integral(times_arr, trans_energy),
        "directions": projection["directions"],
    }


def summarize_energy_flux_proxy(
    *,
    values: Any,
    times: Any | None = None,
    directions: Any | None = None,
    name: str = "signal",
    epsilon: float = 1.0e-30,
) -> dict[str, Any]:
    """Return scalar summaries for component and directional flux proxies."""
    eps = float(epsilon)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be strictly positive and finite.")

    component = compute_component_energy_flux_proxy(values, times=times)
    directional = compute_directional_energy_proxy(values, directions=directions, times=times)
    x_e = float(component["integrated_component_x_energy"])
    y_e = float(component["integrated_component_y_energy"])
    long_e = float(directional["integrated_longitudinal_energy"])
    trans_e = float(directional["integrated_transverse_energy"])

    return {
        "kind": "energy_flux_proxy_summary",
        "name": str(name),
        "scientific_note": SCIENTIFIC_NOTE,
        "integrated_component_x_energy": x_e,
        "integrated_component_y_energy": y_e,
        "integrated_total_component_energy": float(component["integrated_total_component_energy"]),
        "component_x_over_y_energy_ratio": float(x_e / (y_e + eps)),
        "integrated_longitudinal_energy": long_e,
        "integrated_transverse_energy": trans_e,
        "longitudinal_over_transverse_energy_ratio": float(long_e / (trans_e + eps)),
    }
