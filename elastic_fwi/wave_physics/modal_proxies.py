from __future__ import annotations

from typing import Any

import numpy as np

SCIENTIFIC_NOTE = (
    "These quantities are used as component and projection-based modal proxies, "
    "not as a full Lamb-mode decomposition."
)


def _as_vector_values(values: Any, *, name: str = "values") -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim < 2 or arr.shape[-1] != 2:
        raise ValueError(f"{name} must have last dimension equal to 2 components.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _safe_epsilon(epsilon: float) -> float:
    eps = float(epsilon)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be strictly positive and finite.")
    return eps


def component_dominance_ratio(values: Any, *, epsilon: float = 1.0e-30) -> dict[str, Any]:
    """Return component-energy dominance proxies for a vector residual field."""
    arr = _as_vector_values(values)
    eps = _safe_epsilon(epsilon)

    ux = arr[..., 0]
    uy = arr[..., 1]
    energy_x = float(np.sum(ux * ux))
    energy_y = float(np.sum(uy * uy))
    total = energy_x + energy_y

    if energy_x > energy_y:
        dominant = "x"
    elif energy_y > energy_x:
        dominant = "y"
    else:
        dominant = "balanced"

    return {
        "kind": "component_dominance_ratio",
        "scientific_note": SCIENTIFIC_NOTE,
        "energy_x": energy_x,
        "energy_y": energy_y,
        "total_energy": float(total),
        "x_over_y_energy_ratio": float(energy_x / (energy_y + eps)),
        "y_over_x_energy_ratio": float(energy_y / (energy_x + eps)),
        "x_energy_fraction": float(energy_x / (total + eps)),
        "y_energy_fraction": float(energy_y / (total + eps)),
        "dominant_component": dominant,
    }


def _broadcast_directions(values: np.ndarray, directions: Any | None) -> np.ndarray:
    if directions is None:
        direction = np.asarray([1.0, 0.0], dtype=float)
    else:
        direction = np.asarray(directions, dtype=float)

    if direction.shape == (2,):
        out = np.broadcast_to(direction, values.shape)
    else:
        try:
            out = np.broadcast_to(direction, values.shape)
        except ValueError as exc:
            raise ValueError("directions must be shape (2,) or broadcastable to values.shape.") from exc

    if not np.all(np.isfinite(out)):
        raise ValueError("directions must contain only finite values.")

    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("directions must have strictly positive norms.")

    return out / norms


def longitudinal_transverse_projection(
    values: Any,
    directions: Any | None = None,
) -> dict[str, Any]:
    """Project a vector field onto direction-based longitudinal/transverse proxies."""
    arr = _as_vector_values(values)
    dirs = _broadcast_directions(arr, directions)

    longitudinal = np.sum(arr * dirs, axis=-1)
    transverse = -arr[..., 0] * dirs[..., 1] + arr[..., 1] * dirs[..., 0]

    return {
        "kind": "longitudinal_transverse_projection",
        "scientific_note": SCIENTIFIC_NOTE,
        "longitudinal": longitudinal,
        "transverse": transverse,
        "directions": dirs,
    }


def mean_projected_longitudinal_energy(
    values: Any,
    directions: Any | None = None,
) -> float:
    projection = longitudinal_transverse_projection(values, directions)
    longitudinal = np.asarray(projection["longitudinal"], dtype=float)
    return float(np.mean(longitudinal * longitudinal))


def mean_projected_transverse_energy(
    values: Any,
    directions: Any | None = None,
) -> float:
    projection = longitudinal_transverse_projection(values, directions)
    transverse = np.asarray(projection["transverse"], dtype=float)
    return float(np.mean(transverse * transverse))


def apparent_mode_proxy(
    values: Any,
    directions: Any | None = None,
    *,
    epsilon: float = 1.0e-30,
    dominance_threshold: float = 1.25,
) -> dict[str, Any]:
    """Summarize apparent component/projection dominance without claiming modal separation."""
    arr = _as_vector_values(values)
    eps = _safe_epsilon(epsilon)
    threshold = float(dominance_threshold)
    if not np.isfinite(threshold) or threshold <= 1.0:
        raise ValueError("dominance_threshold must be finite and greater than one.")

    comp = component_dominance_ratio(arr, epsilon=eps)
    long_e = mean_projected_longitudinal_energy(arr, directions)
    trans_e = mean_projected_transverse_energy(arr, directions)
    ratio = float(long_e / (trans_e + eps))

    if ratio >= threshold:
        apparent = "longitudinal-proxy-dominated"
    elif ratio <= 1.0 / threshold:
        apparent = "transverse-proxy-dominated"
    else:
        apparent = "mixed-proxy"

    return {
        "kind": "apparent_mode_proxy",
        "scientific_note": SCIENTIFIC_NOTE,
        "mean_projected_longitudinal_energy": float(long_e),
        "mean_projected_transverse_energy": float(trans_e),
        "longitudinal_over_transverse_energy_ratio": ratio,
        "apparent_mode_proxy": apparent,
        "component_dominance": comp,
    }
