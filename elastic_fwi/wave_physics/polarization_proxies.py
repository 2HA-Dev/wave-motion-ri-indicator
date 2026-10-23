from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class PolarizationProxySummary:
    name: str
    energy_x: float
    energy_y: float
    x_over_y_energy_ratio: float
    y_over_x_energy_ratio: float
    mean_orientation_rad: float
    mean_orientation_deg: float
    polarization_coherence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "polarization_proxy_summary",
            "name": self.name,
            "energy_x": float(self.energy_x),
            "energy_y": float(self.energy_y),
            "x_over_y_energy_ratio": float(self.x_over_y_energy_ratio),
            "y_over_x_energy_ratio": float(self.y_over_x_energy_ratio),
            "mean_orientation_rad": float(self.mean_orientation_rad),
            "mean_orientation_deg": float(self.mean_orientation_deg),
            "polarization_coherence": float(self.polarization_coherence),
        }


def _as_vector_values(values: Any, *, name: str = "values") -> np.ndarray:
    arr = np.asarray(values, dtype=float)

    if arr.ndim < 2 or arr.shape[-1] != 2:
        raise ValueError(f"{name} must have last dimension equal to 2 components.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")

    return arr


def _as_times(times: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(times, dtype=float).reshape(-1)

    if arr.size == 0:
        raise ValueError("times must not be empty.")

    if not np.all(np.isfinite(arr)):
        raise ValueError("times must contain only finite values.")

    if not np.all(np.diff(arr) >= 0.0):
        raise ValueError("times must be non-decreasing.")

    if expected_size is not None and arr.shape != (int(expected_size),):
        raise ValueError("times length must match values.shape[0].")

    return arr


def _safe_epsilon(epsilon: float) -> float:
    eps = float(epsilon)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be strictly positive and finite.")
    return eps


def compute_polarization_proxies(
    *,
    values: Any,
    name: str,
    epsilon: float = 1.0e-30,
) -> PolarizationProxySummary:
    arr = _as_vector_values(values)
    epsilon = _safe_epsilon(epsilon)

    ux = arr[..., 0].reshape(-1)
    uy = arr[..., 1].reshape(-1)

    energy_x = float(np.sum(ux * ux))
    energy_y = float(np.sum(uy * uy))

    vectors = np.column_stack([ux, uy])
    norms = np.linalg.norm(vectors, axis=1)
    total_norm = float(np.sum(norms))

    if total_norm > 0.0:
        mean_vec = np.sum(vectors, axis=0) / total_norm
        coherence = float(np.linalg.norm(mean_vec))
        orientation = float(np.arctan2(mean_vec[1], mean_vec[0]))
    else:
        coherence = 0.0
        orientation = 0.0

    return PolarizationProxySummary(
        name=str(name),
        energy_x=energy_x,
        energy_y=energy_y,
        x_over_y_energy_ratio=float(energy_x / (energy_y + epsilon)),
        y_over_x_energy_ratio=float(energy_y / (energy_x + epsilon)),
        mean_orientation_rad=orientation,
        mean_orientation_deg=float(np.degrees(orientation)),
        polarization_coherence=coherence,
    )


def compute_orientation_angles(
    values: Any,
    *,
    min_norm: float = 0.0,
) -> np.ndarray:
    """Return vector-field orientation angles ``atan2(uy, ux)`` in radians.

    The function is NumPy-only and accepts any array whose last dimension is
    ``(ux, uy)``. Zero-amplitude vectors are kept by default and therefore have
    angle ``0``. Use ``min_norm`` to exclude near-zero vectors from histograms.
    """
    arr = _as_vector_values(values)
    min_norm = float(min_norm)
    if not np.isfinite(min_norm) or min_norm < 0.0:
        raise ValueError("min_norm must be finite and non-negative.")

    flat = arr.reshape(-1, 2)
    norms = np.linalg.norm(flat, axis=1)
    if min_norm > 0.0:
        flat = flat[norms > min_norm]

    if flat.size == 0:
        return np.asarray([], dtype=float)

    return np.arctan2(flat[:, 1], flat[:, 0])


def compute_orientation_histogram(
    values: Any,
    bins: int | Iterable[float] = 36,
    *,
    min_norm: float = 0.0,
) -> dict[str, Any]:
    """Compute an orientation histogram over ``[-pi, pi]``.

    Counts are raw sample counts. ``density`` is normalized so that it sums to
    one when at least one vector is retained.
    """
    angles = compute_orientation_angles(values, min_norm=min_norm)

    if isinstance(bins, int):
        if int(bins) <= 0:
            raise ValueError("bins must be a strictly positive integer.")
        bin_edges = np.linspace(-np.pi, np.pi, int(bins) + 1)
    else:
        bin_edges = np.asarray(list(bins), dtype=float)
        if bin_edges.ndim != 1 or bin_edges.size < 2:
            raise ValueError("bins must contain at least two edges.")
        if not np.all(np.diff(bin_edges) > 0.0):
            raise ValueError("bin edges must be strictly increasing.")

    counts, edges = np.histogram(angles, bins=bin_edges)
    total = int(np.sum(counts))
    density = counts.astype(float) / float(total) if total else counts.astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])

    return {
        "kind": "orientation_histogram",
        "num_samples": total,
        "bin_edges": edges,
        "bin_centers": centers,
        "counts": counts.astype(int),
        "density": density,
    }


def compute_hodogram_data(
    times: Any,
    values: Any,
    receiver_index: int | None = None,
    aperture: str = "reflection",
) -> dict[str, Any]:
    """Return time-parametrized ``ux(t), uy(t)`` data for a hodogram.

    If ``values`` has shape ``(T, R, 2)`` and ``receiver_index`` is ``None``,
    the receiver average is used. If a receiver index is provided, the selected
    receiver trace is used. Arrays of shape ``(T, 2)`` are accepted directly.
    """
    arr = _as_vector_values(values)
    times_arr = _as_times(times, expected_size=arr.shape[0])

    if arr.ndim == 2:
        selected = arr
        selected_receiver: int | None = None
        reduction = "single_trace"
    elif arr.ndim == 3:
        if receiver_index is None:
            selected = np.mean(arr, axis=1)
            selected_receiver = None
            reduction = "receiver_average"
        else:
            idx = int(receiver_index)
            if idx < 0 or idx >= arr.shape[1]:
                raise IndexError("receiver_index is outside the receiver axis.")
            selected = arr[:, idx, :]
            selected_receiver = idx
            reduction = "selected_receiver"
    else:
        selected = np.reshape(arr, (arr.shape[0], -1, 2)).mean(axis=1)
        selected_receiver = None
        reduction = "flattened_receiver_average"

    ux = np.asarray(selected[:, 0], dtype=float)
    uy = np.asarray(selected[:, 1], dtype=float)
    amplitude = np.sqrt(ux * ux + uy * uy)
    orientation_rad = np.arctan2(uy, ux)

    return {
        "kind": "hodogram_data",
        "aperture": str(aperture),
        "receiver_index": selected_receiver,
        "receiver_reduction": reduction,
        "num_steps": int(times_arr.size),
        "times": times_arr,
        "ux": ux,
        "uy": uy,
        "amplitude": amplitude,
        "orientation_rad": orientation_rad,
        "orientation_deg": np.degrees(orientation_rad),
    }


def _window_bounds(window: Any) -> tuple[str, float, float]:
    if hasattr(window, "name") and hasattr(window, "start_time") and hasattr(window, "end_time"):
        return str(window.name), float(window.start_time), float(window.end_time)

    if isinstance(window, dict):
        name = str(window.get("name", window.get("window", "window")))
        start = float(window.get("start_time", window.get("start", 0.0)))
        end = float(window.get("end_time", window.get("end", start)))
        return name, start, end

    if isinstance(window, (list, tuple)) and len(window) == 3:
        return str(window[0]), float(window[1]), float(window[2])

    raise TypeError("windows must contain TimeWindow-like objects, dicts or (name, start, end) tuples.")


def compute_windowed_polarization_proxies(
    times: Any,
    values: Any,
    windows: Iterable[Any],
    *,
    name: str = "signal",
    epsilon: float = 1.0e-30,
) -> list[dict[str, Any]]:
    """Compute polarization proxies on a list of time windows."""
    arr = _as_vector_values(values)
    times_arr = _as_times(times, expected_size=arr.shape[0])

    rows: list[dict[str, Any]] = []
    for window in windows:
        window_name, start, end = _window_bounds(window)
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError("window bounds must be finite.")
        if end < start:
            raise ValueError("window end_time must be greater than or equal to start_time.")

        mask = (times_arr >= start) & (times_arr <= end)
        selected = arr[mask]

        if selected.shape[0] == 0:
            proxy = PolarizationProxySummary(
                name=f"{name}:{window_name}",
                energy_x=0.0,
                energy_y=0.0,
                x_over_y_energy_ratio=0.0,
                y_over_x_energy_ratio=0.0,
                mean_orientation_rad=0.0,
                mean_orientation_deg=0.0,
                polarization_coherence=0.0,
            )
        else:
            proxy = compute_polarization_proxies(
                values=selected,
                name=f"{name}:{window_name}",
                epsilon=epsilon,
            )

        row = proxy.as_dict()
        row.update(
            {
                "window": window_name,
                "start_time": float(start),
                "end_time": float(end),
                "num_samples": int(np.count_nonzero(mask)),
            }
        )
        rows.append(row)

    return rows
