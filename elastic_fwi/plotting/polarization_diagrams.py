from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _prepare_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def plot_hodogram(
    hodogram: Mapping[str, Any],
    output_path: str | Path,
    *,
    title: str | None = None,
) -> str:
    """Plot ``ux(t)`` against ``uy(t)`` from ``compute_hodogram_data`` output."""
    path = _ensure_parent(output_path)
    ux = _as_float_array(hodogram["ux"], name="ux").reshape(-1)
    uy = _as_float_array(hodogram["uy"], name="uy").reshape(-1)
    if ux.shape != uy.shape:
        raise ValueError("ux and uy must have the same shape.")

    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot(ux, uy)
    ax.scatter([ux[0]], [uy[0]], marker="o", label="start")
    ax.scatter([ux[-1]], [uy[-1]], marker="x", label="end")
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlabel("ux")
    ax.set_ylabel("uy")
    ax.set_title(title or f"Polarization hodogram — {hodogram.get('aperture', 'signal')}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_orientation_histogram(
    histogram: Mapping[str, Any],
    output_path: str | Path,
    *,
    title: str | None = None,
) -> str:
    """Plot an orientation histogram produced by ``compute_orientation_histogram``."""
    path = _ensure_parent(output_path)
    centers = _as_float_array(histogram["bin_centers"], name="bin_centers").reshape(-1)
    counts = _as_float_array(histogram["counts"], name="counts").reshape(-1)
    if centers.shape != counts.shape:
        raise ValueError("bin_centers and counts must have the same shape.")
    width = float(np.mean(np.diff(centers))) if centers.size > 1 else 0.1

    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(centers, counts, width=0.9 * width)
    ax.set_xlabel("orientation angle atan2(uy, ux) [rad]")
    ax.set_ylabel("count")
    ax.set_title(title or "Polarization orientation histogram")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_windowed_polarization_bars(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    metric: str = "polarization_coherence",
    title: str = "Windowed polarization summary",
) -> str:
    """Plot one scalar polarization metric over time windows and apertures."""
    path = _ensure_parent(output_path)
    labels = [f"{row.get('aperture', row.get('name', 'signal'))}:{row.get('window', '')}" for row in rows]
    values = [float(row.get(metric, 0.0)) for row in rows]
    if not all(np.isfinite(values)):
        raise ValueError(f"{metric} contains non-finite values.")

    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(max(6.0, 0.45 * len(labels)), 4.2))
    ax.bar(labels, values)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_component_energy_ratio(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    title: str = "Component energy ratio",
) -> str:
    """Plot ``x_over_y_energy_ratio`` for aperture-level polarization rows."""
    path = _ensure_parent(output_path)
    labels = [str(row.get("aperture", row.get("name", f"row_{i}"))) for i, row in enumerate(rows)]
    values = [float(row.get("x_over_y_energy_ratio", 0.0)) for row in rows]
    if not all(np.isfinite(values)):
        raise ValueError("x_over_y_energy_ratio contains non-finite values.")

    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.bar(labels, values)
    ax.set_ylabel("ux / uy energy ratio")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)
