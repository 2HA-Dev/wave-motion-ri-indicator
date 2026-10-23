from __future__ import annotations

from pathlib import Path

import numpy as np

from elastic_fwi.plotting.polarization_diagrams import (
    plot_component_energy_ratio,
    plot_hodogram,
    plot_orientation_histogram,
    plot_windowed_polarization_bars,
)
from elastic_fwi.wave_physics.polarization_proxies import (
    compute_hodogram_data,
    compute_orientation_angles,
    compute_orientation_histogram,
    compute_windowed_polarization_proxies,
)
from elastic_fwi.wave_physics.time_windows import TimeWindow


def test_orientation_histogram_and_hodogram_are_finite(tmp_path: Path):
    times = np.linspace(0.0, 1.0, 21)
    values = np.zeros((times.size, 2, 2), dtype=float)
    values[:, :, 0] = np.sin(2.0 * np.pi * times)[:, None]
    values[:, :, 1] = np.cos(2.0 * np.pi * times)[:, None]

    angles = compute_orientation_angles(values)
    histogram = compute_orientation_histogram(values, bins=12)
    hodogram = compute_hodogram_data(times, values, aperture="reflection")

    assert angles.size == values.shape[0] * values.shape[1]
    assert int(np.sum(histogram["counts"])) == angles.size
    assert hodogram["ux"].shape == times.shape
    assert hodogram["uy"].shape == times.shape

    hodo_png = plot_hodogram(hodogram, tmp_path / "hodogram.png")
    hist_png = plot_orientation_histogram(histogram, tmp_path / "hist.png")

    assert Path(hodo_png).exists()
    assert Path(hist_png).exists()


def test_windowed_polarization_plots_are_created(tmp_path: Path):
    times = np.linspace(0.0, 1.0, 11)
    values = np.ones((times.size, 3, 2), dtype=float)
    windows = [TimeWindow("first", 0.0, 0.5), TimeWindow("second", 0.5, 1.0)]

    rows = compute_windowed_polarization_proxies(times, values, windows, name="reflection")
    for row in rows:
        row["aperture"] = "reflection"

    window_png = plot_windowed_polarization_bars(rows, tmp_path / "windows.png")
    ratio_png = plot_component_energy_ratio(
        [{"aperture": "reflection", "x_over_y_energy_ratio": 1.0}],
        tmp_path / "ratio.png",
    )

    assert Path(window_png).exists()
    assert Path(ratio_png).exists()
    assert all(np.isfinite(float(row["polarization_coherence"])) for row in rows)
