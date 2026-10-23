from __future__ import annotations

from pathlib import Path

import numpy as np

from elastic_fwi.diagnostics.plot_vector_array_residuals import (
    plot_vector_array_residuals,
)


def _write_residual_npz(path: Path, scale: float) -> None:
    times = np.array([0.0, 0.1, 0.2], dtype=float)
    steps = np.array([0, 1, 2], dtype=np.int64)
    values = np.zeros((3, 2, 2), dtype=float)
    values[:, :, 0] = scale * np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    values[:, :, 1] = scale * np.array([[0.5, 1.0], [1.0, 1.5], [1.5, 2.0]])

    np.savez(
        path,
        kind=np.array("vector_array_residual_series"),
        times=times,
        steps=steps,
        values=values,
    )


def test_plot_vector_array_residuals_creates_png_pdf_and_summary(tmp_path: Path):
    reflection = tmp_path / "reflection.npz"
    transmission = tmp_path / "transmission.npz"
    output_dir = tmp_path / "figures"

    _write_residual_npz(reflection, scale=1.0)
    _write_residual_npz(transmission, scale=0.1)

    summary = plot_vector_array_residuals(
        reflection_npz=reflection,
        transmission_npz=transmission,
        output_dir=output_dir,
    )

    assert summary["status"] == "ok"
    assert summary["reflection"]["num_steps"] == 3
    assert summary["reflection"]["num_receivers"] == 2
    assert summary["transmission"]["num_steps"] == 3
    assert summary["transmission"]["num_receivers"] == 2

    assert Path(summary["outputs"]["png"]).exists()
    assert Path(summary["outputs"]["pdf"]).exists()
    assert Path(summary["outputs"]["summary"]).exists()
