from __future__ import annotations

import math

import numpy as np
import pytest

from elastic_fwi.metrics.image_quality import compute_image_quality_metrics
from elastic_fwi.metrics.artifact_metrics import write_metrics_csv


def test_image_quality_metrics_detects_defect_peak_and_artifact_ratio(tmp_path):
    centers = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [2.0, 2.0],
        ],
        dtype=float,
    )
    values = np.asarray([0.1, 0.2, 0.15, 0.4, 1.0, 0.5], dtype=float)

    metrics = compute_image_quality_metrics(
        cell_centers=centers,
        image_values=values,
        defect_center_x=0.5,
        defect_center_y=0.5,
        defect_radius=0.25,
        method="transmission_only",
        field_name="gradient_rho",
        threshold_false_max=0.45,
        half_max_fraction=0.5,
    )

    assert metrics.status == "ok"
    assert metrics.max_x == pytest.approx(0.5)
    assert metrics.max_y == pytest.approx(0.5)
    assert metrics.localization_error == pytest.approx(0.0)
    assert metrics.max_value_in_defect == pytest.approx(1.0)
    assert metrics.max_value_outside_defect == pytest.approx(0.5)
    assert metrics.peak_to_artifact_ratio == pytest.approx(2.0)
    assert metrics.energy_in_defect == pytest.approx(1.0)
    assert metrics.energy_outside_defect == pytest.approx(
        0.1**2 + 0.2**2 + 0.15**2 + 0.4**2 + 0.5**2
    )
    assert metrics.false_maxima_count == 1
    assert metrics.half_max_area == pytest.approx(2.0)

    path = write_metrics_csv(tmp_path / "metrics.csv", [metrics.as_dict()])
    assert path.exists()
    assert "peak_to_artifact_ratio" in path.read_text(encoding="utf-8")


def test_image_quality_metrics_handles_zero_image():
    metrics = compute_image_quality_metrics(
        cell_centers=np.asarray([[0.5, 0.5], [1.0, 1.0]], dtype=float),
        image_values=np.zeros(2),
        defect_center_x=0.5,
        defect_center_y=0.5,
        defect_radius=0.25,
        method="reflection_only",
        field_name="gradient_C11",
    )

    assert metrics.status == "zero_image"
    assert metrics.peak_to_artifact_ratio == 0.0
    assert metrics.false_maxima_count == 0


def test_image_quality_metrics_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="size mismatch"):
        compute_image_quality_metrics(
            cell_centers=np.asarray([[0.0, 0.0]], dtype=float),
            image_values=np.asarray([1.0, 2.0], dtype=float),
            defect_center_x=0.0,
            defect_center_y=0.0,
            defect_radius=1.0,
            method="m",
            field_name="f",
        )

    with pytest.raises(ValueError, match="defect_radius"):
        compute_image_quality_metrics(
            cell_centers=np.asarray([[0.0, 0.0]], dtype=float),
            image_values=np.asarray([1.0], dtype=float),
            defect_center_x=0.0,
            defect_center_y=0.0,
            defect_radius=0.0,
            method="m",
            field_name="f",
        )


def test_image_quality_metrics_empty_input():
    metrics = compute_image_quality_metrics(
        cell_centers=np.empty((0, 2), dtype=float),
        image_values=np.empty((0,), dtype=float),
        defect_center_x=0.0,
        defect_center_y=0.0,
        defect_radius=1.0,
        method="m",
        field_name="f",
    )

    assert metrics.status == "empty"
    assert math.isnan(metrics.max_x)
    assert metrics.num_samples == 0
