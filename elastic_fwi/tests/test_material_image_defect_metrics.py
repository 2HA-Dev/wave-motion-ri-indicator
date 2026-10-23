from __future__ import annotations

import math

import numpy as np
import pytest

from elastic_fwi.inversion.material_image_defect_metrics import (
    compute_cell_defect_localization_metrics,
)


def test_compute_cell_defect_localization_metrics_detects_inside_maximum():
    centers = np.asarray(
        [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
            [0.50, 0.50],
        ],
        dtype=float,
    )
    values = np.asarray([0.1, 0.2, 0.3, 0.4, 1.0], dtype=float)

    metrics = compute_cell_defect_localization_metrics(
        cell_centers=centers,
        cell_values=values,
        defect_center_x=0.5,
        defect_center_y=0.5,
        defect_radius=0.2,
    )

    assert metrics.status == "ok"
    assert metrics.num_cells == 5
    assert metrics.num_inside_cells == 1
    assert metrics.num_outside_cells == 4
    assert metrics.max_value == pytest.approx(1.0)
    assert metrics.max_center == pytest.approx((0.5, 0.5))
    assert metrics.max_distance_to_defect_center == pytest.approx(0.0)
    assert metrics.max_inside_defect is True
    assert metrics.total_indicator_mass == pytest.approx(2.0)
    assert metrics.inside_indicator_mass == pytest.approx(1.0)
    assert metrics.outside_indicator_mass == pytest.approx(1.0)
    assert metrics.inside_mass_fraction == pytest.approx(0.5)
    assert metrics.inside_mean == pytest.approx(1.0)
    assert metrics.outside_mean == pytest.approx(0.25)
    assert metrics.inside_outside_mean_ratio == pytest.approx(4.0)


def test_compute_cell_defect_localization_metrics_empty_input():
    metrics = compute_cell_defect_localization_metrics(
        cell_centers=np.empty((0, 2), dtype=float),
        cell_values=np.empty((0,), dtype=float),
        defect_center_x=0.5,
        defect_center_y=0.5,
        defect_radius=0.2,
    )

    assert metrics.status == "empty"
    assert metrics.num_cells == 0
    assert metrics.num_inside_cells == 0
    assert metrics.num_outside_cells == 0
    assert metrics.max_value == 0.0
    assert math.isnan(metrics.max_center[0])
    assert math.isnan(metrics.max_center[1])
    assert math.isnan(metrics.max_distance_to_defect_center)
    assert metrics.total_indicator_mass == 0.0


def test_compute_cell_defect_localization_metrics_as_dict():
    centers = np.asarray([[0.5, 0.5]], dtype=float)
    values = np.asarray([1.0], dtype=float)

    metrics = compute_cell_defect_localization_metrics(
        cell_centers=centers,
        cell_values=values,
        defect_center_x=0.5,
        defect_center_y=0.5,
        defect_radius=0.25,
    )

    data = metrics.as_dict()

    assert data["kind"] == "cell_defect_localization_metrics"
    assert data["status"] == "ok"
    assert data["num_cells"] == 1
    assert data["max_inside_defect"] is True
    assert data["defect_center"] == [0.5, 0.5]


@pytest.mark.parametrize(
    "centers,values,match",
    [
        (np.asarray([0.0, 1.0]), np.asarray([1.0]), "two-dimensional"),
        (np.asarray([[0.0, 0.0]]), np.asarray([1.0, 2.0]), "size mismatch"),
        (np.asarray([[0.0, np.nan]]), np.asarray([1.0]), "non-finite"),
        (np.asarray([[0.0, 0.0]]), np.asarray([np.nan]), "non-finite"),
    ],
)
def test_compute_cell_defect_localization_metrics_rejects_invalid_inputs(
    centers,
    values,
    match,
):
    with pytest.raises(ValueError, match=match):
        compute_cell_defect_localization_metrics(
            cell_centers=centers,
            cell_values=values,
            defect_center_x=0.5,
            defect_center_y=0.5,
            defect_radius=0.25,
        )


def test_compute_cell_defect_localization_metrics_rejects_invalid_defect_radius():
    with pytest.raises(ValueError, match="defect_radius"):
        compute_cell_defect_localization_metrics(
            cell_centers=np.asarray([[0.5, 0.5]], dtype=float),
            cell_values=np.asarray([1.0], dtype=float),
            defect_center_x=0.5,
            defect_center_y=0.5,
            defect_radius=0.0,
        )
