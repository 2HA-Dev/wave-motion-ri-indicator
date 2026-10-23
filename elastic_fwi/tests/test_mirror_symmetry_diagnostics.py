"""Tests for the mirror-symmetry diagnostics of the C66 kernels."""

from __future__ import annotations

import math

import numpy as np
import pytest

from elastic_fwi.postprocessing.mirror_symmetry_diagnostics import (
    DEFAULT_AXIS_POSITION,
    MirrorDiagnosticsError,
    axis_vanishing_metrics,
    extrapolate_exponent_to_zero_floor,
    mirror_orientation_residual,
    mirror_parity_metrics,
    transverse_decay_metrics,
)


AXIS = DEFAULT_AXIS_POSITION
DEFECT_Y = 36.5


def _symmetric_grid(step: float = 0.01, half_width: float = 0.30, half_height: float = 0.80):
    """A mirror-symmetric cell grid, so that parity can be checked exactly."""

    offsets = np.arange(step / 2.0, half_width, step)
    xs = np.concatenate([AXIS - offsets[::-1], AXIS + offsets])
    ys = np.arange(DEFECT_Y - half_height + step / 2.0, DEFECT_Y + half_height, step)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    centers = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    measures = np.full(centers.shape[0], step * step, dtype=float)
    return centers, measures


def test_mirror_parity_metrics_detects_an_even_field():
    centers, measures = _symmetric_grid()
    signed_distance = centers[:, 0] - AXIS
    values = signed_distance**2 + 1.0

    metrics = mirror_parity_metrics(centers, values, measures)

    assert metrics.odd_l2_fraction == pytest.approx(0.0, abs=1e-12)
    assert metrics.even_l2_fraction == pytest.approx(1.0, rel=1e-12)
    assert metrics.pairing_distance_median == pytest.approx(0.0, abs=1e-12)
    assert metrics.num_paired_cells == metrics.num_cells


def test_mirror_parity_metrics_detects_an_odd_field():
    centers, measures = _symmetric_grid()
    values = centers[:, 0] - AXIS

    metrics = mirror_parity_metrics(centers, values, measures)

    assert metrics.odd_l2_fraction == pytest.approx(1.0, rel=1e-12)
    assert metrics.even_l2_fraction == pytest.approx(0.0, abs=1e-12)


def test_mirror_parity_metrics_splits_a_mixed_field():
    centers, measures = _symmetric_grid()
    signed_distance = centers[:, 0] - AXIS
    # Equal-energy even and odd parts, by construction orthogonal in L2.
    values = 1.0 + signed_distance / np.sqrt(np.mean(signed_distance**2))

    metrics = mirror_parity_metrics(centers, values, measures)

    assert metrics.odd_l2_fraction == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-6)
    assert metrics.even_l2_fraction == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-6)


def test_axis_vanishing_ratio_is_small_for_a_quadratically_vanishing_field():
    centers, measures = _symmetric_grid()
    signed_distance = centers[:, 0] - AXIS
    values = signed_distance**2

    metrics = axis_vanishing_metrics(centers, values, measures)

    # A quadratic zero suppresses the axis band by orders of magnitude.
    assert metrics.axis_suppression_ratio < 1.0e-2
    assert metrics.num_axis_cells > 0
    assert metrics.num_reference_cells > 0


def test_axis_vanishing_ratio_is_order_one_for_a_flat_field():
    centers, measures = _symmetric_grid()
    values = np.full(centers.shape[0], 3.0)

    metrics = axis_vanishing_metrics(centers, values, measures)

    assert metrics.axis_suppression_ratio == pytest.approx(1.0, rel=1e-12)


def test_transverse_decay_recovers_a_quadratic_exponent():
    centers, measures = _symmetric_grid(step=0.005)
    distance = np.abs(centers[:, 0] - AXIS)
    values = distance**2

    metrics = transverse_decay_metrics(centers, values, measures)

    assert metrics.decay_exponent == pytest.approx(2.0, abs=0.05)
    assert metrics.fit_r_squared > 0.999


def test_transverse_decay_recovers_a_linear_exponent():
    centers, measures = _symmetric_grid(step=0.005)
    distance = np.abs(centers[:, 0] - AXIS)
    values = distance

    metrics = transverse_decay_metrics(centers, values, measures)

    assert metrics.decay_exponent == pytest.approx(1.0, abs=0.05)


def test_transverse_decay_recovers_a_flat_exponent():
    centers, measures = _symmetric_grid(step=0.005)
    values = np.full(centers.shape[0], -2.5)

    metrics = transverse_decay_metrics(centers, values, measures)

    assert metrics.decay_exponent == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("floor", [0.0, 2.0e-4, 5.0e-4])
def test_floor_corrected_exponent_survives_a_numerical_floor(floor):
    """The plain slope is biased low by a floor; the corrected one is not.

    This reproduces the situation of the campaign maps, where the mesh is not
    generated symmetrically and leaves a non-vanishing residual on the axis.
    """

    centers, measures = _symmetric_grid(step=0.005)
    distance = np.abs(centers[:, 0] - AXIS)
    values = np.sqrt(distance**4 + floor**2)

    metrics = transverse_decay_metrics(centers, values, measures)

    assert metrics.floor_model_converged
    assert metrics.floor_corrected_exponent == pytest.approx(2.0, abs=0.05)
    if floor > 0.0:
        assert metrics.floor_amplitude == pytest.approx(floor, rel=0.15)


def test_floor_corrected_exponent_matches_plain_slope_without_floor():
    centers, measures = _symmetric_grid(step=0.005)
    distance = np.abs(centers[:, 0] - AXIS)
    values = distance**2

    metrics = transverse_decay_metrics(centers, values, measures)

    assert metrics.floor_corrected_exponent == pytest.approx(
        metrics.decay_exponent, abs=0.05
    )


def test_transverse_decay_rejects_an_empty_fit_window():
    centers, measures = _symmetric_grid(step=0.005)
    values = np.abs(centers[:, 0] - AXIS)

    with pytest.raises(MirrorDiagnosticsError):
        transverse_decay_metrics(
            centers,
            values,
            measures,
            fit_min_distance=0.20,
            fit_max_distance=0.22,
            max_distance=0.24,
            num_bins=4,
        )


def test_floor_extrapolation_recovers_the_unbiased_exponent():
    floors = np.array([0.001, 0.005, 0.010, 0.015, 0.020])
    exponents = 2.0 - 40.0 * floors

    result = extrapolate_exponent_to_zero_floor(floors, exponents)

    assert result.exponent_at_zero_floor == pytest.approx(2.0, abs=1e-9)
    assert result.slope == pytest.approx(-40.0, rel=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-12)
    assert result.num_points == 5


def test_floor_extrapolation_on_the_measured_c66_exponents():
    """The eight campaign C66 maps extrapolate to the predicted exponent of 2.

    Values are the ``floor_to_peak_ratio`` and ``floor_corrected_exponent``
    values measured on the campaign C66 maps.
    """

    floors = np.array(
        [0.00146, 0.00317, 0.01030, 0.01122, 0.01122, 0.01229, 0.01953, 0.01998]
    )
    exponents = np.array(
        [1.936, 1.890, 1.754, 1.503, 1.503, 1.646, 1.174, 1.197]
    )

    result = extrapolate_exponent_to_zero_floor(floors, exponents)

    assert result.exponent_at_zero_floor == pytest.approx(2.0, abs=0.15)
    assert result.slope < 0.0
    assert result.r_squared > 0.85


def test_floor_extrapolation_needs_enough_points():
    with pytest.raises(MirrorDiagnosticsError):
        extrapolate_exponent_to_zero_floor(np.array([0.01, 0.02]), np.array([1.5, 1.8]))


@pytest.mark.parametrize(
    ("mode", "orientation"),
    [
        (2, 179.55719761935808),
        (2, 89.96541274421018),
        (3, 30.288836551011045),
        (4, 44.90064778377242),
    ],
)
def test_frozen_dominant_orientations_are_mirror_compatible(mode, orientation):
    """The four campaign C66 orientations sit on the mirror-admissible set.

    Values are the ``dominant_orientation_degrees`` entries of the campaign
    centre-search classification.
    """

    assert mirror_orientation_residual(mode, orientation) < 1.0


def test_mirror_orientation_residual_flags_an_incompatible_orientation():
    # For n = 2 the admissible orientations are 0 and 90 degrees; 45 is the
    # farthest possible, at 45 degrees from both.
    assert mirror_orientation_residual(2, 45.0) == pytest.approx(45.0)


def test_mirror_orientation_residual_rejects_a_non_positive_mode():
    with pytest.raises(MirrorDiagnosticsError):
        mirror_orientation_residual(0, 90.0)


def test_validation_rejects_mismatched_lengths():
    centers, measures = _symmetric_grid(step=0.02)
    with pytest.raises(MirrorDiagnosticsError):
        mirror_parity_metrics(centers, np.zeros(3), measures)


def test_validation_rejects_non_positive_measures():
    centers, measures = _symmetric_grid(step=0.02)
    measures[0] = 0.0
    with pytest.raises(MirrorDiagnosticsError):
        mirror_parity_metrics(centers, np.ones(centers.shape[0]), measures)
