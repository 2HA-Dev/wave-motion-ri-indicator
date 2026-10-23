"""Tests for the shear-strain factor maps and their Cauchy--Schwarz split."""

from __future__ import annotations

import math

import numpy as np
import pytest

from elastic_fwi.postprocessing.shear_factor_maps import (
    ShearFactorError,
    accumulate_pair,
    cauchy_schwarz_split,
    correlation_against_reference,
    factor_spread,
    marginal_profile_x2,
    triangle_shear_strain,
)


def _unit_square_mesh():
    """Two triangles covering the unit square."""

    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return points, triangles


def _linear_displacement(points, a11, a12, a21, a22):
    """u = A x, so that the strains are constant and known analytically."""

    x = points[:, 0]
    y = points[:, 1]
    return np.column_stack([a11 * x + a12 * y, a21 * x + a22 * y, np.zeros_like(x)])


@pytest.mark.parametrize(
    ("a11", "a12", "a21", "a22", "expected"),
    [
        (0.0, 1.0, 0.0, 0.0, 1.0),      # pure du1/dx2
        (0.0, 0.0, 1.0, 0.0, 1.0),      # pure du2/dx1
        (0.0, 2.0, 3.0, 0.0, 5.0),      # both contributions add
        (0.0, 1.0, -1.0, 0.0, 0.0),     # rigid rotation carries no shear
        (7.0, 0.0, 0.0, -4.0, 0.0),     # normal strains carry no shear
        (1.5, -0.5, 2.5, 3.0, 2.0),     # general case
    ],
)
def test_shear_strain_of_a_linear_field(a11, a12, a21, a22, expected):
    points, triangles = _unit_square_mesh()
    displacement = _linear_displacement(points, a11, a12, a21, a22)

    result = triangle_shear_strain(points, triangles, displacement)

    assert result.shear_strain == pytest.approx(np.full(2, expected))
    assert result.areas == pytest.approx(np.full(2, 0.5))


def test_shear_strain_areas_and_centroids():
    points, triangles = _unit_square_mesh()
    displacement = _linear_displacement(points, 0.0, 1.0, 0.0, 0.0)

    result = triangle_shear_strain(points, triangles, displacement)

    assert float(np.sum(result.areas)) == pytest.approx(1.0)
    assert result.centroids.shape == (2, 2)
    assert result.centroids[0] == pytest.approx([2.0 / 3.0, 1.0 / 3.0])


def test_shear_strain_is_invariant_under_vertex_ordering():
    points, triangles = _unit_square_mesh()
    displacement = _linear_displacement(points, 1.0, 2.0, 3.0, 4.0)

    straight = triangle_shear_strain(points, triangles, displacement)
    flipped = triangle_shear_strain(points, triangles[:, [0, 2, 1]], displacement)

    assert flipped.shear_strain == pytest.approx(straight.shear_strain)
    assert flipped.areas == pytest.approx(straight.areas)


def test_shear_strain_rejects_a_degenerate_triangle():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    displacement = np.zeros((3, 3))

    with pytest.raises(ShearFactorError):
        triangle_shear_strain(points, triangles, displacement)


def test_accumulate_pair_integrates_in_time():
    first = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    second = [np.array([5.0, 6.0]), np.array([7.0, 8.0])]
    dt = 0.5

    result = accumulate_pair(first, second, dt)

    assert result["pair_integral"] == pytest.approx(
        dt * np.array([1 * 5 + 3 * 7, 2 * 6 + 4 * 8])
    )
    assert result["first_energy"] == pytest.approx(dt * np.array([1 + 9, 4 + 16]))
    assert result["second_energy"] == pytest.approx(dt * np.array([25 + 49, 36 + 64]))


def test_accumulate_pair_rejects_bad_inputs():
    with pytest.raises(ShearFactorError):
        accumulate_pair([np.ones(2)], [np.ones(2), np.ones(2)], 0.1)
    with pytest.raises(ShearFactorError):
        accumulate_pair([], [], 0.1)
    with pytest.raises(ShearFactorError):
        accumulate_pair([np.ones(2)], [np.ones(2)], 0.0)


def test_coherence_is_one_for_proportional_factors():
    """Perfectly aligned factors saturate Cauchy--Schwarz."""

    signal = [np.array([1.0, -2.0]), np.array([0.5, 3.0])]
    scaled = [3.0 * value for value in signal]

    result = accumulate_pair(signal, scaled, 0.25)
    coherence = cauchy_schwarz_split(
        result["pair_integral"], result["first_energy"], result["second_energy"]
    )

    assert coherence == pytest.approx(np.ones(2))


def test_coherence_is_zero_for_orthogonal_factors():
    first = [np.array([1.0]), np.array([1.0])]
    second = [np.array([1.0]), np.array([-1.0])]

    result = accumulate_pair(first, second, 1.0)
    coherence = cauchy_schwarz_split(
        result["pair_integral"], result["first_energy"], result["second_energy"]
    )

    assert coherence == pytest.approx(np.zeros(1), abs=1e-15)


def test_coherence_is_undefined_where_a_factor_vanishes():
    pair = np.array([1.0, 0.0])
    first = np.array([1.0, 0.0])
    second = np.array([1.0, 1.0])

    coherence = cauchy_schwarz_split(pair, first, second)

    assert coherence[0] == pytest.approx(1.0)
    assert math.isnan(coherence[1])


def test_coherence_rejects_negative_energy():
    with pytest.raises(ShearFactorError):
        cauchy_schwarz_split(np.zeros(2), np.array([-1.0, 1.0]), np.ones(2))


def test_factor_spread_on_a_known_distribution():
    # Mass concentrated on two symmetric points about x2 = 5, one unit apart.
    centroids = np.array([[0.0, 4.5], [0.0, 5.5]])
    weights = np.array([1.0, 1.0])
    areas = np.array([1.0, 1.0])

    spread = factor_spread(centroids, weights, areas)

    assert spread.centroid_x2 == pytest.approx(5.0)
    assert spread.spread_x2 == pytest.approx(0.5)
    assert spread.spread_x1 == pytest.approx(0.0)
    assert spread.total_mass == pytest.approx(2.0)


def test_factor_spread_distinguishes_focused_from_extended():
    x2 = np.linspace(30.0, 43.0, 400)
    centroids = np.column_stack([np.full_like(x2, 36.5), x2])
    areas = np.full_like(x2, 1.0)

    focused = np.exp(-(((x2 - 36.5) / 0.4) ** 2))
    extended = np.ones_like(x2)

    assert (
        factor_spread(centroids, focused, areas).spread_x2
        < factor_spread(centroids, extended, areas).spread_x2
    )


def test_factor_spread_rejects_an_empty_map():
    centroids = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ShearFactorError):
        factor_spread(centroids, np.zeros(2), np.ones(2))


def test_marginal_profile_conserves_mass():
    x2 = np.linspace(30.0, 43.0, 500)
    centroids = np.column_stack([np.zeros_like(x2), x2])
    weights = np.abs(np.sin(x2))
    areas = np.full_like(x2, 0.5)

    _, totals = marginal_profile_x2(centroids, weights, areas, num_bins=40)

    assert float(np.sum(totals)) == pytest.approx(float(np.sum(weights * areas)))


def test_correlation_detects_an_exact_match_up_to_sign():
    rng = np.random.default_rng(0)
    reference = rng.standard_normal(200)
    areas = np.full(200, 0.3)

    same = correlation_against_reference(reference, reference, areas)
    opposite = correlation_against_reference(-reference, reference, areas)

    assert same["correlation"] == pytest.approx(1.0)
    assert same["best_fit_scale"] == pytest.approx(1.0)
    assert opposite["correlation"] == pytest.approx(-1.0)
    assert opposite["best_fit_scale"] == pytest.approx(-1.0)


def test_correlation_recovers_a_scale_factor():
    rng = np.random.default_rng(1)
    computed = rng.standard_normal(300)
    areas = np.full(300, 1.0)

    result = correlation_against_reference(computed, 7.5 * computed, areas)

    assert result["abs_correlation"] == pytest.approx(1.0)
    assert result["best_fit_scale"] == pytest.approx(7.5)
