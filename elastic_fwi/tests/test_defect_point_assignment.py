import numpy as np
import pytest

from elastic_fwi.materials.defects import (
    CircularDefectSpec,
    DefectScenario,
    assign_defect_indices_to_points,
    assign_material_factors_to_points,
    defect_membership_mask,
)
from elastic_fwi.materials.legacy_data import legacy_defect_scenario


def test_defect_membership_mask_is_vectorized_and_euclidean():
    defect = CircularDefectSpec(
        name="unit_disk",
        center=(0.0, 0.0),
        radius=1.0,
        norm_order=2,
    )

    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.6, 0.8],
            [0.8, 0.8],
        ],
        dtype=float,
    )

    mask = defect_membership_mask(points, defect)

    np.testing.assert_array_equal(mask, np.array([True, True, True, False]))


def test_legacy_defect_centers_are_assigned_to_expected_indices():
    scenario = legacy_defect_scenario()

    points = np.array(
        [
            [82.0, 82.0],
            [82.0, 98.0],
            [90.0, 90.0],
            [98.0, 82.0],
            [98.0, 98.0],
            [75.0, 75.0],
        ],
        dtype=float,
    )

    indices = assign_defect_indices_to_points(points, scenario)

    np.testing.assert_array_equal(indices, np.array([0, 1, 2, 3, 4, -1]))


def test_legacy_material_factors_are_assigned_to_defect_centers():
    scenario = legacy_defect_scenario()

    points = np.array(
        [
            [82.0, 82.0],
            [82.0, 98.0],
            [90.0, 90.0],
            [98.0, 82.0],
            [98.0, 98.0],
            [75.0, 75.0],
        ],
        dtype=float,
    )

    factors = assign_material_factors_to_points(points, scenario)

    np.testing.assert_allclose(factors, np.array([0.8, 0.8, 0.8, 0.8, 0.8, 1.0]))


def test_assignment_conflict_policy_first_last_and_error():
    scenario = DefectScenario(
        name="overlap",
        defects=(
            CircularDefectSpec(
                name="first",
                center=(0.0, 0.0),
                radius=1.0,
                material_factor=0.8,
            ),
            CircularDefectSpec(
                name="second",
                center=(0.0, 0.0),
                radius=1.0,
                material_factor=0.6,
            ),
        ),
    )

    points = np.array([[0.0, 0.0]], dtype=float)

    first = assign_defect_indices_to_points(points, scenario, conflict_policy="first")
    last = assign_defect_indices_to_points(points, scenario, conflict_policy="last")

    np.testing.assert_array_equal(first, np.array([0]))
    np.testing.assert_array_equal(last, np.array([1]))

    first_factor = assign_material_factors_to_points(points, scenario, conflict_policy="first")
    last_factor = assign_material_factors_to_points(points, scenario, conflict_policy="last")

    np.testing.assert_allclose(first_factor, np.array([0.8]))
    np.testing.assert_allclose(last_factor, np.array([0.6]))

    with pytest.raises(ValueError):
        assign_defect_indices_to_points(points, scenario, conflict_policy="error")


def test_assignment_rejects_invalid_points_shape():
    scenario = legacy_defect_scenario()

    with pytest.raises(ValueError):
        assign_defect_indices_to_points(np.array([0.0, 1.0]), scenario)

    with pytest.raises(ValueError):
        assign_defect_indices_to_points(np.array([[0.0, 1.0, 2.0]]), scenario)


def test_assignment_rejects_non_finite_points():
    scenario = legacy_defect_scenario()

    with pytest.raises(ValueError):
        assign_defect_indices_to_points(np.array([[np.nan, 0.0]]), scenario)

    with pytest.raises(ValueError):
        assign_defect_indices_to_points(np.array([[np.inf, 0.0]]), scenario)


def test_assignment_rejects_invalid_options():
    scenario = legacy_defect_scenario()
    points = np.array([[75.0, 75.0]], dtype=float)

    with pytest.raises(ValueError):
        assign_defect_indices_to_points(points, scenario, conflict_policy="unknown")

    with pytest.raises(ValueError):
        assign_material_factors_to_points(points, scenario, background_factor=0.0)

    with pytest.raises(ValueError):
        assign_material_factors_to_points(points, scenario, tol=-1.0)
