import numpy as np
import pytest

from elastic_fwi.materials.legacy_data import legacy_defect_scenario
from elastic_fwi.materials.orthotropic import LEGACY_HEALTHY_MATERIAL
from elastic_fwi.materials.point_fields import (
    OrthotropicPointFields,
    defective_material_fields_at_points,
    homogeneous_material_fields_at_points,
    material_fields_from_factors,
    material_indices_and_fields_at_points,
)


def test_homogeneous_material_fields_at_points_use_healthy_values():
    points = np.array(
        [
            [75.0, 75.0],
            [82.0, 82.0],
            [90.0, 90.0],
        ],
        dtype=float,
    )

    fields = homogeneous_material_fields_at_points(points, LEGACY_HEALTHY_MATERIAL)

    assert fields.size == 3
    np.testing.assert_allclose(fields.rho, np.full(3, 4.5))
    np.testing.assert_allclose(fields.C11, np.full(3, 16.24))
    np.testing.assert_allclose(fields.C22, np.full(3, 18.07))
    np.testing.assert_allclose(fields.C12, np.full(3, 6.9))
    np.testing.assert_allclose(fields.C66, np.full(3, 4.67))


def test_defective_material_fields_at_legacy_centers_have_legacy_factor():
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

    fields = defective_material_fields_at_points(points, scenario)

    expected_factors = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 1.0])

    np.testing.assert_allclose(fields.rho, LEGACY_HEALTHY_MATERIAL.rho * expected_factors)
    np.testing.assert_allclose(fields.C11, LEGACY_HEALTHY_MATERIAL.C11 * expected_factors)
    np.testing.assert_allclose(fields.C22, LEGACY_HEALTHY_MATERIAL.C22 * expected_factors)
    np.testing.assert_allclose(fields.C12, LEGACY_HEALTHY_MATERIAL.C12 * expected_factors)
    np.testing.assert_allclose(fields.C66, LEGACY_HEALTHY_MATERIAL.C66 * expected_factors)


def test_material_fields_from_factors():
    factors = np.array([1.0, 0.8, 0.5], dtype=float)

    fields = material_fields_from_factors(factors, LEGACY_HEALTHY_MATERIAL)

    np.testing.assert_allclose(fields.rho, 4.5 * factors)
    np.testing.assert_allclose(fields.C11, 16.24 * factors)
    np.testing.assert_allclose(fields.C22, 18.07 * factors)
    np.testing.assert_allclose(fields.C12, 6.9 * factors)
    np.testing.assert_allclose(fields.C66, 4.67 * factors)


def test_material_indices_and_fields_are_returned_together():
    scenario = legacy_defect_scenario()

    points = np.array(
        [
            [82.0, 82.0],
            [90.0, 90.0],
            [75.0, 75.0],
        ],
        dtype=float,
    )

    indices, fields = material_indices_and_fields_at_points(points, scenario)

    np.testing.assert_array_equal(indices, np.array([0, 2, -1]))
    np.testing.assert_allclose(fields.rho, np.array([0.8 * 4.5, 0.8 * 4.5, 4.5]))


def test_point_fields_as_dict_and_stacked():
    fields = OrthotropicPointFields(
        rho=np.array([1.0, 2.0]),
        C11=np.array([3.0, 4.0]),
        C22=np.array([5.0, 6.0]),
        C12=np.array([7.0, 8.0]),
        C66=np.array([9.0, 10.0]),
    )

    as_dict = fields.as_dict()
    assert set(as_dict) == {"rho", "C11", "C22", "C12", "C66"}

    expected = np.array(
        [
            [1.0, 3.0, 5.0, 7.0, 9.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
        ]
    )

    np.testing.assert_allclose(fields.stacked(), expected)


def test_point_fields_reject_inconsistent_shapes():
    with pytest.raises(ValueError):
        OrthotropicPointFields(
            rho=np.array([1.0, 2.0]),
            C11=np.array([1.0]),
            C22=np.array([1.0, 2.0]),
            C12=np.array([1.0, 2.0]),
            C66=np.array([1.0, 2.0]),
        )


def test_point_fields_reject_non_finite_values():
    with pytest.raises(ValueError):
        OrthotropicPointFields(
            rho=np.array([1.0, np.nan]),
            C11=np.array([1.0, 2.0]),
            C22=np.array([1.0, 2.0]),
            C12=np.array([1.0, 2.0]),
            C66=np.array([1.0, 2.0]),
        )


def test_point_fields_reject_non_positive_density():
    with pytest.raises(ValueError):
        OrthotropicPointFields(
            rho=np.array([1.0, 0.0]),
            C11=np.array([1.0, 2.0]),
            C22=np.array([1.0, 2.0]),
            C12=np.array([1.0, 2.0]),
            C66=np.array([1.0, 2.0]),
        )


def test_material_fields_from_factors_reject_invalid_factors():
    with pytest.raises(ValueError):
        material_fields_from_factors(np.array([[1.0, 0.8]]), LEGACY_HEALTHY_MATERIAL)

    with pytest.raises(ValueError):
        material_fields_from_factors(np.array([1.0, np.nan]), LEGACY_HEALTHY_MATERIAL)

    with pytest.raises(ValueError):
        material_fields_from_factors(np.array([1.0, 0.0]), LEGACY_HEALTHY_MATERIAL)


def test_homogeneous_material_fields_reject_invalid_points():
    with pytest.raises(ValueError):
        homogeneous_material_fields_at_points(np.array([0.0, 1.0]), LEGACY_HEALTHY_MATERIAL)

    with pytest.raises(ValueError):
        homogeneous_material_fields_at_points(np.array([[np.nan, 0.0]]), LEGACY_HEALTHY_MATERIAL)
