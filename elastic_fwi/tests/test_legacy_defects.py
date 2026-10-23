import pytest

from elastic_fwi.materials.defects import CircularDefectSpec, DefectScenario
from elastic_fwi.materials.legacy_data import (
    LEGACY_DEFECT_MATERIAL_FACTOR,
    LEGACY_DEFECT_NORM_ORDER,
    legacy_defect_scenario,
    legacy_defect_specs,
    legacy_defective_material,
    legacy_healthy_material,
)


def test_legacy_defect_geometry_matches_cpp_generator():
    expected = [
        ("legacy_defect_1", (82.0, 82.0), 0.2),
        ("legacy_defect_2", (82.0, 98.0), 0.4),
        ("legacy_defect_3", (90.0, 90.0), 0.28),
        ("legacy_defect_4", (98.0, 82.0), 0.24),
        ("legacy_defect_5", (98.0, 98.0), 0.36),
    ]

    defects = legacy_defect_specs()

    assert len(defects) == len(expected)

    for defect, (name, center, radius) in zip(defects, expected, strict=True):
        assert defect.name == name
        assert defect.center == pytest.approx(center)
        assert defect.radius == pytest.approx(radius)


def test_legacy_defects_use_euclidean_norm_and_twenty_percent_reduction():
    defects = legacy_defect_specs()

    for defect in defects:
        assert defect.norm_order == LEGACY_DEFECT_NORM_ORDER
        assert defect.norm_order == 2
        assert defect.material_factor == pytest.approx(LEGACY_DEFECT_MATERIAL_FACTOR)
        assert defect.material_factor == pytest.approx(0.8)


def test_circular_defect_contains_point_with_euclidean_norm():
    defect = CircularDefectSpec(
        name="test_defect",
        center=(0.0, 0.0),
        radius=1.0,
        norm_order=2,
    )

    assert defect.contains_point((0.0, 0.0))
    assert defect.contains_point((1.0, 0.0))
    assert defect.contains_point((0.6, 0.8))
    assert not defect.contains_point((0.8, 0.8))


def test_circular_defect_contains_point_with_manhattan_norm():
    defect = CircularDefectSpec(
        name="test_defect",
        center=(0.0, 0.0),
        radius=1.0,
        norm_order=1,
    )

    assert defect.contains_point((0.5, 0.5))
    assert not defect.contains_point((0.6, 0.6))


def test_defect_material_constants_are_scaled_from_healthy_material():
    defect = legacy_defect_specs()[0]
    healthy = legacy_healthy_material()
    defective = defect.material_constants(healthy)

    assert defective.rho == pytest.approx(0.8 * healthy.rho)
    assert defective.C11 == pytest.approx(0.8 * healthy.C11)
    assert defective.C22 == pytest.approx(0.8 * healthy.C22)
    assert defective.C12 == pytest.approx(0.8 * healthy.C12)
    assert defective.C66 == pytest.approx(0.8 * healthy.C66)


def test_legacy_defective_material_matches_first_defect_material():
    healthy = legacy_healthy_material()
    expected = legacy_defect_specs()[0].material_constants(healthy)
    actual = legacy_defective_material()

    assert actual.as_tuple() == pytest.approx(expected.as_tuple())


def test_legacy_scenario_preserves_defect_order_and_lookup():
    scenario = legacy_defect_scenario()

    assert isinstance(scenario, DefectScenario)
    assert scenario.name == "legacy_cpp_five_circular_defects"
    assert [defect.name for defect in scenario.defects] == [
        "legacy_defect_1",
        "legacy_defect_2",
        "legacy_defect_3",
        "legacy_defect_4",
        "legacy_defect_5",
    ]

    assert scenario.first_defect_at_point((90.0, 90.0)).name == "legacy_defect_3"
    assert scenario.first_defect_at_point((75.0, 75.0)) is None


def test_invalid_circular_defect_specs_are_rejected():
    with pytest.raises(ValueError):
        CircularDefectSpec(name="", center=(0.0, 0.0), radius=1.0)

    with pytest.raises(ValueError):
        CircularDefectSpec(name="bad_radius", center=(0.0, 0.0), radius=0.0)

    with pytest.raises(ValueError):
        CircularDefectSpec(name="bad_factor", center=(0.0, 0.0), radius=1.0, material_factor=0.0)

    with pytest.raises(ValueError):
        CircularDefectSpec(name="bad_norm", center=(0.0, 0.0), radius=1.0, norm_order=3)


def test_duplicate_defect_names_are_rejected():
    defect = CircularDefectSpec(name="same_name", center=(0.0, 0.0), radius=1.0)

    with pytest.raises(ValueError):
        DefectScenario(name="bad_scenario", defects=(defect, defect))
