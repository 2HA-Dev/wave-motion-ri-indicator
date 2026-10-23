import json
from pathlib import Path

import numpy as np
import pytest

from elastic_fwi.materials.defect_json import load_defect_scenario_json
from elastic_fwi.materials.defects import (
    CircularDefectSpec,
    DefectScenario,
    assign_material_component_factors_to_points,
)
from elastic_fwi.materials.orthotropic import LEGACY_HEALTHY_MATERIAL
from elastic_fwi.materials.point_fields import defective_material_fields_at_points


def test_legacy_common_factor_semantics_remain_unchanged():
    defect = CircularDefectSpec(
        name="legacy",
        center=(0.0, 0.0),
        radius=1.0,
        material_factor=0.8,
    )

    assert defect.material_factor == 0.8
    assert defect.rho_factor == 0.8
    assert defect.stiffness_factor == 0.8

    material = defect.material_constants()
    assert material.rho == pytest.approx(0.8 * LEGACY_HEALTHY_MATERIAL.rho)
    assert material.C11 == pytest.approx(0.8 * LEGACY_HEALTHY_MATERIAL.C11)


def test_component_factors_keep_density_and_scale_stiffness():
    scenario = DefectScenario(
        name="elastic_only",
        defects=(
            CircularDefectSpec(
                name="D1",
                center=(0.0, 0.0),
                radius=1.0,
                rho_factor=1.0,
                stiffness_factor=0.75,
            ),
        ),
    )

    points = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    fields = defective_material_fields_at_points(
        points,
        scenario,
        conflict_policy="error",
    )

    np.testing.assert_allclose(
        fields.rho,
        [LEGACY_HEALTHY_MATERIAL.rho, LEGACY_HEALTHY_MATERIAL.rho],
    )
    np.testing.assert_allclose(
        fields.C11,
        [0.75 * LEGACY_HEALTHY_MATERIAL.C11, LEGACY_HEALTHY_MATERIAL.C11],
    )
    np.testing.assert_allclose(
        fields.C22,
        [0.75 * LEGACY_HEALTHY_MATERIAL.C22, LEGACY_HEALTHY_MATERIAL.C22],
    )
    np.testing.assert_allclose(
        fields.C12,
        [0.75 * LEGACY_HEALTHY_MATERIAL.C12, LEGACY_HEALTHY_MATERIAL.C12],
    )
    np.testing.assert_allclose(
        fields.C66,
        [0.75 * LEGACY_HEALTHY_MATERIAL.C66, LEGACY_HEALTHY_MATERIAL.C66],
    )


def test_component_factor_assignment_supports_distinct_defects():
    scenario = DefectScenario(
        name="two_defects",
        defects=(
            CircularDefectSpec(
                name="D1",
                center=(0.0, 0.0),
                radius=0.5,
                rho_factor=1.0,
                stiffness_factor=0.75,
            ),
            CircularDefectSpec(
                name="D2",
                center=(2.0, 0.0),
                radius=0.5,
                rho_factor=1.0,
                stiffness_factor=0.50,
            ),
        ),
    )

    points = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    rho_factors, stiffness_factors = (
        assign_material_component_factors_to_points(
            points,
            scenario,
            conflict_policy="error",
        )
    )

    np.testing.assert_allclose(rho_factors, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(stiffness_factors, [0.75, 0.50, 1.0])


def test_json_loader_supports_component_factor_schema(tmp_path: Path):
    path = tmp_path / "defects.json"
    path.write_text(
        json.dumps(
            {
                "name": "elastic_only",
                "background_factor": 1.0,
                "conflict_policy": "error",
                "defects": [
                    {
                        "name": "D1",
                        "center_x": 60.0,
                        "center_y": 60.0,
                        "radius": 0.5,
                        "rho_factor": 1.0,
                        "stiffness_factor": 0.75,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    scenario, metadata = load_defect_scenario_json(path)
    defect = scenario.defects[0]

    assert defect.material_factor is None
    assert defect.rho_factor == 1.0
    assert defect.stiffness_factor == 0.75
    assert metadata["defects"][0]["material_factor"] is None
    assert metadata["defects"][0]["rho_factor"] == 1.0
    assert metadata["defects"][0]["stiffness_factor"] == 0.75


def test_json_loader_rejects_ambiguous_factor_schema(tmp_path: Path):
    path = tmp_path / "ambiguous.json"
    path.write_text(
        json.dumps(
            {
                "defects": [
                    {
                        "name": "D1",
                        "center_x": 0.0,
                        "center_y": 0.0,
                        "radius": 1.0,
                        "material_factor": 0.8,
                        "rho_factor": 1.0,
                        "stiffness_factor": 0.75,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mixes"):
        load_defect_scenario_json(path)
