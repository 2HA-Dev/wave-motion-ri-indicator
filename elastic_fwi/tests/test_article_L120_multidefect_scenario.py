from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import numpy as np

from elastic_fwi.materials.defect_json import (
    load_defect_scenario_json,
)
from elastic_fwi.materials.orthotropic import (
    LEGACY_HEALTHY_MATERIAL,
)
from elastic_fwi.materials.point_fields import (
    defective_material_fields_at_points,
)


SCENARIO_PATH = Path(
    "configs/defects/"
    "article_L120_multidefect_5_elastic_only.json"
)


def test_article_L120_multidefect_scenario_geometry() -> None:
    scenario, metadata = load_defect_scenario_json(
        SCENARIO_PATH
    )

    assert len(scenario.defects) == 5
    assert metadata["name"] == (
        "article_L120_multidefect_5_elastic_only"
    )
    assert metadata["conflict_policy"] == "error"

    expected = [
        ("D1_center", 60.0, 60.0, 0.5, 1.0, 0.75),
        ("D2_bottom_left", 56.0, 56.0, 0.4, 1.0, 0.80),
        ("D3_bottom_right", 64.0, 56.0, 0.7, 1.0, 0.60),
        ("D4_top_left", 56.0, 64.0, 0.8, 1.0, 0.90),
        ("D5_top_right", 64.0, 64.0, 0.6, 1.0, 0.50),
    ]

    actual = [
        (
            defect.name,
            defect.center_x,
            defect.center_y,
            defect.radius,
            defect.rho_factor,
            defect.stiffness_factor,
        )
        for defect in scenario.defects
    ]

    assert actual == expected


def test_article_L120_multidefect_scenario_no_overlap() -> None:
    scenario, _ = load_defect_scenario_json(
        SCENARIO_PATH
    )

    for first, second in combinations(
        scenario.defects,
        2,
    ):
        distance = math.hypot(
            second.center_x - first.center_x,
            second.center_y - first.center_y,
        )

        assert distance > first.radius + second.radius


def test_article_L120_multidefect_scenario_inside_roi() -> None:
    scenario, _ = load_defect_scenario_json(
        SCENARIO_PATH
    )

    for defect in scenario.defects:
        assert defect.center_x - defect.radius >= 50.0
        assert defect.center_x + defect.radius <= 70.0
        assert defect.center_y - defect.radius >= 50.0
        assert defect.center_y + defect.radius <= 70.0


def test_article_L120_multidefect_material_values() -> None:
    scenario, _ = load_defect_scenario_json(
        SCENARIO_PATH
    )

    points = np.array(
        [
            [60.0, 60.0],
            [56.0, 56.0],
            [64.0, 56.0],
            [56.0, 64.0],
            [64.0, 64.0],
            [40.0, 40.0],
        ],
        dtype=float,
    )

    expected_stiffness_factors = np.array(
        [0.75, 0.80, 0.60, 0.90, 0.50, 1.00],
        dtype=float,
    )

    fields = defective_material_fields_at_points(
        points,
        scenario,
        background_factor=1.0,
        conflict_policy="error",
    )

    np.testing.assert_allclose(
        fields.rho,
        np.full(
            points.shape[0],
            LEGACY_HEALTHY_MATERIAL.rho,
        ),
    )

    np.testing.assert_allclose(
        fields.C11,
        (
            LEGACY_HEALTHY_MATERIAL.C11
            * expected_stiffness_factors
        ),
    )

    np.testing.assert_allclose(
        fields.C22,
        (
            LEGACY_HEALTHY_MATERIAL.C22
            * expected_stiffness_factors
        ),
    )

    np.testing.assert_allclose(
        fields.C12,
        (
            LEGACY_HEALTHY_MATERIAL.C12
            * expected_stiffness_factors
        ),
    )

    np.testing.assert_allclose(
        fields.C66,
        (
            LEGACY_HEALTHY_MATERIAL.C66
            * expected_stiffness_factors
        ),
    )


def test_article_L120_multidefect_center_positions_symmetric() -> None:
    scenario, _ = load_defect_scenario_json(
        SCENARIO_PATH
    )

    relative_centers = {
        (
            defect.center_x - 60.0,
            defect.center_y - 60.0,
        )
        for defect in scenario.defects
    }

    assert relative_centers == {
        (0.0, 0.0),
        (-4.0, -4.0),
        (+4.0, -4.0),
        (-4.0, +4.0),
        (+4.0, +4.0),
    }
