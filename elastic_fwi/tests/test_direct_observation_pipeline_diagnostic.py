import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_direct_observation_pipeline import (
    run_direct_observation_pipeline_check,
)


def test_direct_observation_pipeline_diagnostic_detects_localized_defect():
    report = run_direct_observation_pipeline_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        simulated_material_kind="healthy",
        observed_material_kind="localized_disk_defect",
        contrast_factor=0.8,
        defect_radius=0.35,
        reflection_weight=2.0,
        transmission_weight=3.0,
    )

    assert report["status"] == "ok"

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    assert report["values"]["observed_defect_cells"] > 0
    assert report["values"]["max_abs_observation_difference"] > 0.0

    values = report["functionals"]["simulated_vs_observed"]

    assert values["reflection"] > 0.0
    assert values["full"] > 0.0
    assert values["full"] == pytest.approx(
        values["reflection"] + values["transmission"]
    )
    assert values["weighted"] == pytest.approx(
        2.0 * values["reflection"] + 3.0 * values["transmission"]
    )


def test_direct_observation_pipeline_diagnostic_zero_for_same_material():
    report = run_direct_observation_pipeline_check(
        nx=2,
        ny=2,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        simulated_material_kind="healthy",
        observed_material_kind="healthy",
    )

    assert report["status"] == "ok"

    values = report["functionals"]["simulated_vs_observed"]

    assert values["reflection"] == pytest.approx(0.0)
    assert values["transmission"] == pytest.approx(0.0)
    assert values["full"] == pytest.approx(0.0)
    assert values["weighted"] == pytest.approx(0.0)


def test_direct_observation_pipeline_diagnostic_homogeneous_contrast():
    report = run_direct_observation_pipeline_check(
        nx=2,
        ny=2,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        simulated_material_kind="healthy",
        observed_material_kind="homogeneous_contrast",
        contrast_factor=0.8,
    )

    assert report["status"] == "ok"

    values = report["functionals"]["simulated_vs_observed"]

    assert values["reflection"] > 0.0
    assert values["full"] > 0.0
    assert np.isfinite(report["values"]["max_abs_observation_difference"])
    assert report["values"]["max_abs_observation_difference"] > 0.0


def test_direct_observation_pipeline_diagnostic_report_is_json_serializable():
    report = run_direct_observation_pipeline_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        simulated_material_kind="healthy",
        observed_material_kind="localized_disk_defect",
        contrast_factor=0.8,
        defect_radius=0.35,
    )

    payload = json.dumps(report, sort_keys=True)

    assert '"status": "ok"' in payload
    assert '"direct_observation_pipeline_result"' in payload
    assert '"simulated_vs_observed"' in payload
