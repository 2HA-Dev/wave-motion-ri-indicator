import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_newmark_adjoint_duality import (
    run_newmark_adjoint_duality_check,
)


def test_newmark_adjoint_duality_diagnostic_healthy_material():
    report = run_newmark_adjoint_duality_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        material_kind="healthy",
        duality_tolerance=1.0e-8,
    )

    assert report["status"] == "ok"

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    assert report["kind"] == "newmark_adjoint_duality_check"
    assert report["matrix"]["rows"] == report["matrix"]["cols"]
    assert report["matrix"]["rows"] > 0

    assert report["values"]["duality_abs"] <= report["values"][
        "duality_tolerance"
    ] * report["values"]["duality_scale"]

    assert np.isfinite(report["values"]["duality_rel"])
    assert report["values"]["max_tangent_norm"] > 0.0
    assert report["values"]["max_adjoint_norm"] > 0.0

    assert report["tangent"]["num_steps"] == 5
    assert report["adjoint"]["kind"] == "newmark_adjoint_state_result"
    assert report["rhs"]["kind"] == "full_adjoint_observation_rhs_result"


def test_newmark_adjoint_duality_diagnostic_localized_material():
    report = run_newmark_adjoint_duality_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        material_kind="localized_disk_defect",
        contrast_factor=0.8,
        defect_radius=0.35,
        duality_tolerance=1.0e-8,
    )

    assert report["status"] == "ok"

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    assert report["material"]["kind"] == "localized_disk_defect"
    assert report["material"]["global_defect_cells"] > 0
    assert report["values"]["duality_abs"] <= report["values"][
        "duality_tolerance"
    ] * report["values"]["duality_scale"]


def test_newmark_adjoint_duality_report_is_json_serializable():
    report = run_newmark_adjoint_duality_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        material_kind="healthy",
    )

    payload = json.dumps(report, sort_keys=True)

    assert '"newmark_adjoint_duality_check"' in payload
    assert '"newmark_adjoint_state_result"' in payload
    assert '"full_adjoint_observation_rhs_result"' in payload
