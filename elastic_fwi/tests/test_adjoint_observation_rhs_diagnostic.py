import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_adjoint_observation_rhs import (
    run_adjoint_observation_rhs_check,
)


def test_adjoint_observation_rhs_diagnostic_default_case():
    report = run_adjoint_observation_rhs_check(
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

    assert report["status"] == "ok"

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    assert report["kind"] == "adjoint_observation_rhs_check"
    assert report["values"]["observed_defect_cells"] > 0
    assert report["values"]["max_abs_residual"] > 0.0
    assert report["values"]["max_rhs_norm"] > 0.0
    assert report["values"]["reflection_max_rhs_norm"] > 0.0
    assert report["values"]["transmission_max_rhs_norm"] >= 0.0

    rhs = report["adjoint_rhs"]

    assert rhs["kind"] == "full_adjoint_observation_rhs_result"
    assert rhs["num_steps"] == 5
    assert rhs["reflection"]["num_steps"] == 5
    assert rhs["transmission"]["num_steps"] == 5

    for block_name in ("reflection", "transmission"):
        block = rhs[block_name]

        for record in block["records"]:
            assert record["rhs_size_global"] > 0
            assert record["rhs_size_local"] >= 0
            assert np.isfinite(record["rhs_norm"])
            assert np.isfinite(record["residual_value"])

    assert report["comparison"]["residuals"]["kind"] == "observation_residual_data"
    assert report["comparison"]["simulated"]["state"][
        "displacement_snapshots_stored"
    ] is True


def test_adjoint_observation_rhs_diagnostic_report_is_json_serializable():
    report = run_adjoint_observation_rhs_check(
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

    assert '"adjoint_observation_rhs_check"' in payload
    assert '"full_adjoint_observation_rhs_result"' in payload
    assert '"observation_residual_data"' in payload
