import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_adjoint_newmark_pipeline import (
    run_adjoint_newmark_pipeline_check,
)


def test_adjoint_newmark_pipeline_diagnostic_default_case():
    report = run_adjoint_newmark_pipeline_check(
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

    assert report["kind"] == "adjoint_newmark_pipeline_check"

    assert report["values"]["matrix_rows"] == report["values"]["matrix_cols"]
    assert report["values"]["matrix_rows"] > 0
    assert report["values"]["observed_defect_cells"] > 0
    assert report["values"]["max_abs_residual"] > 0.0
    assert report["values"]["max_rhs_norm"] > 0.0
    assert report["values"]["max_equation_adjoint_norm"] > 0.0
    assert report["values"]["max_linear_residual_norm"] <= 1.0e-8
    assert np.isfinite(report["values"]["initial_current_adjoint_norm"])
    assert np.isfinite(report["values"]["initial_previous_adjoint_norm"])

    assert report["comparison"]["status"] == "ok"
    assert report["adjoint_rhs"]["kind"] == "full_adjoint_observation_rhs_result"
    assert report["newmark_adjoint"]["kind"] == "newmark_adjoint_state_result"
    assert report["poo_pipeline"]["kind"] == "adjoint_newmark_pipeline_result"

    records = report["newmark_adjoint"]["records"]

    assert len(records) == 5
    assert records[0]["step"] == 1
    assert records[0]["solve_index"] == 5
    assert records[-1]["step"] == 5
    assert records[-1]["solve_index"] == 1

    for record in records:
        assert record["ksp_converged"] is True
        assert record["ksp_iterations"] > 0
        assert np.isfinite(record["observation_rhs_norm"])
        assert np.isfinite(record["accumulated_rhs_norm"])
        assert np.isfinite(record["equation_adjoint_norm"])
        assert np.isfinite(record["carry_current_norm"])
        assert np.isfinite(record["carry_previous_norm"])
        assert np.isfinite(record["linear_residual_norm"])
        assert record["linear_residual_norm"] <= 1.0e-8


def test_adjoint_newmark_pipeline_diagnostic_report_is_json_serializable():
    report = run_adjoint_newmark_pipeline_check(
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

    assert '"adjoint_newmark_pipeline_check"' in payload
    assert '"adjoint_newmark_pipeline_result"' in payload
    assert '"newmark_adjoint_state_result"' in payload
    assert '"full_adjoint_observation_rhs_result"' in payload
    assert '"direct_observation_comparison_result"' in payload
