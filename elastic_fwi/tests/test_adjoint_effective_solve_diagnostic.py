import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_adjoint_effective_solve import (
    run_adjoint_effective_solve_check,
)


def test_adjoint_effective_solve_diagnostic_default_case():
    report = run_adjoint_effective_solve_check(
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

    assert report["kind"] == "adjoint_effective_solve_check"

    assert report["values"]["matrix_rows"] == report["values"]["matrix_cols"]
    assert report["values"]["matrix_rows"] > 0

    assert report["values"]["observed_defect_cells"] > 0
    assert report["values"]["max_abs_residual"] > 0.0
    assert report["values"]["max_rhs_norm"] > 0.0
    assert report["values"]["max_solution_norm"] > 0.0
    assert report["values"]["max_linear_residual_norm"] <= 1.0e-8

    solve = report["adjoint_solve"]

    assert solve["kind"] == "adjoint_effective_solve_result"
    assert solve["num_steps"] == 5
    assert solve["reverse_time"] is True
    assert solve["all_ksp_converged"] is True

    records = solve["records"]

    assert records[0]["step"] == 1
    assert records[0]["solve_index"] == 5
    assert records[-1]["step"] == 5
    assert records[-1]["solve_index"] == 1

    for record in records:
        assert record["ksp_converged"] is True
        assert record["ksp_iterations"] > 0
        assert np.isfinite(record["rhs_norm"])
        assert np.isfinite(record["solution_norm"])
        assert np.isfinite(record["linear_residual_norm"])
        assert record["linear_residual_norm"] <= 1.0e-8

    assert report["adjoint_rhs"]["kind"] == "full_adjoint_observation_rhs_result"
    assert report["comparison"]["residuals"]["kind"] == "observation_residual_data"


def test_adjoint_effective_solve_diagnostic_report_is_json_serializable():
    report = run_adjoint_effective_solve_check(
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

    assert '"adjoint_effective_solve_check"' in payload
    assert '"adjoint_effective_solve_result"' in payload
    assert '"full_adjoint_observation_rhs_result"' in payload
    assert '"observation_residual_data"' in payload
