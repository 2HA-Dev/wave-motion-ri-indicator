import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_newmark_step import run_newmark_step_check


def test_newmark_step_diagnostic_report_is_ok():
    report = run_newmark_step_check(nx=2, ny=2, degree=1, theta=0.25, dt=0.1)

    assert report["status"] == "ok"

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    assert report["space"]["global_size"] > 0
    assert report["space"]["block_size"] == 2

    assert report["operator_config"]["theta"] == pytest.approx(0.25)
    assert report["operator_config"]["dt"] == pytest.approx(0.1)
    assert report["operator_config"]["scale_theta_dt2"] == pytest.approx(0.0025)

    assert report["source"]["tag"] == 1
    assert report["source"]["time"] == pytest.approx(0.25)
    assert report["source"]["value"] == pytest.approx([0.0, 0.5])

    assert report["solve"]["converged"]
    assert report["solve"]["iterations"] >= 0

    values = report["values"]

    assert np.isfinite(values["force_norm"])
    assert np.isfinite(values["rhs_norm"])
    assert np.isfinite(values["displacement_norm"])
    assert np.isfinite(values["linear_residual_norm"])
    assert np.isfinite(values["linear_residual_relative"])

    assert values["force_norm"] > 0.0
    assert values["rhs_norm"] > 0.0
    assert values["displacement_norm"] > 0.0
    assert values["linear_residual_norm"] <= 1.0e-8
    assert values["linear_residual_relative"] <= 1.0e-8
