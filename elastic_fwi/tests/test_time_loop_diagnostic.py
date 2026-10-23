import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_time_loop import run_time_loop_check


def test_time_loop_diagnostic_report_is_ok():
    report = run_time_loop_check(nx=2, ny=2, degree=1, theta=0.25, dt=0.1, num_steps=5)

    assert report["status"] == "ok"

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    assert report["space"]["global_size"] > 0
    assert report["space"]["block_size"] == 2

    assert report["operator_config"]["theta"] == pytest.approx(0.25)
    assert report["operator_config"]["dt"] == pytest.approx(0.1)
    assert report["operator_config"]["scale_theta_dt2"] == pytest.approx(0.0025)

    assert report["loop_config"]["num_steps"] == 5
    assert report["loop_config"]["final_time"] == pytest.approx(0.5)

    assert len(report["records"]) == 5

    times = [record["time"] for record in report["records"]]
    assert times == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])

    values = report["values"]

    assert np.isfinite(values["max_force_norm"])
    assert np.isfinite(values["max_rhs_norm"])
    assert np.isfinite(values["max_displacement_norm"])
    assert np.isfinite(values["final_displacement_norm"])
    assert np.isfinite(values["max_linear_residual_norm"])
    assert np.isfinite(values["max_linear_residual_relative"])

    assert values["max_force_norm"] > 0.0
    assert values["max_rhs_norm"] > 0.0
    assert values["max_displacement_norm"] > 0.0
    assert values["final_displacement_norm"] > 0.0
    assert values["max_linear_residual_norm"] <= 1.0e-8
    assert values["max_linear_residual_relative"] <= 1.0e-8
    assert values["max_ksp_iterations"] >= 0

    for record in report["records"]:
        assert record["ksp_converged"]
        assert record["linear_residual_norm"] <= 1.0e-8
        assert record["linear_residual_relative"] <= 1.0e-8
