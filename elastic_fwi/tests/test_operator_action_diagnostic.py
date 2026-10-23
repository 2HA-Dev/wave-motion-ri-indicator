import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_operator_action import run_operator_action_check


def test_operator_action_diagnostic_report_is_ok():
    report = run_operator_action_check(nx=2, ny=2, degree=1, theta=0.25, dt=0.1)

    assert report["status"] == "ok"

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    assert report["space"]["global_size"] > 0
    assert report["space"]["block_size"] == 2

    assert report["config"]["theta"] == pytest.approx(0.25)
    assert report["config"]["dt"] == pytest.approx(0.1)
    assert report["config"]["scale_theta_dt2"] == pytest.approx(0.0025)

    values = report["values"]

    assert values["mass_size"] == [report["space"]["global_size"]] * 2
    assert values["stiffness_size"] == [report["space"]["global_size"]] * 2
    assert values["effective_size"] == [report["space"]["global_size"]] * 2

    assert np.isfinite(values["mass_frobenius_norm"])
    assert np.isfinite(values["stiffness_frobenius_norm"])
    assert np.isfinite(values["effective_frobenius_norm"])

    assert values["mass_frobenius_norm"] > 0.0
    assert values["stiffness_frobenius_norm"] > 0.0
    assert values["effective_frobenius_norm"] > 0.0

    consistency = values["effective_action_consistency"]

    assert consistency["abs_error"] <= 1.0e-12
    assert consistency["rel_error"] <= 1.0e-12
