import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_boundary_observation import (
    run_boundary_observation_check,
)


def test_boundary_observation_diagnostic_report_is_ok():
    report = run_boundary_observation_check(nx=4, ny=4)

    assert report["status"] == "ok"

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    values = report["values"]

    assert values["norm_gamma1_vector"] == pytest.approx(5.0)
    assert values["norm_gamma1_x"] == pytest.approx(4.0)
    assert values["norm_gamma1_y"] == pytest.approx(1.0)

    assert values["inner_gamma2_vector"] == pytest.approx(-1.0)
    assert values["inner_gamma2_x"] == pytest.approx(2.0)
    assert values["inner_gamma2_y"] == pytest.approx(-3.0)

    assert values["misfit_gamma1_vector"] == pytest.approx(5.0)
    assert values["misfit_gamma1_x"] == pytest.approx(1.0)
    assert values["misfit_gamma1_y"] == pytest.approx(4.0)
    assert values["cost_gamma1_vector"] == pytest.approx(2.5)

    residual = values["residual_report"]
    assert residual["observation"]["name"] == "gamma2_transmission"
    assert residual["observation"]["tag"] == 2
    assert residual["observation"]["component"] == "vector"
    assert residual["norm_squared"] == pytest.approx(5.0)
    assert residual["cost"] == pytest.approx(2.5)
