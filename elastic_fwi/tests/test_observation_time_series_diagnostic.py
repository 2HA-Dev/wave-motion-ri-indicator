import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_observation_time_series import (
    run_observation_time_series_check,
)


def test_observation_time_series_diagnostic_report_is_ok():
    report = run_observation_time_series_check(
        nx=2,
        ny=2,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
    )

    assert report["status"] == "ok"

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    assert report["space"]["global_size"] > 0
    assert report["space"]["block_size"] == 2

    assert report["loop_config"]["num_steps"] == 5
    assert report["loop_config"]["final_time"] == pytest.approx(0.5)
    assert report["loop_config"]["stored_snapshots"] == 5

    assert report["observations"]["gamma1"]["tag"] == 1
    assert report["observations"]["gamma2"]["tag"] == 2

    assert len(report["gamma1_records"]) == 5
    assert len(report["gamma2_records"]) == 5
    assert len(report["time_loop_records"]) == 5

    gamma1_values = [record["norm_squared"] for record in report["gamma1_records"]]
    gamma2_values = [record["norm_squared"] for record in report["gamma2_records"]]

    assert np.all(np.isfinite(gamma1_values))
    assert np.all(np.isfinite(gamma2_values))

    assert min(gamma1_values) >= 0.0
    assert min(gamma2_values) >= 0.0
    assert max(gamma1_values) > 0.0

    assert report["values"]["max_gamma1_norm_squared"] == pytest.approx(
        max(gamma1_values)
    )
    assert report["values"]["max_gamma2_norm_squared"] == pytest.approx(
        max(gamma2_values)
    )

    times = [record["time"] for record in report["gamma1_records"]]
    assert times == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
