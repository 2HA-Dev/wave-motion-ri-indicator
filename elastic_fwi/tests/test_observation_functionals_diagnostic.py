import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_observation_functionals import (
    run_observation_functionals_check,
)


def test_observation_functionals_diagnostic_report_is_ok():
    report = run_observation_functionals_check(
        nx=2,
        ny=2,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        noise_level=1.0e-4,
        noise_seed=123,
        reflection_weight=2.0,
        transmission_weight=3.0,
    )

    assert report["status"] == "ok"

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    clean_clean = report["functionals"]["clean_vs_clean"]
    clean_noisy = report["functionals"]["clean_vs_noisy"]
    noisy_clean = report["functionals"]["noisy_vs_clean"]

    assert clean_clean["reflection"] == pytest.approx(0.0)
    assert clean_clean["transmission"] == pytest.approx(0.0)
    assert clean_clean["full"] == pytest.approx(0.0)
    assert clean_clean["weighted"] == pytest.approx(0.0)

    assert clean_noisy["reflection"] >= 0.0
    assert clean_noisy["transmission"] >= 0.0
    assert clean_noisy["full"] >= 0.0
    assert clean_noisy["weighted"] >= 0.0

    assert clean_noisy["full"] == pytest.approx(
        clean_noisy["reflection"] + clean_noisy["transmission"]
    )

    assert clean_noisy["weighted"] == pytest.approx(
        2.0 * clean_noisy["reflection"] + 3.0 * clean_noisy["transmission"]
    )

    assert clean_noisy["full"] == pytest.approx(noisy_clean["full"])
    assert clean_noisy["weighted"] == pytest.approx(noisy_clean["weighted"])

    assert clean_noisy["reflection_weight"] == pytest.approx(2.0)
    assert clean_noisy["transmission_weight"] == pytest.approx(3.0)

    assert np.isfinite(report["values"]["max_abs_noise"])
    assert report["values"]["max_abs_noise"] > 0.0


def test_observation_functionals_diagnostic_report_is_json_serializable():
    report = run_observation_functionals_check(
        nx=2,
        ny=2,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        noise_level=1.0e-4,
        noise_seed=123,
    )

    payload = json.dumps(report, sort_keys=True)

    assert '"status": "ok"' in payload
    assert '"clean_vs_noisy"' in payload
    assert '"weighted"' in payload
