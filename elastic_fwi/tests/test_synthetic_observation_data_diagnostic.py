import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_synthetic_observation_data import (
    run_synthetic_observation_data_check,
)


def test_synthetic_observation_data_diagnostic_report_is_ok():
    report = run_synthetic_observation_data_check(
        nx=2,
        ny=2,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        noise_level=1.0e-4,
        noise_seed=123,
    )

    assert report["status"] == "ok"

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    assert report["clean_data"]["kind"] == "synthetic_boundary_observation_data"
    assert report["clean_data"]["num_steps"] == 5
    assert report["clean_data"]["final_time"] == pytest.approx(0.5)

    assert report["clean_data"]["reflection"]["tag"] == 1
    assert report["clean_data"]["transmission"]["tag"] == 2

    clean_reflection = report["clean_data"]["reflection"]["values"]
    clean_transmission = report["clean_data"]["transmission"]["values"]
    noisy_reflection = report["noisy_data"]["reflection"]["values"]
    noisy_transmission = report["noisy_data"]["transmission"]["values"]

    assert len(clean_reflection) == 5
    assert len(clean_transmission) == 5
    assert len(noisy_reflection) == 5
    assert len(noisy_transmission) == 5

    assert np.all(np.isfinite(clean_reflection))
    assert np.all(np.isfinite(clean_transmission))
    assert np.all(np.isfinite(noisy_reflection))
    assert np.all(np.isfinite(noisy_transmission))

    assert min(clean_reflection) >= 0.0
    assert min(clean_transmission) >= 0.0
    assert min(noisy_reflection) >= 0.0
    assert min(noisy_transmission) >= 0.0

    assert max(clean_reflection) > 0.0

    assert report["noisy_data"]["metadata"]["noise_model"] == "gaussian"
    assert report["noisy_data"]["metadata"]["noise_level"] == pytest.approx(1.0e-4)
    assert report["noisy_data"]["metadata"]["noise_seed"] == 123

def test_synthetic_observation_data_diagnostic_report_is_json_serializable():
    report = run_synthetic_observation_data_check(
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
    assert '"synthetic_boundary_observation_data"' in payload

