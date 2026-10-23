import json
import subprocess
import sys
from pathlib import Path

import pytest

from elastic_fwi.diagnostics.check_sampling_conditions import (
    check_sampling_conditions,
    estimate_wave_speeds,
)


def test_estimate_wave_speeds_are_positive_and_ordered():
    c_min, c_max = estimate_wave_speeds(
        rho=4.5,
        C11=16.24,
        C22=18.07,
        C66=4.67,
    )

    assert c_min > 0.0
    assert c_max > 0.0
    assert c_min <= c_max


def test_sampling_conditions_accept_default_normalized_case():
    result = check_sampling_conditions(
        nx=8,
        ny=8,
        dt=0.1,
        source_frequency=1.0,
    )

    assert result.kind == "elastic_sampling_conditions"
    assert result.status == "ok"
    assert result.shannon_spatial_ok is True
    assert result.shannon_temporal_ok is True
    assert result.cfl_ok is True
    assert result.points_per_wavelength >= 2.0
    assert result.time_steps_per_period >= 2.0


def test_sampling_conditions_warn_when_practical_ppw_is_low():
    result = check_sampling_conditions(
        nx=8,
        ny=8,
        dt=0.1,
        source_frequency=1.0,
        recommended_min_points_per_wavelength=10.0,
    )

    assert result.status == "ok"
    assert result.recommended_status == "warning"
    assert result.recommended_spatial_ok is False
    assert result.recommended_temporal_ok is True


def test_sampling_conditions_fail_when_spatial_shannon_is_violated():
    result = check_sampling_conditions(
        nx=2,
        ny=2,
        dt=0.1,
        source_frequency=10.0,
    )

    assert result.status == "failed"
    assert result.shannon_spatial_ok is False


def test_sampling_conditions_fail_when_temporal_shannon_is_violated():
    result = check_sampling_conditions(
        nx=64,
        ny=64,
        dt=0.2,
        source_frequency=10.0,
        cfl_max=100.0,
    )

    assert result.status == "failed"
    assert result.shannon_temporal_ok is False


def test_sampling_conditions_fail_when_cfl_is_too_large():
    result = check_sampling_conditions(
        nx=8,
        ny=8,
        dt=1.0,
        source_frequency=1.0,
        cfl_max=1.0,
    )

    assert result.status == "failed"
    assert result.cfl_ok is False


def test_sampling_conditions_strict_recommended_can_fail():
    result = check_sampling_conditions(
        nx=8,
        ny=8,
        dt=0.1,
        source_frequency=1.0,
        strict_recommended=True,
    )

    assert result.status == "failed"
    assert result.recommended_status == "warning"


def test_sampling_conditions_cli_writes_json(tmp_path: Path):
    output_json = tmp_path / "sampling.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "elastic_fwi.diagnostics.check_sampling_conditions",
            "--nx",
            "8",
            "--ny",
            "8",
            "--dt",
            "0.1",
            "--source-frequency",
            "1.0",
            "--output-json",
            str(output_json),
            "--quiet",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert output_json.exists()

    with output_json.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    assert data["kind"] == "elastic_sampling_conditions"
    assert data["status"] == "ok"
    assert data["points_per_wavelength"] >= 2.0


def test_sampling_conditions_reject_invalid_inputs():
    with pytest.raises(ValueError):
        check_sampling_conditions(
            nx=0,
            ny=8,
            dt=0.1,
            source_frequency=1.0,
        )

    with pytest.raises(ValueError):
        check_sampling_conditions(
            nx=8,
            ny=8,
            dt=-0.1,
            source_frequency=1.0,
        )
