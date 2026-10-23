from __future__ import annotations

import numpy as np
import pytest

from elastic_fwi.wave_physics.residual_spectrum import (
    compute_residual_spectrum,
    infer_uniform_dt,
)


def test_infer_uniform_dt():
    times = np.asarray([0.0, 0.1, 0.2, 0.3])

    assert infer_uniform_dt(times) == pytest.approx(0.1)


def test_residual_spectrum_detects_dominant_frequency():
    dt = 0.01
    times = np.arange(0.0, 2.0, dt)
    frequency = 3.0
    signal = np.sin(2.0 * np.pi * frequency * times)
    values = signal[:, None, None]

    summary = compute_residual_spectrum(
        times=times,
        values=values,
        name="test",
    )

    assert summary.dominant_frequency == pytest.approx(frequency, abs=0.1)
    assert summary.peak_power > 0.0
