from __future__ import annotations

import numpy as np
import pytest

from elastic_fwi.wave_physics.scattered_field import (
    compute_scattered_field,
    summarize_scattered_field,
)


def test_compute_scattered_field_uses_defect_minus_reference():
    reference = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    defect = np.asarray([[2.0, 1.0], [5.0, 1.0]])

    scattered = compute_scattered_field(defect=defect, reference=reference)

    assert np.allclose(scattered, [[1.0, -1.0], [2.0, -3.0]])


def test_summarize_scattered_field_reports_peak_time():
    times = np.asarray([0.0, 1.0, 2.0])
    values = np.asarray([[[1.0, 0.0]], [[2.0, 0.0]], [[0.5, 0.0]]])

    summary = summarize_scattered_field(times=times, scattered=values)

    assert summary.energy.tolist() == pytest.approx([1.0, 4.0, 0.25])
    assert summary.index_of_max_energy == 1
    assert summary.time_of_max_energy == pytest.approx(1.0)
