from __future__ import annotations

import numpy as np
import pytest

from elastic_fwi.wave_physics.modal_proxies import (
    SCIENTIFIC_NOTE,
    apparent_mode_proxy,
    component_dominance_ratio,
    longitudinal_transverse_projection,
    mean_projected_longitudinal_energy,
    mean_projected_transverse_energy,
)


def test_component_dominance_ratio_detects_x_dominance():
    values = np.zeros((4, 2, 2), dtype=float)
    values[..., 0] = 2.0
    values[..., 1] = 1.0

    summary = component_dominance_ratio(values)

    assert summary["dominant_component"] == "x"
    assert summary["x_over_y_energy_ratio"] == pytest.approx(4.0)
    assert "not as a full Lamb-mode decomposition" in SCIENTIFIC_NOTE


def test_projection_and_apparent_mode_proxy_are_finite():
    values = np.zeros((5, 2, 2), dtype=float)
    values[..., 0] = 3.0
    directions = np.asarray([1.0, 0.0])

    projection = longitudinal_transverse_projection(values, directions)
    long_energy = mean_projected_longitudinal_energy(values, directions)
    trans_energy = mean_projected_transverse_energy(values, directions)
    proxy = apparent_mode_proxy(values, directions)

    assert np.allclose(projection["transverse"], 0.0)
    assert long_energy > 0.0
    assert trans_energy == pytest.approx(0.0)
    assert proxy["apparent_mode_proxy"] == "longitudinal-proxy-dominated"
