from __future__ import annotations

import numpy as np
import pytest

from elastic_fwi.wave_physics.energy_flux import (
    compute_component_energy_flux_proxy,
    compute_directional_energy_proxy,
    summarize_energy_flux_proxy,
)


def test_component_energy_flux_proxy_integrates_components():
    times = np.asarray([0.0, 1.0, 2.0])
    values = np.ones((3, 2, 2), dtype=float)
    values[..., 1] = 2.0

    component = compute_component_energy_flux_proxy(values, times=times)

    assert component["integrated_component_x_energy"] == pytest.approx(4.0)
    assert component["integrated_component_y_energy"] == pytest.approx(16.0)
    assert component["integrated_total_component_energy"] == pytest.approx(20.0)


def test_directional_energy_flux_proxy_summary_is_prudent_and_finite():
    times = np.linspace(0.0, 1.0, 5)
    values = np.zeros((5, 2, 2), dtype=float)
    values[..., 0] = 1.0
    directions = np.asarray([1.0, 0.0])

    directional = compute_directional_energy_proxy(values, directions=directions, times=times)
    summary = summarize_energy_flux_proxy(
        values=values,
        times=times,
        directions=directions,
        name="reflection",
    )

    assert directional["integrated_longitudinal_energy"] > 0.0
    assert directional["integrated_transverse_energy"] == pytest.approx(0.0)
    assert summary["longitudinal_over_transverse_energy_ratio"] > 1.0
    assert "not a full mechanical energy flux" in summary["scientific_note"]
