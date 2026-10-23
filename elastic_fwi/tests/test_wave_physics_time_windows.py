from __future__ import annotations

import numpy as np
import pytest

from elastic_fwi.wave_physics.time_windows import (
    TimeWindow,
    default_time_windows,
    parse_time_window,
    summarize_windowed_energies,
)


def test_parse_time_window():
    window = parse_time_window("early:0.1:0.5")

    assert window.name == "early"
    assert window.start_time == pytest.approx(0.1)
    assert window.end_time == pytest.approx(0.5)


def test_default_time_windows_include_full():
    windows = default_time_windows(np.linspace(0.0, 1.0, 11))

    assert windows[-1].name == "full"
    assert windows[-1].start_time == pytest.approx(0.0)
    assert windows[-1].end_time == pytest.approx(1.0)


def test_summarize_windowed_energies():
    times = np.linspace(0.0, 1.0, 11)
    reflection = np.ones_like(times)
    transmission = 2.0 * np.ones_like(times)

    rows = summarize_windowed_energies(
        times=times,
        reflection_energy=reflection,
        transmission_energy=transmission,
        windows=[TimeWindow("full", 0.0, 1.0)],
    )

    assert rows[0]["window"] == "full"
    assert rows[0]["reflection_energy"] == pytest.approx(1.0)
    assert rows[0]["transmission_energy"] == pytest.approx(2.0)
    assert rows[0]["reflection_transmission_ratio"] == pytest.approx(0.5)
