from __future__ import annotations

import numpy as np
import pytest

from elastic_fwi.wave_physics.reflected_transmitted_energy import (
    signal_instantaneous_energy,
    summarize_reflected_transmitted_energy,
)
from elastic_fwi.wave_physics.reflection_transmission_ratio import (
    compute_reflection_transmission_ratio,
)


def test_signal_instantaneous_energy_sums_receivers_and_components():
    values = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 0.0], [0.0, 1.0]],
        ]
    )

    energy = signal_instantaneous_energy(values)

    assert energy.tolist() == pytest.approx([30.0, 5.0])


def test_reflection_transmission_energy_and_ratio_are_positive():
    times = np.asarray([0.0, 1.0, 2.0])
    reflection = np.ones((3, 2, 2))
    transmission = 2.0 * np.ones((3, 2, 2))

    energy = summarize_reflected_transmitted_energy(
        times=times,
        reflection_values=reflection,
        transmission_values=transmission,
    )
    ratio = compute_reflection_transmission_ratio(
        reflection=energy.reflection,
        transmission=energy.transmission,
    )

    assert energy.reflection.total_energy > 0.0
    assert energy.transmission.total_energy > energy.reflection.total_energy
    assert ratio.global_ratio == pytest.approx(0.25)
