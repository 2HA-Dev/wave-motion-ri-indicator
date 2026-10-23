from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from elastic_fwi.wave_physics.reflected_transmitted_energy import SignalEnergySummary


@dataclass(frozen=True)
class ReflectionTransmissionRatioSummary:
    times: np.ndarray
    ratio_time: np.ndarray
    global_ratio: float
    epsilon: float
    max_ratio: float
    time_of_max_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "reflection_transmission_ratio_summary",
            "global_ratio": float(self.global_ratio),
            "epsilon": float(self.epsilon),
            "max_ratio": float(self.max_ratio),
            "time_of_max_ratio": float(self.time_of_max_ratio),
            "num_steps": int(self.times.size),
        }


def compute_reflection_transmission_ratio(
    *,
    reflection: SignalEnergySummary,
    transmission: SignalEnergySummary,
    epsilon: float = 1.0e-30,
) -> ReflectionTransmissionRatioSummary:
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive and finite.")

    if reflection.times.shape != transmission.times.shape:
        raise ValueError("reflection and transmission must have same time shape.")

    if not np.allclose(reflection.times, transmission.times, rtol=0.0, atol=1.0e-14):
        raise ValueError("reflection and transmission times must match.")

    ratio_time = reflection.instantaneous_energy / (
        transmission.instantaneous_energy + epsilon
    )
    global_ratio = reflection.total_energy / (transmission.total_energy + epsilon)

    if ratio_time.size:
        idx = int(np.argmax(ratio_time))
        max_ratio = float(ratio_time[idx])
        tmax = float(reflection.times[idx])
    else:
        max_ratio = 0.0
        tmax = 0.0

    return ReflectionTransmissionRatioSummary(
        times=reflection.times.copy(),
        ratio_time=ratio_time,
        global_ratio=float(global_ratio),
        epsilon=epsilon,
        max_ratio=max_ratio,
        time_of_max_ratio=tmax,
    )
