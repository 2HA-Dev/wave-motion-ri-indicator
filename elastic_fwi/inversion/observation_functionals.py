from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np

from elastic_fwi.data.synthetic_observations import (
    BoundaryObservationSeries,
    SyntheticBoundaryObservationData,
)


@dataclass(frozen=True)
class ObservationFunctionalWeights:
    """Weights for reflection/transmission observation misfit functionals."""

    reflection: float = 1.0
    transmission: float = 1.0

    def __post_init__(self) -> None:
        reflection = float(self.reflection)
        transmission = float(self.transmission)

        if not isfinite(reflection) or reflection < 0.0:
            raise ValueError("reflection weight must be finite and non-negative.")

        if not isfinite(transmission) or transmission < 0.0:
            raise ValueError("transmission weight must be finite and non-negative.")

        object.__setattr__(self, "reflection", reflection)
        object.__setattr__(self, "transmission", transmission)

    def as_dict(self) -> dict[str, float]:
        return {
            "reflection": self.reflection,
            "transmission": self.transmission,
        }


@dataclass(frozen=True)
class ObservationFunctionalValues:
    """Scalar values of the reflection/transmission/full/weighted misfits."""

    reflection: float
    transmission: float
    full: float
    weighted: float
    reflection_weight: float
    transmission_weight: float

    def as_dict(self) -> dict[str, float]:
        return {
            "reflection": self.reflection,
            "transmission": self.transmission,
            "full": self.full,
            "weighted": self.weighted,
            "reflection_weight": self.reflection_weight,
            "transmission_weight": self.transmission_weight,
        }


def _as_time_weights(times: np.ndarray, time_weights: Sequence[float] | None) -> np.ndarray:
    if time_weights is None:
        return np.ones_like(times, dtype=float)

    weights = np.asarray(time_weights, dtype=float)

    if weights.ndim != 1:
        raise ValueError("time_weights must be one-dimensional.")

    if weights.size != times.size:
        raise ValueError("time_weights must have the same length as the time series.")

    if not np.all(np.isfinite(weights)):
        raise ValueError("time_weights must be finite.")

    if np.any(weights < 0.0):
        raise ValueError("time_weights must be non-negative.")

    return weights.copy()


def validate_matching_observation_series(
    simulated: BoundaryObservationSeries,
    observed: BoundaryObservationSeries,
) -> None:
    if simulated.name != observed.name:
        raise ValueError("Observation series names must match.")

    if simulated.tag != observed.tag:
        raise ValueError("Observation boundary tags must match.")

    if simulated.component != observed.component:
        raise ValueError("Observation components must match.")

    if simulated.value_kind != observed.value_kind:
        raise ValueError("Observation value kinds must match.")

    if simulated.num_steps != observed.num_steps:
        raise ValueError("Observation series lengths must match.")

    if not np.allclose(simulated.times, observed.times, rtol=0.0, atol=0.0):
        raise ValueError("Observation series times must match exactly.")


def observation_series_misfit(
    simulated: BoundaryObservationSeries,
    observed: BoundaryObservationSeries,
    *,
    time_weights: Sequence[float] | None = None,
) -> float:
    validate_matching_observation_series(simulated, observed)

    weights = _as_time_weights(simulated.times, time_weights)
    residual = np.asarray(simulated.values, dtype=float) - np.asarray(
        observed.values,
        dtype=float,
    )

    return float(0.5 * np.sum(weights * residual**2))


def reflection_misfit(
    simulated: SyntheticBoundaryObservationData,
    observed: SyntheticBoundaryObservationData,
    *,
    time_weights: Sequence[float] | None = None,
) -> float:
    return observation_series_misfit(
        simulated.reflection,
        observed.reflection,
        time_weights=time_weights,
    )


def transmission_misfit(
    simulated: SyntheticBoundaryObservationData,
    observed: SyntheticBoundaryObservationData,
    *,
    time_weights: Sequence[float] | None = None,
) -> float:
    return observation_series_misfit(
        simulated.transmission,
        observed.transmission,
        time_weights=time_weights,
    )


def full_aperture_misfit(
    simulated: SyntheticBoundaryObservationData,
    observed: SyntheticBoundaryObservationData,
    *,
    time_weights: Sequence[float] | None = None,
) -> float:
    return float(
        reflection_misfit(simulated, observed, time_weights=time_weights)
        + transmission_misfit(simulated, observed, time_weights=time_weights)
    )


def weighted_aperture_misfit(
    simulated: SyntheticBoundaryObservationData,
    observed: SyntheticBoundaryObservationData,
    *,
    weights: ObservationFunctionalWeights,
    time_weights: Sequence[float] | None = None,
) -> float:
    j_ref = reflection_misfit(simulated, observed, time_weights=time_weights)
    j_trans = transmission_misfit(simulated, observed, time_weights=time_weights)

    return float(weights.reflection * j_ref + weights.transmission * j_trans)


def compute_observation_functionals(
    simulated: SyntheticBoundaryObservationData,
    observed: SyntheticBoundaryObservationData,
    *,
    weights: ObservationFunctionalWeights | None = None,
    time_weights: Sequence[float] | None = None,
) -> ObservationFunctionalValues:
    if weights is None:
        weights = ObservationFunctionalWeights()

    j_ref = reflection_misfit(simulated, observed, time_weights=time_weights)
    j_trans = transmission_misfit(simulated, observed, time_weights=time_weights)
    j_full = float(j_ref + j_trans)
    j_weighted = float(weights.reflection * j_ref + weights.transmission * j_trans)

    return ObservationFunctionalValues(
        reflection=j_ref,
        transmission=j_trans,
        full=j_full,
        weighted=j_weighted,
        reflection_weight=weights.reflection,
        transmission_weight=weights.transmission,
    )
