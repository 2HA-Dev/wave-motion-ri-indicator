from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from elastic_fwi.data.synthetic_observations import (
    BoundaryObservationSeries,
    SyntheticBoundaryObservationData,
)
from elastic_fwi.inversion.observation_functionals import (
    ObservationFunctionalValues,
    ObservationFunctionalWeights,
    compute_observation_functionals,
    validate_matching_observation_series,
)


@dataclass(frozen=True)
class ObservationResidualSeries:
    """
    Scalar residual time series between simulated and observed boundary data.

    values[n] = simulated[n] - observed[n]
    """

    name: str
    tag: int
    component: str
    times: np.ndarray
    values: np.ndarray
    value_kind: str = "residual"

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        values = np.asarray(self.values, dtype=float)

        if not self.name:
            raise ValueError("Residual series name must be non-empty.")

        if self.tag <= 0:
            raise ValueError("Residual boundary tag must be strictly positive.")

        if not self.component:
            raise ValueError("Residual component must be non-empty.")

        if times.ndim != 1:
            raise ValueError("Residual times must be one-dimensional.")

        if values.ndim != 1:
            raise ValueError("Residual values must be one-dimensional.")

        if times.size == 0:
            raise ValueError("Residual series must be non-empty.")

        if times.size != values.size:
            raise ValueError("Residual times and values must have the same length.")

        if not np.all(np.isfinite(times)):
            raise ValueError("Residual times must be finite.")

        if not np.all(np.isfinite(values)):
            raise ValueError("Residual values must be finite.")

        if times.size > 1 and not np.all(np.diff(times) > 0.0):
            raise ValueError("Residual times must be strictly increasing.")

        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "values", values.copy())

    @property
    def num_steps(self) -> int:
        return int(self.times.size)

    @property
    def final_time(self) -> float:
        return float(self.times[-1])

    @property
    def max_abs_value(self) -> float:
        return float(np.max(np.abs(self.values)))

    @property
    def l2_discrete_norm_squared(self) -> float:
        return float(np.sum(self.values**2))

    @property
    def half_l2_discrete_norm_squared(self) -> float:
        return float(0.5 * self.l2_discrete_norm_squared)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tag": int(self.tag),
            "component": self.component,
            "value_kind": self.value_kind,
            "num_steps": self.num_steps,
            "final_time": self.final_time,
            "max_abs_value": self.max_abs_value,
            "l2_discrete_norm_squared": self.l2_discrete_norm_squared,
            "half_l2_discrete_norm_squared": self.half_l2_discrete_norm_squared,
            "times": self.times.tolist(),
            "values": self.values.tolist(),
        }


@dataclass(frozen=True)
class ObservationResidualData:
    reflection: ObservationResidualSeries
    transmission: ObservationResidualSeries
    functionals: ObservationFunctionalValues

    def __post_init__(self) -> None:
        if self.reflection.num_steps != self.transmission.num_steps:
            raise ValueError("Reflection and transmission residual lengths must match.")

        if not np.allclose(
            self.reflection.times,
            self.transmission.times,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("Reflection and transmission residual times must match.")

    @property
    def times(self) -> np.ndarray:
        return self.reflection.times.copy()

    @property
    def num_steps(self) -> int:
        return self.reflection.num_steps

    @property
    def final_time(self) -> float:
        return self.reflection.final_time

    @property
    def max_abs_residual(self) -> float:
        return float(
            max(
                self.reflection.max_abs_value,
                self.transmission.max_abs_value,
            )
        )

    def stacked_values(self) -> np.ndarray:
        return np.vstack(
            (
                self.reflection.values,
                self.transmission.values,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "observation_residual_data",
            "num_steps": self.num_steps,
            "final_time": self.final_time,
            "max_abs_residual": self.max_abs_residual,
            "reflection": self.reflection.as_dict(),
            "transmission": self.transmission.as_dict(),
            "functionals": self.functionals.as_dict(),
        }


def compute_observation_residual_series(
    simulated: BoundaryObservationSeries,
    observed: BoundaryObservationSeries,
) -> ObservationResidualSeries:
    validate_matching_observation_series(simulated, observed)

    residual = np.asarray(simulated.values, dtype=float) - np.asarray(
        observed.values,
        dtype=float,
    )

    return ObservationResidualSeries(
        name=simulated.name,
        tag=simulated.tag,
        component=simulated.component,
        times=simulated.times,
        values=residual,
    )


def compute_observation_residual_data(
    simulated: SyntheticBoundaryObservationData,
    observed: SyntheticBoundaryObservationData,
    *,
    weights: ObservationFunctionalWeights | None = None,
) -> ObservationResidualData:
    if weights is None:
        weights = ObservationFunctionalWeights()

    reflection = compute_observation_residual_series(
        simulated.reflection,
        observed.reflection,
    )

    transmission = compute_observation_residual_series(
        simulated.transmission,
        observed.transmission,
    )

    functionals = compute_observation_functionals(
        simulated,
        observed,
        weights=weights,
    )

    return ObservationResidualData(
        reflection=reflection,
        transmission=transmission,
        functionals=functionals,
    )
