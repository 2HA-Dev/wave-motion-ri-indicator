from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BoundaryObservationSeries:
    """
    Scalar boundary observation time series.

    In the minimal direct chain, values are typically

        ||H_Gamma u_h(t^n)||^2.

    The class is intentionally NumPy-only and independent of FEniCSx/PETSc.
    """

    name: str
    tag: int
    component: str
    times: Sequence[float]
    values: Sequence[float]
    value_kind: str = "norm_squared"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Observation series name must be non-empty.")

        if self.tag <= 0:
            raise ValueError("Observation boundary tag must be strictly positive.")

        if not self.component:
            raise ValueError("Observation component must be non-empty.")

        if not self.value_kind:
            raise ValueError("Observation value_kind must be non-empty.")

        times = np.asarray(self.times, dtype=float)
        values = np.asarray(self.values, dtype=float)

        if times.ndim != 1:
            raise ValueError("Observation times must be a one-dimensional array.")

        if values.ndim != 1:
            raise ValueError("Observation values must be a one-dimensional array.")

        if times.size == 0:
            raise ValueError("Observation series must contain at least one time.")

        if times.size != values.size:
            raise ValueError("Observation times and values must have the same length.")

        if not np.all(np.isfinite(times)):
            raise ValueError("Observation times must be finite.")

        if not np.all(np.isfinite(values)):
            raise ValueError("Observation values must be finite.")

        if times.size > 1 and not np.all(np.diff(times) > 0.0):
            raise ValueError("Observation times must be strictly increasing.")

        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "values", values.copy())

    @property
    def num_steps(self) -> int:
        return int(self.times.size)

    @property
    def final_time(self) -> float:
        return float(self.times[-1])

    @property
    def max_value(self) -> float:
        return float(np.max(self.values))

    @property
    def final_value(self) -> float:
        return float(self.values[-1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tag": int(self.tag),
            "component": self.component,
            "value_kind": self.value_kind,
            "num_steps": self.num_steps,
            "final_time": self.final_time,
            "max_value": self.max_value,
            "final_value": self.final_value,
            "times": self.times.tolist(),
            "values": self.values.tolist(),
        }


@dataclass(frozen=True)
class SyntheticBoundaryObservationData:
    """
    Synthetic reflection/transmission observation data.

    reflection    -> Gamma1
    transmission  -> Gamma2
    """

    reflection: BoundaryObservationSeries
    transmission: BoundaryObservationSeries
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reflection.num_steps != self.transmission.num_steps:
            raise ValueError("Reflection and transmission series must have the same length.")

        if not np.allclose(
            self.reflection.times,
            self.transmission.times,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("Reflection and transmission series must share identical times.")

        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def times(self) -> np.ndarray:
        return self.reflection.times.copy()

    @property
    def num_steps(self) -> int:
        return self.reflection.num_steps

    @property
    def final_time(self) -> float:
        return self.reflection.final_time

    def stacked_values(self) -> np.ndarray:
        """
        Return values stacked as shape (2, num_steps):

            row 0 = reflection
            row 1 = transmission
        """

        return np.vstack(
            (
                self.reflection.values,
                self.transmission.values,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "synthetic_boundary_observation_data",
            "num_steps": self.num_steps,
            "final_time": self.final_time,
            "metadata": dict(self.metadata),
            "reflection": self.reflection.as_dict(),
            "transmission": self.transmission.as_dict(),
        }


def _record_get(record: Any, key: str) -> Any:
    if isinstance(record, Mapping):
        return record[key]

    return getattr(record, key)


def boundary_observation_series_from_records(
    records: Sequence[Any],
    *,
    name: str | None = None,
    tag: int | None = None,
    component: str | None = None,
    value_kind: str = "norm_squared",
) -> BoundaryObservationSeries:
    if len(records) == 0:
        raise ValueError("records must be non-empty.")

    first = records[0]

    series_name = name if name is not None else str(_record_get(first, "observation_name"))
    series_tag = tag if tag is not None else int(_record_get(first, "tag"))
    series_component = (
        component if component is not None else str(_record_get(first, "component"))
    )

    times = [float(_record_get(record, "time")) for record in records]
    values = [float(_record_get(record, value_kind)) for record in records]

    return BoundaryObservationSeries(
        name=series_name,
        tag=series_tag,
        component=series_component,
        times=times,
        values=values,
        value_kind=value_kind,
    )


def synthetic_data_from_boundary_records(
    *,
    reflection_records: Sequence[Any],
    transmission_records: Sequence[Any],
    metadata: Mapping[str, Any] | None = None,
) -> SyntheticBoundaryObservationData:
    reflection = boundary_observation_series_from_records(reflection_records)
    transmission = boundary_observation_series_from_records(transmission_records)

    return SyntheticBoundaryObservationData(
        reflection=reflection,
        transmission=transmission,
        metadata={} if metadata is None else dict(metadata),
    )


def _noise_scale(values: np.ndarray, noise_level: float, relative: bool) -> float:
    if relative:
        return float(noise_level * max(float(np.max(np.abs(values))), 1.0))

    return float(noise_level)


def add_gaussian_noise(
    data: SyntheticBoundaryObservationData,
    *,
    noise_level: float,
    seed: int | None = None,
    relative: bool = True,
    clip_min: float | None = 0.0,
) -> SyntheticBoundaryObservationData:
    """
    Add reproducible Gaussian noise to synthetic scalar observations.

    If relative=True, the standard deviation is

        noise_level * max(max(abs(values)), 1).

    clip_min=0 is used by default because the values are squared norms.
    """

    noise_level = float(noise_level)

    if not isfinite(noise_level) or noise_level < 0.0:
        raise ValueError("noise_level must be finite and non-negative.")

    if clip_min is not None:
        clip_min = float(clip_min)
        if not isfinite(clip_min):
            raise ValueError("clip_min must be finite when provided.")

    rng = np.random.default_rng(seed)

    def noisy_series(series: BoundaryObservationSeries) -> BoundaryObservationSeries:
        values = np.asarray(series.values, dtype=float)
        scale = _noise_scale(values, noise_level, relative)

        noisy_values = values + rng.normal(loc=0.0, scale=scale, size=values.shape)

        if clip_min is not None:
            noisy_values = np.maximum(noisy_values, clip_min)

        return BoundaryObservationSeries(
            name=series.name,
            tag=series.tag,
            component=series.component,
            times=series.times,
            values=noisy_values,
            value_kind=series.value_kind,
        )

    metadata = dict(data.metadata)
    metadata.update(
        {
            "noise_model": "gaussian",
            "noise_level": noise_level,
            "noise_seed": seed,
            "noise_relative": bool(relative),
            "noise_clip_min": clip_min,
        }
    )

    return SyntheticBoundaryObservationData(
        reflection=noisy_series(data.reflection),
        transmission=noisy_series(data.transmission),
        metadata=metadata,
    )
