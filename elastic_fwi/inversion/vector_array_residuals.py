from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from elastic_fwi.observations.vector_array import VectorArrayObservationTimeSeries


@dataclass(frozen=True)
class VectorArrayResidualSeries:
    """Vector residual time series for one transducer array.

    values[k, j, c] stores the residual at time index k, receiver j,
    component c:

        values = scale * (reference_values - observed_values)

    with c = 0 for x and c = 1 for y.
    """

    name: str
    array_name: str
    role: str
    times: np.ndarray
    steps: np.ndarray
    values: np.ndarray
    reference_values: np.ndarray
    observed_values: np.ndarray
    scale: float = 1.0

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        steps = np.asarray(self.steps, dtype=np.int64)
        values = np.asarray(self.values, dtype=float)
        reference_values = np.asarray(self.reference_values, dtype=float)
        observed_values = np.asarray(self.observed_values, dtype=float)

        if values.ndim != 3:
            raise ValueError("values must have shape (num_steps, num_receivers, 2).")

        if values.shape[-1] != 2:
            raise ValueError("The last dimension of values must contain ux, uy.")

        if reference_values.shape != values.shape:
            raise ValueError("reference_values must have the same shape as values.")

        if observed_values.shape != values.shape:
            raise ValueError("observed_values must have the same shape as values.")

        if times.shape != (values.shape[0],):
            raise ValueError("times must have shape (num_steps,).")

        if steps.shape != (values.shape[0],):
            raise ValueError("steps must have shape (num_steps,).")

        if not np.all(np.isfinite(times)):
            raise ValueError("times must be finite.")

        if not np.all(np.isfinite(values)):
            raise ValueError("values must be finite.")

        if not np.all(np.isfinite(reference_values)):
            raise ValueError("reference_values must be finite.")

        if not np.all(np.isfinite(observed_values)):
            raise ValueError("observed_values must be finite.")

        if not np.isfinite(float(self.scale)):
            raise ValueError("scale must be finite.")

        object.__setattr__(self, "times", times)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "reference_values", reference_values)
        object.__setattr__(self, "observed_values", observed_values)
        object.__setattr__(self, "scale", float(self.scale))

    @property
    def num_steps(self) -> int:
        return int(self.values.shape[0])

    @property
    def num_receivers(self) -> int:
        return int(self.values.shape[1])

    @property
    def num_components(self) -> int:
        return int(self.values.shape[2])

    @property
    def max_abs_residual(self) -> float:
        if self.values.size == 0:
            return 0.0
        return float(np.max(np.abs(self.values)))

    @property
    def l2_norm_squared(self) -> float:
        return float(np.sum(self.values * self.values))

    @property
    def l2_norm(self) -> float:
        return float(np.sqrt(self.l2_norm_squared))

    def reversed_time(self, *, name: str | None = None) -> "VectorArrayResidualSeries":
        """Return the same residual series in reverse time order."""
        new_name = name if name is not None else f"{self.name}_reversed_time"
        return VectorArrayResidualSeries(
            name=new_name,
            array_name=self.array_name,
            role=self.role,
            times=self.times[::-1].copy(),
            steps=self.steps[::-1].copy(),
            values=self.values[::-1, :, :].copy(),
            reference_values=self.reference_values[::-1, :, :].copy(),
            observed_values=self.observed_values[::-1, :, :].copy(),
            scale=self.scale,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "vector_array_residual_series",
            "name": self.name,
            "array_name": self.array_name,
            "role": self.role,
            "num_steps": self.num_steps,
            "num_receivers": self.num_receivers,
            "num_components": self.num_components,
            "shape": list(self.values.shape),
            "scale": float(self.scale),
            "max_abs_residual": self.max_abs_residual,
            "l2_norm": self.l2_norm,
            "l2_norm_squared": self.l2_norm_squared,
            "times": self.times.tolist(),
            "steps": self.steps.tolist(),
        }


@dataclass(frozen=True)
class VectorArrayResidualPair:
    """Reflection/transmission vector residual pair."""

    reflection: VectorArrayResidualSeries | None
    transmission: VectorArrayResidualSeries | None

    @property
    def max_abs_residual(self) -> float:
        values = []
        if self.reflection is not None:
            values.append(self.reflection.max_abs_residual)
        if self.transmission is not None:
            values.append(self.transmission.max_abs_residual)
        if not values:
            return 0.0
        return float(max(values))

    def reversed_time(self) -> "VectorArrayResidualPair":
        return VectorArrayResidualPair(
            reflection=None
            if self.reflection is None
            else self.reflection.reversed_time(),
            transmission=None
            if self.transmission is None
            else self.transmission.reversed_time(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "vector_array_residual_pair",
            "max_abs_residual": self.max_abs_residual,
            "reflection": None
            if self.reflection is None
            else self.reflection.as_dict(),
            "transmission": None
            if self.transmission is None
            else self.transmission.as_dict(),
        }


def _assert_compatible_time_series(
    reference: VectorArrayObservationTimeSeries,
    observed: VectorArrayObservationTimeSeries,
) -> None:
    if reference.values.shape != observed.values.shape:
        raise ValueError(
            "reference and observed time series must have the same value shape."
        )

    if not np.array_equal(reference.steps, observed.steps):
        raise ValueError("reference and observed time series must have the same steps.")

    if not np.allclose(reference.times, observed.times, rtol=0.0, atol=1.0e-14):
        raise ValueError("reference and observed time series must have the same times.")

    if reference.spec.array.name != observed.spec.array.name:
        raise ValueError("reference and observed arrays must have the same name.")

    if reference.spec.role != observed.spec.role:
        raise ValueError("reference and observed arrays must have the same role.")

    ref_centers = np.asarray(reference.spec.centers, dtype=float)
    obs_centers = np.asarray(observed.spec.centers, dtype=float)
    if ref_centers.shape != obs_centers.shape:
        raise ValueError("reference and observed arrays must have the same centers shape.")

    if not np.allclose(ref_centers, obs_centers, rtol=0.0, atol=1.0e-14):
        raise ValueError("reference and observed arrays must have the same centers.")


def compute_vector_array_residual_series(
    *,
    reference: VectorArrayObservationTimeSeries,
    observed: VectorArrayObservationTimeSeries,
    name: str | None = None,
    scale: float = 1.0,
) -> VectorArrayResidualSeries:
    """Compute scale * (reference - observed) for one vector array."""
    _assert_compatible_time_series(reference, observed)

    scale = float(scale)
    if not np.isfinite(scale):
        raise ValueError("scale must be finite.")

    values = scale * (
        np.asarray(reference.values, dtype=float)
        - np.asarray(observed.values, dtype=float)
    )

    residual_name = name
    if residual_name is None:
        residual_name = f"{reference.spec.array.name}_vector_array_residual"

    return VectorArrayResidualSeries(
        name=residual_name,
        array_name=reference.spec.array.name,
        role=reference.spec.role,
        times=reference.times.copy(),
        steps=reference.steps.copy(),
        values=values,
        reference_values=np.asarray(reference.values, dtype=float).copy(),
        observed_values=np.asarray(observed.values, dtype=float).copy(),
        scale=scale,
    )


def reverse_vector_array_residual_series(
    residual: VectorArrayResidualSeries,
    *,
    name: str | None = None,
) -> VectorArrayResidualSeries:
    return residual.reversed_time(name=name)


def build_vector_array_residual_pair(
    *,
    reflection: VectorArrayResidualSeries | None = None,
    transmission: VectorArrayResidualSeries | None = None,
) -> VectorArrayResidualPair:
    if reflection is None and transmission is None:
        raise ValueError("At least one residual series must be provided.")
    return VectorArrayResidualPair(
        reflection=reflection,
        transmission=transmission,
    )
