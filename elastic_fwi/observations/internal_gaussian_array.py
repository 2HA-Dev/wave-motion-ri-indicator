from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from elastic_fwi.observations.time_series import displacement_vector_to_function


_ALLOWED_COMPONENTS = {"vector", "x", "y"}


@dataclass(frozen=True)
class InternalGaussianArrayObservationSpec:
    name: str
    component: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    num_elements: int
    sigma: float
    tag: int = 2

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "kind": "internal_gaussian_array",
            "name": self.name,
            "component": self.component,
            "start_x": float(self.start_x),
            "start_y": float(self.start_y),
            "end_x": float(self.end_x),
            "end_y": float(self.end_y),
            "num_elements": int(self.num_elements),
            "sigma": float(self.sigma),
            "tag": int(self.tag),
        }


@dataclass(frozen=True)
class InternalGaussianArrayObservationTimeSeriesRecord:
    step: int
    time: float
    observation_name: str
    tag: int
    component: str
    norm_squared: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "step": int(self.step),
            "time": float(self.time),
            "observation_name": self.observation_name,
            "tag": int(self.tag),
            "component": self.component,
            "norm_squared": float(self.norm_squared),
        }


def _validate_internal_gaussian_array_spec(spec: InternalGaussianArrayObservationSpec) -> None:
    if spec.component not in _ALLOWED_COMPONENTS:
        raise ValueError(f"component must be one of {sorted(_ALLOWED_COMPONENTS)}.")

    for name in ("start_x", "start_y", "end_x", "end_y", "sigma"):
        value = float(getattr(spec, name))
        if not isfinite(value):
            raise ValueError(f"{name} must be finite.")

    if int(spec.num_elements) <= 0:
        raise ValueError("num_elements must be strictly positive.")

    if float(spec.sigma) <= 0.0:
        raise ValueError("sigma must be strictly positive.")


def internal_gaussian_array_observation(
    *,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    num_elements: int,
    sigma: float,
    component: str = "vector",
    name: str = "gamma2_internal_gaussian_array_transmission",
    tag: int = 2,
) -> InternalGaussianArrayObservationSpec:
    spec = InternalGaussianArrayObservationSpec(
        name=str(name),
        component=str(component),
        start_x=float(start_x),
        start_y=float(start_y),
        end_x=float(end_x),
        end_y=float(end_y),
        num_elements=int(num_elements),
        sigma=float(sigma),
        tag=int(tag),
    )
    _validate_internal_gaussian_array_spec(spec)
    return spec


def _component_inner(u, component: str):
    if component == "vector":
        return ufl.inner(u, u)
    if component == "x":
        return u[0] * u[0]
    if component == "y":
        return u[1] * u[1]
    raise ValueError(f"Unsupported observation component: {component!r}")


def gaussian_array_weight(spec: InternalGaussianArrayObservationSpec, mesh):
    x = ufl.SpatialCoordinate(mesh)

    if spec.num_elements == 1:
        centers = np.array([[spec.start_x, spec.start_y]], dtype=float)
    else:
        xs = np.linspace(spec.start_x, spec.end_x, spec.num_elements)
        ys = np.linspace(spec.start_y, spec.end_y, spec.num_elements)
        centers = np.column_stack([xs, ys])

    sigma2 = float(spec.sigma) ** 2

    weight = 0.0
    for cx, cy in centers:
        r2 = (x[0] - float(cx)) ** 2 + (x[1] - float(cy)) ** 2
        weight = weight + ufl.exp(-0.5 * r2 / sigma2)

    return weight


def internal_gaussian_array_l2_norm_squared(
    u_h: fem.Function,
    spec: InternalGaussianArrayObservationSpec,
) -> float:
    _validate_internal_gaussian_array_spec(spec)

    w = gaussian_array_weight(spec, u_h.function_space.mesh)
    dx = ufl.Measure("dx", domain=u_h.function_space.mesh)

    form = fem.form(w * _component_inner(u_h, spec.component) * dx)
    local_value = fem.assemble_scalar(form)

    comm = u_h.function_space.mesh.comm
    global_value = comm.allreduce(float(local_value), op=MPI.SUM)

    return float(global_value)


def evaluate_internal_gaussian_array_observation_from_vector(
    V,
    vector: PETSc.Vec,
    spec: InternalGaussianArrayObservationSpec,
    *,
    name: str = "u_snapshot",
) -> float:
    function = displacement_vector_to_function(V, vector, name=name)
    return internal_gaussian_array_l2_norm_squared(function, spec)


def evaluate_internal_gaussian_array_observation_time_series(
    V,
    snapshots: list[tuple[int, float, PETSc.Vec]],
    spec: InternalGaussianArrayObservationSpec,
) -> list[InternalGaussianArrayObservationTimeSeriesRecord]:
    records: list[InternalGaussianArrayObservationTimeSeriesRecord] = []

    for step, time, vector in snapshots:
        norm_squared = evaluate_internal_gaussian_array_observation_from_vector(
            V,
            vector,
            spec,
            name=f"{spec.name}_step_{step}",
        )

        records.append(
            InternalGaussianArrayObservationTimeSeriesRecord(
                step=int(step),
                time=float(time),
                observation_name=spec.name,
                tag=int(spec.tag),
                component=spec.component,
                norm_squared=float(norm_squared),
            )
        )

    return records
