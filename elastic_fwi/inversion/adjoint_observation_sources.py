from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc

from elastic_fwi.inversion.observation_residual_series import (
    ObservationResidualData,
    ObservationResidualSeries,
)
from elastic_fwi.observations.time_series import displacement_vector_to_function
from elastic_fwi.observations.internal_gaussian_array import (
    InternalGaussianArrayObservationSpec,
    gaussian_array_weight,
)


_ALLOWED_COMPONENTS = {"vector", "x", "y"}


def _create_zero_rhs_vector(V):
    index_map = V.dofmap.index_map
    block_size = V.dofmap.index_map_bs

    local_size = index_map.size_local * block_size
    global_size = index_map.size_global * block_size

    vector = PETSc.Vec().createMPI(
        (local_size, global_size),
        comm=V.mesh.comm,
    )
    vector.set(0.0)
    vector.assemble()
    return vector


@dataclass(frozen=True)
class AdjointObservationRHSRecord:
    step: int
    time: float
    tag: int
    component: str
    residual_value: float
    rhs_norm: float
    rhs_size_global: int
    rhs_size_local: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "time": float(self.time),
            "tag": int(self.tag),
            "component": self.component,
            "residual_value": float(self.residual_value),
            "rhs_norm": float(self.rhs_norm),
            "rhs_size_global": int(self.rhs_size_global),
            "rhs_size_local": int(self.rhs_size_local),
        }


@dataclass(frozen=True)
class AdjointObservationRHSResult:
    records: tuple[AdjointObservationRHSRecord, ...]
    rhs_vectors: tuple[Any, ...]

    def destroy(self) -> None:
        for vector in self.rhs_vectors:
            vector.destroy()

    @property
    def num_steps(self) -> int:
        return len(self.records)

    @property
    def max_rhs_norm(self) -> float:
        if not self.records:
            return 0.0
        return float(max(record.rhs_norm for record in self.records))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "adjoint_observation_rhs_result",
            "num_steps": self.num_steps,
            "max_rhs_norm": self.max_rhs_norm,
            "records": [record.as_dict() for record in self.records],
        }


def _component_inner(u, v, component: str):
    if component == "vector":
        return ufl.inner(u, v)

    if component == "x":
        return u[0] * v[0]

    if component == "y":
        return u[1] * v[1]

    raise ValueError(f"Unsupported observation component: {component!r}")


def assemble_adjoint_observation_rhs_vector(
    V,
    facet_tags,
    displacement_vector,
    *,
    residual_value: float,
    tag: int,
    component: str = "vector",
    scale: float = 1.0,
):
    residual_value = float(residual_value)
    scale = float(scale)
    tag = int(tag)

    if tag <= 0:
        raise ValueError("Boundary tag must be strictly positive.")

    if component not in _ALLOWED_COMPONENTS:
        raise ValueError(f"component must be one of {sorted(_ALLOWED_COMPONENTS)}.")

    if not np.isfinite(residual_value):
        raise ValueError("residual_value must be finite.")

    if not np.isfinite(scale):
        raise ValueError("scale must be finite.")

    if abs(scale * residual_value) <= 1.0e-30:
        return _create_zero_rhs_vector(V)

    u_h = displacement_vector_to_function(
        V,
        displacement_vector,
        name="u_observation_state",
    )

    v = ufl.TestFunction(V)
    ds = ufl.Measure(
        "ds",
        domain=V.mesh,
        subdomain_data=facet_tags,
    )

    form = fem.form(
        2.0
        * scale
        * residual_value
        * _component_inner(u_h, v, component)
        * ds(tag)
    )

    try:
        rhs = fem_petsc.assemble_vector(form)
    except IndexError:
        rhs = _create_zero_rhs_vector(V)

    rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    return rhs



def assemble_internal_gaussian_array_adjoint_rhs_vector(
    V,
    displacement_vector,
    *,
    residual_value: float,
    spec: InternalGaussianArrayObservationSpec,
    scale: float = 1.0,
):
    residual_value = float(residual_value)
    scale = float(scale)

    if spec.component not in _ALLOWED_COMPONENTS:
        raise ValueError(f"component must be one of {sorted(_ALLOWED_COMPONENTS)}.")

    if not np.isfinite(residual_value):
        raise ValueError("residual_value must be finite.")

    if not np.isfinite(scale):
        raise ValueError("scale must be finite.")

    if abs(scale * residual_value) <= 1.0e-30:
        return _create_zero_rhs_vector(V)

    u_h = displacement_vector_to_function(
        V,
        displacement_vector,
        name="u_internal_observation_state",
    )

    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=V.mesh)
    w = gaussian_array_weight(spec, V.mesh)

    form = fem.form(
        2.0
        * scale
        * residual_value
        * w
        * _component_inner(u_h, v, spec.component)
        * dx
    )

    rhs = fem_petsc.assemble_vector(form)
    rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    return rhs


def assemble_internal_gaussian_array_adjoint_rhs_series(
    V,
    displacement_snapshots,
    residual_series: ObservationResidualSeries,
    *,
    spec: InternalGaussianArrayObservationSpec,
    scale: float = 1.0,
) -> AdjointObservationRHSResult:
    if len(displacement_snapshots) != residual_series.num_steps:
        raise ValueError(
            "Number of displacement snapshots must match residual series length."
        )

    records: list[AdjointObservationRHSRecord] = []
    vectors = []

    for index, displacement_vector in enumerate(displacement_snapshots):
        residual_value = float(residual_series.values[index])
        time = float(residual_series.times[index])
        step = index + 1

        rhs = assemble_internal_gaussian_array_adjoint_rhs_vector(
            V,
            displacement_vector,
            residual_value=residual_value,
            spec=spec,
            scale=scale,
        )

        records.append(
            AdjointObservationRHSRecord(
                step=step,
                time=time,
                tag=int(spec.tag),
                component=spec.component,
                residual_value=residual_value,
                rhs_norm=float(rhs.norm()),
                rhs_size_global=int(rhs.getSize()),
                rhs_size_local=int(rhs.getLocalSize()),
            )
        )
        vectors.append(rhs)

    return AdjointObservationRHSResult(
        records=tuple(records),
        rhs_vectors=tuple(vectors),
    )


def assemble_adjoint_observation_rhs_series(
    V,
    facet_tags,
    displacement_snapshots,
    residual_series: ObservationResidualSeries,
    *,
    scale: float = 1.0,
) -> AdjointObservationRHSResult:
    if len(displacement_snapshots) != residual_series.num_steps:
        raise ValueError(
            "Number of displacement snapshots must match residual series length."
        )

    records: list[AdjointObservationRHSRecord] = []
    vectors = []

    for index, displacement_vector in enumerate(displacement_snapshots):
        residual_value = float(residual_series.values[index])
        time = float(residual_series.times[index])
        step = index + 1

        rhs = assemble_adjoint_observation_rhs_vector(
            V,
            facet_tags,
            displacement_vector,
            residual_value=residual_value,
            tag=residual_series.tag,
            component=residual_series.component,
            scale=scale,
        )

        records.append(
            AdjointObservationRHSRecord(
                step=step,
                time=time,
                tag=residual_series.tag,
                component=residual_series.component,
                residual_value=residual_value,
                rhs_norm=float(rhs.norm()),
                rhs_size_global=int(rhs.getSize()),
                rhs_size_local=int(rhs.getLocalSize()),
            )
        )
        vectors.append(rhs)

    return AdjointObservationRHSResult(
        records=tuple(records),
        rhs_vectors=tuple(vectors),
    )


@dataclass(frozen=True)
class FullAdjointObservationRHSResult:
    reflection: AdjointObservationRHSResult
    transmission: AdjointObservationRHSResult

    def destroy(self) -> None:
        self.reflection.destroy()
        self.transmission.destroy()

    @property
    def num_steps(self) -> int:
        return self.reflection.num_steps

    @property
    def max_rhs_norm(self) -> float:
        return float(max(self.reflection.max_rhs_norm, self.transmission.max_rhs_norm))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "full_adjoint_observation_rhs_result",
            "num_steps": self.num_steps,
            "max_rhs_norm": self.max_rhs_norm,
            "reflection": self.reflection.as_dict(),
            "transmission": self.transmission.as_dict(),
        }


def assemble_full_adjoint_observation_rhs(
    V,
    facet_tags,
    displacement_snapshots,
    residuals: ObservationResidualData,
    *,
    reflection_scale: float = 1.0,
    transmission_scale: float = 1.0,
    internal_transmission_spec: InternalGaussianArrayObservationSpec | None = None,
) -> FullAdjointObservationRHSResult:
    reflection = assemble_adjoint_observation_rhs_series(
        V,
        facet_tags,
        displacement_snapshots,
        residuals.reflection,
        scale=reflection_scale,
    )

    if internal_transmission_spec is None:
        transmission = assemble_adjoint_observation_rhs_series(
            V,
            facet_tags,
            displacement_snapshots,
            residuals.transmission,
            scale=transmission_scale,
        )
    else:
        transmission = assemble_internal_gaussian_array_adjoint_rhs_series(
            V,
            displacement_snapshots,
            residuals.transmission,
            spec=internal_transmission_spec,
            scale=transmission_scale,
        )

    return FullAdjointObservationRHSResult(
        reflection=reflection,
        transmission=transmission,
    )
