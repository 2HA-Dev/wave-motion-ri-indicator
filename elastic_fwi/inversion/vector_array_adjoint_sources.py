from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc

from elastic_fwi.inversion.vector_array_residuals import VectorArrayResidualSeries
from elastic_fwi.observations.vector_array import (
    VectorArrayObservationOperator,
    VectorArrayObservationSpec,
    build_vector_array_observation_operator,
)


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
class VectorArrayAdjointRHSRecord:
    step: int
    time: float
    observation_name: str
    array_name: str
    role: str
    residual_max_abs: float
    residual_l2_norm: float
    rhs_norm: float
    rhs_size_global: int
    rhs_size_local: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "time": float(self.time),
            "observation_name": self.observation_name,
            "array_name": self.array_name,
            "role": self.role,
            "residual_max_abs": float(self.residual_max_abs),
            "residual_l2_norm": float(self.residual_l2_norm),
            "rhs_norm": float(self.rhs_norm),
            "rhs_size_global": int(self.rhs_size_global),
            "rhs_size_local": int(self.rhs_size_local),
        }


@dataclass(frozen=True)
class VectorArrayAdjointRHSResult:
    records: tuple[VectorArrayAdjointRHSRecord, ...]
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
            "kind": "vector_array_adjoint_rhs_result",
            "num_steps": self.num_steps,
            "max_rhs_norm": self.max_rhs_norm,
            "records": [record.as_dict() for record in self.records],
        }


def _validate_residual_values(values: np.ndarray, num_receivers: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if values.shape != (num_receivers, 2):
        raise ValueError(
            "residual_values must have shape (num_receivers, 2), "
            f"got {values.shape} for num_receivers={num_receivers}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("residual_values must be finite.")

    return values


def assemble_vector_array_adjoint_rhs_vector_from_operator(
    operator: VectorArrayObservationOperator,
    residual_values: np.ndarray,
    *,
    scale: float = 1.0,
):
    """Assemble H^T r for one vector-array residual snapshot.

    The residual values have shape (num_receivers, 2).

    The assembled form is

        int_Omega sum_j psi_j/Z_j * (r_jx v_x + r_jy v_y) dx.

    This is the adjoint source associated with the vector observation
    H_j(u) = int psi_j u dx / int psi_j dx.
    """
    V = operator.V
    scale = float(scale)

    if not np.isfinite(scale):
        raise ValueError("scale must be finite.")

    residual_values = _validate_residual_values(
        residual_values,
        operator.num_receivers,
    )

    if np.max(np.abs(scale * residual_values)) <= 1.0e-30:
        return _create_zero_rhs_vector(V)

    v = ufl.TestFunction(V)

    integrand = 0.0
    for j, weight in enumerate(operator.weights):
        inv_norm = 1.0 / float(operator.normalizations[j])
        rx = float(residual_values[j, 0])
        ry = float(residual_values[j, 1])
        integrand = integrand + inv_norm * weight * (rx * v[0] + ry * v[1])

    form = fem.form(scale * integrand * operator.dx)

    rhs = fem_petsc.assemble_vector(form)
    rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    return rhs


def assemble_vector_array_adjoint_rhs_vector(
    V,
    spec: VectorArrayObservationSpec,
    residual_values: np.ndarray,
    *,
    scale: float = 1.0,
):
    operator = build_vector_array_observation_operator(V, spec)
    return assemble_vector_array_adjoint_rhs_vector_from_operator(
        operator,
        residual_values,
        scale=scale,
    )


def assemble_vector_array_adjoint_rhs_series_from_operator(
    operator: VectorArrayObservationOperator,
    residual_series: VectorArrayResidualSeries,
    *,
    scale: float = 1.0,
) -> VectorArrayAdjointRHSResult:
    if residual_series.num_receivers != operator.num_receivers:
        raise ValueError(
            "residual_series and observation operator must have the same number of receivers."
        )

    if residual_series.num_components != 2:
        raise ValueError("residual_series must contain two vector components.")

    records: list[VectorArrayAdjointRHSRecord] = []
    vectors = []

    for index in range(residual_series.num_steps):
        values = residual_series.values[index, :, :]

        rhs = assemble_vector_array_adjoint_rhs_vector_from_operator(
            operator,
            values,
            scale=scale,
        )

        residual_max_abs = float(np.max(np.abs(values))) if values.size else 0.0
        residual_l2_norm = float(np.sqrt(np.sum(values * values)))

        records.append(
            VectorArrayAdjointRHSRecord(
                step=int(residual_series.steps[index]),
                time=float(residual_series.times[index]),
                observation_name=operator.spec.name,
                array_name=residual_series.array_name,
                role=residual_series.role,
                residual_max_abs=residual_max_abs,
                residual_l2_norm=residual_l2_norm,
                rhs_norm=float(rhs.norm()),
                rhs_size_global=int(rhs.getSize()),
                rhs_size_local=int(rhs.getLocalSize()),
            )
        )
        vectors.append(rhs)

    return VectorArrayAdjointRHSResult(
        records=tuple(records),
        rhs_vectors=tuple(vectors),
    )


def assemble_vector_array_adjoint_rhs_series(
    V,
    spec: VectorArrayObservationSpec,
    residual_series: VectorArrayResidualSeries,
    *,
    scale: float = 1.0,
) -> VectorArrayAdjointRHSResult:
    operator = build_vector_array_observation_operator(V, spec)
    return assemble_vector_array_adjoint_rhs_series_from_operator(
        operator,
        residual_series,
        scale=scale,
    )

@dataclass(frozen=True)
class VectorArrayFullAdjointRHSResult:
    """Reflection/transmission vector-array RHS pair.

    This object exposes the structural interface expected by
    solve_newmark_discrete_adjoint_state:

    - reflection.rhs_vectors
    - transmission.rhs_vectors
    - reflection.records
    - transmission.records
    - num_steps
    - max_rhs_norm

    It can therefore be passed directly to the Newmark adjoint solver.
    """

    reflection: VectorArrayAdjointRHSResult
    transmission: VectorArrayAdjointRHSResult

    def __post_init__(self) -> None:
        if self.reflection.num_steps != self.transmission.num_steps:
            raise ValueError(
                "Reflection and transmission vector-array RHS series must have "
                "the same number of steps."
            )

        for index, (rhs_ref, rhs_trans) in enumerate(
            zip(self.reflection.rhs_vectors, self.transmission.rhs_vectors),
            start=1,
        ):
            if rhs_ref.getSize() != rhs_trans.getSize():
                raise ValueError(
                    f"Incompatible reflection/transmission RHS vector sizes "
                    f"at step {index}."
                )

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
            "kind": "vector_array_full_adjoint_rhs_result",
            "num_steps": self.num_steps,
            "max_rhs_norm": self.max_rhs_norm,
            "reflection": self.reflection.as_dict(),
            "transmission": self.transmission.as_dict(),
        }


def assemble_full_vector_array_adjoint_rhs(
    V,
    *,
    reflection_spec: VectorArrayObservationSpec,
    reflection_residual: VectorArrayResidualSeries,
    transmission_spec: VectorArrayObservationSpec,
    transmission_residual: VectorArrayResidualSeries,
    reflection_scale: float = 1.0,
    transmission_scale: float = 1.0,
) -> VectorArrayFullAdjointRHSResult:
    """Assemble a full reflection/transmission vector-array RHS pair."""
    reflection_rhs = assemble_vector_array_adjoint_rhs_series(
        V,
        reflection_spec,
        reflection_residual,
        scale=reflection_scale,
    )

    transmission_rhs = assemble_vector_array_adjoint_rhs_series(
        V,
        transmission_spec,
        transmission_residual,
        scale=transmission_scale,
    )

    try:
        return VectorArrayFullAdjointRHSResult(
            reflection=reflection_rhs,
            transmission=transmission_rhs,
        )
    except Exception:
        reflection_rhs.destroy()
        transmission_rhs.destroy()
        raise

