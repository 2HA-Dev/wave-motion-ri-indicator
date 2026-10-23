from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from petsc4py import PETSc

from elastic_fwi.solvers.elastic_operator import ElasticOperatorConfig


@dataclass(frozen=True)
class MaterialDirectionalGradientRecord:
    step: int
    contribution: float
    forcing_norm: float
    adjoint_norm: float
    current_state_norm: float
    next_state_norm: float
    previous_state_norm: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "contribution": float(self.contribution),
            "forcing_norm": float(self.forcing_norm),
            "adjoint_norm": float(self.adjoint_norm),
            "current_state_norm": float(self.current_state_norm),
            "next_state_norm": float(self.next_state_norm),
            "previous_state_norm": float(self.previous_state_norm),
        }


@dataclass(frozen=True)
class MaterialDirectionalGradientResult:
    value: float
    records: tuple[MaterialDirectionalGradientRecord, ...]

    @property
    def num_steps(self) -> int:
        return len(self.records)

    @property
    def max_abs_contribution(self) -> float:
        if not self.records:
            return 0.0
        return float(max(abs(record.contribution) for record in self.records))

    @property
    def max_forcing_norm(self) -> float:
        if not self.records:
            return 0.0
        return float(max(record.forcing_norm for record in self.records))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "material_directional_gradient_result",
            "num_steps": self.num_steps,
            "value": float(self.value),
            "max_abs_contribution": self.max_abs_contribution,
            "max_forcing_norm": self.max_forcing_norm,
            "records": [record.as_dict() for record in self.records],
        }


def _zero_like(vector: PETSc.Vec) -> PETSc.Vec:
    result = vector.copy()
    result.set(0.0)
    result.assemble()
    return result


def _check_matrix_vector_size(matrix: PETSc.Mat, vector: PETSc.Vec, *, name: str) -> None:
    rows, cols = matrix.getSize()

    if cols != vector.getSize():
        raise ValueError(
            f"Incompatible size for {name}: matrix columns={cols}, "
            f"vector size={vector.getSize()}."
        )


def _check_same_vector_size(left: PETSc.Vec, right: PETSc.Vec, *, name: str) -> None:
    if left.getSize() != right.getSize():
        raise ValueError(
            f"Incompatible vector sizes for {name}: "
            f"{left.getSize()} != {right.getSize()}."
        )


def material_effective_operator_action(
    delta_mass: PETSc.Mat,
    delta_stiffness: PETSc.Mat,
    config: ElasticOperatorConfig,
    vector: PETSc.Vec,
) -> PETSc.Vec:
    """
    Compute δA x with

        δA = δM + θ dt² δK.
    """

    _check_matrix_vector_size(delta_mass, vector, name="delta_mass")
    _check_matrix_vector_size(delta_stiffness, vector, name="delta_stiffness")

    theta = float(config.theta)
    dt = float(config.dt)

    result = delta_mass.createVecLeft()
    tmp = delta_stiffness.createVecLeft()

    try:
        delta_mass.mult(vector, result)

        delta_stiffness.mult(vector, tmp)
        result.axpy(theta * dt * dt, tmp)

        return result

    finally:
        tmp.destroy()


def material_middle_operator_action(
    delta_mass: PETSc.Mat,
    delta_stiffness: PETSc.Mat,
    config: ElasticOperatorConfig,
    vector: PETSc.Vec,
) -> PETSc.Vec:
    """
    Compute δB x with

        δB = 2 δM - dt²(1 - 2θ) δK.
    """

    _check_matrix_vector_size(delta_mass, vector, name="delta_mass")
    _check_matrix_vector_size(delta_stiffness, vector, name="delta_stiffness")

    theta = float(config.theta)
    dt = float(config.dt)

    result = delta_mass.createVecLeft()
    tmp = delta_stiffness.createVecLeft()

    try:
        delta_mass.mult(vector, result)
        result.scale(2.0)

        delta_stiffness.mult(vector, tmp)
        result.axpy(-(dt * dt) * (1.0 - 2.0 * theta), tmp)

        return result

    finally:
        tmp.destroy()


def material_newmark_step_forcing(
    delta_mass: PETSc.Mat,
    delta_stiffness: PETSc.Mat,
    config: ElasticOperatorConfig,
    *,
    next_state: PETSc.Vec,
    current_state: PETSc.Vec,
    previous_state: PETSc.Vec,
) -> PETSc.Vec:
    """
    Compute the material perturbation forcing in the tangent equation:

        δB u^n - δA (u^{n+1} + u^{n-1}).
    """

    _check_same_vector_size(next_state, current_state, name="next/current")
    _check_same_vector_size(next_state, previous_state, name="next/previous")

    forcing = material_middle_operator_action(
        delta_mass,
        delta_stiffness,
        config,
        current_state,
    )

    sum_next_previous = next_state.copy()
    delta_a_action = None

    try:
        sum_next_previous.axpy(1.0, previous_state)

        delta_a_action = material_effective_operator_action(
            delta_mass,
            delta_stiffness,
            config,
            sum_next_previous,
        )

        forcing.axpy(-1.0, delta_a_action)

        return forcing

    finally:
        sum_next_previous.destroy()

        if delta_a_action is not None:
            delta_a_action.destroy()


def compute_newmark_material_directional_gradient(
    delta_mass: PETSc.Mat,
    delta_stiffness: PETSc.Mat,
    config: ElasticOperatorConfig,
    *,
    direct_snapshots: tuple[PETSc.Vec, ...] | list[PETSc.Vec],
    equation_adjoint_vectors: tuple[PETSc.Vec, ...] | list[PETSc.Vec],
) -> MaterialDirectionalGradientResult:
    """
    Compute the scalar directional material derivative:

        Σ_n < μ^{n+1},
              δB u^n - δA(u^{n+1}+u^{n-1}) >.

    The initial direct states u^0 and u^{-1} are assumed to be zero,
    consistently with the Newmark direct pipeline.
    """

    if len(direct_snapshots) != len(equation_adjoint_vectors):
        raise ValueError(
            "direct_snapshots and equation_adjoint_vectors must have same length."
        )

    num_steps = len(direct_snapshots)

    if num_steps <= 0:
        raise ValueError("At least one direct snapshot is required.")

    template = direct_snapshots[0]

    for index, (snapshot, adjoint) in enumerate(
        zip(direct_snapshots, equation_adjoint_vectors),
        start=1,
    ):
        _check_same_vector_size(template, snapshot, name=f"direct snapshot {index}")
        _check_same_vector_size(template, adjoint, name=f"adjoint vector {index}")

    zero_state = _zero_like(template)

    total = 0.0
    records: list[MaterialDirectionalGradientRecord] = []

    try:
        for index in range(num_steps):
            next_state = direct_snapshots[index]

            if index == 0:
                current_state = zero_state
                previous_state = zero_state
            elif index == 1:
                current_state = direct_snapshots[index - 1]
                previous_state = zero_state
            else:
                current_state = direct_snapshots[index - 1]
                previous_state = direct_snapshots[index - 2]

            forcing = material_newmark_step_forcing(
                delta_mass,
                delta_stiffness,
                config,
                next_state=next_state,
                current_state=current_state,
                previous_state=previous_state,
            )

            try:
                contribution = float(equation_adjoint_vectors[index].dot(forcing))
                total += contribution

                records.append(
                    MaterialDirectionalGradientRecord(
                        step=index + 1,
                        contribution=contribution,
                        forcing_norm=float(forcing.norm()),
                        adjoint_norm=float(equation_adjoint_vectors[index].norm()),
                        current_state_norm=float(current_state.norm()),
                        next_state_norm=float(next_state.norm()),
                        previous_state_norm=float(previous_state.norm()),
                    )
                )

            finally:
                forcing.destroy()

        if not np.isfinite(total):
            raise FloatingPointError("Non-finite material directional gradient.")

        return MaterialDirectionalGradientResult(
            value=float(total),
            records=tuple(records),
        )

    finally:
        zero_state.destroy()
