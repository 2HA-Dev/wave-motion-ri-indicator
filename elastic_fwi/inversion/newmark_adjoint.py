from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from petsc4py import PETSc

from elastic_fwi.inversion.adjoint_observation_sources import (
    FullAdjointObservationRHSResult,
)
from elastic_fwi.solvers.elastic_operator import ElasticOperatorMatrices
from elastic_fwi.solvers.newmark import (
    LinearSolveConfig,
    ReusableEffectiveLinearSolver,
    linear_residual_norm,
    solve_effective_system,
)


@dataclass(frozen=True)
class NewmarkAdjointStateStepRecord:
    step: int
    time: float
    solve_index: int
    observation_rhs_norm: float
    accumulated_rhs_norm: float
    equation_adjoint_norm: float
    carry_current_norm: float
    carry_previous_norm: float
    linear_residual_norm: float
    linear_residual_relative: float
    ksp_converged: bool
    ksp_reason: int
    ksp_iterations: int
    ksp_reported_residual_norm: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "time": float(self.time),
            "solve_index": int(self.solve_index),
            "observation_rhs_norm": float(self.observation_rhs_norm),
            "accumulated_rhs_norm": float(self.accumulated_rhs_norm),
            "equation_adjoint_norm": float(self.equation_adjoint_norm),
            "carry_current_norm": float(self.carry_current_norm),
            "carry_previous_norm": float(self.carry_previous_norm),
            "linear_residual_norm": float(self.linear_residual_norm),
            "linear_residual_relative": float(self.linear_residual_relative),
            "ksp_converged": bool(self.ksp_converged),
            "ksp_reason": int(self.ksp_reason),
            "ksp_iterations": int(self.ksp_iterations),
            "ksp_reported_residual_norm": float(self.ksp_reported_residual_norm),
        }


@dataclass(frozen=True)
class NewmarkAdjointStateResult:
    records: tuple[NewmarkAdjointStateStepRecord, ...]
    equation_adjoint_vectors: tuple[PETSc.Vec, ...]
    initial_current_adjoint: PETSc.Vec
    initial_previous_adjoint: PETSc.Vec
    reverse_time: bool = True

    def destroy(self) -> None:
        for vector in self.equation_adjoint_vectors:
            vector.destroy()

        self.initial_current_adjoint.destroy()
        self.initial_previous_adjoint.destroy()

    @property
    def num_steps(self) -> int:
        return len(self.records)

    @property
    def max_equation_adjoint_norm(self) -> float:
        if not self.records:
            return 0.0
        return float(max(record.equation_adjoint_norm for record in self.records))

    @property
    def max_linear_residual_norm(self) -> float:
        if not self.records:
            return 0.0
        return float(max(record.linear_residual_norm for record in self.records))

    @property
    def all_ksp_converged(self) -> bool:
        return bool(all(record.ksp_converged for record in self.records))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "newmark_adjoint_state_result",
            "num_steps": self.num_steps,
            "reverse_time": bool(self.reverse_time),
            "max_equation_adjoint_norm": self.max_equation_adjoint_norm,
            "max_linear_residual_norm": self.max_linear_residual_norm,
            "all_ksp_converged": self.all_ksp_converged,
            "initial_current_adjoint_norm": float(self.initial_current_adjoint.norm()),
            "initial_previous_adjoint_norm": float(self.initial_previous_adjoint.norm()),
            "records": [record.as_dict() for record in self.records],
        }


def _zero_like(vector: PETSc.Vec) -> PETSc.Vec:
    result = vector.copy()
    result.set(0.0)
    result.assemble()
    return result


def _combine_observation_rhs(
    rhs: FullAdjointObservationRHSResult,
    index: int,
) -> PETSc.Vec:
    combined = rhs.reflection.rhs_vectors[index].copy()
    combined.axpy(1.0, rhs.transmission.rhs_vectors[index])
    return combined


def newmark_middle_operator_action(
    matrices: ElasticOperatorMatrices,
    vector: PETSc.Vec,
) -> PETSc.Vec:
    """
    Compute B x with

        B = 2 M - dt^2 (1 - 2 theta) K.
    """

    theta = float(matrices.config.theta)
    dt = float(matrices.config.dt)

    result = matrices.mass.createVecLeft()
    tmp = matrices.stiffness.createVecLeft()

    try:
        matrices.mass.mult(vector, result)
        result.scale(2.0)

        matrices.stiffness.mult(vector, tmp)
        result.axpy(-(dt * dt) * (1.0 - 2.0 * theta), tmp)

        return result

    finally:
        tmp.destroy()


def solve_newmark_discrete_adjoint_state(
    matrices: ElasticOperatorMatrices,
    rhs: FullAdjointObservationRHSResult,
    *,
    solve_config: LinearSolveConfig | None = None,
    reuse_effective_solver: bool = True,
) -> NewmarkAdjointStateResult:
    """
    Reverse accumulation for the discrete Newmark state transition.

    This solves the adjoint equation of the discrete state update

        A u^{n+1} = B u^n - A u^{n-1} + F^{n+1}

    in reverse time, where the observation gradients are supplied as
    RHS vectors at each observed time step.
    """

    if rhs.reflection.num_steps != rhs.transmission.num_steps:
        raise ValueError("Reflection and transmission RHS series must have same length.")

    num_steps = rhs.num_steps

    if num_steps <= 0:
        raise ValueError("The adjoint RHS series must be non-empty.")

    for index, (rhs_ref, rhs_trans) in enumerate(
        zip(rhs.reflection.rhs_vectors, rhs.transmission.rhs_vectors),
        start=1,
    ):
        if rhs_ref.getSize() != rhs_trans.getSize():
            raise ValueError(f"Incompatible RHS vector sizes at step {index}.")

        if rhs_ref.getSize() != matrices.effective.getSize()[1]:
            raise ValueError(
                f"RHS vector size does not match effective operator at step {index}."
            )

    carry_current = _zero_like(rhs.reflection.rhs_vectors[0])
    carry_previous = _zero_like(rhs.reflection.rhs_vectors[0])

    records: list[NewmarkAdjointStateStepRecord | None] = [None] * num_steps
    equation_adjoint_vectors: list[PETSc.Vec | None] = [None] * num_steps
    linear_solver: ReusableEffectiveLinearSolver | None = None

    try:
        if reuse_effective_solver:
            linear_solver = ReusableEffectiveLinearSolver(
                matrices.effective,
                solve_config,
            )

        for solve_index, time_index in enumerate(
            range(num_steps - 1, -1, -1),
            start=1,
        ):
            observation_rhs = _combine_observation_rhs(rhs, time_index)
            accumulated_rhs = observation_rhs.copy()
            accumulated_rhs.axpy(1.0, carry_current)

            try:
                if linear_solver is None:
                    equation_adjoint, report = solve_effective_system(
                        matrices.effective,
                        accumulated_rhs,
                        config=solve_config,
                    )
                else:
                    equation_adjoint, report = linear_solver.solve(accumulated_rhs)

                residual_norm = linear_residual_norm(
                    matrices.effective,
                    equation_adjoint,
                    accumulated_rhs,
                )

                accumulated_norm = float(accumulated_rhs.norm())
                relative_residual = float(residual_norm / max(accumulated_norm, 1.0))

                next_current = newmark_middle_operator_action(
                    matrices,
                    equation_adjoint,
                )
                next_current.axpy(1.0, carry_previous)

                next_previous = accumulated_rhs.copy()
                next_previous.scale(-1.0)

                ref_record = rhs.reflection.records[time_index]

                records[time_index] = NewmarkAdjointStateStepRecord(
                    step=ref_record.step,
                    time=ref_record.time,
                    solve_index=solve_index,
                    observation_rhs_norm=float(observation_rhs.norm()),
                    accumulated_rhs_norm=accumulated_norm,
                    equation_adjoint_norm=float(equation_adjoint.norm()),
                    carry_current_norm=float(next_current.norm()),
                    carry_previous_norm=float(next_previous.norm()),
                    linear_residual_norm=float(residual_norm),
                    linear_residual_relative=relative_residual,
                    ksp_converged=report.converged,
                    ksp_reason=report.reason,
                    ksp_iterations=report.iterations,
                    ksp_reported_residual_norm=report.residual_norm,
                )
                equation_adjoint_vectors[time_index] = equation_adjoint

                carry_current.destroy()
                carry_previous.destroy()

                carry_current = next_current
                carry_previous = next_previous

            finally:
                observation_rhs.destroy()
                accumulated_rhs.destroy()

        if any(record is None for record in records):
            raise RuntimeError("Internal error: missing Newmark adjoint record.")

        if any(vector is None for vector in equation_adjoint_vectors):
            raise RuntimeError("Internal error: missing Newmark adjoint vector.")

        result = NewmarkAdjointStateResult(
            records=tuple(record for record in records if record is not None),
            equation_adjoint_vectors=tuple(
                vector for vector in equation_adjoint_vectors if vector is not None
            ),
            initial_current_adjoint=carry_current,
            initial_previous_adjoint=carry_previous,
            reverse_time=True,
        )

        carry_current = None
        carry_previous = None

        return result

    except Exception:
        for vector in equation_adjoint_vectors:
            if vector is not None:
                vector.destroy()
        raise

    finally:
        if linear_solver is not None:
            linear_solver.destroy()

        if carry_current is not None:
            carry_current.destroy()

        if carry_previous is not None:
            carry_previous.destroy()
