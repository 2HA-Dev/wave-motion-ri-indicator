from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from petsc4py import PETSc

from elastic_fwi.inversion.adjoint_observation_sources import (
    FullAdjointObservationRHSResult,
)
from elastic_fwi.solvers.newmark import (
    LinearSolveConfig,
    LinearSolveReport,
    ReusableEffectiveLinearSolver,
    linear_residual_norm,
    solve_effective_system,
)


@dataclass(frozen=True)
class AdjointEffectiveStepRecord:
    step: int
    time: float
    solve_index: int
    rhs_norm: float
    solution_norm: float
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
            "rhs_norm": float(self.rhs_norm),
            "solution_norm": float(self.solution_norm),
            "linear_residual_norm": float(self.linear_residual_norm),
            "linear_residual_relative": float(self.linear_residual_relative),
            "ksp_converged": bool(self.ksp_converged),
            "ksp_reason": int(self.ksp_reason),
            "ksp_iterations": int(self.ksp_iterations),
            "ksp_reported_residual_norm": float(self.ksp_reported_residual_norm),
        }


@dataclass(frozen=True)
class AdjointEffectiveSolveResult:
    records: tuple[AdjointEffectiveStepRecord, ...]
    adjoint_vectors: tuple[PETSc.Vec, ...]
    reverse_time: bool

    def destroy(self) -> None:
        for vector in self.adjoint_vectors:
            vector.destroy()

    @property
    def num_steps(self) -> int:
        return len(self.records)

    @property
    def max_solution_norm(self) -> float:
        if not self.records:
            return 0.0
        return float(max(record.solution_norm for record in self.records))

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
            "kind": "adjoint_effective_solve_result",
            "num_steps": self.num_steps,
            "reverse_time": bool(self.reverse_time),
            "max_solution_norm": self.max_solution_norm,
            "max_linear_residual_norm": self.max_linear_residual_norm,
            "all_ksp_converged": self.all_ksp_converged,
            "records": [record.as_dict() for record in self.records],
        }


def _check_rhs_compatibility(rhs: FullAdjointObservationRHSResult) -> None:
    if rhs.reflection.num_steps != rhs.transmission.num_steps:
        raise ValueError("Reflection and transmission RHS series must have same length.")

    for index, (rhs_ref, rhs_trans) in enumerate(
        zip(rhs.reflection.rhs_vectors, rhs.transmission.rhs_vectors),
        start=1,
    ):
        if rhs_ref.getSize() != rhs_trans.getSize():
            raise ValueError(f"Incompatible RHS vector sizes at step {index}.")


def _combined_rhs_at_step(
    rhs: FullAdjointObservationRHSResult,
    index: int,
) -> PETSc.Vec:
    combined = rhs.reflection.rhs_vectors[index].copy()
    combined.axpy(1.0, rhs.transmission.rhs_vectors[index])
    return combined


def solve_adjoint_effective_rhs_series(
    matrix: PETSc.Mat,
    rhs: FullAdjointObservationRHSResult,
    *,
    solve_config: LinearSolveConfig | None = None,
    reverse_time: bool = True,
    reuse_effective_solver: bool = True,
) -> AdjointEffectiveSolveResult:
    """
    Solve A_eff p^n = b_adj^n for every adjoint observation RHS.

    The solve is intentionally algebraic and minimal:
    - same effective matrix as the direct Newmark step,
    - combined RHS b_adj = b_Gamma1 + b_Gamma2,
    - optional reverse-time ordering,
    - no full Newmark adjoint recurrence (provided by ``newmark_adjoint``).
    """

    _check_rhs_compatibility(rhs)

    num_steps = rhs.num_steps
    if num_steps <= 0:
        raise ValueError("The adjoint RHS series must be non-empty.")

    order = list(range(num_steps - 1, -1, -1)) if reverse_time else list(range(num_steps))

    records: list[AdjointEffectiveStepRecord | None] = [None] * num_steps
    solutions: list[PETSc.Vec | None] = [None] * num_steps
    linear_solver: ReusableEffectiveLinearSolver | None = None

    try:
        if reuse_effective_solver:
            linear_solver = ReusableEffectiveLinearSolver(matrix, solve_config)

        for solve_index, time_index in enumerate(order, start=1):
            combined_rhs = _combined_rhs_at_step(rhs, time_index)

            try:
                if linear_solver is None:
                    solution, report = solve_effective_system(
                        matrix,
                        combined_rhs,
                        config=solve_config,
                    )
                else:
                    solution, report = linear_solver.solve(combined_rhs)

                residual_norm = linear_residual_norm(
                    matrix,
                    solution,
                    combined_rhs,
                )

                rhs_norm = float(combined_rhs.norm())
                solution_norm = float(solution.norm())
                relative = float(residual_norm / max(rhs_norm, 1.0))

                ref_record = rhs.reflection.records[time_index]

                records[time_index] = AdjointEffectiveStepRecord(
                    step=ref_record.step,
                    time=ref_record.time,
                    solve_index=solve_index,
                    rhs_norm=rhs_norm,
                    solution_norm=solution_norm,
                    linear_residual_norm=float(residual_norm),
                    linear_residual_relative=relative,
                    ksp_converged=report.converged,
                    ksp_reason=report.reason,
                    ksp_iterations=report.iterations,
                    ksp_reported_residual_norm=report.residual_norm,
                )
                solutions[time_index] = solution

            finally:
                combined_rhs.destroy()

        if any(record is None for record in records):
            raise RuntimeError("Internal error: missing adjoint solve record.")

        if any(vector is None for vector in solutions):
            raise RuntimeError("Internal error: missing adjoint solution vector.")

        return AdjointEffectiveSolveResult(
            records=tuple(record for record in records if record is not None),
            adjoint_vectors=tuple(vector for vector in solutions if vector is not None),
            reverse_time=reverse_time,
        )

    except Exception:
        for vector in solutions:
            if vector is not None:
                vector.destroy()
        raise

    finally:
        if linear_solver is not None:
            linear_solver.destroy()
