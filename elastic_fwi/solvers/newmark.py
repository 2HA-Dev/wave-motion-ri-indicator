from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from petsc4py import PETSc

from elastic_fwi.solvers.elastic_operator import ElasticOperatorMatrices


@dataclass(frozen=True)
class LinearSolveConfig:
    """PETSc KSP configuration for the effective Newmark solve.

    Default scientific contract
    ---------------------------
    The repository default is intentionally conservative and transparent:

        ksp_type = "cg"
        pc_type  = "jacobi"

    This is mathematically consistent for the symmetric positive definite
    effective Newmark operator

        A = M + theta * dt**2 * K,

    when the density is positive and the elastic tensor is coercive.  It is
    not claimed to be the fastest production choice.  More advanced PETSc
    choices such as LU/MUMPS, GAMG or HYPRE are exposed through this same
    configuration and through solver-preconditioner diagnostics.
    """

    ksp_type: str = "cg"
    pc_type: str = "jacobi"
    rtol: float = 1.0e-10
    atol: float = 1.0e-12
    max_it: int = 1000
    pc_factor_mat_solver_type: str | None = None

    def __post_init__(self) -> None:
        rtol = float(self.rtol)
        atol = float(self.atol)
        max_it = int(self.max_it)

        if not self.ksp_type:
            raise ValueError("ksp_type must be non-empty.")

        if not self.pc_type:
            raise ValueError("pc_type must be non-empty.")

        if not isfinite(rtol) or rtol < 0.0:
            raise ValueError("rtol must be finite and non-negative.")

        if not isfinite(atol) or atol < 0.0:
            raise ValueError("atol must be finite and non-negative.")

        if max_it <= 0:
            raise ValueError("max_it must be strictly positive.")

        if self.pc_factor_mat_solver_type is not None:
            solver_type = str(self.pc_factor_mat_solver_type)
            if not solver_type:
                raise ValueError("pc_factor_mat_solver_type must be non-empty when provided.")
            object.__setattr__(self, "pc_factor_mat_solver_type", solver_type)

        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "max_it", max_it)


@dataclass(frozen=True)
class LinearSolveReport:
    """Summary of a PETSc linear solve."""

    converged: bool
    reason: int
    iterations: int
    residual_norm: float


@dataclass
class NewmarkStepResult:
    """Result of one Newmark-theta displacement step."""

    displacement: PETSc.Vec
    rhs: PETSc.Vec
    solve_report: LinearSolveReport

    def destroy(self) -> None:
        for vector in (self.displacement, self.rhs):
            try:
                vector.destroy()
            except Exception:
                pass


class ReusableEffectiveLinearSolver:
    """Reusable PETSc KSP for a fixed effective Newmark matrix."""

    def __init__(
        self,
        matrix: PETSc.Mat,
        config: LinearSolveConfig | None = None,
    ) -> None:
        if config is None:
            config = LinearSolveConfig()

        self.matrix = matrix
        self.config = config
        self._destroyed = False

        self.ksp = PETSc.KSP().create(matrix.getComm())
        self.ksp.setOperators(matrix)
        self.ksp.setType(config.ksp_type)

        pc = self.ksp.getPC()
        pc.setType(config.pc_type)

        if config.pc_factor_mat_solver_type is not None:
            pc.setFactorSolverType(config.pc_factor_mat_solver_type)

        self.ksp.setTolerances(
            rtol=config.rtol,
            atol=config.atol,
            max_it=config.max_it,
        )

        self.ksp.setFromOptions()
        self.ksp.setUp()

    def solve(self, rhs: PETSc.Vec) -> tuple[PETSc.Vec, LinearSolveReport]:
        if self._destroyed:
            raise RuntimeError(
                "ReusableEffectiveLinearSolver has already been destroyed."
            )

        matrix_rows, _ = self.matrix.getSize()
        if rhs.getSize() != matrix_rows:
            raise ValueError("RHS vector size does not match matrix row size.")

        solution = self.matrix.createVecRight()
        self.ksp.solve(rhs, solution)

        reason = int(self.ksp.getConvergedReason())
        iterations = int(self.ksp.getIterationNumber())
        residual_norm = float(self.ksp.getResidualNorm())

        report = LinearSolveReport(
            converged=reason > 0,
            reason=reason,
            iterations=iterations,
            residual_norm=residual_norm,
        )

        return solution, report

    def destroy(self) -> None:
        if not self._destroyed:
            try:
                self.ksp.destroy()
            finally:
                self._destroyed = True

    def __enter__(self) -> ReusableEffectiveLinearSolver:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy()


def create_zero_state_vector(matrices: ElasticOperatorMatrices) -> PETSc.Vec:
    vector = matrices.effective.createVecRight()
    vector.set(0.0)
    return vector


def create_constant_state_vector(
    matrices: ElasticOperatorMatrices,
    value: float,
) -> PETSc.Vec:
    value = float(value)

    if not isfinite(value):
        raise ValueError("State vector value must be finite.")

    vector = matrices.effective.createVecRight()
    vector.set(value)
    return vector


def _check_compatible_vectors(reference: PETSc.Vec, *vectors: PETSc.Vec) -> None:
    reference_size = reference.getSize()

    for vector in vectors:
        if vector.getSize() != reference_size:
            raise ValueError("Incompatible PETSc vector sizes.")


def build_newmark_theta_rhs(
    matrices: ElasticOperatorMatrices,
    u_current: PETSc.Vec,
    u_previous: PETSc.Vec,
    force: PETSc.Vec | None = None,
) -> PETSc.Vec:
    """
    Build the RHS of the centered Newmark-theta displacement scheme.

    The scheme is

        A u_next =
            M(2 u_current - u_previous)
            - dt^2 K((1 - 2 theta) u_current + theta u_previous)
            + dt^2 force,

    where A = M + theta dt^2 K.
    """

    theta = matrices.config.theta
    dt2 = matrices.config.dt**2

    _check_compatible_vectors(u_current, u_previous)

    if force is not None:
        _check_compatible_vectors(u_current, force)

    mass_argument = matrices.mass.createVecRight()
    stiffness_argument = matrices.stiffness.createVecRight()
    stiffness_action = matrices.stiffness.createVecLeft()
    rhs = matrices.mass.createVecLeft()

    mass_argument.set(0.0)
    mass_argument.axpy(2.0, u_current)
    mass_argument.axpy(-1.0, u_previous)

    matrices.mass.mult(mass_argument, rhs)

    stiffness_argument.set(0.0)
    stiffness_argument.axpy(1.0 - 2.0 * theta, u_current)
    stiffness_argument.axpy(theta, u_previous)

    matrices.stiffness.mult(stiffness_argument, stiffness_action)
    rhs.axpy(-dt2, stiffness_action)

    if matrices.damping is not None:
        damping_action = matrices.damping.createVecLeft()
        matrices.damping.mult(u_previous, damping_action)
        rhs.axpy(0.5 * matrices.config.dt, damping_action)
        damping_action.destroy()

    if force is not None:
        rhs.axpy(dt2, force)

    mass_argument.destroy()
    stiffness_argument.destroy()
    stiffness_action.destroy()

    return rhs


def solve_effective_system(
    matrix: PETSc.Mat,
    rhs: PETSc.Vec,
    config: LinearSolveConfig | None = None,
) -> tuple[PETSc.Vec, LinearSolveReport]:
    if config is None:
        config = LinearSolveConfig()

    solution = matrix.createVecRight()

    ksp = PETSc.KSP().create(matrix.getComm())
    ksp.setOperators(matrix)
    ksp.setType(config.ksp_type)

    pc = ksp.getPC()
    pc.setType(config.pc_type)

    if config.pc_factor_mat_solver_type is not None:
        pc.setFactorSolverType(config.pc_factor_mat_solver_type)

    ksp.setTolerances(
        rtol=config.rtol,
        atol=config.atol,
        max_it=config.max_it,
    )

    ksp.setFromOptions()
    ksp.solve(rhs, solution)

    reason = int(ksp.getConvergedReason())
    iterations = int(ksp.getIterationNumber())
    residual_norm = float(ksp.getResidualNorm())

    report = LinearSolveReport(
        converged=reason > 0,
        reason=reason,
        iterations=iterations,
        residual_norm=residual_norm,
    )

    ksp.destroy()

    return solution, report


def linear_residual_norm(
    matrix: PETSc.Mat,
    solution: PETSc.Vec,
    rhs: PETSc.Vec,
) -> float:
    _check_compatible_vectors(solution, rhs)

    residual = matrix.createVecLeft()
    matrix.mult(solution, residual)
    residual.axpy(-1.0, rhs)

    norm = float(residual.norm())

    residual.destroy()

    return norm


def newmark_theta_step(
    matrices: ElasticOperatorMatrices,
    u_current: PETSc.Vec,
    u_previous: PETSc.Vec,
    force: PETSc.Vec | None = None,
    solve_config: LinearSolveConfig | None = None,
) -> NewmarkStepResult:
    rhs = build_newmark_theta_rhs(
        matrices,
        u_current,
        u_previous,
        force,
    )

    displacement, report = solve_effective_system(
        matrices.effective,
        rhs,
        solve_config,
    )

    return NewmarkStepResult(
        displacement=displacement,
        rhs=rhs,
        solve_report=report,
    )


def newmark_theta_step_with_solver(
    matrices: ElasticOperatorMatrices,
    u_current: PETSc.Vec,
    u_previous: PETSc.Vec,
    force: PETSc.Vec | None,
    linear_solver: ReusableEffectiveLinearSolver,
) -> NewmarkStepResult:
    rhs = build_newmark_theta_rhs(
        matrices,
        u_current,
        u_previous,
        force,
    )

    displacement, report = linear_solver.solve(rhs)

    return NewmarkStepResult(
        displacement=displacement,
        rhs=rhs,
        solve_report=report,
    )
