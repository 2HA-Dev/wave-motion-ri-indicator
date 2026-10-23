from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from petsc4py import PETSc

from elastic_fwi.solvers.elastic_operator import ElasticOperatorMatrices
from elastic_fwi.solvers.newmark import (
    LinearSolveConfig,
    ReusableEffectiveLinearSolver,
    create_zero_state_vector,
    linear_residual_norm,
    newmark_theta_step,
    newmark_theta_step_with_solver,
)
from elastic_fwi.sources.traction import (
    BoundaryTractionSource,
    assemble_boundary_traction_vector,
)


@dataclass(frozen=True)
class NewmarkTimeLoopConfig:
    """Configuration for a minimal Newmark time loop."""

    num_steps: int
    start_time: float = 0.0
    store_displacements: bool = False
    reuse_effective_solver: bool = True

    def __post_init__(self) -> None:
        num_steps = int(self.num_steps)
        start_time = float(self.start_time)
        store_displacements = bool(self.store_displacements)
        reuse_effective_solver = bool(self.reuse_effective_solver)

        if num_steps <= 0:
            raise ValueError("num_steps must be strictly positive.")

        if not isfinite(start_time):
            raise ValueError("start_time must be finite.")

        object.__setattr__(self, "num_steps", num_steps)
        object.__setattr__(self, "start_time", start_time)
        object.__setattr__(self, "store_displacements", store_displacements)
        object.__setattr__(self, "reuse_effective_solver", reuse_effective_solver)


@dataclass(frozen=True)
class NewmarkTimeStepRecord:
    """Scalar diagnostics for one Newmark time step."""

    step: int
    time: float
    force_norm: float
    rhs_norm: float
    displacement_norm: float
    linear_residual_norm: float
    linear_residual_relative: float
    ksp_converged: bool
    ksp_reason: int
    ksp_iterations: int
    ksp_reported_residual_norm: float

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "step": self.step,
            "time": self.time,
            "force_norm": self.force_norm,
            "rhs_norm": self.rhs_norm,
            "displacement_norm": self.displacement_norm,
            "linear_residual_norm": self.linear_residual_norm,
            "linear_residual_relative": self.linear_residual_relative,
            "ksp_converged": self.ksp_converged,
            "ksp_reason": self.ksp_reason,
            "ksp_iterations": self.ksp_iterations,
            "ksp_reported_residual_norm": self.ksp_reported_residual_norm,
        }


@dataclass
class NewmarkTimeLoopResult:
    """Result of a minimal Newmark time loop."""

    records: list[NewmarkTimeStepRecord]
    final_displacement: PETSc.Vec
    previous_displacement: PETSc.Vec
    displacement_snapshots: list[PETSc.Vec] = field(default_factory=list)

    def destroy(self) -> None:
        for vector in (self.final_displacement, self.previous_displacement):
            try:
                vector.destroy()
            except Exception:
                pass

        for vector in self.displacement_snapshots:
            try:
                vector.destroy()
            except Exception:
                pass

        self.displacement_snapshots.clear()


def _relative_error(abs_error: float, reference_norm: float) -> float:
    denominator = max(float(reference_norm), 1.0)
    return float(abs_error / denominator)


def _assemble_source_vector(*, V, facet_tags, source, t: float):
    """
    Assemble either a boundary traction source or a source object exposing
    assemble_vector(V, facet_tags, t).
    """
    if hasattr(source, "assemble_vector"):
        return source.assemble_vector(V, facet_tags, t)

    return assemble_boundary_traction_vector(
        V,
        facet_tags,
        source,
        t=t,
    )


def run_newmark_time_loop(
    *,
    matrices: ElasticOperatorMatrices,
    V,
    facet_tags,
    source: BoundaryTractionSource,
    config: NewmarkTimeLoopConfig,
    solve_config: LinearSolveConfig | None = None,
) -> NewmarkTimeLoopResult:
    """
    Run a minimal direct Newmark time loop.

    Initial conditions are

        u^{-1} = 0,
        u^0    = 0.

    At each step, the weak traction source is assembled at t_{n+1}.
    """

    if solve_config is None:
        solve_config = LinearSolveConfig()

    dt = matrices.config.dt

    linear_solver = None
    u_previous = None
    u_current = None
    records: list[NewmarkTimeStepRecord] = []
    displacement_snapshots: list[PETSc.Vec] = []

    try:
        if config.reuse_effective_solver:
            linear_solver = ReusableEffectiveLinearSolver(
                matrices.effective,
                solve_config,
            )

        u_previous = create_zero_state_vector(matrices)
        u_current = create_zero_state_vector(matrices)

        for step in range(config.num_steps):
            time = config.start_time + (step + 1) * dt

            force = _assemble_source_vector(
                V=V,
                facet_tags=facet_tags,
                source=source,
                t=time,
            )

            if linear_solver is not None:
                result = newmark_theta_step_with_solver(
                    matrices,
                    u_current,
                    u_previous,
                    force,
                    linear_solver,
                )
            else:
                result = newmark_theta_step(
                    matrices,
                    u_current,
                    u_previous,
                    force,
                    solve_config,
                )

            residual_norm = linear_residual_norm(
                matrices.effective,
                result.displacement,
                result.rhs,
            )

            force_norm = float(force.norm())
            rhs_norm = float(result.rhs.norm())
            displacement_norm = float(result.displacement.norm())
            residual_relative = _relative_error(residual_norm, rhs_norm)

            records.append(
                NewmarkTimeStepRecord(
                    step=step + 1,
                    time=float(time),
                    force_norm=force_norm,
                    rhs_norm=rhs_norm,
                    displacement_norm=displacement_norm,
                    linear_residual_norm=float(residual_norm),
                    linear_residual_relative=float(residual_relative),
                    ksp_converged=bool(result.solve_report.converged),
                    ksp_reason=int(result.solve_report.reason),
                    ksp_iterations=int(result.solve_report.iterations),
                    ksp_reported_residual_norm=float(
                        result.solve_report.residual_norm
                    ),
                )
            )

            if config.store_displacements:
                displacement_snapshots.append(result.displacement.copy())

            old_previous = u_previous
            u_previous = u_current
            u_current = result.displacement

            old_previous.destroy()
            result.rhs.destroy()
            force.destroy()

    except Exception:
        if u_previous is not None:
            try:
                u_previous.destroy()
            except Exception:
                pass

        if u_current is not None:
            try:
                u_current.destroy()
            except Exception:
                pass

        for vector in displacement_snapshots:
            try:
                vector.destroy()
            except Exception:
                pass

        raise

    finally:
        if linear_solver is not None:
            linear_solver.destroy()

    return NewmarkTimeLoopResult(
        records=records,
        final_displacement=u_current,
        previous_displacement=u_previous,
        displacement_snapshots=displacement_snapshots,
    )
