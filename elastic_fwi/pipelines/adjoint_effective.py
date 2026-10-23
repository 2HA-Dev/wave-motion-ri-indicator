from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mpi4py import MPI

from elastic_fwi.geometry.boundary_tags import (
    build_boundary_meshtags,
    unit_square_pitch_catch_boundary_specs,
)
from elastic_fwi.inversion.adjoint_observation_sources import (
    FullAdjointObservationRHSResult,
    assemble_full_adjoint_observation_rhs,
)
from elastic_fwi.inversion.adjoint_solver import (
    AdjointEffectiveSolveResult,
    solve_adjoint_effective_rhs_series,
)
from elastic_fwi.inversion.observation_functionals import ObservationFunctionalWeights
from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    _build_materials,
)
from elastic_fwi.pipelines.observation_comparison import (
    DirectObservationComparisonConfig,
    DirectObservationComparisonResult,
    run_direct_observation_comparison,
)
from elastic_fwi.solvers.elastic_operator import (
    ElasticOperatorConfig,
    assemble_elastic_operator_matrices,
)
from elastic_fwi.solvers.newmark import LinearSolveConfig


@dataclass(frozen=True)
class AdjointEffectivePipelineConfig:
    nx: int = 4
    ny: int = 4
    degree: int = 1
    theta: float = 0.25
    dt: float = 0.1
    num_steps: int = 5

    simulated_material_kind: str = "healthy"
    observed_material_kind: str = "localized_disk_defect"

    contrast_factor: float = 0.8
    defect_center_x: float = 0.5
    defect_center_y: float = 0.5
    defect_radius: float = 0.35

    source_amplitude: float = 1.0
    source_frequency: float = 1.0
    source_num_periods: float = 1.0

    reflection_weight: float = 1.0
    transmission_weight: float = 1.0

    ksp_type: str = "cg"
    pc_type: str = "jacobi"
    ksp_rtol: float = 1.0e-10
    ksp_atol: float = 1.0e-12
    ksp_max_it: int = 1000

    reverse_time: bool = True
    reuse_direct_effective_solver: bool = True
    reuse_adjoint_effective_solver: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "degree": self.degree,
            "theta": self.theta,
            "dt": self.dt,
            "num_steps": self.num_steps,
            "simulated_material_kind": self.simulated_material_kind,
            "observed_material_kind": self.observed_material_kind,
            "contrast_factor": self.contrast_factor,
            "defect_center_x": self.defect_center_x,
            "defect_center_y": self.defect_center_y,
            "defect_radius": self.defect_radius,
            "source_amplitude": self.source_amplitude,
            "source_frequency": self.source_frequency,
            "source_num_periods": self.source_num_periods,
            "reflection_weight": self.reflection_weight,
            "transmission_weight": self.transmission_weight,
            "ksp_type": self.ksp_type,
            "pc_type": self.pc_type,
            "ksp_rtol": self.ksp_rtol,
            "ksp_atol": self.ksp_atol,
            "ksp_max_it": self.ksp_max_it,
            "reverse_time": self.reverse_time,
            "reuse_direct_effective_solver": self.reuse_direct_effective_solver,
            "reuse_adjoint_effective_solver": self.reuse_adjoint_effective_solver,
        }


@dataclass(frozen=True)
class AdjointEffectivePipelineResult:
    config: AdjointEffectivePipelineConfig
    comparison: DirectObservationComparisonResult
    adjoint_rhs: FullAdjointObservationRHSResult
    adjoint_solve: AdjointEffectiveSolveResult
    material_metadata: dict[str, Any]
    matrix_metadata: dict[str, Any]
    mpi_size: int

    def destroy(self) -> None:
        self.adjoint_solve.destroy()
        self.adjoint_rhs.destroy()
        self.comparison.simulated.destroy()
        self.comparison.observed.destroy()

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "adjoint_effective_pipeline_result",
            "config": self.config.as_dict(),
            "mpi_size": int(self.mpi_size),
            "material": dict(self.material_metadata),
            "matrix": dict(self.matrix_metadata),
            "comparison": self.comparison.as_dict(),
            "adjoint_rhs": self.adjoint_rhs.as_dict(),
            "adjoint_solve": self.adjoint_solve.as_dict(),
            "values": {
                "max_abs_residual": self.comparison.residuals.max_abs_residual,
                "max_rhs_norm": self.adjoint_rhs.max_rhs_norm,
                "max_solution_norm": self.adjoint_solve.max_solution_norm,
                "max_linear_residual_norm": self.adjoint_solve.max_linear_residual_norm,
            },
        }


def _direct_config(
    config: AdjointEffectivePipelineConfig,
    *,
    material_kind: str,
    store_displacement_snapshots: bool,
) -> DirectObservationPipelineConfig:
    return DirectObservationPipelineConfig(
        nx=config.nx,
        ny=config.ny,
        degree=config.degree,
        theta=config.theta,
        dt=config.dt,
        num_steps=config.num_steps,
        material_kind=material_kind,
        contrast_factor=config.contrast_factor,
        defect_center_x=config.defect_center_x,
        defect_center_y=config.defect_center_y,
        defect_radius=config.defect_radius,
        source_amplitude=config.source_amplitude,
        source_frequency=config.source_frequency,
        source_num_periods=config.source_num_periods,
        ksp_type=config.ksp_type,
        pc_type=config.pc_type,
        ksp_rtol=config.ksp_rtol,
        ksp_atol=config.ksp_atol,
        ksp_max_it=config.ksp_max_it,
        store_displacement_snapshots=store_displacement_snapshots,
        reuse_effective_solver=config.reuse_direct_effective_solver,
    )


def run_adjoint_effective_pipeline(
    config: AdjointEffectivePipelineConfig | None = None,
    *,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> AdjointEffectivePipelineResult:
    if config is None:
        config = AdjointEffectivePipelineConfig()

    simulated_config = _direct_config(
        config,
        material_kind=config.simulated_material_kind,
        store_displacement_snapshots=True,
    )

    observed_config = _direct_config(
        config,
        material_kind=config.observed_material_kind,
        store_displacement_snapshots=False,
    )

    comparison = None
    adjoint_rhs = None
    adjoint_solve = None
    matrices = None

    try:
        comparison = run_direct_observation_comparison(
            DirectObservationComparisonConfig(
                simulated=simulated_config,
                observed=observed_config,
                weights=ObservationFunctionalWeights(
                    reflection=config.reflection_weight,
                    transmission=config.transmission_weight,
                ),
            ),
            comm=comm,
        )

        V = comparison.simulated.function_space
        if V is None:
            raise RuntimeError(
                "Simulated direct result does not expose function_space."
            )
        if V is not comparison.observed.function_space:
            raise RuntimeError(
                "Adjoint pipeline requires the simulated and observed direct "
                "problems to share the same finite-element space."
            )

        msh = V.mesh
        facet_tags = build_boundary_meshtags(
            msh,
            unit_square_pitch_catch_boundary_specs(),
        )

        materials, material_metadata = _build_materials(msh, simulated_config)

        matrices = assemble_elastic_operator_matrices(
            V,
            materials,
            ElasticOperatorConfig(theta=config.theta, dt=config.dt),
        )

        matrix_rows, matrix_cols = matrices.effective.getSize()
        matrix_local_rows, matrix_local_cols = matrices.effective.getLocalSize()

        adjoint_rhs = assemble_full_adjoint_observation_rhs(
            V,
            facet_tags,
            comparison.simulated.displacement_snapshots,
            comparison.residuals,
            reflection_scale=config.reflection_weight,
            transmission_scale=config.transmission_weight,
        )

        adjoint_solve = solve_adjoint_effective_rhs_series(
            matrices.effective,
            adjoint_rhs,
            solve_config=LinearSolveConfig(
                ksp_type=config.ksp_type,
                pc_type=config.pc_type,
                rtol=config.ksp_rtol,
                atol=config.ksp_atol,
                max_it=config.ksp_max_it,
            ),
            reverse_time=config.reverse_time,
            reuse_effective_solver=config.reuse_adjoint_effective_solver,
        )

        result = AdjointEffectivePipelineResult(
            config=config,
            comparison=comparison,
            adjoint_rhs=adjoint_rhs,
            adjoint_solve=adjoint_solve,
            material_metadata=material_metadata,
            matrix_metadata={
                "rows": int(matrix_rows),
                "cols": int(matrix_cols),
                "local_rows": int(matrix_local_rows),
                "local_cols": int(matrix_local_cols),
                "operator": "effective_newmark",
            },
            mpi_size=comm.size,
        )

        comparison = None
        adjoint_rhs = None
        adjoint_solve = None

        return result

    finally:
        if adjoint_solve is not None:
            adjoint_solve.destroy()

        if adjoint_rhs is not None:
            adjoint_rhs.destroy()

        if matrices is not None:
            matrices.destroy()

        if comparison is not None:
            comparison.simulated.destroy()
            comparison.observed.destroy()
