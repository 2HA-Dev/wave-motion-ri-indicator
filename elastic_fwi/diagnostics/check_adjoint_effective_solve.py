from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from mpi4py import MPI

from elastic_fwi.geometry.boundary_tags import (
    build_boundary_meshtags,
    unit_square_pitch_catch_boundary_specs,
)
from elastic_fwi.inversion.adjoint_observation_sources import (
    assemble_full_adjoint_observation_rhs,
)
from elastic_fwi.inversion.adjoint_solver import solve_adjoint_effective_rhs_series
from elastic_fwi.inversion.observation_functionals import ObservationFunctionalWeights
from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    _build_materials,
)
from elastic_fwi.pipelines.observation_comparison import (
    DirectObservationComparisonConfig,
    run_direct_observation_comparison,
)
from elastic_fwi.solvers.elastic_operator import (
    ElasticOperatorConfig,
    assemble_elastic_operator_matrices,
)
from elastic_fwi.solvers.newmark import LinearSolveConfig


def _records_have_finite_solver_values(records: list[dict[str, Any]]) -> bool:
    return bool(
        all(
            np.isfinite(record["rhs_norm"])
            and np.isfinite(record["solution_norm"])
            and np.isfinite(record["linear_residual_norm"])
            and np.isfinite(record["linear_residual_relative"])
            and np.isfinite(record["ksp_reported_residual_norm"])
            for record in records
        )
    )


def _vectors_have_valid_sizes(vectors) -> bool:
    return bool(
        all(
            vector.getSize() > 0 and vector.getLocalSize() >= 0
            for vector in vectors
        )
    )


def run_adjoint_effective_solve_check(
    *,
    nx: int = 4,
    ny: int = 4,
    degree: int = 1,
    theta: float = 0.25,
    dt: float = 0.1,
    num_steps: int = 5,
    simulated_material_kind: str = "healthy",
    observed_material_kind: str = "localized_disk_defect",
    contrast_factor: float = 0.8,
    defect_center_x: float = 0.5,
    defect_center_y: float = 0.5,
    defect_radius: float = 0.35,
    source_amplitude: float = 1.0,
    source_frequency: float = 1.0,
    source_num_periods: float = 1.0,
    reflection_weight: float = 1.0,
    transmission_weight: float = 1.0,
    ksp_type: str = "cg",
    pc_type: str = "jacobi",
    ksp_rtol: float = 1.0e-10,
    ksp_atol: float = 1.0e-12,
    ksp_max_it: int = 1000,
    max_linear_residual_tolerance: float = 1.0e-8,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    simulated_config = DirectObservationPipelineConfig(
        nx=nx,
        ny=ny,
        degree=degree,
        theta=theta,
        dt=dt,
        num_steps=num_steps,
        material_kind=simulated_material_kind,
        contrast_factor=contrast_factor,
        defect_center_x=defect_center_x,
        defect_center_y=defect_center_y,
        defect_radius=defect_radius,
        source_amplitude=source_amplitude,
        source_frequency=source_frequency,
        source_num_periods=source_num_periods,
        ksp_type=ksp_type,
        pc_type=pc_type,
        ksp_rtol=ksp_rtol,
        ksp_atol=ksp_atol,
        ksp_max_it=ksp_max_it,
        store_displacement_snapshots=True,
    )

    observed_config = DirectObservationPipelineConfig(
        nx=nx,
        ny=ny,
        degree=degree,
        theta=theta,
        dt=dt,
        num_steps=num_steps,
        material_kind=observed_material_kind,
        contrast_factor=contrast_factor,
        defect_center_x=defect_center_x,
        defect_center_y=defect_center_y,
        defect_radius=defect_radius,
        source_amplitude=source_amplitude,
        source_frequency=source_frequency,
        source_num_periods=source_num_periods,
        ksp_type=ksp_type,
        pc_type=pc_type,
        ksp_rtol=ksp_rtol,
        ksp_atol=ksp_atol,
        ksp_max_it=ksp_max_it,
        store_displacement_snapshots=False,
    )

    comparison = None
    rhs = None
    matrices = None
    solve_result = None

    try:
        comparison = run_direct_observation_comparison(
            DirectObservationComparisonConfig(
                simulated=simulated_config,
                observed=observed_config,
                weights=ObservationFunctionalWeights(
                    reflection=reflection_weight,
                    transmission=transmission_weight,
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
                "Adjoint diagnostic requires one common direct finite-element "
                "space for healthy and defective wave problems."
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
            ElasticOperatorConfig(theta=theta, dt=dt),
        )

        rhs = assemble_full_adjoint_observation_rhs(
            V,
            facet_tags,
            comparison.simulated.displacement_snapshots,
            comparison.residuals,
            reflection_scale=reflection_weight,
            transmission_scale=transmission_weight,
        )

        solve_result = solve_adjoint_effective_rhs_series(
            matrices.effective,
            rhs,
            solve_config=LinearSolveConfig(
                ksp_type=ksp_type,
                pc_type=pc_type,
                rtol=ksp_rtol,
                atol=ksp_atol,
                max_it=ksp_max_it,
            ),
            reverse_time=True,
        )

        comparison_report = comparison.as_dict()
        rhs_report = rhs.as_dict()
        solve_report = solve_result.as_dict()

        records = solve_report["records"]
        matrix_rows, matrix_cols = matrices.effective.getSize()
        matrix_local_rows, matrix_local_cols = matrices.effective.getLocalSize()

        observed_defect_cells = int(
            comparison_report["values"]["observed_defect_cells"]
        )

        checks = {
            "comparison_status_ok": comparison_report["status"] == "ok",
            "simulated_snapshots_stored": comparison.simulated.has_displacement_snapshots,
            "snapshot_count_matches_num_steps": comparison.simulated.num_displacement_snapshots
            == num_steps,
            "residuals_have_expected_length": comparison.residuals.num_steps
            == num_steps,
            "rhs_num_steps_match": rhs.num_steps == num_steps,
            "rhs_nonzero": bool(rhs.max_rhs_norm > 0.0),
            "solve_num_steps_match": solve_result.num_steps == num_steps,
            "solve_reverse_time": solve_result.reverse_time is True,
            "solve_records_reverse_ordered": bool(
                records[0]["solve_index"] == num_steps
                and records[-1]["solve_index"] == 1
            ),
            "all_ksp_converged": solve_result.all_ksp_converged,
            "solver_values_finite": _records_have_finite_solver_values(records),
            "solution_vectors_valid_sizes": _vectors_have_valid_sizes(
                solve_result.adjoint_vectors
            ),
            "solution_nonzero": bool(solve_result.max_solution_norm > 0.0),
            "linear_residuals_below_tolerance": bool(
                solve_result.max_linear_residual_norm
                <= float(max_linear_residual_tolerance)
            ),
            "matrix_global_shape_square": bool(matrix_rows == matrix_cols),
            "matrix_local_shape_valid": bool(
                matrix_local_rows >= 0 and matrix_local_cols >= 0
            ),
            "rhs_solution_size_match": bool(
                all(
                    vector.getSize() == matrix_cols
                    for vector in solve_result.adjoint_vectors
                )
            ),
        }

        if observed_material_kind == "localized_disk_defect":
            checks["observed_defect_cells_positive"] = bool(observed_defect_cells > 0)

        status_ok = bool(all(checks.values()))

        return {
            "kind": "adjoint_effective_solve_check",
            "status": "ok" if status_ok else "failed",
            "mpi_size": comm.size,
            "checks": checks,
            "config": {
                "nx": nx,
                "ny": ny,
                "degree": degree,
                "theta": theta,
                "dt": dt,
                "num_steps": num_steps,
                "simulated_material_kind": simulated_material_kind,
                "observed_material_kind": observed_material_kind,
                "contrast_factor": contrast_factor,
                "defect_center_x": defect_center_x,
                "defect_center_y": defect_center_y,
                "defect_radius": defect_radius,
                "source_amplitude": source_amplitude,
                "source_frequency": source_frequency,
                "source_num_periods": source_num_periods,
                "reflection_weight": reflection_weight,
                "transmission_weight": transmission_weight,
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": ksp_rtol,
                "ksp_atol": ksp_atol,
                "ksp_max_it": ksp_max_it,
                "max_linear_residual_tolerance": max_linear_residual_tolerance,
            },
            "values": {
                "matrix_rows": int(matrix_rows),
                "matrix_cols": int(matrix_cols),
                "matrix_local_rows": int(matrix_local_rows),
                "matrix_local_cols": int(matrix_local_cols),
                "max_abs_residual": comparison.residuals.max_abs_residual,
                "max_rhs_norm": rhs.max_rhs_norm,
                "max_solution_norm": solve_result.max_solution_norm,
                "max_linear_residual_norm": solve_result.max_linear_residual_norm,
                "observed_defect_cells": observed_defect_cells,
                "max_abs_observation_difference": comparison_report["values"][
                    "max_abs_observation_difference"
                ],
            },
            "material": material_metadata,
            "adjoint_rhs": rhs_report,
            "adjoint_solve": solve_report,
            "comparison": comparison_report,
        }

    finally:
        if solve_result is not None:
            solve_result.destroy()

        if rhs is not None:
            rhs.destroy()

        if matrices is not None:
            matrices.destroy()

        if comparison is not None:
            comparison.simulated.destroy()
            comparison.observed.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check effective adjoint solves driven by observation RHS."
    )
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--simulated-material-kind", type=str, default="healthy")
    parser.add_argument(
        "--observed-material-kind",
        type=str,
        default="localized_disk_defect",
    )
    parser.add_argument("--contrast-factor", type=float, default=0.8)
    parser.add_argument("--defect-center-x", type=float, default=0.5)
    parser.add_argument("--defect-center-y", type=float, default=0.5)
    parser.add_argument("--defect-radius", type=float, default=0.35)
    parser.add_argument("--source-amplitude", type=float, default=1.0)
    parser.add_argument("--source-frequency", type=float, default=1.0)
    parser.add_argument("--source-num-periods", type=float, default=1.0)
    parser.add_argument("--reflection-weight", type=float, default=1.0)
    parser.add_argument("--transmission-weight", type=float, default=1.0)
    parser.add_argument("--ksp-type", type=str, default="cg")
    parser.add_argument("--pc-type", type=str, default="jacobi")
    parser.add_argument("--ksp-rtol", type=float, default=1.0e-10)
    parser.add_argument("--ksp-atol", type=float, default=1.0e-12)
    parser.add_argument("--ksp-max-it", type=int, default=1000)
    parser.add_argument("--max-linear-residual-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_adjoint_effective_solve_check(
        nx=args.nx,
        ny=args.ny,
        degree=args.degree,
        theta=args.theta,
        dt=args.dt,
        num_steps=args.num_steps,
        simulated_material_kind=args.simulated_material_kind,
        observed_material_kind=args.observed_material_kind,
        contrast_factor=args.contrast_factor,
        defect_center_x=args.defect_center_x,
        defect_center_y=args.defect_center_y,
        defect_radius=args.defect_radius,
        source_amplitude=args.source_amplitude,
        source_frequency=args.source_frequency,
        source_num_periods=args.source_num_periods,
        reflection_weight=args.reflection_weight,
        transmission_weight=args.transmission_weight,
        ksp_type=args.ksp_type,
        pc_type=args.pc_type,
        ksp_rtol=args.ksp_rtol,
        ksp_atol=args.ksp_atol,
        ksp_max_it=args.ksp_max_it,
        max_linear_residual_tolerance=args.max_linear_residual_tolerance,
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "OK" if report["status"] == "ok" else "FAILED"
            print(
                f"[ADJOINT EFFECTIVE SOLVE CHECK] {status} "
                f"for mesh={args.nx}x{args.ny}, degree={args.degree}, "
                f"theta={args.theta}, dt={args.dt}, steps={args.num_steps}, "
                f"simulated={args.simulated_material_kind}, "
                f"observed={args.observed_material_kind}, "
                f"max_residual={report['values']['max_abs_residual']:.6e}, "
                f"rhs_max={report['values']['max_rhs_norm']:.6e}, "
                f"solution_max={report['values']['max_solution_norm']:.6e}, "
                f"linear_residual_max={report['values']['max_linear_residual_norm']:.6e}, "
                f"MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
