from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from dolfinx import mesh
from mpi4py import MPI
from petsc4py import PETSc

from elastic_fwi.inversion.adjoint_observation_sources import (
    AdjointObservationRHSRecord,
    AdjointObservationRHSResult,
    FullAdjointObservationRHSResult,
)
from elastic_fwi.inversion.newmark_adjoint import (
    newmark_middle_operator_action,
    solve_newmark_discrete_adjoint_state,
)
from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    _build_materials,
)
from elastic_fwi.solvers.elastic_operator import (
    ElasticOperatorConfig,
    assemble_elastic_operator_matrices,
    create_vector_lagrange_space,
)
from elastic_fwi.solvers.newmark import LinearSolveConfig, solve_effective_system


def _fill_deterministic_vector(
    vector: PETSc.Vec,
    *,
    phase: float,
    scale: float,
) -> None:
    start, end = vector.getOwnershipRange()

    for gid in range(start, end):
        value = scale * (
            np.sin(0.37 * float(gid + 1) + phase)
            + 0.5 * np.cos(0.13 * float(gid + 1) + 2.0 * phase)
        )
        vector.setValue(gid, float(value))

    vector.assemble()


def _deterministic_vector_like(
    template: PETSc.Vec,
    *,
    phase: float,
    scale: float,
) -> PETSc.Vec:
    vector = template.copy()
    vector.set(0.0)
    _fill_deterministic_vector(vector, phase=phase, scale=scale)
    return vector


def _zero_vector_like(template: PETSc.Vec) -> PETSc.Vec:
    vector = template.copy()
    vector.set(0.0)
    vector.assemble()
    return vector


def _make_deterministic_rhs_series(
    template: PETSc.Vec,
    *,
    num_steps: int,
    dt: float,
    scale: float,
) -> FullAdjointObservationRHSResult:
    reflection_vectors = []
    transmission_vectors = []
    reflection_records = []
    transmission_records = []

    for index in range(num_steps):
        step = index + 1
        time = dt * step

        rhs_ref = _deterministic_vector_like(
            template,
            phase=0.23 * step,
            scale=scale,
        )
        rhs_trans = _deterministic_vector_like(
            template,
            phase=0.51 * step,
            scale=0.1 * scale,
        )

        reflection_vectors.append(rhs_ref)
        transmission_vectors.append(rhs_trans)

        reflection_records.append(
            AdjointObservationRHSRecord(
                step=step,
                time=time,
                tag=1,
                component="vector",
                residual_value=float(scale),
                rhs_norm=float(rhs_ref.norm()),
                rhs_size_global=int(rhs_ref.getSize()),
                rhs_size_local=int(rhs_ref.getLocalSize()),
            )
        )

        transmission_records.append(
            AdjointObservationRHSRecord(
                step=step,
                time=time,
                tag=2,
                component="vector",
                residual_value=float(0.1 * scale),
                rhs_norm=float(rhs_trans.norm()),
                rhs_size_global=int(rhs_trans.getSize()),
                rhs_size_local=int(rhs_trans.getLocalSize()),
            )
        )

    return FullAdjointObservationRHSResult(
        reflection=AdjointObservationRHSResult(
            records=tuple(reflection_records),
            rhs_vectors=tuple(reflection_vectors),
        ),
        transmission=AdjointObservationRHSResult(
            records=tuple(transmission_records),
            rhs_vectors=tuple(transmission_vectors),
        ),
    )


def _tangent_step(
    matrices,
    delta_current: PETSc.Vec,
    delta_previous: PETSc.Vec,
    *,
    solve_config: LinearSolveConfig,
) -> tuple[PETSc.Vec, Any]:
    rhs = newmark_middle_operator_action(matrices, delta_current)
    tmp = matrices.effective.createVecLeft()

    try:
        matrices.effective.mult(delta_previous, tmp)
        rhs.axpy(-1.0, tmp)

        return solve_effective_system(
            matrices.effective,
            rhs,
            config=solve_config,
        )

    finally:
        rhs.destroy()
        tmp.destroy()


def _compute_tangent_sequence(
    matrices,
    delta_current_initial: PETSc.Vec,
    delta_previous_initial: PETSc.Vec,
    *,
    num_steps: int,
    solve_config: LinearSolveConfig,
) -> tuple[tuple[PETSc.Vec, ...], list[dict[str, Any]]]:
    delta_previous = delta_previous_initial.copy()
    delta_current = delta_current_initial.copy()

    tangent_vectors: list[PETSc.Vec] = []
    records: list[dict[str, Any]] = []

    try:
        for step in range(1, num_steps + 1):
            delta_next, solve_report = _tangent_step(
                matrices,
                delta_current,
                delta_previous,
                solve_config=solve_config,
            )

            tangent_vectors.append(delta_next)

            records.append(
                {
                    "step": step,
                    "solution_norm": float(delta_next.norm()),
                    "ksp_converged": bool(solve_report.converged),
                    "ksp_reason": int(solve_report.reason),
                    "ksp_iterations": int(solve_report.iterations),
                    "ksp_reported_residual_norm": float(
                        solve_report.residual_norm
                    ),
                }
            )

            next_previous = delta_current.copy()
            next_current = delta_next.copy()

            delta_previous.destroy()
            delta_current.destroy()

            delta_previous = next_previous
            delta_current = next_current

        return tuple(tangent_vectors), records

    except Exception:
        for vector in tangent_vectors:
            vector.destroy()
        raise

    finally:
        delta_previous.destroy()
        delta_current.destroy()


def _combined_rhs_dot_tangent(
    rhs: FullAdjointObservationRHSResult,
    tangent_vectors: tuple[PETSc.Vec, ...],
) -> float:
    value = 0.0

    for index, tangent_vector in enumerate(tangent_vectors):
        value += float(rhs.reflection.rhs_vectors[index].dot(tangent_vector))
        value += float(rhs.transmission.rhs_vectors[index].dot(tangent_vector))

    return float(value)


def run_newmark_adjoint_duality_check(
    *,
    nx: int = 4,
    ny: int = 4,
    degree: int = 1,
    theta: float = 0.25,
    dt: float = 0.1,
    num_steps: int = 5,
    material_kind: str = "healthy",
    contrast_factor: float = 0.8,
    defect_center_x: float = 0.5,
    defect_center_y: float = 0.5,
    defect_radius: float = 0.35,
    rhs_scale: float = 1.0e-3,
    perturbation_scale: float = 1.0e-2,
    ksp_type: str = "cg",
    pc_type: str = "jacobi",
    ksp_rtol: float = 1.0e-10,
    ksp_atol: float = 1.0e-12,
    ksp_max_it: int = 1000,
    duality_tolerance: float = 1.0e-8,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    msh = mesh.create_unit_square(comm, nx, ny)
    V = create_vector_lagrange_space(msh, degree=degree)

    direct_config = DirectObservationPipelineConfig(
        nx=nx,
        ny=ny,
        degree=degree,
        theta=theta,
        dt=dt,
        num_steps=num_steps,
        material_kind=material_kind,
        contrast_factor=contrast_factor,
        defect_center_x=defect_center_x,
        defect_center_y=defect_center_y,
        defect_radius=defect_radius,
        ksp_type=ksp_type,
        pc_type=pc_type,
        ksp_rtol=ksp_rtol,
        ksp_atol=ksp_atol,
        ksp_max_it=ksp_max_it,
    )

    materials, material_metadata = _build_materials(msh, direct_config)

    matrices = assemble_elastic_operator_matrices(
        V,
        materials,
        ElasticOperatorConfig(theta=theta, dt=dt),
    )

    solve_config = LinearSolveConfig(
        ksp_type=ksp_type,
        pc_type=pc_type,
        rtol=ksp_rtol,
        atol=ksp_atol,
        max_it=ksp_max_it,
    )

    rhs = None
    adjoint_result = None
    tangent_vectors = ()
    delta_current_initial = None
    delta_previous_initial = None

    try:
        template = matrices.effective.createVecRight()

        try:
            template.set(0.0)
            template.assemble()

            rhs = _make_deterministic_rhs_series(
                template,
                num_steps=num_steps,
                dt=dt,
                scale=rhs_scale,
            )

            delta_current_initial = _deterministic_vector_like(
                template,
                phase=0.19,
                scale=perturbation_scale,
            )
            delta_previous_initial = _deterministic_vector_like(
                template,
                phase=0.73,
                scale=perturbation_scale,
            )

        finally:
            template.destroy()

        tangent_vectors, tangent_records = _compute_tangent_sequence(
            matrices,
            delta_current_initial,
            delta_previous_initial,
            num_steps=num_steps,
            solve_config=solve_config,
        )

        adjoint_result = solve_newmark_discrete_adjoint_state(
            matrices,
            rhs,
            solve_config=solve_config,
        )

        left = _combined_rhs_dot_tangent(rhs, tangent_vectors)

        right = float(
            adjoint_result.initial_current_adjoint.dot(delta_current_initial)
            + adjoint_result.initial_previous_adjoint.dot(delta_previous_initial)
        )

        duality_abs = abs(left - right)
        duality_scale = max(abs(left), abs(right), 1.0)
        duality_rel = duality_abs / duality_scale

        matrix_rows, matrix_cols = matrices.effective.getSize()
        matrix_local_rows, matrix_local_cols = matrices.effective.getLocalSize()

        tangent_all_converged = bool(
            all(record["ksp_converged"] for record in tangent_records)
        )

        tangent_values_finite = bool(
            all(
                np.isfinite(record["solution_norm"])
                and np.isfinite(record["ksp_reported_residual_norm"])
                for record in tangent_records
            )
        )

        checks = {
            "matrix_square": bool(matrix_rows == matrix_cols),
            "matrix_local_shape_valid": bool(
                matrix_local_rows >= 0 and matrix_local_cols >= 0
            ),
            "rhs_num_steps_match": rhs.num_steps == num_steps,
            "tangent_num_steps_match": len(tangent_vectors) == num_steps,
            "adjoint_num_steps_match": adjoint_result.num_steps == num_steps,
            "tangent_all_ksp_converged": tangent_all_converged,
            "adjoint_all_ksp_converged": adjoint_result.all_ksp_converged,
            "tangent_values_finite": tangent_values_finite,
            "adjoint_values_finite": bool(
                np.isfinite(adjoint_result.max_equation_adjoint_norm)
                and np.isfinite(adjoint_result.max_linear_residual_norm)
                and np.isfinite(adjoint_result.initial_current_adjoint.norm())
                and np.isfinite(adjoint_result.initial_previous_adjoint.norm())
            ),
            "duality_values_finite": bool(
                np.isfinite(left)
                and np.isfinite(right)
                and np.isfinite(duality_abs)
                and np.isfinite(duality_rel)
            ),
            "duality_close": bool(
                duality_abs <= float(duality_tolerance) * duality_scale
            ),
            "nonzero_tangent": bool(
                max(vector.norm() for vector in tangent_vectors) > 0.0
            ),
            "nonzero_adjoint": bool(
                adjoint_result.max_equation_adjoint_norm > 0.0
            ),
        }

        if material_kind == "localized_disk_defect":
            checks["defect_cells_positive"] = bool(
                material_metadata["global_defect_cells"] > 0
            )

        status_ok = bool(all(checks.values()))

        return {
            "kind": "newmark_adjoint_duality_check",
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
                "material_kind": material_kind,
                "contrast_factor": contrast_factor,
                "defect_center_x": defect_center_x,
                "defect_center_y": defect_center_y,
                "defect_radius": defect_radius,
                "rhs_scale": rhs_scale,
                "perturbation_scale": perturbation_scale,
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": ksp_rtol,
                "ksp_atol": ksp_atol,
                "ksp_max_it": ksp_max_it,
                "duality_tolerance": duality_tolerance,
            },
            "material": material_metadata,
            "matrix": {
                "rows": int(matrix_rows),
                "cols": int(matrix_cols),
                "local_rows": int(matrix_local_rows),
                "local_cols": int(matrix_local_cols),
                "operator": "newmark_state_transition",
            },
            "values": {
                "duality_left": left,
                "duality_right": right,
                "duality_abs": duality_abs,
                "duality_rel": duality_rel,
                "duality_scale": duality_scale,
                "duality_tolerance": duality_tolerance,
                "max_tangent_norm": float(
                    max(vector.norm() for vector in tangent_vectors)
                ),
                "max_adjoint_norm": adjoint_result.max_equation_adjoint_norm,
                "max_adjoint_linear_residual_norm": (
                    adjoint_result.max_linear_residual_norm
                ),
                "initial_current_adjoint_norm": float(
                    adjoint_result.initial_current_adjoint.norm()
                ),
                "initial_previous_adjoint_norm": float(
                    adjoint_result.initial_previous_adjoint.norm()
                ),
            },
            "tangent": {
                "num_steps": len(tangent_vectors),
                "records": tangent_records,
            },
            "adjoint": adjoint_result.as_dict(),
            "rhs": rhs.as_dict(),
        }

    finally:
        if adjoint_result is not None:
            adjoint_result.destroy()

        if rhs is not None:
            rhs.destroy()

        for vector in tangent_vectors:
            vector.destroy()

        if delta_current_initial is not None:
            delta_current_initial.destroy()

        if delta_previous_initial is not None:
            delta_previous_initial.destroy()

        matrices.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check discrete Newmark tangent/adjoint duality."
    )
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--material-kind", type=str, default="healthy")
    parser.add_argument("--contrast-factor", type=float, default=0.8)
    parser.add_argument("--defect-center-x", type=float, default=0.5)
    parser.add_argument("--defect-center-y", type=float, default=0.5)
    parser.add_argument("--defect-radius", type=float, default=0.35)
    parser.add_argument("--rhs-scale", type=float, default=1.0e-3)
    parser.add_argument("--perturbation-scale", type=float, default=1.0e-2)
    parser.add_argument("--ksp-type", type=str, default="cg")
    parser.add_argument("--pc-type", type=str, default="jacobi")
    parser.add_argument("--ksp-rtol", type=float, default=1.0e-10)
    parser.add_argument("--ksp-atol", type=float, default=1.0e-12)
    parser.add_argument("--ksp-max-it", type=int, default=1000)
    parser.add_argument("--duality-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_newmark_adjoint_duality_check(
        nx=args.nx,
        ny=args.ny,
        degree=args.degree,
        theta=args.theta,
        dt=args.dt,
        num_steps=args.num_steps,
        material_kind=args.material_kind,
        contrast_factor=args.contrast_factor,
        defect_center_x=args.defect_center_x,
        defect_center_y=args.defect_center_y,
        defect_radius=args.defect_radius,
        rhs_scale=args.rhs_scale,
        perturbation_scale=args.perturbation_scale,
        ksp_type=args.ksp_type,
        pc_type=args.pc_type,
        ksp_rtol=args.ksp_rtol,
        ksp_atol=args.ksp_atol,
        ksp_max_it=args.ksp_max_it,
        duality_tolerance=args.duality_tolerance,
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "OK" if report["status"] == "ok" else "FAILED"
            values = report["values"]
            print(
                f"[NEWMARK ADJOINT DUALITY CHECK] {status} "
                f"for mesh={args.nx}x{args.ny}, degree={args.degree}, "
                f"theta={args.theta}, dt={args.dt}, steps={args.num_steps}, "
                f"material={args.material_kind}, "
                f"left={values['duality_left']:.6e}, "
                f"right={values['duality_right']:.6e}, "
                f"abs={values['duality_abs']:.6e}, "
                f"rel={values['duality_rel']:.6e}, "
                f"MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
