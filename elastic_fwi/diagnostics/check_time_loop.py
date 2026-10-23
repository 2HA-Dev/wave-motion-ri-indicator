from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

from elastic_fwi.geometry.boundary_tags import (
    build_boundary_meshtags,
    unit_square_pitch_catch_boundary_specs,
)
from elastic_fwi.materials.dg0_fields import homogeneous_dg0_material_functions
from elastic_fwi.materials.orthotropic import LEGACY_HEALTHY_MATERIAL
from elastic_fwi.solvers.elastic_operator import (
    ElasticOperatorConfig,
    assemble_elastic_operator_matrices,
    create_vector_lagrange_space,
)
from elastic_fwi.solvers.newmark import LinearSolveConfig
from elastic_fwi.solvers.time_loop import (
    NewmarkTimeLoopConfig,
    run_newmark_time_loop,
)
from elastic_fwi.sources.traction import gamma1_vertical_traction_source


def run_time_loop_check(
    *,
    nx: int = 2,
    ny: int = 2,
    degree: int = 1,
    theta: float = 0.25,
    dt: float = 0.1,
    num_steps: int = 5,
    source_amplitude: float = 1.0,
    source_frequency: float = 1.0,
    source_num_periods: float = 1.0,
    ksp_rtol: float = 1.0e-10,
    ksp_atol: float = 1.0e-12,
    ksp_max_it: int = 1000,
    residual_atol: float = 1.0e-8,
    residual_rtol: float = 1.0e-8,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be strictly positive.")

    if degree <= 0:
        raise ValueError("degree must be strictly positive.")

    if num_steps <= 0:
        raise ValueError("num_steps must be strictly positive.")

    msh = mesh.create_unit_square(comm, nx, ny)
    facet_tags = build_boundary_meshtags(
        msh,
        unit_square_pitch_catch_boundary_specs(),
    )

    V = create_vector_lagrange_space(msh, degree=degree)
    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)

    operator_config = ElasticOperatorConfig(theta=theta, dt=dt)
    matrices = assemble_elastic_operator_matrices(V, materials, operator_config)

    source = gamma1_vertical_traction_source(
        tag=1,
        amplitude=source_amplitude,
        frequency=source_frequency,
        t0=0.0,
        num_periods=source_num_periods,
    )

    loop_config = NewmarkTimeLoopConfig(num_steps=num_steps, start_time=0.0)
    solve_config = LinearSolveConfig(
        ksp_type="cg",
        pc_type="jacobi",
        rtol=ksp_rtol,
        atol=ksp_atol,
        max_it=ksp_max_it,
    )

    try:
        result = run_newmark_time_loop(
            matrices=matrices,
            V=V,
            facet_tags=facet_tags,
            source=source,
            config=loop_config,
            solve_config=solve_config,
        )

        records = result.records
        record_dicts = [record.as_dict() for record in records]

        times = [record.time for record in records]
        force_norms = [record.force_norm for record in records]
        rhs_norms = [record.rhs_norm for record in records]
        displacement_norms = [record.displacement_norm for record in records]
        residual_norms = [record.linear_residual_norm for record in records]
        residual_relatives = [record.linear_residual_relative for record in records]
        iterations = [record.ksp_iterations for record in records]

        final_displacement_norm = float(result.final_displacement.norm())
        previous_displacement_norm = float(result.previous_displacement.norm())

        expected_final_time = num_steps * dt
        expected_size = int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)

        checks = {
            "num_records": len(records) == num_steps,
            "times_finite": bool(np.all(np.isfinite(times))),
            "times_end_at_expected": bool(
                np.isclose(times[-1], expected_final_time, rtol=1.0e-14, atol=1.0e-14)
            ),
            "force_norms_finite_nonnegative": bool(
                np.all(np.isfinite(force_norms)) and min(force_norms) >= 0.0
            ),
            "some_force_is_nonzero": bool(max(force_norms) > 0.0),
            "rhs_norms_finite_nonnegative": bool(
                np.all(np.isfinite(rhs_norms)) and min(rhs_norms) >= 0.0
            ),
            "displacement_norms_finite_nonnegative": bool(
                np.all(np.isfinite(displacement_norms)) and min(displacement_norms) >= 0.0
            ),
            "some_displacement_is_nonzero": bool(max(displacement_norms) > 0.0),
            "residual_abs": bool(max(residual_norms) <= residual_atol),
            "residual_rel": bool(max(residual_relatives) <= residual_rtol),
            "all_ksp_converged": bool(all(record.ksp_converged for record in records)),
            "iterations_nonnegative": bool(min(iterations) >= 0),
            "final_norm_matches_last_record": bool(
                np.isclose(
                    final_displacement_norm,
                    displacement_norms[-1],
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            ),
            "final_vector_size": bool(result.final_displacement.getSize() == expected_size),
        }

        status_ok = bool(all(checks.values()))

        report = {
            "status": "ok" if status_ok else "failed",
            "mpi_size": comm.size,
            "mesh": {
                "kind": "unit_square",
                "nx": nx,
                "ny": ny,
                "topological_dimension": msh.topology.dim,
                "geometric_dimension": msh.geometry.dim,
            },
            "space": {
                "family": "Lagrange",
                "degree": degree,
                "global_size": expected_size,
                "block_size": int(V.dofmap.index_map_bs),
            },
            "operator_config": {
                "theta": float(operator_config.theta),
                "dt": float(operator_config.dt),
                "scale_theta_dt2": float(operator_config.theta * operator_config.dt**2),
            },
            "loop_config": {
                "num_steps": int(loop_config.num_steps),
                "start_time": float(loop_config.start_time),
                "final_time": float(expected_final_time),
            },
            "source": {
                "name": source.name,
                "tag": source.tag,
                "amplitude": float(source_amplitude),
                "frequency": float(source_frequency),
                "num_periods": float(source_num_periods),
            },
            "solve": {
                "ksp_type": solve_config.ksp_type,
                "pc_type": solve_config.pc_type,
                "rtol": solve_config.rtol,
                "atol": solve_config.atol,
                "max_it": solve_config.max_it,
            },
            "checks": checks,
            "values": {
                "max_force_norm": float(max(force_norms)),
                "max_rhs_norm": float(max(rhs_norms)),
                "max_displacement_norm": float(max(displacement_norms)),
                "final_displacement_norm": final_displacement_norm,
                "previous_displacement_norm": previous_displacement_norm,
                "max_linear_residual_norm": float(max(residual_norms)),
                "max_linear_residual_relative": float(max(residual_relatives)),
                "max_ksp_iterations": int(max(iterations)),
            },
            "records": record_dicts,
        }

    finally:
        result = locals().get("result")
        if result is not None:
            result.destroy()

        matrices.destroy()

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check a short direct Newmark time loop with Gamma1 traction source."
    )
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--source-amplitude", type=float, default=1.0)
    parser.add_argument("--source-frequency", type=float, default=1.0)
    parser.add_argument("--source-num-periods", type=float, default=1.0)
    parser.add_argument("--ksp-rtol", type=float, default=1.0e-10)
    parser.add_argument("--ksp-atol", type=float, default=1.0e-12)
    parser.add_argument("--ksp-max-it", type=int, default=1000)
    parser.add_argument("--residual-atol", type=float, default=1.0e-8)
    parser.add_argument("--residual-rtol", type=float, default=1.0e-8)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_time_loop_check(
        nx=args.nx,
        ny=args.ny,
        degree=args.degree,
        theta=args.theta,
        dt=args.dt,
        num_steps=args.num_steps,
        source_amplitude=args.source_amplitude,
        source_frequency=args.source_frequency,
        source_num_periods=args.source_num_periods,
        ksp_rtol=args.ksp_rtol,
        ksp_atol=args.ksp_atol,
        ksp_max_it=args.ksp_max_it,
        residual_atol=args.residual_atol,
        residual_rtol=args.residual_rtol,
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "OK" if report["status"] == "ok" else "FAILED"
            print(
                f"[TIME LOOP CHECK] {status} "
                f"for mesh={args.nx}x{args.ny}, degree={args.degree}, "
                f"theta={args.theta}, dt={args.dt}, steps={args.num_steps}, "
                f"MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
