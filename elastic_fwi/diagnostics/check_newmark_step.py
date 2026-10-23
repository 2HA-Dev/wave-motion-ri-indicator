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
from elastic_fwi.solvers.newmark import (
    LinearSolveConfig,
    create_zero_state_vector,
    linear_residual_norm,
    newmark_theta_step,
)
from elastic_fwi.sources.traction import (
    assemble_boundary_traction_vector,
    gamma1_vertical_traction_source,
)


def _relative_error(abs_error: float, reference_norm: float) -> float:
    denominator = max(float(reference_norm), 1.0)
    return float(abs_error / denominator)


def run_newmark_step_check(
    *,
    nx: int = 2,
    ny: int = 2,
    degree: int = 1,
    theta: float = 0.25,
    dt: float = 0.1,
    source_time: float = 0.25,
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

    msh = mesh.create_unit_square(comm, nx, ny)
    facet_tags = build_boundary_meshtags(
        msh,
        unit_square_pitch_catch_boundary_specs(),
    )

    V = create_vector_lagrange_space(msh, degree=degree)
    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)

    operator_config = ElasticOperatorConfig(theta=theta, dt=dt)
    solve_config = LinearSolveConfig(
        ksp_type="cg",
        pc_type="jacobi",
        rtol=ksp_rtol,
        atol=ksp_atol,
        max_it=ksp_max_it,
    )

    source = gamma1_vertical_traction_source(
        tag=1,
        amplitude=source_amplitude,
        frequency=source_frequency,
        t0=0.0,
        num_periods=source_num_periods,
    )

    matrices = assemble_elastic_operator_matrices(V, materials, operator_config)

    try:
        u_current = create_zero_state_vector(matrices)
        u_previous = create_zero_state_vector(matrices)

        force = assemble_boundary_traction_vector(
            V,
            facet_tags,
            source,
            t=source_time,
        )

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

        rhs_norm = float(result.rhs.norm())
        displacement_norm = float(result.displacement.norm())
        force_norm = float(force.norm())
        residual_rel = _relative_error(residual_norm, rhs_norm)

        expected_size = int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)

        checks = {
            "force_finite_positive": bool(np.isfinite(force_norm) and force_norm > 0.0),
            "rhs_finite_positive": bool(np.isfinite(rhs_norm) and rhs_norm > 0.0),
            "displacement_finite_positive": bool(
                np.isfinite(displacement_norm) and displacement_norm > 0.0
            ),
            "solve_converged": bool(result.solve_report.converged),
            "solve_residual_finite": bool(np.isfinite(result.solve_report.residual_norm)),
            "linear_residual_abs": bool(residual_norm <= residual_atol),
            "linear_residual_rel": bool(residual_rel <= residual_rtol),
            "vector_size": bool(result.displacement.getSize() == expected_size),
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
            "source": {
                "name": source.name,
                "tag": source.tag,
                "time": float(source_time),
                "amplitude": float(source_amplitude),
                "frequency": float(source_frequency),
                "num_periods": float(source_num_periods),
                "value": list(source.value(source_time)),
            },
            "solve": {
                "ksp_type": solve_config.ksp_type,
                "pc_type": solve_config.pc_type,
                "rtol": solve_config.rtol,
                "atol": solve_config.atol,
                "max_it": solve_config.max_it,
                "converged": result.solve_report.converged,
                "reason": result.solve_report.reason,
                "iterations": result.solve_report.iterations,
                "reported_residual_norm": result.solve_report.residual_norm,
            },
            "checks": checks,
            "values": {
                "force_norm": force_norm,
                "rhs_norm": rhs_norm,
                "displacement_norm": displacement_norm,
                "linear_residual_norm": residual_norm,
                "linear_residual_relative": residual_rel,
            },
        }

    finally:
        for name in ("u_current", "u_previous", "force"):
            vector = locals().get(name)
            if vector is not None:
                try:
                    vector.destroy()
                except Exception:
                    pass

        result = locals().get("result")
        if result is not None:
            result.destroy()

        matrices.destroy()

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check one Newmark-theta step with Gamma1 weak traction source."
    )
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--source-time", type=float, default=0.25)
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

    report = run_newmark_step_check(
        nx=args.nx,
        ny=args.ny,
        degree=args.degree,
        theta=args.theta,
        dt=args.dt,
        source_time=args.source_time,
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
                f"[NEWMARK STEP CHECK] {status} "
                f"for mesh={args.nx}x{args.ny}, degree={args.degree}, "
                f"theta={args.theta}, dt={args.dt}, MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
