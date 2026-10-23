from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from dolfinx import mesh
from mpi4py import MPI
from petsc4py import PETSc

from elastic_fwi.materials.dg0_fields import homogeneous_dg0_material_functions
from elastic_fwi.materials.orthotropic import LEGACY_HEALTHY_MATERIAL
from elastic_fwi.solvers.elastic_operator import (
    ElasticOperatorConfig,
    assemble_elastic_operator_matrices,
    create_vector_lagrange_space,
)
from elastic_fwi.solvers.newmark import linear_residual_norm, solve_effective_system
from elastic_fwi.solvers.solver_presets import (
    get_solver_preset,
    list_solver_presets,
)


def _global_status(comm: MPI.Comm, ok: bool) -> bool:
    return bool(comm.allreduce(bool(ok), op=MPI.LAND))


def _create_test_system(*, nx: int, ny: int, dt: float, theta: float):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
    V = create_vector_lagrange_space(msh, degree=1)
    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)
    config = ElasticOperatorConfig(theta=theta, dt=dt)
    matrices = assemble_elastic_operator_matrices(V, materials, config)

    rhs = matrices.effective.createVecRight()
    rhs.set(1.0)
    rhs.assemble()

    return matrices, rhs


def _check_one_preset(
    *,
    preset_name: str,
    nx: int,
    ny: int,
    dt: float,
    theta: float,
    residual_tol: float,
) -> dict[str, Any]:
    comm = MPI.COMM_WORLD
    preset = get_solver_preset(preset_name)
    matrices = None
    rhs = None
    solution = None

    try:
        matrices, rhs = _create_test_system(nx=nx, ny=ny, dt=dt, theta=theta)
        solution, report = solve_effective_system(
            matrices.effective,
            rhs,
            preset.config,
        )

        residual = linear_residual_norm(matrices.effective, solution, rhs)
        local_ok = bool(report.converged and np.isfinite(residual) and residual <= residual_tol)
        global_ok = _global_status(comm, local_ok)

        status = "ok" if global_ok else "failed"

        return {
            "preset": preset.as_dict(),
            "status": status,
            "converged": bool(report.converged),
            "reason": int(report.reason),
            "iterations": int(report.iterations),
            "ksp_residual_norm": float(report.residual_norm),
            "true_residual_norm": float(residual),
            "residual_tol": float(residual_tol),
            "mpi_size": int(comm.size),
        }

    except PETSc.Error as exc:
        # Optional PETSc external packages, especially MUMPS/HYPRE, may be absent.
        optional = preset.optional_dependency is not None
        return {
            "preset": preset.as_dict(),
            "status": "skipped_unavailable" if optional else "failed_petsc_error",
            "optional": optional,
            "error": str(exc),
            "mpi_size": int(comm.size),
        }

    finally:
        if solution is not None:
            solution.destroy()
        if rhs is not None:
            rhs.destroy()
        if matrices is not None:
            matrices.destroy()


def check_solver_preconditioners(
    *,
    presets: list[str],
    nx: int = 4,
    ny: int = 4,
    dt: float = 0.1,
    theta: float = 0.25,
    residual_tol: float = 1.0e-8,
) -> dict[str, Any]:
    results = [
        _check_one_preset(
            preset_name=name,
            nx=nx,
            ny=ny,
            dt=dt,
            theta=theta,
            residual_tol=residual_tol,
        )
        for name in presets
    ]

    requested = set(presets)

    ok_statuses = {"ok", "skipped_unavailable"}
    nonblocking_failed_presets = {
        # ILU is often sequential/local and may fail in MPI depending on the
        # PETSc build.  It is kept as a small-case candidate but
        # must not invalidate the global solver contract when robust presets
        # pass.
        "gmres_ilu",
    }

    hard_failures = []
    nonblocking_failures = []

    for result in results:
        name = result["preset"]["name"]
        status = result["status"]

        if status in ok_statuses:
            continue

        if name in nonblocking_failed_presets:
            nonblocking_failures.append(result)
        else:
            hard_failures.append(result)

    if "default_cg_jacobi" in requested:
        default_ok = any(
            result["preset"]["name"] == "default_cg_jacobi"
            and result["status"] == "ok"
            for result in results
        )
    else:
        default_ok = True

    global_ok = default_ok and not hard_failures

    return {
        "kind": "solver_preconditioner_diagnostic",
        "status": "ok" if global_ok else "failed",
        "default_solver_contract": {
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "mathematical_operator": "A = M + theta * dt^2 * K",
            "role": "repository default baseline",
        },
        "tested_presets": presets,
        "results": results,
        "hard_failures": hard_failures,
        "nonblocking_failures": nonblocking_failures,
        "mpi_size": int(MPI.COMM_WORLD.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--presets",
        type=str,
        default="default_cg_jacobi,cg_gamg,cg_hypre,gmres_ilu,preonly_lu,preonly_lu_mumps",
        help=f"Comma-separated presets. Available: {','.join(list_solver_presets())}",
    )
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--residual-tol", type=float, default=1.0e-8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    presets = [item.strip() for item in args.presets.split(",") if item.strip()]

    report = check_solver_preconditioners(
        presets=presets,
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        theta=args.theta,
        residual_tol=args.residual_tol,
    )

    comm = MPI.COMM_WORLD
    if args.output is not None and comm.rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))

    if comm.rank == 0 and not args.quiet:
        print("[SOLVER PRECONDITIONER CHECK]", report["status"])
        for result in report["results"]:
            preset = result["preset"]["name"]
            status = result["status"]
            iterations = result.get("iterations", "n/a")
            residual = result.get("true_residual_norm", "n/a")
            print(f"  {preset}: {status}, iterations={iterations}, residual={residual}")

    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
