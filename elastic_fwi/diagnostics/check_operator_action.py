from __future__ import annotations

import argparse
import json
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
    matrix_action_norm,
    matrix_frobenius_norm,
    matrix_global_size,
)


def _relative_error(abs_error: float, reference_norm: float) -> float:
    denominator = max(float(reference_norm), 1.0)
    return float(abs_error / denominator)


def _effective_action_consistency_error(matrices, *, value: float = 1.0) -> dict[str, float]:
    """
    Check A x = M x + theta dt^2 K x for x = constant vector.
    """

    scale = matrices.config.theta * matrices.config.dt**2

    x = matrices.effective.createVecRight()
    y_effective = matrices.effective.createVecLeft()
    y_mass = matrices.mass.createVecLeft()
    y_stiffness = matrices.stiffness.createVecLeft()
    y_expected = matrices.effective.createVecLeft()

    x.set(value)

    matrices.effective.mult(x, y_effective)
    matrices.mass.mult(x, y_mass)
    matrices.stiffness.mult(x, y_stiffness)

    y_expected.set(0.0)
    y_expected.axpy(1.0, y_mass)
    y_expected.axpy(scale, y_stiffness)

    y_difference = y_effective.copy()
    y_difference.axpy(-1.0, y_expected)

    abs_error = float(y_difference.norm())
    reference_norm = float(y_effective.norm())
    rel_error = _relative_error(abs_error, reference_norm)

    x.destroy()
    y_effective.destroy()
    y_mass.destroy()
    y_stiffness.destroy()
    y_expected.destroy()
    y_difference.destroy()

    return {
        "abs_error": abs_error,
        "rel_error": rel_error,
        "reference_norm": reference_norm,
        "scale": float(scale),
    }


def run_operator_action_check(
    *,
    nx: int = 2,
    ny: int = 2,
    degree: int = 1,
    theta: float = 0.25,
    dt: float = 0.1,
    action_value: float = 1.0,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be strictly positive.")

    if degree <= 0:
        raise ValueError("degree must be strictly positive.")

    msh = mesh.create_unit_square(comm, nx, ny)
    V = create_vector_lagrange_space(msh, degree=degree)

    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)
    config = ElasticOperatorConfig(theta=theta, dt=dt)

    matrices = assemble_elastic_operator_matrices(V, materials, config)

    try:
        expected_size = int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)

        mass_size = matrix_global_size(matrices.mass)
        stiffness_size = matrix_global_size(matrices.stiffness)
        effective_size = matrix_global_size(matrices.effective)

        mass_norm = matrix_frobenius_norm(matrices.mass)
        stiffness_norm = matrix_frobenius_norm(matrices.stiffness)
        effective_norm = matrix_frobenius_norm(matrices.effective)

        mass_action_norm = matrix_action_norm(matrices.mass, value=action_value)
        stiffness_action_norm = matrix_action_norm(matrices.stiffness, value=action_value)
        effective_action_norm = matrix_action_norm(matrices.effective, value=action_value)

        consistency = _effective_action_consistency_error(
            matrices,
            value=action_value,
        )

    finally:
        matrices.destroy()

    checks = {
        "mass_size": mass_size == (expected_size, expected_size),
        "stiffness_size": stiffness_size == (expected_size, expected_size),
        "effective_size": effective_size == (expected_size, expected_size),
        "mass_norm_finite_positive": bool(np.isfinite(mass_norm) and mass_norm > 0.0),
        "stiffness_norm_finite_positive": bool(
            np.isfinite(stiffness_norm) and stiffness_norm > 0.0
        ),
        "effective_norm_finite_positive": bool(
            np.isfinite(effective_norm) and effective_norm > 0.0
        ),
        "mass_action_finite_positive": bool(
            np.isfinite(mass_action_norm) and mass_action_norm > 0.0
        ),
        "stiffness_action_finite": bool(np.isfinite(stiffness_action_norm)),
        "effective_action_finite_positive": bool(
            np.isfinite(effective_action_norm) and effective_action_norm > 0.0
        ),
        "effective_action_consistency_abs": bool(consistency["abs_error"] <= atol),
        "effective_action_consistency_rel": bool(consistency["rel_error"] <= rtol),
    }

    status_ok = bool(all(checks.values()))

    return {
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
        "config": {
            "theta": float(config.theta),
            "dt": float(config.dt),
            "scale_theta_dt2": float(config.theta * config.dt**2),
        },
        "checks": checks,
        "values": {
            "mass_size": list(mass_size),
            "stiffness_size": list(stiffness_size),
            "effective_size": list(effective_size),
            "mass_frobenius_norm": mass_norm,
            "stiffness_frobenius_norm": stiffness_norm,
            "effective_frobenius_norm": effective_norm,
            "mass_action_norm": mass_action_norm,
            "stiffness_action_norm": stiffness_action_norm,
            "effective_action_norm": effective_action_norm,
            "effective_action_consistency": consistency,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check assembled elastic operator matrix actions."
    )
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--action-value", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_operator_action_check(
        nx=args.nx,
        ny=args.ny,
        degree=args.degree,
        theta=args.theta,
        dt=args.dt,
        action_value=args.action_value,
        rtol=args.rtol,
        atol=args.atol,
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "OK" if report["status"] == "ok" else "FAILED"
            print(
                f"[OPERATOR ACTION CHECK] {status} "
                f"for mesh={args.nx}x{args.ny}, degree={args.degree}, "
                f"theta={args.theta}, dt={args.dt}, MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
