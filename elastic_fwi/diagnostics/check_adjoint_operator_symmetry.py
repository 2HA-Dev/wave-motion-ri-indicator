from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from dolfinx import mesh
from mpi4py import MPI
from petsc4py import PETSc

from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    _build_materials,
)
from elastic_fwi.solvers.elastic_operator import (
    ElasticOperatorConfig,
    assemble_elastic_operator_matrices,
    create_vector_lagrange_space,
)


def _fill_deterministic_vector(vector: PETSc.Vec, *, phase: float) -> None:
    start, end = vector.getOwnershipRange()

    for gid in range(start, end):
        value = np.sin(0.37 * float(gid + 1) + phase) + 0.5 * np.cos(
            0.11 * float(gid + 1) + phase
        )
        vector.setValue(gid, float(value))

    vector.assemble()


def _matrix_duality_symmetry_report(
    *,
    name: str,
    matrix: PETSc.Mat,
    tolerance: float,
) -> dict[str, Any]:
    rows, cols = matrix.getSize()
    local_rows, local_cols = matrix.getLocalSize()

    x = matrix.createVecRight()
    y = matrix.createVecLeft()
    ax = matrix.createVecLeft()
    aty = matrix.createVecRight()

    try:
        _fill_deterministic_vector(x, phase=0.17)
        _fill_deterministic_vector(y, phase=0.41)

        matrix.mult(x, ax)
        matrix.multTranspose(y, aty)

        left = float(y.dot(ax))
        right = float(x.dot(aty))

        duality_abs = abs(left - right)
        duality_scale = max(abs(left), abs(right), 1.0)
        duality_rel = duality_abs / duality_scale

        matrix_norm = float(matrix.norm(PETSc.NormType.FROBENIUS))

        finite = bool(
            np.isfinite(left)
            and np.isfinite(right)
            and np.isfinite(duality_abs)
            and np.isfinite(duality_rel)
            and np.isfinite(matrix_norm)
        )

        symmetric = bool(finite and duality_abs <= tolerance * duality_scale)

        return {
            "name": name,
            "rows": int(rows),
            "cols": int(cols),
            "local_rows": int(local_rows),
            "local_cols": int(local_cols),
            "frobenius_norm": matrix_norm,
            "duality_left_y_Ax": left,
            "duality_right_x_ATy": right,
            "duality_abs": duality_abs,
            "duality_rel": duality_rel,
            "tolerance": float(tolerance),
            "finite": finite,
            "square": bool(rows == cols),
            "local_shape_valid": bool(local_rows >= 0 and local_cols >= 0),
            "symmetric_in_duality": symmetric,
        }

    finally:
        x.destroy()
        y.destroy()
        ax.destroy()
        aty.destroy()


def run_adjoint_operator_symmetry_check(
    *,
    nx: int = 4,
    ny: int = 4,
    degree: int = 1,
    theta: float = 0.25,
    dt: float = 0.1,
    material_kind: str = "healthy",
    contrast_factor: float = 0.8,
    defect_center_x: float = 0.5,
    defect_center_y: float = 0.5,
    defect_radius: float = 0.35,
    tolerance: float = 1.0e-10,
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
        num_steps=1,
        material_kind=material_kind,
        contrast_factor=contrast_factor,
        defect_center_x=defect_center_x,
        defect_center_y=defect_center_y,
        defect_radius=defect_radius,
    )

    materials, material_metadata = _build_materials(msh, direct_config)

    matrices = assemble_elastic_operator_matrices(
        V,
        materials,
        ElasticOperatorConfig(theta=theta, dt=dt),
    )

    try:
        mass_report = _matrix_duality_symmetry_report(
            name="mass",
            matrix=matrices.mass,
            tolerance=tolerance,
        )
        stiffness_report = _matrix_duality_symmetry_report(
            name="stiffness",
            matrix=matrices.stiffness,
            tolerance=tolerance,
        )
        effective_report = _matrix_duality_symmetry_report(
            name="effective",
            matrix=matrices.effective,
            tolerance=tolerance,
        )

        matrix_reports = {
            "mass": mass_report,
            "stiffness": stiffness_report,
            "effective": effective_report,
        }

        checks = {
            "mass_square": mass_report["square"],
            "stiffness_square": stiffness_report["square"],
            "effective_square": effective_report["square"],
            "mass_local_shape_valid": mass_report["local_shape_valid"],
            "stiffness_local_shape_valid": stiffness_report["local_shape_valid"],
            "effective_local_shape_valid": effective_report["local_shape_valid"],
            "mass_values_finite": mass_report["finite"],
            "stiffness_values_finite": stiffness_report["finite"],
            "effective_values_finite": effective_report["finite"],
            "mass_symmetric_in_duality": mass_report["symmetric_in_duality"],
            "stiffness_symmetric_in_duality": stiffness_report[
                "symmetric_in_duality"
            ],
            "effective_symmetric_in_duality": effective_report[
                "symmetric_in_duality"
            ],
            "effective_norm_positive": bool(effective_report["frobenius_norm"] > 0.0),
        }

        status_ok = bool(all(checks.values()))

        return {
            "kind": "adjoint_operator_symmetry_check",
            "status": "ok" if status_ok else "failed",
            "mpi_size": comm.size,
            "checks": checks,
            "config": {
                "nx": nx,
                "ny": ny,
                "degree": degree,
                "theta": theta,
                "dt": dt,
                "material_kind": material_kind,
                "contrast_factor": contrast_factor,
                "defect_center_x": defect_center_x,
                "defect_center_y": defect_center_y,
                "defect_radius": defect_radius,
                "tolerance": tolerance,
            },
            "material": material_metadata,
            "matrices": matrix_reports,
            "values": {
                "mass_duality_abs": mass_report["duality_abs"],
                "stiffness_duality_abs": stiffness_report["duality_abs"],
                "effective_duality_abs": effective_report["duality_abs"],
                "mass_duality_rel": mass_report["duality_rel"],
                "stiffness_duality_rel": stiffness_report["duality_rel"],
                "effective_duality_rel": effective_report["duality_rel"],
                "effective_frobenius_norm": effective_report["frobenius_norm"],
                "effective_rows": effective_report["rows"],
                "effective_cols": effective_report["cols"],
            },
        }

    finally:
        matrices.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check duality symmetry of mass, stiffness and effective operators."
    )
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--material-kind", type=str, default="healthy")
    parser.add_argument("--contrast-factor", type=float, default=0.8)
    parser.add_argument("--defect-center-x", type=float, default=0.5)
    parser.add_argument("--defect-center-y", type=float, default=0.5)
    parser.add_argument("--defect-radius", type=float, default=0.35)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_adjoint_operator_symmetry_check(
        nx=args.nx,
        ny=args.ny,
        degree=args.degree,
        theta=args.theta,
        dt=args.dt,
        material_kind=args.material_kind,
        contrast_factor=args.contrast_factor,
        defect_center_x=args.defect_center_x,
        defect_center_y=args.defect_center_y,
        defect_radius=args.defect_radius,
        tolerance=args.tolerance,
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "OK" if report["status"] == "ok" else "FAILED"
            values = report["values"]
            print(
                f"[ADJOINT OPERATOR SYMMETRY CHECK] {status} "
                f"for mesh={args.nx}x{args.ny}, degree={args.degree}, "
                f"theta={args.theta}, dt={args.dt}, material={args.material_kind}, "
                f"mass_abs={values['mass_duality_abs']:.6e}, "
                f"stiffness_abs={values['stiffness_duality_abs']:.6e}, "
                f"effective_abs={values['effective_duality_abs']:.6e}, "
                f"effective_rel={values['effective_duality_rel']:.6e}, "
                f"MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
