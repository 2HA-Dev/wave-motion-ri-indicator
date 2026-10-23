from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from mpi4py import MPI

from elastic_fwi.inversion.material_imaging import (
    compute_material_qualitative_dof_image,
)
from elastic_fwi.pipelines.adjoint_newmark import (
    AdjointNewmarkPipelineConfig,
    run_adjoint_newmark_pipeline,
)


def run_material_qualitative_image_check(
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
    ksp_type: str = "cg",
    pc_type: str = "jacobi",
    ksp_rtol: float = 1.0e-10,
    ksp_atol: float = 1.0e-12,
    ksp_max_it: int = 1000,
    normalize: bool = True,
    reuse_direct_effective_solver: bool = True,
    reuse_adjoint_effective_solver: bool = True,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    pipeline = run_adjoint_newmark_pipeline(
        AdjointNewmarkPipelineConfig(
            nx=nx,
            ny=ny,
            degree=degree,
            theta=theta,
            dt=dt,
            num_steps=num_steps,
            simulated_material_kind=simulated_material_kind,
            observed_material_kind=observed_material_kind,
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
            reuse_direct_effective_solver=reuse_direct_effective_solver,
            reuse_adjoint_effective_solver=reuse_adjoint_effective_solver,
        ),
        comm=comm,
    )

    try:
        image = compute_material_qualitative_dof_image(
            direct_snapshots=pipeline.comparison.simulated.displacement_snapshots,
            equation_adjoint_vectors=pipeline.newmark_adjoint.equation_adjoint_vectors,
            function_space=pipeline.comparison.simulated.function_space,
            comm=comm,
            absolute_value=True,
            normalize=normalize,
        )

        pipeline_report = pipeline.as_dict()

        max_residual = float(pipeline_report["values"]["max_abs_residual"])
        max_adjoint_norm = float(pipeline_report["values"]["max_equation_adjoint_norm"])

        matrix_rows = int(pipeline_report["matrix"]["rows"])
        matrix_local_rows = int(pipeline_report["matrix"]["local_rows"])

        has_coordinates = image.local_coordinates is not None

        coordinates_shape_ok = (
            has_coordinates
            and image.local_coordinates is not None
            and image.local_coordinates.ndim == 2
            and image.local_coordinates.shape[0] == image.local_size
            and image.local_coordinates.shape[1] == 2
        )

        coordinates_finite = (
            has_coordinates
            and image.local_coordinates is not None
            and bool(np.all(np.isfinite(image.local_coordinates)))
        )

        checks = {
            "pipeline_status_ok": pipeline_report["comparison"]["status"] == "ok",
            "simulated_snapshots_stored": bool(
                pipeline_report["comparison"]["simulated"]["state"][
                    "displacement_snapshots_stored"
                ]
            ),
            "snapshot_count_matches_num_steps": int(
                pipeline_report["comparison"]["simulated"]["state"][
                    "num_displacement_snapshots"
                ]
            )
            == num_steps,
            "adjoint_num_steps_match": int(
                pipeline_report["newmark_adjoint"]["num_steps"]
            )
            == num_steps,
            "image_num_steps_match": image.num_steps == num_steps,
            "image_local_size_matches_matrix": image.local_size == matrix_local_rows,
            "image_global_size_matches_matrix": image.global_size == matrix_rows,
            "image_values_finite": bool(np.all(np.isfinite(image.local_values))),
            "image_global_l2_finite": bool(np.isfinite(image.global_l2_norm)),
            "image_global_l2_positive": bool(image.global_l2_norm > 0.0),
            "image_global_max_positive": bool(image.global_max > 0.0),
            "image_global_sum_positive": bool(image.global_sum > 0.0),
            "observation_residual_nonzero": bool(max_residual > 0.0),
            "adjoint_nonzero": bool(max_adjoint_norm > 0.0),
            "image_has_coordinates": bool(has_coordinates),
            "image_coordinates_shape_ok": bool(coordinates_shape_ok),
            "image_coordinates_finite": bool(coordinates_finite),
        }

        status_ok = bool(all(checks.values()))

        return {
            "kind": "material_qualitative_image_check",
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
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": ksp_rtol,
                "ksp_atol": ksp_atol,
                "ksp_max_it": ksp_max_it,
                "normalize": normalize,
                "reuse_direct_effective_solver": reuse_direct_effective_solver,
                "reuse_adjoint_effective_solver": reuse_adjoint_effective_solver,
            },
            "values": {
                "image_global_min": image.global_min,
                "image_global_max": image.global_max,
                "image_global_l2_norm": image.global_l2_norm,
                "image_global_sum": image.global_sum,
                "image_coordinate_dimension": image.coordinate_dimension,
                "max_observation_residual": max_residual,
                "max_adjoint_norm": max_adjoint_norm,
            },
            "image": image.as_dict(include_values=False),
        }

    finally:
        pipeline.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check qualitative material image construction."
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

    parser.add_argument("--ksp-type", type=str, default="cg")
    parser.add_argument("--pc-type", type=str, default="jacobi")
    parser.add_argument("--ksp-rtol", type=float, default=1.0e-10)
    parser.add_argument("--ksp-atol", type=float, default=1.0e-12)
    parser.add_argument("--ksp-max-it", type=int, default=1000)

    reuse_group = parser.add_mutually_exclusive_group()
    reuse_group.add_argument(
        "--reuse-effective-solver",
        dest="reuse_effective_solver",
        action="store_true",
        default=True,
        help="Reuse direct and adjoint effective Newmark solvers. Default: enabled.",
    )
    reuse_group.add_argument(
        "--no-reuse-effective-solver",
        dest="reuse_effective_solver",
        action="store_false",
        help="Use legacy one-shot effective solvers in direct and adjoint paths.",
    )

    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_material_qualitative_image_check(
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
        ksp_type=args.ksp_type,
        pc_type=args.pc_type,
        ksp_rtol=args.ksp_rtol,
        ksp_atol=args.ksp_atol,
        ksp_max_it=args.ksp_max_it,
        normalize=not args.no_normalize,
        reuse_direct_effective_solver=args.reuse_effective_solver,
        reuse_adjoint_effective_solver=args.reuse_effective_solver,
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(
                "[MATERIAL QUALITATIVE IMAGE CHECK] "
                f"{report['status'].upper()} "
                f"for N=({args.nx},{args.ny}), degree={args.degree}, "
                f"steps={args.num_steps}, MPI size={comm.size}, "
                f"image_l2={report['values']['image_global_l2_norm']:.6e}, "
                f"image_max={report['values']['image_global_max']:.6e}, "
                f"coordinate_dimension={report['values']['image_coordinate_dimension']}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
