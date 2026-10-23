from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from mpi4py import MPI

from elastic_fwi.inversion.material_directions import (
    HomogeneousMaterialDirection,
    assemble_homogeneous_material_direction_matrices,
)
from elastic_fwi.inversion.material_gradient import (
    compute_newmark_material_directional_gradient,
)
from elastic_fwi.gradient_observation_modes import (
    TRUE_GRADIENT_OBSERVATION_MODES,
    resolve_gradient_observation_mode,
)
from elastic_fwi.pipelines.adjoint_newmark import (
    AdjointNewmarkPipelineConfig,
    run_adjoint_newmark_pipeline,
)


def run_material_gradient_directional_check(
    *,
    nx: int = 4,
    ny: int = 4,
    degree: int = 1,
    theta: float = 0.25,
    dt: float = 0.1,
    num_steps: int = 5,
    observation_mode: str = "classical_full",
    simulated_material_kind: str = "healthy",
    observed_material_kind: str = "localized_disk_defect",
    contrast_factor: float = 0.8,
    defect_center_x: float = 0.5,
    defect_center_y: float = 0.5,
    defect_radius: float = 0.35,
    direction_rho: float = 0.0,
    direction_C11: float = 1.0,
    direction_C22: float = 0.0,
    direction_C12: float = 0.0,
    direction_C66: float = 0.0,
    ksp_type: str = "cg",
    pc_type: str = "jacobi",
    ksp_rtol: float = 1.0e-10,
    ksp_atol: float = 1.0e-12,
    ksp_max_it: int = 1000,
    reuse_direct_effective_solver: bool = True,
    reuse_adjoint_effective_solver: bool = True,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    mode = resolve_gradient_observation_mode(observation_mode)
    pipeline = None
    perturbation = None

    try:
        pipeline = run_adjoint_newmark_pipeline(
            AdjointNewmarkPipelineConfig(
                nx=nx,
                ny=ny,
                degree=degree,
                theta=theta,
                dt=dt,
                num_steps=num_steps,
                reflection_weight=mode.reflection_weight,
                transmission_weight=mode.transmission_weight,
                simulated_material_kind=simulated_material_kind,
                observed_material_kind=observed_material_kind,
                contrast_factor=contrast_factor,
                defect_center_x=defect_center_x,
                defect_center_y=defect_center_y,
                defect_radius=defect_radius,
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

        direction = HomogeneousMaterialDirection(
            rho=direction_rho,
            C11=direction_C11,
            C22=direction_C22,
            C12=direction_C12,
            C66=direction_C66,
        )

        V = pipeline.comparison.simulated.function_space
        operator_config = pipeline.comparison.simulated.operator_config

        if V is None:
            raise RuntimeError("Simulated pipeline result does not expose function_space.")

        if operator_config is None:
            raise RuntimeError("Simulated pipeline result does not expose operator_config.")

        perturbation = assemble_homogeneous_material_direction_matrices(
            V,
            direction,
        )

        gradient = compute_newmark_material_directional_gradient(
            perturbation.mass,
            perturbation.stiffness,
            operator_config,
            direct_snapshots=pipeline.comparison.simulated.displacement_snapshots,
            equation_adjoint_vectors=pipeline.newmark_adjoint.equation_adjoint_vectors,
        )

        pipeline_report = pipeline.as_dict()
        perturbation_report = perturbation.as_dict()
        gradient_report = gradient.as_dict()

        direction_values = direction.as_dict()
        direction_nonzero = bool(
            any(abs(value) > 0.0 for value in direction_values.values())
        )

        checks = {
            "pipeline_status_ok": pipeline_report["comparison"]["status"] == "ok",
            "simulated_snapshots_stored": pipeline_report["comparison"]["simulated"][
                "state"
            ]["displacement_snapshots_stored"],
            "snapshot_count_matches_num_steps": pipeline_report["comparison"][
                "simulated"
            ]["state"]["num_displacement_snapshots"]
            == num_steps,
            "adjoint_num_steps_match": pipeline_report["newmark_adjoint"][
                "num_steps"
            ]
            == num_steps,
            "gradient_num_steps_match": gradient_report["num_steps"] == num_steps,
            "direction_nonzero": direction_nonzero,
            "perturbation_mass_shape_valid": perturbation_report["mass"]["rows"]
            == perturbation_report["mass"]["cols"],
            "perturbation_stiffness_shape_valid": perturbation_report["stiffness"][
                "rows"
            ]
            == perturbation_report["stiffness"]["cols"],
            "gradient_value_finite": bool(np.isfinite(gradient_report["value"])),
            "gradient_records_finite": bool(
                all(
                    np.isfinite(record["contribution"])
                    and np.isfinite(record["forcing_norm"])
                    and np.isfinite(record["adjoint_norm"])
                    for record in gradient_report["records"]
                )
            ),
        }

        if direction_nonzero:
            checks["directional_forcing_nonzero"] = bool(
                gradient_report["max_forcing_norm"] > 0.0
            )

        status_ok = bool(all(checks.values()))

        return {
            "kind": "material_gradient_directional_check",
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
                "observation_mode": mode.name,
                "observation_mode_definition": mode.as_dict(),
                "simulated_material_kind": simulated_material_kind,
                "observed_material_kind": observed_material_kind,
                "contrast_factor": contrast_factor,
                "defect_center_x": defect_center_x,
                "defect_center_y": defect_center_y,
                "defect_radius": defect_radius,
                "direction": direction_values,
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": ksp_rtol,
                "ksp_atol": ksp_atol,
                "ksp_max_it": ksp_max_it,
                "reuse_direct_effective_solver": reuse_direct_effective_solver,
                "reuse_adjoint_effective_solver": reuse_adjoint_effective_solver,
            },
            "values": {
                "gradient_value": gradient_report["value"],
                "gradient_abs": abs(gradient_report["value"]),
                "max_abs_contribution": gradient_report["max_abs_contribution"],
                "max_forcing_norm": gradient_report["max_forcing_norm"],
                "max_adjoint_norm": pipeline_report["values"][
                    "max_equation_adjoint_norm"
                ],
                "max_observation_residual": pipeline_report["values"][
                    "max_abs_residual"
                ],
            },
            "direction": direction_values,
            "perturbation": perturbation_report,
            "gradient": gradient_report,
            "pipeline": pipeline_report,
        }

    finally:
        if perturbation is not None:
            perturbation.destroy()

        if pipeline is not None:
            pipeline.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check material directional gradient on the Newmark adjoint pipeline."
    )

    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument(
        "--observation-mode",
        choices=TRUE_GRADIENT_OBSERVATION_MODES,
        default="classical_full",
    )
    parser.add_argument("--simulated-material-kind", type=str, default="healthy")
    parser.add_argument("--observed-material-kind", type=str, default="localized_disk_defect")
    parser.add_argument("--contrast-factor", type=float, default=0.8)
    parser.add_argument("--defect-center-x", type=float, default=0.5)
    parser.add_argument("--defect-center-y", type=float, default=0.5)
    parser.add_argument("--defect-radius", type=float, default=0.35)

    parser.add_argument("--direction-rho", type=float, default=0.0)
    parser.add_argument("--direction-C11", type=float, default=1.0)
    parser.add_argument("--direction-C22", type=float, default=0.0)
    parser.add_argument("--direction-C12", type=float, default=0.0)
    parser.add_argument("--direction-C66", type=float, default=0.0)

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

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_material_gradient_directional_check(
        nx=args.nx,
        ny=args.ny,
        degree=args.degree,
        theta=args.theta,
        dt=args.dt,
        num_steps=args.num_steps,
        observation_mode=args.observation_mode,
        simulated_material_kind=args.simulated_material_kind,
        observed_material_kind=args.observed_material_kind,
        contrast_factor=args.contrast_factor,
        defect_center_x=args.defect_center_x,
        defect_center_y=args.defect_center_y,
        defect_radius=args.defect_radius,
        direction_rho=args.direction_rho,
        direction_C11=args.direction_C11,
        direction_C22=args.direction_C22,
        direction_C12=args.direction_C12,
        direction_C66=args.direction_C66,
        ksp_type=args.ksp_type,
        pc_type=args.pc_type,
        ksp_rtol=args.ksp_rtol,
        ksp_atol=args.ksp_atol,
        ksp_max_it=args.ksp_max_it,
        reuse_direct_effective_solver=args.reuse_effective_solver,
        reuse_adjoint_effective_solver=args.reuse_effective_solver,
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "OK" if report["status"] == "ok" else "FAILED"
            print(
                f"[MATERIAL GRADIENT DIRECTIONAL CHECK] {status} "
                f"for mesh={args.nx}x{args.ny}, degree={args.degree}, "
                f"theta={args.theta}, dt={args.dt}, steps={args.num_steps}, "
                f"mode={args.observation_mode}, "
                f"gradient={report['values']['gradient_value']:.6e}, "
                f"max_forcing={report['values']['max_forcing_norm']:.6e}, "
                f"MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
