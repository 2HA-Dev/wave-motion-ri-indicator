from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from mpi4py import MPI

from elastic_fwi.pipelines.adjoint_newmark import (
    AdjointNewmarkPipelineConfig,
    run_adjoint_newmark_pipeline,
)


def _records_have_finite_values(records: list[dict[str, Any]]) -> bool:
    return bool(
        all(
            np.isfinite(record["observation_rhs_norm"])
            and np.isfinite(record["accumulated_rhs_norm"])
            and np.isfinite(record["equation_adjoint_norm"])
            and np.isfinite(record["carry_current_norm"])
            and np.isfinite(record["carry_previous_norm"])
            and np.isfinite(record["linear_residual_norm"])
            and np.isfinite(record["linear_residual_relative"])
            and np.isfinite(record["ksp_reported_residual_norm"])
            for record in records
        )
    )


def _records_are_reverse_ordered(records: list[dict[str, Any]]) -> bool:
    if not records:
        return False

    num_steps = len(records)

    return bool(
        records[0]["step"] == 1
        and records[0]["solve_index"] == num_steps
        and records[-1]["step"] == num_steps
        and records[-1]["solve_index"] == 1
    )


def _vectors_have_valid_sizes(vectors, *, expected_size: int) -> bool:
    return bool(
        all(
            vector.getSize() == expected_size and vector.getLocalSize() >= 0
            for vector in vectors
        )
    )


def run_adjoint_newmark_pipeline_check(
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
    result = None

    try:
        result = run_adjoint_newmark_pipeline(
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
                reflection_weight=reflection_weight,
                transmission_weight=transmission_weight,
                ksp_type=ksp_type,
                pc_type=pc_type,
                ksp_rtol=ksp_rtol,
                ksp_atol=ksp_atol,
                ksp_max_it=ksp_max_it,
            ),
            comm=comm,
        )

        pipeline_report = result.as_dict()

        comparison = pipeline_report["comparison"]
        rhs = pipeline_report["adjoint_rhs"]
        newmark = pipeline_report["newmark_adjoint"]
        matrix = pipeline_report["matrix"]
        records = newmark["records"]

        observed_defect_cells = int(
            comparison["values"]["observed_defect_cells"]
        )

        checks = {
            "comparison_status_ok": comparison["status"] == "ok",
            "simulated_snapshots_stored": comparison["simulated"]["state"][
                "displacement_snapshots_stored"
            ],
            "snapshot_count_matches_num_steps": comparison["simulated"]["state"][
                "num_displacement_snapshots"
            ]
            == num_steps,
            "residuals_have_expected_length": comparison["residuals"]["num_steps"]
            == num_steps,
            "rhs_num_steps_match": rhs["num_steps"] == num_steps,
            "rhs_nonzero": bool(rhs["max_rhs_norm"] > 0.0),
            "newmark_num_steps_match": newmark["num_steps"] == num_steps,
            "newmark_reverse_time": newmark["reverse_time"] is True,
            "newmark_records_reverse_ordered": _records_are_reverse_ordered(
                records
            ),
            "all_ksp_converged": newmark["all_ksp_converged"],
            "newmark_values_finite": _records_have_finite_values(records),
            "equation_adjoint_vectors_valid_sizes": _vectors_have_valid_sizes(
                result.newmark_adjoint.equation_adjoint_vectors,
                expected_size=int(matrix["cols"]),
            ),
            "equation_adjoint_nonzero": bool(
                newmark["max_equation_adjoint_norm"] > 0.0
            ),
            "initial_current_adjoint_finite": bool(
                np.isfinite(newmark["initial_current_adjoint_norm"])
            ),
            "initial_previous_adjoint_finite": bool(
                np.isfinite(newmark["initial_previous_adjoint_norm"])
            ),
            "linear_residuals_below_tolerance": bool(
                newmark["max_linear_residual_norm"]
                <= float(max_linear_residual_tolerance)
            ),
            "matrix_global_shape_square": bool(matrix["rows"] == matrix["cols"]),
            "matrix_local_shape_valid": bool(
                matrix["local_rows"] >= 0 and matrix["local_cols"] >= 0
            ),
            "pipeline_values_finite": bool(
                np.isfinite(pipeline_report["values"]["max_abs_residual"])
                and np.isfinite(pipeline_report["values"]["max_rhs_norm"])
                and np.isfinite(
                    pipeline_report["values"]["max_equation_adjoint_norm"]
                )
                and np.isfinite(
                    pipeline_report["values"]["max_linear_residual_norm"]
                )
            ),
        }

        if observed_material_kind == "localized_disk_defect":
            checks["observed_defect_cells_positive"] = bool(
                observed_defect_cells > 0
            )

        status_ok = bool(all(checks.values()))

        return {
            "kind": "adjoint_newmark_pipeline_check",
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
                "matrix_rows": int(matrix["rows"]),
                "matrix_cols": int(matrix["cols"]),
                "matrix_local_rows": int(matrix["local_rows"]),
                "matrix_local_cols": int(matrix["local_cols"]),
                "max_abs_residual": pipeline_report["values"][
                    "max_abs_residual"
                ],
                "max_rhs_norm": pipeline_report["values"]["max_rhs_norm"],
                "max_equation_adjoint_norm": pipeline_report["values"][
                    "max_equation_adjoint_norm"
                ],
                "max_linear_residual_norm": pipeline_report["values"][
                    "max_linear_residual_norm"
                ],
                "initial_current_adjoint_norm": pipeline_report["values"][
                    "initial_current_adjoint_norm"
                ],
                "initial_previous_adjoint_norm": pipeline_report["values"][
                    "initial_previous_adjoint_norm"
                ],
                "observed_defect_cells": observed_defect_cells,
                "max_abs_observation_difference": comparison["values"][
                    "max_abs_observation_difference"
                ],
            },
            "material": pipeline_report["material"],
            "matrix": pipeline_report["matrix"],
            "comparison": comparison,
            "adjoint_rhs": rhs,
            "newmark_adjoint": newmark,
            "poo_pipeline": pipeline_report,
        }

    finally:
        if result is not None:
            result.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the full discrete Newmark adjoint pipeline."
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

    report = run_adjoint_newmark_pipeline_check(
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
                f"[ADJOINT NEWMARK PIPELINE CHECK] {status} "
                f"for mesh={args.nx}x{args.ny}, degree={args.degree}, "
                f"theta={args.theta}, dt={args.dt}, steps={args.num_steps}, "
                f"simulated={args.simulated_material_kind}, "
                f"observed={args.observed_material_kind}, "
                f"max_residual={report['values']['max_abs_residual']:.6e}, "
                f"rhs_max={report['values']['max_rhs_norm']:.6e}, "
                f"adjoint_max={report['values']['max_equation_adjoint_norm']:.6e}, "
                f"linear_residual_max={report['values']['max_linear_residual_norm']:.6e}, "
                f"MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
