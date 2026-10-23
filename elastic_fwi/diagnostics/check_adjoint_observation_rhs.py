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
from elastic_fwi.inversion.observation_functionals import ObservationFunctionalWeights
from elastic_fwi.pipelines.direct_observation import DirectObservationPipelineConfig
from elastic_fwi.pipelines.observation_comparison import (
    DirectObservationComparisonConfig,
    run_direct_observation_comparison,
)


def _records_have_finite_norms(records: list[dict[str, Any]]) -> bool:
    return bool(
        all(
            np.isfinite(record["rhs_norm"])
            and np.isfinite(record["residual_value"])
            for record in records
        )
    )


def _records_have_valid_sizes(records: list[dict[str, Any]]) -> bool:
    return bool(
        all(
            record["rhs_size_global"] > 0
            and record["rhs_size_local"] >= 0
            for record in records
        )
    )


def run_adjoint_observation_rhs_check(
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
        store_displacement_snapshots=False,
    )

    comparison = None
    rhs = None

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

        rhs = assemble_full_adjoint_observation_rhs(
            V,
            facet_tags,
            comparison.simulated.displacement_snapshots,
            comparison.residuals,
            reflection_scale=reflection_weight,
            transmission_scale=transmission_weight,
        )

        comparison_report = comparison.as_dict()
        rhs_report = rhs.as_dict()

        reflection_records = rhs_report["reflection"]["records"]
        transmission_records = rhs_report["transmission"]["records"]

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
            "residual_functionals_match": bool(
                comparison_report["checks"]["residual_functionals_match"]
            ),
            "rhs_num_steps_match": rhs.num_steps == num_steps,
            "reflection_rhs_num_steps_match": rhs.reflection.num_steps == num_steps,
            "transmission_rhs_num_steps_match": rhs.transmission.num_steps
            == num_steps,
            "rhs_norms_finite": bool(
                np.isfinite(rhs.max_rhs_norm)
                and _records_have_finite_norms(reflection_records)
                and _records_have_finite_norms(transmission_records)
            ),
            "rhs_sizes_valid": bool(
                _records_have_valid_sizes(reflection_records)
                and _records_have_valid_sizes(transmission_records)
            ),
            "rhs_nonzero": bool(rhs.max_rhs_norm > 0.0),
            "reflection_rhs_nonzero": bool(rhs.reflection.max_rhs_norm > 0.0),
            "residual_nonzero": bool(comparison.residuals.max_abs_residual > 0.0),
        }

        if observed_material_kind == "localized_disk_defect":
            checks["observed_defect_cells_positive"] = bool(observed_defect_cells > 0)

        status_ok = bool(all(checks.values()))

        return {
            "kind": "adjoint_observation_rhs_check",
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
            },
            "values": {
                "max_rhs_norm": rhs.max_rhs_norm,
                "reflection_max_rhs_norm": rhs.reflection.max_rhs_norm,
                "transmission_max_rhs_norm": rhs.transmission.max_rhs_norm,
                "max_abs_residual": comparison.residuals.max_abs_residual,
                "observed_defect_cells": observed_defect_cells,
                "max_abs_observation_difference": comparison_report["values"][
                    "max_abs_observation_difference"
                ],
            },
            "adjoint_rhs": rhs_report,
            "comparison": comparison_report,
        }

    finally:
        if rhs is not None:
            rhs.destroy()

        if comparison is not None:
            comparison.simulated.destroy()
            comparison.observed.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check adjoint-compatible observation RHS assembly."
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
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_adjoint_observation_rhs_check(
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
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "OK" if report["status"] == "ok" else "FAILED"
            print(
                f"[ADJOINT OBSERVATION RHS CHECK] {status} "
                f"for mesh={args.nx}x{args.ny}, degree={args.degree}, "
                f"theta={args.theta}, dt={args.dt}, steps={args.num_steps}, "
                f"simulated={args.simulated_material_kind}, "
                f"observed={args.observed_material_kind}, "
                f"max_residual={report['values']['max_abs_residual']:.6e}, "
                f"rhs_ref={report['values']['reflection_max_rhs_norm']:.6e}, "
                f"rhs_trans={report['values']['transmission_max_rhs_norm']:.6e}, "
                f"rhs_max={report['values']['max_rhs_norm']:.6e}, "
                f"MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
