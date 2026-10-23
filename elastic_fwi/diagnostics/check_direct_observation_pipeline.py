from __future__ import annotations

import argparse
import json
from typing import Any

from mpi4py import MPI

from elastic_fwi.inversion.observation_functionals import (
    ObservationFunctionalWeights,
)
from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
)
from elastic_fwi.pipelines.observation_comparison import (
    DirectObservationComparisonConfig,
    run_direct_observation_comparison,
)


def _make_direct_config(
    *,
    nx: int,
    ny: int,
    degree: int,
    theta: float,
    dt: float,
    num_steps: int,
    material_kind: str,
    contrast_factor: float,
    defect_center_x: float,
    defect_center_y: float,
    defect_radius: float,
    source_amplitude: float,
    source_frequency: float,
    source_num_periods: float,
) -> DirectObservationPipelineConfig:
    return DirectObservationPipelineConfig(
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
        source_amplitude=source_amplitude,
        source_frequency=source_frequency,
        source_num_periods=source_num_periods,
    )


def run_direct_observation_pipeline_check(
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
    simulated_config = _make_direct_config(
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
    )

    observed_config = _make_direct_config(
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
    )

    comparison_config = DirectObservationComparisonConfig(
        simulated=simulated_config,
        observed=observed_config,
        weights=ObservationFunctionalWeights(
            reflection=reflection_weight,
            transmission=transmission_weight,
        ),
    )

    result = run_direct_observation_comparison(
        comparison_config,
        comm=comm,
    )

    report = result.as_dict()

    same_material = simulated_material_kind == observed_material_kind
    values = report["functionals"]["simulated_vs_observed"]
    max_diff = report["values"]["max_abs_observation_difference"]

    report["checks"]["misfit_consistent_with_material_choice"] = bool(
        (
            same_material
            and values["full"] == 0.0
            and max_diff == 0.0
        )
        or (
            not same_material
            and values["full"] > 0.0
            and max_diff > 0.0
        )
    )

    if simulated_material_kind == "localized_disk_defect":
        report["checks"]["simulated_defect_cells_positive"] = bool(
            report["values"]["simulated_defect_cells"] > 0
        )

    if observed_material_kind == "localized_disk_defect":
        report["checks"]["observed_defect_cells_positive"] = bool(
            report["values"]["observed_defect_cells"] > 0
        )

    report["status"] = "ok" if all(report["checks"].values()) else "failed"

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the reusable direct observation pipeline comparison."
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

    report = run_direct_observation_pipeline_check(
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
            values = report["functionals"]["simulated_vs_observed"]
            print(
                f"[DIRECT OBSERVATION PIPELINE CHECK] {status} "
                f"for mesh={args.nx}x{args.ny}, degree={args.degree}, "
                f"theta={args.theta}, dt={args.dt}, steps={args.num_steps}, "
                f"simulated={args.simulated_material_kind}, "
                f"observed={args.observed_material_kind}, "
                f"J_ref={values['reflection']:.6e}, "
                f"J_trans={values['transmission']:.6e}, "
                f"J_full={values['full']:.6e}, "
                f"J_weighted={values['weighted']:.6e}, "
                f"MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
