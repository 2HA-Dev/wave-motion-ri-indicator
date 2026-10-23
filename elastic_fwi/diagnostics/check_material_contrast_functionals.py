from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from mpi4py import MPI

from elastic_fwi.inversion.observation_functionals import (
    ObservationFunctionalWeights,
)
from elastic_fwi.pipelines.direct_observation import DirectObservationPipelineConfig
from elastic_fwi.pipelines.observation_comparison import (
    DirectObservationComparisonConfig,
    run_direct_observation_comparison,
)


def run_material_contrast_functionals_check(
    *,
    nx: int = 2,
    ny: int = 2,
    degree: int = 1,
    theta: float = 0.25,
    dt: float = 0.1,
    num_steps: int = 5,
    contrast_factor: float = 0.8,
    source_amplitude: float = 1.0,
    source_frequency: float = 1.0,
    source_num_periods: float = 1.0,
    reflection_weight: float = 1.0,
    transmission_weight: float = 1.0,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    """
    Backward-compatible diagnostic facade for homogeneous material contrast.

    Internally, the diagnostic relies on the reusable object-oriented comparison pipeline:

        healthy direct pipeline
        homogeneous_contrast direct pipeline
        observation functionals

    The returned report keeps the keys expected by the existing tests.
    """

    healthy_config = DirectObservationPipelineConfig(
        nx=nx,
        ny=ny,
        degree=degree,
        theta=theta,
        dt=dt,
        num_steps=num_steps,
        material_kind="healthy",
        contrast_factor=contrast_factor,
        source_amplitude=source_amplitude,
        source_frequency=source_frequency,
        source_num_periods=source_num_periods,
    )

    contrast_config = DirectObservationPipelineConfig(
        nx=nx,
        ny=ny,
        degree=degree,
        theta=theta,
        dt=dt,
        num_steps=num_steps,
        material_kind="homogeneous_contrast",
        contrast_factor=contrast_factor,
        source_amplitude=source_amplitude,
        source_frequency=source_frequency,
        source_num_periods=source_num_periods,
    )

    comparison = run_direct_observation_comparison(
        DirectObservationComparisonConfig(
            simulated=healthy_config,
            observed=contrast_config,
            weights=ObservationFunctionalWeights(
                reflection=reflection_weight,
                transmission=transmission_weight,
            ),
        ),
        comm=comm,
    )

    comparison_report = comparison.as_dict()

    healthy_vs_contrast = comparison_report["functionals"]["simulated_vs_observed"]
    contrast_vs_healthy = comparison_report["functionals"]["observed_vs_simulated"]

    max_abs_difference = float(
        comparison_report["values"]["max_abs_observation_difference"]
    )

    checks = {
        "healthy_values_finite": bool(
            comparison_report["checks"]["simulated_values_finite"]
        ),
        "contrast_values_finite": bool(
            comparison_report["checks"]["observed_values_finite"]
        ),
        "healthy_values_nonnegative": bool(
            comparison_report["checks"]["simulated_values_nonnegative"]
        ),
        "contrast_values_nonnegative": bool(
            comparison_report["checks"]["observed_values_nonnegative"]
        ),
        "contrast_changes_observations": bool(max_abs_difference > 0.0),
        "reflection_misfit_positive": bool(healthy_vs_contrast["reflection"] > 0.0),
        "transmission_misfit_positive": bool(
            healthy_vs_contrast["transmission"] > 0.0
        ),
        "full_misfit_positive": bool(healthy_vs_contrast["full"] > 0.0),
        "weighted_misfit_positive": bool(healthy_vs_contrast["weighted"] > 0.0),
        "full_is_sum": bool(comparison_report["checks"]["full_is_sum"]),
        "weighted_formula": bool(comparison_report["checks"]["weighted_formula"]),
        "symmetry_of_squared_misfit": bool(
            comparison_report["checks"]["symmetry_of_squared_misfit"]
        ),
    }

    status_ok = bool(all(checks.values()))

    return {
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
            "contrast_factor": contrast_factor,
            "source_amplitude": source_amplitude,
            "source_frequency": source_frequency,
            "source_num_periods": source_num_periods,
            "reflection_weight": reflection_weight,
            "transmission_weight": transmission_weight,
        },
        "values": {
            "max_abs_observation_difference": max_abs_difference,
            "healthy_max_reflection": comparison_report["values"][
                "simulated_max_reflection"
            ],
            "healthy_max_transmission": comparison_report["values"][
                "simulated_max_transmission"
            ],
            "contrast_max_reflection": comparison_report["values"][
                "observed_max_reflection"
            ],
            "contrast_max_transmission": comparison_report["values"][
                "observed_max_transmission"
            ],
        },
        "functionals": {
            "healthy_vs_contrast": healthy_vs_contrast,
            "contrast_vs_healthy": contrast_vs_healthy,
        },
        "healthy_data": comparison_report["simulated"]["data"],
        "contrast_data": comparison_report["observed"]["data"],
        "healthy_time_loop_records": comparison_report["simulated"][
            "time_loop_records"
        ],
        "contrast_time_loop_records": comparison_report["observed"][
            "time_loop_records"
        ],
        "poo_comparison": comparison_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check observation functionals between healthy and contrasted materials."
    )
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--contrast-factor", type=float, default=0.8)
    parser.add_argument("--source-amplitude", type=float, default=1.0)
    parser.add_argument("--source-frequency", type=float, default=1.0)
    parser.add_argument("--source-num-periods", type=float, default=1.0)
    parser.add_argument("--reflection-weight", type=float, default=1.0)
    parser.add_argument("--transmission-weight", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_material_contrast_functionals_check(
        nx=args.nx,
        ny=args.ny,
        degree=args.degree,
        theta=args.theta,
        dt=args.dt,
        num_steps=args.num_steps,
        contrast_factor=args.contrast_factor,
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
            values = report["functionals"]["healthy_vs_contrast"]
            print(
                f"[MATERIAL CONTRAST FUNCTIONALS CHECK] {status} "
                f"for mesh={args.nx}x{args.ny}, degree={args.degree}, "
                f"theta={args.theta}, dt={args.dt}, steps={args.num_steps}, "
                f"contrast={args.contrast_factor}, "
                f"J_ref={values['reflection']:.6e}, "
                f"J_trans={values['transmission']:.6e}, "
                f"J_full={values['full']:.6e}, "
                f"J_weighted={values['weighted']:.6e}, "
                f"MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
