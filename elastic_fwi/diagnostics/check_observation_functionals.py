from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from mpi4py import MPI

from elastic_fwi.data.synthetic_observations import (
    add_gaussian_noise,
    synthetic_data_from_boundary_records,
)
from elastic_fwi.diagnostics.check_observation_time_series import (
    run_observation_time_series_check,
)
from elastic_fwi.inversion.observation_functionals import (
    ObservationFunctionalWeights,
    compute_observation_functionals,
)


def run_observation_functionals_check(
    *,
    nx: int = 2,
    ny: int = 2,
    degree: int = 1,
    theta: float = 0.25,
    dt: float = 0.1,
    num_steps: int = 5,
    source_amplitude: float = 1.0,
    source_frequency: float = 1.0,
    source_num_periods: float = 1.0,
    noise_level: float = 1.0e-4,
    noise_seed: int = 123,
    reflection_weight: float = 1.0,
    transmission_weight: float = 1.0,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    if noise_level < 0.0:
        raise ValueError("noise_level must be non-negative.")

    observation_report = run_observation_time_series_check(
        nx=nx,
        ny=ny,
        degree=degree,
        theta=theta,
        dt=dt,
        num_steps=num_steps,
        source_amplitude=source_amplitude,
        source_frequency=source_frequency,
        source_num_periods=source_num_periods,
        comm=comm,
    )

    clean_data = synthetic_data_from_boundary_records(
        reflection_records=observation_report["gamma1_records"],
        transmission_records=observation_report["gamma2_records"],
        metadata={
            "source": "check_observation_functionals",
            "mesh_nx": nx,
            "mesh_ny": ny,
            "degree": degree,
            "theta": theta,
            "dt": dt,
            "num_steps": num_steps,
        },
    )

    noisy_data = add_gaussian_noise(
        clean_data,
        noise_level=noise_level,
        seed=noise_seed,
        relative=True,
        clip_min=0.0,
    )

    weights = ObservationFunctionalWeights(
        reflection=reflection_weight,
        transmission=transmission_weight,
    )

    clean_vs_clean = compute_observation_functionals(
        clean_data,
        clean_data,
        weights=weights,
    )

    clean_vs_noisy = compute_observation_functionals(
        clean_data,
        noisy_data,
        weights=weights,
    )

    noisy_vs_clean = compute_observation_functionals(
        noisy_data,
        clean_data,
        weights=weights,
    )

    clean_stacked = clean_data.stacked_values()
    noisy_stacked = noisy_data.stacked_values()
    difference = noisy_stacked - clean_stacked

    checks = {
        "observation_report_ok": observation_report["status"] == "ok",
        "num_steps": clean_data.num_steps == num_steps,
        "clean_values_finite": bool(np.all(np.isfinite(clean_stacked))),
        "noisy_values_finite": bool(np.all(np.isfinite(noisy_stacked))),
        "misfit_clean_clean_zero": bool(
            np.isclose(clean_vs_clean.full, 0.0, rtol=0.0, atol=0.0)
            and np.isclose(clean_vs_clean.weighted, 0.0, rtol=0.0, atol=0.0)
        ),
        "misfit_clean_noisy_finite": bool(
            np.isfinite(clean_vs_noisy.reflection)
            and np.isfinite(clean_vs_noisy.transmission)
            and np.isfinite(clean_vs_noisy.full)
            and np.isfinite(clean_vs_noisy.weighted)
        ),
        "misfit_clean_noisy_nonnegative": bool(
            clean_vs_noisy.reflection >= 0.0
            and clean_vs_noisy.transmission >= 0.0
            and clean_vs_noisy.full >= 0.0
            and clean_vs_noisy.weighted >= 0.0
        ),
        "full_is_sum": bool(
            np.isclose(
                clean_vs_noisy.full,
                clean_vs_noisy.reflection + clean_vs_noisy.transmission,
                rtol=1.0e-14,
                atol=1.0e-14,
            )
        ),
        "weighted_formula": bool(
            np.isclose(
                clean_vs_noisy.weighted,
                reflection_weight * clean_vs_noisy.reflection
                + transmission_weight * clean_vs_noisy.transmission,
                rtol=1.0e-14,
                atol=1.0e-14,
            )
        ),
        "symmetry_of_squared_misfit": bool(
            np.isclose(
                clean_vs_noisy.full,
                noisy_vs_clean.full,
                rtol=1.0e-14,
                atol=1.0e-14,
            )
            and np.isclose(
                clean_vs_noisy.weighted,
                noisy_vs_clean.weighted,
                rtol=1.0e-14,
                atol=1.0e-14,
            )
        ),
        "noise_changes_at_least_one_value": bool(np.max(np.abs(difference)) > 0.0),
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
            "source_amplitude": source_amplitude,
            "source_frequency": source_frequency,
            "source_num_periods": source_num_periods,
            "noise_level": noise_level,
            "noise_seed": noise_seed,
            "reflection_weight": reflection_weight,
            "transmission_weight": transmission_weight,
        },
        "values": {
            "max_abs_noise": float(np.max(np.abs(difference))),
            "clean_max_reflection": clean_data.reflection.max_value,
            "clean_max_transmission": clean_data.transmission.max_value,
            "noisy_max_reflection": noisy_data.reflection.max_value,
            "noisy_max_transmission": noisy_data.transmission.max_value,
        },
        "functionals": {
            "clean_vs_clean": clean_vs_clean.as_dict(),
            "clean_vs_noisy": clean_vs_noisy.as_dict(),
            "noisy_vs_clean": noisy_vs_clean.as_dict(),
        },
        "clean_data": clean_data.as_dict(),
        "noisy_data": noisy_data.as_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check scalar reflection/transmission/full/weighted observation functionals."
    )
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--source-amplitude", type=float, default=1.0)
    parser.add_argument("--source-frequency", type=float, default=1.0)
    parser.add_argument("--source-num-periods", type=float, default=1.0)
    parser.add_argument("--noise-level", type=float, default=1.0e-4)
    parser.add_argument("--noise-seed", type=int, default=123)
    parser.add_argument("--reflection-weight", type=float, default=1.0)
    parser.add_argument("--transmission-weight", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_observation_functionals_check(
        nx=args.nx,
        ny=args.ny,
        degree=args.degree,
        theta=args.theta,
        dt=args.dt,
        num_steps=args.num_steps,
        source_amplitude=args.source_amplitude,
        source_frequency=args.source_frequency,
        source_num_periods=args.source_num_periods,
        noise_level=args.noise_level,
        noise_seed=args.noise_seed,
        reflection_weight=args.reflection_weight,
        transmission_weight=args.transmission_weight,
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "OK" if report["status"] == "ok" else "FAILED"
            values = report["functionals"]["clean_vs_noisy"]
            print(
                f"[OBSERVATION FUNCTIONALS CHECK] {status} "
                f"for mesh={args.nx}x{args.ny}, degree={args.degree}, "
                f"theta={args.theta}, dt={args.dt}, steps={args.num_steps}, "
                f"J_ref={values['reflection']:.6e}, "
                f"J_trans={values['transmission']:.6e}, "
                f"J_full={values['full']:.6e}, "
                f"J_weighted={values['weighted']:.6e}, "
                f"MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
