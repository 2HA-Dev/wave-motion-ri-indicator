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


def run_synthetic_observation_data_check(
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
            "source": "check_synthetic_observation_data",
            "mesh_nx": nx,
            "mesh_ny": ny,
            "degree": degree,
            "theta": theta,
            "dt": dt,
            "num_steps": num_steps,
        },
    )

    noisy_data_a = add_gaussian_noise(
        clean_data,
        noise_level=noise_level,
        seed=noise_seed,
        relative=True,
        clip_min=0.0,
    )

    noisy_data_b = add_gaussian_noise(
        clean_data,
        noise_level=noise_level,
        seed=noise_seed,
        relative=True,
        clip_min=0.0,
    )

    clean_stacked = clean_data.stacked_values()
    noisy_stacked_a = noisy_data_a.stacked_values()
    noisy_stacked_b = noisy_data_b.stacked_values()

    checks = {
        "observation_report_ok": observation_report["status"] == "ok",
        "num_steps": clean_data.num_steps == num_steps,
        "final_time": bool(np.isclose(clean_data.final_time, num_steps * dt)),
        "reflection_tag": clean_data.reflection.tag == 1,
        "transmission_tag": clean_data.transmission.tag == 2,
        "stacked_shape": clean_stacked.shape == (2, num_steps),
        "clean_values_finite": bool(np.all(np.isfinite(clean_stacked))),
        "clean_values_nonnegative": bool(np.min(clean_stacked) >= 0.0),
        "noisy_values_finite": bool(np.all(np.isfinite(noisy_stacked_a))),
        "noisy_values_nonnegative": bool(np.min(noisy_stacked_a) >= 0.0),
        "noise_reproducible": bool(np.allclose(noisy_stacked_a, noisy_stacked_b)),
        "clean_data_not_mutated": bool(
            np.allclose(clean_stacked, clean_data.stacked_values())
        ),
        "noise_metadata": bool(
            noisy_data_a.metadata.get("noise_seed") == noise_seed
            and np.isclose(noisy_data_a.metadata.get("noise_level"), noise_level)
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
            "source_amplitude": source_amplitude,
            "source_frequency": source_frequency,
            "source_num_periods": source_num_periods,
            "noise_level": noise_level,
            "noise_seed": noise_seed,
        },
        "values": {
            "clean_max_reflection": clean_data.reflection.max_value,
            "clean_max_transmission": clean_data.transmission.max_value,
            "clean_final_reflection": clean_data.reflection.final_value,
            "clean_final_transmission": clean_data.transmission.final_value,
            "noisy_max_reflection": noisy_data_a.reflection.max_value,
            "noisy_max_transmission": noisy_data_a.transmission.max_value,
            "noisy_final_reflection": noisy_data_a.reflection.final_value,
            "noisy_final_transmission": noisy_data_a.transmission.final_value,
        },
        "clean_data": clean_data.as_dict(),
        "noisy_data": noisy_data_a.as_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check synthetic reflection/transmission observation data."
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
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_synthetic_observation_data_check(
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
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "OK" if report["status"] == "ok" else "FAILED"
            print(
                f"[SYNTHETIC OBSERVATION DATA CHECK] {status} "
                f"for mesh={args.nx}x{args.ny}, degree={args.degree}, "
                f"theta={args.theta}, dt={args.dt}, steps={args.num_steps}, "
                f"noise={args.noise_level}, MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
