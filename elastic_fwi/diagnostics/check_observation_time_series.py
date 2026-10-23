from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

from elastic_fwi.geometry.boundary_tags import (
    build_boundary_meshtags,
    unit_square_pitch_catch_boundary_specs,
)
from elastic_fwi.materials.dg0_fields import homogeneous_dg0_material_functions
from elastic_fwi.materials.orthotropic import LEGACY_HEALTHY_MATERIAL
from elastic_fwi.observations.boundary_observation import (
    gamma1_reflection_observation,
    gamma2_transmission_observation,
)
from elastic_fwi.observations.time_series import (
    evaluate_boundary_observation_time_series,
)
from elastic_fwi.solvers.elastic_operator import (
    ElasticOperatorConfig,
    assemble_elastic_operator_matrices,
    create_vector_lagrange_space,
)
from elastic_fwi.solvers.newmark import LinearSolveConfig
from elastic_fwi.solvers.time_loop import (
    NewmarkTimeLoopConfig,
    run_newmark_time_loop,
)
from elastic_fwi.sources.traction import gamma1_vertical_traction_source


def _records_to_dicts(records):
    return [record.as_dict() for record in records]


def run_observation_time_series_check(
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
    ksp_rtol: float = 1.0e-10,
    ksp_atol: float = 1.0e-12,
    ksp_max_it: int = 1000,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be strictly positive.")

    if degree <= 0:
        raise ValueError("degree must be strictly positive.")

    if num_steps <= 0:
        raise ValueError("num_steps must be strictly positive.")

    msh = mesh.create_unit_square(comm, nx, ny)
    facet_tags = build_boundary_meshtags(
        msh,
        unit_square_pitch_catch_boundary_specs(),
    )

    V = create_vector_lagrange_space(msh, degree=degree)
    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)

    operator_config = ElasticOperatorConfig(theta=theta, dt=dt)
    matrices = assemble_elastic_operator_matrices(V, materials, operator_config)

    source = gamma1_vertical_traction_source(
        tag=1,
        amplitude=source_amplitude,
        frequency=source_frequency,
        t0=0.0,
        num_periods=source_num_periods,
    )

    solve_config = LinearSolveConfig(
        ksp_type="cg",
        pc_type="jacobi",
        rtol=ksp_rtol,
        atol=ksp_atol,
        max_it=ksp_max_it,
    )

    try:
        loop_result = run_newmark_time_loop(
            matrices=matrices,
            V=V,
            facet_tags=facet_tags,
            source=source,
            config=NewmarkTimeLoopConfig(
                num_steps=num_steps,
                start_time=0.0,
                store_displacements=True,
            ),
            solve_config=solve_config,
        )

        snapshots = [
            (record.step, record.time, vector)
            for record, vector in zip(
                loop_result.records,
                loop_result.displacement_snapshots,
                strict=True,
            )
        ]

        gamma1 = gamma1_reflection_observation(component="vector")
        gamma2 = gamma2_transmission_observation(component="vector")

        gamma1_records = evaluate_boundary_observation_time_series(
            V,
            snapshots,
            facet_tags,
            gamma1,
        )

        gamma2_records = evaluate_boundary_observation_time_series(
            V,
            snapshots,
            facet_tags,
            gamma2,
        )

        gamma1_values = [record.norm_squared for record in gamma1_records]
        gamma2_values = [record.norm_squared for record in gamma2_records]
        times = [record.time for record in gamma1_records]

        expected_final_time = num_steps * dt
        expected_size = int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)

        checks = {
            "snapshots_stored": len(loop_result.displacement_snapshots) == num_steps,
            "gamma1_records": len(gamma1_records) == num_steps,
            "gamma2_records": len(gamma2_records) == num_steps,
            "times_finite": bool(np.all(np.isfinite(times))),
            "times_end_at_expected": bool(
                np.isclose(times[-1], expected_final_time, rtol=1.0e-14, atol=1.0e-14)
            ),
            "gamma1_finite_nonnegative": bool(
                np.all(np.isfinite(gamma1_values)) and min(gamma1_values) >= 0.0
            ),
            "gamma2_finite_nonnegative": bool(
                np.all(np.isfinite(gamma2_values)) and min(gamma2_values) >= 0.0
            ),
            "some_gamma1_signal_is_nonzero": bool(max(gamma1_values) > 0.0),
            "all_ksp_converged": bool(
                all(record.ksp_converged for record in loop_result.records)
            ),
            "final_vector_size": bool(
                loop_result.final_displacement.getSize() == expected_size
            ),
        }

        status_ok = bool(all(checks.values()))

        report = {
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
            "operator_config": {
                "theta": float(operator_config.theta),
                "dt": float(operator_config.dt),
                "scale_theta_dt2": float(operator_config.theta * operator_config.dt**2),
            },
            "loop_config": {
                "num_steps": int(num_steps),
                "final_time": float(expected_final_time),
                "stored_snapshots": len(loop_result.displacement_snapshots),
            },
            "source": {
                "name": source.name,
                "tag": source.tag,
                "amplitude": float(source_amplitude),
                "frequency": float(source_frequency),
                "num_periods": float(source_num_periods),
            },
            "observations": {
                "gamma1": {
                    "name": gamma1.name,
                    "tag": gamma1.tag,
                    "component": gamma1.component,
                },
                "gamma2": {
                    "name": gamma2.name,
                    "tag": gamma2.tag,
                    "component": gamma2.component,
                },
            },
            "checks": checks,
            "values": {
                "max_gamma1_norm_squared": float(max(gamma1_values)),
                "max_gamma2_norm_squared": float(max(gamma2_values)),
                "final_gamma1_norm_squared": float(gamma1_values[-1]),
                "final_gamma2_norm_squared": float(gamma2_values[-1]),
            },
            "gamma1_records": _records_to_dicts(gamma1_records),
            "gamma2_records": _records_to_dicts(gamma2_records),
            "time_loop_records": [record.as_dict() for record in loop_result.records],
        }

    finally:
        loop_result = locals().get("loop_result")
        if loop_result is not None:
            loop_result.destroy()

        matrices.destroy()

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check boundary observation time series on Gamma1 and Gamma2."
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
    parser.add_argument("--ksp-rtol", type=float, default=1.0e-10)
    parser.add_argument("--ksp-atol", type=float, default=1.0e-12)
    parser.add_argument("--ksp-max-it", type=int, default=1000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_observation_time_series_check(
        nx=args.nx,
        ny=args.ny,
        degree=args.degree,
        theta=args.theta,
        dt=args.dt,
        num_steps=args.num_steps,
        source_amplitude=args.source_amplitude,
        source_frequency=args.source_frequency,
        source_num_periods=args.source_num_periods,
        ksp_rtol=args.ksp_rtol,
        ksp_atol=args.ksp_atol,
        ksp_max_it=args.ksp_max_it,
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "OK" if report["status"] == "ok" else "FAILED"
            print(
                f"[OBSERVATION TIME SERIES CHECK] {status} "
                f"for mesh={args.nx}x{args.ny}, degree={args.degree}, "
                f"theta={args.theta}, dt={args.dt}, steps={args.num_steps}, "
                f"MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
