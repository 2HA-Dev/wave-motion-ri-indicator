from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI

from elastic_fwi.geometry.boundary_tags import (
    build_boundary_meshtags,
    unit_square_pitch_catch_boundary_specs,
)
from elastic_fwi.observations.boundary_observation import (
    boundary_inner_product,
    boundary_l2_misfit_cost,
    boundary_l2_misfit_norm_squared,
    boundary_l2_norm_squared,
    gamma1_reflection_observation,
    gamma2_transmission_observation,
)
from elastic_fwi.observations.residuals import boundary_residual_report


def _create_vector_space(msh):
    return fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))


def _constant_vector_function(V, values: tuple[float, float], name: str):
    function = fem.Function(V)
    function.name = name

    vx, vy = values

    def expr(x):
        return np.vstack(
            (
                np.full(x.shape[1], vx, dtype=float),
                np.full(x.shape[1], vy, dtype=float),
            )
        )

    function.interpolate(expr)
    function.x.scatter_forward()
    return function


def _close(value: float, expected: float, *, rtol: float, atol: float) -> bool:
    return bool(np.isclose(value, expected, rtol=rtol, atol=atol))


def run_boundary_observation_check(
    *,
    nx: int = 4,
    ny: int = 4,
    rtol: float = 1.0e-14,
    atol: float = 1.0e-14,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be strictly positive.")

    msh = mesh.create_unit_square(comm, nx, ny)
    facet_tags = build_boundary_meshtags(
        msh,
        unit_square_pitch_catch_boundary_specs(),
    )

    V = _create_vector_space(msh)

    u = _constant_vector_function(V, (2.0, -1.0), "u")
    v = _constant_vector_function(V, (1.0, 3.0), "v")
    observed = _constant_vector_function(V, (1.0, 1.0), "observed")

    gamma1_vector = gamma1_reflection_observation(component="vector")
    gamma1_x = gamma1_reflection_observation(component="x")
    gamma1_y = gamma1_reflection_observation(component="y")

    gamma2_vector = gamma2_transmission_observation(component="vector")
    gamma2_x = gamma2_transmission_observation(component="x")
    gamma2_y = gamma2_transmission_observation(component="y")

    norm_gamma1_vector = boundary_l2_norm_squared(u, facet_tags, gamma1_vector)
    norm_gamma1_x = boundary_l2_norm_squared(u, facet_tags, gamma1_x)
    norm_gamma1_y = boundary_l2_norm_squared(u, facet_tags, gamma1_y)

    inner_gamma2_vector = boundary_inner_product(u, v, facet_tags, gamma2_vector)
    inner_gamma2_x = boundary_inner_product(u, v, facet_tags, gamma2_x)
    inner_gamma2_y = boundary_inner_product(u, v, facet_tags, gamma2_y)

    misfit_gamma1_vector = boundary_l2_misfit_norm_squared(
        u,
        observed,
        facet_tags,
        gamma1_vector,
    )
    misfit_gamma1_x = boundary_l2_misfit_norm_squared(
        u,
        observed,
        facet_tags,
        gamma1_x,
    )
    misfit_gamma1_y = boundary_l2_misfit_norm_squared(
        u,
        observed,
        facet_tags,
        gamma1_y,
    )
    cost_gamma1_vector = boundary_l2_misfit_cost(
        u,
        observed,
        facet_tags,
        gamma1_vector,
    )

    residual_report = boundary_residual_report(
        u,
        observed,
        facet_tags,
        gamma2_vector,
    )

    checks = {
        "norm_gamma1_vector": _close(norm_gamma1_vector, 5.0, rtol=rtol, atol=atol),
        "norm_gamma1_x": _close(norm_gamma1_x, 4.0, rtol=rtol, atol=atol),
        "norm_gamma1_y": _close(norm_gamma1_y, 1.0, rtol=rtol, atol=atol),
        "inner_gamma2_vector": _close(inner_gamma2_vector, -1.0, rtol=rtol, atol=atol),
        "inner_gamma2_x": _close(inner_gamma2_x, 2.0, rtol=rtol, atol=atol),
        "inner_gamma2_y": _close(inner_gamma2_y, -3.0, rtol=rtol, atol=atol),
        "misfit_gamma1_vector": _close(misfit_gamma1_vector, 5.0, rtol=rtol, atol=atol),
        "misfit_gamma1_x": _close(misfit_gamma1_x, 1.0, rtol=rtol, atol=atol),
        "misfit_gamma1_y": _close(misfit_gamma1_y, 4.0, rtol=rtol, atol=atol),
        "cost_gamma1_vector": _close(cost_gamma1_vector, 2.5, rtol=rtol, atol=atol),
        "residual_report_norm": _close(
            float(residual_report["norm_squared"]),
            5.0,
            rtol=rtol,
            atol=atol,
        ),
        "residual_report_cost": _close(
            float(residual_report["cost"]),
            2.5,
            rtol=rtol,
            atol=atol,
        ),
    }

    status_ok = bool(all(checks.values()))

    return {
        "status": "ok" if status_ok else "failed",
        "mpi_size": comm.size,
        "mesh": {
            "kind": "unit_square",
            "nx": nx,
            "ny": ny,
            "topological_dimension": msh.topology.dim,
            "geometric_dimension": msh.geometry.dim,
        },
        "checks": checks,
        "values": {
            "norm_gamma1_vector": norm_gamma1_vector,
            "norm_gamma1_x": norm_gamma1_x,
            "norm_gamma1_y": norm_gamma1_y,
            "inner_gamma2_vector": inner_gamma2_vector,
            "inner_gamma2_x": inner_gamma2_x,
            "inner_gamma2_y": inner_gamma2_y,
            "misfit_gamma1_vector": misfit_gamma1_vector,
            "misfit_gamma1_x": misfit_gamma1_x,
            "misfit_gamma1_y": misfit_gamma1_y,
            "cost_gamma1_vector": cost_gamma1_vector,
            "residual_report": residual_report,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check minimal boundary observation operators on Gamma1/Gamma2."
    )
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--rtol", type=float, default=1.0e-14)
    parser.add_argument("--atol", type=float, default=1.0e-14)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_boundary_observation_check(
        nx=args.nx,
        ny=args.ny,
        rtol=args.rtol,
        atol=args.atol,
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "OK" if report["status"] == "ok" else "FAILED"
            print(
                f"[BOUNDARY OBSERVATION CHECK] {status} "
                f"for mesh={args.nx}x{args.ny} on MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
