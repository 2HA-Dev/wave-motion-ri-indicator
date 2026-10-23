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
from elastic_fwi.sources.traction import (
    BoundaryTractionSource,
    ConstantTimeSignal,
    boundary_traction_l2_norm_squared,
    boundary_traction_work,
    assemble_boundary_traction_vector,
    gamma1_vertical_traction_source,
)


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


def run_traction_source_check(
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
    test_function = _constant_vector_function(V, (2.0, -1.0), "test_function")

    constant_vertical = BoundaryTractionSource(
        name="constant_vertical",
        tag=1,
        direction=(0.0, 1.0),
        time_signal=ConstantTimeSignal(3.0),
    )

    stronger_vertical = BoundaryTractionSource(
        name="stronger_vertical",
        tag=1,
        direction=(0.0, 2.0),
        time_signal=ConstantTimeSignal(3.0),
    )

    pulse = gamma1_vertical_traction_source(
        tag=1,
        amplitude=2.0,
        frequency=1.0,
        t0=0.0,
        num_periods=1.0,
    )

    work_constant = boundary_traction_work(
        test_function,
        facet_tags,
        constant_vertical,
        t=0.0,
    )

    norm_stronger = boundary_traction_l2_norm_squared(
        V,
        facet_tags,
        stronger_vertical,
        t=0.0,
    )

    pulse_value_before = pulse.value(-0.1)
    pulse_value_start = pulse.value(0.0)
    pulse_value_mid = pulse.value(0.25)
    pulse_value_end = pulse.value(1.0)
    pulse_value_after = pulse.value(1.1)

    pulse_norm_mid = boundary_traction_l2_norm_squared(
        V,
        facet_tags,
        pulse,
        t=0.25,
    )

    rhs = assemble_boundary_traction_vector(
        V,
        facet_tags,
        constant_vertical,
        t=0.0,
    )

    rhs_global_size = int(rhs.getSize())
    rhs_norm = float(rhs.norm())

    expected_global_size = int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)

    try:
        rhs.destroy()
    except Exception:
        pass

    checks = {
        "constant_work": _close(work_constant, -3.0, rtol=rtol, atol=atol),
        "stronger_norm": _close(norm_stronger, 36.0, rtol=rtol, atol=atol),
        "pulse_before_zero": _close(pulse_value_before[0], 0.0, rtol=rtol, atol=atol)
        and _close(pulse_value_before[1], 0.0, rtol=rtol, atol=atol),
        "pulse_start_zero": _close(pulse_value_start[0], 0.0, rtol=rtol, atol=atol)
        and _close(pulse_value_start[1], 0.0, rtol=rtol, atol=atol),
        "pulse_mid_value": _close(pulse_value_mid[0], 0.0, rtol=rtol, atol=atol)
        and _close(pulse_value_mid[1], 1.0, rtol=rtol, atol=atol),
        "pulse_end_zero": _close(pulse_value_end[0], 0.0, rtol=rtol, atol=atol)
        and _close(pulse_value_end[1], 0.0, rtol=rtol, atol=atol),
        "pulse_after_zero": _close(pulse_value_after[0], 0.0, rtol=rtol, atol=atol)
        and _close(pulse_value_after[1], 0.0, rtol=rtol, atol=atol),
        "pulse_norm_mid": _close(pulse_norm_mid, 1.0, rtol=rtol, atol=atol),
        "rhs_size": rhs_global_size == expected_global_size,
        "rhs_norm_positive": rhs_norm > 0.0,
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
            "constant_work": work_constant,
            "stronger_norm": norm_stronger,
            "pulse_value_before": list(pulse_value_before),
            "pulse_value_start": list(pulse_value_start),
            "pulse_value_mid": list(pulse_value_mid),
            "pulse_value_end": list(pulse_value_end),
            "pulse_value_after": list(pulse_value_after),
            "pulse_norm_mid": pulse_norm_mid,
            "rhs_global_size": rhs_global_size,
            "expected_global_size": expected_global_size,
            "rhs_norm": rhs_norm,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check weak boundary traction source on Gamma1."
    )
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--rtol", type=float, default=1.0e-14)
    parser.add_argument("--atol", type=float, default=1.0e-14)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_traction_source_check(
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
                f"[TRACTION SOURCE CHECK] {status} "
                f"for mesh={args.nx}x{args.ny} on MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
