from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from mpi4py import MPI
from dolfinx import mesh

from elastic_fwi.materials.defects import CircularDefectSpec, DefectScenario
from elastic_fwi.materials.dg0_fields import (
    defective_dg0_material_functions,
    homogeneous_dg0_material_functions,
    material_indices_and_dg0_functions,
)
from elastic_fwi.materials.orthotropic import LEGACY_HEALTHY_MATERIAL


EXPECTED_FIELD_NAMES = ("rho", "C11", "C22", "C12", "C66")


def _owned_values(function) -> np.ndarray:
    V = function.function_space
    owned_size = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    return np.asarray(function.x.array[:owned_size], dtype=float)


def _global_field_stats(function, comm: MPI.Comm) -> dict[str, Any]:
    values = _owned_values(function)

    local_count = int(values.size)
    local_nonfinite = int(np.count_nonzero(~np.isfinite(values)))

    finite_values = values[np.isfinite(values)]

    if finite_values.size > 0:
        local_min = float(np.min(finite_values))
        local_max = float(np.max(finite_values))
    else:
        local_min = float("inf")
        local_max = float("-inf")

    global_count = int(comm.allreduce(local_count, op=MPI.SUM))
    global_nonfinite = int(comm.allreduce(local_nonfinite, op=MPI.SUM))
    global_min = float(comm.allreduce(local_min, op=MPI.MIN))
    global_max = float(comm.allreduce(local_max, op=MPI.MAX))

    if global_count == 0:
        min_value = None
        max_value = None
    else:
        min_value = global_min
        max_value = global_max

    return {
        "global_count": global_count,
        "global_nonfinite": global_nonfinite,
        "global_min": min_value,
        "global_max": max_value,
    }


def _check_constant_field(
    stats: dict[str, Any],
    expected_value: float,
    *,
    rtol: float,
    atol: float,
) -> bool:
    if stats["global_count"] <= 0:
        return False

    if stats["global_nonfinite"] != 0:
        return False

    if stats["global_min"] is None or stats["global_max"] is None:
        return False

    return bool(
        np.isclose(stats["global_min"], expected_value, rtol=rtol, atol=atol)
        and np.isclose(stats["global_max"], expected_value, rtol=rtol, atol=atol)
    )


def _all_fields_are_strictly_positive(
    stats_by_name: dict[str, dict[str, Any]],
) -> bool:
    for stats in stats_by_name.values():
        if stats["global_min"] is None:
            return False
        if stats["global_min"] <= 0.0:
            return False
    return True


def _large_defect_scenario() -> DefectScenario:
    return DefectScenario(
        name="diagnostic_all_cells_defective",
        defects=(
            CircularDefectSpec(
                name="large_defect",
                center=(0.5, 0.5),
                radius=10.0,
                material_factor=0.8,
            ),
        ),
    )


def run_material_dg0_check(
    *,
    nx: int = 2,
    ny: int = 2,
    rtol: float = 1.0e-14,
    atol: float = 1.0e-14,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be strictly positive.")

    msh = mesh.create_unit_square(comm, nx, ny)

    homogeneous = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)

    scenario = _large_defect_scenario()
    defective = defective_dg0_material_functions(msh, scenario)

    indices, defective_from_indices = material_indices_and_dg0_functions(msh, scenario)

    expected_names_ok = (
        tuple(sorted(homogeneous)) == tuple(sorted(EXPECTED_FIELD_NAMES))
        and tuple(sorted(defective)) == tuple(sorted(EXPECTED_FIELD_NAMES))
        and tuple(sorted(defective_from_indices)) == tuple(sorted(EXPECTED_FIELD_NAMES))
    )

    homogeneous_stats = {
        name: _global_field_stats(function, comm)
        for name, function in homogeneous.items()
    }

    defective_stats = {
        name: _global_field_stats(function, comm)
        for name, function in defective.items()
    }

    defective_from_indices_stats = {
        name: _global_field_stats(function, comm)
        for name, function in defective_from_indices.items()
    }

    expected_healthy = {
        "rho": LEGACY_HEALTHY_MATERIAL.rho,
        "C11": LEGACY_HEALTHY_MATERIAL.C11,
        "C22": LEGACY_HEALTHY_MATERIAL.C22,
        "C12": LEGACY_HEALTHY_MATERIAL.C12,
        "C66": LEGACY_HEALTHY_MATERIAL.C66,
    }

    expected_defective = {
        name: 0.8 * value
        for name, value in expected_healthy.items()
    }

    homogeneous_constant_ok = all(
        _check_constant_field(
            homogeneous_stats[name],
            expected_healthy[name],
            rtol=rtol,
            atol=atol,
        )
        for name in EXPECTED_FIELD_NAMES
    )

    defective_constant_ok = all(
        _check_constant_field(
            defective_stats[name],
            expected_defective[name],
            rtol=rtol,
            atol=atol,
        )
        for name in EXPECTED_FIELD_NAMES
    )

    defective_from_indices_constant_ok = all(
        _check_constant_field(
            defective_from_indices_stats[name],
            expected_defective[name],
            rtol=rtol,
            atol=atol,
        )
        for name in EXPECTED_FIELD_NAMES
    )

    local_indices_all_zero = bool(np.all(indices == 0))
    global_indices_all_zero = bool(
        comm.allreduce(local_indices_all_zero, op=MPI.LAND)
    )

    all_positive = (
        _all_fields_are_strictly_positive(homogeneous_stats)
        and _all_fields_are_strictly_positive(defective_stats)
        and _all_fields_are_strictly_positive(defective_from_indices_stats)
    )

    status_ok = bool(
        expected_names_ok
        and homogeneous_constant_ok
        and defective_constant_ok
        and defective_from_indices_constant_ok
        and global_indices_all_zero
        and all_positive
    )

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
        "expected_names_ok": expected_names_ok,
        "homogeneous_constant_ok": homogeneous_constant_ok,
        "defective_constant_ok": defective_constant_ok,
        "defective_from_indices_constant_ok": defective_from_indices_constant_ok,
        "global_indices_all_zero": global_indices_all_zero,
        "all_positive": all_positive,
        "homogeneous": homogeneous_stats,
        "defective": defective_stats,
        "defective_from_indices": defective_from_indices_stats,
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check DG0 material field construction."
    )
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--rtol", type=float, default=1.0e-14)
    parser.add_argument("--atol", type=float, default=1.0e-14)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    report = run_material_dg0_check(
        nx=args.nx,
        ny=args.ny,
        rtol=args.rtol,
        atol=args.atol,
        comm=comm,
    )

    if comm.rank == 0:
        if args.json and not args.quiet:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "OK" if report["status"] == "ok" else "FAILED"
            print(
                f"[MATERIAL DG0 CHECK] {status} "
                f"for mesh={args.nx}x{args.ny} on MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
