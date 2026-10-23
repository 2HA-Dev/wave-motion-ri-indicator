from __future__ import annotations

import argparse
import json
from typing import Any

from mpi4py import MPI
from dolfinx import mesh

from elastic_fwi.geometry.boundary_tags import (
    boundary_measures_by_tag,
    boundary_tag_counts_global,
    build_boundary_meshtags,
    unit_square_pitch_catch_boundary_specs,
)


EXPECTED_TAGS = (1, 2, 3, 4)
EXPECTED_MEASURE_UNIT_SQUARE = {
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
}


def _measure_close(value: float, expected: float, *, rtol: float, atol: float) -> bool:
    return abs(value - expected) <= atol + rtol * abs(expected)


def run_boundary_tags_check(
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

    specs = unit_square_pitch_catch_boundary_specs()
    facet_tags = build_boundary_meshtags(msh, specs)

    counts = boundary_tag_counts_global(facet_tags, EXPECTED_TAGS, comm)
    measures = boundary_measures_by_tag(msh, facet_tags, EXPECTED_TAGS)

    counts_positive = all(counts[tag] > 0 for tag in EXPECTED_TAGS)

    measures_ok = all(
        _measure_close(
            measures[tag],
            EXPECTED_MEASURE_UNIT_SQUARE[tag],
            rtol=rtol,
            atol=atol,
        )
        for tag in EXPECTED_TAGS
    )

    local_present_tags = tuple(sorted(set(int(value) for value in facet_tags.values)))

    tags_present_by_count = {
        int(tag): bool(counts[int(tag)] > 0)
        for tag in EXPECTED_TAGS
    }

    tags_present = all(tags_present_by_count.values())

    status_ok = bool(counts_positive and measures_ok and tags_present)

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
        "expected_tags": list(EXPECTED_TAGS),
        "local_present_tags": list(local_present_tags),
        "tags_present_by_count": tags_present_by_count,
        "tags_present": tags_present,
        "counts_positive": counts_positive,
        "measures_ok": measures_ok,
        "counts": counts,
        "measures": measures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check pitch-catch boundary tags on a unit-square mesh."
    )
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--rtol", type=float, default=1.0e-14)
    parser.add_argument("--atol", type=float, default=1.0e-14)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_boundary_tags_check(
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
                f"[BOUNDARY TAGS CHECK] {status} "
                f"for mesh={args.nx}x{args.ny} on MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
