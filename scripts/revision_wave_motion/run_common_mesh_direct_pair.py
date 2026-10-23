from __future__ import annotations

import argparse
from pathlib import Path

from mpi4py import MPI

from elastic_fwi.diagnostics.export_common_mesh_direct_pair import (
    export_common_mesh_direct_pair,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run two direct elastic solves on one shared mesh and export "
            "healthy/defective PVD time series."
        )
    )
    parser.add_argument(
        "--template-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--every-k",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Optional short-run override used for smoke validation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comm = MPI.COMM_WORLD

    manifest = export_common_mesh_direct_pair(
        template_manifest=args.template_manifest,
        output_dir=args.output_dir,
        every_k=args.every_k,
        num_steps_override=args.num_steps,
        comm=comm,
    )

    if comm.rank == 0:
        print("[COMMON MESH DIRECT PAIR] OK")
        print(f"manifest={manifest['manifest']}")
        print(
            "simulated_pvd="
            f"{manifest['exports']['direct_simulated']['pvd']}"
        )
        print(
            "observed_pvd="
            f"{manifest['exports']['direct_observed']['pvd']}"
        )
        print(
            "snapshots="
            f"{manifest['num_snapshots_exported']}"
        )


if __name__ == "__main__":
    main()
