from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from elastic_fwi.campaigns.run_wave_defect_analysis import run_analysis



def _mpi_rank_size() -> tuple[int, int]:
    try:
        from mpi4py import MPI
    except Exception:
        return 0, 1
    comm = MPI.COMM_WORLD
    return int(comm.rank), int(comm.size)


def _mpi_barrier() -> None:
    try:
        from mpi4py import MPI
    except Exception:
        return
    MPI.COMM_WORLD.Barrier()


REQUIRED_OUTPUTS = (
    "scattered_field_summary_npz",
    "energy_summary_csv",
    "reflection_transmission_ratio_csv",
    "time_window_summary_csv",
    "residual_spectrum_summary_csv",
    "polarization_proxy_summary_csv",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check wave-defect interaction analysis outputs.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--from-inverse-manifest",
        type=Path,
        default=None,
        help="Read real vector-array residual NPZ files from an inverse_time_series_manifest.json or vector_array_manifest.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/wave_physics/manual_check"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.quick and args.from_inverse_manifest is not None:
        raise SystemExit("Use either --quick or --from-inverse-manifest, not both.")

    if not args.quick and args.from_inverse_manifest is None:
        raise SystemExit("Provide --quick or --from-inverse-manifest.")

    rank, size = _mpi_rank_size()

    if rank == 0:
        manifest = run_analysis(
            output_dir=args.output_dir,
            quick=args.quick,
            from_inverse_manifest=args.from_inverse_manifest,
        )
    else:
        manifest = {}

    _mpi_barrier()

    manifest_path = args.output_dir / "campaign_manifest.json"
    if size > 1:
        if not manifest_path.exists():
            raise RuntimeError(f"MPI rank {rank}: missing analysis manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["campaign_manifest"] = str(manifest_path)

    missing = []
    for key in REQUIRED_OUTPUTS:
        path = Path(manifest[key])
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path))

    npz_path = Path(manifest["scattered_field_summary_npz"])
    with np.load(npz_path, allow_pickle=False) as data:
        for key in ["reflection_energy", "transmission_energy"]:
            values = np.asarray(data[key], dtype=float)
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"Non-finite values in {key}.")
            if np.any(values < -1.0e-14):
                raise RuntimeError(f"Negative energy in {key}.")

    ratio = float(manifest["ratio"]["global_ratio"])
    if not np.isfinite(ratio) or ratio < 0.0:
        raise RuntimeError("Invalid global reflection/transmission ratio.")

    if missing:
        raise RuntimeError("Missing output files: " + ", ".join(missing))

    check_manifest = {
        "kind": "wave_defect_interaction_check",
        "status": "ok",
        "analysis_manifest": manifest["campaign_manifest"],
        "global_reflection_transmission_ratio": ratio,
        "checked_outputs": {key: manifest[key] for key in REQUIRED_OUTPUTS},
    }

    check_path = args.output_dir / "check_manifest.json"

    if rank == 0:
        check_path.write_text(
            json.dumps(check_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        print("[WAVE DEFECT INTERACTION CHECK] OK")
        print("output_dir        =", args.output_dir)
        print("check_manifest    =", check_path)
        print("analysis_manifest =", manifest["campaign_manifest"])
        print("global_R_T_ratio  =", ratio)

    _mpi_barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
