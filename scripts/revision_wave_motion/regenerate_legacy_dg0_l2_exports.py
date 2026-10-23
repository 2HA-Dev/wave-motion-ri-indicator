from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from elastic_fwi.postprocessing.dg0_l2_reexport import regenerate_legacy_campaign_dg0_l2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the existing 20 legacy DG0 cell-integral raw_signed maps to "
            "DG0/L2 cell-average densities and rebuild the 20 article_safe maps. "
            "This is VTK-only post-processing and does not rerun any wave solve."
        )
    )
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--backup-suffix", default="_legacy_cell_integral")
    parser.add_argument("--overwrite-backup", action="store_true")
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = regenerate_legacy_campaign_dg0_l2(
        args.campaign_dir,
        backup_suffix=args.backup_suffix,
        overwrite_backup=args.overwrite_backup,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(
        "[DG0/L2 REEXPORT] status=ok raw_signed=20 article_safe=20 "
        "wave_solves_rerun=false"
    )
    print(f"[DG0/L2 REEXPORT] manifest={report['manifest_path']}")
    print(f"[DG0/L2 REEXPORT] raw_backup={report['raw_backup']}")
    print(f"[DG0/L2 REEXPORT] article_safe_backup={report['article_safe_backup']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
