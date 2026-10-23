from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from elastic_fwi.postprocessing.dg0_l2_sensitivity_validation import (
    INPUT_SEMANTICS,
    validate_campaign_20_maps,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate all 20 signed material-sensitivity maps against the DG0/L2 "
            "cell-average density contract and quantify the cell-measure bias."
        )
    )
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--analysis-mode",
        choices=("line-only", "auto"),
        default="line-only",
    )
    parser.add_argument(
        "--input-semantics",
        choices=INPUT_SEMANTICS,
        default="l2-density",
        help=(
            "Use cell-integral for maps stored as DG0 cell integrals; use "
            "l2-density for cell-average DG0/L2 maps."
        ),
    )
    parser.add_argument("--defect-center-x", type=float)
    parser.add_argument("--defect-center-y", type=float)
    parser.add_argument("--defect-radius", type=float)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    campaign_dir = args.campaign_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else campaign_dir / "dg0_l2_20_map_validation"
    )
    report = validate_campaign_20_maps(
        campaign_dir=campaign_dir,
        output_dir=output_dir,
        manifest_path=args.manifest,
        analysis_mode=args.analysis_mode,
        input_semantics=args.input_semantics,
        defect_center_x=args.defect_center_x,
        defect_center_y=args.defect_center_y,
        defect_radius=args.defect_radius,
    )
    if not args.quiet:
        print(
            f"[DG0/L2 20 MAPS] status={report['status']} "
            f"processed={report['num_processed']} ok={report['num_ok']} "
            f"semantics={report['input_semantics']}"
        )
        print(f"[DG0/L2 20 MAPS] csv={report['csv_path']}")
        print(f"[DG0/L2 20 MAPS] json={report['json_path']}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
