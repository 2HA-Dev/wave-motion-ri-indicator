#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Permit direct execution from a source checkout without an editable install.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from elastic_fwi.postprocessing.signed_sensitivity_fields import (
    AnalysisConfig,
    run_signed_sensitivity_analysis,
)


def _patterns(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ("*",)
    patterns: list[str] = []
    for value in values:
        patterns.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(patterns) or ("*",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze signed DG0 material-sensitivity fields, export divergent "
            "maps and spatially weighted metrics, and validate reconstruction "
            "of the paired article_safe maps."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analysis-mode",
        choices=("auto", "line-only", "line-box"),
        default="auto",
        help=(
            "Select a genuine line-only analysis or require a line/box set. "
            "In line-only mode, line is selected automatically when --geometry "
            "is omitted."
        ),
    )
    parser.add_argument(
        "--geometry",
        action="append",
        default=None,
        help="Optional geometry/campaign glob; repeat or use comma-separated values.",
    )
    parser.add_argument(
        "--field-pattern",
        action="append",
        default=None,
        help=(
            "Glob selecting field IDs, parameters, observation modes, filenames, "
            "or full paths; repeat or use comma-separated values."
        ),
    )
    parser.add_argument(
        "--article-safe-dir",
        type=Path,
        default=None,
        help=(
            "Optional root corresponding to gradient_combinations; manifest pairing "
            "is used by default."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional gradient-combination campaign manifest or inverse export "
            "manifest."
        ),
    )
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--zero-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    geometry_patterns = () if not args.geometry else _patterns(args.geometry)
    if args.analysis_mode == "line-only" and not geometry_patterns:
        geometry_patterns = ("line",)
    config = AnalysisConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        analysis_mode=args.analysis_mode,
        geometry_patterns=geometry_patterns,
        field_patterns=_patterns(args.field_pattern),
        article_safe_dir=args.article_safe_dir,
        manifest=args.manifest,
        rtol=args.rtol,
        atol=args.atol,
        zero_tolerance=args.zero_tolerance,
        overwrite=args.overwrite,
        quiet=args.quiet,
    )
    try:
        result = run_signed_sensitivity_analysis(config)
    except Exception as exc:
        print(f"[SIGNED][ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
