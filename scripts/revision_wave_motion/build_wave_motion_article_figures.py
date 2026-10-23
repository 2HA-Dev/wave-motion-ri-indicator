#!/usr/bin/env python3
from __future__ import annotations

"""Assemble the reproducible figures of the accompanying Wave Motion article."""

import argparse
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from elastic_fwi.postprocessing.article_figures import (
    ArticleFigureConfig,
    DEFAULT_GROUPS,
    DEFAULT_MODALITIES,
    DEFAULT_PARAMETERS,
    DEFAULT_SIGNED_MODALITIES,
    build_article_figures,
)


def _strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble final Wave Motion article figures from existing campaign and "
            "post-processing outputs without rerunning simulations or gradients."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--campaign-dir",
        action="append",
        type=Path,
        default=None,
        help="Optional explicit campaign root; repeat for line and box.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--figure-groups",
        default=",".join(DEFAULT_GROUPS),
        help="Comma-separated groups: " + ",".join(DEFAULT_GROUPS),
    )
    parser.add_argument("--roi", default="auto", help="auto, full, or xmin,xmax,ymin,ymax")
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--modalities", default=",".join(DEFAULT_MODALITIES))
    parser.add_argument("--parameters", default=",".join(DEFAULT_PARAMETERS))
    parser.add_argument("--signed-modalities", default=",".join(DEFAULT_SIGNED_MODALITIES))
    parser.add_argument(
        "--field-rendering",
        choices=("auto", "vector", "rasterized"),
        default="auto",
        help=(
            "auto uses vector point collections for small maps and an area-weighted "
            "raster field layer for dense DG0 maps. PDF text and annotations remain vector."
        ),
    )
    parser.add_argument("--max-vector-points", type=int, default=50_000)
    parser.add_argument("--map-grid-size", type=int, default=500)
    parser.add_argument("--manifest-name", default="article_figures_manifest.json")
    parser.add_argument("--caption-summary", default="article_figures_summary.md")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero code if any requested figure group is skipped.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ArticleFigureConfig(
        input_root=args.input_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        campaign_dirs=tuple(path.expanduser().resolve() for path in (args.campaign_dir or ())),
        figure_groups=_strings(args.figure_groups),
        roi_spec=args.roi,
        formats=_strings(args.formats),
        dpi=args.dpi,
        overwrite=args.overwrite,
        quiet=args.quiet,
        strict=args.strict,
        manifest_name=args.manifest_name,
        caption_summary_name=args.caption_summary,
        modalities=_strings(args.modalities),
        parameters=_strings(args.parameters),
        signed_modalities=_strings(args.signed_modalities),
        field_rendering=args.field_rendering,
        max_vector_points=args.max_vector_points,
        map_grid_size=args.map_grid_size,
    )
    try:
        result = build_article_figures(config, command=[sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])])
    except Exception as exc:
        print(f"[ARTICLE FIGURES][ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
