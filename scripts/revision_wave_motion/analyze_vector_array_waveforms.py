#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from elastic_fwi.postprocessing.receiver_aperture_comparison import (
    ALLOWED_ANALYSIS_GEOMETRIES,
    ALLOWED_TIME_POLICIES,
)
from elastic_fwi.postprocessing.vector_array_waveforms import (
    ALLOWED_RESIDUAL_CONVENTIONS,
    WaveformAnalysisConfig,
    WaveformAnalysisInputs,
    WaveformOutputPaths,
    run_vector_array_waveform_analysis,
    write_waveform_analysis_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze healthy/reference, defective/observed, and vector residual "
            "waveforms in genuine line-only mode or from the validated historical "
            "line/box vector-array CSV exports."
        )
    )
    parser.add_argument(
        "--geometry",
        choices=ALLOWED_ANALYSIS_GEOMETRIES,
        default="line-box",
        help=(
            "line performs a genuine line-only analysis. line-box preserves the "
            "historical two-campaign analysis."
        ),
    )
    parser.add_argument("--line-residuals", type=Path, required=True)
    parser.add_argument(
        "--box-residuals",
        type=Path,
        default=None,
        help="Required only when --geometry line-box.",
    )
    parser.add_argument("--line-manifest", type=Path, required=True)
    parser.add_argument(
        "--box-manifest",
        type=Path,
        default=None,
        help="Required only when --geometry line-box.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.025,
        help=(
            "Solver time step of the campaign being analysed. The article "
            "campaigns use 0.025 (config.dt of both manifests)."
        ),
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=20.0,
        help=(
            "Physical final time of the campaign. The article campaigns use "
            "20.0 = 800 * 0.025 (temporal_sampling.T_final of both manifests)."
        ),
    )
    parser.add_argument("--arrival-threshold", type=float, default=0.1)
    parser.add_argument(
        "--residual-convention",
        choices=ALLOWED_RESIDUAL_CONVENTIONS,
        default="source",
        help=(
            "source preserves the validated repository convention. "
            "healthy-minus-defect uses reference-observed; defect-minus-healthy "
            "uses observed-reference. Source CSV columns are never modified."
        ),
    )
    parser.add_argument(
        "--time-column-policy",
        choices=ALLOWED_TIME_POLICIES,
        default="strict-physical",
        help=(
            "strict-physical requires CSV time=step*dt. legacy-step-index explicitly "
            "accepts the validated historical export where CSV time equals step."
        ),
    )
    parser.add_argument("--time-atol", type=float, default=1.0e-12)
    parser.add_argument("--time-rtol", type=float, default=1.0e-12)
    parser.add_argument("--value-atol", type=float, default=1.0e-14)
    parser.add_argument("--value-rtol", type=float, default=1.0e-12)
    parser.add_argument("--geometry-atol", type=float, default=1.0e-12)
    parser.add_argument("--ratio-epsilon", type=float, default=1.0e-30)
    parser.add_argument(
        "--receiver-ids",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional explicit receiver IDs to plot wherever they exist. "
            "Without this option, deterministic representative receivers are used."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _configure_logger(
    paths: WaveformOutputPaths,
    *,
    quiet: bool,
) -> logging.Logger:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = paths.logs_dir / "vector_array_waveform_analysis.log"
    logger = logging.getLogger("vector_array_waveform_analysis")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if not quiet:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    paths = WaveformOutputPaths.from_root(output_dir)

    try:
        if output_dir.exists() and any(output_dir.iterdir()):
            if not args.overwrite:
                raise FileExistsError(
                    f"Output directory is not empty: {output_dir}. Use --overwrite."
                )
            shutil.rmtree(output_dir)
        paths.create(overwrite=False)
        logger = _configure_logger(paths, quiet=args.quiet)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    command = (
        [sys.executable, *sys.argv]
        if argv is None
        else [sys.executable, sys.argv[0], *argv]
    )
    try:
        config = WaveformAnalysisConfig(
            inputs=WaveformAnalysisInputs(
                line_residuals=args.line_residuals,
                box_residuals=args.box_residuals,
                line_manifest=args.line_manifest,
                box_manifest=args.box_manifest,
            ),
            output_dir=output_dir,
            dt=args.dt,
            final_time=args.final_time,
            arrival_threshold=args.arrival_threshold,
            residual_convention=args.residual_convention,
            time_column_policy=args.time_column_policy,
            time_atol=args.time_atol,
            time_rtol=args.time_rtol,
            value_atol=args.value_atol,
            value_rtol=args.value_rtol,
            geometry_atol=args.geometry_atol,
            ratio_epsilon=args.ratio_epsilon,
            receiver_ids=None
            if args.receiver_ids is None
            else tuple(args.receiver_ids),
            overwrite=args.overwrite,
            quiet=args.quiet,
            geometry=args.geometry,
        )
        logger.info(
            "Starting vector-array waveform analysis in %s mode.",
            "line-only" if args.geometry == "line" else "line-box",
        )
        logger.info("geometry=%s", args.geometry)
        logger.info("line_residuals=%s", args.line_residuals)
        logger.info("line_manifest=%s", args.line_manifest)
        if args.geometry == "line-box":
            logger.info("box_residuals=%s", args.box_residuals)
            logger.info("box_manifest=%s", args.box_manifest)
        logger.info("dt=%s", args.dt)
        logger.info("final_time=%s", args.final_time)
        logger.info("time_column_policy=%s", args.time_column_policy)
        logger.info("residual_convention=%s", args.residual_convention)
        logger.info("arrival_threshold=%s", args.arrival_threshold)
        logger.info("output_dir=%s", output_dir)

        result = run_vector_array_waveform_analysis(config, logger=logger)
        outputs = write_waveform_analysis_outputs(
            result,
            config,
            command=command,
            repository_root=_REPOSITORY_ROOT,
            logger=logger,
        )
        outputs["log"] = paths.logs_dir / "vector_array_waveform_analysis.log"
        logger.info("Status: OK")
        if not args.quiet:
            print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
        return 0
    except Exception as exc:
        logger.exception("Vector-array waveform analysis failed: %s", exc)
        if args.quiet:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
