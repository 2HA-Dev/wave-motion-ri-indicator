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

from elastic_fwi.postprocessing.scattered_wavefield_reconstruction import (
    AnalysisBox,
    ScatteredOutputPaths,
    ScatteredWavefieldConfig,
    ScatteredWavefieldInputs,
    run_scattered_wavefield_analysis,
    write_scattered_wavefield_outputs,
)


def _pair(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated values: x,y")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _box(value: str) -> AnalysisBox:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Expected four comma-separated values: xmin,xmax,ymin,ymax"
        )
    try:
        box = AnalysisBox(*(float(part) for part in parts))
        box.validate()
        return box
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the physical scattered displacement field "
            "u_defect-u_healthy from paired PVD/PVTU/VTU direct wavefields."
        )
    )
    parser.add_argument(
        "--geometry",
        dest="geometry_kind",
        choices=("line", "box", "unknown"),
        default="unknown",
        help=(
            "Geometry metadata for this single-campaign reconstruction. "
            "Use line for a genuine line-only run; no box input is required."
        ),
    )
    parser.add_argument("--healthy-pvd", type=Path, required=True)
    parser.add_argument("--defect-pvd", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, default=None)
    parser.add_argument(
        "--healthy-field", default="direct_simulated_displacement"
    )
    parser.add_argument(
        "--defect-field", default="direct_observed_displacement"
    )
    parser.add_argument("--defect-center", type=_pair, default=None)
    parser.add_argument("--defect-radius", type=float, default=None)
    parser.add_argument("--analysis-box", type=_box, default=None)
    parser.add_argument("--zone-width", type=float, default=0.65)
    parser.add_argument("--arrival-time", type=float, default=None)
    parser.add_argument("--snapshot-times", type=float, nargs="*", default=None)
    parser.add_argument("--time-atol", type=float, default=1.0e-12)
    parser.add_argument("--time-rtol", type=float, default=1.0e-12)
    parser.add_argument("--geometry-atol", type=float, default=1.0e-12)
    parser.add_argument("--geometry-rtol", type=float, default=0.0)
    parser.add_argument("--pre-atol", type=float, default=1.0e-14)
    parser.add_argument("--pre-rtol", type=float, default=1.0e-10)
    parser.add_argument("--detection-fraction", type=float, default=1.0e-6)
    parser.add_argument("--zone-detection-fraction", type=float, default=0.1)
    parser.add_argument("--ratio-epsilon", type=float, default=1.0e-30)
    parser.add_argument(
        "--vtk-export-mode",
        choices=("none", "selected", "all"),
        default="selected",
    )
    parser.add_argument("--raster-resolution", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _configure_logger(paths: ScatteredOutputPaths, *, quiet: bool) -> logging.Logger:
    paths.create()
    log_path = paths.logs_dir / "scattered_wavefield_reconstruction.log"
    logger = logging.getLogger("scattered_wavefield_reconstruction")
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
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            print(
                f"ERROR: Output directory is not empty: {output_dir}. Use --overwrite.",
                file=sys.stderr,
            )
            return 1
        shutil.rmtree(output_dir)
    paths = ScatteredOutputPaths.from_root(output_dir)
    try:
        logger = _configure_logger(paths, quiet=args.quiet)
        command = (
            [sys.executable, *sys.argv]
            if argv is None
            else [sys.executable, sys.argv[0], *argv]
        )
        config = ScatteredWavefieldConfig(
            inputs=ScatteredWavefieldInputs(
                healthy_pvd=args.healthy_pvd,
                defect_pvd=args.defect_pvd,
                campaign_manifest=args.campaign_manifest,
            ),
            output_dir=output_dir,
            geometry_kind=args.geometry_kind,
            healthy_field=args.healthy_field,
            defect_field=args.defect_field,
            time_atol=args.time_atol,
            time_rtol=args.time_rtol,
            geometry_atol=args.geometry_atol,
            geometry_rtol=args.geometry_rtol,
            pre_atol=args.pre_atol,
            pre_rtol=args.pre_rtol,
            detection_fraction=args.detection_fraction,
            zone_detection_fraction=args.zone_detection_fraction,
            ratio_epsilon=args.ratio_epsilon,
            defect_center=args.defect_center,
            defect_radius=args.defect_radius,
            analysis_box=args.analysis_box,
            zone_width=args.zone_width,
            arrival_time=args.arrival_time,
            snapshot_times=None
            if args.snapshot_times is None
            else tuple(args.snapshot_times),
            vtk_export_mode=args.vtk_export_mode,
            raster_resolution=args.raster_resolution,
            overwrite=args.overwrite,
            quiet=args.quiet,
        )
        logger.info("Starting physical scattered-wavefield reconstruction.")
        logger.info("geometry=%s", args.geometry_kind)
        logger.info(
            "analysis_mode=%s",
            {"line": "line-only", "box": "box-only", "unknown": "single-campaign"}[
                args.geometry_kind
            ],
        )
        logger.info("healthy_pvd=%s", args.healthy_pvd)
        logger.info("defect_pvd=%s", args.defect_pvd)
        logger.info("campaign_manifest=%s", args.campaign_manifest)
        logger.info("healthy_field=%s", args.healthy_field)
        logger.info("defect_field=%s", args.defect_field)
        logger.info("vtk_export_mode=%s", args.vtk_export_mode)
        logger.info("output_dir=%s", output_dir)
        result = run_scattered_wavefield_analysis(config, logger=logger)
        outputs = write_scattered_wavefield_outputs(
            result,
            config,
            command=command,
            repository_root=_REPOSITORY_ROOT,
            logger=logger,
        )
        outputs["log"] = paths.logs_dir / "scattered_wavefield_reconstruction.log"
        logger.info("Pre-interaction validation: %s", result.validation["status"])
        logger.info("Status: OK")
        if not args.quiet:
            print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
        return 0
    except Exception as exc:
        if "logger" in locals():
            logger.exception("Scattered-wavefield reconstruction failed: %s", exc)
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
