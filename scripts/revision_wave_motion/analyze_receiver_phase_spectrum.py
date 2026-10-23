#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from elastic_fwi.postprocessing.receiver_phase_spectrum import (
    SpectrumConfig,
    analyze_trace_pair,
    compare_line_box_cases,
    load_campaign_trace_pairs,
    load_observation_npz_pair,
    load_residual_csv,
    load_residual_npz,
    write_analysis_outputs,
)


def _comma_list(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze receiver FFT amplitude, spectral phase, cross-correlation delay, "
            "temporal amplitudes, and two-component polarization from validated "
            "vector-array exports."
        )
    )
    parser.add_argument(
        "--geometry",
        choices=("auto", "line", "line-box"),
        default="auto",
        help=(
            "Analysis scope. line performs a genuine line-only analysis and "
            "does not expect box traces; line-box requires both geometries."
        ),
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Campaign root containing inverse/vector_array. Repeat for line and box."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        help=(
            "Standalone vector_array_residual_series NPZ or long residual CSV. "
            "Use --geometry-kind and optionally --residual-family."
        ),
    )
    parser.add_argument("--reference-traces", type=Path)
    parser.add_argument("--observed-traces", type=Path)
    parser.add_argument("--centers-csv", type=Path)
    parser.add_argument("--campaign-name", default="receiver_case")
    parser.add_argument(
        "--geometry-kind", choices=("line", "box", "unknown"), default="unknown"
    )
    parser.add_argument(
        "--residual-family",
        default="reflection,transmission",
        help="Comma-separated reflection/transmission families for campaign mode.",
    )
    parser.add_argument(
        "--standalone-family",
        default="unknown",
        help="Family label for --input or paired --reference-traces/--observed-traces.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-frequency", type=float)
    parser.add_argument("--source-frequency-tolerance", type=float, default=0.10)
    parser.add_argument(
        "--source-num-periods",
        type=float,
        help="Hann-sine source duration in carrier periods; inferred from campaign metadata when possible.",
    )
    parser.add_argument(
        "--source-t0",
        type=float,
        help="Hann-sine source start time; inferred from campaign metadata when possible.",
    )
    parser.add_argument(
        "--source-band-amplitude-threshold",
        type=float,
        default=0.05,
        help="Relative source-spectrum amplitude defining the effective source band.",
    )
    parser.add_argument("--phase-amplitude-threshold", type=float, default=0.05)
    parser.add_argument(
        "--significant-amplitude-threshold", type=float, default=0.10
    )
    parser.add_argument(
        "--amplitude-ratio-reference-floor-relative",
        type=float,
        default=1.0e-4,
        help="Suppress component amplitude ratios when the reference RMS is below this fraction of the receiver vector RMS.",
    )
    parser.add_argument(
        "--amplitude-ratio-reference-floor-absolute",
        type=float,
        default=1.0e-30,
        help="Absolute RMS floor used in addition to the relative amplitude-ratio floor.",
    )
    parser.add_argument("--window", choices=("hann", "none"), default="hann")
    parser.add_argument("--no-detrend", action="store_true")
    parser.add_argument("--max-lag-seconds", type=float)
    parser.add_argument("--max-plot-frequency", type=float)
    parser.add_argument(
        "--require-final-time",
        type=float,
        default=20.0,
        help="Required campaign final time. Use a negative value to disable.",
    )
    parser.add_argument(
        "--expected-geometries",
        type=_comma_list,
        default=None,
        help="Optional expected geometries, e.g. line,box. Missing ones are reported.",
    )
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--no-npz", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _validate_input_mode(args: argparse.Namespace) -> None:
    if args.expected_geometries is not None:
        expected = set(args.expected_geometries)
        if args.geometry == "line" and expected != {"line"}:
            raise ValueError(
                "--geometry line is compatible only with --expected-geometries line."
            )
        if args.geometry == "line-box" and expected != {"line", "box"}:
            raise ValueError(
                "--geometry line-box requires --expected-geometries line,box."
            )
    modes = 0
    if args.campaign_dir:
        modes += 1
    if args.input is not None:
        modes += 1
    if args.reference_traces is not None or args.observed_traces is not None:
        modes += 1
        if args.reference_traces is None or args.observed_traces is None:
            raise ValueError(
                "--reference-traces and --observed-traces must be provided together."
            )
    if modes != 1:
        raise ValueError(
            "Choose exactly one input mode: --campaign-dir, --input, or the paired "
            "--reference-traces/--observed-traces options."
        )


def _load_inputs(args: argparse.Namespace) -> tuple[list[Any], list[dict[str, Any]]]:
    pairs = []
    missing_inputs: list[dict[str, Any]] = []
    require_final_time = (
        None if args.require_final_time is not None and args.require_final_time < 0 else args.require_final_time
    )
    if args.campaign_dir:
        families = _comma_list(args.residual_family)
        contexts = []
        for campaign_dir in args.campaign_dir:
            loaded, context = load_campaign_trace_pairs(
                campaign_dir,
                residual_families=families,
                require_final_time=require_final_time,
            )
            pairs.extend(loaded)
            contexts.append(context)
            for path in context.get("missing_inputs", []):
                missing_inputs.append(
                    {
                        "status": "missing_input",
                        "campaign_dir": str(Path(campaign_dir).expanduser().resolve()),
                        "path": path,
                    }
                )
        expected_geometries = args.expected_geometries
        if expected_geometries is None:
            if args.geometry == "line":
                expected_geometries = ("line",)
            elif args.geometry == "line-box":
                expected_geometries = ("line", "box")
        if expected_geometries:
            present = {pair.geometry_kind for pair in pairs}
            for geometry in expected_geometries:
                if geometry not in present:
                    missing_inputs.append(
                        {
                            "status": "missing_input",
                            "geometry_kind": geometry,
                            "message": f"No {geometry} campaign was provided or loadable.",
                        }
                    )
        return pairs, missing_inputs

    if args.input is not None:
        source = args.input.expanduser().resolve()
        if source.suffix.lower() == ".npz":
            pair = load_residual_npz(
                source,
                case_name=args.campaign_name,
                geometry_kind=args.geometry_kind,
                centers_csv=args.centers_csv,
            )
        elif source.suffix.lower() == ".csv":
            pair = load_residual_csv(
                source,
                case_name=args.campaign_name,
                geometry_kind=args.geometry_kind,
                residual_family=args.standalone_family,
                centers_csv=args.centers_csv,
            )
        else:
            raise ValueError(
                f"Unsupported standalone input format {source.suffix!r}; expected NPZ or CSV."
            )
        pairs.append(pair)
        return pairs, missing_inputs

    pairs.append(
        load_observation_npz_pair(
            args.reference_traces,
            args.observed_traces,
            case_name=args.campaign_name,
            geometry_kind=args.geometry_kind,
            residual_family=args.standalone_family,
        )
    )
    return pairs, missing_inputs



def _resolve_analysis_mode(
    args: argparse.Namespace,
    pairs: list[Any],
) -> str:
    geometries = {str(pair.geometry_kind).lower() for pair in pairs}
    if args.geometry == "line":
        if geometries.difference({"line"}):
            raise ValueError(
                "--geometry line accepts only line trace pairs; "
                f"discovered geometries={sorted(geometries)}."
            )
        return "line-only"
    if args.geometry == "line-box":
        if not args.campaign_dir:
            raise ValueError("--geometry line-box requires campaign mode.")
        if not {"line", "box"}.issubset(geometries):
            raise ValueError(
                "--geometry line-box requires both line and box trace pairs; "
                f"discovered geometries={sorted(geometries)}."
            )
        return "line-box"
    if geometries == {"line"}:
        return "line-only"
    if geometries == {"box"}:
        return "box-only"
    if geometries == {"line", "box"}:
        return "line-box"
    return "multi-geometry"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_input_mode(args)
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            if not args.overwrite:
                raise FileExistsError(
                    f"Output directory is not empty: {output_dir}. Use --overwrite."
                )
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pairs, missing_inputs = _load_inputs(args)
        if not pairs:
            empty_mode = "line-only" if args.geometry == "line" else (
                "line-box" if args.geometry == "line-box" else "auto"
            )
            outputs = write_analysis_outputs(
                [],
                output_dir=output_dir,
                analysis_mode=empty_mode,
                missing_inputs=missing_inputs,
                save_pdf=not args.no_pdf,
                save_npz=not args.no_npz,
                command=[sys.executable, sys.argv[0], *(argv or sys.argv[1:])],
            )
            if not args.quiet:
                print("[RECEIVER PHASE SPECTRUM] processed=0 status=missing_input")
                print(f"[RECEIVER PHASE SPECTRUM] manifest={outputs['manifest_json']}")
            return 2

        analysis_mode = _resolve_analysis_mode(args, pairs)
        inferred_source_frequencies = {
            float(pair.metadata["campaign_source_frequency"])
            for pair in pairs
            if pair.metadata.get("campaign_source_frequency") is not None
        }
        inferred_source_num_periods = {
            float(pair.metadata["campaign_source_num_periods"])
            for pair in pairs
            if pair.metadata.get("campaign_source_num_periods") is not None
        }
        inferred_source_t0 = {
            float(pair.metadata["campaign_source_t0"])
            for pair in pairs
            if pair.metadata.get("campaign_source_t0") is not None
        }
        expected_source_frequency = args.expected_source_frequency
        if expected_source_frequency is None and len(inferred_source_frequencies) == 1:
            expected_source_frequency = inferred_source_frequencies.pop()
        source_num_periods = args.source_num_periods
        if source_num_periods is None and len(inferred_source_num_periods) == 1:
            source_num_periods = inferred_source_num_periods.pop()
        if source_num_periods is None:
            source_num_periods = 1.0
        source_t0 = args.source_t0
        if source_t0 is None and len(inferred_source_t0) == 1:
            source_t0 = inferred_source_t0.pop()
        if source_t0 is None:
            source_t0 = 0.0
        config = SpectrumConfig(
            expected_source_frequency=expected_source_frequency,
            source_frequency_tolerance=args.source_frequency_tolerance,
            source_num_periods=source_num_periods,
            source_t0=source_t0,
            source_band_amplitude_threshold=args.source_band_amplitude_threshold,
            phase_amplitude_threshold=args.phase_amplitude_threshold,
            significant_amplitude_threshold=args.significant_amplitude_threshold,
            amplitude_ratio_reference_floor_relative=(
                args.amplitude_ratio_reference_floor_relative
            ),
            amplitude_ratio_reference_floor_absolute=(
                args.amplitude_ratio_reference_floor_absolute
            ),
            window=args.window,
            detrend=not args.no_detrend,
            max_lag_seconds=args.max_lag_seconds,
            max_plot_frequency=args.max_plot_frequency,
        )
        analyses = []
        for pair in pairs:
            if not args.quiet:
                print(
                    "[RECEIVER PHASE SPECTRUM] "
                    f"{pair.case_name}: {pair.residual_family} / {pair.geometry_kind}"
                )
            analyses.append(analyze_trace_pair(pair, config))
        comparisons = (
            compare_line_box_cases(analyses)
            if analysis_mode == "line-box"
            else []
        )
        command = [sys.executable, sys.argv[0], *(argv or sys.argv[1:])]
        outputs = write_analysis_outputs(
            analyses,
            output_dir=output_dir,
            analysis_mode=analysis_mode,
            line_box_comparisons=comparisons,
            missing_inputs=missing_inputs,
            save_pdf=not args.no_pdf,
            save_npz=not args.no_npz,
            command=command,
        )
        warning_count = sum(
            analysis.summary["status"] != "ok" for analysis in analyses
        )
        if not args.quiet:
            print(
                "[RECEIVER PHASE SPECTRUM] "
                f"processed={len(analyses)} warnings={warning_count} "
                f"missing={len(missing_inputs)} output={output_dir}"
            )
            print(f"[RECEIVER PHASE SPECTRUM] summary_csv={outputs['summary_csv']}")
            print(f"[RECEIVER PHASE SPECTRUM] manifest={outputs['manifest_json']}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
