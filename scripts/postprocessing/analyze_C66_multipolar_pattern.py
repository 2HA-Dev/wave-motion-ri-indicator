#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from elastic_fwi.postprocessing.c66_multipolar_pattern import (
    CaseAnalysis,
    CaseInput,
    CaseMetadata,
    DefectGeometry,
    MultipolarConfig,
    analyze_c66_case,
    discover_c66_cases_from_campaign,
    load_c66_field,
    make_composite_figure,
    write_metrics,
)


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not result:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return result


def _comma_separated_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not result:
        raise argparse.ArgumentTypeError("At least one float is required.")
    return result


def _comma_separated_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the signed normalized C66 multipolar pattern using area-weighted "
            "radial profiles, annular energy, angular Fourier modes, pi-rotation "
            "symmetry, lobe orientation, and local symmetry-center estimation."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        help="Single PVD/PVTU/VTU, NPZ, or CSV C66 field.",
    )
    source.add_argument(
        "--campaign-dir",
        type=Path,
        action="append",
        help=(
            "T20 gradient-combination campaign directory. Repeat for line and box. "
            "C66 paths and defect geometry are read from the manifest."
        ),
    )

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--geometry",
        choices=("auto", "line", "line-box"),
        default="auto",
        help=(
            "Campaign scope. line performs a genuine line-only analysis and "
            "does not expect box cases; line-box preserves the comparative mode."
        ),
    )
    parser.add_argument("--field-name", default=None)
    parser.add_argument("--defect-center-x", type=float, default=None)
    parser.add_argument("--defect-center-y", type=float, default=None)
    parser.add_argument("--defect-radius", type=float, default=None)
    parser.add_argument("--case-name", default=None)
    parser.add_argument(
        "--modality",
        default="classical",
        help="Single-input label: classical, interaction, reflection, or transmission.",
    )
    parser.add_argument(
        "--aperture",
        default="unknown",
        help="Single-input label, normally line or box.",
    )
    parser.add_argument(
        "--observation-mode",
        action="append",
        default=None,
        help=(
            "Campaign mode(s) to analyze. Repeat or use comma-separated values. "
            "Default: classical_full,residual_interaction."
        ),
    )
    parser.add_argument(
        "--expected-apertures",
        type=_comma_separated_strings,
        default=None,
        help=(
            "Campaign-mode apertures expected in the comparison table. "
            "Optional explicit expected apertures. When omitted, --geometry line "
            "implies line and --geometry line-box implies line,box."
        ),
    )
    parser.add_argument(
        "--require-final-time",
        type=float,
        default=20.0,
        help="Reject campaign manifests whose dt*num_steps differs from this value.",
    )
    parser.add_argument("--final-time-tolerance", type=float, default=1.0e-10)

    parser.add_argument("--num-radial-bins", type=int, default=80)
    parser.add_argument("--num-angular-bins", type=int, default=180)
    parser.add_argument(
        "--radial-bin-values",
        type=_comma_separated_ints,
        default=(60, 80, 100),
    )
    parser.add_argument("--annulus-width-factor", type=float, default=0.15)
    parser.add_argument(
        "--annulus-width-values",
        type=_comma_separated_floats,
        default=(0.10, 0.15, 0.20),
    )
    parser.add_argument(
        "--annulus-radius-factors",
        type=_comma_separated_floats,
        default=(1.0, 1.5, 2.0, 2.5, 3.0),
    )
    parser.add_argument("--max-radius-factor", type=float, default=8.0)
    parser.add_argument("--primary-radius-min-factor", type=float, default=0.75)
    parser.add_argument("--primary-radius-max-factor", type=float, default=3.0)
    parser.add_argument("--near-defect-radius-factor", type=float, default=2.0)
    parser.add_argument("--max-fourier-mode", type=int, default=8)
    parser.add_argument("--minimum-mode-power-fraction", type=float, default=0.05)
    parser.add_argument("--minimum-angular-coverage", type=float, default=0.75)

    parser.add_argument("--center-search-extent-factor", type=float, default=0.5)
    parser.add_argument("--center-search-steps", type=int, default=7)
    parser.add_argument("--orientation-stability-degrees", type=float, default=10.0)
    parser.add_argument("--symmetry-stability-tolerance", type=float, default=0.15)
    parser.add_argument("--peak-radius-stability-factor", type=float, default=0.25)
    parser.add_argument("--dominant-mode-agreement-threshold", type=float, default=0.75)
    parser.add_argument(
        "--minimum-valid-robustness-configurations",
        type=int,
        default=2,
        help="Minimum number of interpretable configurations required for consensus.",
    )
    parser.add_argument(
        "--minimum-valid-robustness-fraction",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of requested configurations that must be interpretable. "
            "Under-sampled annuli remain reported but are excluded from mode consensus."
        ),
    )
    parser.add_argument("--center-proximity-factor", type=float, default=0.5)
    parser.add_argument("--max-single-cell-energy-fraction", type=float, default=0.05)
    parser.add_argument(
        "--compare-discretizations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate all radial-bin and annulus-width combinations.",
    )

    parser.add_argument("--save-pdf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _observation_modes(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ("classical_full", "residual_interaction")
    modes: list[str] = []
    for value in values:
        modes.extend(_comma_separated_strings(value))
    return tuple(dict.fromkeys(modes))


def _config(args: argparse.Namespace) -> MultipolarConfig:
    return MultipolarConfig(
        num_radial_bins=args.num_radial_bins,
        num_angular_bins=args.num_angular_bins,
        radial_bin_values=tuple(args.radial_bin_values),
        annulus_width_factor=args.annulus_width_factor,
        annulus_width_values=tuple(args.annulus_width_values),
        annulus_radius_factors=tuple(args.annulus_radius_factors),
        max_radius_factor=args.max_radius_factor,
        primary_radius_min_factor=args.primary_radius_min_factor,
        primary_radius_max_factor=args.primary_radius_max_factor,
        near_defect_radius_factor=args.near_defect_radius_factor,
        max_fourier_mode=args.max_fourier_mode,
        minimum_mode_power_fraction=args.minimum_mode_power_fraction,
        minimum_angular_coverage=args.minimum_angular_coverage,
        center_search_extent_factor=args.center_search_extent_factor,
        center_search_steps=args.center_search_steps,
        orientation_stability_degrees=args.orientation_stability_degrees,
        symmetry_stability_tolerance=args.symmetry_stability_tolerance,
        peak_radius_stability_factor=args.peak_radius_stability_factor,
        dominant_mode_agreement_threshold=args.dominant_mode_agreement_threshold,
        minimum_valid_robustness_configurations=(
            args.minimum_valid_robustness_configurations
        ),
        minimum_valid_robustness_fraction=args.minimum_valid_robustness_fraction,
        center_proximity_factor=args.center_proximity_factor,
        max_single_cell_energy_fraction=args.max_single_cell_energy_fraction,
        compare_discretizations=args.compare_discretizations,
    )


def _single_case(args: argparse.Namespace) -> tuple[CaseInput, ...]:
    missing = [
        name
        for name, value in (
            ("--defect-center-x", args.defect_center_x),
            ("--defect-center-y", args.defect_center_y),
            ("--defect-radius", args.defect_radius),
        )
        if value is None
    ]
    if missing:
        raise ValueError("Single-input mode requires " + ", ".join(missing) + ".")
    field = load_c66_field(args.input, field_name=args.field_name)
    modality = str(args.modality)
    aliases = {
        "classical": "classical_full",
        "interaction": "residual_interaction",
        "reflection": "standard_reflection_only",
        "transmission": "standard_transmission_only",
    }
    observation_mode = aliases.get(modality, modality)
    case_name = args.case_name or f"{args.aperture}_{modality}_C66"
    return (
        CaseInput(
            field=field,
            defect=DefectGeometry(
                center_x=args.defect_center_x,
                center_y=args.defect_center_y,
                radius=args.defect_radius,
            ),
            metadata=CaseMetadata(
                case_name=case_name,
                observation_mode=observation_mode,
                modality=modality,
                aperture=args.aperture,
            ),
        ),
    )


def _campaign_cases(args: argparse.Namespace) -> tuple[CaseInput, ...]:
    cases: list[CaseInput] = []
    for campaign in args.campaign_dir:
        cases.extend(
            discover_c66_cases_from_campaign(
                campaign,
                observation_modes=_observation_modes(args.observation_mode),
                require_final_time=args.require_final_time,
                final_time_tolerance=args.final_time_tolerance,
            )
        )
    names = [case.metadata.case_name for case in cases]
    if len(set(names)) != len(names):
        raise ValueError(
            "Duplicate case names were discovered. Supply campaigns with distinct line/box "
            "apertures or analyze them in separate output directories."
        )
    return tuple(cases)


def _missing_campaign_cases(
    cases: tuple[CaseInput, ...],
    *,
    expected_apertures: tuple[str, ...],
    requested_modes: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    aliases = {
        "classical": "classical_full",
        "interaction": "residual_interaction",
        "reflection": "standard_reflection_only",
        "transmission": "standard_transmission_only",
    }
    labels = {
        "classical_full": "classical",
        "residual_interaction": "interaction",
        "standard_reflection_only": "reflection",
        "standard_transmission_only": "transmission",
    }
    normalized_modes = tuple(aliases.get(mode, mode) for mode in requested_modes)
    present = {(case.metadata.aperture, case.metadata.observation_mode) for case in cases}
    missing: list[dict[str, object]] = []
    for aperture in expected_apertures:
        for mode in normalized_modes:
            if (aperture, mode) in present:
                continue
            modality = labels.get(mode, mode)
            missing.append(
                {
                    "analysis_date_utc": "",
                    "format_version": "c66_multipolar_metrics_v2",
                    "status": "MISSING_CAMPAIGN",
                    "case_name": f"{aperture}_T20_{modality}_C66",
                    "observation_mode": mode,
                    "modality": modality,
                    "aperture": aperture,
                    "final_time": 20.0,
                    "message": (
                        "No matching T20 campaign was supplied or discovered; "
                        "no T26 result was substituted."
                    ),
                }
            )
    return tuple(missing)




def _resolve_campaign_scope(
    args: argparse.Namespace,
    cases: tuple[CaseInput, ...],
) -> tuple[str, tuple[str, ...]]:
    present = {str(case.metadata.aperture).lower() for case in cases}
    explicit = (
        None
        if args.expected_apertures is None
        else tuple(str(value).lower() for value in args.expected_apertures)
    )
    if args.geometry == "line":
        if present.difference({"line"}):
            raise ValueError(
                "--geometry line accepts only line campaigns; "
                f"discovered apertures={sorted(present)}."
            )
        expected = ("line",) if explicit is None else explicit
        if set(expected) != {"line"}:
            raise ValueError(
                "--geometry line is incompatible with expected apertures other than line."
            )
        return "line-only", expected
    if args.geometry == "line-box":
        expected = ("line", "box") if explicit is None else explicit
        if set(expected) != {"line", "box"}:
            raise ValueError(
                "--geometry line-box requires expected apertures line and box."
            )
        return "line-box", expected

    if explicit is None:
        expected = tuple(
            aperture
            for aperture in ("line", "box")
            if aperture in present
        ) or tuple(sorted(present))
    else:
        expected = explicit
    if present == {"line"}:
        mode = "line-only"
    elif present == {"box"}:
        mode = "box-only"
    elif present == {"line", "box"}:
        mode = "line-box"
    else:
        mode = "multi-aperture"
    return mode, expected
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    protected = (
        output_dir / "C66_multipolar_summary.csv",
        output_dir / "C66_multipolar_metrics.json",
    )
    if not args.overwrite and any(path.exists() for path in protected):
        print(
            f"[C66 MULTIPOLAR][ERROR] Outputs already exist in {output_dir}; use --overwrite.",
            file=sys.stderr,
        )
        return 2
    if args.overwrite:
        for path in (
            output_dir / "C66_multipolar_summary.csv",
            output_dir / "C66_multipolar_metrics.json",
            output_dir / "C66_multipolar_comparisons.csv",
        ):
            path.unlink(missing_ok=True)
    try:
        config = _config(args)
        config.validate()
        requested_modes = _observation_modes(args.observation_mode)
        cases = _single_case(args) if args.input is not None else _campaign_cases(args)
        if not cases:
            raise RuntimeError("No C66 case was selected.")
        if args.input is not None:
            aperture = str(cases[0].metadata.aperture).lower()
            if args.geometry == "line-box":
                raise ValueError(
                    "--geometry line-box is not valid with a single input field."
                )
            if args.geometry == "line" and aperture != "line":
                raise ValueError(
                    "--geometry line requires --aperture line in single-input mode."
                )
            analysis_mode = (
                "line-only" if aperture == "line"
                else "box-only" if aperture == "box"
                else "single-case"
            )
            expected_apertures = (aperture,)
        else:
            analysis_mode, expected_apertures = _resolve_campaign_scope(args, cases)
        missing_cases = (
            ()
            if args.input is not None
            else _missing_campaign_cases(
                cases,
                expected_apertures=expected_apertures,
                requested_modes=requested_modes,
            )
        )
        missing_cases = tuple(
            {**dict(item), "analysis_mode": analysis_mode}
            for item in missing_cases
        )
        analyses: list[CaseAnalysis] = []
        for case in cases:
            if not args.quiet:
                print(
                    f"[C66 MULTIPOLAR] {case.metadata.case_name}: "
                    f"{case.metadata.observation_mode} / {case.metadata.aperture}",
                    flush=True,
                )
            analysis = analyze_c66_case(case, config=config)
            analysis.summary["analysis_mode"] = analysis_mode
            identifier = "".join(
                character if character.isalnum() or character in "._-" else "_"
                for character in case.metadata.case_name
            ).strip("_")
            png = output_dir / f"C66_multipolar_{identifier}.png"
            pdf = output_dir / f"C66_multipolar_{identifier}.pdf" if args.save_pdf else None
            make_composite_figure(
                case,
                analysis,
                output_png=png,
                output_pdf=pdf,
                config=config,
            )
            analysis = replace(analysis, figure_png=png, figure_pdf=pdf)
            analysis.summary["figure_png"] = str(png)
            analysis.summary["figure_pdf"] = "" if pdf is None else str(pdf)
            analyses.append(analysis)
        paths = write_metrics(
            output_dir,
            analyses,
            cases,
            config=config,
            missing_cases=missing_cases,
            analysis_mode=analysis_mode,
            save_json=args.save_json,
            save_csv=args.save_csv,
        )
    except Exception as exc:
        print(f"[C66 MULTIPOLAR][ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    warning_count = (
        sum(analysis.summary["status"] != "OK" for analysis in analyses)
        + len(missing_cases)
    )
    if not args.quiet:
        print(
            f"[C66 MULTIPOLAR] processed={len(analyses)} warnings={warning_count} "
            f"missing={len(missing_cases)} output={output_dir}",
            flush=True,
        )
        for item in missing_cases:
            print(
                f"[C66 MULTIPOLAR] missing={item['case_name']}: {item['message']}",
                file=sys.stderr,
            )
        for key, path in sorted(paths.items()):
            print(f"[C66 MULTIPOLAR] {key}={path}")
    return 0 if warning_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
