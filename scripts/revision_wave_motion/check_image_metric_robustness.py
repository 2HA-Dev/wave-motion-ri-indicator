#!/usr/bin/env python3
from __future__ import annotations

"""Check robustness of rankings derived from normalized qualitative images."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Sequence

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from elastic_fwi.postprocessing.image_metric_robustness import (
    DEFAULT_CIRCLE_RADIUS_FACTORS,
    DEFAULT_RANKING_METRICS,
    DEFAULT_THRESHOLDS,
    RobustnessConfig,
    analyze_image_map,
    analyze_ranking_robustness,
    discover_campaign_maps,
    load_campaign_map,
    load_explicit_mask,
    load_reference_rankings,
    load_standalone_maps,
    make_circle_zones,
    make_explicit_mask_zones,
    normalize_metric_names,
    validate_thresholds,
    write_outputs,
)


def _comma_strings(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = []
        for entry in value:
            items.extend(str(entry).split(","))
    return tuple(item.strip() for item in items if item.strip())


def _comma_floats(value: str | Sequence[float] | None) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = []
        for entry in value:
            items.extend(str(entry).split(","))
    return tuple(float(item.strip()) for item in items if item.strip())


def _comma_ints(value: str | Sequence[int] | None) -> tuple[int, ...]:
    return tuple(int(value) for value in _comma_floats(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether rankings derived from normalized qualitative image maps "
            "are stable under threshold and target-zone variations."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--campaign-dir",
        action="append",
        type=Path,
        help=(
            "Campaign root containing gradient_combination_campaign_manifest.json. "
            "Repeat for line and box campaigns."
        ),
    )
    input_group.add_argument(
        "--maps",
        type=Path,
        help="Standalone NPY, NPZ, CSV, or directory of normalized/normalizable maps.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        help="Standalone binary mask in NPY, NPZ, or CSV format. Required with --maps.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--geometry",
        choices=("auto", "line", "line-box"),
        default="auto",
        help=(
            "Campaign scope. line performs a genuine line-only robustness "
            "analysis; line-box requires both geometries. Standalone maps use auto."
        ),
    )
    parser.add_argument(
        "--modalities",
        default="classical_full,standard_reflection_only,standard_transmission_only,residual_interaction",
        help="Campaign modalities to analyze, comma-separated.",
    )
    parser.add_argument(
        "--parameters",
        default="rho,C11,C22,C12,C66",
        help="Material parameters to analyze, comma-separated.",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help="Normalized image thresholds in [0,1], comma-separated.",
    )
    parser.add_argument(
        "--target-zone-mode",
        default="exact,dilated,eroded",
        help="Standalone mask modes: exact,dilated,eroded.",
    )
    parser.add_argument(
        "--dilation-pixels",
        default="1,2,3",
        help="Standalone regular-grid dilation sizes, comma-separated.",
    )
    parser.add_argument(
        "--erosion-pixels",
        default="1,2,3",
        help="Standalone regular-grid erosion sizes, comma-separated.",
    )
    parser.add_argument(
        "--zone-radius-factors",
        default=",".join(str(value) for value in DEFAULT_CIRCLE_RADIUS_FACTORS),
        help=(
            "Campaign target-radius factors. Values below, equal to, and above 1 "
            "represent eroded, exact, and dilated defect zones."
        ),
    )
    parser.add_argument(
        "--metric",
        default=",".join(DEFAULT_RANKING_METRICS),
        help=(
            "Ranking metrics, comma-separated. The first metric is primary. "
            "Thresholded aliases include thresholded_target_energy_ratio and "
            "thresholded_target_mass_ratio. Non-thresholded metrics are evaluated "
            "once per target zone rather than repeated for every threshold."
        ),
    )
    parser.add_argument("--reference-threshold", type=float, default=0.30)
    parser.add_argument("--reference-zone-id", default="exact")
    parser.add_argument(
        "--reference-ranking",
        type=Path,
        help="Optional JSON list or object defining an external reference ranking.",
    )
    parser.add_argument(
        "--ranking-group",
        choices=("auto", "all", "parameter"),
        default="auto",
        help=(
            "Ranking scope. auto compares campaign modalities per parameter and "
            "compares all standalone maps together."
        ),
    )
    parser.add_argument("--false-max-threshold-fraction", type=float, default=0.50)
    parser.add_argument("--false-max-min-distance-factor", type=float, default=0.50)
    parser.add_argument("--fail-top-change-fraction", type=float, default=0.25)
    parser.add_argument("--score-sensitivity-warning", type=float, default=0.50)
    parser.add_argument("--ranking-tie-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--minimum-maps-per-group", type=int, default=2)
    parser.add_argument(
        "--require-final-time",
        type=float,
        help="Optional required value of dt*num_steps for every campaign.",
    )
    parser.add_argument("--defect-center-x", type=float)
    parser.add_argument("--defect-center-y", type=float)
    parser.add_argument("--defect-radius", type=float)
    parser.add_argument(
        "--figure-format",
        default="png,pdf",
        help="Figure formats, comma-separated: png,pdf.",
    )
    parser.add_argument("--csv-name", default="image_metric_robustness.csv")
    parser.add_argument(
        "--summary-name", default="image_metric_robustness_summary.json"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _defect_override(args: argparse.Namespace) -> tuple[tuple[float, float] | None, float | None]:
    supplied = [
        args.defect_center_x is not None,
        args.defect_center_y is not None,
        args.defect_radius is not None,
    ]
    if any(supplied) and not all(supplied):
        raise ValueError(
            "--defect-center-x, --defect-center-y, and --defect-radius must be supplied together."
        )
    if all(supplied):
        return (
            (float(args.defect_center_x), float(args.defect_center_y)),
            float(args.defect_radius),
        )
    return None, None


def _ranking_group(
    *,
    mode: str,
    campaign_id: str,
    geometry_kind: str,
    parameter_name: str,
) -> str:
    if mode == "all":
        return "all"
    if mode == "parameter":
        return parameter_name
    if campaign_id == "standalone":
        return "all"
    return f"{campaign_id}|{geometry_kind}|{parameter_name}"


def _config_from_args(args: argparse.Namespace) -> RobustnessConfig:
    thresholds = validate_thresholds(_comma_floats(args.thresholds))
    metrics = normalize_metric_names(_comma_strings(args.metric))
    if args.fail_top_change_fraction < 0.0 or args.fail_top_change_fraction > 1.0:
        raise ValueError("--fail-top-change-fraction must lie in [0,1].")
    if args.score_sensitivity_warning < 0.0:
        raise ValueError("--score-sensitivity-warning must be non-negative.")
    if args.minimum_maps_per_group < 2:
        raise ValueError("--minimum-maps-per-group must be at least 2.")
    return RobustnessConfig(
        thresholds=thresholds,
        ranking_metrics=metrics,
        reference_threshold=float(args.reference_threshold),
        reference_zone_id=str(args.reference_zone_id),
        false_max_threshold_fraction=float(args.false_max_threshold_fraction),
        false_max_min_distance_factor=float(args.false_max_min_distance_factor),
        fail_top_change_fraction=float(args.fail_top_change_fraction),
        score_sensitivity_warning=float(args.score_sensitivity_warning),
        ranking_tie_tolerance=float(args.ranking_tie_tolerance),
        minimum_maps_per_group=int(args.minimum_maps_per_group),
    )



def _resolve_analysis_mode(
    requested: str,
    inventory: list[dict],
    *,
    campaign_mode: bool,
) -> tuple[str, list[str]]:
    geometries = sorted(
        {
            str(item.get("geometry_kind", "unknown")).lower()
            for item in inventory
            if str(item.get("geometry_kind", "unknown")).strip()
        }
    )
    geometry_set = set(geometries)
    if not campaign_mode:
        if requested != "auto":
            raise ValueError(
                "--geometry line/line-box is only valid with --campaign-dir."
            )
        return "standalone", geometries
    if requested == "line":
        if geometry_set != {"line"}:
            raise ValueError(
                "--geometry line requires exclusively line campaign maps; "
                f"discovered geometries={geometries}."
            )
        return "line-only", geometries
    if requested == "line-box":
        if not {"line", "box"}.issubset(geometry_set):
            raise ValueError(
                "--geometry line-box requires both line and box campaign maps; "
                f"discovered geometries={geometries}."
            )
        extra = geometry_set.difference({"line", "box"})
        if extra:
            raise ValueError(
                "--geometry line-box received additional geometries: "
                f"{sorted(extra)}."
            )
        return "line-box", geometries
    if geometry_set == {"line"}:
        return "line-only", geometries
    if geometry_set == {"box"}:
        return "box-only", geometries
    if geometry_set == {"line", "box"}:
        return "line-box", geometries
    return "multi-geometry", geometries


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            if not args.overwrite:
                raise FileExistsError(
                    f"Output directory is not empty: {output_dir}. Use --overwrite."
                )
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config = _config_from_args(args)
        detail_rows: list[dict] = []
        inventory: list[dict] = []
        defect_center, defect_radius = _defect_override(args)

        if args.campaign_dir:
            modalities = _comma_strings(args.modalities)
            parameters = _comma_strings(args.parameters)
            radius_factors = _comma_floats(args.zone_radius_factors)
            for campaign_dir in args.campaign_dir:
                sources = discover_campaign_maps(
                    campaign_dir,
                    modalities=modalities,
                    parameters=parameters,
                    defect_center=defect_center,
                    defect_radius=defect_radius,
                    require_final_time=args.require_final_time,
                )
                for source in sources:
                    if not source.source_path.exists():
                        raise FileNotFoundError(
                            f"Campaign map listed in the manifest is missing: {source.source_path}"
                        )
                    if not args.quiet:
                        print(
                            "[IMAGE METRIC ROBUSTNESS] "
                            f"{source.campaign_id}: {source.modality} / {source.parameter_name}",
                            flush=True,
                        )
                    image = load_campaign_map(source)
                    zones = make_circle_zones(
                        image.centers,
                        source.defects,
                        radius_factors=radius_factors,
                    )
                    group = _ranking_group(
                        mode=args.ranking_group,
                        campaign_id=image.campaign_id,
                        geometry_kind=image.geometry_kind,
                        parameter_name=image.parameter_name,
                    )
                    detail_rows.extend(
                        analyze_image_map(image, zones, config, ranking_group=group)
                    )
                    inventory.append(
                        {
                            "campaign_id": image.campaign_id,
                            "geometry_kind": image.geometry_kind,
                            "map_name": image.map_name,
                            "parameter_name": image.parameter_name,
                            "modality": image.modality,
                            "source_path": image.source_path,
                            "source_format": image.source_format,
                            "normalization_factor": image.metadata.get(
                                "normalization_factor"
                            ),
                        }
                    )
                    del image
        else:
            if args.mask is None:
                raise ValueError("--mask is required with --maps.")
            images = load_standalone_maps(args.maps)
            if not images:
                raise ValueError("No standalone maps were loaded.")
            first = images[0]
            base_mask = load_explicit_mask(
                args.mask, expected_size=int(first.values.size)
            )
            reference_centers = first.centers
            reference_weights = first.weights
            for image in images:
                if image.values.size != first.values.size:
                    raise ValueError("Standalone maps must use the same number of samples.")
                if not (
                    (image.centers.shape == reference_centers.shape)
                    and (image.weights.shape == reference_weights.shape)
                ):
                    raise ValueError("Standalone maps must share a common discretization.")
                if not (
                    np.allclose(image.centers, reference_centers)
                    and np.allclose(image.weights, reference_weights)
                ):
                    raise ValueError("Standalone maps have incompatible coordinates or weights.")
                zones = make_explicit_mask_zones(
                    image,
                    base_mask,
                    modes=_comma_strings(args.target_zone_mode),
                    dilation_pixels=_comma_ints(args.dilation_pixels),
                    erosion_pixels=_comma_ints(args.erosion_pixels),
                )
                group = _ranking_group(
                    mode=args.ranking_group,
                    campaign_id=image.campaign_id,
                    geometry_kind=image.geometry_kind,
                    parameter_name=image.parameter_name,
                )
                if not args.quiet:
                    print(
                        f"[IMAGE METRIC ROBUSTNESS] standalone: {image.map_name}",
                        flush=True,
                    )
                detail_rows.extend(
                    analyze_image_map(image, zones, config, ranking_group=group)
                )
                inventory.append(
                    {
                        "campaign_id": image.campaign_id,
                        "geometry_kind": image.geometry_kind,
                        "map_name": image.map_name,
                        "parameter_name": image.parameter_name,
                        "modality": image.modality,
                        "source_path": image.source_path,
                        "source_format": image.source_format,
                        "normalization_factor": image.metadata.get(
                            "normalization_factor"
                        ),
                    }
                )

        analysis_mode, geometry_kinds = _resolve_analysis_mode(
            args.geometry,
            inventory,
            campaign_mode=bool(args.campaign_dir),
        )
        references = load_reference_rankings(args.reference_ranking)
        result = analyze_ranking_robustness(
            detail_rows,
            config,
            external_reference_rankings=references or None,
        )
        result.summary["analysis_mode"] = analysis_mode
        result.summary["geometry_kinds"] = geometry_kinds
        result.summary["configuration"] = {
            "thresholds": list(config.thresholds),
            "ranking_metrics": list(config.ranking_metrics),
            "reference_threshold": config.reference_threshold,
            "reference_zone_id": config.reference_zone_id,
            "false_max_threshold_fraction": config.false_max_threshold_fraction,
            "false_max_min_distance_factor": config.false_max_min_distance_factor,
            "fail_top_change_fraction": config.fail_top_change_fraction,
            "score_sensitivity_warning": config.score_sensitivity_warning,
            "ranking_tie_tolerance": config.ranking_tie_tolerance,
            "minimum_maps_per_group": config.minimum_maps_per_group,
            "analysis_mode_requested": args.geometry,
            "ranking_group": args.ranking_group,
            "metric_dependencies": result.summary.get("metric_dependencies", {}),
        }
        result.summary["map_inventory"] = inventory
        outputs = write_outputs(
            result,
            output_dir=output_dir,
            csv_name=args.csv_name,
            summary_name=args.summary_name,
            figure_formats=_comma_strings(args.figure_format),
        )
        status = str(result.summary["status"])
        if not args.quiet:
            if status == "ok":
                print("[IMAGE METRIC ROBUSTNESS] OK")
            elif status == "warning":
                print("[IMAGE METRIC ROBUSTNESS] WARNING: ranking, tie, or score sensitivity detected")
            else:
                print("[IMAGE METRIC ROBUSTNESS] FAIL")
            print(
                "[IMAGE METRIC ROBUSTNESS] "
                f"maps={result.summary['num_maps']} rows={result.summary['num_detail_rows']} "
                f"unique_inversions={result.summary['num_unique_inversion_patterns']} "
                f"raw_inversion_occurrences={result.summary['num_raw_inversion_occurrences']} "
                f"tie_changes={result.summary['num_tie_change_patterns']} output={output_dir}"
            )
            print(f"[IMAGE METRIC ROBUSTNESS] summary={outputs['summary_json']}")
        return 2 if status == "failed" else 0
    except Exception as exc:
        if not args.quiet:
            print(f"[IMAGE METRIC ROBUSTNESS] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
