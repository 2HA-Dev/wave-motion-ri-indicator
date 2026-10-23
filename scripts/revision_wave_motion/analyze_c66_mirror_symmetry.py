#!/usr/bin/env python3
"""Validate the C66 mirror-symmetry theorem on the frozen signed DG0 maps.

The theorem is verified symbolically by ``verify_mirror_symmetry_symbolic.py``.
This script measures its three observable consequences on maps that already
exist: it performs no PDE solve, regenerates no figure and leaves the campaign
directories untouched.

For every (geometry, modality, coefficient) triple it reports

* the mirror-odd energy fraction, which the theorem predicts to be a pure
  discretisation residual for *all* coefficients;
* the axis suppression ratio, which the theorem predicts to be small for
  ``C66`` only;
* the transverse decay exponent near the axis, predicted to be close to 2 for
  ``C66`` and close to 0 for the normal-strain coefficients.

Example
-------

    PYTHONPATH=. python scripts/revision_wave_motion/analyze_c66_mirror_symmetry.py \
        --freeze-dir path/to/reference_campaigns \
        --output-dir path/to/output_directory
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

from elastic_fwi.postprocessing.mirror_symmetry_diagnostics import (
    DEFAULT_AXIS_POSITION,
    DEFAULT_DEFECT_CENTER,
    DEFAULT_DEFECT_RADIUS,
    MirrorDiagnosticsError,
    axis_vanishing_metrics,
    extrapolate_exponent_to_zero_floor,
    mirror_orientation_residual,
    mirror_parity_metrics,
    transverse_decay_metrics,
)
from elastic_fwi.postprocessing.signed_sensitivity_fields import load_vtk_cell_field


FORMAT_VERSION = "c66_mirror_symmetry_v1"

GEOMETRIES = ("line", "box")
MODALITIES = (
    "classical_full",
    "standard_reflection_only",
    "standard_transmission_only",
    "residual_interaction",
)
COEFFICIENTS = ("rho", "C11", "C22", "C12", "C66")

CAMPAIGN_DIRECTORY_TEMPLATE = (
    "final_{geometry}_L73_gmsh_p1_ppw15_near40_T20_dt0p025_everyk1"
    "_mpi32_cg_gamg_both_exports"
)

CSV_COLUMNS = (
    "geometry",
    "modality",
    "coefficient",
    "num_cells",
    "max_abs_value",
    "pairing_distance_median",
    "pairing_distance_p95",
    "odd_l2_fraction",
    "num_axis_cells",
    "num_reference_cells",
    "axis_rms",
    "reference_rms",
    "axis_suppression_ratio",
    "decay_exponent",
    "decay_exponent_stderr",
    "fit_r_squared",
    "floor_corrected_exponent",
    "floor_corrected_exponent_stderr",
    "floor_amplitude",
    "floor_to_peak_ratio",
    "floor_model_converged",
    "num_bins_used",
    "input_path",
)


@dataclass(frozen=True)
class MapReference:
    geometry: str
    modality: str
    coefficient: str
    path: Path
    field_name: str


def _campaign_dir(
    freeze_dir: Path, geometry: str, campaign_dir: Path | None = None
) -> Path:
    """Resolve a campaign directory.

    ``campaign_dir``, when given, points straight at one campaign and bypasses
    the frozen-campaign naming convention entirely. This allows the
    diagnostics to run on a campaign outside the reference set, for instance
    the symmetry-reversal campaign, whose directory name encodes different
    parameters.
    """

    if campaign_dir is not None:
        return campaign_dir
    return freeze_dir / CAMPAIGN_DIRECTORY_TEMPLATE.format(geometry=geometry)


def discover_maps(
    freeze_dir: Path,
    geometries: Iterable[str],
    modalities: Iterable[str],
    coefficients: Iterable[str],
    campaign_dir: Path | None = None,
) -> tuple[list[MapReference], list[str]]:
    """Locate the signed DG0 maps, reporting every missing file explicitly."""

    found: list[MapReference] = []
    missing: list[str] = []

    for geometry in geometries:
        signed_root = (
            _campaign_dir(freeze_dir, geometry, campaign_dir)
            / "inverse"
            / "gradient_combinations_raw_signed"
        )
        for modality in modalities:
            for coefficient in coefficients:
                if modality == "residual_interaction":
                    stem = f"residual_interaction_{coefficient}"
                    field_name = f"residual_interaction_{coefficient}"
                else:
                    stem = f"material_parameter_gradients_gradient_{coefficient}"
                    field_name = f"gradient_{coefficient}"

                path = signed_root / modality / "vtk" / f"{stem}.pvd"
                if path.is_file():
                    found.append(
                        MapReference(
                            geometry=geometry,
                            modality=modality,
                            coefficient=coefficient,
                            path=path,
                            field_name=field_name,
                        )
                    )
                else:
                    missing.append(str(path))

    return found, missing


def analyse_map(
    reference: MapReference,
    *,
    axis_position: float,
    defect_center: tuple[float, float],
    span_half_height: float,
    axis_band_half_width: float,
    reference_band_min: float,
    reference_band_max: float,
    fit_min_distance: float,
    fit_max_distance: float,
    max_pairing_distance: float | None,
) -> dict[str, Any]:
    loaded = load_vtk_cell_field(reference.path, reference.field_name)
    centers = np.asarray(loaded.centers, dtype=np.float64)
    values = np.asarray(loaded.values, dtype=np.float64)
    measures = np.asarray(loaded.cell_measures, dtype=np.float64)

    row: dict[str, Any] = {
        "geometry": reference.geometry,
        "modality": reference.modality,
        "coefficient": reference.coefficient,
        "num_cells": int(values.size),
        "max_abs_value": float(np.max(np.abs(values))) if values.size else float("nan"),
        "input_path": str(reference.path),
    }

    parity = mirror_parity_metrics(
        centers,
        values,
        measures,
        axis_position=axis_position,
        max_pairing_distance=max_pairing_distance,
    )
    row.update(
        {
            "pairing_distance_median": parity.pairing_distance_median,
            "pairing_distance_p95": parity.pairing_distance_p95,
            "odd_l2_fraction": parity.odd_l2_fraction,
        }
    )

    vanishing = axis_vanishing_metrics(
        centers,
        values,
        measures,
        axis_position=axis_position,
        defect_center=defect_center,
        axis_band_half_width=axis_band_half_width,
        reference_band_min=reference_band_min,
        reference_band_max=reference_band_max,
        span_half_height=span_half_height,
    )
    row.update(
        {
            "num_axis_cells": vanishing.num_axis_cells,
            "num_reference_cells": vanishing.num_reference_cells,
            "axis_rms": vanishing.axis_rms,
            "reference_rms": vanishing.reference_rms,
            "axis_suppression_ratio": vanishing.axis_suppression_ratio,
        }
    )

    decay = transverse_decay_metrics(
        centers,
        values,
        measures,
        axis_position=axis_position,
        defect_center=defect_center,
        span_half_height=span_half_height,
        fit_min_distance=fit_min_distance,
        fit_max_distance=fit_max_distance,
    )
    row.update(
        {
            "decay_exponent": decay.decay_exponent,
            "decay_exponent_stderr": decay.decay_exponent_stderr,
            "fit_r_squared": decay.fit_r_squared,
            "floor_corrected_exponent": decay.floor_corrected_exponent,
            "floor_corrected_exponent_stderr": decay.floor_corrected_exponent_stderr,
            "floor_amplitude": decay.floor_amplitude,
            "floor_to_peak_ratio": decay.floor_to_peak_ratio,
            "floor_model_converged": decay.floor_model_converged,
            "num_bins_used": decay.num_bins_used,
        }
    )

    row["_decay_profile"] = {
        "bin_centers": list(decay.bin_centers),
        "bin_rms": list(decay.bin_rms),
        "bin_cell_counts": list(decay.bin_cell_counts),
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def _summarise(
    rows: list[dict[str, Any]], expected_vanishing: Iterable[str] = ("C66",)
) -> dict[str, Any]:
    """Contrast C66 with the four other coefficients on the two key metrics."""

    def _collect(predicate) -> list[float]:
        return [
            float(row["axis_suppression_ratio"])
            for row in rows
            if predicate(row) and np.isfinite(row["axis_suppression_ratio"])
        ]

    def _collect_exponent(predicate) -> list[float]:
        # The floor-corrected exponent is the one to trust when it is
        # identifiable; on a flat profile the correction degenerates and the
        # plain log-log slope is the meaningful measurement.
        values: list[float] = []
        for row in rows:
            if not predicate(row):
                continue
            corrected = float(row["floor_corrected_exponent"])
            plain = float(row["decay_exponent"])
            chosen = corrected if np.isfinite(corrected) else plain
            if np.isfinite(chosen):
                values.append(chosen)
        return values

    # Which coefficients are *expected* to vanish on the axis is a property of
    # the source polarisation, not a constant of the problem.  With the
    # mirror-even polarisation (0,1) of the reference campaign it is C66 alone.
    # With the mirror-odd polarisation (1,0) of the symmetry-reversal campaign
    # the parities swap and it becomes C11, C22 and C12 instead. The summary
    # key names below keep the ``c66_`` prefix, which is a format contract;
    # ``expected_vanishing_coefficients`` records what they actually designate
    # so the meaning is never ambiguous.
    vanishing = tuple(expected_vanishing)
    is_c66 = lambda row: row["coefficient"] in vanishing  # noqa: E731
    is_other = lambda row: row["coefficient"] not in vanishing  # noqa: E731

    c66_ratios = _collect(is_c66)
    other_ratios = _collect(is_other)
    c66_exponents = _collect_exponent(is_c66)
    other_exponents = _collect_exponent(is_other)

    odd_fractions = [
        float(row["odd_l2_fraction"])
        for row in rows
        if np.isfinite(row["odd_l2_fraction"])
    ]

    c66_rows = [
        row
        for row in rows
        if is_c66(row)
        and np.isfinite(row["floor_corrected_exponent"])
        and np.isfinite(row["floor_to_peak_ratio"])
    ]
    extrapolation = None
    if len(c66_rows) >= 3:
        try:
            extrapolation = extrapolate_exponent_to_zero_floor(
                np.array([row["floor_to_peak_ratio"] for row in c66_rows]),
                np.array([row["floor_corrected_exponent"] for row in c66_rows]),
            ).as_dict()
        except MirrorDiagnosticsError:
            extrapolation = None

    summary = {
        "num_maps": len(rows),
        "expected_vanishing_coefficients": list(vanishing),
        "c66_exponent_floor_extrapolation": extrapolation,
        "c66_axis_suppression_max": max(c66_ratios) if c66_ratios else None,
        "other_axis_suppression_min": min(other_ratios) if other_ratios else None,
        "axis_suppression_separation": (
            min(other_ratios) / max(c66_ratios)
            if c66_ratios and other_ratios and max(c66_ratios) > 0.0
            else None
        ),
        "c66_decay_exponent_min": min(c66_exponents) if c66_exponents else None,
        "c66_decay_exponent_max": max(c66_exponents) if c66_exponents else None,
        "other_decay_exponent_max_abs": (
            max(abs(value) for value in other_exponents) if other_exponents else None
        ),
        "odd_l2_fraction_max": max(odd_fractions) if odd_fractions else None,
    }

    # Thresholds are deliberately loose on P1 and P3, and tight on P2.
    #
    # P1 is limited by the mesh: gmsh does not generate a mirror-symmetric
    # triangulation, so pairing each cell with the cell nearest its mirror
    # image introduces an O(h |grad S|) error that shows up as a spurious odd
    # component.  A small odd fraction is therefore corroboration, not proof;
    # the sharp evidence for mirror evenness comes from the frozen barycentre
    # and orientation cross-checks reported separately.
    #
    # P3 is limited by the same floor: the measured exponent approaches 2 from
    # below and the residual gap is systematic, not statistical, so the quoted
    # standard errors understate the true uncertainty.  What the check asserts
    # is the qualitative contrast (a clearly positive exponent for C66
    # against a flat profile for the other coefficients), not the value 2.
    checks = {
        "P1_all_kernels_mirror_even": (
            summary["odd_l2_fraction_max"] is not None
            and summary["odd_l2_fraction_max"] < 0.15
        ),
        "P2_axis_vanishing_specific_to_C66": (
            summary["axis_suppression_separation"] is not None
            and summary["axis_suppression_separation"] > 10.0
        ),
        "P3_transverse_decay_specific_to_C66": (
            summary["c66_decay_exponent_min"] is not None
            and summary["c66_decay_exponent_min"] > 1.0
            and summary["c66_decay_exponent_max"] < 2.6
            and summary["other_decay_exponent_max_abs"] is not None
            and summary["other_decay_exponent_max_abs"] < 0.4
        ),
        # The individual exponents sit below 2 in proportion to their own
        # numerical floor.  Regressing them on that floor and reading the
        # intercept is what tests the predicted value itself.
        "P4_exponent_extrapolates_to_two": (
            extrapolation is not None
            and abs(extrapolation["exponent_at_zero_floor"] - 2.0) < 0.3
            and extrapolation["r_squared"] > 0.7
        ),
    }
    summary["checks"] = checks
    summary["all_checks_passed"] = all(checks.values())
    return summary


def frozen_cross_checks(
    signed_metrics_csv: Path | None,
    center_search_csv: Path | None,
    *,
    axis_position: float,
) -> dict[str, Any]:
    """Read the two frozen post-processing tables and test them against the theorem.

    These checks are sharper than anything recomputed here, because both tables
    were produced before the theorem was formulated and neither analysis knows
    about the mirror.  Mirror evenness forces

    * every signed barycentre to sit on the axis, whatever the coefficient;
    * every fitted centre of symmetry to sit on the axis, so that the reported
      displacement of the classical C66 pattern is purely along e2;
    * every dominant angular orientation phi_n to satisfy
      phi_n = 90 degrees modulo 180/n degrees.
    """

    import csv

    report: dict[str, Any] = {}

    if signed_metrics_csv is not None and signed_metrics_csv.is_file():
        offsets: list[dict[str, Any]] = []
        with signed_metrics_csv.open(newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                entry: dict[str, Any] = {
                    "geometry": record.get("geometry_id", ""),
                    "modality": record.get("observation_mode", ""),
                    "coefficient": record.get("parameter_name", ""),
                }
                worst = 0.0
                for kind in ("positive", "negative", "absolute"):
                    raw = record.get(f"{kind}_barycenter_x", "")
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        continue
                    offset = abs(value - float(axis_position))
                    entry[f"{kind}_barycenter_x_offset"] = offset
                    worst = max(worst, offset)
                entry["max_barycenter_x_offset"] = worst
                offsets.append(entry)
        if offsets:
            worst_offsets = [item["max_barycenter_x_offset"] for item in offsets]
            report["signed_barycenters"] = {
                "num_fields": len(offsets),
                "max_barycenter_x_offset": max(worst_offsets),
                "median_barycenter_x_offset": float(np.median(worst_offsets)),
                "rows": offsets,
            }

    if center_search_csv is not None and center_search_csv.is_file():
        centres: list[dict[str, Any]] = []
        with center_search_csv.open(newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                try:
                    mode = int(float(record["dominant_mode"]))
                    orientation = float(record["dominant_orientation_degrees"])
                    offset_x = abs(float(record["offset_x_over_R"]))
                    offset_y = abs(float(record["offset_y_over_R"]))
                except (KeyError, TypeError, ValueError):
                    continue
                centres.append(
                    {
                        "case_name": record.get("case_name", ""),
                        "dominant_mode": mode,
                        "orientation_degrees": orientation,
                        "orientation_mirror_residual_degrees": (
                            mirror_orientation_residual(mode, orientation)
                        ),
                        "abs_offset_x_over_R": offset_x,
                        "abs_offset_y_over_R": offset_y,
                    }
                )
        if centres:
            report["symmetry_centres"] = {
                "num_cases": len(centres),
                "max_abs_offset_x_over_R": max(
                    item["abs_offset_x_over_R"] for item in centres
                ),
                "max_abs_offset_y_over_R": max(
                    item["abs_offset_y_over_R"] for item in centres
                ),
                "max_orientation_mirror_residual_degrees": max(
                    item["orientation_mirror_residual_degrees"] for item in centres
                ),
                "rows": centres,
            }

    checks: dict[str, bool] = {}
    if "signed_barycenters" in report:
        checks["F1_signed_barycentres_lie_on_the_axis"] = (
            report["signed_barycenters"]["max_barycenter_x_offset"] < 0.05
        )
    if "symmetry_centres" in report:
        checks["F2_fitted_centres_lie_on_the_axis"] = (
            report["symmetry_centres"]["max_abs_offset_x_over_R"] < 1.0e-9
        )
        checks["F3_drift_is_purely_along_e2"] = (
            report["symmetry_centres"]["max_abs_offset_y_over_R"] > 0.1
        )
        checks["F4_orientations_are_mirror_compatible"] = (
            report["symmetry_centres"]["max_orientation_mirror_residual_degrees"] < 1.0
        )
    report["checks"] = checks
    report["all_checks_passed"] = bool(checks) and all(checks.values())
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=Path("outputs/revision_wave_motion/final_freeze_20260727"),
        help="Directory holding the frozen line and box campaigns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision_analyses/c66_mirror_symmetry"),
        help="Destination for the CSV and JSON reports (never inside outputs/).",
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=None,
        help=(
            "Analyse one campaign directly, bypassing the frozen-campaign "
            "naming convention. Required for any campaign outside "
            "final_freeze_20260727, such as the polarisation-reversal campaign."
        ),
    )
    parser.add_argument("--geometry", default=",".join(GEOMETRIES))
    parser.add_argument("--modality", default=",".join(MODALITIES))
    parser.add_argument("--coefficient", default=",".join(COEFFICIENTS))
    parser.add_argument(
        "--no-frozen-cross-checks",
        action="store_true",
        help=(
            "Skip the cross-checks against the signed-metrics and "
            "centre-search tables. Use when those tables are not available "
            "for the campaign being analysed."
        ),
    )
    parser.add_argument(
        "--expected-vanishing-coefficients",
        default="C66",
        help=(
            "Comma-separated coefficients expected to vanish on the mirror "
            "axis. C66 for a mirror-even source polarisation (0,1), the frozen "
            "case; C11,C22,C12 for the mirror-odd polarisation (1,0) of the "
            "polarisation-reversal campaign, where the parities swap."
        ),
    )
    parser.add_argument("--axis-position", type=float, default=DEFAULT_AXIS_POSITION)
    parser.add_argument("--defect-center-x", type=float, default=DEFAULT_DEFECT_CENTER[0])
    parser.add_argument("--defect-center-y", type=float, default=DEFAULT_DEFECT_CENTER[1])
    parser.add_argument(
        "--span-half-height",
        type=float,
        default=3.0 * DEFAULT_DEFECT_RADIUS,
        help="Half-height in x2 of the analysis strip, default three defect radii.",
    )
    parser.add_argument("--axis-band-half-width", type=float, default=0.02)
    parser.add_argument("--reference-band-min", type=float, default=0.10)
    parser.add_argument("--reference-band-max", type=float, default=0.30)
    parser.add_argument("--fit-min-distance", type=float, default=0.03)
    parser.add_argument("--fit-max-distance", type=float, default=0.15)
    parser.add_argument(
        "--max-pairing-distance",
        type=float,
        default=None,
        help="Discard mirror pairs farther apart than this, in length units.",
    )
    parser.add_argument(
        "--signed-metrics-csv",
        type=Path,
        default=Path(
            "outputs/Priorité 6 — analyze_signed_sensitivity_fields.py"
            "/final_freeze_20260727/P6_SIGNED_METRICS_LINE_BOX.csv"
        ),
        help="Signed-metrics table, read only, for the signed-barycentre cross-check.",
    )
    parser.add_argument(
        "--center-search-csv",
        type=Path,
        default=Path(
            "outputs/Priorité 7 — analyze_C66_multipolar_pattern.py"
            "/final_freeze_20260727/P7_CENTER_SEARCH_FINAL_CLASSIFICATION.csv"
        ),
        help="Centre-search table, read only, for the centre and orientation checks.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    campaign_dir = (
        args.campaign_dir.expanduser().resolve()
        if args.campaign_dir is not None
        else None
    )
    if campaign_dir is not None and not campaign_dir.is_dir():
        print(f"Campaign directory not found: {campaign_dir}", file=sys.stderr)
        return 2

    freeze_dir = args.freeze_dir.expanduser().resolve()
    if campaign_dir is None and not freeze_dir.is_dir():
        print(f"Freeze directory not found: {freeze_dir}", file=sys.stderr)
        return 2

    # A campaign outside the reference set must not be silently cross-checked
    # against the frozen signed-metrics and centre-search tables, whose
    # defaults point at the reference campaigns.
    if campaign_dir is not None and not args.no_frozen_cross_checks:
        stale = [
            name
            for name, value in (
                ("--signed-metrics-csv", args.signed_metrics_csv),
                ("--center-search-csv", args.center_search_csv),
            )
            if value is not None and "final_freeze_20260727" in str(value)
        ]
        if stale:
            print(
                "Refusing to cross-check a non-frozen campaign against the "
                f"frozen tables still selected for {', '.join(stale)}. Pass the "
                "CSV paths regenerated for this campaign, or "
                "--no-frozen-cross-checks to skip the cross-checks entirely.",
                file=sys.stderr,
            )
            return 2

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not args.overwrite and any(output_dir.iterdir()):
        print(
            f"Output directory {output_dir} is not empty; pass --overwrite.",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    geometries = tuple(item for item in args.geometry.split(",") if item)
    modalities = tuple(item for item in args.modality.split(",") if item)
    coefficients = tuple(item for item in args.coefficient.split(",") if item)

    references, missing = discover_maps(
        freeze_dir, geometries, modalities, coefficients, campaign_dir=campaign_dir
    )
    if missing:
        for path in missing:
            print(f"MISSING_MAP {path}", file=sys.stderr)
    if not references:
        print("No signed map discovered; nothing to do.", file=sys.stderr)
        return 2

    defect_center = (float(args.defect_center_x), float(args.defect_center_y))

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, reference in enumerate(references, start=1):
        label = f"{reference.geometry}/{reference.modality}/{reference.coefficient}"
        print(f"[{index}/{len(references)}] {label}", flush=True)
        try:
            rows.append(
                analyse_map(
                    reference,
                    axis_position=float(args.axis_position),
                    defect_center=defect_center,
                    span_half_height=float(args.span_half_height),
                    axis_band_half_width=float(args.axis_band_half_width),
                    reference_band_min=float(args.reference_band_min),
                    reference_band_max=float(args.reference_band_max),
                    fit_min_distance=float(args.fit_min_distance),
                    fit_max_distance=float(args.fit_max_distance),
                    max_pairing_distance=args.max_pairing_distance,
                )
            )
        except (MirrorDiagnosticsError, OSError, ValueError) as error:
            print(f"  FAILED: {error}", file=sys.stderr)
            failures.append({"case": label, "error": str(error)})

    if not rows:
        print("Every map failed to analyse.", file=sys.stderr)
        return 1

    _write_csv(output_dir / "c66_mirror_symmetry_summary.csv", rows)

    expected_vanishing = tuple(
        item for item in args.expected_vanishing_coefficients.split(",") if item
    )
    summary = _summarise(rows, expected_vanishing=expected_vanishing)
    frozen = frozen_cross_checks(
        args.signed_metrics_csv.expanduser().resolve()
        if args.signed_metrics_csv is not None and not args.no_frozen_cross_checks
        else None,
        args.center_search_csv.expanduser().resolve()
        if args.center_search_csv is not None and not args.no_frozen_cross_checks
        else None,
        axis_position=float(args.axis_position),
    )
    payload = {
        "format_version": FORMAT_VERSION,
        "analysis_date_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_dir": str(freeze_dir),
        "axis_position": float(args.axis_position),
        "defect_center": list(defect_center),
        "parameters": {
            "span_half_height": float(args.span_half_height),
            "axis_band_half_width": float(args.axis_band_half_width),
            "reference_band_min": float(args.reference_band_min),
            "reference_band_max": float(args.reference_band_max),
            "fit_min_distance": float(args.fit_min_distance),
            "fit_max_distance": float(args.fit_max_distance),
            "max_pairing_distance": args.max_pairing_distance,
        },
        "missing_maps": missing,
        "failures": failures,
        "summary": summary,
        "frozen_cross_checks": frozen,
        "rows": rows,
    }
    (output_dir / "c66_mirror_symmetry_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("-" * 72)
    print("Recomputed from the signed DG0 maps:")
    for name, passed in summary["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if frozen.get("checks"):
        print("Cross-checked against the frozen post-processing tables:")
        for name, passed in frozen["checks"].items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"Reports written to {output_dir}")

    everything_passed = (
        summary["all_checks_passed"] and frozen.get("all_checks_passed", False)
    )
    return 0 if everything_passed and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
