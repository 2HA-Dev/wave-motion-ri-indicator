from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.revision_wave_motion.validate_gradient_decomposition import (
    run_decomposition_validation,
)
from elastic_fwi.postprocessing.dg0_l2_sensitivity_validation import (
    INPUT_SEMANTICS,
    validate_campaign_20_maps,
)
from scripts.revision_wave_motion.validate_vtk_partition_integrity import (
    run_partition_validation,
)


DEFAULT_LINE = Path(
    "outputs/revision_wave_motion/"
    "final_line_L73_gmsh_p1_ppw15_near40_T20_dt0p025_mpi32_cg_gamg_both_exports"
)
DEFAULT_BOX = Path(
    "outputs/revision_wave_motion/"
    "final_box_L73_gmsh_p1_ppw15_near40_T20_dt0p025_mpi32_cg_gamg_both_exports"
)
MANIFEST_NAME = "gradient_combination_campaign_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the non-multipolar T20 validation locks for line and box: "
            "VTK partition integrity, 20-map DG0/L2 sensitivity audit, and "
            "pointwise gradient decomposition."
        )
    )
    parser.add_argument("--line-dir", type=Path, default=DEFAULT_LINE)
    parser.add_argument("--box-dir", type=Path, default=DEFAULT_BOX)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/revision_wave_motion/T20_gradient_validation"),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help=(
            "Process the available T20 campaigns and record missing campaigns as "
            "status=partial instead of stopping before analysis. This does not "
            "validate the absent geometry."
        ),
    )
    parser.add_argument("--coordinate-tolerance", type=float, default=1.0e-11)
    parser.add_argument("--area-rtol", type=float, default=1.0e-9)
    parser.add_argument("--area-atol", type=float, default=1.0e-8)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-18)
    parser.add_argument("--geometry-atol", type=float, default=1.0e-12)
    parser.add_argument(
        "--dg0-input-semantics",
        choices=INPUT_SEMANTICS,
        default="l2-density",
        help=(
            "Use l2-density for cell-average DG0/L2 maps; use cell-integral "
            "when the maps are stored as DG0 cell integrals."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _campaign_preflight(
    campaigns: dict[str, Path],
) -> tuple[dict[str, dict[str, Path]], list[dict[str, str]]]:
    """Resolve campaign paths and report absent directories or manifests."""

    available: dict[str, dict[str, Path]] = {}
    missing: list[dict[str, str]] = []
    for label, raw_path in campaigns.items():
        campaign = raw_path.expanduser().resolve()
        manifest = campaign / MANIFEST_NAME
        if not campaign.is_dir():
            missing.append(
                {
                    "campaign_label": label,
                    "campaign_dir": str(campaign),
                    "manifest_path": str(manifest),
                    "reason": "campaign_directory_missing",
                }
            )
            continue
        if not manifest.is_file():
            missing.append(
                {
                    "campaign_label": label,
                    "campaign_dir": str(campaign),
                    "manifest_path": str(manifest),
                    "reason": "campaign_manifest_missing",
                }
            )
            continue
        available[label] = {"campaign": campaign, "manifest": manifest}
    return available, missing


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    if not columns:
        columns = ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_preflight_manifest(
    *,
    output_dir: Path,
    status: str,
    available: dict[str, dict[str, Path]],
    missing: list[dict[str, str]],
    partition_rows: list[dict[str, Any]] | None = None,
    decomposition_rows: list[dict[str, Any]] | None = None,
    dg0_l2_reports: list[dict[str, Any]] | None = None,
) -> Path:
    payload = {
        "kind": "T20_line_box_gradient_validation_suite",
        "format_version": "1.2",
        "analysis_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "scope": {
            "final_time": 20.0,
            "required_geometries": ["line", "box"],
            "priority_7_multipolar_analysis": "explicitly_excluded",
            "residual_interaction_taylor_test": "explicitly_excluded",
        },
        "available_campaigns": {
            label: {
                "campaign_dir": str(paths["campaign"]),
                "manifest_path": str(paths["manifest"]),
            }
            for label, paths in available.items()
        },
        "missing_campaigns": missing,
        "partition_reports": partition_rows or [],
        "decomposition_rows": decomposition_rows or [],
        "dg0_l2_20_map_reports": dg0_l2_reports or [],
    }
    path = output_dir / "t20_gradient_validation_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_campaigns = {"line": args.line_dir, "box": args.box_dir}
    available, missing = _campaign_preflight(requested_campaigns)

    if missing and not args.skip_missing:
        manifest_path = _write_preflight_manifest(
            output_dir=output_dir,
            status="missing_required_campaign",
            available=available,
            missing=missing,
        )
        if not args.quiet:
            print("[T20 GRADIENT SUITE] missing required T20 campaign(s):", file=sys.stderr)
            for item in missing:
                print(
                    f"  - {item['campaign_label']}: {item['reason']} "
                    f"({item['campaign_dir']})",
                    file=sys.stderr,
                )
            print(
                "[T20 GRADIENT SUITE] No T26 campaign will be substituted. "
                "Generate/provide the missing T20 outputs, or rerun with "
                "--skip-missing for a partial validation of the available geometry.",
                file=sys.stderr,
            )
            print(f"[T20 GRADIENT SUITE] manifest={manifest_path}", file=sys.stderr)
        return 2

    if not available:
        manifest_path = _write_preflight_manifest(
            output_dir=output_dir,
            status="no_available_campaign",
            available=available,
            missing=missing,
        )
        if not args.quiet:
            print("[T20 GRADIENT SUITE] no T20 campaign is available.", file=sys.stderr)
            print(f"[T20 GRADIENT SUITE] manifest={manifest_path}", file=sys.stderr)
        return 2

    partition_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    dg0_l2_reports: list[dict[str, Any]] = []
    for expected_geometry, paths in available.items():
        campaign = paths["campaign"]
        manifest = paths["manifest"]
        partition = run_partition_validation(
            campaign_dir=campaign,
            manifest_path=manifest,
            coordinate_tolerance=args.coordinate_tolerance,
            area_rtol=args.area_rtol,
            area_atol=args.area_atol,
        )
        if partition["receiver_geometry"] != expected_geometry and not (
            expected_geometry == "box"
            and partition["receiver_geometry"] == "box_except_gamma1"
        ):
            partition["status"] = "failed"
            partition.setdefault("warnings", []).append(
                f"Expected {expected_geometry} geometry, observed "
                f"{partition['receiver_geometry']}."
            )
        partition["campaign_label"] = expected_geometry
        partition_rows.append(partition)

        dg0_l2 = validate_campaign_20_maps(
            campaign_dir=campaign,
            manifest_path=manifest,
            output_dir=output_dir / expected_geometry / "dg0_l2_20_maps",
            analysis_mode="line-only" if expected_geometry == "line" else "auto",
            input_semantics=args.dg0_input_semantics,
        )
        dg0_l2_reports.append({
            "campaign_label": expected_geometry,
            "status": dg0_l2["status"],
            "num_processed": dg0_l2["num_processed"],
            "num_ok": dg0_l2["num_ok"],
            "input_semantics": dg0_l2["input_semantics"],
            "density_semantics": dg0_l2["density_semantics"],
            "csv_path": dg0_l2["csv_path"],
            "json_path": dg0_l2["json_path"],
        })

        decomposition = run_decomposition_validation(
            campaign_dir=campaign,
            manifest_path=manifest,
            output_dir=output_dir / expected_geometry / "decomposition",
            rtol=args.rtol,
            atol=args.atol,
            geometry_atol=args.geometry_atol,
            input_semantics=args.dg0_input_semantics,
        )
        for row in decomposition["rows"]:
            decomposition_rows.append({"campaign_label": expected_geometry, **row})

        partition_path = output_dir / expected_geometry / "vtk_partition_integrity.json"
        partition_path.parent.mkdir(parents=True, exist_ok=True)
        partition_path.write_text(
            json.dumps(partition, indent=2, sort_keys=True), encoding="utf-8"
        )

    _write_csv(output_dir / "t20_vtk_partition_integrity.csv", partition_rows)
    _write_csv(output_dir / "t20_gradient_decomposition.csv", decomposition_rows)
    _write_csv(output_dir / "t20_dg0_l2_20_map_validation.csv", dg0_l2_reports)

    partition_ok = bool(partition_rows) and all(
        row["status"] == "ok" for row in partition_rows
    )
    decomposition_ok = bool(decomposition_rows) and all(
        row.get("status") == "ok" for row in decomposition_rows
    )
    dg0_l2_ok = bool(dg0_l2_reports) and all(
        row.get("status") == "ok" and row.get("num_processed") == 20
        for row in dg0_l2_reports
    )
    if not partition_ok or not decomposition_ok or not dg0_l2_ok:
        status = "failed"
    elif missing:
        status = "partial"
    else:
        status = "ok"

    manifest_path = _write_preflight_manifest(
        output_dir=output_dir,
        status=status,
        available=available,
        missing=missing,
        partition_rows=partition_rows,
        decomposition_rows=decomposition_rows,
        dg0_l2_reports=dg0_l2_reports,
    )
    if not args.quiet:
        expected_decomposition = 5 * len(available)
        print(
            f"[T20 GRADIENT SUITE] status={status} "
            f"partition={sum(row['status'] == 'ok' for row in partition_rows)}/"
            f"{len(available)} "
            f"decomposition={sum(row.get('status') == 'ok' for row in decomposition_rows)}/"
            f"{expected_decomposition} "
            f"dg0_l2_20_maps={sum(row.get('status') == 'ok' for row in dg0_l2_reports)}/"
            f"{len(available)}"
        )
        if missing:
            labels = ", ".join(item["campaign_label"] for item in missing)
            print(
                f"[T20 GRADIENT SUITE] missing={labels}; partial results do not "
                "validate the absent geometry."
            )
        print(f"[T20 GRADIENT SUITE] manifest={manifest_path}")
    return 0 if status in {"ok", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
