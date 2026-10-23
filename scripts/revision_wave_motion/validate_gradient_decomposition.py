from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any

import numpy as np

from elastic_fwi.postprocessing.gradient_validation import (
    MATERIAL_PARAMETERS,
    validate_gradient_decomposition,
)
from elastic_fwi.postprocessing.signed_sensitivity_fields import (
    AnalysisConfig,
    discover_signed_fields,
)


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _final_time(payload: dict[str, Any]) -> float:
    temporal = payload.get("temporal_sampling", {})
    config = payload.get("config", {})
    if not isinstance(temporal, dict):
        temporal = {}
    if not isinstance(config, dict):
        config = {}
    dt = float(temporal.get("dt", config.get("dt", np.nan)))
    steps = int(temporal.get("num_steps", config.get("num_steps", 0)))
    return float(temporal.get("T_final", dt * steps))


def run_decomposition_validation(
    *,
    campaign_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-18,
    geometry_atol: float = 1.0e-12,
    acceptance_relative_l2_tolerance: float = 1.0e-6,
    acceptance_relative_linf_tolerance: float = 1.0e-6,
    input_semantics: str = "l2-density",
) -> dict[str, Any]:
    payload = _read_manifest(manifest_path)
    final_time = _final_time(payload)
    if not np.isclose(final_time, 20.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"Only T20 campaigns are accepted; observed T_final={final_time}.")

    discovery = discover_signed_fields(
        AnalysisConfig(
            input_dir=campaign_dir,
            output_dir=output_dir / "_unused_decomposition_discovery",
            manifest=manifest_path,
        )
    )
    lookup = {
        (item.observation_mode, item.parameter_name): item
        for item in discovery.fields
    }
    rows: list[dict[str, Any]] = []
    for parameter in MATERIAL_PARAMETERS:
        required = {
            mode: lookup.get((mode, parameter))
            for mode in (
                "classical_full",
                "standard_reflection_only",
                "standard_transmission_only",
            )
        }
        missing = [mode for mode, item in required.items() if item is None]
        if missing:
            rows.append(
                {
                    "parameter_name": parameter,
                    "status": "failed",
                    "pointwise_allclose": False,
                    "message": f"Missing fields: {missing}",
                }
            )
            continue
        field_name = f"gradient_{parameter}"
        report = validate_gradient_decomposition(
            classical_path=required["classical_full"].input_path,  # type: ignore[union-attr]
            reflection_path=required["standard_reflection_only"].input_path,  # type: ignore[union-attr]
            transmission_path=required["standard_transmission_only"].input_path,  # type: ignore[union-attr]
            field_name=field_name,
            rtol=rtol,
            atol=atol,
            geometry_atol=geometry_atol,
            acceptance_relative_l2_tolerance=acceptance_relative_l2_tolerance,
            acceptance_relative_linf_tolerance=acceptance_relative_linf_tolerance,
            input_semantics=input_semantics,
        )
        report["parameter_name"] = parameter
        rows.append(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gradient_decomposition_metrics.csv"
    columns = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    status = "ok" if len(rows) == len(MATERIAL_PARAMETERS) and all(
        row.get("status") == "ok" for row in rows
    ) else "failed"
    manifest = {
        "kind": "gradient_decomposition_campaign_validation",
        "format_version": "1.1",
        "analysis_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "campaign_dir": str(campaign_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "final_time": final_time,
        "rtol": rtol,
        "atol": atol,
        "geometry_atol": geometry_atol,
        "acceptance_relative_l2_tolerance": acceptance_relative_l2_tolerance,
        "acceptance_relative_linf_tolerance": acceptance_relative_linf_tolerance,
        "input_semantics": str(input_semantics),
        "validation_basis": "solver_aware_relative_l2_and_linf",
        "spatial_representation": "DG0_L2_cell_average_density",
        "norm_measure": "cell_area_weighted",
        "strict_allclose_is_diagnostic_only": True,
        "identity": (
            "gradient_density_classical_full = "
            "gradient_density_reflection_only + gradient_density_transmission_only"
        ),
        "residual_interaction_excluded": True,
        "rows": rows,
        "csv_path": str(csv_path),
    }
    manifest_path_out = output_dir / "gradient_decomposition_manifest.json"
    manifest_path_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the pointwise T20 identity classical gradient = reflection "
            "gradient + transmission gradient."
        )
    )
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-18)
    parser.add_argument("--geometry-atol", type=float, default=1.0e-12)
    parser.add_argument(
        "--input-semantics",
        choices=("cell-integral", "l2-density"),
        default="l2-density",
        help=(
            "Use cell-integral for maps stored as DG0 cell integrals "
            "(converted on the fly to DG0/L2 density); use l2-density for "
            "cell-average DG0/L2 maps."
        ),
    )
    parser.add_argument(
        "--acceptance-relative-l2-tolerance",
        type=float,
        default=1.0e-6,
        help=(
            "Principal acceptance threshold for the area-weighted relative L2 "
            "decomposition residual."
        ),
    )
    parser.add_argument(
        "--acceptance-relative-linf-tolerance",
        type=float,
        default=1.0e-6,
        help=(
            "Principal acceptance threshold for the relative pointwise L-infinity "
            "decomposition residual."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    campaign_dir = args.campaign_dir.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else campaign_dir / "gradient_combination_campaign_manifest.json"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else campaign_dir / "gradient_decomposition_validation"
    )
    report = run_decomposition_validation(
        campaign_dir=campaign_dir,
        manifest_path=manifest,
        output_dir=output_dir,
        rtol=args.rtol,
        atol=args.atol,
        geometry_atol=args.geometry_atol,
        acceptance_relative_l2_tolerance=args.acceptance_relative_l2_tolerance,
        acceptance_relative_linf_tolerance=args.acceptance_relative_linf_tolerance,
        input_semantics=args.input_semantics,
    )
    if not args.quiet:
        print(
            f"[GRADIENT DECOMPOSITION] status={report['status']} "
            f"validated={sum(row.get('status') == 'ok' for row in report['rows'])}/5 "
            f"strict={sum(row.get('strict_pointwise_allclose') is True for row in report['rows'])}/5"
        )
        print(f"[GRADIENT DECOMPOSITION] output={output_dir}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
