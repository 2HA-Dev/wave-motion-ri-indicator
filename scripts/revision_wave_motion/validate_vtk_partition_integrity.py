from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any

import numpy as np

from elastic_fwi.postprocessing.gradient_validation import (
    inspect_vtk_partition_integrity,
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


def _manifest_context(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config", {})
    temporal = payload.get("temporal_sampling", {})
    mesh = payload.get("mesh_discretization", {})
    if not isinstance(config, dict):
        config = {}
    if not isinstance(temporal, dict):
        temporal = {}
    if not isinstance(mesh, dict):
        mesh = {}

    dt = float(temporal.get("dt", config.get("dt", np.nan)))
    num_steps = int(temporal.get("num_steps", config.get("num_steps", 0)))
    final_time = float(temporal.get("T_final", dt * num_steps))
    length_x = float(config.get("domain_length_x", np.nan))
    length_y = float(config.get("domain_length_y", np.nan))
    expected_area = (
        float(length_x * length_y)
        if np.isfinite(length_x) and np.isfinite(length_y)
        else None
    )
    expected_num_cells = mesh.get("global_owned_cells")
    if expected_num_cells is not None:
        expected_num_cells = int(expected_num_cells)
    return {
        "dt": dt,
        "num_steps": num_steps,
        "final_time": final_time,
        "geometry": str(config.get("receiver_array_geometry", "unknown")),
        "expected_area": expected_area,
        "expected_num_cells": expected_num_cells,
    }


def run_partition_validation(
    *,
    campaign_dir: Path,
    manifest_path: Path,
    parameter: str = "C11",
    coordinate_tolerance: float = 1.0e-11,
    area_rtol: float = 1.0e-9,
    area_atol: float = 1.0e-8,
) -> dict[str, Any]:
    payload = _read_manifest(manifest_path)
    context = _manifest_context(payload)
    if not np.isclose(context["final_time"], 20.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"Only T20 campaigns are accepted; observed T_final={context['final_time']}."
        )

    discovery = discover_signed_fields(
        AnalysisConfig(
            input_dir=campaign_dir,
            output_dir=campaign_dir / "_unused_partition_discovery",
            manifest=manifest_path,
        )
    )
    candidates = [
        item
        for item in discovery.fields
        if item.observation_mode == "classical_full"
        and item.parameter_name == parameter
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one classical_full/{parameter} field, found {len(candidates)}."
        )

    report = inspect_vtk_partition_integrity(
        candidates[0].input_path,
        coordinate_tolerance=coordinate_tolerance,
        expected_num_cells=context["expected_num_cells"],
        expected_area=context["expected_area"],
        area_rtol=area_rtol,
        area_atol=area_atol,
    )
    report.update(
        {
            "campaign_dir": str(campaign_dir.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "campaign_final_time": context["final_time"],
            "campaign_dt": context["dt"],
            "campaign_num_steps": context["num_steps"],
            "receiver_geometry": context["geometry"],
            "representative_parameter": parameter,
        }
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate MPI/PVTU cell uniqueness and global area for one T20 "
            "signed-gradient campaign."
        )
    )
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--parameter", default="C11")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--coordinate-tolerance", type=float, default=1.0e-11)
    parser.add_argument("--area-rtol", type=float, default=1.0e-9)
    parser.add_argument("--area-atol", type=float, default=1.0e-8)
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
    report = run_partition_validation(
        campaign_dir=campaign_dir,
        manifest_path=manifest,
        parameter=args.parameter,
        coordinate_tolerance=args.coordinate_tolerance,
        area_rtol=args.area_rtol,
        area_atol=args.area_atol,
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else campaign_dir / "vtk_partition_integrity.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not args.quiet:
        print(
            "[VTK PARTITION] "
            f"geometry={report['receiver_geometry']} status={report['status']} "
            f"cells={report['num_cells_after_ghost_filter']} "
            f"duplicates={report['duplicate_geometry_cells']} "
            f"area_rel={report['area_relative_error']}"
        )
        print(f"[VTK PARTITION] report={output}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
