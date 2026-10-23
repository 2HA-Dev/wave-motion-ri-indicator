from __future__ import annotations

"""Validation of DG0/L2 sensitivity-density semantics on non-uniform meshes.

The raw signed export of a material sensitivity must represent the DG0 cell
average (the L2 density), not the assembled cell contribution.  This module
provides campaign-level diagnostics for all 20 primary maps and can also check
outputs that store per-cell integrals.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import csv
import json
import math

import numpy as np

from elastic_fwi.postprocessing.signed_sensitivity_fields import (
    AnalysisConfig,
    DiscoveredSignedField,
    discover_signed_fields,
)
from elastic_fwi.postprocessing.vtk_cell_fields import (
    CellFieldChunk,
    iter_vtk_cell_field_chunks,
)


FORMAT_VERSION = "dg0_l2_sensitivity_validation_v1"
EXPECTED_MODALITIES = (
    "classical_full",
    "standard_reflection_only",
    "standard_transmission_only",
    "residual_interaction",
)
EXPECTED_PARAMETERS = ("rho", "C11", "C22", "C12", "C66")
INPUT_SEMANTICS = ("cell-integral", "l2-density")


@dataclass(frozen=True)
class DefectGeometry:
    center_x: float
    center_y: float
    radius: float

    @property
    def center(self) -> np.ndarray:
        return np.asarray([self.center_x, self.center_y], dtype=np.float64)


@dataclass(frozen=True)
class FieldRepresentationMetrics:
    num_cells: int
    cell_measure_min: float
    cell_measure_max: float
    cell_measure_ratio: float
    integrated_peak_abs: float
    integrated_peak_x: float
    integrated_peak_y: float
    integrated_peak_distance_to_defect: float
    integrated_peak_cell_measure: float
    density_peak_abs: float
    density_peak_x: float
    density_peak_y: float
    density_peak_distance_to_defect: float
    density_peak_cell_measure: float
    density_peak_inside_defect: bool
    density_peak_inside_defect_abs: float
    density_peak_outside_defect_abs: float
    density_inside_outside_peak_ratio: float


def _finite_positive_measures(measures: np.ndarray) -> None:
    if measures.size == 0:
        raise ValueError("A DG0 field cannot be validated without cells.")
    if not np.all(np.isfinite(measures)) or np.any(measures <= 0.0):
        raise ValueError("DG0 cell measures must be finite and strictly positive.")


def _representations(
    values: np.ndarray,
    measures: np.ndarray,
    *,
    input_semantics: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    measures = np.asarray(measures, dtype=np.float64)
    if values.shape != measures.shape:
        raise ValueError("DG0 values and cell measures must have identical shapes.")
    _finite_positive_measures(measures)

    semantics = str(input_semantics)
    if semantics == "cell-integral":
        integrated = values
        density = values / measures
    elif semantics == "l2-density":
        density = values
        integrated = values * measures
    else:
        raise ValueError(
            f"input_semantics must be one of {INPUT_SEMANTICS}, got {semantics!r}."
        )
    return integrated, density


def compute_field_representation_metrics(
    chunks: Iterable[CellFieldChunk],
    *,
    defect: DefectGeometry,
    input_semantics: str,
) -> FieldRepresentationMetrics:
    center = defect.center
    if not np.isfinite(center).all() or not np.isfinite(defect.radius) or defect.radius <= 0.0:
        raise ValueError("Defect center/radius must be finite and radius strictly positive.")

    num_cells = 0
    measure_min = math.inf
    measure_max = 0.0

    integrated_peak_abs = -1.0
    integrated_peak_xy = np.asarray([math.nan, math.nan])
    integrated_peak_measure = math.nan

    density_peak_abs = -1.0
    density_peak_xy = np.asarray([math.nan, math.nan])
    density_peak_measure = math.nan

    density_inside_peak = 0.0
    density_outside_peak = 0.0

    for chunk in chunks:
        centers = np.asarray(chunk.centers, dtype=np.float64)
        values = np.asarray(chunk.values, dtype=np.float64)
        measures = np.asarray(chunk.cell_measures, dtype=np.float64)

        if centers.ndim != 2 or centers.shape[1] < 2:
            raise ValueError(f"Invalid DG0 cell-center array shape: {centers.shape}.")
        if centers.shape[0] != values.size or values.size != measures.size:
            raise ValueError("DG0 centers, values and cell measures must have matching lengths.")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(centers[:, :2])):
            raise ValueError("DG0 field contains non-finite values or cell centers.")
        _finite_positive_measures(measures)

        integrated, density = _representations(
            values,
            measures,
            input_semantics=input_semantics,
        )
        abs_integrated = np.abs(integrated)
        abs_density = np.abs(density)

        num_cells += int(values.size)
        measure_min = min(measure_min, float(np.min(measures)))
        measure_max = max(measure_max, float(np.max(measures)))

        if values.size:
            i = int(np.argmax(abs_integrated))
            if float(abs_integrated[i]) > integrated_peak_abs:
                integrated_peak_abs = float(abs_integrated[i])
                integrated_peak_xy = centers[i, :2].copy()
                integrated_peak_measure = float(measures[i])

            j = int(np.argmax(abs_density))
            if float(abs_density[j]) > density_peak_abs:
                density_peak_abs = float(abs_density[j])
                density_peak_xy = centers[j, :2].copy()
                density_peak_measure = float(measures[j])

        distances = np.linalg.norm(centers[:, :2] - center[None, :], axis=1)
        inside = distances <= float(defect.radius)
        if np.any(inside):
            density_inside_peak = max(
                density_inside_peak,
                float(np.max(abs_density[inside])),
            )
        if np.any(~inside):
            density_outside_peak = max(
                density_outside_peak,
                float(np.max(abs_density[~inside])),
            )

    if num_cells <= 0:
        raise ValueError("No DG0 cell values were discovered.")

    integrated_distance = float(np.linalg.norm(integrated_peak_xy - center))
    density_distance = float(np.linalg.norm(density_peak_xy - center))
    ratio = (
        float(density_inside_peak / density_outside_peak)
        if density_outside_peak > 0.0
        else math.inf if density_inside_peak > 0.0 else math.nan
    )

    return FieldRepresentationMetrics(
        num_cells=num_cells,
        cell_measure_min=float(measure_min),
        cell_measure_max=float(measure_max),
        cell_measure_ratio=float(measure_max / measure_min),
        integrated_peak_abs=float(integrated_peak_abs),
        integrated_peak_x=float(integrated_peak_xy[0]),
        integrated_peak_y=float(integrated_peak_xy[1]),
        integrated_peak_distance_to_defect=integrated_distance,
        integrated_peak_cell_measure=float(integrated_peak_measure),
        density_peak_abs=float(density_peak_abs),
        density_peak_x=float(density_peak_xy[0]),
        density_peak_y=float(density_peak_xy[1]),
        density_peak_distance_to_defect=density_distance,
        density_peak_cell_measure=float(density_peak_measure),
        density_peak_inside_defect=bool(density_distance <= float(defect.radius)),
        density_peak_inside_defect_abs=float(density_inside_peak),
        density_peak_outside_defect_abs=float(density_outside_peak),
        density_inside_outside_peak_ratio=ratio,
    )


def _resolve_defect_geometry(
    fields: tuple[DiscoveredSignedField, ...],
    *,
    center_x: float | None,
    center_y: float | None,
    radius: float | None,
) -> DefectGeometry:
    if center_x is not None and center_y is not None and radius is not None:
        return DefectGeometry(float(center_x), float(center_y), float(radius))

    for item in fields:
        config = item.configuration
        try:
            cx = float(config["defect_center_x"])
            cy = float(config["defect_center_y"])
            rr = float(config["defect_radius"])
        except (KeyError, TypeError, ValueError):
            continue
        return DefectGeometry(
            float(center_x) if center_x is not None else cx,
            float(center_y) if center_y is not None else cy,
            float(radius) if radius is not None else rr,
        )

    missing = [
        name
        for name, value in (("center_x", center_x), ("center_y", center_y), ("radius", radius))
        if value is None
    ]
    raise ValueError(
        "Defect geometry could not be resolved from the campaign manifest; "
        f"provide explicit values for: {', '.join(missing)}."
    )


def _expected_20_contract(fields: tuple[DiscoveredSignedField, ...]) -> dict[str, Any]:
    keys = {(item.observation_mode, item.parameter_name) for item in fields}
    expected = {(mode, parameter) for mode in EXPECTED_MODALITIES for parameter in EXPECTED_PARAMETERS}
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    duplicates = len(fields) - len(keys)
    return {
        "num_fields": len(fields),
        "num_unique_mode_parameter_pairs": len(keys),
        "missing": [{"observation_mode": m, "parameter_name": p} for m, p in missing],
        "extra": [{"observation_mode": m, "parameter_name": p} for m, p in extra],
        "duplicates": int(duplicates),
        "status": "ok" if len(fields) == 20 and not missing and not extra and duplicates == 0 else "failed",
    }


def validate_campaign_20_maps(
    *,
    campaign_dir: Path,
    output_dir: Path,
    manifest_path: Path | None = None,
    analysis_mode: str = "line-only",
    input_semantics: str = "l2-density",
    defect_center_x: float | None = None,
    defect_center_y: float | None = None,
    defect_radius: float | None = None,
) -> dict[str, Any]:
    campaign_dir = Path(campaign_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else campaign_dir / "gradient_combination_campaign_manifest.json"
    )
    if input_semantics not in INPUT_SEMANTICS:
        raise ValueError(f"input_semantics must be one of {INPUT_SEMANTICS}.")

    patterns = ("line",) if analysis_mode == "line-only" else ()
    discovery = discover_signed_fields(
        AnalysisConfig(
            input_dir=campaign_dir,
            output_dir=output_dir / "_unused_discovery",
            analysis_mode=analysis_mode,
            geometry_patterns=patterns,
            manifest=manifest,
        )
    )
    fields = discovery.fields
    contract = _expected_20_contract(fields)
    defect = _resolve_defect_geometry(
        fields,
        center_x=defect_center_x,
        center_y=defect_center_y,
        radius=defect_radius,
    )

    rows: list[dict[str, Any]] = []
    for item in fields:
        try:
            metrics = compute_field_representation_metrics(
                iter_vtk_cell_field_chunks(
                    item.input_path,
                    (
                        f"residual_interaction_{item.parameter_name}"
                        if item.observation_mode == "residual_interaction"
                        else f"gradient_{item.parameter_name}"
                    ),
                    remove_ghost_cells=True,
                ),
                defect=defect,
                input_semantics=input_semantics,
            )
            row = {
                "status": "ok",
                "message": "",
                "geometry_id": item.geometry_id,
                "campaign_id": item.campaign_id,
                "observation_mode": item.observation_mode,
                "parameter_name": item.parameter_name,
                "input_path": str(item.input_path),
                "input_semantics": input_semantics,
                "density_semantics": "DG0_L2_cell_average_density",
                **metrics.__dict__,
            }
        except Exception as exc:  # preserve one row per discovered map
            row = {
                "status": "failed",
                "message": repr(exc),
                "geometry_id": item.geometry_id,
                "campaign_id": item.campaign_id,
                "observation_mode": item.observation_mode,
                "parameter_name": item.parameter_name,
                "input_path": str(item.input_path),
                "input_semantics": input_semantics,
                "density_semantics": "DG0_L2_cell_average_density",
            }
        rows.append(row)

    rows.sort(key=lambda row: (str(row.get("observation_mode")), str(row.get("parameter_name"))))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dg0_l2_20_map_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rows_ok = len(rows) == 20 and all(row.get("status") == "ok" for row in rows)
    status = "ok" if contract["status"] == "ok" and rows_ok else "failed"
    report = {
        "kind": "dg0_l2_20_map_validation",
        "format_version": FORMAT_VERSION,
        "analysis_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "campaign_dir": str(campaign_dir),
        "manifest_path": str(manifest),
        "analysis_mode": analysis_mode,
        "input_semantics": input_semantics,
        "density_semantics": "DG0_L2_cell_average_density",
        "defect_geometry": {
            "center_x": defect.center_x,
            "center_y": defect.center_y,
            "radius": defect.radius,
        },
        "expected_20_map_contract": contract,
        "num_processed": len(rows),
        "num_ok": sum(row.get("status") == "ok" for row in rows),
        "rows": rows,
        "csv_path": str(csv_path),
        "warnings": list(discovery.warnings),
    }
    json_path = output_dir / "dg0_l2_20_map_validation.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["json_path"] = str(json_path)
    return report
