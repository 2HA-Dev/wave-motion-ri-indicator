from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from elastic_fwi.postprocessing.vtk_cell_fields import (
    iter_vtk_cell_field_chunks,
    resolve_vtk_dataset,
)


REQUIRED_COMBINATIONS = (
    "classical_full",
    "standard_reflection_only",
    "standard_transmission_only",
    "residual_interaction",
)

REQUIRED_GRADIENT_FIELDS = (
    "gradient_rho",
    "gradient_C11",
    "gradient_C22",
    "gradient_C12",
    "gradient_C66",
)

REQUIRED_KERNEL_FIELDS = (
    "residual_interaction_rho",
    "residual_interaction_C11",
    "residual_interaction_C22",
    "residual_interaction_C12",
    "residual_interaction_C66",
)


def required_fields_for_combination(combination: str) -> tuple[str, ...]:
    if str(combination) == "residual_interaction":
        return REQUIRED_KERNEL_FIELDS
    return REQUIRED_GRADIENT_FIELDS


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def first_dataset_from_pvd(pvd_path: Path) -> Path:
    root = ET.parse(pvd_path).getroot()
    datasets = root.findall(".//DataSet")
    if not datasets:
        raise RuntimeError(f"No DataSet found in {pvd_path}")
    rel = datasets[0].attrib["file"]
    return (pvd_path.parent / rel).resolve()


def piece_paths_from_pvtu(pvtu_path: Path) -> list[Path]:
    root = ET.parse(pvtu_path).getroot()
    pieces = []
    for piece in root.findall(".//Piece"):
        src = piece.attrib.get("Source")
        if src:
            pieces.append((pvtu_path.parent / src).resolve())
    return pieces


def read_vtu_points_cells_and_celldata(vtu_path: Path, expected_field: str):
    root = ET.parse(vtu_path).getroot()

    points = None
    for da in root.findall(".//Points/DataArray"):
        vals = np.fromstring(da.text or "", sep=" ")
        if vals.size > 0 and vals.size % 3 == 0:
            points = vals.reshape((-1, 3))[:, :2]
            break

    if points is None:
        raise RuntimeError(f"No points found in {vtu_path}")

    connectivity = None
    offsets = None

    for da in root.findall(".//Cells/DataArray"):
        name = da.attrib.get("Name", "")
        vals = np.fromstring(da.text or "", sep=" ", dtype=np.int64)
        if name == "connectivity":
            connectivity = vals
        elif name == "offsets":
            offsets = vals

    if connectivity is None or offsets is None:
        raise RuntimeError(f"No cell connectivity/offsets found in {vtu_path}")

    candidate_names = [
        expected_field,
        "residual_wave_" + expected_field,
    ]

    cell_arrays: dict[str, np.ndarray] = {}
    for cdata in root.findall(".//CellData"):
        for da in cdata.findall("DataArray"):
            name = da.attrib.get("Name", "")
            if not name:
                continue
            vals = np.fromstring(da.text or "", sep=" ")
            if vals.size == 0:
                continue
            ncomp = int(da.attrib.get("NumberOfComponents", "1"))
            if ncomp > 1:
                if vals.size % ncomp != 0:
                    continue
                vals = vals.reshape((-1, ncomp))
                vals = np.linalg.norm(vals, axis=1)
            cell_arrays[name] = vals

    selected_name = None
    selected_values = None

    for name in candidate_names:
        if name in cell_arrays:
            selected_name = name
            selected_values = cell_arrays[name]
            break

    if selected_values is None:
        available = ", ".join(sorted(cell_arrays))
        raise RuntimeError(
            f"Could not find CellData field for {expected_field} in {vtu_path}. "
            f"Available CellData: {available}"
        )

    # Cell centres.
    starts = np.concatenate(([0], offsets[:-1]))
    centers = []
    for start, end in zip(starts, offsets):
        ids = connectivity[int(start):int(end)]
        centers.append(points[ids].mean(axis=0))
    centers = np.vstack(centers)

    if centers.shape[0] != selected_values.shape[0]:
        raise RuntimeError(
            f"Cell/value size mismatch in {vtu_path}: "
            f"{centers.shape[0]} centers vs {selected_values.shape[0]} values."
        )

    return centers, selected_values.astype(float), selected_name


def read_pvd_cell_cloud(pvd_path: Path, expected_field: str):
    """Read one scalar DG0 map with exact cell measures.

    MPI ghost cells are removed by the shared VTK reader.  This is essential on
    locally refined meshes: image metrics must use the actual cell areas rather
    than an average area inferred from the cloud extent.
    """

    all_centers: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    all_measures: list[np.ndarray] = []
    actual_field_name: str | None = None

    for chunk in iter_vtk_cell_field_chunks(
        pvd_path,
        expected_field,
        remove_ghost_cells=True,
        require_finite=True,
    ):
        if actual_field_name is None:
            actual_field_name = chunk.field_name
        elif chunk.field_name != actual_field_name:
            raise RuntimeError(
                f"Inconsistent field names in {pvd_path}: "
                f"{actual_field_name!r} then {chunk.field_name!r}."
            )
        if chunk.num_cells:
            all_centers.append(np.asarray(chunk.centers, dtype=np.float64))
            all_values.append(np.asarray(chunk.values, dtype=np.float64))
            all_measures.append(np.asarray(chunk.cell_measures, dtype=np.float64))

    if not all_values or actual_field_name is None:
        raise RuntimeError(f"No owned DG0 cells found in {pvd_path}.")

    centers = np.concatenate(all_centers, axis=0)
    values = np.concatenate(all_values, axis=0)
    measures = np.concatenate(all_measures, axis=0)
    if centers.shape[0] != values.size or values.size != measures.size:
        raise RuntimeError("DG0 centers, values, and measures have incompatible sizes.")
    if np.any(measures <= 0.0) or not np.all(np.isfinite(measures)):
        raise RuntimeError("DG0 cell measures must be finite and strictly positive.")

    dataset = resolve_vtk_dataset(pvd_path)
    return centers, values, measures, actual_field_name, dataset.dataset_path


def count_false_maxima(
    centers: np.ndarray,
    image: np.ndarray,
    outside_mask: np.ndarray,
    threshold: float,
    min_distance: float,
    max_candidates: int = 5000,
) -> int:
    candidate_indices = np.flatnonzero(outside_mask & (image >= threshold))
    if candidate_indices.size == 0:
        return 0

    if candidate_indices.size > max_candidates:
        strongest = np.argsort(image[candidate_indices])[-max_candidates:]
        candidate_indices = candidate_indices[strongest]

    ordered = candidate_indices[np.argsort(image[candidate_indices])[::-1]]

    selected = []
    for idx in ordered:
        p = centers[int(idx)]
        if not selected:
            selected.append(p)
            continue
        prev = np.vstack(selected)
        d = np.linalg.norm(prev - p[None, :], axis=1)
        if np.all(d >= min_distance):
            selected.append(p)

    return int(len(selected))


def compute_metrics(
    *,
    centers: np.ndarray,
    values: np.ndarray,
    cell_measures: np.ndarray,
    defect_center: tuple[float, float],
    defect_radius: float,
    false_max_threshold_fraction: float,
    false_max_min_distance_factor: float,
) -> dict:
    """Compute image metrics with the physical DG0 area measure.

    Pointwise extrema and localization are unchanged by weighting.  Spatial
    means, quadratic energies, and thresholded areas are true area integrals on
    the non-uniform mesh.
    """

    image = np.abs(values.astype(float))
    measures = np.asarray(cell_measures, dtype=np.float64)
    finite = np.isfinite(image) & np.isfinite(measures) & (measures > 0.0)

    centers = np.asarray(centers, dtype=np.float64)[finite]
    image = image[finite]
    measures = measures[finite]

    if image.size == 0:
        raise RuntimeError("Empty finite image.")

    center = np.array(defect_center, dtype=float)
    distances = np.linalg.norm(centers - center[None, :], axis=1)

    defect_mask = distances <= defect_radius
    outside_mask = ~defect_mask

    eps = 1.0e-300

    global_peak_idx = int(np.argmax(image))
    global_peak_value = float(image[global_peak_idx])
    global_peak_x = float(centers[global_peak_idx, 0])
    global_peak_y = float(centers[global_peak_idx, 1])
    localization_error = float(np.linalg.norm(centers[global_peak_idx] - center))

    area_in = float(np.sum(measures[defect_mask], dtype=np.float64)) if np.any(defect_mask) else 0.0
    area_out = float(np.sum(measures[outside_mask], dtype=np.float64)) if np.any(outside_mask) else 0.0

    peak_in_defect = float(np.max(image[defect_mask])) if np.any(defect_mask) else 0.0
    mean_in_defect = (
        float(np.sum(image[defect_mask] * measures[defect_mask], dtype=np.float64) / area_in)
        if area_in > 0.0 else 0.0
    )
    energy_in_defect = (
        float(np.sum(image[defect_mask] ** 2 * measures[defect_mask], dtype=np.float64))
        if np.any(defect_mask) else 0.0
    )

    peak_outside = float(np.max(image[outside_mask])) if np.any(outside_mask) else 0.0
    mean_outside = (
        float(np.sum(image[outside_mask] * measures[outside_mask], dtype=np.float64) / area_out)
        if area_out > 0.0 else 0.0
    )
    energy_outside = (
        float(np.sum(image[outside_mask] ** 2 * measures[outside_mask], dtype=np.float64))
        if np.any(outside_mask) else 0.0
    )

    peak_to_artifact_ratio = float(peak_in_defect / max(peak_outside, eps))
    contrast_defect_to_background = float(mean_in_defect / max(mean_outside, eps))

    total_energy = energy_in_defect + energy_outside
    energy_fraction_in_defect = float(energy_in_defect / max(total_energy, eps))

    half_max_threshold = 0.5 * global_peak_value
    half_max_area = float(np.sum(measures[image >= half_max_threshold], dtype=np.float64))

    false_max_threshold = false_max_threshold_fraction * global_peak_value
    false_max_min_distance = false_max_min_distance_factor * defect_radius

    false_maxima_count = count_false_maxima(
        centers=centers,
        image=image,
        outside_mask=outside_mask,
        threshold=false_max_threshold,
        min_distance=false_max_min_distance,
    )

    return {
        "num_cells": int(centers.shape[0]),
        "global_peak_value": global_peak_value,
        "global_peak_x": global_peak_x,
        "global_peak_y": global_peak_y,
        "localization_error": localization_error,
        "peak_in_defect": peak_in_defect,
        "peak_outside_defect": peak_outside,
        "peak_to_artifact_ratio": peak_to_artifact_ratio,
        "mean_in_defect": mean_in_defect,
        "mean_outside_defect": mean_outside,
        "contrast_defect_to_background": contrast_defect_to_background,
        "energy_in_defect": energy_in_defect,
        "energy_outside_defect": energy_outside,
        "energy_fraction_in_defect": energy_fraction_in_defect,
        "half_max_threshold": half_max_threshold,
        "half_max_area": half_max_area,
        "false_max_threshold": false_max_threshold,
        "false_maxima_count": false_maxima_count,
        "area_in_defect": area_in,
        "area_outside_defect": area_out,
        "domain_area": area_in + area_out,
        "cell_measure_min": float(np.min(measures)),
        "cell_measure_max": float(np.max(measures)),
        "spatial_measure": "exact_cell_area_weighted",
    }


def extract_found_paths(campaign_manifest: dict) -> dict[str, dict[str, str]]:
    validation = campaign_manifest.get("validation", {})
    found = validation.get("found", {})
    if not isinstance(found, dict):
        raise RuntimeError("Missing validation.found in campaign manifest.")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--campaign-name", type=str, required=True)
    parser.add_argument("--defect-center-x", type=float, default=36.5)
    parser.add_argument("--defect-center-y", type=float, default=36.5)
    parser.add_argument("--defect-radius", type=float, default=0.25)
    parser.add_argument("--false-max-threshold-fraction", type=float, default=0.5)
    parser.add_argument("--false-max-min-distance-factor", type=float, default=1.0)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    campaign_manifest_path = input_dir / "gradient_combination_campaign_manifest.json"
    campaign_manifest = read_json(campaign_manifest_path)
    found = extract_found_paths(campaign_manifest)

    rows = []

    for combination in REQUIRED_COMBINATIONS:
        for field in required_fields_for_combination(combination):
            pvd_path = Path(found[combination][field])

            print(f"[IMAGE METRICS] {combination} / {field} -> {pvd_path}", flush=True)

            centers, values, cell_measures, actual_field_name, data_path = read_pvd_cell_cloud(
                pvd_path,
                expected_field=field,
            )

            metrics = compute_metrics(
                centers=centers,
                values=values,
                cell_measures=cell_measures,
                defect_center=(args.defect_center_x, args.defect_center_y),
                defect_radius=args.defect_radius,
                false_max_threshold_fraction=args.false_max_threshold_fraction,
                false_max_min_distance_factor=args.false_max_min_distance_factor,
            )

            rows.append(
                {
                    "campaign_name": args.campaign_name,
                    "combination": combination,
                    "expected_field": field,
                    "actual_field": actual_field_name,
                    "pvd_path": str(pvd_path),
                    "data_path": str(data_path),
                    **metrics,
                }
            )

    csv_path = output_dir / "gradient_image_quality_metrics.csv"
    json_path = output_dir / "gradient_image_quality_metrics.json"
    manifest_path = output_dir / "gradient_image_quality_metrics_manifest.json"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "status": "ok",
        "campaign_name": args.campaign_name,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "campaign_manifest": str(campaign_manifest_path),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "num_rows": len(rows),
        "defect_center": [args.defect_center_x, args.defect_center_y],
        "defect_radius": args.defect_radius,
        "false_max_threshold_fraction": args.false_max_threshold_fraction,
        "false_max_min_distance_factor": args.false_max_min_distance_factor,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
