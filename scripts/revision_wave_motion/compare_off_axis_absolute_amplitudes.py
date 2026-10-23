#!/usr/bin/env python3
"""Compare absolute (unnormalised) sensitivity amplitudes on and off the mirror axis.

Why this script exists.  The image-quality metrics produced by
``extract_gradient_image_quality_metrics.py`` are computed on maps normalised
to a unit maximum, so its ``global_peak_value`` column is identically 1 and its
energy fractions are ratios to a total that itself changes between campaigns.
Those columns cannot separate "the response inside the target fell" from "the
artefact background grew" when two campaigns are compared.  This script reads
the ``gradient_combinations_raw_signed`` fields, which carry the physical
scale, and reports the in-target peak and the out-of-target peak separately for
each combination and each material parameter.

It performs no PDE solve and leaves the campaign directories untouched.

Example
-------
    PYTHONPATH=. python3 scripts/revision_wave_motion/compare_off_axis_absolute_amplitudes.py \
        --reference-dir path/to/on_axis_campaign/inverse/gradient_combinations_raw_signed \
        --reference-defect 36.5 36.5 \
        --candidate-dir path/to/off_axis_campaign/inverse/gradient_combinations_raw_signed \
        --candidate-defect 37.5 36.5 \
        --defect-radius 0.25 \
        --output-dir path/to/output_directory
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from elastic_fwi.postprocessing.signed_sensitivity_fields import load_vtk_cell_field

COMBINATIONS = (
    "classical_full",
    "standard_reflection_only",
    "standard_transmission_only",
    "residual_interaction",
)
PARAMETERS = ("C11", "C12", "C22", "C66", "rho")


def field_path(base: Path, combination: str, parameter: str) -> Path:
    if combination == "residual_interaction":
        stem = f"residual_interaction_{parameter}"
    else:
        stem = f"material_parameter_gradients_gradient_{parameter}"
    return base / combination / "vtk" / f"{stem}.pvd"


def field_name(combination: str, parameter: str) -> str:
    if combination == "residual_interaction":
        return f"residual_interaction_{parameter}"
    return f"gradient_{parameter}"


def measure(path: Path, name: str, cx: float, cy: float, radius: float) -> dict[str, float]:
    """Peak and mean-square density of |field| inside and outside the target disk."""
    field = load_vtk_cell_field(path, name)
    values = np.abs(np.asarray(field.values, dtype=float))
    centers = np.asarray(field.centers, dtype=float)
    weights = np.asarray(field.cell_measures, dtype=float)

    inside = np.hypot(centers[:, 0] - cx, centers[:, 1] - cy) <= radius
    if not inside.any():
        raise RuntimeError(f"No cell centre falls inside the target disk for {path}.")

    area_in = float(weights[inside].sum())
    area_out = float(weights[~inside].sum())
    return {
        "peak_in_target": float(values[inside].max()),
        "peak_outside_target": float(values[~inside].max()),
        "energy_density_in_target": float((values[inside] ** 2 * weights[inside]).sum()) / area_in,
        "energy_density_outside_target": float((values[~inside] ** 2 * weights[~inside]).sum()) / area_out,
        "num_cells_in_target": int(inside.sum()),
        "area_in_target": area_in,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="raw_signed directory of the on-axis reference campaign.")
    parser.add_argument("--reference-defect", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--candidate-dir", type=Path, required=True,
                        help="raw_signed directory of the campaign under test.")
    parser.add_argument("--candidate-defect", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--defect-radius", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for combination in COMBINATIONS:
        for parameter in PARAMETERS:
            name = field_name(combination, parameter)
            ref = measure(field_path(args.reference_dir, combination, parameter), name,
                          *args.reference_defect, args.defect_radius)
            cand = measure(field_path(args.candidate_dir, combination, parameter), name,
                           *args.candidate_defect, args.defect_radius)
            row: dict[str, object] = {"combination": combination, "parameter": parameter}
            row.update({f"reference_{k}": v for k, v in ref.items()})
            row.update({f"candidate_{k}": v for k, v in cand.items()})
            row["target_peak_ratio"] = cand["peak_in_target"] / ref["peak_in_target"]
            row["artefact_peak_ratio"] = cand["peak_outside_target"] / ref["peak_outside_target"]
            row["target_density_ratio"] = (
                cand["energy_density_in_target"] / ref["energy_density_in_target"]
            )
            rows.append(row)
            if not args.quiet:
                print(f"{combination:26s}{parameter:5s} "
                      f"target x{row['target_peak_ratio']:7.3f}  "
                      f"artefact x{row['artefact_peak_ratio']:7.3f}", flush=True)

    csv_path = args.output_dir / "off_axis_absolute_amplitudes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "kind": "off_axis_absolute_amplitude_comparison",
        "reference_dir": str(args.reference_dir),
        "reference_defect": list(args.reference_defect),
        "candidate_dir": str(args.candidate_dir),
        "candidate_defect": list(args.candidate_defect),
        "defect_radius": args.defect_radius,
        "combinations": list(COMBINATIONS),
        "parameters": list(PARAMETERS),
        "csv_path": str(csv_path),
        "num_rows": len(rows),
        "status": "ok",
    }
    (args.output_dir / "off_axis_absolute_amplitudes_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not args.quiet:
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
