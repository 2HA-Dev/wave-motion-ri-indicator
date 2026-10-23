#!/usr/bin/env python3
"""Unanchored (blind) symmetry-centre scan for the C66 sensitivity kernels.

Context
-------
A robustness sweep of the *size* of the centre-search window is a local
search anchored on the true defect position: it does not test whether the
pattern's centre can be recovered without already knowing the true position.

This script performs that test: for each of the four reference C66 fields
(line/box apertures, classical/residual-interaction constructions), it scans
a symmetry-centre objective on a grid that is built from the fixed geometry
of the receiver arrays alone (never from `defect_center_x`/`defect_center_y`)
and reports the grid-wide maximiser.

The campaign directories are read only: only VTK cell fields are loaded via
the existing, tested `load_c66_field()`. No PDE is solved and no campaign
result is modified.

IMPORTANT CAVEAT (read before interpreting the numbers)
--------------------------------------------------------
For this campaign, `defect_center_x = defect_center_y = 36.5`, exactly the
geometric centre of the domain Omega=[0,73]^2 and of the receiver-array box
[30,43]x[30,43] (Gamma1=[(30,30),(43,30)], Gamma2 likewise on the `line`
aperture), as recorded in the `config` block of the campaign's
`gradient_combination_campaign_manifest.json`.
Consequently *any* grid built from array/domain geometry alone reproduces the
true defect location as its own geometric centre. A scan restricted to this
box cannot demonstrate that the method would find a defect placed elsewhere;
it can only show that the objective used here does not secretly special-case
the true position (the code path below never reads `defect_center_x/y` to
build the candidate grid) and that, on this single already-symmetric
configuration, the unanchored maximiser coincides with the known answer.
Establishing genuine blind-localisation capability requires new campaigns
with the defect placed off that geometric centre.

Objective
---------
Mode-agnostic on purpose: unlike `estimate_symmetry_center()` in
`c66_multipolar_pattern.py` (which weights the search by the dominant angular
mode measured *at* the true centre, itself information derived from
knowing the true location), this scan uses the absolute 180-degree signed
symmetry `abs(signed_symmetry_180)` as the parity score, so that no
mode number derived from an anchored measurement enters the search.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import numpy as np
from scipy.spatial import cKDTree

from elastic_fwi.postprocessing.c66_multipolar_pattern import (
    DefectGeometry,
    MultipolarConfig,
    _analyze_annulus,
    compute_polar_coordinates,
    load_c66_field,
)

CASES: dict[str, dict[str, str]] = {
    "line_classical": {
        "pvd": (
            "outputs/revision_wave_motion/final_freeze_20260727/"
            "final_line_L73_gmsh_p1_ppw15_near40_T20_dt0p025_everyk1_mpi32_cg_gamg_both_exports/"
            "inverse/gradient_combinations_raw_signed/classical_full/vtk/"
            "material_parameter_gradients_gradient_C66.pvd"
        ),
    },
    "line_residual_interaction": {
        "pvd": (
            "outputs/revision_wave_motion/final_freeze_20260727/"
            "final_line_L73_gmsh_p1_ppw15_near40_T20_dt0p025_everyk1_mpi32_cg_gamg_both_exports/"
            "inverse/gradient_combinations_raw_signed/residual_interaction/vtk/"
            "residual_interaction_C66.pvd"
        ),
    },
    "box_classical": {
        "pvd": (
            "outputs/revision_wave_motion/final_freeze_20260727/"
            "final_box_L73_gmsh_p1_ppw15_near40_T20_dt0p025_everyk1_mpi32_cg_gamg_both_exports/"
            "inverse/gradient_combinations_raw_signed/classical_full/vtk/"
            "material_parameter_gradients_gradient_C66.pvd"
        ),
    },
    "box_residual_interaction": {
        "pvd": (
            "outputs/revision_wave_motion/final_freeze_20260727/"
            "final_box_L73_gmsh_p1_ppw15_near40_T20_dt0p025_everyk1_mpi32_cg_gamg_both_exports/"
            "inverse/gradient_combinations_raw_signed/residual_interaction/vtk/"
            "residual_interaction_C66.pvd"
        ),
    },
}

# Receiver-array box geometry, fixed by the experimental set-up
# (Gamma1=[(30,30),(43,30)], Gamma2 on [30,43] as well for `line` and `box`).
# NOT derived from defect_center_x/y; see module docstring.
GRID_XMIN, GRID_XMAX = 30.0, 43.0
GRID_YMIN, GRID_YMAX = 30.0, 43.0

DEFECT_RADIUS = 0.25  # known model parameter (R), not a position
TRUE_CENTER = (36.5, 36.5)  # used ONLY for post-hoc distance reporting


@dataclass(frozen=True)
class ScanConfig:
    # At the true centre, with the narrow default annulus (width_factor=0.15),
    # signed_symmetry_180 reaches ~0.997 (near-perfect), and a local fine probe
    # (step 0.05, +-1 length unit) confirms the peak sits exactly at dx=dy=0:
    # a sharp, narrow feature. Widening the annulus to make a coarse grid
    # tractable (width_factor=1.0) dilutes this peak entirely (score ~0.68 at
    # truth, higher off-centre), so that option is not used. Per-candidate
    # cost with the narrow annulus and a small query radius is ~0.4 ms, so a
    # single-pass FINE grid over the whole box is affordable directly, with
    # no coarse-to-fine staging. Kept identical to the default local-search
    # parameters (center_factor=1.5, width_factor=0.15) so the
    # only thing that changes relative to `estimate_symmetry_center()` is the
    # candidate grid's origin (array/domain geometry, not defect_center_x/y).
    grid_step: float = 0.05              # length units; 0.2R
    annulus_center_factor: float = 1.5   # matches the MultipolarConfig default
    annulus_width_factor: float = 0.15   # matches the MultipolarConfig default
    query_radius: float = 0.5            # KD-tree neighbourhood radius (length units), covers the annulus with margin
    roi_max_factor: float = 3.0          # only affects the unused annular_energy_fraction denominator


def _grid_candidates(step: float) -> np.ndarray:
    xs = np.arange(GRID_XMIN, GRID_XMAX + 1e-9, step)
    ys = np.arange(GRID_YMIN, GRID_YMAX + 1e-9, step)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def scan_case(pvd_path: Path, scan: ScanConfig) -> dict[str, Any]:
    field = load_c66_field(pvd_path)
    centers = np.asarray(field.centers, dtype=np.float64)
    values = np.asarray(field.values, dtype=np.float64)
    measures = np.asarray(field.cell_measures, dtype=np.float64)

    tree = cKDTree(centers)
    candidates = _grid_candidates(scan.grid_step)

    config = MultipolarConfig(
        annulus_width_factor=scan.annulus_width_factor,
        max_radius_factor=scan.roi_max_factor,
    )
    config.validate()

    records: list[dict[str, Any]] = []
    for cx, cy in candidates:
        idx = tree.query_ball_point([cx, cy], r=scan.query_radius)
        if len(idx) < 32:
            continue
        idx = np.asarray(idx, dtype=np.int64)
        local_centers = centers[idx]
        local_values = values[idx]
        local_measures = measures[idx]
        radii, theta = compute_polar_coordinates(local_centers, cx, cy)
        annulus = _analyze_annulus(
            radii,
            theta,
            local_values,
            local_measures,
            defect_radius=DEFECT_RADIUS,
            center_factor=scan.annulus_center_factor,
            width_factor=scan.annulus_width_factor,
            config=config,
        )
        symmetry = float(annulus.get("signed_symmetry_180", math.nan))
        if not math.isfinite(symmetry):
            continue
        parity_score = abs(symmetry)  # mode-agnostic: see module docstring
        records.append(
            {
                "center_x": float(cx),
                "center_y": float(cy),
                "signed_symmetry_180": symmetry,
                "absolute_symmetry_180": float(annulus.get("absolute_symmetry_180", math.nan)),
                "angular_coverage": float(annulus.get("angular_coverage", 0.0)),
                "annular_energy": float(annulus.get("annular_energy", 0.0)),
                "cell_count": int(annulus.get("cell_count", 0)),
                "parity_score": parity_score,
            }
        )

    if not records:
        raise RuntimeError(f"No valid candidate produced a finite score for {pvd_path}")

    max_energy = max(row["annular_energy"] for row in records)
    for row in records:
        normalized_energy = row["annular_energy"] / max_energy if max_energy > 0.0 else 0.0
        row["objective"] = (
            max(0.0, row["parity_score"]) * row["angular_coverage"] * math.sqrt(normalized_energy)
        )

    best = max(records, key=lambda row: row["objective"])
    distance_to_truth = math.hypot(
        best["center_x"] - TRUE_CENTER[0], best["center_y"] - TRUE_CENTER[1]
    )

    return {
        "pvd_path": str(pvd_path),
        "num_candidates_evaluated": len(records),
        "num_candidates_on_grid": int(candidates.shape[0]),
        "grid_step": scan.grid_step,
        "best_center_x": best["center_x"],
        "best_center_y": best["center_y"],
        "best_objective": best["objective"],
        "best_signed_symmetry_180": best["signed_symmetry_180"],
        "best_annular_energy": best["annular_energy"],
        "distance_to_true_center": distance_to_truth,
        "distance_to_true_center_over_R": distance_to_truth / DEFECT_RADIUS,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-step", type=float, default=0.25)
    parser.add_argument("--repo-root", type=Path, default=_REPOSITORY_ROOT)
    args = parser.parse_args()

    scan = ScanConfig(grid_step=args.grid_step)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    for case_name, spec in CASES.items():
        pvd_path = args.repo_root / spec["pvd"]
        print(f"[{case_name}] scanning {pvd_path} ...", flush=True)
        result = scan_case(pvd_path, scan)
        records = result.pop("records")
        summary[case_name] = result
        print(
            f"[{case_name}] best=({result['best_center_x']:.3f},{result['best_center_y']:.3f}) "
            f"distance_to_truth={result['distance_to_true_center_over_R']:.4f} R "
            f"over {result['num_candidates_evaluated']}/{result['num_candidates_on_grid']} candidates",
            flush=True,
        )
        records_path = args.output_dir / f"blind_center_search_{case_name}_records.csv"
        with open(records_path, "w", encoding="utf-8") as fh:
            fh.write(
                "center_x,center_y,signed_symmetry_180,absolute_symmetry_180,"
                "angular_coverage,annular_energy,cell_count,parity_score,objective\n"
            )
            for row in records:
                fh.write(
                    f"{row['center_x']},{row['center_y']},{row['signed_symmetry_180']},"
                    f"{row['absolute_symmetry_180']},{row['angular_coverage']},"
                    f"{row['annular_energy']},{row['cell_count']},{row['parity_score']},"
                    f"{row['objective']}\n"
                )

    with open(args.output_dir / "blind_center_search_summary.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "scan_config": asdict(scan),
                "grid_box": {
                    "xmin": GRID_XMIN, "xmax": GRID_XMAX,
                    "ymin": GRID_YMIN, "ymax": GRID_YMAX,
                },
                "true_center_used_for_reporting_only": TRUE_CENTER,
                "defect_radius": DEFECT_RADIUS,
                "cases": summary,
            },
            fh,
            indent=2,
        )
    print(f"Summary written to {args.output_dir / 'blind_center_search_summary.json'}")


if __name__ == "__main__":
    main()
