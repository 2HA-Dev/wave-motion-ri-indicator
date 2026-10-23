#!/usr/bin/env python
"""
Measured scaling exponents of the sensitivity-kernel L2 norms.

PURPOSE
-------
This script produces the scaling exponents reported in the accompanying
article (``tab:scaling_exponents``). It recomputes the exponents from the
reference and contrast-sweep campaigns and writes both the per-coefficient
norms and the fitted exponents to disk.

WHAT IT COMPUTES
----------------
For every campaign of the contrast sweep, every imaging modality and every
material coefficient, the area-weighted L2 norm of the signed DG0 sensitivity
density,

    ||g||_2 = sqrt( sum_K |K| g_K^2 ),

as prescribed by the DG0 L2 sensitivity-density convention of this code.
Cell-integrated values are NOT used: on a locally refined mesh they are
proportional to cell area and would bias the norm toward coarse cells.

The exponent is then a log-log least-squares fit of that norm against the
perturbation amplitude delta = 1 - contrast_factor. The fit is restricted to
the points closest to the reference medium (delta <= 0.10): on the three
coarse points alone (delta >= 0.25) the same fit returns 1.45 and 3.45, which
would wrongly appear to refute the predicted asymptotic scaling. The
restriction is therefore a statement about the asymptotic regime, not a
convenience, and ``--max-delta`` defaults to it.

No PDE is solved. The campaign directories are read only.

USAGE
-----
    PYTHONPATH=. python path/to/extract_contrast_scaling_exponents.py \
        --output-dir path/to/output_directory --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from elastic_fwi.postprocessing.signed_sensitivity_fields import load_vtk_cell_field

# The reference point (contrast 0.75) comes from the reference campaign, the
# six others from the contrast sweep. The absence of contrast_0p75_line from
# the sweep is deliberate, not a gap.
FREEZE_LINE = (
    "outputs/revision_wave_motion/final_freeze_20260727/"
    "final_line_L73_gmsh_p1_ppw15_near40_T20_dt0p025_everyk1_mpi32_cg_gamg_both_exports"
)
SWEEP_ROOT = "outputs/revision_wave_motion/point47_scaling_test"

CONTRASTS: tuple[tuple[str, float], ...] = (
    ("0p25", 0.25),
    ("0p50", 0.50),
    ("0p75", 0.75),
    ("0p90", 0.90),
    ("0p95", 0.95),
    ("0p975", 0.975),
    ("0p9875", 0.9875),
)

COEFFICIENTS = ("rho", "C11", "C22", "C12", "C66")

# Field-name prefix differs between the "true gradient" modes and the
# residual-interaction mode; see export_inverse_time_series_xdmf.py.
MODALITIES: dict[str, str] = {
    "classical_full": "gradient",
    "standard_reflection_only": "gradient",
    "standard_transmission_only": "gradient",
    "residual_interaction": "residual_interaction",
}


def campaign_dir(repo: Path, tag: str) -> Path:
    if tag == "0p75":
        return repo / FREEZE_LINE
    return repo / SWEEP_ROOT / f"contrast_{tag}_line"


def kernel_path(camp: Path, modality: str, coefficient: str) -> Path:
    prefix = MODALITIES[modality]
    stem = (
        f"material_parameter_gradients_{prefix}_{coefficient}"
        if prefix == "gradient"
        else f"{prefix}_{coefficient}"
    )
    return camp / "inverse" / "gradient_combinations_raw_signed" / modality / "vtk" / f"{stem}.pvd"


def field_name(modality: str, coefficient: str) -> str:
    return f"{MODALITIES[modality]}_{coefficient}"


def area_weighted_l2(path: Path, name: str) -> tuple[float, int]:
    """Area-weighted L2 norm of a signed DG0 density, ghost cells removed."""
    loaded = load_vtk_cell_field(path, name)
    values = np.asarray(loaded.values, dtype=np.float64)
    measures = np.asarray(loaded.cell_measures, dtype=np.float64)
    if values.shape != measures.shape:
        raise RuntimeError(f"Shape mismatch for {path}: {values.shape} vs {measures.shape}")
    return float(np.sqrt(np.sum(measures * values * values))), int(values.size)


def loglog_fit(deltas: list[float], norms: list[float]) -> dict[str, float]:
    """Least-squares fit of log(norm) against log(delta); returns slope and R^2."""
    x = np.log(np.asarray(deltas, dtype=np.float64))
    y = np.log(np.asarray(norms, dtype=np.float64))
    n = x.size
    if n < 2:
        raise ValueError("At least two points are required for a log-log fit.")
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan
    return {
        "exponent": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "num_points": int(n),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-delta",
        type=float,
        default=0.10,
        help="Fit only points with delta <= this value (0.10 for Table 4).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = args.output_dir
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    norms: dict[str, dict[str, dict[str, float]]] = {}
    num_cells_seen: set[int] = set()

    for modality in MODALITIES:
        norms[modality] = {}
        for coefficient in COEFFICIENTS:
            norms[modality][coefficient] = {}
            for tag, contrast in CONTRASTS:
                camp = campaign_dir(args.repo_root, tag)
                path = kernel_path(camp, modality, coefficient)
                if not path.exists():
                    raise SystemExit(f"Missing kernel export: {path}")
                value, num_cells = area_weighted_l2(path, field_name(modality, coefficient))
                num_cells_seen.add(num_cells)
                norms[modality][coefficient][tag] = value
                print(
                    f"[NORM] {modality:26s} {coefficient:4s} "
                    f"contrast={contrast:<7g} ||g||_2={value:.9e}",
                    flush=True,
                )

    fits: dict[str, dict[str, dict[str, float]]] = {}
    summary: dict[str, dict[str, float]] = {}
    for modality in MODALITIES:
        fits[modality] = {}
        exponents: list[float] = []
        r_squareds: list[float] = []
        for coefficient in COEFFICIENTS:
            deltas: list[float] = []
            values: list[float] = []
            for tag, contrast in CONTRASTS:
                delta = 1.0 - contrast
                if delta <= args.max_delta + 1e-12:
                    deltas.append(delta)
                    values.append(norms[modality][coefficient][tag])
            fit = loglog_fit(deltas, values)
            fits[modality][coefficient] = fit
            exponents.append(fit["exponent"])
            r_squareds.append(fit["r_squared"])
        summary[modality] = {
            "exponent_min": float(min(exponents)),
            "exponent_max": float(max(exponents)),
            "r_squared_min": float(min(r_squareds)),
        }

    payload = {
        "analysis_date_utc": datetime.now(timezone.utc).isoformat(),
        "format_version": "contrast_scaling_exponents_v1",
        "purpose": "Reproduce Table 4 (tab:scaling_exponents).",
        "norm_definition": "area-weighted L2 of the signed DG0 density: sqrt(sum_K |K| g_K^2)",
        "contract": "DG0/L2 sensitivity-density convention (cell-average density)",
        "max_delta_for_fit": float(args.max_delta),
        "contrasts": {tag: contrast for tag, contrast in CONTRASTS},
        "num_cells": sorted(num_cells_seen),
        "norms": norms,
        "fits": fits,
        "summary": summary,
    }
    (out / "contrast_scaling_exponents.json").write_text(json.dumps(payload, indent=1))

    lines = ["modality,coefficient,contrast_factor,delta,area_weighted_l2_norm"]
    for modality in MODALITIES:
        for coefficient in COEFFICIENTS:
            for tag, contrast in CONTRASTS:
                lines.append(
                    f"{modality},{coefficient},{contrast},{1.0 - contrast:.4f},"
                    f"{norms[modality][coefficient][tag]:.12e}"
                )
    (out / "contrast_scaling_norms.csv").write_text("\n".join(lines) + "\n")

    print("\n" + "-" * 78)
    print(f"Fit restricted to delta <= {args.max_delta}")
    print(f"{'modality':<28}{'exponent range':>22}{'min R^2':>12}")
    print("-" * 78)
    for modality in MODALITIES:
        s = summary[modality]
        rng = f"{s['exponent_min']:.3f} -- {s['exponent_max']:.3f}"
        print(f"{modality:<28}{rng:>22}{s['r_squared_min']:>12.5f}")
    print("-" * 78)
    print(f"Reports written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
