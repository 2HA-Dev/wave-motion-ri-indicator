#!/usr/bin/env python3
"""Build the shear-strain factor maps of the C66 kernels and split them.

For each of the two C66 constructions this script recomputes, from the frozen
wavefield snapshots, the three maps that Cauchy--Schwarz separates:

    |S(x)| = rho(x) * sqrt(A_first(x)) * sqrt(A_second(x)),

with ``A_f(x) = int_0^T gamma12(f)^2 dt``.  The classical pair is
``(u, lambda)`` and the residual-interaction pair is ``(W1, Lambda2)``.

The point of the exercise is the extent of each factor along the propagation
coordinate x2.  If the mechanism proposed for the drift is right, the reference
forward field ``u`` spreads over the whole path between the two arrays while the
three residual-driven fields concentrate near the defect, so that the classical
kernel inherits the geometry of the beam and the residual-interaction kernel
does not.

No PDE is solved and the campaign directories are left untouched. The recomputed
kernels are checked against the frozen exported C66 maps, which also fixes the
time convention used to pair the two factors.

Example
-------

    PYTHONPATH=. python scripts/revision_wave_motion/analyze_c66_shear_factor_maps.py \
        --time-stride 20 --output-dir path/to/output_directory
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from elastic_fwi.postprocessing.shear_factor_maps import (
    ShearFactorError,
    cauchy_schwarz_split,
    correlation_against_reference,
    factor_spread,
    marginal_profile_x2,
    triangle_shear_strain,
)
from elastic_fwi.postprocessing.signed_sensitivity_fields import load_vtk_cell_field
from elastic_fwi.postprocessing.vtk_residual_wave_io import (
    read_raw_residual_wave_snapshot,
)


FORMAT_VERSION = "c66_shear_factor_maps_v1"

CAMPAIGN_DIRECTORY_TEMPLATE = (
    "final_{geometry}_L73_gmsh_p1_ppw15_near40_T20_dt0p025_everyk1"
    "_mpi32_cg_gamg_both_exports"
)

# The four snapshot families, in the notation of the accompanying article.
FIELD_U = "direct_simulated_displacement"          # u,        reference forward
FIELD_LAMBDA = "adjoint_displacement"              # lambda,   classical adjoint
FIELD_W1 = "reflection_residual_wavefield"         # W1,       reflection residual
FIELD_LAMBDA2 = "transmission_residual_wavefield"  # Lambda2,  transmission residual

FIELDS = (FIELD_U, FIELD_LAMBDA, FIELD_W1, FIELD_LAMBDA2)

PAIRS = {
    "classical": (FIELD_U, FIELD_LAMBDA),
    "residual_interaction": (FIELD_W1, FIELD_LAMBDA2),
}

FROZEN_C66_MAP = {
    "classical": (
        "classical_full",
        "material_parameter_gradients_gradient_C66",
        "gradient_C66",
    ),
    "residual_interaction": (
        "residual_interaction",
        "residual_interaction_C66",
        "residual_interaction_C66",
    ),
}


@dataclass(frozen=True)
class PieceResult:
    piece: int
    centroids: np.ndarray
    areas: np.ndarray
    maps: dict[str, np.ndarray]


def _resolve_campaign_dir(
    freeze_dir: Path, geometry: str, campaign_dir: Path | None = None
) -> Path:
    """Resolve a campaign directory, bypassing the freeze convention if asked.

    ``campaign_dir`` lets the diagnostics run on a campaign outside the
    reference set, for instance the symmetry-reversal campaign, whose
    directory name encodes different parameters and therefore cannot match
    ``CAMPAIGN_DIRECTORY_TEMPLATE``.
    """

    if campaign_dir is not None:
        return campaign_dir
    return freeze_dir / CAMPAIGN_DIRECTORY_TEMPLATE.format(geometry=geometry)


def _vtk_dir(
    freeze_dir: Path, geometry: str, campaign_dir: Path | None = None
) -> Path:
    return _resolve_campaign_dir(freeze_dir, geometry, campaign_dir) / "inverse" / "vtk"


def _snapshot_path(vtk_dir: Path, field: str, piece: int, step: int) -> Path:
    return vtk_dir / f"{field}_p{piece}_{step:06d}.vtu"


def select_pieces(
    vtk_dir: Path,
    *,
    num_pieces: int,
    roi: tuple[float, float, float, float],
    probe_step: int,
) -> list[int]:
    """Keep only the MPI pieces whose bounding box meets the region of interest.

    The partition is spatially coherent, so this discards most of the data
    without touching the result: a piece disjoint from the region contributes
    nothing to any map restricted to it.
    """

    x_min, x_max, y_min, y_max = roi
    selected: list[int] = []
    for piece in range(num_pieces):
        path = _snapshot_path(vtk_dir, FIELD_U, piece, probe_step)
        if not path.is_file():
            continue
        snapshot = read_raw_residual_wave_snapshot(path, field_name=FIELD_U)
        points = snapshot.points[:, :2]
        if (
            float(np.max(points[:, 0])) >= x_min
            and float(np.min(points[:, 0])) <= x_max
            and float(np.max(points[:, 1])) >= y_min
            and float(np.min(points[:, 1])) <= y_max
        ):
            selected.append(piece)
    return selected


def _process_piece(task: dict[str, Any]) -> PieceResult:
    """Accumulate every factor and pair map for one MPI piece."""

    vtk_dir = Path(task["vtk_dir"])
    piece = int(task["piece"])
    steps = list(task["steps"])
    dt = float(task["dt"])
    roi = tuple(task["roi"])

    reference = read_raw_residual_wave_snapshot(
        _snapshot_path(vtk_dir, FIELD_U, piece, steps[0]), field_name=FIELD_U
    )
    points = reference.points
    triangles = reference.triangles_owned

    geometry = triangle_shear_strain(points, triangles, reference.field)
    centroids = geometry.centroids
    areas = geometry.areas

    x_min, x_max, y_min, y_max = roi
    inside = (
        (centroids[:, 0] >= x_min)
        & (centroids[:, 0] <= x_max)
        & (centroids[:, 1] >= y_min)
        & (centroids[:, 1] <= y_max)
    )
    if not np.any(inside):
        return PieceResult(
            piece=piece,
            centroids=np.zeros((0, 2)),
            areas=np.zeros(0),
            maps={},
        )

    num_cells = int(np.count_nonzero(inside))
    num_steps = len(steps)

    # gamma12 time series per field, restricted to the region of interest.
    series = {
        field: np.empty((num_steps, num_cells), dtype=np.float32) for field in FIELDS
    }
    for index, step in enumerate(steps):
        for field in FIELDS:
            snapshot = read_raw_residual_wave_snapshot(
                _snapshot_path(vtk_dir, field, piece, step), field_name=field
            )
            shear = triangle_shear_strain(points, triangles, snapshot.field)
            series[field][index] = shear.shear_strain[inside].astype(np.float32)

    maps: dict[str, np.ndarray] = {}
    for field in FIELDS:
        values = series[field].astype(np.float64)
        maps[f"energy__{field}"] = dt * np.sum(values * values, axis=0)

    for name, (first_field, second_field) in PAIRS.items():
        first = series[first_field].astype(np.float64)
        second = series[second_field].astype(np.float64)
        maps[f"pair__{name}__as_stored"] = dt * np.sum(first * second, axis=0)
        # The adjoint family may be stored in its forward-in-tau order; pairing
        # against the reversed sequence is the other admissible convention.
        maps[f"pair__{name}__reversed"] = dt * np.sum(first * second[::-1], axis=0)

    return PieceResult(
        piece=piece,
        centroids=centroids[inside],
        areas=areas[inside],
        maps=maps,
    )


def _load_frozen_reference(
    freeze_dir: Path,
    geometry: str,
    pair_name: str,
    centroids: np.ndarray,
    campaign_dir: Path | None = None,
) -> tuple[np.ndarray, float] | None:
    """Sample the exported C66 map at the recomputed cell centroids."""

    from scipy.spatial import cKDTree

    modality, stem, field_name = FROZEN_C66_MAP[pair_name]
    path = (
        _resolve_campaign_dir(freeze_dir, geometry, campaign_dir)
        / "inverse"
        / "gradient_combinations_raw_signed"
        / modality
        / "vtk"
        / f"{stem}.pvd"
    )
    if not path.is_file():
        return None

    loaded = load_vtk_cell_field(path, field_name)
    tree = cKDTree(np.asarray(loaded.centers, dtype=np.float64)[:, :2])
    distances, indices = tree.query(centroids, k=1)
    return np.asarray(loaded.values, dtype=np.float64)[indices], float(
        np.max(distances)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=Path("outputs/revision_wave_motion/final_freeze_20260727"),
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
    parser.add_argument("--geometry", default="line")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision_analyses/c66_shear_factors"),
    )
    parser.add_argument("--num-pieces", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=801)
    parser.add_argument(
        "--time-stride",
        type=int,
        default=1,
        help="Use every k-th snapshot; 1 uses the full every_k=1 export.",
    )
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument(
        "--roi",
        default="29,44,29,44",
        help="x_min,x_max,y_min,y_max; defaults to the band spanned by the arrays.",
    )
    parser.add_argument("--defect-center-x", type=float, default=36.5)
    parser.add_argument("--defect-center-y", type=float, default=36.5)
    parser.add_argument("--defect-radius", type=float, default=0.25)
    parser.add_argument(
        "--local-window-radii",
        default="3,8,20",
        help="Analysis windows around the defect, in defect radii.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--profile-bins", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    freeze_dir = args.freeze_dir.expanduser().resolve()
    campaign_dir = (
        args.campaign_dir.expanduser().resolve()
        if args.campaign_dir is not None
        else None
    )
    vtk_dir = _vtk_dir(freeze_dir, args.geometry, campaign_dir)
    if not vtk_dir.is_dir():
        print(f"Snapshot directory not found: {vtk_dir}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not args.overwrite and any(output_dir.iterdir()):
        print(f"{output_dir} is not empty; pass --overwrite.", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        roi = tuple(float(value) for value in args.roi.split(","))
        if len(roi) != 4:
            raise ValueError
    except ValueError:
        print("--roi must be four comma-separated numbers.", file=sys.stderr)
        return 2

    steps = list(range(0, int(args.num_steps), max(1, int(args.time_stride))))
    effective_dt = float(args.dt) * max(1, int(args.time_stride))

    print(f"Selecting MPI pieces meeting {roi} ...", flush=True)
    pieces = select_pieces(
        vtk_dir, num_pieces=int(args.num_pieces), roi=roi, probe_step=steps[0]
    )
    if not pieces:
        print("No MPI piece meets the region of interest.", file=sys.stderr)
        return 2
    print(f"  {len(pieces)} of {args.num_pieces} pieces retained: {pieces}", flush=True)
    print(f"Reading {len(steps)} snapshots x {len(FIELDS)} fields ...", flush=True)

    tasks = [
        {
            "vtk_dir": str(vtk_dir),
            "piece": piece,
            "steps": steps,
            "dt": effective_dt,
            "roi": list(roi),
        }
        for piece in pieces
    ]

    results: list[PieceResult] = []
    with ProcessPoolExecutor(max_workers=int(args.max_workers)) as pool:
        for result in pool.map(_process_piece, tasks):
            if result.areas.size:
                results.append(result)
            print(f"  piece {result.piece} done ({result.areas.size} cells)", flush=True)

    if not results:
        print("Every piece came back empty.", file=sys.stderr)
        return 1

    centroids = np.concatenate([item.centroids for item in results], axis=0)
    areas = np.concatenate([item.areas for item in results], axis=0)
    map_names = sorted(results[0].maps)
    maps = {
        name: np.concatenate([item.maps[name] for item in results], axis=0)
        for name in map_names
    }
    print(f"Assembled {areas.size} cells over the region of interest.", flush=True)

    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "analysis_date_utc": datetime.now(timezone.utc).isoformat(),
        "geometry": args.geometry,
        "freeze_dir": str(freeze_dir),
        "campaign_dir": str(campaign_dir) if campaign_dir else None,
        "roi": list(roi),
        "num_steps_used": len(steps),
        "time_stride": int(args.time_stride),
        "effective_dt": effective_dt,
        "pieces": [item.piece for item in results],
        "num_cells": int(areas.size),
    }

    # Which time convention reproduces the frozen export?
    conventions: dict[str, Any] = {}
    for pair_name in PAIRS:
        frozen = _load_frozen_reference(
            freeze_dir, args.geometry, pair_name, centroids,
            campaign_dir=campaign_dir,
        )
        if frozen is None:
            continue
        reference, max_distance = frozen
        entry: dict[str, Any] = {"max_centroid_match_distance": max_distance}
        for convention in ("as_stored", "reversed"):
            entry[convention] = correlation_against_reference(
                maps[f"pair__{pair_name}__{convention}"], reference, areas
            )
        entry["selected"] = max(
            ("as_stored", "reversed"),
            key=lambda name: (
                entry[name]["abs_correlation"]
                if math.isfinite(entry[name]["abs_correlation"])
                else -1.0
            ),
        )
        conventions[pair_name] = entry
    report["time_convention"] = conventions

    # Spread of every factor and of the resulting kernels.
    spreads: dict[str, Any] = {}
    for field in FIELDS:
        spreads[f"energy__{field}"] = factor_spread(
            centroids, maps[f"energy__{field}"], areas
        ).as_dict()

    coherence_summary: dict[str, Any] = {}
    for pair_name, (first_field, second_field) in PAIRS.items():
        convention = conventions.get(pair_name, {}).get("selected", "as_stored")
        kernel = maps[f"pair__{pair_name}__{convention}"]
        spreads[f"kernel__{pair_name}"] = factor_spread(
            centroids, np.abs(kernel), areas
        ).as_dict()

        coherence = cauchy_schwarz_split(
            kernel,
            maps[f"energy__{first_field}"],
            maps[f"energy__{second_field}"],
        )
        finite = np.isfinite(coherence)
        weights = areas[finite]
        coherence_summary[pair_name] = {
            "convention": convention,
            "num_defined_cells": int(np.count_nonzero(finite)),
            "area_weighted_mean_coherence": float(
                np.sum(weights * coherence[finite]) / np.sum(weights)
            )
            if np.any(finite)
            else math.nan,
            "median_coherence": float(np.median(coherence[finite]))
            if np.any(finite)
            else math.nan,
        }
        maps[f"coherence__{pair_name}"] = coherence

    report["factor_spread"] = spreads
    report["coherence"] = coherence_summary

    # The decomposition that answers the question, evaluated on windows centred
    # on the defect.  Writing |K| = rho * sqrt(A1 * A2) splits the displacement
    # of a kernel along e2 into an amplitude part, carried by the envelope, and
    # a timing part, carried by the coherence.  The r <= 8R window is the one
    # used by the frozen C66 multipolar analysis.
    defect = (float(args.defect_center_x), float(args.defect_center_y))
    radius = float(args.defect_radius)
    distance = np.hypot(centroids[:, 0] - defect[0], centroids[:, 1] - defect[1])

    windows: dict[str, Any] = {}
    for factor in [float(value) for value in args.local_window_radii.split(",")]:
        inside = distance <= factor * radius
        if int(np.count_nonzero(inside)) < 100:
            continue
        window_centroids = centroids[inside]
        window_areas = areas[inside]

        def offset_over_radius(
            weights: np.ndarray,
            *,
            points: np.ndarray = window_centroids,
            measures: np.ndarray = window_areas,
        ) -> float:
            """Signed x2 displacement of a map's centroid, in defect radii."""

            spread = factor_spread(points, weights, measures)
            return (spread.centroid_x2 - defect[1]) / radius

        entry: dict[str, Any] = {"num_cells": int(np.count_nonzero(inside))}
        for pair_name, (first_field, second_field) in PAIRS.items():
            convention = conventions.get(pair_name, {}).get("selected", "as_stored")
            kernel = maps[f"pair__{pair_name}__{convention}"][inside]
            first_energy = maps[f"energy__{first_field}"][inside]
            second_energy = maps[f"energy__{second_field}"][inside]

            envelope = np.sqrt(first_energy * second_energy)
            coherence = cauchy_schwarz_split(kernel, first_energy, second_energy)
            defined = np.isfinite(coherence)
            envelope_weight = window_areas * envelope

            entry[pair_name] = {
                "envelope_offset_x2_over_R": offset_over_radius(envelope),
                "coherence_offset_x2_over_R": offset_over_radius(
                    coherence[defined],
                    points=window_centroids[defined],
                    measures=window_areas[defined],
                ),
                "kernel_offset_x2_over_R": offset_over_radius(np.abs(kernel)),
                "envelope_weighted_mean_coherence": float(
                    np.sum(envelope_weight[defined] * coherence[defined])
                    / np.sum(envelope_weight[defined])
                )
                if np.any(defined)
                else math.nan,
            }
        windows[f"r_le_{factor:g}R"] = entry
    report["defect_window_decomposition"] = windows

    # The mechanism, stated as a comparison of x2 extents.
    report["mechanism"] = {
        "spread_x2_forward_field_u": spreads[f"energy__{FIELD_U}"]["spread_x2"],
        "spread_x2_adjoint_lambda": spreads[f"energy__{FIELD_LAMBDA}"]["spread_x2"],
        "spread_x2_reflection_W1": spreads[f"energy__{FIELD_W1}"]["spread_x2"],
        "spread_x2_transmission_Lambda2": spreads[f"energy__{FIELD_LAMBDA2}"][
            "spread_x2"
        ],
        "centroid_x2_forward_field_u": spreads[f"energy__{FIELD_U}"]["centroid_x2"],
        "defect_center_x2": float(args.defect_center_y),
    }

    # x2 marginals, for the figure and for direct inspection.
    profiles: dict[str, list[float]] = {}
    centers = None
    for name in list(maps):
        if not (name.startswith("energy__") or name.startswith("pair__")):
            continue
        values = np.abs(maps[name])
        centers, totals = marginal_profile_x2(
            centroids,
            values,
            areas,
            num_bins=int(args.profile_bins),
            x2_min=roi[2],
            x2_max=roi[3],
        )
        peak = float(np.max(totals))
        profiles[name] = [float(value / peak) if peak > 0.0 else 0.0 for value in totals]
    report["x2_profile_bin_centers"] = (
        [float(value) for value in centers] if centers is not None else []
    )
    report["x2_profiles_normalised"] = profiles

    (output_dir / "c66_shear_factor_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "c66_shear_factor_maps.npz",
        centroids=centroids,
        areas=areas,
        **maps,
    )

    print("-" * 72)
    for pair_name, entry in conventions.items():
        selected = entry["selected"]
        print(
            f"{pair_name:22s} convention={selected:10s} "
            f"|corr| = {entry[selected]['abs_correlation']:.6f}"
        )
    print()
    for field in FIELDS:
        spread = spreads[f"energy__{field}"]
        print(
            f"{field:32s} x2 centroid = {spread['centroid_x2']:7.3f}  "
            f"x2 spread = {spread['spread_x2']:6.3f}"
        )

    print("\nCauchy-Schwarz split of the x2 displacement, in defect radii:")
    header = f"{'window':>10s} {'pair':>22s} {'envelope':>10s} {'coherence':>10s} {'kernel':>10s}"
    print(header)
    for window_name, entry in windows.items():
        for pair_name in PAIRS:
            values = entry.get(pair_name)
            if values is None:
                continue
            print(
                f"{window_name:>10s} {pair_name:>22s} "
                f"{values['envelope_offset_x2_over_R']:+10.3f} "
                f"{values['coherence_offset_x2_over_R']:+10.3f} "
                f"{values['kernel_offset_x2_over_R']:+10.3f}"
            )

    print(f"\nReports written to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
