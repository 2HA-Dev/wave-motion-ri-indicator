#!/usr/bin/env python
"""
Quantify the effect of the initial zero-padding on the K^RI density kernel.

THE PADDING ISSUE
-----------------
``compute_material_parameter_correlation_kernels_dg0`` (the production K^RI
kernel, ``elastic_fwi/inversion/material_parameter_gradients.py``) computes the
density term via a Newmark second difference with implicit zero-padding for
the two steps before the stored series begins (``_newmark_second_difference_
vector``, itself documented as a legacy, non-exact acceleration). For the
"true gradient" modes this padding pads a field genuinely at rest at t=0. For
``residual_interaction`` it pads ``W1``/``Lambda2``, which are near their
TERMINAL rest state at the LAST exported index, not the first. A direct nodal
measurement shows that the real first kernel index (VTU export index 1; export index 0 is an
extra initial-state snapshot outside ``equation_adjoint_vectors``, per
``num_snapshots_available=800`` vs ``num_snapshots_exported=801`` in the
campaign manifest) sits at 96.9% of its own peak amplitude, not at rest.
Padding therefore assigns a near-zero past to a field that is near its
maximum. The resulting candidate error is confined to ``kernel_rho`` of the
``residual_interaction`` combination only (the four stiffness kernels use
undifferenced strains, no padding is involved).

WHAT THIS SCRIPT DOES
----------------------
It does NOT re-run the full 800-step accumulation (that already exists on
disk, exported as the "raw_signed" (signed, non-normalised) kernel_rho
field, which is the reference field this script corrects). It instead:

  1. Regenerates the campaign's mesh and function space independently, via
     the SAME production code path the original campaign used
     (``build_direct_observation_spatial_context``), under a single serial
     communicator. Validated against the manifest's own cell/vertex counts
     before any further work.
  2. Reads ONLY the two padding-affected snapshots (VTU export indices 1 and
     2, i.e. kernel steps 0 and 1) of ``reflection_residual_wavefield``
     (W1) and ``transmission_residual_wavefield`` (Lambda2) across all MPI
     pieces, and interpolates them onto the regenerated mesh by nearest
     geometric match (the two meshes are the same geometry; matching by
     coordinate rather than by index is robust to any internal reordering
     between the original 32-rank run and this serial reconstruction).
  3. Computes, via the PRODUCTION helpers imported directly (not
     reimplemented) from ``material_parameter_gradients.py``, the exact same
     UFL density ``inner(a, p)`` integrated to a DG0 cell-average density,
     for steps 0 and 1 only: this is precisely the contribution the
     padding assumption controls.
  4. Subtracts that two-step contribution from the already-exported raw
     signed kernel_rho field (matched cell-by-cell by centroid) to obtain a
     "corrected" field in which steps 0 and 1 make no contribution at all,
     rather than a padded one.
  5. Recomputes the image-quality metrics of the accompanying article (localisation error,
     peak-to-artifact ratio, energy fraction in the target) on both the
     baseline and corrected fields, via the actual production function
     ``compute_metrics`` imported from
     ``extract_gradient_image_quality_metrics.py`` (not reimplemented).

No PDE is solved. The campaign directory is read only.

USAGE (single process; mesh construction is itself serial inside gmsh, so a
COMM_SELF/COMM_WORLD-of-size-1 run reproduces the same geometry as the
original 32-rank campaign without needing to match its partition)
-----
    PYTHONPATH=. python path/to/quantify_ri_rho_boundary_padding_effect.py \
        --campaign-dir path/to/line_campaign_directory \
        --output-dir path/to/output_directory --overwrite
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from dataclasses import fields
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from elastic_fwi.pipelines.direct_observation import (  # noqa: E402
    DirectObservationPipelineConfig,
    build_direct_observation_spatial_context,
)
from elastic_fwi.inversion.material_parameter_gradients import (  # noqa: E402
    _assemble_dg0_cell_integrals,
    _assemble_dg0_cell_measures,
    _cell_average_density_from_integrals,
    _newmark_second_difference_vector,
    _owned_local_size,
)
from elastic_fwi.io.manifests import read_manifest  # noqa: E402
from elastic_fwi.postprocessing.signed_sensitivity_fields import (  # noqa: E402
    load_vtk_cell_field,
)

# extract_gradient_image_quality_metrics.py is a script, not a package member;
# compute_metrics is loaded directly by path rather than duplicating its formulas.
_METRICS_SCRIPT = _REPO_ROOT / "scripts" / "revision_wave_motion" / "extract_gradient_image_quality_metrics.py"
_spec = importlib.util.spec_from_file_location("_gradient_image_quality_metrics", _METRICS_SCRIPT)
_metrics_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics_module)
compute_metrics = _metrics_module.compute_metrics

_CONFIG_FIELDS = {item.name for item in fields(DirectObservationPipelineConfig)}

DEFECT_CENTER = (36.5, 36.5)
DEFECT_RADIUS = 0.25
FALSE_MAX_THRESHOLD_FRACTION = 0.5
FALSE_MAX_MIN_DISTANCE_FACTOR = 1.0


# --------------------------------------------------------------------------
# Minimal ASCII-VTU point-cloud reader (meshio does not support the VTU 2.2
# header this solver emits). Only Points and PointData are read, not Cells:
# cell topology is taken from the freshly regenerated mesh, so VTU
# connectivity never needs to be parsed.
# --------------------------------------------------------------------------
def _read_vtu_points_and_vector(path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text()
    points_match = re.search(r"<Points>.*?<DataArray[^>]*>(.*?)</DataArray>.*?</Points>", text, re.S)
    if points_match is None:
        raise RuntimeError(f"No <Points> block in {path}")
    points = np.fromstring(points_match.group(1), sep=" ").reshape(-1, 3)[:, :2]

    data_match = re.search(r"<PointData[^>]*>(.*?)</PointData>", text, re.S)
    if data_match is None:
        raise RuntimeError(f"No <PointData> block in {path}")
    # PointData also carries vtkOriginalPointIds/vtkGhostType blocks (no
    # NumberOfComponents attribute); this matches the first array that has
    # one, i.e. the actual displacement field.
    array_match = re.search(
        r'<DataArray[^>]*NumberOfComponents="(\d+)"[^>]*>(.*?)</DataArray>', data_match.group(1), re.S
    )
    if array_match is None:
        raise RuntimeError(f"No vector DataArray in <PointData> of {path}")
    nc = int(array_match.group(1))
    values = np.fromstring(array_match.group(2), sep=" ").reshape(-1, nc)
    # VTK pads 2-D vectors to 3 components (zero z); V has block size 2.
    return points, values[:, :2]


def _read_all_pieces(directory: Path, field: str, export_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate Points/PointData for one snapshot across every MPI piece."""
    pieces = sorted(directory.glob(f"{field}_p*_{export_index:06d}.vtu"))
    if not pieces:
        raise SystemExit(f"No pieces found for {field} at export index {export_index} in {directory}")
    all_points, all_values = [], []
    for piece in pieces:
        pts, vals = _read_vtu_points_and_vector(piece)
        all_points.append(pts)
        all_values.append(vals)
    return np.concatenate(all_points, axis=0), np.concatenate(all_values, axis=0)


def _build_config(raw: dict, *, num_steps: int) -> DirectObservationPipelineConfig:
    kwargs = {name: raw[name] for name in _CONFIG_FIELDS if name in raw}
    kwargs["store_displacement_snapshots"] = True
    kwargs["num_steps"] = int(num_steps)
    return DirectObservationPipelineConfig(**kwargs)


def _petsc_vec_from_function(fn: fem.Function) -> PETSc.Vec:
    owned = fn.function_space.dofmap.index_map.size_local * fn.function_space.dofmap.index_map_bs
    vec = PETSc.Vec().createSeq(owned, comm=MPI.COMM_SELF)
    vec.setArray(np.asarray(fn.x.array[:owned], dtype=np.float64))
    vec.assemble()
    return vec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = args.output_dir
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = args.campaign_dir / "gradient_combination_campaign_manifest.json"
    manifest = read_manifest(manifest_path)
    raw_config = manifest["config"]
    dt = float(raw_config["dt"])
    num_steps = int(raw_config["num_steps"])
    expected_cells = int(manifest["mesh_discretization"]["global_owned_cells"])
    expected_vertices = int(manifest["mesh_discretization"]["global_owned_vertices"])

    print(f"[STEP 1] Regenerating mesh from manifest config (dt={dt}, num_steps={num_steps}) ...", flush=True)
    config = _build_config(raw_config, num_steps=num_steps)
    ctx = build_direct_observation_spatial_context(config, comm=MPI.COMM_SELF)
    msh, V = ctx.mesh, ctx.function_space

    n_cells = msh.topology.index_map(msh.topology.dim).size_local
    n_vertices = msh.topology.index_map(0).size_local
    print(f"[STEP 1] Regenerated: {n_cells} cells, {n_vertices} vertices "
          f"(expected {expected_cells} / {expected_vertices}).", flush=True)
    if n_cells != expected_cells or n_vertices != expected_vertices:
        raise SystemExit(
            "MESH VALIDATION FAILED -- regenerated mesh does not match the campaign's "
            "recorded discretisation. Aborting before any further computation."
        )
    print("[STEP 1] Mesh validated against manifest.", flush=True)

    Q = fem.functionspace(msh, ("DG", 0))
    cell_measures = _assemble_dg0_cell_measures(Q)
    my_centroids = Q.tabulate_dof_coordinates()[: _owned_local_size(Q), :2]

    print("[STEP 2] Loading baseline exported kernel_rho (raw signed, residual_interaction) ...", flush=True)
    baseline_path = (
        args.campaign_dir
        / "inverse"
        / "gradient_combinations_raw_signed"
        / "residual_interaction"
        / "vtk"
        / "residual_interaction_rho.pvd"
    )
    baseline = load_vtk_cell_field(baseline_path, "residual_interaction_rho")
    print(f"[STEP 2] Baseline: {baseline.values.size} cells.", flush=True)

    print("[STEP 3] Matching regenerated DG0 cells to baseline cells by centroid ...", flush=True)
    baseline_tree = cKDTree(baseline.centers)
    dist, match_idx = baseline_tree.query(my_centroids)
    max_dist = float(np.max(dist))
    print(f"[STEP 3] Max centroid-match distance: {max_dist:.3e}.", flush=True)
    if max_dist > 1.0e-6:
        raise SystemExit(
            f"CELL MATCH VALIDATION FAILED -- max distance {max_dist:.3e} exceeds tolerance. "
            "The regenerated mesh's cells do not correspond 1:1 to the baseline's. Aborting."
        )
    print("[STEP 3] Cell correspondence validated (near-zero geometric distance).", flush=True)

    # -----------------------------------------------------------------
    # STEP 4: read the two padding-affected snapshots. VTU export index 0
    # is the extra initial-state export (num_snapshots_exported=801)
    # outside equation_adjoint_vectors (num_snapshots_available=800), so
    # kernel step n corresponds to VTU export index n+1. Measured amplitudes:
    # VTU index 0 at 1.7% of peak, VTU index 1 at 96.9%, VTU index 800 at
    # 7.8e-15, consistent with terminal rest at the LAST exported index.
    # -----------------------------------------------------------------
    vtk_dir = args.campaign_dir / "inverse" / "vtk"
    print("[STEP 4] Reading W1 (reflection) and Lambda2 (transmission), export indices 1-2 ...", flush=True)
    w1_points0, w1_vals0 = _read_all_pieces(vtk_dir, "reflection_residual_wavefield", 1)
    w1_points1, w1_vals1 = _read_all_pieces(vtk_dir, "reflection_residual_wavefield", 2)
    l2_points0, l2_vals0 = _read_all_pieces(vtk_dir, "transmission_residual_wavefield", 1)
    l2_points1, l2_vals1 = _read_all_pieces(vtk_dir, "transmission_residual_wavefield", 2)
    print(f"[STEP 4] Read {w1_points0.shape[0]} W1 points and {l2_points0.shape[0]} Lambda2 points per step.",
          flush=True)

    print("[STEP 5] Validating nodal geometric correspondence ...", flush=True)
    n_owned_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    dof_coords = V.tabulate_dof_coordinates()[:n_owned_dofs, :2]
    # tabulate_dof_coordinates() may repeat a node's coordinate once per
    # blocked component; querying the (possibly redundant) full array is
    # harmless for a max-distance validation and avoids any assumption about
    # dolfinx's internal block layout.
    point_tree = cKDTree(w1_points0)
    node_dist, _ = point_tree.query(dof_coords)
    max_node_dist = float(np.max(node_dist))
    print(f"[STEP 5] Max W1 node-match distance: {max_node_dist:.3e}.", flush=True)
    l2_tree0 = cKDTree(l2_points0)
    l2_node_dist, _ = l2_tree0.query(dof_coords)
    l2_tree1 = cKDTree(l2_points1)
    l2_node_dist1, _ = l2_tree1.query(dof_coords)
    max_l2_dist = float(max(np.max(l2_node_dist), np.max(l2_node_dist1)))
    print(f"[STEP 5] Max Lambda2 node-match distance: {max_l2_dist:.3e}.", flush=True)
    if max(max_node_dist, max_l2_dist) > 1.0e-6:
        raise SystemExit("NODE MATCH VALIDATION FAILED. Aborting.")
    print("[STEP 5] Node correspondence validated.", flush=True)

    # Exact for P1 Lagrange: dolfinx evaluates the callable at the nodal
    # coordinates and assigns the nodal values directly (no projection),
    # and .interpolate() handles the blocked vector-DOF layout internally,
    # which avoids the block-indexing errors of a hand-rolled assignment.
    def make_interpolator(points_2d: np.ndarray, values: np.ndarray):
        tree = cKDTree(points_2d)

        def f(x: np.ndarray) -> np.ndarray:
            idx = tree.query(x[:2, :].T)[1]
            return np.ascontiguousarray(values[idx].T)

        return f

    print("[STEP 5] Interpolating snapshots onto the regenerated mesh ...", flush=True)
    u0 = fem.Function(V, name="W1_step0")
    u0.interpolate(make_interpolator(w1_points0, w1_vals0))
    u1 = fem.Function(V, name="W1_step1")
    u1.interpolate(make_interpolator(w1_points1, w1_vals1))
    p0 = fem.Function(V, name="Lambda2_step0")
    p0.interpolate(make_interpolator(l2_points0, l2_vals0))
    p1 = fem.Function(V, name="Lambda2_step1")
    p1.interpolate(make_interpolator(l2_points1, l2_vals1))

    print("[STEP 6] Computing the two padding-affected accumulation terms via the "
          "production Newmark-difference helper ...", flush=True)
    u0_vec = _petsc_vec_from_function(u0)
    u1_vec = _petsc_vec_from_function(u1)
    try:
        a0_vec = _newmark_second_difference_vector((u0_vec, u1_vec), 0, dt)
        a1_vec = _newmark_second_difference_vector((u0_vec, u1_vec), 1, dt)
        try:
            a0_fn = fem.Function(V, name="a0")
            a0_fn.x.array[:n_owned_dofs] = np.asarray(a0_vec.getArray(readonly=True))
            a1_fn = fem.Function(V, name="a1")
            a1_fn.x.array[:n_owned_dofs] = np.asarray(a1_vec.getArray(readonly=True))

            integrals0 = _assemble_dg0_cell_integrals(Q, ufl.inner(a0_fn, p0))
            integrals1 = _assemble_dg0_cell_integrals(Q, ufl.inner(a1_fn, p1))
            density0 = _cell_average_density_from_integrals(integrals0, cell_measures)
            density1 = _cell_average_density_from_integrals(integrals1, cell_measures)
        finally:
            a0_vec.destroy()
            a1_vec.destroy()
    finally:
        u0_vec.destroy()
        u1_vec.destroy()

    padded_contribution = float(dt) * (density0 + density1)
    print(f"[STEP 6] Padded contribution: L2-ish max|.| = {np.max(np.abs(padded_contribution)):.6e}, "
          f"sum = {np.sum(padded_contribution):.6e}.", flush=True)

    # -----------------------------------------------------------------
    # STEP 7: corrected field = baseline minus what the current padding
    # rule contributes at steps 0-1. This variant drops the ill-defined
    # boundary terms; it does not re-estimate what those two steps would
    # contribute without padding.
    # -----------------------------------------------------------------
    baseline_matched = baseline.values[match_idx]
    corrected = baseline_matched - padded_contribution

    baseline_metrics = compute_metrics(
        centers=my_centroids,
        values=baseline_matched,
        cell_measures=cell_measures,
        defect_center=DEFECT_CENTER,
        defect_radius=DEFECT_RADIUS,
        false_max_threshold_fraction=FALSE_MAX_THRESHOLD_FRACTION,
        false_max_min_distance_factor=FALSE_MAX_MIN_DISTANCE_FACTOR,
    )
    corrected_metrics = compute_metrics(
        centers=my_centroids,
        values=corrected,
        cell_measures=cell_measures,
        defect_center=DEFECT_CENTER,
        defect_radius=DEFECT_RADIUS,
        false_max_threshold_fraction=FALSE_MAX_THRESHOLD_FRACTION,
        false_max_min_distance_factor=FALSE_MAX_MIN_DISTANCE_FACTOR,
    )

    print("\n" + "-" * 78)
    print(f"{'metric':<28}{'baseline (published)':>24}{'corrected':>18}{'rel. change':>14}")
    print("-" * 78)
    for key in ("localization_error", "peak_to_artifact_ratio", "energy_fraction_in_defect",
                "global_peak_value", "contrast_defect_to_background"):
        b, c = baseline_metrics[key], corrected_metrics[key]
        rel = (c - b) / b if b != 0 else float("nan")
        print(f"{key:<28}{b:>24.6e}{c:>18.6e}{rel:>14.4%}")
    print("-" * 78)

    payload = {
        "analysis_date_utc": datetime.now(timezone.utc).isoformat(),
        "format_version": "ri_rho_padding_effect_v1",
        "purpose": "Quantify the effect of the boundary padding of the residual_interaction "
                   "kernel_rho Newmark second difference at steps 0-1.",
        "campaign_dir": str(args.campaign_dir),
        "dt": dt,
        "num_cells": int(baseline.values.size),
        "max_centroid_match_distance": max_dist,
        "max_node_match_distance": max(max_node_dist, float(np.max(l2_node_dist)), float(np.max(l2_node_dist1))),
        "padded_contribution_max_abs": float(np.max(np.abs(padded_contribution))),
        "padded_contribution_sum": float(np.sum(padded_contribution)),
        "baseline_metrics": baseline_metrics,
        "corrected_metrics": corrected_metrics,
    }
    (out / "c1_padding_effect.json").write_text(json.dumps(payload, indent=1))
    print(f"\nReport written to {(out / 'c1_padding_effect.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
