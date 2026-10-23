#!/usr/bin/env python3
"""Measure the eps11 / eps22 profiles of u_ref across the mirror axis.

This is the numerical counterpart of
``derive_normal_strain_curvature_asymmetry.py``, which derives, from the
elastodynamic equation alone,

    kappa_11 / kappa_22 = -(C12 + C66)/C11 * (eps22|axis / eps11|axis)

where kappa_ii is the relative curvature in s = x1 - 36.5 of the normal strain
eps_ii of the reference field, evaluated on the mirror axis.  For the healthy
medium (C12 = 6.9, C66 = 4.67, C11 = 16.24) the stiffness prefactor is 0.7124.

Two predictions are tested here, both on u_ref alone.  u_ref lives in the
healthy medium and does not depend on the defect at all, so this measurement is
independent of where the defect sits -- and in particular it is *not* a
restatement of the off-axis campaign's result.

    P1.  eps11 and eps22 are both stationary at s = 0 (mirror parity, already
         proved; measured here as a control on the discretisation).
    P2.  kappa_11 and kappa_22 have OPPOSITE signs, with the ratio above.

P2 is falsifiable: it follows from the derivation alone, independently of the measurement.

Strains are computed exactly per P1 triangle (the gradient of a P1 field is
cell-wise constant), squared, and integrated in time by trapezoid over the
exported snapshots.  The profile is then formed over cells lying in a band
around the defect's x2, binned in s and averaged with cell areas as weights.
Ghost cells are excluded per MPI piece.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import meshio
import numpy as np

from elastic_fwi.metrics.artifact_metrics import _meshio_read_serial_vtu_compat

AXIS = 36.5
STIFFNESS_PREFACTOR = (6.9 + 4.67) / 16.24  # (C12 + C66) / C11, healthy medium


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _pvd_entries(pvd_path: Path) -> list[tuple[float, Path]]:
    root = ET.parse(pvd_path).getroot()
    entries = [
        (float(node.attrib["timestep"]), pvd_path.parent / node.attrib["file"])
        for node in root.iter()
        if _local_name(node.tag) == "DataSet"
    ]
    entries.sort(key=lambda item: item[0])
    return entries


def _piece_paths(pvtu_path: Path) -> list[Path]:
    root = ET.parse(pvtu_path).getroot()
    return [
        pvtu_path.parent / node.attrib["Source"]
        for node in root.iter()
        if _local_name(node.tag) == "Piece"
    ]


def _triangles_and_owned(mesh) -> tuple[np.ndarray, np.ndarray]:
    for i, block in enumerate(mesh.cells):
        data = np.asarray(block.data, dtype=np.int64)
        if str(block.type) in ("triangle", "VTK_TRIANGLE", "VTK_LAGRANGE_TRIANGLE") and data.shape[1] >= 3:
            ghost = mesh.cell_data.get("vtkGhostType")
            owned = (
                np.asarray(ghost[i]).reshape(-1) == 0
                if ghost is not None
                else np.ones(data.shape[0], dtype=bool)
            )
            return data[:, :3], owned
    raise RuntimeError("No triangle cell block found.")


def _cell_gradients(points: np.ndarray, tri: np.ndarray, nodal: np.ndarray) -> np.ndarray:
    """Constant per-cell gradient of a P1 scalar field. Returns (n_cells, 2)."""
    p0 = points[tri[:, 0], :2]
    e1 = points[tri[:, 1], :2] - p0
    e2 = points[tri[:, 2], :2] - p0
    df1 = nodal[tri[:, 1]] - nodal[tri[:, 0]]
    df2 = nodal[tri[:, 2]] - nodal[tri[:, 0]]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    grad_x = (df1 * e2[:, 1] - df2 * e1[:, 1]) / det
    grad_y = (-df1 * e2[:, 0] + df2 * e1[:, 0]) / det
    return np.column_stack((grad_x, grad_y))


def _cell_areas(points: np.ndarray, tri: np.ndarray) -> np.ndarray:
    p0 = points[tri[:, 0], :2]
    e1 = points[tri[:, 1], :2] - p0
    e2 = points[tri[:, 2], :2] - p0
    return 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])


def accumulate(vtk_dir: Path, field_name: str, pvd_name: str, *, stride: int,
               band_half_width: float, band_center_x2: float, s_max: float):
    """Time-integrated eps11^2 and eps22^2 per cell, restricted to the band."""
    entries = _pvd_entries(vtk_dir / pvd_name)[::stride]
    times = np.array([t for t, _ in entries])

    centroids_s = None
    areas = None
    acc11: list[np.ndarray] = []
    acc22: list[np.ndarray] = []

    for index, (time_value, pvtu) in enumerate(entries):
        e11_parts, e22_parts, s_parts, area_parts = [], [], [], []
        for piece in _piece_paths(pvtu):
            mesh = _meshio_read_serial_vtu_compat(meshio, piece)
            values = np.asarray(mesh.point_data[field_name], dtype=np.float64)
            points = np.asarray(mesh.points, dtype=np.float64)
            tri, owned = _triangles_and_owned(mesh)
            tri = tri[owned]
            if tri.size == 0:
                continue
            centroid = points[tri, :2].mean(axis=1)
            in_band = np.abs(centroid[:, 1] - band_center_x2) <= band_half_width
            in_band &= np.abs(centroid[:, 0] - AXIS) <= s_max
            if not in_band.any():
                continue
            tri = tri[in_band]
            centroid = centroid[in_band]
            eps11 = _cell_gradients(points, tri, values[:, 0])[:, 0]
            eps22 = _cell_gradients(points, tri, values[:, 1])[:, 1]
            e11_parts.append(eps11)
            e22_parts.append(eps22)
            s_parts.append(centroid[:, 0] - AXIS)
            area_parts.append(_cell_areas(points, tri))

        e11 = np.concatenate(e11_parts) if e11_parts else np.zeros(0)
        e22 = np.concatenate(e22_parts) if e22_parts else np.zeros(0)
        if centroids_s is None:
            centroids_s = np.concatenate(s_parts)
            areas = np.concatenate(area_parts)
        acc11.append(e11**2)
        acc22.append(e22**2)
        print(f"    t={time_value:6.2f}  cells_in_band={e11.size}", flush=True)

    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    energy11 = trapezoid(np.vstack(acc11), times, axis=0)
    energy22 = trapezoid(np.vstack(acc22), times, axis=0)
    return centroids_s, areas, energy11, energy22


def bin_profile(s_values, areas, energy, num_bins, s_max):
    edges = np.linspace(-s_max, s_max, num_bins + 1)
    index = np.digitize(s_values, edges) - 1
    centers, means = [], []
    for k in range(num_bins):
        mask = index == k
        if mask.sum() < 5:
            continue
        centers.append(0.5 * (edges[k] + edges[k + 1]))
        means.append(float(np.average(energy[mask], weights=areas[mask])))
    return np.array(centers), np.array(means)


def fit_relative_curvature(centers, means, fit_half_width):
    """Fit a + b s^2 near the axis; the relative curvature of the strain is b/a."""
    mask = np.abs(centers) <= fit_half_width
    design = np.column_stack((np.ones(mask.sum()), centers[mask] ** 2))
    coeffs, *_ = np.linalg.lstsq(design, means[mask], rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    residual = means[mask] - design @ coeffs
    r_squared = 1.0 - float((residual**2).sum()) / float(((means[mask] - means[mask].mean()) ** 2).sum())
    # Odd part, which parity forces to vanish: a control on the discretisation.
    linear_design = np.column_stack((np.ones(mask.sum()), centers[mask], centers[mask] ** 2))
    linear_coeffs, *_ = np.linalg.lstsq(linear_design, means[mask], rcond=None)
    return {
        "value_on_axis": a,
        "quadratic_coefficient": b,
        "relative_curvature": b / a,
        "r_squared": r_squared,
        "odd_coefficient": float(linear_coeffs[1]),
        "odd_over_even_ratio": abs(float(linear_coeffs[1])) / abs(a),
        "num_bins_used": int(mask.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vtk-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--band-half-width", type=float, default=0.5,
                        help="half-width in x2 of the band centred on the defect's x2")
    parser.add_argument("--band-center-x2", type=float, default=36.5)
    parser.add_argument("--s-max", type=float, default=2.0)
    parser.add_argument("--num-bins", type=int, default=40)
    parser.add_argument("--fit-half-width", type=float, default=1.0)
    args = parser.parse_args()

    print(f"[u_ref] direct_simulated_displacement.pvd, band |x2 - {args.band_center_x2}| <= "
          f"{args.band_half_width}, |s| <= {args.s_max}", flush=True)
    s_values, areas, energy11, energy22 = accumulate(
        args.vtk_dir, "direct_simulated_displacement", "direct_simulated_displacement.pvd",
        stride=args.stride, band_half_width=args.band_half_width,
        band_center_x2=args.band_center_x2, s_max=args.s_max,
    )

    centers11, means11 = bin_profile(s_values, areas, energy11, args.num_bins, args.s_max)
    centers22, means22 = bin_profile(s_values, areas, energy22, args.num_bins, args.s_max)
    fit11 = fit_relative_curvature(centers11, means11, args.fit_half_width)
    fit22 = fit_relative_curvature(centers22, means22, args.fit_half_width)

    # eps_ii|axis is the RMS strain, i.e. the square root of the fitted value of
    # the time-integrated squared strain on the axis.
    strain_ratio = float(np.sqrt(fit22["value_on_axis"] / fit11["value_on_axis"]))
    measured_kappa_ratio = fit11["relative_curvature"] / fit22["relative_curvature"]
    predicted_kappa_ratio = -STIFFNESS_PREFACTOR * strain_ratio

    result = {
        "kind": "normal_strain_axis_profile",
        "vtk_dir": str(args.vtk_dir),
        "axis": AXIS,
        "band_center_x2": args.band_center_x2,
        "band_half_width": args.band_half_width,
        "s_max": args.s_max,
        "fit_half_width": args.fit_half_width,
        "stiffness_prefactor_C12_plus_C66_over_C11": STIFFNESS_PREFACTOR,
        "eps11_fit": fit11,
        "eps22_fit": fit22,
        "eps22_over_eps11_on_axis": strain_ratio,
        "measured_kappa_ratio": measured_kappa_ratio,
        "predicted_kappa_ratio": predicted_kappa_ratio,
        "signs_are_opposite": bool(fit11["relative_curvature"] * fit22["relative_curvature"] < 0),
        "profile_eps11": {"s": centers11.tolist(), "time_integrated_square": means11.tolist()},
        "profile_eps22": {"s": centers22.tolist(), "time_integrated_square": means22.tolist()},
    }

    print("\n--- eps11 ---")
    for key, value in fit11.items():
        print(f"  {key:26s} {value}")
    print("--- eps22 ---")
    for key, value in fit22.items():
        print(f"  {key:26s} {value}")
    print(f"\n  eps22/eps11 on axis         {strain_ratio:.4f}")
    print(f"  kappa_11 / kappa_22 measured {measured_kappa_ratio:.4f}")
    print(f"  kappa_11 / kappa_22 predicted {predicted_kappa_ratio:.4f}")
    print(f"  opposite signs (P2)          {result['signs_are_opposite']}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
