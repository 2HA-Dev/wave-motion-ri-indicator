#!/usr/bin/env python3
"""Domain-integrated x1/x2 displacement-component energy of a full wavefield.

Why.  The RMS(x)/RMS(y) ratio at the two receiver arrays (see
``check_c11_off_axis_weak_component_hypothesis.py``) cannot decide whether
the weakness of a field's x1-component explains why C11 collapses off axis:
the direct reference field u_ref has an x/y ratio of 3.5e-5 at Gamma1 (y=30,
close to the polarised source) but 0.26 at Gamma2 (y=43, after 13 length
units of propagation), a factor 7400 apart for the *same* field.  A
two-point sample cannot characterise a field whose
component balance itself varies that much in space; a domain integral is
needed.

This script reads the full nodal displacement snapshots already exported by
the campaign (``direct_simulated_displacement.pvd``,
``reflection_residual_wavefield.pvd``, ``transmission_residual_wavefield.pvd``)
and accumulates, over a chosen set of snapshot times and the *whole* mesh,
the P1-exact spatial integral of u_x^2 and u_y^2, trapezoidally summed over
time.  That is exactly the quantity the material-gradient kernel integrates
against (up to the strain operator), so it is a far more defensible proxy for
"how lopsided is this field" than any point sample.  No PDE solve is
performed; every array read here was produced by campaigns already run.

Ghost cells (vtkGhostType != 0) are excluded per MPI piece so that shared
partition boundaries are not double-counted; shared points are handled
correctly because only *owned* triangles contribute to the P1 quadrature.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import meshio
import numpy as np

from elastic_fwi.metrics.artifact_metrics import _meshio_read_serial_vtu_compat


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _pvd_entries(pvd_path: Path) -> list[tuple[float, Path]]:
    root = ET.parse(pvd_path).getroot()
    entries: list[tuple[float, Path]] = []
    for node in root.iter():
        if _local_name(node.tag) != "DataSet":
            continue
        entries.append((float(node.attrib["timestep"]), pvd_path.parent / node.attrib["file"]))
    entries.sort(key=lambda item: item[0])
    return entries


def _piece_paths(pvtu_path: Path) -> list[Path]:
    root = ET.parse(pvtu_path).getroot()
    pieces = []
    for node in root.iter():
        if _local_name(node.tag) == "Piece":
            pieces.append(pvtu_path.parent / node.attrib["Source"])
    return pieces


def _triangle_area(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    xy = points[triangles, :2]
    cross = (
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 1, 1] - xy[:, 0, 1]) * (xy[:, 2, 0] - xy[:, 0, 0])
    )
    return 0.5 * np.abs(cross)


def _p1_triangle_square_integral(values: np.ndarray, area: np.ndarray) -> np.ndarray:
    squares = np.sum(values * values, axis=1)
    pairs = values[:, 0] * values[:, 1] + values[:, 0] * values[:, 2] + values[:, 1] * values[:, 2]
    return area * (squares + pairs) / 6.0


def _owned_triangles(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return (triangle connectivity, owned mask) for the first triangle block."""
    block_index = None
    triangles = None
    for i, block in enumerate(mesh.cells):
        data = np.asarray(block.data, dtype=np.int64)
        if str(block.type) in ("triangle", "VTK_TRIANGLE", "VTK_LAGRANGE_TRIANGLE") and data.shape[1] >= 3:
            block_index = i
            triangles = data[:, :3]
            break
    if triangles is None:
        raise RuntimeError("No triangle cell block found.")

    ghost_blocks = mesh.cell_data.get("vtkGhostType")
    if ghost_blocks is not None and block_index is not None:
        owned = np.asarray(ghost_blocks[block_index]).reshape(-1) == 0
    else:
        owned = np.ones(triangles.shape[0], dtype=bool)
    return triangles, owned


def snapshot_energy(pvtu_path: Path, field_name: str) -> tuple[float, float]:
    """Spatial integral of u_x^2 and u_y^2 over the whole (multi-piece) mesh."""
    energy_x = 0.0
    energy_y = 0.0
    for piece in _piece_paths(pvtu_path):
        mesh = _meshio_read_serial_vtu_compat(meshio, piece)
        if field_name not in mesh.point_data:
            raise RuntimeError(f"Field {field_name!r} not found in {piece}. Available: {list(mesh.point_data)}")
        values = np.asarray(mesh.point_data[field_name], dtype=np.float64)
        points = np.asarray(mesh.points, dtype=np.float64)
        triangles, owned = _owned_triangles(mesh)
        if not owned.any():
            continue
        tri = triangles[owned]
        area = _triangle_area(points, tri)
        vx = values[tri, 0]
        vy = values[tri, 1]
        energy_x += float(_p1_triangle_square_integral(vx, area).sum())
        energy_y += float(_p1_triangle_square_integral(vy, area).sum())
    return energy_x, energy_y


def series_energy_balance(pvd_path: Path, field_name: str, *, stride: int) -> dict[str, float]:
    entries = _pvd_entries(pvd_path)[::stride]
    if len(entries) < 2:
        raise RuntimeError(f"Not enough snapshots after striding: {pvd_path}")
    times = np.array([t for t, _ in entries])
    ex = np.zeros(len(entries))
    ey = np.zeros(len(entries))
    for i, (_, path) in enumerate(entries):
        ex[i], ey[i] = snapshot_energy(path, field_name)
        print(f"    t={times[i]:6.2f}  E_x={ex[i]:.4e}  E_y={ey[i]:.4e}", flush=True)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    time_integrated_ex = float(trapezoid(ex, times))
    time_integrated_ey = float(trapezoid(ey, times))
    return {
        "num_snapshots": len(entries),
        "time_integrated_energy_x": time_integrated_ex,
        "time_integrated_energy_y": time_integrated_ey,
        "energy_x_over_y_ratio": time_integrated_ex / time_integrated_ey,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vtk-dir", type=Path, required=True, help="campaign's inverse/vtk directory")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1, help="use every Nth snapshot in each PVD series")
    args = parser.parse_args()

    fields = {
        "u_ref": ("direct_simulated_displacement.pvd", "direct_simulated_displacement"),
        "W_gamma1_reflection_residual": ("reflection_residual_wavefield.pvd", "reflection_residual_wavefield"),
        "Lambda_gamma2_transmission_residual": (
            "transmission_residual_wavefield.pvd",
            "transmission_residual_wavefield",
        ),
    }
    result: dict[str, object] = {"vtk_dir": str(args.vtk_dir), "stride": args.stride, "fields": {}}
    for label, (pvd_name, field_name) in fields.items():
        print(f"[{label}] {pvd_name}", flush=True)
        result["fields"][label] = series_energy_balance(args.vtk_dir / pvd_name, field_name, stride=args.stride)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
