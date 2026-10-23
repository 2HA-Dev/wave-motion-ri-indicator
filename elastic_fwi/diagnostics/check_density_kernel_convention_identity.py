from __future__ import annotations

"""
Validation check of the density-kernel conventions.

Checks, on stored residual-wave snapshots of a completed campaign, the identity
relating the two density-kernel conventions:

    K^acc_rho(l1, l2) = B_rho(l1, l2) - d_t l1(0) . l2(0)

with K^acc_rho(a, b) = sum d_tt a . b dt (acceleration form, implemented in
material_parameter_gradients.py) and B_rho(a, b) = - sum d_t a . d_t b dt
(velocity form). Two time discretisations of B_rho are compared: a staggered
one, the exact discrete primitive of the production second-difference
operator, for which the identity must hold to machine precision; and a
naive centred one, for which it must not. This contrast is what makes the
check probative rather than tautological.

Time convention, read from the exported PVD timesteps: both series carry terminal adjoint conditions l(T) = d_t l(T) = 0
at their last stored index, so index n corresponds to adjoint time t = n * dt
and the boundary term of the identity sits at index 0.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

REFLECTION_FIELD = "reflection_residual_wavefield"
TRANSMISSION_FIELD = "transmission_residual_wavefield"


def _read_partition(path: Path, field: str) -> tuple[np.ndarray, np.ndarray]:
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()

    point_data = grid.GetPointData()
    values = point_data.GetArray(field)
    if values is None:
        raise KeyError(f"Field {field!r} missing from {path}")

    ghost = point_data.GetArray("vtkGhostType")
    ghost_mask = (
        vtk_to_numpy(ghost).astype(bool)
        if ghost is not None
        else np.zeros(grid.GetNumberOfPoints(), dtype=bool)
    )

    return np.asarray(vtk_to_numpy(values), dtype=np.float64), ~ghost_mask


def _identity_from_series(
    a_series: np.ndarray,
    b_series: np.ndarray,
    *,
    dt: float,
    owned: np.ndarray,
) -> dict[str, Any]:
    """
    Core algebra, decoupled from file I/O so it can be unit-tested on small
    synthetic series. a_series and b_series have shape (n_snapshots, n_points,
    n_components), ordered by increasing adjoint time (index 0 = t = 0).
    """
    n_files = a_series.shape[0]
    n_pts = a_series.shape[1]

    k_acc = np.zeros(n_pts, dtype=np.float64)
    b_staggered = np.zeros(n_pts, dtype=np.float64)
    b_centred = np.zeros(n_pts, dtype=np.float64)

    a_m1 = b_m1 = None
    a_0, b_0 = a_series[0], b_series[0]
    first_a, first_b = a_0.copy(), b_0.copy()

    for index in range(1, n_files):
        a_p1, b_p1 = a_series[index], b_series[index]

        da_half = (a_p1 - a_0) / dt
        db_half = (b_p1 - b_0) / dt
        b_staggered -= np.einsum("ij,ij->i", da_half, db_half) * dt

        if a_m1 is not None:
            acc = (a_p1 - 2.0 * a_0 + a_m1) / (dt * dt)
            k_acc += np.einsum("ij,ij->i", acc, b_0) * dt

            da_c = (a_p1 - a_m1) / (2.0 * dt)
            db_c = (b_p1 - b_m1) / (2.0 * dt)
            b_centred -= np.einsum("ij,ij->i", da_c, db_c) * dt

        a_m1, b_m1 = a_0, b_0
        a_0, b_0 = a_p1, b_p1

    da_first_half = (a_series[1] - first_a) / dt
    boundary = -np.einsum("ij,ij->i", da_first_half, first_b)

    def norm(x: np.ndarray) -> float:
        return float(np.sqrt(np.sum(x[owned] ** 2)))

    res_staggered = k_acc - b_staggered - boundary
    res_centred = k_acc - b_centred - boundary
    k_acc_norm = norm(k_acc)

    return {
        "terminal_condition_check": {
            "norm_lambda1_at_last_index": float(np.linalg.norm(a_0[owned])),
            "norm_lambda2_at_last_index": float(np.linalg.norm(b_0[owned])),
        },
        "values": {
            "K_acc_norm": k_acc_norm,
            "B_staggered_norm": norm(b_staggered),
            "B_centred_norm": norm(b_centred),
            "boundary_term_norm": norm(boundary),
            "staggered_residual_relative": norm(res_staggered) / k_acc_norm,
            "centred_residual_relative": norm(res_centred) / k_acc_norm,
            "convention_gap_boundary_over_K_acc": norm(boundary) / k_acc_norm,
        },
    }


def run_density_kernel_convention_identity_check(
    *,
    vtk_dir: Path,
    rank: int = 0,
    dt: float,
    tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    files_a = sorted(vtk_dir.glob(f"{REFLECTION_FIELD}_p{rank}_*.vtu"))
    files_b = sorted(vtk_dir.glob(f"{TRANSMISSION_FIELD}_p{rank}_*.vtu"))

    if len(files_a) != len(files_b) or not files_a:
        raise RuntimeError(
            f"Inconsistent or empty snapshot series for rank {rank}: "
            f"{len(files_a)} vs {len(files_b)} files."
        )

    n_files = len(files_a)
    owned = None
    a_series = []
    b_series = []

    for index in range(n_files):
        a, owned_a = _read_partition(files_a[index], REFLECTION_FIELD)
        b, _ = _read_partition(files_b[index], TRANSMISSION_FIELD)
        if owned is None:
            owned = owned_a
        a_series.append(a)
        b_series.append(b)

    identity = _identity_from_series(
        np.stack(a_series), np.stack(b_series), dt=dt, owned=owned
    )

    staggered_relative = identity["values"]["staggered_residual_relative"]
    status = "ok" if staggered_relative <= tolerance else "failed"

    return {
        "kind": "density_kernel_convention_identity",
        "protocol": "residual-interaction bilinear identity, test 1",
        "identity": "K^acc_rho = B_rho - d_t lambda1(0) . lambda2(0)",
        "status": status,
        "parameters": {
            "vtk_dir": str(vtk_dir),
            "rank": rank,
            "dt": dt,
            "tolerance": tolerance,
            "n_snapshots": n_files,
            "n_points_owned": int(np.count_nonzero(owned)),
        },
        **identity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check the density-kernel convention identity "
            "K^acc_rho = B_rho - d_t lambda1(0).lambda2(0) "
            "on stored residual-wave snapshots of a frozen campaign."
        )
    )
    parser.add_argument("--vtk-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)

    args = parser.parse_args()

    report = run_density_kernel_convention_identity_check(
        vtk_dir=args.vtk_dir,
        rank=args.rank,
        dt=args.dt,
        tolerance=args.tolerance,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.quiet:
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "OK" if report["status"] == "ok" else "FAILED"
            values = report["values"]
            print(
                f"[DENSITY KERNEL CONVENTION IDENTITY] {status} "
                f"rank={args.rank}, n_snapshots={report['parameters']['n_snapshots']}, "
                f"staggered_rel={values['staggered_residual_relative']:.3e}, "
                f"centred_rel={values['centred_residual_relative']:.3e}, "
                f"convention_gap={values['convention_gap_boundary_over_K_acc']:.3e}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
