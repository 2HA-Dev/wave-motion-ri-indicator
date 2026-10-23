from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from math import isfinite, log10, sqrt
from pathlib import Path
from typing import Any

import numpy as np
from mpi4py import MPI

from elastic_fwi.pipelines.direct_observation import (
    LEGACY_HEALTHY_MATERIAL,
    DirectObservationPipelineConfig,
    run_direct_observation_pipeline,
)


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be strictly positive and finite.")
    return value


def _get_time_values(result) -> np.ndarray:
    for name in ("time_values", "times", "t_values"):
        if hasattr(result, name):
            values = np.asarray(getattr(result, name), dtype=float)
            if values.ndim == 1 and values.size > 0:
                return values
    raise RuntimeError("Unable to find time values in DirectObservationPipelineResult.")


def _get_snapshots(result) -> list[np.ndarray]:
    for name in (
        "displacement_snapshots",
        "snapshots",
        "u_snapshots",
        "solution_snapshots",
    ):
        if hasattr(result, name):
            values = getattr(result, name)
            if values is not None:
                return [np.asarray(v, dtype=float).copy() for v in values]
    raise RuntimeError("Unable to find displacement snapshots in DirectObservationPipelineResult.")


def _snapshot_norm2(snapshot: np.ndarray) -> float:
    values = np.asarray(snapshot, dtype=float).ravel()
    if values.size == 0:
        return 0.0
    return float(np.dot(values, values) / values.size)


def _energy_series_from_snapshots(snapshots: list[np.ndarray]) -> np.ndarray:
    return np.asarray([_snapshot_norm2(s) for s in snapshots], dtype=float)


def _source_center_from_config(config: DirectObservationPipelineConfig) -> tuple[float, float]:
    if config.source_kind == "internal_gaussian_array":
        return (
            0.5 * (float(config.source_array_start_x) + float(config.source_array_end_x)),
            0.5 * (float(config.source_array_start_y) + float(config.source_array_end_y)),
        )

    return (
        float(config.source_center_x),
        float(config.source_center_y),
    )


def _estimate_cmax() -> float:
    material = LEGACY_HEALTHY_MATERIAL
    rho = _positive_float(material.rho, "rho")
    C11 = _positive_float(material.C11, "C11")
    C22 = _positive_float(material.C22, "C22")
    C66 = _positive_float(material.C66, "C66")
    return sqrt(max(C11, C22, C66) / rho)


def _estimate_return_time(config: DirectObservationPipelineConfig) -> dict[str, float]:
    sx, sy = _source_center_from_config(config)

    distance_to_boundary = min(
        sx,
        float(config.domain_length_x) - sx,
        sy,
        float(config.domain_length_y) - sy,
    )

    cmax = _estimate_cmax()
    return_time = 2.0 * distance_to_boundary / cmax

    return {
        "source_center_x": sx,
        "source_center_y": sy,
        "nearest_boundary_distance": distance_to_boundary,
        "cmax_estimate": cmax,
        "return_time_estimate": return_time,
    }


def _build_config(
    *,
    absorbing_layer: bool,
    nx: int,
    ny: int,
    dt: float,
    num_steps: int,
    absorbing_layer_width_wavelengths: float,
    absorbing_layer_strength: float,
    absorbing_layer_polynomial_order: float,
    absorbing_layer_min_cells: int,
) -> DirectObservationPipelineConfig:
    return DirectObservationPipelineConfig(
        mesh_kind="gmsh_refined",
        mesh_auto_from_wavelength=True,
        mesh_target_ppw=10.0,
        mesh_near_target_ppw=20.0,
        absorbing_layer=absorbing_layer,
        absorbing_layer_width_wavelengths=absorbing_layer_width_wavelengths,
        absorbing_layer_min_cells=absorbing_layer_min_cells,
        absorbing_layer_strength=absorbing_layer_strength,
        absorbing_layer_polynomial_order=absorbing_layer_polynomial_order,
        mesh_refine_source=True,
        mesh_refine_defect=True,
        mesh_refine_source_radius=1.25,
        mesh_refine_defect_radius=0.75,
        nx=nx,
        ny=ny,
        domain_length_x=10.0,
        domain_length_y=10.0,
        source_kind="internal_gaussian_array",
        source_array_start_x=4.0,
        source_array_start_y=5.0,
        source_array_end_x=6.0,
        source_array_end_y=5.0,
        source_array_num_elements=9,
        source_sigma=0.08,
        source_direction_x=0.0,
        source_direction_y=1.0,
        receiver_kind="boundary_segment",
        receiver_side="top",
        receiver_center=5.0,
        receiver_length=2.0,
        material_kind="localized_disk_defect",
        defect_center_x=2.0,
        defect_center_y=2.0,
        defect_radius=0.25,
        contrast_factor=0.8,
        dt=dt,
        num_steps=num_steps,
        source_frequency=1.0,
        store_displacement_snapshots=True,
    )


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0.0:
        return float("inf")
    return float(num / den)


def _db_from_ratio(ratio: float) -> float:
    if ratio <= 0.0:
        return float("-inf")
    return float(10.0 * log10(ratio))


def run_absorbing_layer_reflection_check(
    *,
    output_dir: Path,
    nx: int,
    ny: int,
    dt: float,
    num_steps: int,
    every_k: int,
    acceptance_ratio: float,
    absorbing_layer_width_wavelengths: float = 1.5,
    absorbing_layer_strength: float = 3.0,
    absorbing_layer_polynomial_order: float = 2.0,
    absorbing_layer_min_cells: int = 6,
    comm: MPI.Comm,
) -> dict[str, Any]:
    rank = comm.rank
    output_dir.mkdir(parents=True, exist_ok=True)

    every_k = int(every_k)
    if every_k <= 0:
        raise ValueError("every_k must be strictly positive.")

    config_no_abs = _build_config(
        absorbing_layer=False,
        nx=nx,
        ny=ny,
        dt=dt,
        num_steps=num_steps,
        absorbing_layer_width_wavelengths=absorbing_layer_width_wavelengths,
        absorbing_layer_strength=absorbing_layer_strength,
        absorbing_layer_polynomial_order=absorbing_layer_polynomial_order,
        absorbing_layer_min_cells=absorbing_layer_min_cells,
    )

    config_abs = _build_config(
        absorbing_layer=True,
        nx=nx,
        ny=ny,
        dt=dt,
        num_steps=num_steps,
        absorbing_layer_width_wavelengths=absorbing_layer_width_wavelengths,
        absorbing_layer_strength=absorbing_layer_strength,
        absorbing_layer_polynomial_order=absorbing_layer_polynomial_order,
        absorbing_layer_min_cells=absorbing_layer_min_cells,
    )

    result_no_abs = run_direct_observation_pipeline(config=config_no_abs, comm=comm)
    result_abs = run_direct_observation_pipeline(config=config_abs, comm=comm)

    raw_snapshots_no_abs = _get_snapshots(result_no_abs)
    raw_snapshots_abs = _get_snapshots(result_abs)

    if len(raw_snapshots_no_abs) != len(raw_snapshots_abs):
        raise RuntimeError("The two runs do not have the same number of raw snapshots.")

    raw_times = dt * np.arange(1, len(raw_snapshots_abs) + 1, dtype=float)

    times_no_abs = raw_times[::every_k]
    times_abs = raw_times[::every_k]

    snapshots_no_abs = raw_snapshots_no_abs[::every_k]
    snapshots_abs = raw_snapshots_abs[::every_k]

    if len(snapshots_no_abs) != len(snapshots_abs):
        raise RuntimeError("The two runs do not have the same number of snapshots.")

    if times_no_abs.shape != times_abs.shape or not np.allclose(times_no_abs, times_abs):
        raise RuntimeError("The two runs do not have matching time grids.")

    energy_no_abs = _energy_series_from_snapshots(snapshots_no_abs)
    energy_abs = _energy_series_from_snapshots(snapshots_abs)

    return_info = _estimate_return_time(config_abs)
    t_return = return_info["return_time_estimate"]

    late_mask = times_abs >= t_return
    if not np.any(late_mask):
        raise RuntimeError(
            "No exported snapshot occurs after the estimated boundary-return time. "
            "Increase num_steps or decrease every_k."
        )

    peak_no_abs_total = float(np.max(energy_no_abs))
    peak_abs_total = float(np.max(energy_abs))

    late_peak_no_abs = float(np.max(energy_no_abs[late_mask]))
    late_peak_abs = float(np.max(energy_abs[late_mask]))

    suppression_ratio = _safe_ratio(late_peak_abs, late_peak_no_abs)
    total_ratio = _safe_ratio(peak_abs_total, peak_no_abs_total)

    status = "ok" if suppression_ratio <= acceptance_ratio else "warning"

    summary = {
        "kind": "absorbing_layer_reflection_check",
        "status": status,
        "acceptance_ratio": float(acceptance_ratio),
        "time": {
            "dt": float(dt),
            "num_steps": int(num_steps),
            "final_time": float(dt * num_steps),
            "every_k": int(every_k),
            "num_snapshots": int(len(times_abs)),
            "return_time_estimate": float(t_return),
        },
        "return_time_estimate": return_info,
        "energy": {
            "peak_no_abs_total": peak_no_abs_total,
            "peak_abs_total": peak_abs_total,
            "total_peak_ratio_abs_over_no_abs": total_ratio,
            "late_peak_no_abs": late_peak_no_abs,
            "late_peak_abs": late_peak_abs,
            "late_energy_suppression_ratio": suppression_ratio,
            "late_energy_suppression_db": _db_from_ratio(suppression_ratio),
        },
        "absorbing_layer_parameters": {
            "width_wavelengths": float(absorbing_layer_width_wavelengths),
            "strength": float(absorbing_layer_strength),
            "polynomial_order": float(absorbing_layer_polynomial_order),
            "min_cells": int(absorbing_layer_min_cells),
        },
        "absorbing_layer": result_abs.mesh_metadata.get("absorbing_layer", {"enabled": False}),
        "config_no_abs": config_no_abs.to_dict() if hasattr(config_no_abs, "to_dict") else asdict(config_no_abs),
        "config_abs": config_abs.to_dict() if hasattr(config_abs, "to_dict") else asdict(config_abs),
    }

    if rank == 0:
        json_path = output_dir / "absorbing_layer_reflection_summary.json"
        csv_path = output_dir / "absorbing_layer_reflection_energy.csv"

        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "time",
                    "energy_no_abs",
                    "energy_abs",
                    "late_window",
                ],
            )
            writer.writeheader()
            for t, e0, e1, late in zip(times_abs, energy_no_abs, energy_abs, late_mask):
                writer.writerow(
                    {
                        "time": float(t),
                        "energy_no_abs": float(e0),
                        "energy_abs": float(e1),
                        "late_window": int(bool(late)),
                    }
                )

        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            fig_path = output_dir / "absorbing_layer_reflection_energy.png"

            plt.figure()
            plt.semilogy(times_abs, energy_no_abs + 1e-300, label="without absorbing layer")
            plt.semilogy(times_abs, energy_abs + 1e-300, label="with absorbing layer")
            plt.axvline(t_return, linestyle="--", label="estimated return time")
            plt.xlabel("time")
            plt.ylabel("mean squared displacement")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=200)
            plt.close()

            summary["figure"] = str(fig_path)
            json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        except Exception as exc:
            summary["figure_error"] = repr(exc)
            json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whether the absorbing layer suppresses artificial boundary reflections."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--every-k", type=int, default=1)
    parser.add_argument("--acceptance-ratio", type=float, default=0.2)
    parser.add_argument("--absorbing-layer-width-wavelengths", type=float, default=1.5)
    parser.add_argument("--absorbing-layer-strength", type=float, default=3.0)
    parser.add_argument("--absorbing-layer-polynomial-order", type=float, default=2.0)
    parser.add_argument("--absorbing-layer-min-cells", type=int, default=6)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)

    summary = run_absorbing_layer_reflection_check(
        output_dir=args.output_dir,
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        num_steps=args.num_steps,
        every_k=args.every_k,
        acceptance_ratio=args.acceptance_ratio,
        absorbing_layer_width_wavelengths=args.absorbing_layer_width_wavelengths,
        absorbing_layer_strength=args.absorbing_layer_strength,
        absorbing_layer_polynomial_order=args.absorbing_layer_polynomial_order,
        absorbing_layer_min_cells=args.absorbing_layer_min_cells,
        comm=MPI.COMM_WORLD,
    )

    if MPI.COMM_WORLD.rank == 0 and not args.quiet:
        energy = summary["energy"]
        print("[ABSORBING LAYER REFLECTION CHECK]", summary["status"])
        print("late_energy_suppression_ratio =", energy["late_energy_suppression_ratio"])
        print("late_energy_suppression_db =", energy["late_energy_suppression_db"])
        print("return_time_estimate =", summary["time"]["return_time_estimate"])
        print("final_time =", summary["time"]["final_time"])

    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
