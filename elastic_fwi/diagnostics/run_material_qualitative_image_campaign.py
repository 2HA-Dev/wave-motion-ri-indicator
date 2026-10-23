from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from mpi4py import MPI

from elastic_fwi.diagnostics.plot_material_qualitative_image_campaign import (
    plot_material_qualitative_image_campaign_summary,
)
from elastic_fwi.diagnostics.run_material_qualitative_image_artifacts import (
    run_material_qualitative_image_artifacts,
)


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())

    if not values:
        raise ValueError("At least one integer value is required.")

    if any(item <= 0 for item in values):
        raise ValueError("All integer values must be strictly positive.")

    return values


def _ranked_path(path: Path, comm: MPI.Comm) -> Path:
    if comm.size == 1:
        return path

    return path.with_name(
        f"{path.stem}_rank{comm.rank:04d}of{comm.size:04d}{path.suffix}"
    )


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "nx",
        "ny",
        "degree",
        "num_steps",
        "mpi_rank",
        "mpi_size",
        "status",
        "npz",
        "dof_png",
        "dof_pdf",
        "node_png",
        "node_pdf",
        "image_global_l2_norm",
        "image_global_max",
        "image_node_count",
        "image_node_aggregation",
        "image_cell_count",
        "image_cell_aggregation",
        "defect_metric_status",
        "defect_metric_num_inside_cells",
        "defect_metric_num_outside_cells",
        "defect_metric_max_distance_to_center",
        "defect_metric_max_inside_defect",
        "defect_metric_inside_mass_fraction",
        "defect_metric_inside_mean",
        "defect_metric_outside_mean",
        "defect_metric_inside_outside_mean_ratio",
    ]

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def _safe_ratios(values: list[float]) -> list[float]:
    ratios: list[float] = []

    for previous, current in zip(values[:-1], values[1:], strict=False):
        if previous == 0.0:
            ratios.append(float("nan"))
        else:
            ratios.append(float(current / previous))

    return ratios


def _all_finite(values: list[float]) -> bool:
    return all(value == value and abs(value) < float("inf") for value in values)


def _monotone_non_decreasing(values: list[float]) -> bool:
    return all(
        current >= previous
        for previous, current in zip(values[:-1], values[1:], strict=False)
    )


def _monotone_non_increasing(values: list[float]) -> bool:
    return all(
        current <= previous
        for previous, current in zip(values[:-1], values[1:], strict=False)
    )


def _index_of_max(values: list[float]) -> int | None:
    if not values:
        return None

    return int(max(range(len(values)), key=lambda index: values[index]))


def _index_of_min(values: list[float]) -> int | None:
    if not values:
        return None

    return int(min(range(len(values)), key=lambda index: values[index]))


def _compute_final_assessment(diagnostics: dict[str, Any]) -> dict[str, Any]:
    n_values = [int(value) for value in diagnostics["n_values"]]

    inside_mass_fractions = [
        float(value)
        for value in diagnostics.get("defect_inside_mass_fractions", [])
    ]
    max_distances_ok = [
        float(value)
        for value in diagnostics.get("defect_max_distances_to_center_ok", [])
    ]

    statuses = [
        str(value)
        for value in diagnostics.get("defect_metric_statuses", [])
    ]

    ok_n_values = [
        n_values[index]
        for index, status in enumerate(statuses)
        if status == "ok"
    ]

    best_inside_index = _index_of_max(inside_mass_fractions)
    best_distance_index = _index_of_min(max_distances_ok)

    inside_mass_fraction_trend_non_decreasing = _monotone_non_decreasing(
        inside_mass_fractions
    )
    max_distance_trend_non_increasing = _monotone_non_increasing(max_distances_ok)

    best_n_by_inside_mass_fraction = (
        n_values[best_inside_index] if best_inside_index is not None else None
    )
    best_n_by_max_distance = (
        ok_n_values[best_distance_index] if best_distance_index is not None else None
    )

    if (
        diagnostics.get("status") == "ok"
        and diagnostics.get("defect_metric_finite") is True
        and inside_mass_fractions
        and max_distances_ok
    ):
        qualitative_status = "measured"
    else:
        qualitative_status = "incomplete"

    return {
        "kind": "material_qualitative_image_campaign_final_assessment",
        "status": qualitative_status,
        "num_cases": len(n_values),
        "n_values": n_values,
        "ok_n_values_for_distance": ok_n_values,
        "inside_mass_fraction_trend_non_decreasing": (
            inside_mass_fraction_trend_non_decreasing
        ),
        "max_distance_trend_non_increasing": max_distance_trend_non_increasing,
        "best_n_by_inside_mass_fraction": best_n_by_inside_mass_fraction,
        "best_n_by_max_distance": best_n_by_max_distance,
        "best_inside_mass_fraction": (
            inside_mass_fractions[best_inside_index]
            if best_inside_index is not None
            else None
        ),
        "best_max_distance_to_center": (
            max_distances_ok[best_distance_index]
            if best_distance_index is not None
            else None
        ),
        "interpretation": (
            "diagnostic_only_no_success_assertion"
            if qualitative_status == "measured"
            else "insufficient_data"
        ),
    }


def _compute_campaign_diagnostics(
    *,
    rows: list[dict[str, Any]],
    normalize: bool,
    comm: MPI.Comm,
) -> dict[str, Any]:
    n_values = [int(row["nx"]) for row in rows]
    l2_values = [float(row["image_global_l2_norm"]) for row in rows]
    max_values = [float(row["image_global_max"]) for row in rows]
    defect_metric_statuses = [
        str(row["defect_metric_status"]) for row in rows
    ]
    inside_mass_fractions = [
        float(row["defect_metric_inside_mass_fraction"]) for row in rows
    ]
    max_distances_to_defect = [
        float(row["defect_metric_max_distance_to_center"]) for row in rows
    ]
    inside_outside_mean_ratios = [
        float(row["defect_metric_inside_outside_mean_ratio"]) for row in rows
    ]

    defect_ok_indices = [
        index
        for index, status in enumerate(defect_metric_statuses)
        if status == "ok"
    ]
    defect_valid_statuses = all(
        status in {"ok", "empty"} for status in defect_metric_statuses
    )

    max_distances_to_defect_ok = [
        max_distances_to_defect[index] for index in defect_ok_indices
    ]
    inside_outside_mean_ratios_ok = [
        inside_outside_mean_ratios[index] for index in defect_ok_indices
    ]
    local_node_counts = [int(row["image_node_count"]) for row in rows]
    local_cell_counts = [int(row["image_cell_count"]) for row in rows]

    global_node_counts = [
        int(comm.allreduce(local_count, op=MPI.SUM))
        for local_count in local_node_counts
    ]
    global_cell_counts = [
        int(comm.allreduce(local_count, op=MPI.SUM))
        for local_count in local_cell_counts
    ]

    l2_ratios = _safe_ratios(l2_values)
    node_count_ratios = _safe_ratios([float(value) for value in global_node_counts])
    cell_count_ratios = _safe_ratios([float(value) for value in global_cell_counts])

    l2_positive = all(value > 0.0 for value in l2_values)
    l2_finite = _all_finite(l2_values)

    max_positive = all(value > 0.0 for value in max_values)
    max_finite = _all_finite(max_values)
    defect_metric_finite = (
        defect_valid_statuses
        and _all_finite(inside_mass_fractions)
        and _all_finite(max_distances_to_defect_ok)
        and _all_finite(inside_outside_mean_ratios_ok)
    )

    global_node_count_positive = all(value > 0 for value in global_node_counts)
    global_node_count_monotone = all(
        current >= previous
        for previous, current in zip(
            global_node_counts[:-1],
            global_node_counts[1:],
            strict=False,
        )
    )

    global_cell_count_positive = all(value > 0 for value in global_cell_counts)
    global_cell_count_monotone = all(
        current >= previous
        for previous, current in zip(
            global_cell_counts[:-1],
            global_cell_counts[1:],
            strict=False,
        )
    )

    max_abs_deviation_from_one = max(
        (abs(value - 1.0) for value in max_values),
        default=0.0,
    )

    max_close_to_one = (
        bool(max_abs_deviation_from_one <= 1.0e-12)
        if normalize
        else None
    )

    status = "ok"

    if not l2_positive or not l2_finite:
        status = "failed"

    if not max_positive or not max_finite:
        status = "failed"

    if not global_node_count_positive:
        status = "failed"

    if not global_node_count_monotone:
        status = "failed"

    if not global_cell_count_positive:
        status = "failed"

    if not global_cell_count_monotone:
        status = "failed"

    if normalize and not max_close_to_one:
        status = "failed"

    if not defect_metric_finite:
        status = "failed"

    return {
        "kind": "material_qualitative_image_campaign_diagnostics",
        "status": status,
        "mpi_rank": comm.rank,
        "mpi_size": comm.size,
        "n_values": n_values,
        "l2_values": l2_values,
        "max_values": max_values,
        "local_node_counts": local_node_counts,
        "global_node_counts": global_node_counts,
        "local_cell_counts": local_cell_counts,
        "global_cell_counts": global_cell_counts,
        "l2_growth_ratios": l2_ratios,
        "global_node_count_growth_ratios": node_count_ratios,
        "global_cell_count_growth_ratios": cell_count_ratios,
        "l2_positive": l2_positive,
        "l2_finite": l2_finite,
        "max_positive": max_positive,
        "max_finite": max_finite,
        "defect_metric_finite": defect_metric_finite,
        "defect_metric_statuses": defect_metric_statuses,
        "defect_metric_valid_statuses": defect_valid_statuses,
        "defect_inside_mass_fractions": inside_mass_fractions,
        "defect_max_distances_to_center": max_distances_to_defect,
        "defect_inside_outside_mean_ratios": inside_outside_mean_ratios,
        "defect_max_distances_to_center_ok": max_distances_to_defect_ok,
        "defect_inside_outside_mean_ratios_ok": inside_outside_mean_ratios_ok,
        "normalize": normalize,
        "max_abs_deviation_from_one": max_abs_deviation_from_one,
        "max_close_to_one": max_close_to_one,
        "global_node_count_positive": global_node_count_positive,
        "global_node_count_monotone_non_decreasing": global_node_count_monotone,
        "global_cell_count_positive": global_cell_count_positive,
        "global_cell_count_monotone_non_decreasing": global_cell_count_monotone,
    }


def run_material_qualitative_image_campaign(
    *,
    output_dir: str | Path,
    n_values: tuple[int, ...] = (2, 4, 8, 16, 32),
    degree: int = 1,
    theta: float = 0.25,
    dt: float = 0.1,
    num_steps: int = 5,
    simulated_material_kind: str = "healthy",
    observed_material_kind: str = "localized_disk_defect",
    contrast_factor: float = 0.8,
    defect_center_x: float = 0.5,
    defect_center_y: float = 0.5,
    defect_radius: float = 0.35,
    source_amplitude: float = 1.0,
    source_frequency: float = 1.0,
    source_num_periods: float = 1.0,
    ksp_type: str = "cg",
    pc_type: str = "jacobi",
    ksp_rtol: float = 1.0e-10,
    ksp_atol: float = 1.0e-12,
    ksp_max_it: int = 1000,
    normalize: bool = True,
    node_aggregation: str = "mean",
    node_tolerance: float = 1.0e-12,
    cell_aggregation: str = "mean",
    cell_tolerance: float = 1.0e-12,
    cell_min_nodes_per_cell: int = 1,
    cell_plot_style: str = "grid",
    show_defect_overlay: bool = True,
    generate_summary_plots: bool = True,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    if cell_plot_style not in {"grid", "scatter"}:
        raise ValueError("cell_plot_style must be either 'grid' or 'scatter'.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    for n_value in n_values:
        case_dir = output_dir / f"N{n_value:04d}"

        result = run_material_qualitative_image_artifacts(
            output_dir=case_dir,
            nx=n_value,
            ny=n_value,
            degree=degree,
            theta=theta,
            dt=dt,
            num_steps=num_steps,
            simulated_material_kind=simulated_material_kind,
            observed_material_kind=observed_material_kind,
            contrast_factor=contrast_factor,
            defect_center_x=defect_center_x,
            defect_center_y=defect_center_y,
            defect_radius=defect_radius,
            source_amplitude=source_amplitude,
            source_frequency=source_frequency,
            source_num_periods=source_num_periods,
            ksp_type=ksp_type,
            pc_type=pc_type,
            ksp_rtol=ksp_rtol,
            ksp_atol=ksp_atol,
            ksp_max_it=ksp_max_it,
            normalize=normalize,
            node_aggregation=node_aggregation,
            node_tolerance=node_tolerance,
            cell_aggregation=cell_aggregation,
            cell_tolerance=cell_tolerance,
            cell_min_nodes_per_cell=cell_min_nodes_per_cell,
            cell_plot_style=cell_plot_style,
            show_defect_overlay=show_defect_overlay,
            comm=comm,
        )

        row = {
            "nx": n_value,
            "ny": n_value,
            "degree": degree,
            "num_steps": num_steps,
            "mpi_rank": comm.rank,
            "mpi_size": comm.size,
            "status": result["status"],
            "npz": result["files"]["npz"],
            "dof_png": result["files"]["dof_png"],
            "dof_pdf": result["files"]["dof_pdf"],
            "node_png": result["files"]["node_png"],
            "node_pdf": result["files"]["node_pdf"],
            "image_global_l2_norm": result["image_global_l2_norm"],
            "image_global_max": result["image_global_max"],
            "image_node_count": result["image_node_count"],
            "image_node_aggregation": result["image_node_aggregation"],
            "image_cell_count": result["image_cell_count"],
            "image_cell_aggregation": result["image_cell_aggregation"],
            "defect_metric_status": result["defect_metric_status"],
            "defect_metric_num_inside_cells": result[
                "defect_metric_num_inside_cells"
            ],
            "defect_metric_num_outside_cells": result[
                "defect_metric_num_outside_cells"
            ],
            "defect_metric_max_distance_to_center": result[
                "defect_metric_max_distance_to_center"
            ],
            "defect_metric_max_inside_defect": result[
                "defect_metric_max_inside_defect"
            ],
            "defect_metric_inside_mass_fraction": result[
                "defect_metric_inside_mass_fraction"
            ],
            "defect_metric_inside_mean": result["defect_metric_inside_mean"],
            "defect_metric_outside_mean": result["defect_metric_outside_mean"],
            "defect_metric_inside_outside_mean_ratio": result[
                "defect_metric_inside_outside_mean_ratio"
            ],
        }

        rows.append(row)
        cases.append(
            {
                "N": n_value,
                "artifact_manifest": result["manifest"],
                "files": result["files"],
                "summary": row,
            }
        )

    summary_csv = _ranked_path(output_dir / "material_image_campaign_summary.csv", comm)
    manifest_json = _ranked_path(
        output_dir / "material_image_campaign_manifest.json",
        comm,
    )

    _write_summary_csv(summary_csv, rows)

    summary_plot_manifest: dict[str, Any] | None = None

    if generate_summary_plots:
        summary_plot_dir = output_dir / "summary_plots"

        if comm.size > 1:
            summary_plot_dir = output_dir / f"summary_plots_rank{comm.rank:04d}of{comm.size:04d}"

        summary_plot_manifest = plot_material_qualitative_image_campaign_summary(
            summary_csv=summary_csv,
            output_dir=summary_plot_dir,
            prefix="material_image_campaign",
        )

    campaign_diagnostics = _compute_campaign_diagnostics(
        rows=rows,
        normalize=normalize,
        comm=comm,
    )

    if campaign_diagnostics["status"] != "ok":
        raise RuntimeError("Material qualitative image campaign diagnostics failed.")

    final_assessment = _compute_final_assessment(campaign_diagnostics)

    manifest = {
        "kind": "material_qualitative_image_campaign",
        "status": "ok",
        "mpi_rank": comm.rank,
        "mpi_size": comm.size,
        "config": {
            "n_values": list(n_values),
            "degree": degree,
            "theta": theta,
            "dt": dt,
            "num_steps": num_steps,
            "simulated_material_kind": simulated_material_kind,
            "observed_material_kind": observed_material_kind,
            "contrast_factor": contrast_factor,
            "defect_center_x": defect_center_x,
            "defect_center_y": defect_center_y,
            "defect_radius": defect_radius,
            "source_amplitude": source_amplitude,
            "source_frequency": source_frequency,
            "source_num_periods": source_num_periods,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": ksp_rtol,
            "ksp_atol": ksp_atol,
            "ksp_max_it": ksp_max_it,
            "normalize": normalize,
            "node_aggregation": node_aggregation,
            "node_tolerance": node_tolerance,
            "cell_aggregation": cell_aggregation,
            "cell_tolerance": cell_tolerance,
            "cell_min_nodes_per_cell": cell_min_nodes_per_cell,
            "cell_plot_style": cell_plot_style,
            "show_defect_overlay": show_defect_overlay,
            "generate_summary_plots": generate_summary_plots,
        },
        "files": {
            "summary_csv": str(summary_csv),
            "manifest_json": str(manifest_json),
        },
        "cases": cases,
        "summary_plots": summary_plot_manifest,
        "diagnostics": campaign_diagnostics,
        "final_assessment": final_assessment,
    }

    if summary_plot_manifest is not None:
        manifest["files"]["summary_plot_manifest"] = summary_plot_manifest["files"][
            "manifest_json"
        ]

    _write_json(manifest_json, manifest)

    return {
        "kind": "material_qualitative_image_campaign",
        "status": "ok",
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv),
        "manifest_json": str(manifest_json),
        "manifest": str(manifest_json),
        "summary_plot_manifest": (
            summary_plot_manifest["files"]["manifest_json"]
            if summary_plot_manifest is not None
            else ""
        ),
        "num_cases": len(rows),
        "mpi_rank": comm.rank,
        "mpi_size": comm.size,
        "rows": rows,
        "summary_plots": summary_plot_manifest,
        "diagnostics": campaign_diagnostics,
        "final_assessment": final_assessment,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a qualitative material image campaign over several mesh sizes."
    )

    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--N-values", type=str, default="2,4,8,16,32")

    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=5)

    parser.add_argument("--simulated-material-kind", type=str, default="healthy")
    parser.add_argument(
        "--observed-material-kind",
        type=str,
        default="localized_disk_defect",
    )
    parser.add_argument("--contrast-factor", type=float, default=0.8)
    parser.add_argument("--defect-center-x", type=float, default=0.5)
    parser.add_argument("--defect-center-y", type=float, default=0.5)
    parser.add_argument("--defect-radius", type=float, default=0.35)

    parser.add_argument("--source-amplitude", type=float, default=1.0)
    parser.add_argument("--source-frequency", type=float, default=1.0)
    parser.add_argument("--source-num-periods", type=float, default=1.0)

    parser.add_argument("--ksp-type", type=str, default="cg")
    parser.add_argument("--pc-type", type=str, default="jacobi")
    parser.add_argument("--ksp-rtol", type=float, default=1.0e-10)
    parser.add_argument("--ksp-atol", type=float, default=1.0e-12)
    parser.add_argument("--ksp-max-it", type=int, default=1000)

    parser.add_argument("--node-aggregation", type=str, default="mean")
    parser.add_argument("--node-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--cell-aggregation", type=str, default="mean")
    parser.add_argument("--cell-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--cell-min-nodes-per-cell", type=int, default=1)
    parser.add_argument(
        "--cell-plot-style",
        choices=("grid", "scatter"),
        default="grid",
    )

    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--no-defect-overlay", action="store_true")
    parser.add_argument("--no-summary-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    result = run_material_qualitative_image_campaign(
        output_dir=args.output_dir,
        n_values=_parse_int_list(args.N_values),
        degree=args.degree,
        theta=args.theta,
        dt=args.dt,
        num_steps=args.num_steps,
        simulated_material_kind=args.simulated_material_kind,
        observed_material_kind=args.observed_material_kind,
        contrast_factor=args.contrast_factor,
        defect_center_x=args.defect_center_x,
        defect_center_y=args.defect_center_y,
        defect_radius=args.defect_radius,
        source_amplitude=args.source_amplitude,
        source_frequency=args.source_frequency,
        source_num_periods=args.source_num_periods,
        ksp_type=args.ksp_type,
        pc_type=args.pc_type,
        ksp_rtol=args.ksp_rtol,
        ksp_atol=args.ksp_atol,
        ksp_max_it=args.ksp_max_it,
        normalize=not args.no_normalize,
        node_aggregation=args.node_aggregation,
        node_tolerance=args.node_tolerance,
        cell_aggregation=args.cell_aggregation,
        cell_tolerance=args.cell_tolerance,
        cell_min_nodes_per_cell=args.cell_min_nodes_per_cell,
        cell_plot_style=args.cell_plot_style,
        show_defect_overlay=not args.no_defect_overlay,
        generate_summary_plots=not args.no_summary_plots,
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        print(
            "[MATERIAL QUALITATIVE IMAGE CAMPAIGN] OK "
            f"for output_dir={args.output_dir}, "
            f"MPI size={comm.size}, "
            f"num_cases={result['num_cases']}, "
            f"summary={result['summary_csv']}, "
            f"summary_plots={result['summary_plot_manifest'] or 'disabled'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
