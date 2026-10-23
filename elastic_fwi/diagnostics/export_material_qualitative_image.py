from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from mpi4py import MPI

from elastic_fwi.diagnostics.check_material_qualitative_image import (
    run_material_qualitative_image_check,
)


def _ranked_output_path(output: Path, comm: MPI.Comm) -> Path:
    if comm.size == 1:
        return output

    stem = output.stem
    suffix = output.suffix or ".npz"

    return output.with_name(
        f"{stem}_rank{comm.rank:04d}of{comm.size:04d}{suffix}"
    )


def _to_numpy_string(value: str) -> np.ndarray:
    return np.asarray(value, dtype=np.str_)


def _to_numpy_bool(value: bool) -> np.ndarray:
    return np.asarray(value, dtype=np.bool_)


def _to_numpy_int(value: int) -> np.ndarray:
    return np.asarray(value, dtype=np.int64)


def _to_numpy_float(value: float) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _empty_local_coordinates() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float64)


def _empty_local_values() -> np.ndarray:
    return np.empty((0,), dtype=np.float64)


def export_material_qualitative_image_npz(
    *,
    output: str | Path,
    nx: int = 4,
    ny: int = 4,
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
    reuse_direct_effective_solver: bool = True,
    reuse_adjoint_effective_solver: bool = True,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    output_path = Path(output)
    rank_output_path = _ranked_output_path(output_path, comm)

    report = run_material_qualitative_image_check(
        nx=nx,
        ny=ny,
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
        reuse_direct_effective_solver=reuse_direct_effective_solver,
        reuse_adjoint_effective_solver=reuse_adjoint_effective_solver,
        comm=comm,
    )

    if report["status"] != "ok":
        raise RuntimeError("Material qualitative image check failed; export aborted.")

    # Local recomputation, deliberately kept separate from the compact report:
    # the JSON report omits the local values to avoid a massive CLI output.
    from elastic_fwi.inversion.material_image_cells import (
        aggregate_nodal_image_to_structured_cells,
    )
    from elastic_fwi.inversion.material_image_defect_metrics import (
        compute_cell_defect_localization_metrics,
    )
    from elastic_fwi.inversion.material_image_nodes import (
        aggregate_dof_image_to_nodes,
    )
    from elastic_fwi.inversion.material_imaging import (
        compute_material_qualitative_dof_image,
    )
    from elastic_fwi.pipelines.adjoint_newmark import (
        AdjointNewmarkPipelineConfig,
        run_adjoint_newmark_pipeline,
    )

    pipeline = run_adjoint_newmark_pipeline(
        AdjointNewmarkPipelineConfig(
            nx=nx,
            ny=ny,
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
            reuse_direct_effective_solver=reuse_direct_effective_solver,
            reuse_adjoint_effective_solver=reuse_adjoint_effective_solver,
        ),
        comm=comm,
    )

    try:
        image = compute_material_qualitative_dof_image(
            direct_snapshots=pipeline.comparison.simulated.displacement_snapshots,
            equation_adjoint_vectors=pipeline.newmark_adjoint.equation_adjoint_vectors,
            function_space=pipeline.comparison.simulated.function_space,
            comm=comm,
            absolute_value=True,
            normalize=normalize,
        )

        if image.local_coordinates is None:
            local_coordinates = _empty_local_coordinates()
        else:
            local_coordinates = np.asarray(
                image.local_coordinates,
                dtype=np.float64,
            )

        local_values = np.asarray(image.local_values, dtype=np.float64)

        if local_coordinates.size == 0:
            node_coordinates = _empty_local_coordinates()
            node_values = _empty_local_values()
            node_count = 0
            node_local_min = 0.0
            node_local_max = 0.0
            node_local_sum = 0.0
            node_local_l2_norm = 0.0
        else:
            nodal_image = aggregate_dof_image_to_nodes(
                local_coordinates=local_coordinates,
                local_values=local_values,
                aggregation=node_aggregation,
                tolerance=node_tolerance,
            )

            node_coordinates = np.asarray(
                nodal_image.node_coordinates,
                dtype=np.float64,
            )
            node_values = np.asarray(
                nodal_image.node_values,
                dtype=np.float64,
            )
            node_count = int(nodal_image.num_nodes)
            node_local_min = float(nodal_image.local_min)
            node_local_max = float(nodal_image.local_max)
            node_local_sum = float(nodal_image.local_sum)
            node_local_l2_norm = float(nodal_image.local_l2_norm)

        if node_coordinates.size == 0:
            cell_centers = _empty_local_coordinates()
            cell_values = _empty_local_values()
            cell_count = 0
            cell_local_min = 0.0
            cell_local_max = 0.0
            cell_local_sum = 0.0
            cell_local_l2_norm = 0.0
        else:
            cell_image = aggregate_nodal_image_to_structured_cells(
                node_coordinates=node_coordinates,
                node_values=node_values,
                nx=nx,
                ny=ny,
                aggregation=cell_aggregation,
                tolerance=cell_tolerance,
                min_nodes_per_cell=cell_min_nodes_per_cell,
            )

            cell_centers = np.asarray(
                cell_image.cell_centers,
                dtype=np.float64,
            )
            cell_values = np.asarray(
                cell_image.cell_values,
                dtype=np.float64,
            )
            cell_count = int(cell_image.num_cells)
            cell_local_min = float(cell_image.local_min)
            cell_local_max = float(cell_image.local_max)
            cell_local_sum = float(cell_image.local_sum)
            cell_local_l2_norm = float(cell_image.local_l2_norm)

        rank_output_path.parent.mkdir(parents=True, exist_ok=True)

        defect_metrics = compute_cell_defect_localization_metrics(
            cell_centers=cell_centers,
            cell_values=cell_values,
            defect_center_x=defect_center_x,
            defect_center_y=defect_center_y,
            defect_radius=defect_radius,
        )

        np.savez(
            rank_output_path,
            kind=_to_numpy_string("material_qualitative_image_export"),
            status=_to_numpy_string(report["status"]),
            mpi_rank=_to_numpy_int(comm.rank),
            mpi_size=_to_numpy_int(comm.size),
            nx=_to_numpy_int(nx),
            ny=_to_numpy_int(ny),
            degree=_to_numpy_int(degree),
            theta=_to_numpy_float(theta),
            dt=_to_numpy_float(dt),
            num_steps=_to_numpy_int(num_steps),
            simulated_material_kind=_to_numpy_string(simulated_material_kind),
            observed_material_kind=_to_numpy_string(observed_material_kind),
            contrast_factor=_to_numpy_float(contrast_factor),
            defect_center_x=_to_numpy_float(defect_center_x),
            defect_center_y=_to_numpy_float(defect_center_y),
            defect_radius=_to_numpy_float(defect_radius),
            source_amplitude=_to_numpy_float(source_amplitude),
            source_frequency=_to_numpy_float(source_frequency),
            source_num_periods=_to_numpy_float(source_num_periods),
            ksp_type=_to_numpy_string(ksp_type),
            pc_type=_to_numpy_string(pc_type),
            ksp_rtol=_to_numpy_float(ksp_rtol),
            ksp_atol=_to_numpy_float(ksp_atol),
            ksp_max_it=_to_numpy_int(ksp_max_it),
            reuse_direct_effective_solver=_to_numpy_bool(
                reuse_direct_effective_solver
            ),
            reuse_adjoint_effective_solver=_to_numpy_bool(
                reuse_adjoint_effective_solver
            ),
            normalized=_to_numpy_bool(image.normalized),
            image_local_values=local_values,
            image_local_coordinates=local_coordinates,
            image_coordinate_dim=_to_numpy_int(image.coordinate_dimension),
            image_local_size=_to_numpy_int(image.local_size),
            image_global_size=_to_numpy_int(image.global_size),
            image_local_min=_to_numpy_float(image.local_min),
            image_local_max=_to_numpy_float(image.local_max),
            image_global_min=_to_numpy_float(image.global_min),
            image_global_max=_to_numpy_float(image.global_max),
            image_global_l2_norm=_to_numpy_float(image.global_l2_norm),
            image_global_sum=_to_numpy_float(image.global_sum),
            image_node_coordinates=node_coordinates,
            image_node_values=node_values,
            image_node_count=_to_numpy_int(node_count),
            image_node_aggregation=_to_numpy_string(node_aggregation),
            image_node_tolerance=_to_numpy_float(node_tolerance),
            image_node_local_min=_to_numpy_float(node_local_min),
            image_node_local_max=_to_numpy_float(node_local_max),
            image_node_local_sum=_to_numpy_float(node_local_sum),
            image_node_local_l2_norm=_to_numpy_float(node_local_l2_norm),
            image_cell_centers=cell_centers,
            image_cell_values=cell_values,
            image_cell_count=_to_numpy_int(cell_count),
            image_cell_aggregation=_to_numpy_string(cell_aggregation),
            image_cell_tolerance=_to_numpy_float(cell_tolerance),
            image_cell_min_nodes_per_cell=_to_numpy_int(cell_min_nodes_per_cell),
            image_cell_local_min=_to_numpy_float(cell_local_min),
            image_cell_local_max=_to_numpy_float(cell_local_max),
            image_cell_local_sum=_to_numpy_float(cell_local_sum),
            image_cell_local_l2_norm=_to_numpy_float(cell_local_l2_norm),
            defect_metric_status=_to_numpy_string(defect_metrics.status),
            defect_metric_num_cells=_to_numpy_int(defect_metrics.num_cells),
            defect_metric_num_inside_cells=_to_numpy_int(
                defect_metrics.num_inside_cells
            ),
            defect_metric_num_outside_cells=_to_numpy_int(
                defect_metrics.num_outside_cells
            ),
            defect_metric_max_value=_to_numpy_float(defect_metrics.max_value),
            defect_metric_max_center=np.asarray(
                defect_metrics.max_center,
                dtype=np.float64,
            ),
            defect_metric_max_distance_to_center=_to_numpy_float(
                defect_metrics.max_distance_to_defect_center
            ),
            defect_metric_max_inside_defect=_to_numpy_bool(
                defect_metrics.max_inside_defect
            ),
            defect_metric_total_indicator_mass=_to_numpy_float(
                defect_metrics.total_indicator_mass
            ),
            defect_metric_inside_indicator_mass=_to_numpy_float(
                defect_metrics.inside_indicator_mass
            ),
            defect_metric_outside_indicator_mass=_to_numpy_float(
                defect_metrics.outside_indicator_mass
            ),
            defect_metric_inside_mass_fraction=_to_numpy_float(
                defect_metrics.inside_mass_fraction
            ),
            defect_metric_inside_mean=_to_numpy_float(defect_metrics.inside_mean),
            defect_metric_outside_mean=_to_numpy_float(defect_metrics.outside_mean),
            defect_metric_inside_outside_mean_ratio=_to_numpy_float(
                defect_metrics.inside_outside_mean_ratio
            ),
            max_observation_residual=_to_numpy_float(
                report["values"]["max_observation_residual"]
            ),
            max_adjoint_norm=_to_numpy_float(report["values"]["max_adjoint_norm"]),
        )

        return {
            "kind": "material_qualitative_image_export",
            "status": "ok",
            "path": str(rank_output_path),
            "mpi_rank": comm.rank,
            "mpi_size": comm.size,
            "image_local_size": image.local_size,
            "image_global_size": image.global_size,
            "image_global_l2_norm": image.global_l2_norm,
            "image_global_max": image.global_max,
            "normalized": image.normalized,
            "reuse_direct_effective_solver": reuse_direct_effective_solver,
            "reuse_adjoint_effective_solver": reuse_adjoint_effective_solver,
            "has_coordinates": image.local_coordinates is not None,
            "coordinate_dimension": image.coordinate_dimension,
            "image_node_count": node_count,
            "image_node_aggregation": node_aggregation,
            "image_node_local_l2_norm": node_local_l2_norm,
            "image_cell_count": cell_count,
            "image_cell_aggregation": cell_aggregation,
            "image_cell_local_l2_norm": cell_local_l2_norm,
            "defect_metric_status": defect_metrics.status,
            "defect_metric_num_inside_cells": defect_metrics.num_inside_cells,
            "defect_metric_num_outside_cells": defect_metrics.num_outside_cells,
            "defect_metric_max_distance_to_center": (
                defect_metrics.max_distance_to_defect_center
            ),
            "defect_metric_max_inside_defect": defect_metrics.max_inside_defect,
            "defect_metric_inside_mass_fraction": defect_metrics.inside_mass_fraction,
            "defect_metric_inside_mean": defect_metrics.inside_mean,
            "defect_metric_outside_mean": defect_metrics.outside_mean,
            "defect_metric_inside_outside_mean_ratio": (
                defect_metrics.inside_outside_mean_ratio
            ),
        }

    finally:
        pipeline.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a qualitative material image to a pickle-free NPZ file."
    )

    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
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

    reuse_group = parser.add_mutually_exclusive_group()
    reuse_group.add_argument(
        "--reuse-effective-solver",
        dest="reuse_effective_solver",
        action="store_true",
        default=True,
        help="Reuse direct and adjoint effective Newmark solvers. Default: enabled.",
    )
    reuse_group.add_argument(
        "--no-reuse-effective-solver",
        dest="reuse_effective_solver",
        action="store_false",
        help="Use legacy one-shot effective solvers in direct and adjoint paths.",
    )

    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    export = export_material_qualitative_image_npz(
        output=args.output,
        nx=args.nx,
        ny=args.ny,
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
        reuse_direct_effective_solver=args.reuse_effective_solver,
        reuse_adjoint_effective_solver=args.reuse_effective_solver,
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        print(
            "[MATERIAL QUALITATIVE IMAGE EXPORT] OK "
            f"for output={args.output}, MPI size={comm.size}, "
            f"image_l2={export['image_global_l2_norm']:.6e}, "
            f"image_max={export['image_global_max']:.6e}, "
            f"node_count={export['image_node_count']}, "
            f"node_aggregation={export['image_node_aggregation']}, "
            f"cell_count={export['image_cell_count']}, "
            f"cell_aggregation={export['image_cell_aggregation']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
