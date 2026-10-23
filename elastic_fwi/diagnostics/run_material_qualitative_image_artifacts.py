from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mpi4py import MPI

from elastic_fwi.diagnostics.export_material_qualitative_image import (
    export_material_qualitative_image_npz,
)
from elastic_fwi.diagnostics.plot_material_qualitative_image import (
    plot_material_qualitative_image_npz,
)


def _ranked_artifact_path(path: Path, comm: MPI.Comm) -> Path:
    if comm.size == 1:
        return path

    stem = path.stem
    suffix = path.suffix

    return path.with_name(
        f"{stem}_rank{comm.rank:04d}of{comm.size:04d}{suffix}"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def run_material_qualitative_image_artifacts(
    *,
    output_dir: str | Path,
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
    cell_plot_style: str = "grid",
    show_defect_overlay: bool = True,
    reuse_direct_effective_solver: bool = True,
    reuse_adjoint_effective_solver: bool = True,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    if cell_plot_style not in {"grid", "scatter"}:
        raise ValueError("cell_plot_style must be either 'grid' or 'scatter'.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_npz = output_dir / "material_image.npz"
    base_dof_png = output_dir / "material_image_dof.png"
    base_dof_pdf = output_dir / "material_image_dof.pdf"
    base_node_png = output_dir / "material_image_node.png"
    base_node_pdf = output_dir / "material_image_node.pdf"
    base_cell_png = output_dir / "material_image_cell.png"
    base_cell_pdf = output_dir / "material_image_cell.pdf"
    base_manifest = output_dir / "material_image_manifest.json"

    dof_png = _ranked_artifact_path(base_dof_png, comm)
    dof_pdf = _ranked_artifact_path(base_dof_pdf, comm)
    node_png = _ranked_artifact_path(base_node_png, comm)
    node_pdf = _ranked_artifact_path(base_node_pdf, comm)
    cell_png = _ranked_artifact_path(base_cell_png, comm)
    cell_pdf = _ranked_artifact_path(base_cell_pdf, comm)
    manifest_path = _ranked_artifact_path(base_manifest, comm)

    export = export_material_qualitative_image_npz(
        output=base_npz,
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
        node_aggregation=node_aggregation,
        node_tolerance=node_tolerance,
        cell_aggregation=cell_aggregation,
        cell_tolerance=cell_tolerance,
        cell_min_nodes_per_cell=cell_min_nodes_per_cell,
        reuse_direct_effective_solver=reuse_direct_effective_solver,
        reuse_adjoint_effective_solver=reuse_adjoint_effective_solver,
        comm=comm,
    )

    npz_path = Path(export["path"])

    dof_plot = plot_material_qualitative_image_npz(
        input_path=npz_path,
        output_png=dof_png,
        output_pdf=dof_pdf,
        show_defect_overlay=show_defect_overlay,
        use_nodal_image=False,
    )

    node_plot = plot_material_qualitative_image_npz(
        input_path=npz_path,
        output_png=node_png,
        output_pdf=node_pdf,
        show_defect_overlay=show_defect_overlay,
        use_nodal_image=True,
    )

    cell_plot = plot_material_qualitative_image_npz(
        input_path=npz_path,
        output_png=cell_png,
        output_pdf=cell_pdf,
        show_defect_overlay=show_defect_overlay,
        use_cell_image=True,
        cell_plot_style=cell_plot_style,
    )

    manifest = {
        "kind": "material_qualitative_image_artifacts",
        "status": "ok",
        "mpi_rank": comm.rank,
        "mpi_size": comm.size,
        "config": {
            "nx": nx,
            "ny": ny,
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
            "reuse_direct_effective_solver": reuse_direct_effective_solver,
            "reuse_adjoint_effective_solver": reuse_adjoint_effective_solver,
        },
        "files": {
            "npz": str(npz_path),
            "png": str(dof_png),
            "pdf": str(dof_pdf),
            "dof_png": str(dof_png),
            "dof_pdf": str(dof_pdf),
            "node_png": str(node_png),
            "node_pdf": str(node_pdf),
            "cell_png": str(cell_png),
            "cell_pdf": str(cell_pdf),
        },
        "export": export,
        "defect_metrics": {
            "status": export["defect_metric_status"],
            "num_inside_cells": export["defect_metric_num_inside_cells"],
            "num_outside_cells": export["defect_metric_num_outside_cells"],
            "max_distance_to_center": export[
                "defect_metric_max_distance_to_center"
            ],
            "max_inside_defect": export["defect_metric_max_inside_defect"],
            "inside_mass_fraction": export[
                "defect_metric_inside_mass_fraction"
            ],
            "inside_mean": export["defect_metric_inside_mean"],
            "outside_mean": export["defect_metric_outside_mean"],
            "inside_outside_mean_ratio": export[
                "defect_metric_inside_outside_mean_ratio"
            ],
        },
        "plots": {
            "dof": dof_plot,
            "node": node_plot,
            "cell": cell_plot,
        },
    }

    _write_json(manifest_path, manifest)

    return {
        "kind": "material_qualitative_image_artifacts",
        "status": "ok",
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "files": manifest["files"],
        "mpi_rank": comm.rank,
        "mpi_size": comm.size,
        "image_global_l2_norm": export["image_global_l2_norm"],
        "image_global_max": export["image_global_max"],
        "image_node_count": export["image_node_count"],
        "image_node_aggregation": export["image_node_aggregation"],
        "image_cell_count": export["image_cell_count"],
        "image_cell_aggregation": export["image_cell_aggregation"],
        "cell_plot_style": cell_plot["cell_plot_style"],
        "defect_metric_status": export["defect_metric_status"],
        "defect_metric_num_inside_cells": export[
            "defect_metric_num_inside_cells"
        ],
        "defect_metric_num_outside_cells": export[
            "defect_metric_num_outside_cells"
        ],
        "defect_metric_max_distance_to_center": export[
            "defect_metric_max_distance_to_center"
        ],
        "defect_metric_max_inside_defect": export[
            "defect_metric_max_inside_defect"
        ],
        "defect_metric_inside_mass_fraction": export[
            "defect_metric_inside_mass_fraction"
        ],
        "defect_metric_inside_mean": export["defect_metric_inside_mean"],
        "defect_metric_outside_mean": export["defect_metric_outside_mean"],
        "defect_metric_inside_outside_mean_ratio": export[
            "defect_metric_inside_outside_mean_ratio"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate qualitative material image artifacts: NPZ, "
            "DOF plot, nodal plot and JSON manifest."
        )
    )

    parser.add_argument("--output-dir", type=str, required=True)

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
    parser.add_argument(
        "--cell-plot-style",
        choices=("grid", "scatter"),
        default="grid",
    )

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
    parser.add_argument("--no-defect-overlay", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    result = run_material_qualitative_image_artifacts(
        output_dir=args.output_dir,
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
        cell_plot_style=args.cell_plot_style,
        show_defect_overlay=not args.no_defect_overlay,
        reuse_direct_effective_solver=args.reuse_effective_solver,
        reuse_adjoint_effective_solver=args.reuse_effective_solver,
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        print(
            "[MATERIAL QUALITATIVE IMAGE ARTIFACTS] OK "
            f"for output_dir={args.output_dir}, MPI size={comm.size}, "
            f"image_l2={result['image_global_l2_norm']:.6e}, "
            f"image_max={result['image_global_max']:.6e}, "
            f"node_count={result['image_node_count']}, "
            f"node_aggregation={result['image_node_aggregation']}, "
            f"cell_count={result['image_cell_count']}, "
            f"cell_aggregation={result['image_cell_aggregation']}, "
            f"cell_plot_style={result['cell_plot_style']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
