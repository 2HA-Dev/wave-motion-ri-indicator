from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from elastic_fwi.diagnostics.run_material_qualitative_image_artifacts import (
    run_material_qualitative_image_artifacts,
)


def test_material_qualitative_image_artifacts_generate_dof_and_nodal_outputs(
    tmp_path: Path,
):
    comm = MPI.COMM_WORLD

    result = run_material_qualitative_image_artifacts(
        output_dir=tmp_path,
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        node_aggregation="mean",
        cell_aggregation="mean",
        cell_min_nodes_per_cell=1,
        cell_plot_style="grid",
        comm=comm,
    )

    manifest_path = Path(result["manifest"])

    assert result["status"] == "ok"
    assert result["mpi_rank"] == comm.rank
    assert result["mpi_size"] == comm.size
    assert result["image_global_l2_norm"] > 0.0
    assert result["image_global_max"] > 0.0
    assert result["image_node_count"] >= 0
    assert result["image_node_aggregation"] == "mean"
    assert result["image_cell_count"] >= 0
    assert result["image_cell_aggregation"] == "mean"
    assert result["cell_plot_style"] in {"grid", "scatter-fallback"}
    assert result["defect_metric_status"] in {"ok", "empty"}
    assert result["defect_metric_num_inside_cells"] >= 0
    assert result["defect_metric_num_outside_cells"] >= 0
    assert result["defect_metric_inside_mass_fraction"] >= 0.0

    global_node_count = comm.allreduce(result["image_node_count"], op=MPI.SUM)
    global_cell_count = comm.allreduce(result["image_cell_count"], op=MPI.SUM)

    assert global_node_count > 0
    assert global_cell_count > 0

    assert manifest_path.exists()

    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    assert manifest["kind"] == "material_qualitative_image_artifacts"
    assert manifest["status"] == "ok"
    assert manifest["mpi_rank"] == comm.rank
    assert manifest["mpi_size"] == comm.size

    files = manifest["files"]

    for key in (
        "npz",
        "png",
        "pdf",
        "dof_png",
        "dof_pdf",
        "node_png",
        "node_pdf",
        "cell_png",
        "cell_pdf",
    ):
        path = Path(files[key])
        assert path.exists()
        assert path.stat().st_size > 0

    assert files["png"] == files["dof_png"]
    assert files["pdf"] == files["dof_pdf"]

    assert manifest["plots"]["dof"]["image_mode"] == "dof"
    assert manifest["plots"]["dof"]["use_nodal_image"] is False
    assert manifest["plots"]["dof"]["defect_overlay_added"] is True

    assert manifest["plots"]["node"]["image_mode"] == "node"
    assert manifest["plots"]["node"]["use_nodal_image"] is True
    assert manifest["plots"]["node"]["node_aggregation"] == "mean"
    assert manifest["plots"]["node"]["defect_overlay_added"] is True

    assert manifest["plots"]["cell"]["image_mode"] == "cell"
    assert manifest["plots"]["cell"]["use_cell_image"] is True
    assert manifest["plots"]["cell"]["cell_aggregation"] == "mean"
    assert manifest["plots"]["cell"]["cell_plot_style"] in {"grid", "scatter-fallback"}
    assert manifest["plots"]["cell"]["defect_overlay_added"] is True

    assert manifest["defect_metrics"]["status"] in {"ok", "empty"}
    assert manifest["defect_metrics"]["num_inside_cells"] >= 0
    assert manifest["defect_metrics"]["num_outside_cells"] >= 0
    assert manifest["defect_metrics"]["inside_mass_fraction"] >= 0.0

    with np.load(files["npz"], allow_pickle=False) as data:
        dof_values = data["image_local_values"]
        dof_coordinates = data["image_local_coordinates"]
        node_values = data["image_node_values"]
        node_coordinates = data["image_node_coordinates"]
        cell_values = data["image_cell_values"]
        cell_centers = data["image_cell_centers"]

        assert dof_values.ndim == 1
        assert dof_coordinates.ndim == 2
        assert dof_coordinates.shape[0] == dof_values.size
        assert dof_coordinates.shape[1] == 2

        assert node_values.ndim == 1
        assert node_coordinates.ndim == 2
        assert node_coordinates.shape[0] == node_values.size
        assert node_coordinates.shape[1] == 2

        assert cell_values.ndim == 1
        assert cell_centers.ndim == 2
        assert cell_centers.shape[0] == cell_values.size
        assert cell_centers.shape[1] == 2

        assert int(data["image_node_count"]) == node_values.size
        assert str(data["image_node_aggregation"]) == "mean"
        assert int(data["image_cell_count"]) == cell_values.size
        assert str(data["image_cell_aggregation"]) == "mean"

        assert node_values.size <= dof_values.size
        assert cell_values.size <= int(manifest["config"]["nx"]) * int(manifest["config"]["ny"])

        assert np.all(np.isfinite(dof_values))
        assert np.all(np.isfinite(dof_coordinates))
        assert np.all(np.isfinite(node_values))
        assert np.all(np.isfinite(node_coordinates))
        assert np.all(np.isfinite(cell_values))
        assert np.all(np.isfinite(cell_centers))


def test_material_qualitative_image_artifacts_can_disable_defect_overlay(
    tmp_path: Path,
):
    comm = MPI.COMM_WORLD

    result = run_material_qualitative_image_artifacts(
        output_dir=tmp_path,
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        show_defect_overlay=False,
        comm=comm,
    )

    with Path(result["manifest"]).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    assert manifest["status"] == "ok"
    assert manifest["plots"]["dof"]["defect_overlay_available"] is True
    assert manifest["plots"]["dof"]["defect_overlay_added"] is False
    assert manifest["plots"]["node"]["defect_overlay_available"] is True
    assert manifest["plots"]["node"]["defect_overlay_added"] is False
    assert manifest["plots"]["cell"]["defect_overlay_available"] is True
    assert manifest["plots"]["cell"]["defect_overlay_added"] is False
    assert manifest["plots"]["cell"]["cell_plot_style"] in {"grid", "scatter-fallback"}


def test_material_qualitative_image_artifacts_can_use_cell_scatter_style(
    tmp_path: Path,
):
    comm = MPI.COMM_WORLD

    result = run_material_qualitative_image_artifacts(
        output_dir=tmp_path,
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        node_aggregation="mean",
        cell_aggregation="mean",
        cell_min_nodes_per_cell=1,
        cell_plot_style="scatter",
        comm=comm,
    )

    assert result["status"] == "ok"
    assert result["cell_plot_style"] == "scatter"

    with Path(result["manifest"]).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    assert manifest["config"]["cell_plot_style"] == "scatter"
    assert manifest["plots"]["cell"]["image_mode"] == "cell"
    assert manifest["plots"]["cell"]["cell_plot_style"] == "scatter"
    assert Path(manifest["files"]["cell_png"]).exists()
    assert Path(manifest["files"]["cell_pdf"]).exists()
