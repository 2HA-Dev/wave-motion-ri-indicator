from __future__ import annotations

from pathlib import Path

import numpy as np
from mpi4py import MPI

from elastic_fwi.diagnostics.export_material_qualitative_image import (
    export_material_qualitative_image_npz,
)


def test_material_qualitative_image_export_contains_cell_image(tmp_path: Path):
    comm = MPI.COMM_WORLD

    output = tmp_path / "material_image.npz"

    result = export_material_qualitative_image_npz(
        output=output,
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        node_aggregation="mean",
        cell_aggregation="mean",
        cell_min_nodes_per_cell=1,
        comm=comm,
    )

    path = Path(result["path"])

    assert path.exists()
    assert result["status"] == "ok"
    assert result["image_cell_count"] >= 0
    assert result["image_cell_aggregation"] == "mean"
    assert result["image_cell_local_l2_norm"] >= 0.0

    global_cell_count = comm.allreduce(result["image_cell_count"], op=MPI.SUM)
    assert global_cell_count > 0

    with np.load(path, allow_pickle=False) as data:
        cell_values = data["image_cell_values"]
        cell_centers = data["image_cell_centers"]

        assert cell_values.ndim == 1
        assert cell_centers.ndim == 2
        assert cell_centers.shape[0] == cell_values.size
        assert cell_centers.shape[1] == 2

        assert int(data["image_cell_count"]) == cell_values.size
        assert str(data["image_cell_aggregation"]) == "mean"
        assert float(data["image_cell_tolerance"]) > 0.0
        assert int(data["image_cell_min_nodes_per_cell"]) == 1

        assert np.all(np.isfinite(cell_values))
        assert np.all(np.isfinite(cell_centers))

        assert float(data["image_cell_local_l2_norm"]) >= 0.0
        assert np.isfinite(float(data["image_cell_local_min"]))
        assert np.isfinite(float(data["image_cell_local_max"]))
        assert np.isfinite(float(data["image_cell_local_sum"]))


def test_material_qualitative_image_export_cell_count_is_not_larger_than_structured_cells(
    tmp_path: Path,
):
    comm = MPI.COMM_WORLD

    nx = 4
    ny = 4
    output = tmp_path / "material_image_cells.npz"

    result = export_material_qualitative_image_npz(
        output=output,
        nx=nx,
        ny=ny,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        cell_aggregation="mean",
        cell_min_nodes_per_cell=1,
        comm=comm,
    )

    assert result["image_cell_count"] <= nx * ny

    global_cell_count = comm.allreduce(result["image_cell_count"], op=MPI.SUM)

    assert global_cell_count >= nx * ny
