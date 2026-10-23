from __future__ import annotations

from pathlib import Path

import numpy as np
from mpi4py import MPI

from elastic_fwi.diagnostics.export_material_qualitative_image import (
    export_material_qualitative_image_npz,
)


def test_material_qualitative_image_export_contains_nodal_image(tmp_path: Path):
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
        comm=comm,
    )

    path = Path(result["path"])

    assert path.exists()
    assert result["status"] == "ok"
    assert result["image_node_count"] > 0
    assert result["image_node_aggregation"] == "mean"
    assert result["image_node_local_l2_norm"] > 0.0

    with np.load(path, allow_pickle=False) as data:
        dof_values = data["image_local_values"]
        dof_coordinates = data["image_local_coordinates"]
        node_values = data["image_node_values"]
        node_coordinates = data["image_node_coordinates"]

        assert dof_values.ndim == 1
        assert dof_coordinates.ndim == 2
        assert dof_coordinates.shape[0] == dof_values.size
        assert dof_coordinates.shape[1] == 2

        assert node_values.ndim == 1
        assert node_coordinates.ndim == 2
        assert node_coordinates.shape[0] == node_values.size
        assert node_coordinates.shape[1] == 2

        assert int(data["image_node_count"]) == node_values.size
        assert str(data["image_node_aggregation"]) == "mean"
        assert float(data["image_node_tolerance"]) > 0.0

        assert node_values.size <= dof_values.size
        assert node_values.size > 0

        assert np.all(np.isfinite(node_values))
        assert np.all(np.isfinite(node_coordinates))

        assert float(data["image_node_local_l2_norm"]) > 0.0
        assert np.isfinite(float(data["image_node_local_min"]))
        assert np.isfinite(float(data["image_node_local_max"]))
        assert np.isfinite(float(data["image_node_local_sum"]))
