from __future__ import annotations

from pathlib import Path

import numpy as np
from mpi4py import MPI

from elastic_fwi.diagnostics.export_material_qualitative_image import (
    export_material_qualitative_image_npz,
)


def test_material_qualitative_image_export_contains_defect_metrics(tmp_path: Path):
    comm = MPI.COMM_WORLD

    output = tmp_path / "material_image_metrics.npz"

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
    assert result["defect_metric_status"] in {"ok", "empty"}
    assert result["defect_metric_num_inside_cells"] >= 0
    assert result["defect_metric_num_outside_cells"] >= 0
    assert result["defect_metric_inside_mass_fraction"] >= 0.0
    assert np.isfinite(result["defect_metric_inside_mean"])
    assert np.isfinite(result["defect_metric_outside_mean"])

    global_inside_cells = comm.allreduce(
        result["defect_metric_num_inside_cells"],
        op=MPI.SUM,
    )
    global_outside_cells = comm.allreduce(
        result["defect_metric_num_outside_cells"],
        op=MPI.SUM,
    )

    assert global_inside_cells >= 0
    assert global_outside_cells >= 0
    assert global_inside_cells + global_outside_cells > 0

    with np.load(path, allow_pickle=False) as data:
        assert str(data["defect_metric_status"]) in {"ok", "empty"}

        assert int(data["defect_metric_num_cells"]) == int(data["image_cell_count"])
        assert int(data["defect_metric_num_inside_cells"]) >= 0
        assert int(data["defect_metric_num_outside_cells"]) >= 0

        max_center = data["defect_metric_max_center"]

        assert max_center.shape == (2,)
        assert bool(data["defect_metric_max_inside_defect"]) in {True, False}
        assert np.isfinite(float(data["defect_metric_max_value"]))

        if str(data["defect_metric_status"]) == "ok":
            assert np.all(np.isfinite(max_center))
            assert np.isfinite(float(data["defect_metric_max_distance_to_center"]))
        else:
            assert int(data["image_cell_count"]) == 0

        assert float(data["defect_metric_total_indicator_mass"]) >= 0.0
        assert float(data["defect_metric_inside_indicator_mass"]) >= 0.0
        assert float(data["defect_metric_outside_indicator_mass"]) >= 0.0
        assert float(data["defect_metric_inside_mass_fraction"]) >= 0.0
        assert np.isfinite(float(data["defect_metric_inside_mean"]))
        assert np.isfinite(float(data["defect_metric_outside_mean"]))
