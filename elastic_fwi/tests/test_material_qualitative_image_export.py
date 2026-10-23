from __future__ import annotations

from pathlib import Path

import numpy as np
from mpi4py import MPI

from elastic_fwi.diagnostics.export_material_qualitative_image import (
    export_material_qualitative_image_npz,
)


def _expected_output_path(base: Path, comm: MPI.Comm) -> Path:
    if comm.size == 1:
        return base

    return base.with_name(
        f"{base.stem}_rank{comm.rank:04d}of{comm.size:04d}{base.suffix}"
    )


def test_material_qualitative_image_export_npz_is_pickle_free(tmp_path: Path):
    comm = MPI.COMM_WORLD
    output = tmp_path / "material_image.npz"

    export = export_material_qualitative_image_npz(
        output=output,
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        comm=comm,
    )

    expected_path = _expected_output_path(output, comm)

    assert export["kind"] == "material_qualitative_image_export"
    assert export["status"] == "ok"
    assert export["mpi_rank"] == comm.rank
    assert export["mpi_size"] == comm.size
    assert Path(export["path"]) == expected_path
    assert expected_path.exists()

    with np.load(expected_path, allow_pickle=False) as data:
        assert str(data["kind"]) == "material_qualitative_image_export"
        assert str(data["status"]) == "ok"

        assert int(data["mpi_rank"]) == comm.rank
        assert int(data["mpi_size"]) == comm.size

        assert int(data["nx"]) == 4
        assert int(data["ny"]) == 4
        assert int(data["degree"]) == 1
        assert int(data["num_steps"]) == 5

        assert float(data["theta"]) == 0.25
        assert float(data["dt"]) == 0.1

        assert str(data["simulated_material_kind"]) == "healthy"
        assert str(data["observed_material_kind"]) == "localized_disk_defect"

        local_values = data["image_local_values"]
        local_coordinates = data["image_local_coordinates"]

        assert local_values.ndim == 1
        assert local_values.size == int(data["image_local_size"])
        assert np.all(np.isfinite(local_values))

        assert local_coordinates.ndim == 2
        assert local_coordinates.shape[0] == int(data["image_local_size"])
        assert local_coordinates.shape[1] == 2
        assert np.all(np.isfinite(local_coordinates))

        assert int(data["image_coordinate_dim"]) == 2
        assert int(data["image_global_size"]) > 0
        assert float(data["image_global_l2_norm"]) > 0.0
        assert float(data["image_global_max"]) > 0.0
        assert float(data["image_global_sum"]) > 0.0

        assert np.isfinite(float(data["max_observation_residual"]))
        assert np.isfinite(float(data["max_adjoint_norm"]))
        assert float(data["max_observation_residual"]) > 0.0
        assert float(data["max_adjoint_norm"]) > 0.0


def test_material_qualitative_image_export_without_normalization(tmp_path: Path):
    comm = MPI.COMM_WORLD
    output = tmp_path / "material_image_raw.npz"

    export = export_material_qualitative_image_npz(
        output=output,
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        normalize=False,
        comm=comm,
    )

    expected_path = _expected_output_path(output, comm)

    assert export["status"] == "ok"
    assert expected_path.exists()
    assert export["normalized"] is False

    with np.load(expected_path, allow_pickle=False) as data:
        assert str(data["kind"]) == "material_qualitative_image_export"
        assert str(data["status"]) == "ok"
        assert bool(data["normalized"]) is False
        assert data["image_local_values"].ndim == 1
        assert np.all(np.isfinite(data["image_local_values"]))
        assert float(data["image_global_l2_norm"]) > 0.0
