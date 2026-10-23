from __future__ import annotations

from pathlib import Path

import numpy as np
from mpi4py import MPI

from elastic_fwi.diagnostics.export_material_qualitative_image import (
    export_material_qualitative_image_npz,
)


_NUMERIC_ARTIFACT_KEYS = (
    "image_local_values",
    "image_local_coordinates",
    "image_global_min",
    "image_global_max",
    "image_global_l2_norm",
    "image_global_sum",
    "image_node_coordinates",
    "image_node_values",
    "image_node_count",
    "image_node_local_min",
    "image_node_local_max",
    "image_node_local_sum",
    "image_node_local_l2_norm",
    "image_cell_centers",
    "image_cell_values",
    "image_cell_count",
    "image_cell_local_min",
    "image_cell_local_max",
    "image_cell_local_sum",
    "image_cell_local_l2_norm",
    "defect_metric_max_value",
    "defect_metric_max_center",
    "defect_metric_max_distance_to_center",
    "defect_metric_total_indicator_mass",
    "defect_metric_inside_indicator_mass",
    "defect_metric_outside_indicator_mass",
    "defect_metric_inside_mass_fraction",
    "defect_metric_inside_mean",
    "defect_metric_outside_mean",
    "defect_metric_inside_outside_mean_ratio",
    "max_observation_residual",
    "max_adjoint_norm",
)


def _relative_array_error(new: np.ndarray, old: np.ndarray) -> float:
    new_values = np.asarray(new, dtype=np.float64)
    old_values = np.asarray(old, dtype=np.float64)

    if new_values.shape != old_values.shape:
        raise AssertionError(
            f"shape mismatch: old={old_values.shape}, new={new_values.shape}"
        )

    if old_values.size == 0 and new_values.size == 0:
        return 0.0

    diff = np.max(np.abs(new_values - old_values))
    reference = max(float(np.max(np.abs(old_values))), 1.0)
    return float(diff / reference)


def test_material_image_npz_artifact_reuse_matches_old_path(tmp_path: Path):
    comm = MPI.COMM_WORLD

    old = export_material_qualitative_image_npz(
        output=tmp_path / "old_path" / "material_image.npz",
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        ksp_type="cg",
        pc_type="jacobi",
        ksp_rtol=1.0e-10,
        ksp_atol=1.0e-12,
        ksp_max_it=1000,
        normalize=True,
        reuse_direct_effective_solver=False,
        reuse_adjoint_effective_solver=False,
        comm=comm,
    )

    new = export_material_qualitative_image_npz(
        output=tmp_path / "reuse_ksp" / "material_image.npz",
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        ksp_type="cg",
        pc_type="jacobi",
        ksp_rtol=1.0e-10,
        ksp_atol=1.0e-12,
        ksp_max_it=1000,
        normalize=True,
        reuse_direct_effective_solver=True,
        reuse_adjoint_effective_solver=True,
        comm=comm,
    )

    assert old["status"] == "ok"
    assert new["status"] == "ok"

    with np.load(old["path"], allow_pickle=False) as old_data, np.load(
        new["path"],
        allow_pickle=False,
    ) as new_data:
        assert bool(old_data["reuse_direct_effective_solver"]) is False
        assert bool(old_data["reuse_adjoint_effective_solver"]) is False
        assert bool(new_data["reuse_direct_effective_solver"]) is True
        assert bool(new_data["reuse_adjoint_effective_solver"]) is True

        max_rel = 0.0
        for key in _NUMERIC_ARTIFACT_KEYS:
            assert key in old_data.files
            assert key in new_data.files
            assert np.all(np.isfinite(old_data[key]))
            assert np.all(np.isfinite(new_data[key]))
            max_rel = max(
                max_rel,
                _relative_array_error(new_data[key], old_data[key]),
            )

        assert max_rel <= 1.0e-12
