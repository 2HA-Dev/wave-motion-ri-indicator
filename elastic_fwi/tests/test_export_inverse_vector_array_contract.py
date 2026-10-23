from __future__ import annotations

import shutil
from pathlib import Path

from mpi4py import MPI

from elastic_fwi.diagnostics.export_inverse_time_series_xdmf import (
    export_inverse_time_series_xdmf,
)


def _shared_output_dir(tmp_path: Path, name: str) -> Path:
    """Return the same filesystem path on all MPI ranks.

    pytest's tmp_path can differ between ranks when pytest is launched under
    mpirun. XDMF/HDF5 collective output requires one shared path, so rank 0
    chooses the directory and broadcasts it.
    """
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        output_dir = tmp_path / name
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path_str = str(output_dir)
    else:
        path_str = None

    path_str = comm.bcast(path_str, root=0)
    output_dir = Path(path_str)

    comm.Barrier()
    return output_dir


def test_export_inverse_time_series_vector_array_contract(tmp_path: Path):
    output_dir = _shared_output_dir(tmp_path, "inverse_vector_array")

    manifest = export_inverse_time_series_xdmf(
        output_dir=output_dir,
        mesh_kind="structured",
        domain_length_x=10.0,
        domain_length_y=10.0,
        absorbing_layer=True,
        absorbing_layer_width=1.0,
        absorbing_layer_strength=6.0,
        absorbing_layer_polynomial_order=2.0,
        absorbing_layer_min_cells=4,
        auto_place_probes_inside_absorbing_layer=True,
        observation_kind="vector_array",
        source_kind="internal_gaussian_array",
        source_array_start_x=3.0,
        source_array_end_x=7.0,
        source_array_num_elements=3,
        source_sigma=0.25,
        source_direction_x=0.0,
        source_direction_y=1.0,
        receiver_kind="internal_gaussian_array",
        receiver_array_start_x=3.0,
        receiver_array_end_x=7.0,
        receiver_array_num_elements=3,
        receiver_sigma=0.25,
        simulated_material_kind="healthy",
        observed_material_kind="localized_disk_defect",
        defect_center_x=5.0,
        defect_center_y=5.0,
        defect_radius=0.5,
        contrast_factor=0.7,
        nx=8,
        ny=8,
        dt=0.1,
        num_steps=3,
        source_frequency=1.0,
        every_k=1,
        export_material_fields=False,
        export_mesh=False,
        export_sampling_conditions=False,
        comm=MPI.COMM_WORLD,
    )

    assert manifest["status"] == "ok"
    assert manifest["common_mesh_direct_pair"] is True
    assert manifest["direct_mesh_contract"]["status"] == "ok"
    assert manifest["direct_mesh_contract"]["policy"] == "shared_observed_refined"
    assert manifest["direct_mesh_contract"]["mesh_created_once"] is True
    assert manifest["direct_mesh_contract"]["shared_mesh_object"] is True
    assert manifest["direct_mesh_contract"]["shared_function_space"] is True
    assert manifest["direct_mesh_contract"]["same_mpi_partition"] is True
    fingerprint = manifest["direct_mesh_contract"]["partition_fingerprint"]
    assert fingerprint["global_owned_vertices"] > 0
    assert fingerprint["global_owned_cells"] > 0
    assert len(fingerprint["rank_ordered_partition_sha256"]) == 64
    assert (
        manifest["direct_mesh_contract"][
            "all_imaging_wave_problems_use_shared_space"
        ]
        is True
    )
    assert manifest["observation_kind"] == "vector_array"
    assert manifest["resolved_receiver_sigma"] == 0.25
    assert manifest["vector_array_comparison"] is not None
    assert manifest["adjoint_rhs"]["kind"] == "vector_array_full_adjoint_rhs_result"

    exports = manifest["exports"]
    assert exports["vector_array"] is not None
    assert exports["vector_array"]["status"] == "ok"

    if MPI.COMM_WORLD.rank == 0:
        vector_export = exports["vector_array"]

        assert Path(vector_export["manifest"]).exists()
        assert Path(vector_export["residual_csv"]).exists()

        assert Path(vector_export["signals"]["simulated_reflection"]).exists()
        assert Path(vector_export["signals"]["observed_reflection"]).exists()
        assert Path(vector_export["signals"]["simulated_transmission"]).exists()
        assert Path(vector_export["signals"]["observed_transmission"]).exists()

        assert Path(vector_export["residual_npz"]["reflection"]).exists()
        assert Path(vector_export["residual_npz"]["transmission"]).exists()

        assert "residual_figures" in vector_export
        assert Path(vector_export["residual_figures"]["png"]).exists()
        assert Path(vector_export["residual_figures"]["pdf"]).exists()
        assert Path(vector_export["residual_figures"]["summary"]).exists()

    MPI.COMM_WORLD.Barrier()


def test_export_inverse_time_series_scalar_energy_compatibility_contract(tmp_path: Path):
    output_dir = _shared_output_dir(tmp_path, "inverse_scalar_energy")

    manifest = export_inverse_time_series_xdmf(
        output_dir=output_dir,
        mesh_kind="structured",
        domain_length_x=10.0,
        domain_length_y=10.0,
        observation_kind="scalar_energy",
        source_kind="boundary_segment",
        receiver_kind="boundary_segment",
        simulated_material_kind="healthy",
        observed_material_kind="localized_disk_defect",
        defect_center_x=5.0,
        defect_center_y=5.0,
        defect_radius=0.5,
        contrast_factor=0.7,
        nx=6,
        ny=6,
        dt=0.1,
        num_steps=2,
        source_frequency=1.0,
        every_k=1,
        export_material_fields=False,
        export_mesh=False,
        export_sampling_conditions=False,
        comm=MPI.COMM_WORLD,
    )

    assert manifest["status"] == "ok"
    assert manifest["common_mesh_direct_pair"] is True
    assert manifest["direct_mesh_contract"]["status"] == "ok"
    assert manifest["direct_mesh_contract"]["shared_function_space"] is True
    assert manifest["observation_kind"] == "scalar_energy"
    assert manifest["vector_array_comparison"] is None
    assert manifest["exports"]["vector_array"] is None
    assert manifest["adjoint_rhs"]["kind"] == "full_adjoint_observation_rhs_result"

    MPI.COMM_WORLD.Barrier()
