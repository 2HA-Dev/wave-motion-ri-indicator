from __future__ import annotations

from pathlib import Path

from mpi4py import MPI

from elastic_fwi.diagnostics.export_inverse_time_series_xdmf import (
    export_inverse_time_series_xdmf,
)


def _shared_output_dir(tmp_path: Path, name: str) -> Path:
    comm = MPI.COMM_WORLD
    root = tmp_path / name
    root_str = comm.bcast(str(root), root=0)
    return Path(root_str)


def test_residual_wave_interaction_export_contract(tmp_path: Path):
    output_dir = _shared_output_dir(tmp_path, "residual_wave_interaction_export")

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
        observation_kind="vector_array",
        source_kind="internal_gaussian_array",
        source_array_start_x=3.0,
        source_array_start_y=2.0,
        source_array_end_x=7.0,
        source_array_end_y=2.0,
        source_array_num_elements=3,
        source_sigma=0.25,
        source_direction_x=0.0,
        source_direction_y=1.0,
        receiver_kind="internal_gaussian_array",
        receiver_array_geometry="box_except_gamma1",
        receiver_box_xmin=2.0,
        receiver_box_xmax=8.0,
        receiver_box_ymin=2.0,
        receiver_box_ymax=8.0,
        receiver_box_left_num_elements=3,
        receiver_box_top_num_elements=3,
        receiver_box_right_num_elements=3,
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
    assert manifest["observation_kind"] == "vector_array"

    residual_wave = manifest["exports"]["residual_wave_interaction"]
    assert residual_wave is not None
    assert residual_wave["status"] == "ok"
    assert residual_wave["imaging_mode"] == "residual_wave_interaction"

    assert residual_wave["reflection_residual_wavefield"]["field_name"] == "reflection_residual_wavefield"
    assert residual_wave["transmission_residual_wavefield"]["field_name"] == "transmission_residual_wavefield"

    assert residual_wave["interaction_image"]["scientific_role"] == "residual_wave_interaction_image"
    assert (
        residual_wave["material_parameter_kernels"]["scientific_role"]
        == "residual_wave_interaction_indicator"
    )

    assert manifest["residual_wave_rhs"]["kind"] == "residual_wave_split_rhs"
    assert manifest["residual_wave_rhs"]["reflection"]["keep"] == "reflection"
    assert manifest["residual_wave_rhs"]["transmission"]["keep"] == "transmission"
    assert manifest["residual_wave_rhs"]["transmission"]["active_scale"] == 1.0
