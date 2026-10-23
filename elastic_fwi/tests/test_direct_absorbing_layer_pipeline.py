from mpi4py import MPI

from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    run_direct_observation_pipeline,
)


def test_direct_pipeline_accepts_absorbing_layer():
    config = DirectObservationPipelineConfig(
        mesh_kind="gmsh_refined",
        mesh_auto_from_wavelength=True,
        mesh_target_ppw=10.0,
        mesh_near_target_ppw=20.0,
        absorbing_layer=True,
        absorbing_layer_width_wavelengths=1.5,
        absorbing_layer_min_cells=6,
        absorbing_layer_strength=3.0,
        absorbing_layer_polynomial_order=2.0,
        mesh_refine_source=True,
        mesh_refine_defect=True,
        mesh_refine_source_radius=1.25,
        mesh_refine_defect_radius=0.75,
        nx=20,
        ny=20,
        domain_length_x=10.0,
        domain_length_y=10.0,
        source_kind="internal_gaussian_array",
        source_array_start_x=4.0,
        source_array_start_y=5.0,
        source_array_end_x=6.0,
        source_array_end_y=5.0,
        source_array_num_elements=5,
        source_sigma=0.10,
        source_direction_x=0.0,
        source_direction_y=1.0,
        receiver_kind="boundary_segment",
        receiver_side="top",
        receiver_center=5.0,
        receiver_length=2.0,
        material_kind="localized_disk_defect",
        defect_center_x=2.0,
        defect_center_y=2.0,
        defect_radius=0.25,
        contrast_factor=0.8,
        dt=0.05,
        num_steps=3,
        source_frequency=1.0,
        store_displacement_snapshots=True,
    )

    result = run_direct_observation_pipeline(config=config, comm=MPI.COMM_WORLD)

    absorbing = result.mesh_metadata["absorbing_layer"]

    assert result.has_displacement_snapshots
    assert absorbing["enabled"] is True
    assert absorbing["kind"] == "absorbing_sponge_layer"
    assert absorbing["plan"]["width"] > 0.0
    assert absorbing["plan"]["sigma_max"] > 0.0
    assert absorbing["field_summary"]["global_max"] > 0.0
    assert absorbing["field_summary"]["global_nonzero_cells"] > 0
