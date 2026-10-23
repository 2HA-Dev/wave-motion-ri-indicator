from mpi4py import MPI

from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    run_direct_observation_pipeline,
)


def test_structured_direct_pipeline_auto_sizes_mesh_from_wavelength():
    config = DirectObservationPipelineConfig(
        mesh_kind="structured",
        mesh_auto_from_wavelength=True,
        mesh_target_ppw=10.0,
        mesh_near_target_ppw=20.0,
        nx=4,
        ny=4,
        domain_length_x=10.0,
        domain_length_y=10.0,
        source_kind="internal_gaussian",
        source_center_x=5.0,
        source_center_y=5.0,
        source_sigma=0.25,
        receiver_kind="boundary_segment",
        receiver_side="top",
        receiver_center=5.0,
        receiver_length=2.0,
        material_kind="healthy",
        dt=0.05,
        num_steps=2,
        source_frequency=1.0,
        store_displacement_snapshots=True,
    )

    result = run_direct_observation_pipeline(config=config, comm=MPI.COMM_WORLD)

    assert result.config.mesh_auto_from_wavelength is True
    assert result.config.wavelength_mesh_sizing is not None
    assert result.config.nx == result.config.wavelength_mesh_sizing["nx"]
    assert result.config.ny == result.config.wavelength_mesh_sizing["ny"]
    assert result.mesh_metadata["mesh_auto_from_wavelength"] is True
    assert result.mesh_metadata["effective_nx"] == result.config.nx


def test_gmsh_direct_pipeline_auto_sets_far_and_near_h_from_wavelength():
    config = DirectObservationPipelineConfig(
        mesh_kind="gmsh_refined",
        mesh_auto_from_wavelength=True,
        mesh_target_ppw=10.0,
        mesh_near_target_ppw=20.0,
        mesh_refine_source=True,
        mesh_refine_defect=True,
        mesh_refine_source_radius=0.75,
        mesh_refine_defect_radius=0.75,
        nx=20,
        ny=20,
        domain_length_x=10.0,
        domain_length_y=10.0,
        source_kind="internal_gaussian",
        source_center_x=5.0,
        source_center_y=5.0,
        source_sigma=0.25,
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
        num_steps=2,
        source_frequency=1.0,
        store_displacement_snapshots=True,
    )

    result = run_direct_observation_pipeline(config=config, comm=MPI.COMM_WORLD)

    sizing = result.config.wavelength_mesh_sizing
    assert sizing is not None
    assert result.config.mesh_far_h == sizing["mesh_far_h"]
    assert result.config.mesh_near_h == sizing["mesh_near_h"]
    assert result.mesh_metadata["mesh_auto_from_wavelength"] is True
    assert result.mesh_metadata["wavelength_mesh_sizing"]["target_ppw"] == 10.0
