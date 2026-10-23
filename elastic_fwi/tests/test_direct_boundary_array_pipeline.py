from mpi4py import MPI

from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    run_direct_observation_pipeline,
)


def test_direct_pipeline_accepts_boundary_point_array_source_and_receiver():
    comm = MPI.COMM_WORLD

    config = DirectObservationPipelineConfig(
        nx=40,
        ny=40,
        degree=1,
        domain_length_x=10.0,
        domain_length_y=10.0,
        source_kind="boundary_point_array",
        source_side="bottom",
        source_array_start=4.0,
        source_array_end=6.0,
        source_array_num_elements=5,
        source_array_element_length=0.5,
        receiver_kind="boundary_point_array",
        receiver_side="top",
        receiver_array_start=4.0,
        receiver_array_end=6.0,
        receiver_array_num_elements=5,
        receiver_array_element_length=0.5,
        material_kind="localized_disk_defect",
        defect_center_x=5.0,
        defect_center_y=5.0,
        defect_radius=1.0,
        contrast_factor=0.8,
        dt=0.05,
        num_steps=3,
        source_frequency=1.0,
        store_displacement_snapshots=True,
    )

    result = run_direct_observation_pipeline(config=config, comm=comm)

    assert result.has_displacement_snapshots
    assert result.num_displacement_snapshots == 3

    assert result.source_metadata["geometry"]["kind"] == "boundary_point_array"
    assert result.source_metadata["geometry"]["num_elements"] == 5
    assert result.source_metadata["geometry"]["centers"] == [4.0, 4.5, 5.0, 5.5, 6.0]

    assert result.observation_metadata["transmission"]["geometry"]["kind"] == "boundary_point_array"
    assert result.observation_metadata["transmission"]["geometry"]["num_elements"] == 5


def test_direct_pipeline_accepts_boundary_segment_source_and_receiver():
    comm = MPI.COMM_WORLD

    config = DirectObservationPipelineConfig(
        nx=20,
        ny=20,
        degree=1,
        domain_length_x=10.0,
        domain_length_y=10.0,
        source_kind="boundary_segment",
        source_side="bottom",
        source_center=5.0,
        source_length=2.0,
        receiver_kind="boundary_segment",
        receiver_side="top",
        receiver_center=5.0,
        receiver_length=2.0,
        material_kind="healthy",
        dt=0.05,
        num_steps=3,
        source_frequency=1.0,
        store_displacement_snapshots=True,
    )

    result = run_direct_observation_pipeline(config=config, comm=comm)

    assert result.source_metadata["geometry"]["kind"] == "boundary_segment"
    assert result.source_metadata["geometry"]["center"] == 5.0
    assert result.source_metadata["geometry"]["length"] == 2.0

    assert result.observation_metadata["transmission"]["geometry"]["kind"] == "boundary_segment"
    assert result.observation_metadata["transmission"]["geometry"]["center"] == 5.0
    assert result.observation_metadata["transmission"]["geometry"]["length"] == 2.0
