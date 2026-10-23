from mpi4py import MPI

from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    run_direct_observation_pipeline,
)


def test_direct_pipeline_accepts_internal_disk_source():
    comm = MPI.COMM_WORLD

    config = DirectObservationPipelineConfig(
        nx=20,
        ny=20,
        degree=1,
        domain_length_x=10.0,
        domain_length_y=10.0,
        source_kind="internal_disk",
        source_center_x=5.0,
        source_center_y=5.0,
        source_radius=0.5,
        source_direction_x=0.0,
        source_direction_y=1.0,
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

    assert result.has_displacement_snapshots
    assert result.num_displacement_snapshots == 3
    assert result.source_metadata["geometry"]["kind"] == "internal_disk"
    assert result.source_metadata["geometry"]["center_x"] == 5.0
    assert result.source_metadata["geometry"]["center_y"] == 5.0
    assert result.source_metadata["geometry"]["radius"] == 0.5


def test_direct_pipeline_accepts_internal_gaussian_source():
    comm = MPI.COMM_WORLD

    config = DirectObservationPipelineConfig(
        nx=20,
        ny=20,
        degree=1,
        domain_length_x=10.0,
        domain_length_y=10.0,
        source_kind="internal_gaussian",
        source_center_x=5.0,
        source_center_y=5.0,
        source_sigma=0.5,
        source_direction_x=0.0,
        source_direction_y=1.0,
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

    assert result.has_displacement_snapshots
    assert result.num_displacement_snapshots == 3
    assert result.source_metadata["geometry"]["kind"] == "internal_gaussian"
    assert result.source_metadata["geometry"]["center_x"] == 5.0
    assert result.source_metadata["geometry"]["center_y"] == 5.0
    assert result.source_metadata["geometry"]["sigma"] == 0.5
