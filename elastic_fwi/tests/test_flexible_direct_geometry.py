from mpi4py import MPI

from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    run_direct_observation_pipeline,
)


def test_direct_pipeline_accepts_rectangular_domain_segments_and_defect():
    comm = MPI.COMM_WORLD

    config = DirectObservationPipelineConfig(
        nx=10,
        ny=10,
        degree=1,
        domain_length_x=10.0,
        domain_length_y=10.0,
        source_segment_xmin=4.0,
        source_segment_xmax=6.0,
        receiver_segment_xmin=4.0,
        receiver_segment_xmax=6.0,
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

    assert result.mesh_metadata["kind"] == "rectangle"
    assert result.mesh_metadata["domain_length_x"] == 10.0
    assert result.mesh_metadata["domain_length_y"] == 10.0

    assert result.source_metadata["segment_xmin"] == 4.0
    assert result.source_metadata["segment_xmax"] == 6.0

    assert result.observation_metadata["transmission"]["segment_xmin"] == 4.0
    assert result.observation_metadata["transmission"]["segment_xmax"] == 6.0
    assert result.observation_metadata["transmission"]["segment_y"] == 10.0

    assert result.material_metadata["kind"] == "localized_disk_defect"
    assert result.material_metadata["center_x"] == 5.0
    assert result.material_metadata["center_y"] == 5.0
    assert result.material_metadata["radius"] == 1.0

    assert result.has_displacement_snapshots
    assert result.num_displacement_snapshots == 3
