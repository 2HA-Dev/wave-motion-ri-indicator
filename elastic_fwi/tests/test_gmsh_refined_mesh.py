import importlib.util

import pytest
from mpi4py import MPI

from elastic_fwi.meshing import (
    RefinementDisk,
    create_gmsh_refined_rectangle_mesh,
)
from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    run_direct_observation_pipeline,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("gmsh") is None,
    reason="gmsh Python module is not installed",
)


def test_gmsh_refined_rectangle_mesh_is_created():
    comm = MPI.COMM_WORLD

    msh = create_gmsh_refined_rectangle_mesh(
        comm=comm,
        domain_length_x=10.0,
        domain_length_y=10.0,
        far_h=0.5,
        refinement_disks=(
            RefinementDisk(
                name="source",
                center_x=5.0,
                center_y=5.0,
                radius=0.75,
                target_h=0.15,
            ),
            RefinementDisk(
                name="defect",
                center_x=2.0,
                center_y=2.0,
                radius=0.75,
                target_h=0.15,
            ),
        ),
    )

    assert msh.topology.dim == 2
    assert msh.geometry.dim == 2
    assert msh.geometry.x.shape[0] > 0


def test_direct_pipeline_accepts_gmsh_refined_mesh_with_internal_source_and_defect():
    comm = MPI.COMM_WORLD

    config = DirectObservationPipelineConfig(
        mesh_kind="gmsh_refined",
        mesh_far_h=0.5,
        mesh_near_h=0.15,
        mesh_refine_source=True,
        mesh_refine_defect=True,
        mesh_refine_source_radius=0.75,
        mesh_refine_defect_radius=0.75,
        nx=20,
        ny=20,
        degree=1,
        domain_length_x=10.0,
        domain_length_y=10.0,
        source_kind="internal_gaussian",
        source_center_x=5.0,
        source_center_y=5.0,
        source_sigma=0.25,
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

    result = run_direct_observation_pipeline(config=config, comm=comm)

    assert result.has_displacement_snapshots
    assert result.mesh_metadata["mesh_kind"] == "gmsh_refined"
    assert len(result.mesh_metadata["refinement_disks"]) == 2
    assert result.source_metadata["geometry"]["kind"] == "internal_gaussian"
