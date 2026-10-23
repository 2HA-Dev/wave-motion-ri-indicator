from mpi4py import MPI
from dolfinx import mesh

from elastic_fwi.geometry.boundary_tags import (
    build_boundary_meshtags,
    boundary_tag_counts_global,
)
from elastic_fwi.sources.boundary_arrays import (
    BoundaryPointArraySource,
    BoundarySegmentSource,
    lateral_boundary_specs,
)


def test_boundary_segment_source_specs_on_bottom():
    source = BoundarySegmentSource(
        name="gamma1_source",
        side="bottom",
        center=5.0,
        length=2.0,
        tag=1,
    )

    specs = source.boundary_specs(domain_length_x=10.0, domain_length_y=10.0)

    assert len(specs) == 1
    assert specs[0].tag == 1
    assert specs[0].fixed_axis == 1
    assert specs[0].fixed_value == 0.0
    assert specs[0].interval_min == 4.0
    assert specs[0].interval_max == 6.0

    metadata = source.metadata()
    assert metadata["kind"] == "boundary_segment"
    assert metadata["center"] == 5.0
    assert metadata["length"] == 2.0


def test_boundary_point_array_source_specs_share_physical_tag():
    source = BoundaryPointArraySource(
        name="gamma1_array",
        side="bottom",
        start=4.0,
        end=6.0,
        num_elements=5,
        element_length=0.1,
        tag=1,
    )

    specs = source.boundary_specs(domain_length_x=10.0, domain_length_y=10.0)

    assert len(specs) == 5
    assert {spec.tag for spec in specs} == {1}
    assert len({spec.name for spec in specs}) == 5
    assert source.centers == (4.0, 4.5, 5.0, 5.5, 6.0)

    metadata = source.metadata()
    assert metadata["kind"] == "boundary_point_array"
    assert metadata["num_elements"] == 5
    assert metadata["centers"] == [4.0, 4.5, 5.0, 5.5, 6.0]


def test_boundary_point_array_meshtags_mpi():
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(
        comm,
        ((0.0, 0.0), (10.0, 10.0)),
        (40, 40),
        cell_type=mesh.CellType.triangle,
    )

    source = BoundaryPointArraySource(
        name="gamma1_array",
        side="bottom",
        start=4.0,
        end=6.0,
        num_elements=5,
        element_length=0.5,
        tag=1,
    )

    receiver = BoundaryPointArraySource(
        name="gamma2_array",
        side="top",
        start=4.0,
        end=6.0,
        num_elements=5,
        element_length=0.5,
        tag=2,
    )

    specs = (
        *source.boundary_specs(domain_length_x=10.0, domain_length_y=10.0),
        *receiver.boundary_specs(domain_length_x=10.0, domain_length_y=10.0),
        *lateral_boundary_specs(domain_length_x=10.0, domain_length_y=10.0),
    )

    facet_tags = build_boundary_meshtags(msh, specs)
    counts = boundary_tag_counts_global(facet_tags, (1, 2, 3, 4), comm)

    assert counts[1] > 0
    assert counts[2] > 0
    assert counts[3] > 0
    assert counts[4] > 0
