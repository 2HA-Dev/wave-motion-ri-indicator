from mpi4py import MPI
from dolfinx import mesh

from elastic_fwi.solvers.elastic_operator import create_vector_lagrange_space
from elastic_fwi.sources.internal import (
    internal_disk_hann_source,
    internal_gaussian_hann_source,
)


def test_internal_disk_source_assembles_nonzero_vector():
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(
        comm,
        ((0.0, 0.0), (10.0, 10.0)),
        (20, 20),
        cell_type=mesh.CellType.triangle,
    )
    V = create_vector_lagrange_space(msh, degree=1)

    source = internal_disk_hann_source(
        center_x=5.0,
        center_y=5.0,
        radius=0.5,
        direction=(0.0, 1.0),
        amplitude=1.0,
        frequency=1.0,
        t0=0.0,
        num_periods=1.0,
    )

    vector = source.assemble_vector(V, None, t=0.25)
    try:
        assert vector.norm() > 0.0
    finally:
        vector.destroy()

    metadata = source.metadata()
    assert metadata["kind"] == "internal_disk"
    assert metadata["center_x"] == 5.0
    assert metadata["center_y"] == 5.0
    assert metadata["radius"] == 0.5


def test_internal_gaussian_source_assembles_nonzero_vector():
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(
        comm,
        ((0.0, 0.0), (10.0, 10.0)),
        (20, 20),
        cell_type=mesh.CellType.triangle,
    )
    V = create_vector_lagrange_space(msh, degree=1)

    source = internal_gaussian_hann_source(
        center_x=5.0,
        center_y=5.0,
        sigma=0.5,
        direction=(0.0, 1.0),
        amplitude=1.0,
        frequency=1.0,
        t0=0.0,
        num_periods=1.0,
    )

    vector = source.assemble_vector(V, None, t=0.25)
    try:
        assert vector.norm() > 0.0
    finally:
        vector.destroy()

    metadata = source.metadata()
    assert metadata["kind"] == "internal_gaussian"
    assert metadata["center_x"] == 5.0
    assert metadata["center_y"] == 5.0
    assert metadata["sigma"] == 0.5
