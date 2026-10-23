from mpi4py import MPI
from dolfinx import mesh

from elastic_fwi.sources.internal import internal_gaussian_array_hann_source
from elastic_fwi.solvers.elastic_operator import create_vector_lagrange_space


def test_internal_gaussian_array_metadata_and_assembly():
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [10.0, 10.0]],
        [8, 8],
        cell_type=mesh.CellType.triangle,
    )
    V = create_vector_lagrange_space(msh, degree=1)

    source = internal_gaussian_array_hann_source(
        start_x=4.0,
        start_y=5.0,
        end_x=6.0,
        end_y=5.0,
        num_elements=5,
        sigma=0.10,
        direction=(0.0, 1.0),
        amplitude=1.0,
        frequency=1.0,
    )

    metadata = source.metadata()

    assert metadata["kind"] == "internal_gaussian_array"
    assert metadata["num_elements"] == 5
    assert metadata["start_x"] == 4.0
    assert metadata["start_y"] == 5.0
    assert metadata["end_x"] == 6.0
    assert metadata["end_y"] == 5.0
    assert metadata["center_x"] == 5.0
    assert metadata["center_y"] == 5.0
    assert len(metadata["points"]) == 5

    vector = source.assemble_vector(V, None, t=0.25)
    assert vector.getSize() > 0
