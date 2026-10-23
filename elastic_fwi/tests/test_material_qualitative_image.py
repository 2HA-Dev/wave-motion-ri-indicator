from __future__ import annotations

import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI
from petsc4py import PETSc

from elastic_fwi.diagnostics.check_material_qualitative_image import (
    run_material_qualitative_image_check,
)
from elastic_fwi.inversion.material_imaging import (
    compute_material_qualitative_dof_image,
    owned_scalar_dof_coordinates_2d,
)
from elastic_fwi.solvers.elastic_operator import create_vector_lagrange_space


def _constant_vec(comm: MPI.Comm, size: int, value: float) -> PETSc.Vec:
    vec = PETSc.Vec().createMPI(size, comm=comm)
    vec.set(value)
    vec.assemble()
    return vec


def test_compute_material_qualitative_dof_image_basic_contract():
    comm = MPI.COMM_WORLD

    u1 = _constant_vec(comm, 6, 2.0)
    u2 = _constant_vec(comm, 6, 3.0)
    p1 = _constant_vec(comm, 6, 5.0)
    p2 = _constant_vec(comm, 6, 7.0)

    try:
        image = compute_material_qualitative_dof_image(
            direct_snapshots=(u1, u2),
            equation_adjoint_vectors=(p1, p2),
            comm=comm,
            absolute_value=True,
            normalize=True,
        )

        expected_local_size = int(u1.getLocalSize())

        assert image.num_steps == 2
        assert image.local_size == expected_local_size
        assert image.global_size == 6
        assert image.normalized
        assert image.local_coordinates is None
        assert image.coordinate_dimension == 0
        assert np.all(np.isfinite(image.local_values))
        assert image.global_l2_norm > 0.0
        assert image.global_sum > 0.0
        assert image.global_max == pytest.approx(1.0)

        data = image.as_dict(include_values=True)

        assert data["kind"] == "material_qualitative_dof_image"
        assert data["num_steps"] == 2
        assert data["normalized"] is True
        assert data["has_coordinates"] is False
        assert data["coordinate_dimension"] == 0
        assert "local_values" in data
        assert "local_coordinates" not in data

    finally:
        for vec in (u1, u2, p1, p2):
            vec.destroy()


def test_owned_scalar_dof_coordinates_2d_matches_owned_scalar_dofs():
    comm = MPI.COMM_WORLD

    msh = mesh.create_unit_square(comm, 2, 2)
    V = create_vector_lagrange_space(msh, degree=1)

    coordinates = owned_scalar_dof_coordinates_2d(V)

    block_size = int(getattr(V.dofmap, "index_map_bs", getattr(V.dofmap, "bs", 1)))
    expected_local_size = int(V.dofmap.index_map.size_local) * block_size

    assert coordinates.ndim == 2
    assert coordinates.shape == (expected_local_size, 2)
    assert np.all(np.isfinite(coordinates))

    if coordinates.size:
        assert np.min(coordinates[:, 0]) >= -1.0e-14
        assert np.max(coordinates[:, 0]) <= 1.0 + 1.0e-14
        assert np.min(coordinates[:, 1]) >= -1.0e-14
        assert np.max(coordinates[:, 1]) <= 1.0 + 1.0e-14


def test_compute_material_qualitative_dof_image_with_function_space_coordinates():
    comm = MPI.COMM_WORLD

    msh = mesh.create_unit_square(comm, 2, 2)
    V = create_vector_lagrange_space(msh, degree=1)

    coordinates = owned_scalar_dof_coordinates_2d(V)
    local_size = int(coordinates.shape[0])
    global_size = int(comm.allreduce(local_size, op=MPI.SUM))

    u1 = PETSc.Vec().createMPI((local_size, global_size), comm=comm)
    u2 = PETSc.Vec().createMPI((local_size, global_size), comm=comm)
    p1 = PETSc.Vec().createMPI((local_size, global_size), comm=comm)
    p2 = PETSc.Vec().createMPI((local_size, global_size), comm=comm)

    for vec, value in ((u1, 2.0), (u2, 3.0), (p1, 5.0), (p2, 7.0)):
        vec.set(value)
        vec.assemble()

    try:
        image = compute_material_qualitative_dof_image(
            direct_snapshots=(u1, u2),
            equation_adjoint_vectors=(p1, p2),
            function_space=V,
            comm=comm,
            absolute_value=True,
            normalize=True,
        )

        assert image.num_steps == 2
        assert image.local_size == int(u1.getLocalSize())
        assert image.global_size == global_size
        assert image.normalized
        assert image.coordinate_dimension == 2
        assert image.local_coordinates is not None
        assert image.local_coordinates.shape == (image.local_size, 2)
        assert np.all(np.isfinite(image.local_coordinates))
        assert np.all(np.isfinite(image.local_values))
        assert image.global_l2_norm > 0.0
        assert image.global_sum > 0.0
        assert image.global_max == pytest.approx(1.0)

        data = image.as_dict(include_values=True)

        assert data["kind"] == "material_qualitative_dof_image"
        assert data["has_coordinates"] is True
        assert data["coordinate_dimension"] == 2
        assert data["local_coordinate_count"] == image.local_size
        assert "local_values" in data
        assert "local_coordinates" in data

    finally:
        for vec in (u1, u2, p1, p2):
            vec.destroy()


def test_compute_material_qualitative_dof_image_rejects_empty_series():
    with pytest.raises(ValueError, match="At least one direct snapshot"):
        compute_material_qualitative_dof_image(
            direct_snapshots=(),
            equation_adjoint_vectors=(),
            comm=MPI.COMM_WORLD,
        )


def test_compute_material_qualitative_dof_image_rejects_length_mismatch():
    comm = MPI.COMM_WORLD

    u = _constant_vec(comm, 4, 1.0)
    p1 = _constant_vec(comm, 4, 1.0)
    p2 = _constant_vec(comm, 4, 1.0)

    try:
        with pytest.raises(ValueError, match="same length"):
            compute_material_qualitative_dof_image(
                direct_snapshots=(u,),
                equation_adjoint_vectors=(p1, p2),
                comm=comm,
            )

    finally:
        for vec in (u, p1, p2):
            vec.destroy()


def test_material_qualitative_image_check_reports_ok():
    report = run_material_qualitative_image_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        comm=MPI.COMM_WORLD,
    )

    assert report["kind"] == "material_qualitative_image_check"
    assert report["status"] == "ok"
    assert report["mpi_size"] == MPI.COMM_WORLD.size

    checks = report["checks"]

    assert checks["pipeline_status_ok"]
    assert checks["simulated_snapshots_stored"]
    assert checks["snapshot_count_matches_num_steps"]
    assert checks["adjoint_num_steps_match"]
    assert checks["image_num_steps_match"]
    assert checks["image_local_size_matches_matrix"]
    assert checks["image_global_size_matches_matrix"]
    assert checks["image_values_finite"]
    assert checks["image_global_l2_finite"]
    assert checks["image_global_l2_positive"]
    assert checks["image_global_max_positive"]
    assert checks["image_global_sum_positive"]
    assert checks["observation_residual_nonzero"]
    assert checks["adjoint_nonzero"]
    assert checks["image_has_coordinates"]
    assert checks["image_coordinates_shape_ok"]
    assert checks["image_coordinates_finite"]

    values = report["values"]

    assert np.isfinite(values["image_global_min"])
    assert np.isfinite(values["image_global_max"])
    assert np.isfinite(values["image_global_l2_norm"])
    assert np.isfinite(values["image_global_sum"])
    assert np.isfinite(values["max_observation_residual"])
    assert np.isfinite(values["max_adjoint_norm"])

    assert values["image_global_l2_norm"] > 0.0
    assert values["image_global_max"] > 0.0
    assert values["image_global_sum"] > 0.0
    assert values["max_observation_residual"] > 0.0
    assert values["max_adjoint_norm"] > 0.0
    assert values["image_coordinate_dimension"] == 2

    image = report["image"]

    assert image["kind"] == "material_qualitative_dof_image"
    assert image["num_steps"] == 5
    assert image["normalized"] is True
    assert image["has_coordinates"] is True
    assert image["coordinate_dimension"] == 2
    assert image["local_coordinate_count"] == image["local_size"]
    assert image["global_size"] > 0
    assert image["global_l2_norm"] > 0.0
    assert image["global_max"] == pytest.approx(1.0)


def test_material_qualitative_image_check_without_normalization_reports_ok():
    report = run_material_qualitative_image_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        normalize=False,
        comm=MPI.COMM_WORLD,
    )

    assert report["status"] == "ok"
    assert report["checks"]["image_values_finite"]
    assert report["checks"]["image_global_l2_positive"]
    assert report["checks"]["image_has_coordinates"]
    assert report["checks"]["image_coordinates_shape_ok"]
    assert report["checks"]["image_coordinates_finite"]
    assert report["image"]["normalized"] is False
    assert report["image"]["has_coordinates"] is True
    assert report["image"]["coordinate_dimension"] == 2
    assert np.isfinite(report["values"]["image_global_l2_norm"])
    assert report["values"]["image_global_l2_norm"] > 0.0
