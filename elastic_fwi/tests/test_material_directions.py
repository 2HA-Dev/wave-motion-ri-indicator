import json

import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh

pytest.importorskip("dolfinx")

from elastic_fwi.inversion.material_directions import (
    HomogeneousMaterialDirection,
    assemble_homogeneous_material_direction_matrices,
    homogeneous_material_direction_fields,
)
from elastic_fwi.solvers.elastic_operator import create_vector_lagrange_space


def test_homogeneous_material_direction_fields_are_constant_dg0():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = create_vector_lagrange_space(msh, degree=1)

    direction = HomogeneousMaterialDirection(
        rho=1.0,
        C11=2.0,
        C22=3.0,
        C12=4.0,
        C66=5.0,
    )

    fields = homogeneous_material_direction_fields(V, direction)

    assert set(fields) == {"rho", "C11", "C22", "C12", "C66"}

    expected = direction.as_dict()

    for key, function in fields.items():
        local_values = function.x.array

        assert local_values.size >= 0

        if local_values.size > 0:
            assert np.allclose(local_values, expected[key])


def test_homogeneous_material_direction_matrices_have_valid_sizes():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = create_vector_lagrange_space(msh, degree=1)

    direction = HomogeneousMaterialDirection(
        rho=1.0,
        C11=0.1,
        C22=0.2,
        C12=0.03,
        C66=0.04,
    )

    perturbation = assemble_homogeneous_material_direction_matrices(V, direction)

    try:
        report = perturbation.as_dict()

        assert report["kind"] == "material_perturbation_matrices"
        assert report["metadata"]["kind"] == "homogeneous_material_direction"

        assert report["mass"]["rows"] == report["mass"]["cols"]
        assert report["stiffness"]["rows"] == report["stiffness"]["cols"]
        assert report["mass"]["rows"] == report["stiffness"]["rows"]
        assert report["mass"]["rows"] > 0

        assert perturbation.mass.norm() > 0.0
        assert perturbation.stiffness.norm() > 0.0

        payload = json.dumps(report, sort_keys=True)

        assert '"material_perturbation_matrices"' in payload
        assert '"homogeneous_material_direction"' in payload

    finally:
        perturbation.destroy()


def test_zero_homogeneous_material_direction_gives_zero_matrices():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = create_vector_lagrange_space(msh, degree=1)

    direction = HomogeneousMaterialDirection()

    perturbation = assemble_homogeneous_material_direction_matrices(V, direction)

    try:
        assert np.isclose(perturbation.mass.norm(), 0.0)
        assert np.isclose(perturbation.stiffness.norm(), 0.0)

    finally:
        perturbation.destroy()


def test_signed_homogeneous_material_direction_is_allowed():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = create_vector_lagrange_space(msh, degree=1)

    direction = HomogeneousMaterialDirection(
        rho=-1.0,
        C11=-0.1,
        C22=0.2,
        C12=-0.03,
        C66=0.04,
    )

    perturbation = assemble_homogeneous_material_direction_matrices(V, direction)

    try:
        report = perturbation.as_dict()

        assert report["mass"]["rows"] > 0
        assert report["stiffness"]["rows"] > 0

    finally:
        perturbation.destroy()
