import numpy as np
import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI
from dolfinx import mesh

from elastic_fwi.materials.dg0_fields import homogeneous_dg0_material_functions
from elastic_fwi.materials.orthotropic import LEGACY_HEALTHY_MATERIAL
from elastic_fwi.solvers.elastic_operator import (
    ElasticOperatorConfig,
    assemble_elastic_operator_matrices,
    create_vector_lagrange_space,
    matrix_action_norm,
    matrix_frobenius_norm,
    matrix_global_size,
    validate_material_fields,
)


def _create_mesh():
    return mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)


def test_elastic_operator_config_validation():
    config = ElasticOperatorConfig(theta=0.25, dt=0.1)

    assert config.theta == pytest.approx(0.25)
    assert config.dt == pytest.approx(0.1)

    with pytest.raises(ValueError):
        ElasticOperatorConfig(theta=-1.0, dt=0.1)

    with pytest.raises(ValueError):
        ElasticOperatorConfig(theta=0.25, dt=0.0)


def test_create_vector_lagrange_space_size():
    msh = _create_mesh()
    V = create_vector_lagrange_space(msh, degree=1)

    global_size = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    assert global_size > 0
    assert V.dofmap.index_map_bs == msh.geometry.dim


def test_validate_material_fields_accepts_homogeneous_dg0_fields():
    msh = _create_mesh()
    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)

    validate_material_fields(materials)


def test_validate_material_fields_rejects_missing_field():
    msh = _create_mesh()
    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)

    materials.pop("C66")

    with pytest.raises(ValueError):
        validate_material_fields(materials)


def test_validate_material_fields_rejects_non_positive_density():
    msh = _create_mesh()
    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)

    rho = materials["rho"]
    rho.x.array[:] = 0.0
    rho.x.scatter_forward()

    with pytest.raises(ValueError):
        validate_material_fields(materials)


def test_validate_material_fields_accepts_reference_stiffness():
    msh = _create_mesh()
    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)

    validate_material_fields(materials)


def test_validate_material_fields_rejects_non_positive_C11():
    msh = _create_mesh()
    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)

    C11 = materials["C11"]
    C11.x.array[:] = 0.0
    C11.x.scatter_forward()

    with pytest.raises(ValueError):
        validate_material_fields(materials)


def test_validate_material_fields_rejects_non_positive_C66():
    msh = _create_mesh()
    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)

    C66 = materials["C66"]
    C66.x.array[:] = 0.0
    C66.x.scatter_forward()

    with pytest.raises(ValueError):
        validate_material_fields(materials)


def test_validate_material_fields_rejects_non_positive_definite_voigt_matrix():
    msh = _create_mesh()
    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)

    # C11 = 16.24 > 0 and C66 = 4.67 > 0 individually, but C12 is set large
    # enough that C11*C22 - C12**2 <= 0: the Sylvester minor fails while the
    # two single-coefficient checks alone would not catch it.
    C12 = materials["C12"]
    C12.x.array[:] = 20.0
    C12.x.scatter_forward()

    with pytest.raises(ValueError):
        validate_material_fields(materials)


def test_assemble_elastic_operator_matrices_have_consistent_sizes_and_norms():
    msh = _create_mesh()
    V = create_vector_lagrange_space(msh, degree=1)
    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)

    config = ElasticOperatorConfig(theta=0.25, dt=0.1)
    matrices = assemble_elastic_operator_matrices(V, materials, config)

    try:
        expected_size = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

        for matrix in (matrices.mass, matrices.stiffness, matrices.effective):
            assert matrix_global_size(matrix) == (expected_size, expected_size)

            norm = matrix_frobenius_norm(matrix)
            assert np.isfinite(norm)
            assert norm > 0.0

        assert matrices.config.theta == pytest.approx(0.25)
        assert matrices.config.dt == pytest.approx(0.1)
    finally:
        matrices.destroy()


def test_mass_and_effective_matrix_actions_are_finite():
    msh = _create_mesh()
    V = create_vector_lagrange_space(msh, degree=1)
    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)

    matrices = assemble_elastic_operator_matrices(V, materials)

    try:
        mass_action = matrix_action_norm(matrices.mass, value=1.0)
        effective_action = matrix_action_norm(matrices.effective, value=1.0)

        assert np.isfinite(mass_action)
        assert np.isfinite(effective_action)
        assert mass_action > 0.0
        assert effective_action > 0.0
    finally:
        matrices.destroy()
