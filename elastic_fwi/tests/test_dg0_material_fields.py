import numpy as np
import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI
from dolfinx import mesh

from elastic_fwi.geometry.cell_centers import (
    local_cell_indices,
    local_cell_midpoints,
    local_cell_midpoints_2d,
)
from elastic_fwi.materials.defects import CircularDefectSpec, DefectScenario
from elastic_fwi.materials.dg0_fields import (
    defective_dg0_material_functions,
    homogeneous_dg0_material_functions,
    material_indices_and_dg0_functions,
)
from elastic_fwi.materials.orthotropic import LEGACY_HEALTHY_MATERIAL


def _create_unit_square_mesh():
    return mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)


def _owned_dof_values(function):
    V = function.function_space
    owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    return function.x.array[:owned].copy()


def test_local_cell_indices_and_midpoints_have_consistent_shapes():
    msh = _create_unit_square_mesh()

    cells = local_cell_indices(msh, include_ghosts=True)
    midpoints = local_cell_midpoints(msh, include_ghosts=True)
    midpoints_2d = local_cell_midpoints_2d(msh, include_ghosts=True)

    assert cells.ndim == 1
    assert midpoints.ndim == 2
    assert midpoints_2d.ndim == 2
    assert midpoints.shape[0] == cells.shape[0]
    assert midpoints_2d.shape[0] == cells.shape[0]
    assert midpoints_2d.shape[1] == 2
    assert np.all(np.isfinite(midpoints_2d))


def test_homogeneous_dg0_material_functions_have_healthy_values():
    msh = _create_unit_square_mesh()

    functions = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)

    assert set(functions) == {"rho", "C11", "C22", "C12", "C66"}

    expected = {
        "rho": 4.5,
        "C11": 16.24,
        "C22": 18.07,
        "C12": 6.9,
        "C66": 4.67,
    }

    for name, value in expected.items():
        owned_values = _owned_dof_values(functions[name])
        assert owned_values.size > 0 or MPI.COMM_WORLD.size > 1
        np.testing.assert_allclose(owned_values, value)


def test_defective_dg0_material_functions_can_mark_all_cells():
    msh = _create_unit_square_mesh()

    scenario = DefectScenario(
        name="all_cells_defective",
        defects=(
            CircularDefectSpec(
                name="large_defect",
                center=(0.5, 0.5),
                radius=10.0,
                material_factor=0.8,
            ),
        ),
    )

    functions = defective_dg0_material_functions(msh, scenario)

    expected = {
        "rho": 0.8 * 4.5,
        "C11": 0.8 * 16.24,
        "C22": 0.8 * 18.07,
        "C12": 0.8 * 6.9,
        "C66": 0.8 * 4.67,
    }

    for name, value in expected.items():
        owned_values = _owned_dof_values(functions[name])
        np.testing.assert_allclose(owned_values, value)


def test_material_indices_and_dg0_functions_return_consistent_objects():
    msh = _create_unit_square_mesh()

    scenario = DefectScenario(
        name="all_cells_defective",
        defects=(
            CircularDefectSpec(
                name="large_defect",
                center=(0.5, 0.5),
                radius=10.0,
                material_factor=0.8,
            ),
        ),
    )

    indices, functions = material_indices_and_dg0_functions(msh, scenario)

    cells = local_cell_indices(msh, include_ghosts=True)

    assert indices.shape == cells.shape
    assert np.all(indices == 0)
    assert set(functions) == {"rho", "C11", "C22", "C12", "C66"}
