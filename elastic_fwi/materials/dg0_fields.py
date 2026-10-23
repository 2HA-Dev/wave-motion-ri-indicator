from __future__ import annotations

import numpy as np
from dolfinx import fem

from elastic_fwi.geometry.cell_centers import local_cell_indices, local_cell_midpoints_2d
from elastic_fwi.materials.defects import DefectScenario
from elastic_fwi.materials.orthotropic import (
    LEGACY_HEALTHY_MATERIAL,
    OrthotropicPlaneStrainConstants,
)
from elastic_fwi.materials.point_fields import (
    OrthotropicPointFields,
    defective_material_fields_at_points,
    homogeneous_material_fields_at_points,
    material_indices_and_fields_at_points,
)


def create_scalar_dg0_space(msh):
    """
    Create a scalar DG0 function space.

    The tuple syntax is supported by current DOLFINx versions.
    """

    return fem.functionspace(msh, ("DG", 0))


def create_named_dg0_function(V, name: str):
    function = fem.Function(V)
    function.name = name
    return function


def assign_cellwise_values_to_dg0(function, values: object, *, include_ghosts: bool = True):
    """
    Assign one scalar value per local cell to a scalar DG0 Function.

    For DG0 scalar spaces, each cell has exactly one cell dof.
    """

    values_array = np.asarray(values, dtype=float)

    if values_array.ndim != 1:
        raise ValueError("DG0 cellwise values must be one-dimensional.")

    if not np.all(np.isfinite(values_array)):
        raise ValueError("DG0 cellwise values must be finite.")

    V = function.function_space
    msh = V.mesh
    cells = local_cell_indices(msh, include_ghosts=include_ghosts)

    if values_array.shape[0] != cells.shape[0]:
        raise ValueError(
            "Number of DG0 values does not match the number of selected local cells: "
            f"{values_array.shape[0]} values for {cells.shape[0]} cells."
        )

    array = function.x.array

    for cell, value in zip(cells, values_array, strict=True):
        dofs = V.dofmap.cell_dofs(int(cell))

        if len(dofs) != 1:
            raise RuntimeError("Expected exactly one scalar DG0 dof per cell.")

        array[dofs[0]] = value

    function.x.scatter_forward()
    return function


def dg0_functions_from_point_fields(
    msh,
    fields: OrthotropicPointFields,
    *,
    include_ghosts: bool = True,
) -> dict[str, object]:
    """
    Convert material point fields to DOLFINx scalar DG0 functions.
    """

    V0 = create_scalar_dg0_space(msh)

    functions = {
        "rho": create_named_dg0_function(V0, "rho"),
        "C11": create_named_dg0_function(V0, "C11"),
        "C22": create_named_dg0_function(V0, "C22"),
        "C12": create_named_dg0_function(V0, "C12"),
        "C66": create_named_dg0_function(V0, "C66"),
    }

    data = fields.as_dict()

    for name, function in functions.items():
        assign_cellwise_values_to_dg0(
            function,
            data[name],
            include_ghosts=include_ghosts,
        )

    return functions


def homogeneous_dg0_material_functions(
    msh,
    material: OrthotropicPlaneStrainConstants = LEGACY_HEALTHY_MATERIAL,
    *,
    include_ghosts: bool = True,
) -> dict[str, object]:
    """
    Build homogeneous DG0 material functions on a mesh.
    """

    points = local_cell_midpoints_2d(msh, include_ghosts=include_ghosts)
    fields = homogeneous_material_fields_at_points(points, material)
    return dg0_functions_from_point_fields(msh, fields, include_ghosts=include_ghosts)


def defective_dg0_material_functions(
    msh,
    scenario: DefectScenario,
    *,
    healthy_material: OrthotropicPlaneStrainConstants = LEGACY_HEALTHY_MATERIAL,
    background_factor: float = 1.0,
    tol: float = 0.0,
    conflict_policy: str = "last",
    include_ghosts: bool = True,
) -> dict[str, object]:
    """
    Build defective DG0 material functions from analytical defects.
    """

    points = local_cell_midpoints_2d(msh, include_ghosts=include_ghosts)

    fields = defective_material_fields_at_points(
        points,
        scenario,
        healthy_material=healthy_material,
        background_factor=background_factor,
        tol=tol,
        conflict_policy=conflict_policy,
    )

    return dg0_functions_from_point_fields(msh, fields, include_ghosts=include_ghosts)


def material_indices_and_dg0_functions(
    msh,
    scenario: DefectScenario,
    *,
    healthy_material: OrthotropicPlaneStrainConstants = LEGACY_HEALTHY_MATERIAL,
    background_factor: float = 1.0,
    tol: float = 0.0,
    conflict_policy: str = "last",
    include_ghosts: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Return defect indices at local cell midpoints and associated DG0 functions.
    """

    points = local_cell_midpoints_2d(msh, include_ghosts=include_ghosts)

    indices, fields = material_indices_and_fields_at_points(
        points,
        scenario,
        healthy_material=healthy_material,
        background_factor=background_factor,
        tol=tol,
        conflict_policy=conflict_policy,
    )

    functions = dg0_functions_from_point_fields(
        msh,
        fields,
        include_ghosts=include_ghosts,
    )

    return indices, functions
