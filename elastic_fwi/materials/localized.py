from __future__ import annotations

from typing import Any

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI

from elastic_fwi.materials.orthotropic import (
    LEGACY_HEALTHY_MATERIAL,
    OrthotropicPlaneStrainConstants,
)


_MATERIAL_NAMES = ("rho", "C11", "C22", "C12", "C66")


def _material_value(constants: OrthotropicPlaneStrainConstants, name: str) -> float:
    return float(getattr(constants, name))


def localized_disk_dg0_material_functions(
    msh,
    *,
    center_x: float = 0.5,
    center_y: float = 0.5,
    radius: float = 0.35,
    contrast_factor: float = 0.8,
    healthy_material: OrthotropicPlaneStrainConstants = LEGACY_HEALTHY_MATERIAL,
) -> tuple[dict[str, fem.Function], dict[str, Any]]:
    """
    Build DG0 material fields with one localized circular defect.

    The material is equal to healthy_material outside the disk and to

        contrast_factor * healthy_material

    inside the disk, cellwise at DG0 cell centers.
    """

    center_x = float(center_x)
    center_y = float(center_y)
    radius = float(radius)
    contrast_factor = float(contrast_factor)

    if radius <= 0.0:
        raise ValueError("radius must be strictly positive.")

    if contrast_factor <= 0.0:
        raise ValueError("contrast_factor must be strictly positive.")

    q0 = fem.functionspace(msh, ("DG", 0))
    tdim = msh.topology.dim
    cell_map = msh.topology.index_map(tdim)
    num_owned_cells = int(cell_map.size_local)

    owned_cells = np.arange(num_owned_cells, dtype=np.int32)

    if owned_cells.size > 0:
        midpoints = mesh.compute_midpoints(msh, tdim, owned_cells)[:, :2]
        dx = midpoints[:, 0] - center_x
        dy = midpoints[:, 1] - center_y
        local_mask = dx * dx + dy * dy <= radius**2
    else:
        local_mask = np.zeros(0, dtype=bool)

    local_defect_cells = int(np.count_nonzero(local_mask))
    global_defect_cells = int(
        msh.comm.allreduce(local_defect_cells, op=MPI.SUM)
    )

    materials: dict[str, fem.Function] = {}

    for name in _MATERIAL_NAMES:
        function = fem.Function(q0)
        function.name = name

        base_value = _material_value(healthy_material, name)
        function.x.array[:] = base_value

        for cell, is_defect in zip(owned_cells, local_mask, strict=True):
            if is_defect:
                dof = q0.dofmap.cell_dofs(int(cell))[0]
                function.x.array[dof] = contrast_factor * base_value

        function.x.scatter_forward()
        materials[name] = function

    metadata = {
        "kind": "localized_disk_defect",
        "center_x": center_x,
        "center_y": center_y,
        "radius": radius,
        "contrast_factor": contrast_factor,
        "local_defect_cells": local_defect_cells,
        "global_defect_cells": global_defect_cells,
    }

    return materials, metadata
