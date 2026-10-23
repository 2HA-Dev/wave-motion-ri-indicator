from __future__ import annotations

import numpy as np
from dolfinx.mesh import compute_midpoints


def local_cell_indices(msh, *, include_ghosts: bool = True) -> np.ndarray:
    """
    Return local cell indices for a DOLFINx mesh.

    If include_ghosts=True, the returned array includes owned and ghost cells.
    This is useful for deterministic local DG0 construction.
    """

    tdim = msh.topology.dim
    index_map = msh.topology.index_map(tdim)

    if index_map is None:
        raise RuntimeError("Cell index map is not available on this mesh.")

    num_cells = index_map.size_local

    if include_ghosts:
        num_cells += index_map.num_ghosts

    return np.arange(num_cells, dtype=np.int32)


def local_cell_midpoints(msh, *, include_ghosts: bool = True) -> np.ndarray:
    """
    Return local cell midpoints as an array of shape (n_cells, gdim).
    """

    tdim = msh.topology.dim
    cells = local_cell_indices(msh, include_ghosts=include_ghosts)
    midpoints = compute_midpoints(msh, tdim, cells)

    if midpoints.ndim != 2:
        raise RuntimeError("DOLFINx returned invalid midpoint array.")

    return np.asarray(midpoints, dtype=float)


def local_cell_midpoints_2d(msh, *, include_ghosts: bool = True) -> np.ndarray:
    """
    Return local cell midpoints restricted to their first two coordinates.

    This is the format expected by the pure Python material/defect layer.
    """

    midpoints = local_cell_midpoints(msh, include_ghosts=include_ghosts)

    if midpoints.shape[1] < 2:
        raise ValueError("The mesh geometric dimension must be at least 2.")

    return midpoints[:, :2].copy()
