from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


def owned_scalar_dof_coordinates_2d(V) -> np.ndarray:
    """
    Return the owned scalar dof coordinates as an array of shape (n_local_dofs, 2).

    The implementation is robust to blocked vector spaces and to possible
    ghost entries returned by DOLFINx coordinate tabulation.
    """

    coords = np.asarray(V.tabulate_dof_coordinates(), dtype=float)

    if coords.ndim != 2:
        raise ValueError("tabulate_dof_coordinates() must return a 2D array.")

    if coords.shape[1] < 2:
        raise ValueError("The function space must live on a mesh of geometric dimension >= 2.")

    coords = coords[:, :2].copy()

    index_map = V.dofmap.index_map
    block_size = int(getattr(V.dofmap, "index_map_bs", getattr(V.dofmap, "bs", 1)))

    num_owned_blocks = int(index_map.size_local)
    num_ghost_blocks = int(index_map.num_ghosts)

    num_owned_scalar_dofs = num_owned_blocks * block_size
    num_all_blocks = num_owned_blocks + num_ghost_blocks
    num_all_scalar_dofs = num_all_blocks * block_size

    # Several DOLFINx layouts are possible depending on element/blocking.
    if coords.shape[0] == num_owned_scalar_dofs:
        owned = coords

    elif coords.shape[0] == num_all_scalar_dofs:
        owned = coords[:num_owned_scalar_dofs]

    elif coords.shape[0] == num_owned_blocks:
        owned = np.repeat(coords, block_size, axis=0)

    elif coords.shape[0] == num_all_blocks:
        owned = np.repeat(coords[:num_owned_blocks], block_size, axis=0)

    else:
        raise ValueError(
            "Unexpected number of tabulated dof coordinates: "
            f"got {coords.shape[0]}, expected one of "
            f"{num_owned_scalar_dofs}, {num_all_scalar_dofs}, "
            f"{num_owned_blocks}, {num_all_blocks}."
        )

    return np.ascontiguousarray(owned, dtype=float)


@dataclass(frozen=True)
class MaterialQualitativeDofImageResult:
    local_values: np.ndarray
    num_steps: int
    local_size: int
    global_size: int
    local_min: float
    local_max: float
    global_min: float
    global_max: float
    global_l2_norm: float
    global_sum: float
    normalized: bool
    local_coordinates: np.ndarray | None = None
    coordinate_dimension: int = 0

    def as_dict(self, *, include_values: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": "material_qualitative_dof_image",
            "num_steps": self.num_steps,
            "local_size": self.local_size,
            "global_size": self.global_size,
            "local_min": self.local_min,
            "local_max": self.local_max,
            "global_min": self.global_min,
            "global_max": self.global_max,
            "global_l2_norm": self.global_l2_norm,
            "global_sum": self.global_sum,
            "normalized": self.normalized,
            "has_coordinates": self.local_coordinates is not None,
            "coordinate_dimension": self.coordinate_dimension,
        }

        if self.local_coordinates is not None:
            data["local_coordinate_count"] = int(self.local_coordinates.shape[0])

        if include_values:
            data["local_values"] = self.local_values.tolist()
            if self.local_coordinates is not None:
                data["local_coordinates"] = self.local_coordinates.tolist()

        return data


def _readonly_array(vector: PETSc.Vec) -> np.ndarray:
    return np.asarray(vector.getArray(readonly=True), dtype=float)


def compute_material_qualitative_dof_image(
    *,
    direct_snapshots: tuple[PETSc.Vec, ...],
    equation_adjoint_vectors: tuple[PETSc.Vec, ...],
    function_space: Any | None = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
    absolute_value: bool = True,
    normalize: bool = True,
) -> MaterialQualitativeDofImageResult:
    if len(direct_snapshots) == 0:
        raise ValueError("At least one direct snapshot is required.")

    if len(direct_snapshots) != len(equation_adjoint_vectors):
        raise ValueError(
            "direct_snapshots and equation_adjoint_vectors must have the same length."
        )

    local_size = int(direct_snapshots[0].getLocalSize())

    image = np.zeros(local_size, dtype=float)

    for step, (u_vec, adj_vec) in enumerate(
        zip(direct_snapshots, equation_adjoint_vectors, strict=True),
        start=1,
    ):
        if int(u_vec.getLocalSize()) != local_size:
            raise ValueError(f"Direct snapshot local size mismatch at step {step}.")

        if int(adj_vec.getLocalSize()) != local_size:
            raise ValueError(f"Adjoint vector local size mismatch at step {step}.")

        u_local = _readonly_array(u_vec)
        adj_local = _readonly_array(adj_vec)

        if u_local.shape != adj_local.shape:
            raise ValueError(f"Direct/adjoint local array shape mismatch at step {step}.")

        contribution = u_local * adj_local

        if absolute_value:
            contribution = np.abs(contribution)

        image += contribution

    local_raw_abs_max = float(np.max(np.abs(image))) if image.size else 0.0
    global_raw_abs_max = float(comm.allreduce(local_raw_abs_max, op=MPI.MAX))

    normalized = False
    if normalize and global_raw_abs_max > 0.0:
        image = image / global_raw_abs_max
        normalized = True

    local_min = float(np.min(image)) if image.size else 0.0
    local_max = float(np.max(image)) if image.size else 0.0
    local_sum = float(np.sum(image))
    local_l2_squared = float(np.dot(image, image))

    global_min = float(comm.allreduce(local_min, op=MPI.MIN))
    global_max = float(comm.allreduce(local_max, op=MPI.MAX))
    global_sum = float(comm.allreduce(local_sum, op=MPI.SUM))
    global_l2_squared = float(comm.allreduce(local_l2_squared, op=MPI.SUM))
    global_size = int(comm.allreduce(local_size, op=MPI.SUM))

    local_coordinates = None
    coordinate_dimension = 0

    if function_space is not None:
        local_coordinates = owned_scalar_dof_coordinates_2d(function_space)

        if local_coordinates.ndim != 2 or local_coordinates.shape[1] != 2:
            raise ValueError("local dof coordinates must have shape (n_local_dofs, 2).")

        if local_coordinates.shape[0] != local_size:
            raise ValueError(
                "The number of local dof coordinates does not match the image local size: "
                f"{local_coordinates.shape[0]} != {local_size}."
            )

        if not np.all(np.isfinite(local_coordinates)):
            raise ValueError("local dof coordinates contain non-finite values.")

        coordinate_dimension = 2

    return MaterialQualitativeDofImageResult(
        local_values=image,
        num_steps=len(direct_snapshots),
        local_size=local_size,
        global_size=global_size,
        local_min=local_min,
        local_max=local_max,
        global_min=global_min,
        global_max=global_max,
        global_l2_norm=sqrt(max(global_l2_squared, 0.0)),
        global_sum=global_sum,
        normalized=normalized,
        local_coordinates=local_coordinates,
        coordinate_dimension=coordinate_dimension,
    )
