from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc

from elastic_fwi.observations.time_series import displacement_vector_to_function


@dataclass(frozen=True)
class DirectAdjointInteractionImage:
    fields: dict[str, fem.Function]
    metadata: dict[str, Any]


def _owned_local_size(Q) -> int:
    return int(Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs)


def _assemble_dg0_density(Q, expr) -> np.ndarray:
    q = ufl.TestFunction(Q)
    form = fem.form(expr * q * ufl.dx)

    vec = fem_petsc.assemble_vector(form)
    try:
        vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        local_size = _owned_local_size(Q)
        return np.asarray(vec.getArray(readonly=True)[:local_size], dtype=float).copy()
    finally:
        vec.destroy()


def _make_dg0_function(Q, name: str, owned_values: np.ndarray) -> fem.Function:
    f = fem.Function(Q)
    f.name = name

    n = min(len(owned_values), len(f.x.array))
    f.x.array[:n] = owned_values[:n]

    if len(f.x.array) > n:
        f.x.array[n:] = 0.0

    f.x.scatter_forward()
    return f


def _global_max_abs(values: np.ndarray, comm: MPI.Comm) -> float:
    local = float(np.max(np.abs(values))) if values.size else 0.0
    return float(comm.allreduce(local, op=MPI.MAX))


def compute_direct_adjoint_interaction_image_dg0(
    *,
    V,
    direct_snapshots,
    adjoint_snapshots,
    dt: float,
    eps: float = 1.0e-12,
) -> DirectAdjointInteractionImage:
    if len(direct_snapshots) == 0:
        raise ValueError("At least one direct snapshot is required.")

    if len(direct_snapshots) != len(adjoint_snapshots):
        raise ValueError("direct_snapshots and adjoint_snapshots must have the same length.")

    comm = V.mesh.comm
    Q = fem.functionspace(V.mesh, ("DG", 0))
    local_size = _owned_local_size(Q)

    interaction = np.zeros(local_size, dtype=float)
    signed_interaction = np.zeros(local_size, dtype=float)
    direct_energy = np.zeros(local_size, dtype=float)
    adjoint_energy = np.zeros(local_size, dtype=float)

    for u_vec, p_vec in zip(direct_snapshots, adjoint_snapshots, strict=True):
        u = displacement_vector_to_function(V, u_vec, name="u_direct_for_interaction")
        p = displacement_vector_to_function(V, p_vec, name="p_adjoint_for_interaction")

        interaction += dt * _assemble_dg0_density(
            Q,
            ufl.sqrt(ufl.inner(u, u) + eps) * ufl.sqrt(ufl.inner(p, p) + eps),
        )

        signed_interaction += dt * _assemble_dg0_density(
            Q,
            ufl.inner(u, p),
        )

        direct_energy += dt * _assemble_dg0_density(
            Q,
            ufl.inner(u, u),
        )

        adjoint_energy += dt * _assemble_dg0_density(
            Q,
            ufl.inner(p, p),
        )

    max_interaction = _global_max_abs(interaction, comm)
    max_signed = _global_max_abs(signed_interaction, comm)
    max_direct = _global_max_abs(direct_energy, comm)
    max_adjoint = _global_max_abs(adjoint_energy, comm)

    normalized = interaction / max_interaction if max_interaction > 0.0 else interaction.copy()
    signed_normalized = signed_interaction / max_signed if max_signed > 0.0 else signed_interaction.copy()
    direct_normalized = direct_energy / max_direct if max_direct > 0.0 else direct_energy.copy()
    adjoint_normalized = adjoint_energy / max_adjoint if max_adjoint > 0.0 else adjoint_energy.copy()

    log_interaction = np.log10(eps + np.maximum(normalized, 0.0))

    fields = {
        "interaction_image": _make_dg0_function(Q, "interaction_image", normalized),
        "log_interaction_image": _make_dg0_function(Q, "log_interaction_image", log_interaction),
        "signed_interaction_image": _make_dg0_function(Q, "signed_interaction_image", signed_normalized),
        "direct_energy_image": _make_dg0_function(Q, "direct_energy_image", direct_normalized),
        "adjoint_energy_image": _make_dg0_function(Q, "adjoint_energy_image", adjoint_normalized),
    }

    metadata = {
        "kind": "direct_adjoint_interaction_image_dg0",
        "num_steps": int(len(direct_snapshots)),
        "dt": float(dt),
        "eps": float(eps),
        "raw_max_interaction": float(max_interaction),
        "raw_max_signed_interaction": float(max_signed),
        "raw_max_direct_energy": float(max_direct),
        "raw_max_adjoint_energy": float(max_adjoint),
        "field_names": sorted(fields.keys()),
    }

    return DirectAdjointInteractionImage(fields=fields, metadata=metadata)
