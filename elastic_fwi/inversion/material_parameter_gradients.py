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
from elastic_fwi.solvers.elastic_operator import strain_voigt


@dataclass(frozen=True)
class MaterialParameterGradientFields:
    fields: dict[str, fem.Function]
    metadata: dict[str, Any]


_FIELD_SUFFIXES = ("rho", "C11", "C22", "C12", "C66")
_GRADIENT_FIELD_NAMES = tuple(f"gradient_{suffix}" for suffix in _FIELD_SUFFIXES)
_KERNEL_FIELD_NAMES = tuple(f"kernel_{suffix}" for suffix in _FIELD_SUFFIXES)


def _owned_local_size(Q) -> int:
    return int(Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs)


def _assemble_dg0_cell_integrals(Q, expr) -> np.ndarray:
    """Assemble per-cell DG0 linear-form contributions ``∫_K expr dx``."""
    q = ufl.TestFunction(Q)
    form = fem.form(expr * q * ufl.dx)

    vec = fem_petsc.assemble_vector(form)
    try:
        vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        local_size = _owned_local_size(Q)
        return np.asarray(vec.getArray(readonly=True)[:local_size], dtype=float).copy()
    finally:
        vec.destroy()


def _assemble_dg0_cell_measures(Q) -> np.ndarray:
    """Return owned DG0 cell measures ``|K|`` in dof order.

    For scalar DG0, the mass matrix is diagonal and its diagonal entry on a
    cell K is exactly ``|K|``.  Dividing an assembled DG0 linear-form vector
    by these measures therefore gives the exact DG0 L2 projection / cell
    average of the underlying density.
    """
    q = ufl.TestFunction(Q)
    form = fem.form(1.0 * q * ufl.dx)

    vec = fem_petsc.assemble_vector(form)
    try:
        vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        local_size = _owned_local_size(Q)
        measures = np.asarray(
            vec.getArray(readonly=True)[:local_size], dtype=float
        ).copy()
    finally:
        vec.destroy()

    if measures.size and (
        not np.all(np.isfinite(measures)) or np.any(measures <= 0.0)
    ):
        raise FloatingPointError(
            "DG0 cell measures must be finite and strictly positive."
        )
    return measures


def _cell_average_density_from_integrals(
    integrals: np.ndarray,
    cell_measures: np.ndarray,
) -> np.ndarray:
    """Convert per-cell derivative contributions to an L2 density."""
    integrals = np.asarray(integrals, dtype=float)
    cell_measures = np.asarray(cell_measures, dtype=float)
    if integrals.shape != cell_measures.shape:
        raise ValueError(
            "DG0 integral and cell-measure arrays must have identical shapes."
        )
    if cell_measures.size and np.any(cell_measures <= 0.0):
        raise FloatingPointError(
            "DG0 cell measures must be strictly positive."
        )
    return integrals / cell_measures


def _normalize_global(values: np.ndarray, comm: MPI.Comm) -> tuple[np.ndarray, float]:
    local_max = float(np.max(np.abs(values))) if values.size else 0.0
    global_max = float(comm.allreduce(local_max, op=MPI.MAX))

    if global_max > 0.0:
        return values / global_max, global_max

    return values, global_max


def _make_dg0_function(Q, name: str, owned_values: np.ndarray) -> fem.Function:
    f = fem.Function(Q)
    f.name = name

    n = min(len(owned_values), len(f.x.array))
    f.x.array[:n] = owned_values[:n]

    if len(f.x.array) > n:
        f.x.array[n:] = 0.0

    f.x.scatter_forward()
    return f


def _zero_like(vector: PETSc.Vec) -> PETSc.Vec:
    result = vector.copy()
    result.set(0.0)
    result.assemble()
    return result


def _linear_combination(template: PETSc.Vec, terms: tuple[tuple[float, PETSc.Vec], ...]) -> PETSc.Vec:
    result = template.copy()
    result.set(0.0)
    for coefficient, vector in terms:
        result.axpy(float(coefficient), vector)
    result.assemble()
    return result


def _newmark_second_difference_vector(snapshots, index: int, dt: float) -> PETSc.Vec:
    """
    C++ legacy density-kernel acceleration.

    The stored snapshots are u^{n+1}. The historical C++ density image uses

        a^{n+1} = (u^{n+1} - 2 u^n + u^{n-1}) / dt^2,

    with homogeneous initial values u^0 = u^{-1} = 0 for the first stored steps.

    This helper is used only for the legacy correlation-kernel export. It is not
    the exact discrete Newmark gradient density unless the global Newmark sign
    and time weights are applied separately.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive when computing Newmark accelerations.")

    if index < 0 or index >= len(snapshots):
        raise IndexError("Snapshot index out of range.")

    inv_dt2 = 1.0 / (float(dt) * float(dt))

    acceleration = snapshots[index].copy()
    acceleration.scale(inv_dt2)

    if index >= 1:
        acceleration.axpy(-2.0 * inv_dt2, snapshots[index - 1])

    if index >= 2:
        acceleration.axpy(inv_dt2, snapshots[index - 2])

    acceleration.assemble()
    return acceleration


def _finalize_dg0_fields(
    *,
    Q,
    accum: dict[str, np.ndarray],
    comm: MPI.Comm,
    absolute_value: bool,
    normalize: bool,
    metadata: dict[str, Any],
    cell_measures: np.ndarray | None = None,
) -> MaterialParameterGradientFields:
    fields: dict[str, fem.Function] = {}
    metadata = dict(metadata)
    metadata.update(
        {
            "absolute_value": bool(absolute_value),
            "absolute_value_stage": "after_time_accumulation",
            "normalize": bool(normalize),
            "fields": {},
        }
    )

    for name, values in accum.items():
        raw_local_min = float(np.min(values)) if values.size else 0.0
        raw_local_max = float(np.max(values)) if values.size else 0.0
        if cell_measures is None:
            raw_local_l2_sq = float(np.dot(values, values))
            l2_kind = "euclidean_dof_norm"
        else:
            if values.shape != cell_measures.shape:
                raise ValueError(
                    "DG0 field values and cell measures must have identical shapes."
                )
            raw_local_l2_sq = float(
                np.dot(values * values, cell_measures)
            )
            l2_kind = "area_weighted_L2_density_norm"

        raw_global_min = float(comm.allreduce(raw_local_min, op=MPI.MIN))
        raw_global_max = float(comm.allreduce(raw_local_max, op=MPI.MAX))
        raw_global_l2 = sqrt(
            float(comm.allreduce(raw_local_l2_sq, op=MPI.SUM))
        )

        exported = values.copy()

        if absolute_value:
            exported = np.abs(exported)

        normalizer = 0.0

        if normalize:
            exported, normalizer = _normalize_global(exported, comm)

        fields[name] = _make_dg0_function(Q, name, exported)

        metadata["fields"][name] = {
            "raw_global_min": raw_global_min,
            "raw_global_max": raw_global_max,
            "raw_global_l2_norm": raw_global_l2,
            "raw_global_l2_norm_kind": l2_kind,
            "normalizer": float(normalizer),
            "exported_global_max": float(
                comm.allreduce(
                    float(np.max(exported)) if exported.size else 0.0,
                    op=MPI.MAX,
                )
            ),
        }

    return MaterialParameterGradientFields(fields=fields, metadata=metadata)


def compute_material_parameter_correlation_kernels_dg0(
    *,
    V,
    direct_snapshots,
    adjoint_snapshots,
    dt: float,
    absolute_value: bool = True,
    normalize: bool = True,
    field_prefix: str = "kernel",
) -> MaterialParameterGradientFields:
    """
    Export legacy direct-adjoint correlation images on DG0 cells.

    These fields are sensitivities/kernels in the imaging sense. They preserve
    the historical C++ products

        + integral p · a_u dt,
        + integral eps(u) : eps(p) dt,

    and therefore must not be interpreted as the exact discrete Newmark gradient
    unless an explicitly documented global sign and Newmark time weighting are
    applied outside this routine.
    """
    if len(direct_snapshots) == 0:
        raise ValueError("At least one direct snapshot is required.")

    if len(direct_snapshots) != len(adjoint_snapshots):
        raise ValueError("direct_snapshots and adjoint_snapshots must have the same length.")

    if field_prefix not in {"kernel", "gradient"}:
        raise ValueError("field_prefix must be 'kernel' or 'gradient'.")

    comm = V.mesh.comm
    Q = fem.functionspace(V.mesh, ("DG", 0))
    local_size = _owned_local_size(Q)
    cell_measures = _assemble_dg0_cell_measures(Q)

    names = {suffix: f"{field_prefix}_{suffix}" for suffix in _FIELD_SUFFIXES}
    accum = {name: np.zeros(local_size, dtype=float) for name in names.values()}

    for step_index, (u_vec, p_vec) in enumerate(
        zip(direct_snapshots, adjoint_snapshots, strict=True)
    ):
        u = displacement_vector_to_function(V, u_vec, name="u_direct_for_correlation_kernel")
        p = displacement_vector_to_function(V, p_vec, name="p_adjoint_for_correlation_kernel")

        a_vec = _newmark_second_difference_vector(direct_snapshots, step_index, dt)
        try:
            a_u = displacement_vector_to_function(
                V,
                a_vec,
                name="a_direct_for_density_kernel",
            )

            e11_u, e22_u, gamma12_u = strain_voigt(u)
            e11_p, e22_p, gamma12_p = strain_voigt(p)

            densities = {
                names["rho"]: ufl.inner(a_u, p),
                names["C11"]: e11_u * e11_p,
                names["C22"]: e22_u * e22_p,
                names["C12"]: e11_u * e22_p + e22_u * e11_p,
                names["C66"]: gamma12_u * gamma12_p,
            }

            for name, expr in densities.items():
                integrals = _assemble_dg0_cell_integrals(Q, expr)
                values = _cell_average_density_from_integrals(
                    integrals, cell_measures
                )

                if values.shape[0] != local_size:
                    raise ValueError(
                        f"Unexpected DG0 kernel local size for {name}: "
                        f"{values.shape[0]} != {local_size}."
                    )

                accum[name] += float(dt) * values
        finally:
            a_vec.destroy()

    return _finalize_dg0_fields(
        Q=Q,
        accum=accum,
        comm=comm,
        absolute_value=absolute_value,
        normalize=normalize,
        cell_measures=cell_measures,
        metadata={
            "kind": "material_parameter_correlation_kernels_dg0",
            "num_steps": int(len(direct_snapshots)),
            "dt": float(dt),
            "field_prefix": field_prefix,
            "mathematical_role": "direct_adjoint_correlation_kernel_not_fd_validated_gradient",
            "is_true_discrete_gradient": False,
            "density_kernel_uses": "newmark_second_difference_acceleration",
            "stiffness_kernel_uses": "strain_interaction",
            "time_weighting": "legacy_dt_integral",
            "sign_convention": "+ direct_adjoint_correlation",
            "spatial_representation": "DG0_L2_cell_average_density",
            "cell_measure_normalization": True,
            "cell_measure_min": float(
                comm.allreduce(
                    float(np.min(cell_measures)) if cell_measures.size else np.inf,
                    op=MPI.MIN,
                )
            ),
            "cell_measure_max": float(
                comm.allreduce(
                    float(np.max(cell_measures)) if cell_measures.size else 0.0,
                    op=MPI.MAX,
                )
            ),
        },
    )


def compute_material_parameter_gradients_dg0(
    *,
    V,
    direct_snapshots,
    adjoint_snapshots,
    dt: float,
    theta: float,
    absolute_value: bool = True,
    normalize: bool = True,
) -> MaterialParameterGradientFields:
    """
    Export the exact signed DG0 densities of the discrete Newmark gradient.

    The formulas match compute_newmark_material_directional_gradient. For any
    DG0 material perturbation δm, the area-weighted cell contraction of the
    raw exported density with δm equals

        Σ_n < μ^{n+1}, δB u^n - δA(u^{n+1}+u^{n-1}) >.

    The DG0 coefficients are true cell-average/L2 gradient densities, not
    per-cell integrated derivative contributions. This distinction is essential
    on locally refined meshes.
    """
    if len(direct_snapshots) == 0:
        raise ValueError("At least one direct snapshot is required.")

    if len(direct_snapshots) != len(adjoint_snapshots):
        raise ValueError("direct_snapshots and adjoint_snapshots must have the same length.")

    dt = float(dt)
    theta = float(theta)

    if dt <= 0.0:
        raise ValueError("dt must be strictly positive.")

    comm = V.mesh.comm
    Q = fem.functionspace(V.mesh, ("DG", 0))
    local_size = _owned_local_size(Q)
    cell_measures = _assemble_dg0_cell_measures(Q)

    accum = {name: np.zeros(local_size, dtype=float) for name in _GRADIENT_FIELD_NAMES}
    zero_state = _zero_like(direct_snapshots[0])

    try:
        for step_index, (next_vec, p_vec) in enumerate(
            zip(direct_snapshots, adjoint_snapshots, strict=True)
        ):
            if step_index == 0:
                current_vec = zero_state
                previous_vec = zero_state
            elif step_index == 1:
                current_vec = direct_snapshots[step_index - 1]
                previous_vec = zero_state
            else:
                current_vec = direct_snapshots[step_index - 1]
                previous_vec = direct_snapshots[step_index - 2]

            mass_vec = _linear_combination(
                next_vec,
                (
                    (2.0, current_vec),
                    (-1.0, next_vec),
                    (-1.0, previous_vec),
                ),
            )
            stiffness_vec = _linear_combination(
                next_vec,
                (
                    (-(dt * dt) * (1.0 - 2.0 * theta), current_vec),
                    (-(dt * dt) * theta, next_vec),
                    (-(dt * dt) * theta, previous_vec),
                ),
            )

            try:
                mass_state = displacement_vector_to_function(
                    V,
                    mass_vec,
                    name="newmark_density_gradient_state",
                )
                stiffness_state = displacement_vector_to_function(
                    V,
                    stiffness_vec,
                    name="newmark_stiffness_gradient_state",
                )
                p = displacement_vector_to_function(
                    V,
                    p_vec,
                    name="p_adjoint_for_discrete_gradient",
                )

                e11_s, e22_s, gamma12_s = strain_voigt(stiffness_state)
                e11_p, e22_p, gamma12_p = strain_voigt(p)

                densities = {
                    "gradient_rho": ufl.inner(mass_state, p),
                    "gradient_C11": e11_s * e11_p,
                    "gradient_C22": e22_s * e22_p,
                    "gradient_C12": e11_s * e22_p + e22_s * e11_p,
                    "gradient_C66": gamma12_s * gamma12_p,
                }

                for name, expr in densities.items():
                    integrals = _assemble_dg0_cell_integrals(Q, expr)
                    values = _cell_average_density_from_integrals(
                        integrals, cell_measures
                    )

                    if values.shape[0] != local_size:
                        raise ValueError(
                            f"Unexpected DG0 gradient local size for {name}: "
                            f"{values.shape[0]} != {local_size}."
                        )

                    accum[name] += values
            finally:
                mass_vec.destroy()
                stiffness_vec.destroy()
    finally:
        zero_state.destroy()

    return _finalize_dg0_fields(
        Q=Q,
        accum=accum,
        comm=comm,
        absolute_value=absolute_value,
        normalize=normalize,
        cell_measures=cell_measures,
        metadata={
            "kind": "material_parameter_discrete_gradients_dg0",
            "num_steps": int(len(direct_snapshots)),
            "dt": dt,
            "theta": theta,
            "field_prefix": "gradient",
            "mathematical_role": "fd_validated_discrete_newmark_gradient_density",
            "is_true_discrete_gradient": True,
            "gradient_identity": (
                "sum_cells cell_measure * gradient_density_m * delta_m "
                "== material_directional_gradient"
            ),
            "density_gradient_uses": "2*u^n - u^{n+1} - u^{n-1}",
            "stiffness_gradient_uses": "-dt^2*((1-2*theta)*u^n + theta*(u^{n+1}+u^{n-1}))",
            "time_weighting": "exact_discrete_newmark_no_extra_dt_integral",
            "sign_convention": "matches_positive_H_transpose_residual_adjoint_and_forward_fd_derivative",
            "spatial_representation": "DG0_L2_cell_average_density",
            "cell_measure_normalization": True,
            "cell_measure_min": float(
                comm.allreduce(
                    float(np.min(cell_measures)) if cell_measures.size else np.inf,
                    op=MPI.MIN,
                )
            ),
            "cell_measure_max": float(
                comm.allreduce(
                    float(np.max(cell_measures)) if cell_measures.size else 0.0,
                    op=MPI.MAX,
                )
            ),
        },
    )
