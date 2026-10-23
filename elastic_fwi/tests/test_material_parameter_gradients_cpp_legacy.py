from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc

from elastic_fwi.inversion.material_parameter_gradients import (
    compute_material_parameter_correlation_kernels_dg0,
)
from elastic_fwi.solvers.elastic_operator import create_vector_lagrange_space


def _constant_vec_for_space(V, value: float) -> PETSc.Vec:
    comm = V.mesh.comm
    bs = int(V.dofmap.index_map_bs)
    local_size = int(V.dofmap.index_map.size_local) * bs
    global_size = int(V.dofmap.index_map.size_global) * bs

    vec = PETSc.Vec().createMPI((local_size, global_size), comm=comm)
    vec.set(float(value))
    vec.assemble()
    return vec


def _owned_dg0_values(field) -> np.ndarray:
    Q = field.function_space
    local_size = int(Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs)
    return np.asarray(field.x.array[:local_size], dtype=float)


def _owned_dg0_cell_measures(Q) -> np.ndarray:
    q = ufl.TestFunction(Q)
    vec = fem_petsc.assemble_vector(fem.form(1.0 * q * ufl.dx))
    try:
        vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        local_size = int(Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs)
        return np.asarray(vec.getArray(readonly=True)[:local_size], dtype=float).copy()
    finally:
        vec.destroy()


def test_density_gradient_uses_newmark_acceleration_not_displacement():
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, 1, 1)
    V = create_vector_lagrange_space(msh, degree=1)

    dt = 0.5

    # Stored snapshots are interpreted as u^1 and u^2.
    # With u^0 = u^{-1} = 0:
    #
    # a^1 = (u^1 - 2u^0 + u^{-1}) / dt^2 = 1 / 0.25 = 4
    # a^2 = (u^2 - 2u^1 + u^0) / dt^2 = (3 - 2) / 0.25 = 4
    #
    # p = 2 in both vector components, hence inner(a,p)=4*2+4*2=16.
    # Time integration over two steps with dt=0.5 gives global integral 16
    # over the unit square.
    u1 = _constant_vec_for_space(V, 1.0)
    u2 = _constant_vec_for_space(V, 3.0)
    p1 = _constant_vec_for_space(V, 2.0)
    p2 = _constant_vec_for_space(V, 2.0)

    try:
        result = compute_material_parameter_correlation_kernels_dg0(
            V=V,
            direct_snapshots=(u1, u2),
            adjoint_snapshots=(p1, p2),
            dt=dt,
            absolute_value=False,
            normalize=False,
        )

        rho_field = result.fields["kernel_rho"]
        rho_values = _owned_dg0_values(rho_field)
        cell_measures = _owned_dg0_cell_measures(rho_field.function_space)
        global_integral = float(
            comm.allreduce(float(np.sum(rho_values * cell_measures)), op=MPI.SUM)
        )

        assert np.allclose(rho_values, 16.0, rtol=1e-12, atol=1e-12)
        assert global_integral == pytest.approx(16.0, rel=1e-12, abs=1e-12)
        assert result.metadata["spatial_representation"] == "DG0_L2_cell_average_density"
        assert result.metadata["cell_measure_normalization"] is True

        for name in ("kernel_C11", "kernel_C22", "kernel_C12", "kernel_C66"):
            values = _owned_dg0_values(result.fields[name])
            assert np.allclose(values, 0.0)

        assert (
            result.metadata["density_kernel_uses"]
            == "newmark_second_difference_acceleration"
        )
        assert result.metadata["stiffness_kernel_uses"] == "strain_interaction"
        assert result.metadata["absolute_value_stage"] == "after_time_accumulation"

    finally:
        for vec in (u1, u2, p1, p2):
            vec.destroy()


def test_absolute_value_is_applied_after_time_integration():
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, 1, 1)
    V = create_vector_lagrange_space(msh, degree=1)

    dt = 1.0

    # u1=1, u2=1 gives accelerations:
    # a1 = 1
    # a2 = 1 - 2*1 = -1
    # With p1=p2=1 in both components, signed time sum cancels.
    # If abs were taken before accumulation, the result would be nonzero.
    u1 = _constant_vec_for_space(V, 1.0)
    u2 = _constant_vec_for_space(V, 1.0)
    p1 = _constant_vec_for_space(V, 1.0)
    p2 = _constant_vec_for_space(V, 1.0)

    try:
        result = compute_material_parameter_correlation_kernels_dg0(
            V=V,
            direct_snapshots=(u1, u2),
            adjoint_snapshots=(p1, p2),
            dt=dt,
            absolute_value=True,
            normalize=False,
        )

        rho_values = _owned_dg0_values(result.fields["kernel_rho"])
        global_sum = float(comm.allreduce(float(np.sum(rho_values)), op=MPI.SUM))

        assert global_sum == pytest.approx(0.0, abs=1e-12)

    finally:
        for vec in (u1, u2, p1, p2):
            vec.destroy()
