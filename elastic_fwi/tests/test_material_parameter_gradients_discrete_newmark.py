from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import petsc as fem_petsc

from elastic_fwi.diagnostics.check_material_gradient_directional import (
    run_material_gradient_directional_check,
)
from elastic_fwi.inversion.material_directions import (
    HomogeneousMaterialDirection,
    assemble_homogeneous_material_direction_matrices,
)
from elastic_fwi.inversion.material_gradient import (
    compute_newmark_material_directional_gradient,
)
from elastic_fwi.inversion.material_parameter_gradients import (
    compute_material_parameter_gradients_dg0,
)
from elastic_fwi.pipelines.adjoint_newmark import (
    AdjointNewmarkPipelineConfig,
    run_adjoint_newmark_pipeline,
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
        return np.asarray(
            vec.getArray(readonly=True)[:local_size], dtype=float
        ).copy()
    finally:
        vec.destroy()


def test_density_discrete_gradient_has_newmark_global_sign_and_no_extra_dt():
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, 1, 1)
    V = create_vector_lagrange_space(msh, degree=1)

    dt = 0.5
    theta = 0.25

    u1 = _constant_vec_for_space(V, 1.0)
    u2 = _constant_vec_for_space(V, 3.0)
    p1 = _constant_vec_for_space(V, 2.0)
    p2 = _constant_vec_for_space(V, 2.0)

    try:
        result = compute_material_parameter_gradients_dg0(
            V=V,
            direct_snapshots=(u1, u2),
            adjoint_snapshots=(p1, p2),
            dt=dt,
            theta=theta,
            absolute_value=False,
            normalize=False,
        )

        rho_field = result.fields["gradient_rho"]
        rho_values = _owned_dg0_values(rho_field)
        cell_measures = _owned_dg0_cell_measures(rho_field.function_space)
        global_integral = float(
            comm.allreduce(
                float(np.sum(rho_values * cell_measures)),
                op=MPI.SUM,
            )
        )

        # For each stored step, 2 u^n - u^{n+1} - u^{n-1} = -1.
        # With p=(2,2), the density is -4 per step, hence -8 after
        # two steps. Its area integral over the unit square is also -8.
        assert np.allclose(rho_values, -8.0, rtol=1e-12, atol=1e-12)
        assert global_integral == pytest.approx(-8.0, rel=1e-12, abs=1e-12)
        assert result.metadata["spatial_representation"] == (
            "DG0_L2_cell_average_density"
        )
        assert result.metadata["cell_measure_normalization"] is True

        assert result.metadata["kind"] == "material_parameter_discrete_gradients_dg0"
        assert result.metadata["is_true_discrete_gradient"] is True
        assert (
            result.metadata["mathematical_role"]
            == "fd_validated_discrete_newmark_gradient_density"
        )

    finally:
        for vec in (u1, u2, p1, p2):
            vec.destroy()


def test_dg0_raw_gradient_matches_scalar_directional_gradient_for_homogeneous_direction():
    comm = MPI.COMM_WORLD
    pipeline = None
    perturbation = None

    try:
        pipeline = run_adjoint_newmark_pipeline(
            AdjointNewmarkPipelineConfig(
                nx=4,
                ny=4,
                degree=1,
                theta=0.25,
                dt=0.1,
                num_steps=5,
                simulated_material_kind="healthy",
                observed_material_kind="localized_disk_defect",
                contrast_factor=0.8,
                defect_center_x=0.5,
                defect_center_y=0.5,
                defect_radius=0.35,
                ksp_type="cg",
                pc_type="jacobi",
                ksp_rtol=1.0e-10,
                ksp_atol=1.0e-12,
                ksp_max_it=1000,
            ),
            comm=comm,
        )

        V = pipeline.comparison.simulated.function_space
        operator_config = pipeline.comparison.simulated.operator_config
        assert V is not None
        assert operator_config is not None

        direction = HomogeneousMaterialDirection(
            rho=-0.1,
            C11=-0.1,
            C22=0.2,
            C12=-0.03,
            C66=0.04,
        )

        perturbation = assemble_homogeneous_material_direction_matrices(V, direction)

        scalar_gradient = compute_newmark_material_directional_gradient(
            perturbation.mass,
            perturbation.stiffness,
            operator_config,
            direct_snapshots=pipeline.comparison.simulated.displacement_snapshots,
            equation_adjoint_vectors=pipeline.newmark_adjoint.equation_adjoint_vectors,
        )

        dg0_result = compute_material_parameter_gradients_dg0(
            V=V,
            direct_snapshots=pipeline.comparison.simulated.displacement_snapshots,
            adjoint_snapshots=pipeline.newmark_adjoint.equation_adjoint_vectors,
            dt=operator_config.dt,
            theta=operator_config.theta,
            absolute_value=False,
            normalize=False,
        )

        raw_by_parameter = {}
        for name, field in dg0_result.fields.items():
            values = _owned_dg0_values(field)
            measures = _owned_dg0_cell_measures(field.function_space)
            raw_by_parameter[name.removeprefix("gradient_")] = float(
                comm.allreduce(
                    float(np.sum(values * measures)),
                    op=MPI.SUM,
                )
            )

        dg0_contraction = sum(
            raw_by_parameter[key] * value
            for key, value in direction.as_dict().items()
        )

        assert dg0_contraction == pytest.approx(
            scalar_gradient.value,
            rel=1e-9,
            abs=1e-11,
        )

    finally:
        if perturbation is not None:
            perturbation.destroy()
        if pipeline is not None:
            pipeline.destroy()
