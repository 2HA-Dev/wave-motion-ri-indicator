from __future__ import annotations

"""
Validation check of the exact-derivative identity of the residual-interaction
indicator.

The residual-interaction indicator is the exact derivative, in the perturbation
variable, of a linearised least-squares functional with a virtual illuminating
source (Theorem 1 of the accompanying article):

    <r2, F_{lambda1} dm> = - <K^RI, dm>,    hence   grad_{dm} Q(0) = K^RI.

In the proof, lambda1 enters only through the source term of

    A_{m_ref} du = - dA[dm] lambda1,   sigma(du).n = 0,   du(0) = d_t du(0) = 0

so lambda1 is arbitrary. Only lambda2 (terminal conditions) and du (initial
conditions) are constrained. Generating lambda2 by solving A lambda2 = G for a
chosen source G therefore makes <A lambda2, du> = <G, du> exactly known, and
the theorem becomes checkable without any campaign data.

The right-hand side is evaluated with the production kernel routine
compute_material_parameter_correlation_kernels_dg0 (raw signed, unnormalised),
so this check validates the shipped code path, not a parallel reimplementation.

Discrete bookkeeping. The scheme is

    forward  : A u^n - B u^{n-1} + A u^{n-2} = F^n,  u^0 = u^{-1} = 0
    backward : A l^n - B l^{n+1} + A l^{n+2} = G^n,  l^{N+1} = l^{N+2} = 0

whose exact duality gives sum_n <G^n, u^n> = sum_n <l^n, F^n>. With
F^n = dt^2 f^n, f^n = -[M_drho a1^n + K_dC lambda1^n], and the production
kernel already carrying one factor dt, the identity checked here reads

    sum_n <G^n, du^n> = - dt * <K^RI, dm>.

The dt factor is discrete bookkeeping, not physics. The residual tracks the
linear-solver tolerance: with a direct LU solve it falls to machine precision,
which is what establishes that the identity is exact rather than approximate.
"""

import argparse
import json
from typing import Any

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

from elastic_fwi.inversion.material_parameter_gradients import (
    _assemble_dg0_cell_measures,
    _owned_local_size,
    compute_material_parameter_correlation_kernels_dg0,
)
from elastic_fwi.inversion.newmark_adjoint import newmark_middle_operator_action
from elastic_fwi.materials.orthotropic import LEGACY_HEALTHY_MATERIAL
from elastic_fwi.solvers.elastic_operator import (
    ElasticOperatorConfig,
    assemble_elastic_operator_matrices,
    assemble_matrix_from_form,
    create_vector_lagrange_space,
    strain_voigt,
)
from elastic_fwi.solvers.newmark import LinearSolveConfig, solve_effective_system

FIELDS = ("rho", "C11", "C22", "C12", "C66")


def _dg0_fields(Q, values: dict[str, np.ndarray]) -> dict[str, fem.Function]:
    fields = {}
    for name in FIELDS:
        function = fem.Function(Q, name=name)
        function.x.array[: values[name].shape[0]] = values[name]
        function.x.scatter_forward()
        fields[name] = function
    return fields


def _deterministic_vector(template: PETSc.Vec, seed: int) -> PETSc.Vec:
    """
    A reproducible pseudo-random vector, drawn over the global index range so
    the values do not depend on the MPI partition.
    """
    vector = template.duplicate()
    start, end = vector.getOwnershipRange()
    values = np.random.default_rng(seed).normal(size=vector.getSize())
    vector.setValues(list(range(start, end)), values[start:end])
    vector.assemble()
    return vector


def _perturbation_operators(V, Q, dm_values: dict[str, np.ndarray]):
    """
    Assemble the linearisation operators M_drho and K_dC of dA[dm].

    dm is a perturbation, not a medium: drho vanishes outside its compact
    support, so the physical-material validation (rho > 0) enforced by
    mass_form / stiffness_form does not apply. The same UFL expressions are
    therefore instantiated directly here.
    """
    dm_fields = _dg0_fields(Q, dm_values)

    trial = ufl.TrialFunction(V)
    test = ufl.TestFunction(V)

    mass_form = fem.form(dm_fields["rho"] * ufl.inner(trial, test) * ufl.dx)

    e11_u, e22_u, gamma12_u = strain_voigt(trial)
    e11_v, e22_v, gamma12_v = strain_voigt(test)
    stiffness_form = fem.form(
        (
            dm_fields["C11"] * e11_u * e11_v
            + dm_fields["C22"] * e22_u * e22_v
            + dm_fields["C12"] * (e11_u * e22_v + e22_u * e11_v)
            + dm_fields["C66"] * gamma12_u * gamma12_v
        )
        * ufl.dx
    )

    return assemble_matrix_from_form(mass_form), assemble_matrix_from_form(
        stiffness_form
    )


def run_residual_interaction_gradient_identity_check(
    *,
    n_cells: int = 8,
    num_steps: int = 10,
    dt: float = 0.05,
    theta: float = 0.25,
    support_radius: float = 0.25,
    direct_solver: bool = True,
    tolerance: float = 1.0e-12,
    comm: MPI.Comm | None = None,
) -> dict[str, Any]:
    comm = MPI.COMM_WORLD if comm is None else comm

    msh = mesh.create_unit_square(comm, n_cells, n_cells)
    V = create_vector_lagrange_space(msh, degree=1)
    Q = fem.functionspace(msh, ("DG", 0))
    local_size = _owned_local_size(Q)
    cell_measures = _assemble_dg0_cell_measures(Q)

    config = ElasticOperatorConfig(theta=theta, dt=dt)

    reference_values = {
        name: np.full(local_size, getattr(LEGACY_HEALTHY_MATERIAL, name))
        for name in FIELDS
    }
    matrices_ref = assemble_elastic_operator_matrices(
        V, _dg0_fields(Q, reference_values), config
    )

    # Compactly supported perturbation, strictly interior to the domain.
    centroids = Q.tabulate_dof_coordinates()[:local_size]
    radius_squared = (centroids[:, 0] - 0.5) ** 2 + (centroids[:, 1] - 0.5) ** 2
    bump = np.exp(-radius_squared / (2.0 * 0.12**2))
    bump[radius_squared > support_radius**2] = 0.0

    scales = {"rho": 0.30, "C11": 0.20, "C22": 0.17, "C12": 0.11, "C66": 0.23}
    dm_values = {
        name: scales[name] * getattr(LEGACY_HEALTHY_MATERIAL, name) * bump
        for name in FIELDS
    }

    mass_dm, stiffness_dm = _perturbation_operators(V, Q, dm_values)

    solve_config = (
        LinearSolveConfig(ksp_type="preonly", pc_type="lu")
        if direct_solver
        else LinearSolveConfig()
    )

    template = matrices_ref.effective.createVecRight()

    # lambda1 is arbitrary by the theorem; lambda2 is generated from an
    # arbitrary source G so that <A lambda2, du> = <G, du> exactly.
    lambda1 = [_deterministic_vector(template, 100 + n) for n in range(num_steps)]
    g_source = [_deterministic_vector(template, 500 + n) for n in range(num_steps)]

    def _recur(sources, reverse: bool) -> list[PETSc.Vec]:
        states: list[PETSc.Vec] = [None] * num_steps  # type: ignore[list-item]
        near = template.duplicate()
        near.set(0.0)
        far = template.duplicate()
        far.set(0.0)

        order = range(num_steps - 1, -1, -1) if reverse else range(num_steps)
        for step in order:
            rhs = newmark_middle_operator_action(matrices_ref, near)
            tmp = matrices_ref.effective.createVecLeft()
            matrices_ref.effective.mult(far, tmp)
            rhs.axpy(-1.0, tmp)
            rhs.axpy(1.0, sources[step])
            solution, _ = solve_effective_system(
                matrices_ref.effective, rhs, config=solve_config
            )
            states[step] = solution
            far.destroy()
            far = near
            near = solution.copy()
            rhs.destroy()
            tmp.destroy()

        near.destroy()
        far.destroy()
        return states

    lambda2 = _recur(g_source, reverse=True)

    # Tangent source term: f^n = -[ M_drho a1^n + K_dC lambda1^n ], with the same
    # backward second difference the production kernel uses.
    forces: list[PETSc.Vec] = []
    for step in range(num_steps):
        acceleration = lambda1[step].copy()
        acceleration.scale(1.0 / (dt * dt))
        if step >= 1:
            acceleration.axpy(-2.0 / (dt * dt), lambda1[step - 1])
        if step >= 2:
            acceleration.axpy(1.0 / (dt * dt), lambda1[step - 2])

        force = mass_dm.createVecLeft()
        mass_dm.mult(acceleration, force)
        stiffness_action = stiffness_dm.createVecLeft()
        stiffness_dm.mult(lambda1[step], stiffness_action)
        force.axpy(1.0, stiffness_action)
        force.scale(-1.0)
        forces.append(force)

        acceleration.destroy()
        stiffness_action.destroy()

    scaled_forces = []
    for force in forces:
        scaled = force.copy()
        scaled.scale(dt * dt)
        scaled_forces.append(scaled)

    delta_u = _recur(scaled_forces, reverse=False)

    left = sum(float(g_source[n].dot(delta_u[n])) for n in range(num_steps))

    kernels = compute_material_parameter_correlation_kernels_dg0(
        V=V,
        direct_snapshots=lambda1,
        adjoint_snapshots=lambda2,
        dt=dt,
        absolute_value=False,
        normalize=False,
        field_prefix="kernel",
    )

    # DG0/L2 density contract: dJ[dm] = sum_K |K| g_K dm_K.
    pairing_local = 0.0
    for name in FIELDS:
        kernel_values = np.asarray(
            kernels.fields[f"kernel_{name}"].x.array[:local_size], dtype=float
        )
        pairing_local += float(np.sum(cell_measures * kernel_values * dm_values[name]))
    pairing = comm.allreduce(pairing_local, op=MPI.SUM)

    right = -dt * pairing

    absolute = abs(left - right)
    scale = max(abs(left), abs(right), 1.0)
    relative = absolute / scale

    return {
        "kind": "residual_interaction_gradient_identity_check",
        "protocol": "residual-interaction bilinear identity, test 4",
        "identity": "sum_n <G^n, du^n> = - dt * <K^RI, dm>",
        "status": "ok" if relative <= tolerance else "failed",
        "parameters": {
            "n_cells_per_side": n_cells,
            "num_steps": num_steps,
            "dt": dt,
            "theta": theta,
            "support_radius": support_radius,
            "direct_solver": direct_solver,
            "ksp_type": solve_config.ksp_type,
            "pc_type": solve_config.pc_type,
            "tolerance": tolerance,
            "mpi_size": int(comm.size),
        },
        "values": {
            "duality_pairing": left,
            "kernel_contraction": right,
            "kernel_dm_pairing": pairing,
            "absolute_difference": absolute,
            "relative_difference": relative,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check theorem 2: the residual-interaction indicator is the exact "
            "gradient in dm of a linearised least-squares functional."
        )
    )
    parser.add_argument("--n-cells", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--support-radius", type=float, default=0.25)
    parser.add_argument("--iterative-solver", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_residual_interaction_gradient_identity_check(
        n_cells=args.n_cells,
        num_steps=args.num_steps,
        dt=args.dt,
        theta=args.theta,
        support_radius=args.support_radius,
        direct_solver=not args.iterative_solver,
        tolerance=args.tolerance,
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "OK" if report["status"] == "ok" else "FAILED"
            values = report["values"]
            print(
                f"[RESIDUAL INTERACTION GRADIENT IDENTITY] {status} "
                f"lhs={values['duality_pairing']:.12e}, "
                f"rhs={values['kernel_contraction']:.12e}, "
                f"rel={values['relative_difference']:.3e}, "
                f"solver={report['parameters']['pc_type']}, "
                f"MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
