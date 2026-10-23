from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc


REQUIRED_MATERIAL_FIELDS = ("rho", "C11", "C22", "C12", "C66")


@dataclass(frozen=True)
class ElasticOperatorConfig:
    """
    Configuration for the semi-discrete elastodynamic operator.

    The effective Newmark-type matrix is

        A = M + theta * dt**2 * K,

    to which effective_operator_form adds 0.5 * dt * C when a damping
    coefficient is supplied, C being the damping matrix
    rho * sigma * inner(u, v). The article presets set
    absorbing_layer = False and never take that branch; the default preset
    of run_gradient_combination_campaign.py does.
    """

    theta: float = 0.25
    dt: float = 1.0

    def __post_init__(self) -> None:
        theta = float(self.theta)
        dt = float(self.dt)

        if not isfinite(theta):
            raise ValueError("theta must be finite.")

        if not isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be strictly positive and finite.")

        if theta < 0.0:
            raise ValueError("theta must be non-negative.")

        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "dt", dt)


@dataclass
class ElasticOperatorMatrices:
    """
    Assembled PETSc matrices for the linear elastic wave operator.

    keepalive stores the FEniCSx objects from which the matrices were assembled.
    This avoids PETSc invalid-communicator errors when matrices are used after a
    local factory function returns.
    """

    mass: PETSc.Mat
    stiffness: PETSc.Mat
    effective: PETSc.Mat
    config: ElasticOperatorConfig
    damping: PETSc.Mat | None = None
    keepalive: tuple[object, ...] = ()

    def destroy(self) -> None:
        for matrix in (self.mass, self.stiffness, self.effective, self.damping):
            try:
                if matrix is not None:
                    matrix.destroy()
            except Exception:
                pass

        self.keepalive = ()


def create_vector_lagrange_space(msh, degree: int = 1):
    if degree <= 0:
        raise ValueError("The Lagrange degree must be strictly positive.")

    return fem.functionspace(msh, ("Lagrange", degree, (msh.geometry.dim,)))


def _function_space_comm(function_space) -> MPI.Comm:
    return function_space.mesh.comm


def _global_bool(comm: MPI.Comm, local_value: bool) -> bool:
    return bool(comm.allreduce(bool(local_value), op=MPI.LOR))


def validate_material_fields(materials: dict[str, object]) -> None:
    """
    Validate DG0 material fields.

    Beyond finiteness and a strictly positive density, the orthotropic Voigt
    matrix [[C11, C12, 0], [C12, C22, 0], [0, 0, C66]] is checked for positive
    definiteness via Sylvester's criterion on its block-diagonal structure:
    C11 > 0, C11*C22 - C12**2 > 0, C66 > 0.

    The validation is MPI-global: if one rank detects a violation, all ranks
    raise the same ValueError. This avoids rank-dependent behavior when some
    ranks own no DG0 entries.
    """

    missing = [name for name in REQUIRED_MATERIAL_FIELDS if name not in materials]

    if missing:
        raise ValueError(f"Missing material fields: {missing}.")

    reference_space = materials["rho"].function_space
    comm = _function_space_comm(reference_space)

    for name in REQUIRED_MATERIAL_FIELDS:
        field = materials[name]

        if field.function_space is not reference_space:
            raise ValueError("All material fields must share the same DG0 function space.")

        values = np.asarray(field.x.array, dtype=float)
        local_non_finite = not np.all(np.isfinite(values))
        global_non_finite = _global_bool(comm, local_non_finite)

        if global_non_finite:
            raise ValueError(f"Material field {name} contains non-finite values.")

    rho_values = np.asarray(materials["rho"].x.array, dtype=float)
    local_non_positive_rho = bool(np.any(rho_values <= 0.0))
    global_non_positive_rho = _global_bool(comm, local_non_positive_rho)

    if global_non_positive_rho:
        raise ValueError("Density field rho must be strictly positive.")

    C11_values = np.asarray(materials["C11"].x.array, dtype=float)
    C22_values = np.asarray(materials["C22"].x.array, dtype=float)
    C12_values = np.asarray(materials["C12"].x.array, dtype=float)
    C66_values = np.asarray(materials["C66"].x.array, dtype=float)

    local_non_positive_C11 = bool(np.any(C11_values <= 0.0))
    global_non_positive_C11 = _global_bool(comm, local_non_positive_C11)

    if global_non_positive_C11:
        raise ValueError("Stiffness field C11 must be strictly positive.")

    local_non_positive_C66 = bool(np.any(C66_values <= 0.0))
    global_non_positive_C66 = _global_bool(comm, local_non_positive_C66)

    if global_non_positive_C66:
        raise ValueError("Stiffness field C66 must be strictly positive.")

    minor2 = C11_values * C22_values - C12_values**2
    local_non_positive_minor = bool(np.any(minor2 <= 0.0))
    global_non_positive_minor = _global_bool(comm, local_non_positive_minor)

    if global_non_positive_minor:
        raise ValueError(
            "Voigt matrix is not positive definite: "
            "C11*C22 - C12^2 must be strictly positive."
        )


def strain_voigt(u) -> tuple[object, object, object]:
    """
    Return engineering Voigt components:

        e11     = du_x/dx
        e22     = du_y/dy
        gamma12 = du_x/dy + du_y/dx
    """

    grad_u = ufl.grad(u)

    e11 = grad_u[0, 0]
    e22 = grad_u[1, 1]
    gamma12 = grad_u[0, 1] + grad_u[1, 0]

    return e11, e22, gamma12


def mass_form(V, materials: dict[str, object]):
    validate_material_fields(materials)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    rho = materials["rho"]

    return fem.form(rho * ufl.inner(u, v) * ufl.dx)


def stiffness_form(V, materials: dict[str, object]):
    validate_material_fields(materials)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    C11 = materials["C11"]
    C22 = materials["C22"]
    C12 = materials["C12"]
    C66 = materials["C66"]

    e11_u, e22_u, gamma12_u = strain_voigt(u)
    e11_v, e22_v, gamma12_v = strain_voigt(v)

    density = (
        C11 * e11_u * e11_v
        + C22 * e22_u * e22_v
        + C12 * (e11_u * e22_v + e22_u * e11_v)
        + C66 * gamma12_u * gamma12_v
    )

    return fem.form(density * ufl.dx)


def damping_form(V, materials: dict[str, object], damping_coefficient):
    validate_material_fields(materials)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    rho = materials["rho"]

    return fem.form(rho * damping_coefficient * ufl.inner(u, v) * ufl.dx)


def effective_operator_form(
    V,
    materials: dict[str, object],
    config: ElasticOperatorConfig,
    damping_coefficient=None,
):
    validate_material_fields(materials)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    rho = materials["rho"]
    C11 = materials["C11"]
    C22 = materials["C22"]
    C12 = materials["C12"]
    C66 = materials["C66"]

    e11_u, e22_u, gamma12_u = strain_voigt(u)
    e11_v, e22_v, gamma12_v = strain_voigt(v)

    mass_density = rho * ufl.inner(u, v)

    stiffness_density = (
        C11 * e11_u * e11_v
        + C22 * e22_u * e22_v
        + C12 * (e11_u * e22_v + e22_u * e11_v)
        + C66 * gamma12_u * gamma12_v
    )

    effective_density = mass_density + config.theta * config.dt**2 * stiffness_density

    if damping_coefficient is not None:
        effective_density = effective_density + 0.5 * config.dt * rho * damping_coefficient * ufl.inner(u, v)

    return fem.form(effective_density * ufl.dx)


def assemble_matrix_from_form(form) -> PETSc.Mat:
    matrix = fem_petsc.assemble_matrix(form)
    matrix.assemble()
    return matrix


def assemble_elastic_operator_matrices(
    V,
    materials: dict[str, object],
    config: ElasticOperatorConfig | None = None,
    damping_coefficient=None,
) -> ElasticOperatorMatrices:
    """
    Assemble M, K, the effective matrix A and, on request, the damping matrix C.

    With damping_coefficient=None, A = M + theta * dt**2 * K and no damping
    matrix is built. When a damping coefficient is supplied, the damping matrix
    C = rho * sigma * inner(u, v) is assembled as well and

        A = M + theta * dt**2 * K + 0.5 * dt * C,

    the matching 0.5 * dt * C * u_previous term being applied to the right-hand
    side by build_newmark_theta_rhs.
    """

    if config is None:
        config = ElasticOperatorConfig()

    mass = assemble_matrix_from_form(mass_form(V, materials))
    stiffness = assemble_matrix_from_form(stiffness_form(V, materials))

    damping = None
    if damping_coefficient is not None:
        damping = assemble_matrix_from_form(
            damping_form(V, materials, damping_coefficient)
        )

    effective = assemble_matrix_from_form(
        effective_operator_form(
            V,
            materials,
            config,
            damping_coefficient=damping_coefficient,
        )
    )

    keepalive = (V, materials)
    if damping_coefficient is not None:
        keepalive = (V, materials, damping_coefficient)

    return ElasticOperatorMatrices(
        mass=mass,
        stiffness=stiffness,
        damping=damping,
        effective=effective,
        config=config,
        keepalive=keepalive,
    )


def matrix_global_size(matrix: PETSc.Mat) -> tuple[int, int]:
    rows, cols = matrix.getSize()
    return int(rows), int(cols)


def matrix_frobenius_norm(matrix: PETSc.Mat) -> float:
    return float(matrix.norm(PETSc.NormType.FROBENIUS))


def matrix_action_norm(matrix: PETSc.Mat, value: float = 1.0) -> float:
    x = matrix.createVecRight()
    y = matrix.createVecLeft()

    x.set(value)
    matrix.mult(x, y)

    norm = float(y.norm())

    x.destroy()
    y.destroy()

    return norm
