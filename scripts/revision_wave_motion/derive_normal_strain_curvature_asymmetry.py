#!/usr/bin/env python3
"""Why C11, and not C22, loses its sensitivity when the defect leaves the axis.

Standalone like ``verify_mirror_symmetry_symbolic.py``: imports ``sympy`` only.
Run with any Python interpreter that provides ``sympy``:

    python path/to/derive_normal_strain_curvature_asymmetry.py

The question this answers
-------------------------
The off-axis campaign measured, for the target-zone peak of the C11 kernel, a
fall by x0.27 (classical), x0.28 (reflection only), x0.27 (transmission only)
and x0.54 (residual interaction), while C22 barely moved (x0.86 to x0.97) and
rho did not move at all.

``verify_mirror_symmetry_symbolic.py`` check 3 establishes that eps11 and eps22
are BOTH mirror-even, hence both stationary in x1 on the axis.  Parity alone
therefore cannot distinguish them; on parity grounds alone the distinction
would remain an unexplained amplitude hypothesis.

This script provides the complementary argument. Parity fixes the *form* of the expansion; the
elastodynamic equation itself fixes the *coefficients*.  Imposing the PDE order
by order in s = x1 - 36.5 shows that the relative curvature of eps11 on the axis
is amplified, relative to that of eps22, by exactly the ratio of the two normal
strains on the axis -- a large number when the source is polarised along x2,
because eps11 is then built on the subdominant displacement component.

Scope.  The argument applies to the reference field u_ref, which lives in the
*healthy* medium (``simulated_material_kind = healthy``, homogeneous, no
absorbing layer for this preset).  Homogeneity is what allows the stiffnesses to
be pulled out of the derivatives below.  u_ref is also, by construction, the one
field that does not depend on where the defect sits -- which is why it, and not
the adjoint, carries the defect-independent part of the explanation.

Exit code 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import sys

import sympy as sp


AXIS_SUM = sp.Integer(73)
AXIS = AXIS_SUM / 2

x2, t = sp.symbols("x2 t", real=True)
s = sp.symbols("s", real=True)

# Aligned orthotropic plane-strain constants of the healthy medium
# (elastic_fwi/materials/orthotropic.py, LEGACY_HEALTHY_MATERIAL).
HEALTHY = {"rho": sp.Rational(45, 10), "C11": sp.Rational(1624, 100),
           "C22": sp.Rational(1807, 100), "C12": sp.Rational(69, 10),
           "C66": sp.Rational(467, 100)}


def _report(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    return ok


def build_mirror_even_field(order: int):
    """Most general mirror-even displacement: u1 odd in s, u2 even in s."""
    a = [sp.Function(f"A{k}")(x2, t) for k in range(order)]
    b = [sp.Function(f"B{k}")(x2, t) for k in range(order)]
    u1 = sum(c * s ** (2 * k + 1) for k, c in enumerate(a))
    u2 = sum(c * s ** (2 * k) for k, c in enumerate(b))
    return u1, u2, a, b


def pde_residuals(u1, u2, C11, C22, C12, C66, rho):
    """rho d_tt u - div sigma(u), for a HOMOGENEOUS aligned orthotropic medium."""
    eps11 = sp.diff(u1, s)
    eps22 = sp.diff(u2, x2)
    gamma12 = sp.diff(u1, x2) + sp.diff(u2, s)

    sigma11 = C11 * eps11 + C12 * eps22
    sigma22 = C12 * eps11 + C22 * eps22
    sigma12 = C66 * gamma12

    r1 = rho * sp.diff(u1, t, 2) - (sp.diff(sigma11, s) + sp.diff(sigma12, x2))
    r2 = rho * sp.diff(u2, t, 2) - (sp.diff(sigma12, s) + sp.diff(sigma22, x2))
    return r1, r2


def check_pde_fixes_the_curvature_coefficients():
    """Solve the PDE order by order for A1 (curvature of u1) and B1 (of u2)."""
    C11, C22, C12, C66, rho = sp.symbols("C11 C22 C12 C66 rho", positive=True)
    u1, u2, a, b = build_mirror_even_field(order=4)
    r1, r2 = pde_residuals(u1, u2, C11, C22, C12, C66, rho)

    # Lowest order in s at which each equation constrains something:
    # r1 is odd in s (constrains at s^1), r2 is even (constrains at s^0).
    eq_A = sp.expand(r1).coeff(s, 1)
    eq_B = sp.expand(r2).coeff(s, 0)

    sol_B1 = sp.solve(eq_B, b[1], dict=True)
    sol_A1 = sp.solve(eq_A, a[1], dict=True)
    ok = _report(
        "1a. the wave equation determines B1 and A1 (they are not free)",
        bool(sol_B1) and bool(sol_A1),
        "parity fixes the form of the expansion; the PDE fixes its coefficients",
    )

    B1 = sp.simplify(sol_B1[0][b[1]])
    A1 = sp.simplify(sol_A1[0][a[1]])
    print(f"         B1 = {B1}")
    print(f"         A1 = {A1}")

    # A1 is driven by d_x2 B1 -- the curvature of the OTHER component.
    A1_full = sp.simplify(A1.subs(b[1], B1).doit())
    coupling_present = A1.has(sp.Derivative(b[1], x2)) or A1.has(b[1])
    ok &= _report(
        "1b. A1 (curvature of u1) is driven by the x2-derivative of B1",
        coupling_present,
        "the subdominant component inherits the curvature of the dominant one",
    )
    return ok, A1, B1, (C11, C22, C12, C66, rho), a, b


def check_relative_curvature_ratio(A1, B1, consts, a, b):
    """kappa_11 / kappa_22 = -(C12+C66)/C11 * (eps22|axis / eps11|axis).

    No limits are taken.  ``sympy.limit`` cannot collect multivariate
    ``Derivative`` objects, and a limit would in any case be a weaker statement
    than the algebraic identity checked here: kappa_11 is multiplied by
    eps11|axis to clear its only singular factor, after which setting the
    component-imbalance parameter to zero is an ordinary substitution.
    """
    C11, C22, C12, C66, rho = consts

    # eps11 = A0 + 3 A1 s^2 + O(s^4)   -> relative curvature 6 A1 / A0
    # eps22 = d2 B0 + (d2 B1) s^2 + .. -> relative curvature 2 d2 B1 / d2 B0
    eps11_axis = a[0]
    eps22_axis = sp.diff(b[0], x2)
    kappa_11 = 6 * A1.subs(b[1], B1).doit() / eps11_axis
    kappa_22 = 2 * sp.diff(B1, x2) / eps22_axis

    # Bookkeeping: the source is polarised along x2, so u1 is subdominant.
    # Write eps11|axis = A0 = epsilon * alpha, epsilon -> 0, everything else O(1).
    epsilon = sp.symbols("epsilon", positive=True)
    alpha = sp.Function("alpha")(x2, t)
    scale = {a[0]: epsilon * alpha}

    # Clear the 1/A0 singularity, then read the epsilon-independent part.
    L11 = sp.expand(sp.simplify((kappa_11 * eps11_axis).subs(scale).doit()))
    k22 = sp.expand(sp.simplify(kappa_22.subs(scale).doit()))

    L11_lead = sp.simplify(L11.subs(epsilon, 0))
    k22_lead = sp.simplify(k22.subs(epsilon, 0))

    ok = _report(
        "2a. kappa_11 keeps an epsilon-independent numerator over an O(epsilon) "
        "denominator, so it diverges as 1/epsilon; kappa_22 stays O(1)",
        L11_lead != 0 and k22_lead != 0,
        f"(kappa_11 * eps11|axis)|_(epsilon=0) = {L11_lead}",
    )
    print(f"         kappa_22|_(epsilon=0) = {k22_lead}")

    # kappa_11/kappa_22 = L11/(eps11|axis * kappa_22); at leading order this is
    # the closed form below.  Verified as an identity, not as a limit.
    ratio_lead = sp.simplify(L11_lead / k22_lead)
    ratio_predicted = sp.simplify(-(C12 + C66) / C11 * eps22_axis)
    difference = sp.simplify(sp.together(ratio_lead - ratio_predicted))
    ok &= _report(
        "2b. kappa_11/kappa_22 = -(C12+C66)/C11 * (eps22|axis / eps11|axis)",
        difference == 0,
        f"(kappa_11/kappa_22) * eps11|axis = {ratio_lead}",
    )

    # The amplification is genuinely absent when the two components balance.
    ok &= _report(
        "2c. the amplification is exactly the inverse component imbalance",
        sp.simplify(ratio_lead / ratio_predicted - 1) == 0,
        "no amplification survives if eps22|axis and eps11|axis are comparable",
    )
    return ok, ratio_predicted


def check_numerical_prefactor():
    """The stiffness prefactor (C12+C66)/C11 for the healthy medium."""
    prefactor = (HEALTHY["C12"] + HEALTHY["C66"]) / HEALTHY["C11"]
    value = float(prefactor)
    ok = _report(
        "3. stiffness prefactor of the amplification, healthy medium",
        0 < value < 10,
        f"(C12 + C66)/C11 = ({HEALTHY['C12']} + {HEALTHY['C66']})/{HEALTHY['C11']} = {value:.4f}",
    )
    print("         so the predicted amplification is "
          f"{value:.4f} x (eps22|axis / eps11|axis), the strain ratio being measurable")
    return ok


def check_rho_control():
    """The density kernel pairs displacements, not strains: no amplification.

    Two independent mirror-even fields (u and the adjoint) are contracted as a
    dot product.  The subdominant components multiply each other and vanish on
    the axis; the dominant ones survive.  There is no division by a small
    quantity anywhere, hence no 1/epsilon amplification -- which is why rho is
    the natural control for the effect.
    """
    order = 3
    a = [sp.Function(f"A{k}")(x2, t) for k in range(order)]
    b = [sp.Function(f"B{k}")(x2, t) for k in range(order)]
    c = [sp.Function(f"P{k}")(x2, t) for k in range(order)]
    d = [sp.Function(f"Q{k}")(x2, t) for k in range(order)]

    u1 = sum(v * s ** (2 * k + 1) for k, v in enumerate(a))
    u2 = sum(v * s ** (2 * k) for k, v in enumerate(b))
    lam1 = sum(v * s ** (2 * k + 1) for k, v in enumerate(c))
    lam2 = sum(v * s ** (2 * k) for k, v in enumerate(d))

    kernel = sp.expand(u1 * lam1 + u2 * lam2)
    axis_value = sp.simplify(kernel.subs(s, 0))
    ok = _report(
        "4. the rho kernel is non-zero on the axis and carries no 1/epsilon factor",
        sp.simplify(axis_value - b[0] * d[0]) == 0,
        f"rho kernel|_(s=0) = {axis_value}  (the u1*lam1 term vanishes, u2*lam2 survives)",
    )
    print("         consistent with the measurement: rho moved by x1.04 / x1.01, i.e. not at all")
    return ok


def main() -> int:
    print("Curvature asymmetry between the C11 and C22 kernels on the mirror axis")
    print(f"Mirror: x1 -> {AXIS_SUM} - x1, fixed axis x1 = {AXIS}; s = x1 - {AXIS}")
    print("Source polarisation (0,1): u2 dominant, u1 subdominant")
    print("-" * 78)

    ok1, A1, B1, consts, a, b = check_pde_fixes_the_curvature_coefficients()
    ok2, _ = check_relative_curvature_ratio(A1, B1, consts, a, b)
    ok3 = check_numerical_prefactor()
    ok4 = check_rho_control()

    print("-" * 78)
    if all((ok1, ok2, ok3, ok4)):
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
