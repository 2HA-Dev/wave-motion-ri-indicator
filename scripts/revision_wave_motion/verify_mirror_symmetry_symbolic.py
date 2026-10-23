#!/usr/bin/env python3
"""Symbolic verification of the mirror-symmetry theorem for the C66 kernels.

This script is deliberately standalone: it imports ``sympy`` only and nothing
from ``elastic_fwi``, so it does not need the FEniCSx environment and runs
under any Python interpreter that provides ``sympy``:

    python path/to/verify_mirror_symmetry_symbolic.py

What is verified, in the order the proof needs it:

1.  The aligned orthotropic plane-strain stiffness tensor is invariant under
    the mirror ``M = diag(-1, 1)``, and this invariance is *equivalent* to the
    vanishing of the two monoclinic couplings ``C16`` and ``C26``.
2.  The elastodynamic operator is equivariant: pushing a displacement field
    through the mirror and pushing the residual of the momentum equation
    through the mirror commute.
3.  A mirror-even displacement field has odd ``gamma12`` and even ``eps11``,
    ``eps22``; consequently ``gamma12`` vanishes identically on the mirror axis
    while ``eps11`` and ``eps22`` do not.
4.  A product of two such shear strains --- which is exactly the structure of
    the C66 kernels, classical and residual-interaction alike --- is mirror
    *even* and vanishes to second order on the axis.  It therefore does **not**
    change sign across the axis.

In particular, point 4 rules out a forced sign change across the axis: the
product of two odd factors is even.

6.  The symmetry-reversal prediction, derived symbolically, independently of
    the campaign that tests it.  A
    purely horizontal source polarisation ``p = (1, 0)`` is mirror-ANTIsymmetric
    (``M p = -p``) rather than merely non-symmetric.  Every wavefield then lands
    in the opposite parity class, and the theorem **reverses** term by term:
    ``eps11`` and ``eps22`` become odd and vanish on the axis, ``gamma12``
    becomes even and does not, so the double zero moves from the C66 kernel to
    the C11, C22 and C12 kernels.  The density kernel, a dot product of two
    fields of the same class, is non-zero on the axis in *both* classes and
    therefore acts as a control: it must not move.

Exit code is 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import sys

import sympy as sp


# The mirror axis of the `line` and `box` configurations: x1 -> AXIS_SUM - x1,
# with AXIS_SUM = 73 so that the fixed line is x1 = 36.5, the defect centre.
AXIS_SUM = sp.Integer(73)
AXIS = AXIS_SUM / 2

x1, x2, t = sp.symbols("x1 x2 t", real=True)


def _report(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    return ok


# ---------------------------------------------------------------------------
# 1. Invariance of the stiffness tensor
# ---------------------------------------------------------------------------


def _voigt_to_tensor(C11, C22, C12, C66, C16, C26):
    """Build the 2-D fourth-order stiffness from plane-strain Voigt entries."""

    # Voigt pairs: 1 <-> (0,0), 2 <-> (1,1), 6 <-> (0,1)=(1,0).
    voigt = {
        (0, 0): {(0, 0): C11, (1, 1): C12, (0, 1): C16, (1, 0): C16},
        (1, 1): {(0, 0): C12, (1, 1): C22, (0, 1): C26, (1, 0): C26},
        (0, 1): {(0, 0): C16, (1, 1): C26, (0, 1): C66, (1, 0): C66},
        (1, 0): {(0, 0): C16, (1, 1): C26, (0, 1): C66, (1, 0): C66},
    }
    return lambda i, j, k, l: voigt[(i, j)][(k, l)]


def check_stiffness_invariance() -> bool:
    C11, C22, C12, C66, C16, C26 = sp.symbols("C11 C22 C12 C66 C16 C26", real=True)
    C = _voigt_to_tensor(C11, C22, C12, C66, C16, C26)

    M = sp.diag(-1, 1)

    residuals = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    pushed = sum(
                        M[i, p] * M[j, q] * M[k, r] * M[l, s] * C(p, q, r, s)
                        for p in range(2)
                        for q in range(2)
                        for r in range(2)
                        for s in range(2)
                    )
                    diff = sp.simplify(pushed - C(i, j, k, l))
                    if diff != 0:
                        residuals[(i, j, k, l)] = diff

    # The residual must be a combination of C16 and C26 only, and must vanish
    # identically once those two are set to zero.
    aligned_ok = all(
        sp.simplify(diff.subs({C16: 0, C26: 0})) == 0 for diff in residuals.values()
    )
    ok = _report(
        "1a. aligned orthotropic stiffness is mirror-invariant (C16 = C26 = 0)",
        aligned_ok,
        f"{len(residuals)} tensor components constrain the monoclinic couplings",
    )

    # Equivalence: the invariance genuinely forces C16 = C26 = 0, so the
    # hypothesis is sharp rather than merely sufficient.
    solution = sp.solve(list(residuals.values()), [C16, C26], dict=True)
    forced_ok = bool(solution) and all(
        sp.simplify(sol.get(C16, C16)) == 0 and sp.simplify(sol.get(C26, C26)) == 0
        for sol in solution
    )
    ok &= _report(
        "1b. mirror invariance forces C16 = C26 = 0 (hypothesis is sharp)",
        forced_ok,
        f"solve -> {solution}",
    )
    return ok


# ---------------------------------------------------------------------------
# 2. Equivariance of the elastodynamic operator
# ---------------------------------------------------------------------------


def _strain(u1e, u2e):
    e11 = sp.diff(u1e, x1)
    e22 = sp.diff(u2e, x2)
    g12 = sp.diff(u1e, x2) + sp.diff(u2e, x1)
    return e11, e22, g12


def _divergence_of_stress(u1e, u2e, C11, C22, C12, C66):
    """Return (div sigma)_1, (div sigma)_2 for the aligned orthotropic law."""

    e11, e22, g12 = _strain(u1e, u2e)
    s11 = C11 * e11 + C12 * e22
    s22 = C12 * e11 + C22 * e22
    s12 = C66 * g12
    d1 = sp.diff(s11, x1) + sp.diff(s12, x2)
    d2 = sp.diff(s12, x1) + sp.diff(s22, x2)
    return d1, d2


def check_operator_equivariance() -> bool:
    C11, C22, C12, C66, rho = sp.symbols("C11 C22 C12 C66 rho", real=True, positive=True)

    u1 = sp.Function("u1")(x1, x2, t)
    u2 = sp.Function("u2")(x1, x2, t)

    # Mirror push-forward of a vector field: (M*u)(x) = M u(Mx).
    mu1 = -u1.subs(x1, AXIS_SUM - x1)
    mu2 = u2.subs(x1, AXIS_SUM - x1)

    # Momentum residual L[u] = rho d_tt u - div sigma(u).
    d1, d2 = _divergence_of_stress(u1, u2, C11, C22, C12, C66)
    L1 = rho * sp.diff(u1, t, 2) - d1
    L2 = rho * sp.diff(u2, t, 2) - d2

    # Left-hand side: operator applied to the mirrored field.
    md1, md2 = _divergence_of_stress(mu1, mu2, C11, C22, C12, C66)
    lhs1 = rho * sp.diff(mu1, t, 2) - md1
    lhs2 = rho * sp.diff(mu2, t, 2) - md2

    # Right-hand side: mirror push-forward of the operator residual.
    rhs1 = -L1.subs(x1, AXIS_SUM - x1)
    rhs2 = L2.subs(x1, AXIS_SUM - x1)

    ok1 = sp.simplify(sp.expand(lhs1 - rhs1)) == 0
    ok2 = sp.simplify(sp.expand(lhs2 - rhs2)) == 0

    return _report(
        "2. elastodynamic operator commutes with the mirror push-forward",
        ok1 and ok2,
        "L[M*u] = M*(L[u]) componentwise, for arbitrary u",
    )


# ---------------------------------------------------------------------------
# 3. Strain parities and vanishing on the axis
# ---------------------------------------------------------------------------


def check_strain_parities() -> bool:
    """A mirror-even field has odd gamma12 and even eps11, eps22."""

    # Work in the shifted coordinate s = x1 - AXIS, so that the mirror is the
    # plain reflection s -> -s and d/dx1 = d/ds.  A mirror-even displacement is
    # exactly one whose first component is odd in s and second component even;
    # the truncated parity-respecting series below is the general such form to
    # the order needed to read off values and first derivatives at s = 0.
    s = sp.symbols("s", real=True)

    order = 6
    a_coeffs = sp.symbols(f"a1:{order}", real=True)  # odd series in s
    b_coeffs = sp.symbols(f"b0:{order}", real=True)  # even series in s

    A_series = sum(c * s ** (2 * k + 1) for k, c in enumerate(a_coeffs))
    B_series = sum(c * s ** (2 * k) for k, c in enumerate(b_coeffs))

    ok = _report(
        "3a. odd/even series representation respects the parity identities",
        sp.simplify(A_series.subs(s, -s) + A_series) == 0
        and sp.simplify(B_series.subs(s, -s) - B_series) == 0,
    )

    # Let the coefficients depend on x2 and t, which is the general form of
    # a mirror-even displacement field.
    a_funcs = [sp.Function(f"A{k}")(x2, t) for k in range(len(a_coeffs))]
    b_funcs = [sp.Function(f"B{k}")(x2, t) for k in range(len(b_coeffs))]
    A_gen = sum(c * s ** (2 * k + 1) for k, c in enumerate(a_funcs))
    B_gen = sum(c * s ** (2 * k) for k, c in enumerate(b_funcs))

    g12_gen = sp.diff(A_gen, x2) + sp.diff(B_gen, s)
    e11_gen = sp.diff(A_gen, s)
    e22_gen = sp.diff(B_gen, x2)

    g12_axis = sp.simplify(g12_gen.subs(s, 0))
    e11_axis = sp.simplify(e11_gen.subs(s, 0))
    e22_axis = sp.simplify(e22_gen.subs(s, 0))

    ok &= _report(
        "3b. gamma12 vanishes identically on the mirror axis",
        g12_axis == 0,
        f"gamma12|_(s=0) = {g12_axis}",
    )
    ok &= _report(
        "3c. eps11 and eps22 do not vanish on the axis (generically non-zero)",
        e11_axis != 0 and e22_axis != 0,
        f"eps11|_(s=0) = {e11_axis}, eps22|_(s=0) = {e22_axis}",
    )

    # Parity of the strains as functions of s.
    g12_parity_ok = sp.simplify(g12_gen.subs(s, -s) + g12_gen) == 0
    e11_parity_ok = sp.simplify(e11_gen.subs(s, -s) - e11_gen) == 0
    e22_parity_ok = sp.simplify(e22_gen.subs(s, -s) - e22_gen) == 0
    ok &= _report(
        "3d. gamma12 is odd in s while eps11, eps22 are even",
        g12_parity_ok and e11_parity_ok and e22_parity_ok,
    )

    # gamma12 vanishes to first order only: its s-derivative on the axis is the
    # generically non-zero quantity that controls the quadratic kernel below.
    dg12_axis = sp.simplify(sp.diff(g12_gen, s).subs(s, 0))
    ok &= _report(
        "3e. gamma12 vanishes linearly, not to higher order",
        dg12_axis != 0,
        f"d_s gamma12|_(s=0) = {dg12_axis}",
    )
    return ok


# ---------------------------------------------------------------------------
# 4. Structure of the C66 kernels
# ---------------------------------------------------------------------------


def check_kernel_structure() -> bool:
    """gamma12(f) * gamma12(g) is mirror-even with a double zero on the axis."""

    s = sp.symbols("s", real=True)
    order = 4
    # Two independent shear strains, each odd in s with x2/t-dependent
    # coefficients: this covers (u, lambda) and (W1, Lambda2) alike.
    p_funcs = [sp.Function(f"P{k}")(x2, t) for k in range(order)]
    q_funcs = [sp.Function(f"Q{k}")(x2, t) for k in range(order)]
    g_first = sum(c * s ** (2 * k + 1) for k, c in enumerate(p_funcs))
    g_second = sum(c * s ** (2 * k + 1) for k, c in enumerate(q_funcs))

    kernel_density = sp.expand(g_first * g_second)

    even_ok = sp.simplify(kernel_density.subs(s, -s) - kernel_density) == 0
    ok = _report(
        "4a. the C66 kernel density is mirror-EVEN (no sign change across the axis)",
        even_ok,
        "product of two odd factors, hence no forced sign change across the axis",
    )

    value_axis = sp.simplify(kernel_density.subs(s, 0))
    slope_axis = sp.simplify(sp.diff(kernel_density, s).subs(s, 0))
    curv_axis = sp.simplify(sp.diff(kernel_density, s, 2).subs(s, 0))

    ok &= _report(
        "4b. the kernel density has a double zero on the axis",
        value_axis == 0 and slope_axis == 0 and curv_axis != 0,
        f"value = {value_axis}, slope = {slope_axis}, curvature = {curv_axis}",
    )

    # Leading behaviour: quadratic in the distance to the axis.
    leading = sp.simplify(sp.series(kernel_density, s, 0, 3).removeO())
    ok &= _report(
        "4c. leading behaviour is quadratic in the distance to the axis",
        sp.simplify(leading - p_funcs[0] * q_funcs[0] * s**2) == 0,
        f"kernel ~ {p_funcs[0] * q_funcs[0]} * s^2",
    )

    # By contrast, the normal-strain kernels are even and generically non-zero
    # on the axis, which is why the effect is specific to C66.
    r_funcs = [sp.Function(f"R{k}")(x2, t) for k in range(order)]
    t_funcs = [sp.Function(f"T{k}")(x2, t) for k in range(order)]
    e_first = sum(c * s ** (2 * k) for k, c in enumerate(r_funcs))
    e_second = sum(c * s ** (2 * k) for k, c in enumerate(t_funcs))
    normal_kernel = sp.expand(e_first * e_second)
    ok &= _report(
        "4d. normal-strain kernels do NOT vanish on the axis (C66 specificity)",
        sp.simplify(normal_kernel.subs(s, 0)) == r_funcs[0] * t_funcs[0],
        f"eps-eps kernel|_(s=0) = {sp.simplify(normal_kernel.subs(s, 0))}",
    )
    return ok


# ---------------------------------------------------------------------------
# 5. Source polarisation: why the symmetry-reversal test can falsify
# ---------------------------------------------------------------------------


def check_source_polarisation() -> bool:
    """The (0,1) polarisation is mirror-even; a rotated one is not."""

    M = sp.diag(-1, 1)

    e2 = sp.Matrix([0, 1])
    even_ok = sp.simplify(M * e2 - e2) == sp.zeros(2, 1)
    ok = _report(
        "5a. the (0,1) source polarisation is mirror-even",
        even_ok,
        "M e2 = e2, so the synchronous array excitation respects the mirror",
    )

    theta = sp.symbols("theta", real=True)
    rotated = sp.Matrix([sp.sin(theta), sp.cos(theta)])
    mismatch = sp.simplify((M * rotated - rotated)[0])
    broken_ok = sp.simplify(mismatch.subs(theta, sp.pi / 4)) != 0
    ok &= _report(
        "5b. a rotated polarisation breaks the mirror (falsification test)",
        broken_ok,
        f"(M p - p)_1 = {mismatch}, non-zero unless sin(theta) = 0",
    )
    return ok


# ---------------------------------------------------------------------------
# 6. Symmetry-reversal prediction: an antisymmetric source reverses the theorem
# ---------------------------------------------------------------------------


def check_antisymmetric_source_reverses_theorem() -> bool:
    """p = (1, 0) is mirror-ODD, and the vanishing set swaps accordingly.

    This is the falsifiable prediction of the symmetry-reversal campaign,
    established here symbolically, without propagating any wave.  The mirror
    structure is *preserved*:
    every kernel stays mirror-even and the fitted centres stay on the axis.
    Only which coefficient carries the double zero changes.
    """

    M = sp.diag(-1, 1)

    e1 = sp.Matrix([1, 0])
    anti_ok = sp.simplify(M * e1 + e1) == sp.zeros(2, 1)
    ok = _report(
        "6a. the (1,0) source polarisation is mirror-ODD, not merely non-even",
        anti_ok,
        "M p = -p, so the array excitation lands in the opposite parity class",
    )

    # A mirror-ODD displacement is the exact converse of the mirror-even one of
    # check 3: first component EVEN in s, second component ODD.
    s = sp.symbols("s", real=True)
    order = 6
    c_funcs = [sp.Function(f"C{k}")(x2, t) for k in range(order)]  # u1, even
    d_funcs = [sp.Function(f"D{k}")(x2, t) for k in range(order)]  # u2, odd
    u1_anti = sum(c * s ** (2 * k) for k, c in enumerate(c_funcs))
    u2_anti = sum(c * s ** (2 * k + 1) for k, c in enumerate(d_funcs))

    g12_anti = sp.diff(u1_anti, x2) + sp.diff(u2_anti, s)
    e11_anti = sp.diff(u1_anti, s)
    e22_anti = sp.diff(u2_anti, x2)

    e11_axis = sp.simplify(e11_anti.subs(s, 0))
    e22_axis = sp.simplify(e22_anti.subs(s, 0))
    ok &= _report(
        "6b. eps11 and eps22 now vanish identically on the mirror axis",
        e11_axis == 0 and e22_axis == 0,
        f"eps11|_(s=0) = {e11_axis}, eps22|_(s=0) = {e22_axis}",
    )

    g12_axis = sp.simplify(g12_anti.subs(s, 0))
    ok &= _report(
        "6c. gamma12 no longer vanishes on the axis (generically non-zero)",
        g12_axis != 0,
        f"gamma12|_(s=0) = {g12_axis}",
    )

    # Parities are exactly swapped with respect to check 3d.
    parity_ok = (
        sp.simplify(g12_anti.subs(s, -s) - g12_anti) == 0
        and sp.simplify(e11_anti.subs(s, -s) + e11_anti) == 0
        and sp.simplify(e22_anti.subs(s, -s) + e22_anti) == 0
    )
    ok &= _report(
        "6d. gamma12 is now even while eps11, eps22 are odd (parities swapped)",
        parity_ok,
    )

    # Kernel structure: the double zero moves to the normal-strain kernels.
    r_funcs = [sp.Function(f"U{k}")(x2, t) for k in range(4)]
    v_funcs = [sp.Function(f"V{k}")(x2, t) for k in range(4)]
    e_first = sum(c * s ** (2 * k + 1) for k, c in enumerate(r_funcs))  # odd eps
    e_second = sum(c * s ** (2 * k + 1) for k, c in enumerate(v_funcs))
    normal_kernel = sp.expand(e_first * e_second)

    value_axis = sp.simplify(normal_kernel.subs(s, 0))
    slope_axis = sp.simplify(sp.diff(normal_kernel, s).subs(s, 0))
    curv_axis = sp.simplify(sp.diff(normal_kernel, s, 2).subs(s, 0))
    ok &= _report(
        "6e. the C11/C22/C12 kernels now carry the double zero on the axis",
        value_axis == 0 and slope_axis == 0 and curv_axis != 0,
        f"value = {value_axis}, slope = {slope_axis}, curvature = {curv_axis}",
    )

    g_first = sum(c * s ** (2 * k) for k, c in enumerate(r_funcs))  # even gamma
    g_second = sum(c * s ** (2 * k) for k, c in enumerate(v_funcs))
    shear_kernel = sp.expand(g_first * g_second)
    ok &= _report(
        "6f. the C66 kernel no longer vanishes on the axis (reversal complete)",
        sp.simplify(shear_kernel.subs(s, 0)) == r_funcs[0] * v_funcs[0],
        f"gamma-gamma kernel|_(s=0) = {sp.simplify(shear_kernel.subs(s, 0))}",
    )

    # Control: the density kernel pairs two fields of the same parity class as a
    # dot product, so one term survives on the axis in EITHER class.  It must
    # not move between the reference campaign and the symmetry-reversal one.
    a_funcs = [sp.Function(f"W{k}")(x2, t) for k in range(3)]
    b_funcs = [sp.Function(f"Z{k}")(x2, t) for k in range(3)]
    odd_1 = sum(c * s ** (2 * k + 1) for k, c in enumerate(a_funcs))
    even_1 = sum(c * s ** (2 * k) for k, c in enumerate(a_funcs))
    odd_2 = sum(c * s ** (2 * k + 1) for k, c in enumerate(b_funcs))
    even_2 = sum(c * s ** (2 * k) for k, c in enumerate(b_funcs))

    rho_even_class = sp.expand(odd_1 * odd_2 + even_1 * even_2)  # p = (0,1)
    rho_odd_class = sp.expand(even_1 * even_2 + odd_1 * odd_2)  # p = (1,0)
    ok &= _report(
        "6g. the density kernel is non-zero on the axis in BOTH classes (control)",
        sp.simplify(rho_even_class.subs(s, 0)) != 0
        and sp.simplify(rho_odd_class.subs(s, 0)) != 0,
        "rho must NOT move between the two campaigns; if it does, the observed "
        "effect is not the one the theorem predicts",
    )
    return ok


def main() -> int:
    print("Mirror-symmetry theorem for the C66 sensitivity kernels")
    print(f"Mirror: x1 -> {AXIS_SUM} - x1, fixed axis x1 = {AXIS}")
    print("-" * 72)

    results = [
        check_stiffness_invariance(),
        check_operator_equivariance(),
        check_strain_parities(),
        check_kernel_structure(),
        check_source_polarisation(),
        check_antisymmetric_source_reverses_theorem(),
    ]

    print("-" * 72)
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
