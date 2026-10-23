import numpy as np

from elastic_fwi.diagnostics.check_density_kernel_convention_identity import (
    _identity_from_series,
)


def _terminal_zero_series(rng, *, n_points, n_components, n_steps):
    """
    A discrete series with arbitrary interior values but the last two stored
    snapshots exactly zero: the discrete signature of the strong-form adjoint
    terminal conditions lambda(T) = d_t lambda(T) = 0 assumed by the
    residual-interaction bilinear identity. The identity checked by
    _identity_from_series only ever subtracts a boundary term at the t = 0
    end, so it is exact precisely when the t = T end vanishes in this way. On
    campaign data the identity holds to a relative residual of 1.2e-15.
    """
    series = rng.normal(size=(n_steps, n_points, n_components))
    series[-1] = 0.0
    series[-2] = 0.0
    return series


def test_identity_holds_at_near_machine_precision_when_terminal_conditions_hold():
    rng = np.random.default_rng(0)
    n_points, n_components, n_steps = 6, 2, 40
    dt = 0.03

    a_series = _terminal_zero_series(
        rng, n_points=n_points, n_components=n_components, n_steps=n_steps
    )
    b_series = _terminal_zero_series(
        rng, n_points=n_points, n_components=n_components, n_steps=n_steps
    )
    owned = np.ones(n_points, dtype=bool)

    report = _identity_from_series(a_series, b_series, dt=dt, owned=owned)

    # Staggered (compatible) discretisation: exact discrete primitive of the
    # production second-difference operator -- must close near machine
    # precision.
    assert report["values"]["staggered_residual_relative"] < 1.0e-8

    # Naive centred discretisation is a genuinely different discrete
    # operator: it must NOT satisfy the identity. This contrast is what
    # makes the check probative rather than tautological.
    assert report["values"]["centred_residual_relative"] > 1.0e-2

    # A generic (non-degenerate) origin: the boundary term is not zero.
    assert report["values"]["boundary_term_norm"] > 0.0


def test_boundary_term_vanishes_exactly_when_origin_step_is_stationary():
    rng = np.random.default_rng(1)
    n_points, n_components, n_steps = 4, 2, 30
    dt = 0.05

    a_series = _terminal_zero_series(
        rng, n_points=n_points, n_components=n_components, n_steps=n_steps
    )
    b_series = _terminal_zero_series(
        rng, n_points=n_points, n_components=n_components, n_steps=n_steps
    )

    # Force the first discrete step of `a` to be exactly stationary:
    # Da_{1/2} = (a_1 - a_0) / dt = 0 bit-for-bit. The boundary term
    # - Da_{1/2} . b_0 must then vanish exactly, and the identity collapses
    # to K_acc = B_staggered with no correction.
    a_series[1] = a_series[0]
    owned = np.ones(n_points, dtype=bool)

    report = _identity_from_series(a_series, b_series, dt=dt, owned=owned)

    assert report["values"]["boundary_term_norm"] == 0.0
    assert report["values"]["staggered_residual_relative"] < 1.0e-8
