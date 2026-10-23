import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_residual_interaction_gradient_identity import (
    run_residual_interaction_gradient_identity_check,
)


def test_theorem2_identity_holds_to_machine_precision_with_direct_solver():
    report = run_residual_interaction_gradient_identity_check(
        n_cells=8,
        num_steps=10,
        dt=0.05,
        direct_solver=True,
        tolerance=1.0e-12,
    )

    assert report["status"] == "ok"
    assert report["values"]["relative_difference"] < 1.0e-12

    # A non-degenerate check: both sides must actually be non-zero, otherwise
    # the identity would hold trivially.
    assert abs(report["values"]["duality_pairing"]) > 1.0e-6
    assert abs(report["values"]["kernel_dm_pairing"]) > 1.0e-6


def test_identity_residual_tracks_the_linear_solver_tolerance():
    """
    The decisive evidence that the identity is exact rather than approximate:
    its residual is governed by the linear solve, not by the discretisation.
    Swapping the direct solve for the default iterative one must degrade the
    residual by orders of magnitude while leaving both sides unchanged to
    solver accuracy.
    """
    direct = run_residual_interaction_gradient_identity_check(
        n_cells=8, num_steps=10, dt=0.05, direct_solver=True
    )
    iterative = run_residual_interaction_gradient_identity_check(
        n_cells=8, num_steps=10, dt=0.05, direct_solver=False, tolerance=1.0e-6
    )

    assert direct["status"] == "ok"
    assert iterative["status"] == "ok"

    assert (
        iterative["values"]["relative_difference"]
        > 100.0 * direct["values"]["relative_difference"]
    )

    # Same physical quantity in both runs.
    assert direct["values"]["duality_pairing"] == pytest.approx(
        iterative["values"]["duality_pairing"], rel=1.0e-6
    )


def test_identity_is_insensitive_to_time_step_and_mesh_refinement():
    """
    An exact algebraic identity, unlike a consistent discretisation, must not
    improve or degrade with dt or h.
    """
    coarse = run_residual_interaction_gradient_identity_check(
        n_cells=6, num_steps=8, dt=0.08, direct_solver=True
    )
    fine = run_residual_interaction_gradient_identity_check(
        n_cells=10, num_steps=14, dt=0.02, direct_solver=True
    )

    assert coarse["status"] == "ok"
    assert fine["status"] == "ok"
    assert coarse["values"]["relative_difference"] < 1.0e-12
    assert fine["values"]["relative_difference"] < 1.0e-12
