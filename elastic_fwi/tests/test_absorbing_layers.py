from mpi4py import MPI
from dolfinx import mesh

from elastic_fwi.wave_physics import (
    absorbing_layer_summary_from_function,
    create_absorbing_layer_dg0,
    plan_absorbing_layer,
)


def test_absorbing_layer_plan_and_dg0_field_are_valid():
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [10.0, 10.0]],
        [8, 8],
        cell_type=mesh.CellType.triangle,
    )

    plan = plan_absorbing_layer(
        domain_length_x=10.0,
        domain_length_y=10.0,
        source_frequency=1.0,
        rho=4.5,
        C11=16.24,
        C22=18.07,
        C66=4.67,
        h_reference=10.0 / 8.0,
    )

    sigma = create_absorbing_layer_dg0(
        msh,
        domain_length_x=10.0,
        domain_length_y=10.0,
        plan=plan,
    )

    summary = absorbing_layer_summary_from_function(sigma)

    assert plan.width > 0.0
    assert plan.lambda_min > 0.0
    assert plan.sigma_max > 0.0
    assert summary["global_max"] > 0.0
    assert summary["global_nonzero_cells"] > 0
