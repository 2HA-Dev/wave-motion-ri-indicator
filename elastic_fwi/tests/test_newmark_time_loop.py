import numpy as np
import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI
from dolfinx import mesh

from elastic_fwi.geometry.boundary_tags import (
    build_boundary_meshtags,
    unit_square_pitch_catch_boundary_specs,
)
from elastic_fwi.materials.dg0_fields import homogeneous_dg0_material_functions
from elastic_fwi.materials.orthotropic import LEGACY_HEALTHY_MATERIAL
from elastic_fwi.solvers.elastic_operator import (
    ElasticOperatorConfig,
    assemble_elastic_operator_matrices,
    create_vector_lagrange_space,
)
from elastic_fwi.solvers.newmark import LinearSolveConfig
from elastic_fwi.solvers.time_loop import (
    NewmarkTimeLoopConfig,
    NewmarkTimeStepRecord,
    run_newmark_time_loop,
)
from elastic_fwi.sources.traction import gamma1_vertical_traction_source


def _vector_difference_norm(left, right):
    diff = left.copy()
    diff.axpy(-1.0, right)
    norm = float(diff.norm())
    diff.destroy()
    return norm


def _create_loop_inputs():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    facet_tags = build_boundary_meshtags(
        msh,
        unit_square_pitch_catch_boundary_specs(),
    )

    V = create_vector_lagrange_space(msh, degree=1)
    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)

    operator_config = ElasticOperatorConfig(theta=0.25, dt=0.1)
    matrices = assemble_elastic_operator_matrices(V, materials, operator_config)

    source = gamma1_vertical_traction_source(
        tag=1,
        amplitude=1.0,
        frequency=1.0,
        t0=0.0,
        num_periods=1.0,
    )

    return matrices, V, facet_tags, source


def test_newmark_time_loop_config_validation():
    config = NewmarkTimeLoopConfig(num_steps=3, start_time=0.0)

    assert config.num_steps == 3
    assert config.start_time == pytest.approx(0.0)
    assert config.reuse_effective_solver is True

    no_reuse_config = NewmarkTimeLoopConfig(
        num_steps=3,
        start_time=0.0,
        reuse_effective_solver=False,
    )
    assert no_reuse_config.reuse_effective_solver is False

    with pytest.raises(ValueError):
        NewmarkTimeLoopConfig(num_steps=0)

    with pytest.raises(ValueError):
        NewmarkTimeLoopConfig(num_steps=1, start_time=np.nan)


def test_newmark_time_step_record_as_dict():
    record = NewmarkTimeStepRecord(
        step=1,
        time=0.1,
        force_norm=1.0,
        rhs_norm=0.01,
        displacement_norm=0.02,
        linear_residual_norm=1.0e-14,
        linear_residual_relative=1.0e-14,
        ksp_converged=True,
        ksp_reason=3,
        ksp_iterations=10,
        ksp_reported_residual_norm=1.0e-14,
    )

    data = record.as_dict()

    assert data["step"] == 1
    assert data["time"] == pytest.approx(0.1)
    assert data["ksp_converged"] is True


def test_run_newmark_time_loop_returns_consistent_records():
    matrices, V, facet_tags, source = _create_loop_inputs()

    try:
        result = run_newmark_time_loop(
            matrices=matrices,
            V=V,
            facet_tags=facet_tags,
            source=source,
            config=NewmarkTimeLoopConfig(num_steps=3),
            solve_config=LinearSolveConfig(
                ksp_type="cg",
                pc_type="jacobi",
                rtol=1.0e-10,
                atol=1.0e-12,
                max_it=1000,
            ),
        )

        try:
            assert len(result.records) == 3

            times = [record.time for record in result.records]
            assert times == pytest.approx([0.1, 0.2, 0.3])

            displacement_norms = [
                record.displacement_norm for record in result.records
            ]

            assert max(displacement_norms) > 0.0
            assert float(result.final_displacement.norm()) == pytest.approx(
                displacement_norms[-1]
            )

            for record in result.records:
                assert record.ksp_converged
                assert np.isfinite(record.force_norm)
                assert np.isfinite(record.rhs_norm)
                assert np.isfinite(record.displacement_norm)
                assert np.isfinite(record.linear_residual_norm)
                assert np.isfinite(record.linear_residual_relative)

                assert record.force_norm >= 0.0
                assert record.rhs_norm >= 0.0
                assert record.displacement_norm >= 0.0
                assert record.linear_residual_norm <= 1.0e-8
                assert record.linear_residual_relative <= 1.0e-8

        finally:
            result.destroy()

    finally:
        matrices.destroy()


def test_run_newmark_time_loop_reuse_matches_old_path():
    matrices, V, facet_tags, source = _create_loop_inputs()
    result_old = None
    result_new = None

    try:
        solve_config = LinearSolveConfig(
            ksp_type="cg",
            pc_type="jacobi",
            rtol=1.0e-10,
            atol=1.0e-12,
            max_it=1000,
        )

        result_old = run_newmark_time_loop(
            matrices=matrices,
            V=V,
            facet_tags=facet_tags,
            source=source,
            config=NewmarkTimeLoopConfig(
                num_steps=3,
                reuse_effective_solver=False,
            ),
            solve_config=solve_config,
        )

        result_new = run_newmark_time_loop(
            matrices=matrices,
            V=V,
            facet_tags=facet_tags,
            source=source,
            config=NewmarkTimeLoopConfig(
                num_steps=3,
                reuse_effective_solver=True,
            ),
            solve_config=solve_config,
        )

        diff_norm = _vector_difference_norm(
            result_new.final_displacement,
            result_old.final_displacement,
        )
        reference_norm = max(float(result_old.final_displacement.norm()), 1.0)

        assert diff_norm / reference_norm <= 1.0e-12
        assert len(result_old.records) == len(result_new.records) == 3

        for old_record, new_record in zip(result_old.records, result_new.records):
            assert old_record.ksp_converged
            assert new_record.ksp_converged
            assert old_record.ksp_converged == new_record.ksp_converged
            assert old_record.ksp_reason == new_record.ksp_reason
            assert np.isfinite(old_record.ksp_reported_residual_norm)
            assert np.isfinite(new_record.ksp_reported_residual_norm)
            assert new_record.linear_residual_relative <= 1.0e-8

    finally:
        if result_old is not None:
            result_old.destroy()
        if result_new is not None:
            result_new.destroy()
        matrices.destroy()

