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
from elastic_fwi.observations.boundary_observation import (
    gamma1_reflection_observation,
    gamma2_transmission_observation,
)
from elastic_fwi.observations.time_series import (
    BoundaryObservationTimeSeriesRecord,
    displacement_vector_to_function,
    evaluate_boundary_observation_from_vector,
    evaluate_boundary_observation_time_series,
)
from elastic_fwi.solvers.elastic_operator import (
    ElasticOperatorConfig,
    assemble_elastic_operator_matrices,
    create_vector_lagrange_space,
)
from elastic_fwi.solvers.newmark import LinearSolveConfig, create_zero_state_vector
from elastic_fwi.solvers.time_loop import (
    NewmarkTimeLoopConfig,
    run_newmark_time_loop,
)
from elastic_fwi.sources.traction import gamma1_vertical_traction_source


def _create_inputs():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    facet_tags = build_boundary_meshtags(
        msh,
        unit_square_pitch_catch_boundary_specs(),
    )

    V = create_vector_lagrange_space(msh, degree=1)
    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)

    matrices = assemble_elastic_operator_matrices(
        V,
        materials,
        ElasticOperatorConfig(theta=0.25, dt=0.1),
    )

    source = gamma1_vertical_traction_source(
        tag=1,
        amplitude=1.0,
        frequency=1.0,
        t0=0.0,
        num_periods=1.0,
    )

    return matrices, V, facet_tags, source


def test_boundary_observation_time_series_record_as_dict():
    record = BoundaryObservationTimeSeriesRecord(
        step=1,
        time=0.1,
        observation_name="gamma1_reflection",
        tag=1,
        component="vector",
        norm_squared=2.0,
    )

    data = record.as_dict()

    assert data["step"] == 1
    assert data["time"] == pytest.approx(0.1)
    assert data["observation_name"] == "gamma1_reflection"
    assert data["tag"] == 1
    assert data["component"] == "vector"
    assert data["norm_squared"] == pytest.approx(2.0)


def test_zero_displacement_vector_gives_zero_boundary_observation():
    matrices, V, facet_tags, _source = _create_inputs()

    try:
        vector = create_zero_state_vector(matrices)
        observation = gamma1_reflection_observation(component="vector")

        function = displacement_vector_to_function(V, vector)
        assert function.x.array.size > 0 or V.dofmap.index_map.size_local == 0

        norm_squared = evaluate_boundary_observation_from_vector(
            V,
            vector,
            facet_tags,
            observation,
        )

        assert norm_squared == pytest.approx(0.0)

    finally:
        vector = locals().get("vector")
        if vector is not None:
            vector.destroy()
        matrices.destroy()


def test_observation_time_series_from_newmark_snapshots():
    matrices, V, facet_tags, source = _create_inputs()

    try:
        result = run_newmark_time_loop(
            matrices=matrices,
            V=V,
            facet_tags=facet_tags,
            source=source,
            config=NewmarkTimeLoopConfig(
                num_steps=3,
                start_time=0.0,
                store_displacements=True,
            ),
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
            assert len(result.displacement_snapshots) == 3

            snapshots = [
                (record.step, record.time, vector)
                for record, vector in zip(
                    result.records,
                    result.displacement_snapshots,
                    strict=True,
                )
            ]

            gamma1 = gamma1_reflection_observation(component="vector")
            gamma2 = gamma2_transmission_observation(component="vector")

            gamma1_records = evaluate_boundary_observation_time_series(
                V,
                snapshots,
                facet_tags,
                gamma1,
            )

            gamma2_records = evaluate_boundary_observation_time_series(
                V,
                snapshots,
                facet_tags,
                gamma2,
            )

            assert len(gamma1_records) == 3
            assert len(gamma2_records) == 3

            assert [record.time for record in gamma1_records] == pytest.approx(
                [0.1, 0.2, 0.3]
            )

            gamma1_norms = [record.norm_squared for record in gamma1_records]
            gamma2_norms = [record.norm_squared for record in gamma2_records]

            assert np.all(np.isfinite(gamma1_norms))
            assert np.all(np.isfinite(gamma2_norms))

            assert min(gamma1_norms) >= 0.0
            assert min(gamma2_norms) >= 0.0

            assert max(gamma1_norms) > 0.0

            for record in gamma1_records:
                assert record.observation_name == "gamma1_reflection"
                assert record.tag == 1
                assert record.component == "vector"

            for record in gamma2_records:
                assert record.observation_name == "gamma2_transmission"
                assert record.tag == 2
                assert record.component == "vector"

        finally:
            result.destroy()

    finally:
        matrices.destroy()
