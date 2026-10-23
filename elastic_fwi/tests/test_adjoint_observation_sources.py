import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.geometry.boundary_tags import (
    build_boundary_meshtags,
    unit_square_pitch_catch_boundary_specs,
)
from elastic_fwi.inversion.adjoint_observation_sources import (
    assemble_adjoint_observation_rhs_series,
    assemble_full_adjoint_observation_rhs,
)
from elastic_fwi.inversion.observation_functionals import ObservationFunctionalWeights
from elastic_fwi.pipelines.direct_observation import DirectObservationPipelineConfig
from elastic_fwi.pipelines.observation_comparison import (
    DirectObservationComparisonConfig,
    run_direct_observation_comparison,
)


def _comparison_with_snapshots():
    config = DirectObservationComparisonConfig(
        simulated=DirectObservationPipelineConfig(
            nx=4,
            ny=4,
            degree=1,
            theta=0.25,
            dt=0.1,
            num_steps=5,
            material_kind="healthy",
            store_displacement_snapshots=True,
        ),
        observed=DirectObservationPipelineConfig(
            nx=4,
            ny=4,
            degree=1,
            theta=0.25,
            dt=0.1,
            num_steps=5,
            material_kind="localized_disk_defect",
            defect_radius=0.35,
            contrast_factor=0.8,
            store_displacement_snapshots=True,
        ),
        weights=ObservationFunctionalWeights(
            reflection=1.0,
            transmission=1.0,
        ),
    )

    return run_direct_observation_comparison(config)


def test_adjoint_observation_rhs_series_for_reflection_residual():
    comparison = _comparison_with_snapshots()

    V = comparison.simulated.function_space
    assert V is comparison.observed.function_space
    assert V is not None
    facet_tags = build_boundary_meshtags(
        V.mesh,
        unit_square_pitch_catch_boundary_specs(),
    )

    result = assemble_adjoint_observation_rhs_series(
        V,
        facet_tags,
        comparison.simulated.displacement_snapshots,
        comparison.residuals.reflection,
    )

    try:
        assert result.num_steps == comparison.residuals.num_steps
        assert len(result.rhs_vectors) == comparison.residuals.num_steps
        assert result.max_rhs_norm > 0.0

        report = result.as_dict()

        assert report["kind"] == "adjoint_observation_rhs_result"
        assert report["num_steps"] == 5
        assert report["max_rhs_norm"] > 0.0

        for record in report["records"]:
            assert record["tag"] == 1
            assert record["component"] == "vector"
            assert np.isfinite(record["rhs_norm"])
            assert record["rhs_size_global"] > 0
            assert record["rhs_size_local"] >= 0

        payload = json.dumps(report, sort_keys=True)

        assert '"adjoint_observation_rhs_result"' in payload

    finally:
        result.destroy()
        comparison.simulated.destroy()
        comparison.observed.destroy()


def test_full_adjoint_observation_rhs_for_reflection_and_transmission():
    comparison = _comparison_with_snapshots()

    V = comparison.simulated.function_space
    assert V is comparison.observed.function_space
    assert V is not None
    facet_tags = build_boundary_meshtags(
        V.mesh,
        unit_square_pitch_catch_boundary_specs(),
    )

    rhs = assemble_full_adjoint_observation_rhs(
        V,
        facet_tags,
        comparison.simulated.displacement_snapshots,
        comparison.residuals,
    )

    try:
        assert rhs.num_steps == 5
        assert rhs.reflection.num_steps == 5
        assert rhs.transmission.num_steps == 5
        assert rhs.max_rhs_norm > 0.0

        report = rhs.as_dict()

        assert report["kind"] == "full_adjoint_observation_rhs_result"
        assert report["reflection"]["num_steps"] == 5
        assert report["transmission"]["num_steps"] == 5

        assert report["reflection"]["max_rhs_norm"] > 0.0
        assert report["transmission"]["max_rhs_norm"] >= 0.0

        payload = json.dumps(report, sort_keys=True)

        assert '"full_adjoint_observation_rhs_result"' in payload
        assert '"reflection"' in payload
        assert '"transmission"' in payload

    finally:
        rhs.destroy()
        comparison.simulated.destroy()
        comparison.observed.destroy()


def test_adjoint_observation_rhs_rejects_mismatched_snapshot_count():
    comparison = _comparison_with_snapshots()

    V = comparison.simulated.function_space
    assert V is comparison.observed.function_space
    assert V is not None
    facet_tags = build_boundary_meshtags(
        V.mesh,
        unit_square_pitch_catch_boundary_specs(),
    )

    with pytest.raises(ValueError):
        assemble_adjoint_observation_rhs_series(
            V,
            facet_tags,
            comparison.simulated.displacement_snapshots[:-1],
            comparison.residuals.reflection,
        )

    comparison.simulated.destroy()
    comparison.observed.destroy()
