import json
from dataclasses import replace

import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
)
from elastic_fwi.pipelines.observation_comparison import (
    DirectObservationComparisonConfig,
    run_common_mesh_direct_observation_comparison,
    run_direct_observation_comparison,
)


def _common_pair_config(
    *,
    simulated_dt: float = 0.1,
    observed_dt: float = 0.1,
    store_snapshots: bool = True,
) -> DirectObservationComparisonConfig:
    common = {
        "nx": 4,
        "ny": 4,
        "degree": 1,
        "theta": 0.25,
        "num_steps": 4,
        "store_displacement_snapshots": store_snapshots,
    }

    return DirectObservationComparisonConfig(
        simulated=DirectObservationPipelineConfig(
            **common,
            dt=simulated_dt,
            material_kind="healthy",
        ),
        observed=DirectObservationPipelineConfig(
            **common,
            dt=observed_dt,
            material_kind="localized_disk_defect",
            defect_center_x=0.5,
            defect_center_y=0.5,
            defect_radius=0.35,
            contrast_factor=0.8,
        ),
    )


def test_common_mesh_pair_reuses_mesh_partition_and_function_space():
    result = run_common_mesh_direct_observation_comparison(
        _common_pair_config()
    )

    try:
        report = result.as_dict()

        assert report["status"] == "ok"
        assert report["common_mesh"]["enabled"] is True
        assert report["common_mesh"]["common_mesh"] is True
        assert report["common_mesh"]["mesh_created_once"] is True
        assert report["common_mesh"]["shared_mesh_object"] is True
        assert report["common_mesh"]["shared_function_space"] is True
        assert report["common_mesh"]["same_mpi_partition"] is True
        assert report["common_mesh"]["same_export_times"] is True

        assert result.simulated.function_space is result.observed.function_space
        assert (
            result.simulated.function_space.mesh
            is result.observed.function_space.mesh
        )

        assert result.simulated.has_displacement_snapshots
        assert result.observed.has_displacement_snapshots
        assert (
            result.simulated.num_displacement_snapshots
            == result.observed.num_displacement_snapshots
            == 4
        )

        assert report["values"]["simulated_defect_cells"] == 0
        assert report["values"]["observed_defect_cells"] > 0
        assert report["values"]["max_abs_observation_difference"] > 0.0

        json.dumps(report, sort_keys=True)

    finally:
        result.simulated.destroy()
        result.observed.destroy()


def test_common_mesh_pair_rejects_time_discretization_mismatch():
    config = _common_pair_config(
        simulated_dt=0.2,
        observed_dt=0.1,
        store_snapshots=False,
    )

    with pytest.raises(
        ValueError,
        match="Incompatible direct configurations for a common-mesh pair",
    ):
        run_common_mesh_direct_observation_comparison(config)


def test_common_mesh_pair_rejects_different_declared_defect_geometry():
    base = _common_pair_config(store_snapshots=False)
    simulated = replace(
        base.simulated,
        defect_center_x=0.45,
    )
    config = DirectObservationComparisonConfig(
        simulated=simulated,
        observed=base.observed,
    )

    with pytest.raises(
        ValueError,
        match="defect_center_x",
    ):
        run_common_mesh_direct_observation_comparison(config)


def test_default_comparison_uses_shared_observed_refined_mesh():
    result = run_direct_observation_comparison(
        _common_pair_config(store_snapshots=False)
    )

    try:
        assert result.simulated.function_space is result.observed.function_space
        report = result.as_dict()
        assert report["config"]["mesh_policy"] == "shared_observed_refined"
        assert report["common_mesh"]["common_mesh"] is True
        assert report["common_mesh"]["mesh_source"] == "observed_configuration"
        assert (
            report["common_mesh"]["refinement_semantics"]
            == "physical_experiment_geometry"
        )
    finally:
        result.simulated.destroy()
        result.observed.destroy()


def test_legacy_independent_policy_remains_explicitly_available():
    base = _common_pair_config(store_snapshots=False)
    config = DirectObservationComparisonConfig(
        simulated=base.simulated,
        observed=base.observed,
        weights=base.weights,
        mesh_policy="independent",
    )
    result = run_direct_observation_comparison(config)

    try:
        assert result.simulated.function_space is not result.observed.function_space
        assert result.as_dict()["common_mesh"] == {}
    finally:
        result.simulated.destroy()
        result.observed.destroy()
