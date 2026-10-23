import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.inversion.observation_functionals import (
    compute_observation_functionals,
)
from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    run_direct_observation_pipeline,
)


def test_direct_observation_pipeline_healthy_report_is_ok_and_json_serializable():
    config = DirectObservationPipelineConfig(
        nx=2,
        ny=2,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        material_kind="healthy",
    )

    result = run_direct_observation_pipeline(config)
    report = result.as_dict()

    assert report["kind"] == "direct_observation_pipeline_result"
    assert report["config"]["material_kind"] == "healthy"
    assert report["data"]["kind"] == "synthetic_boundary_observation_data"
    assert report["data"]["num_steps"] == 5
    assert report["data"]["final_time"] == pytest.approx(0.5)

    assert report["data"]["reflection"]["tag"] == 1
    assert report["data"]["transmission"]["tag"] == 2

    assert len(report["time_loop_records"]) == 5

    reflection = np.asarray(report["data"]["reflection"]["values"], dtype=float)
    transmission = np.asarray(report["data"]["transmission"]["values"], dtype=float)

    assert np.all(np.isfinite(reflection))
    assert np.all(np.isfinite(transmission))
    assert np.min(reflection) >= 0.0
    assert np.min(transmission) >= 0.0
    assert np.max(reflection) > 0.0

    payload = json.dumps(report, sort_keys=True)

    assert '"direct_observation_pipeline_result"' in payload
    assert '"gamma1_reflection"' in payload
    assert '"gamma2_transmission"' in payload


def test_direct_observation_pipeline_homogeneous_contrast_changes_observations():
    healthy = run_direct_observation_pipeline(
        DirectObservationPipelineConfig(
            nx=2,
            ny=2,
            degree=1,
            theta=0.25,
            dt=0.1,
            num_steps=5,
            material_kind="healthy",
        )
    )

    contrast = run_direct_observation_pipeline(
        DirectObservationPipelineConfig(
            nx=2,
            ny=2,
            degree=1,
            theta=0.25,
            dt=0.1,
            num_steps=5,
            material_kind="homogeneous_contrast",
            contrast_factor=0.8,
        )
    )

    values = compute_observation_functionals(
        healthy.data,
        contrast.data,
    )

    assert values.reflection > 0.0
    assert values.full > 0.0
    assert values.full == pytest.approx(values.reflection + values.transmission)

    diff = contrast.data.stacked_values() - healthy.data.stacked_values()

    assert np.max(np.abs(diff)) > 0.0


def test_direct_observation_pipeline_localized_defect_changes_observations():
    healthy = run_direct_observation_pipeline(
        DirectObservationPipelineConfig(
            nx=4,
            ny=4,
            degree=1,
            theta=0.25,
            dt=0.1,
            num_steps=5,
            material_kind="healthy",
        )
    )

    localized = run_direct_observation_pipeline(
        DirectObservationPipelineConfig(
            nx=4,
            ny=4,
            degree=1,
            theta=0.25,
            dt=0.1,
            num_steps=5,
            material_kind="localized_disk_defect",
            defect_center_x=0.5,
            defect_center_y=0.5,
            defect_radius=0.35,
            contrast_factor=0.8,
        )
    )

    assert localized.material_metadata["kind"] == "localized_disk_defect"
    assert localized.material_metadata["global_defect_cells"] > 0

    values = compute_observation_functionals(
        healthy.data,
        localized.data,
    )

    assert values.reflection > 0.0
    assert values.full > 0.0
    assert values.full == pytest.approx(values.reflection + values.transmission)

    diff = localized.data.stacked_values() - healthy.data.stacked_values()

    assert np.max(np.abs(diff)) > 0.0


def test_direct_observation_pipeline_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        DirectObservationPipelineConfig(nx=0)

    with pytest.raises(ValueError):
        DirectObservationPipelineConfig(material_kind="unknown")

    with pytest.raises(ValueError):
        DirectObservationPipelineConfig(defect_radius=-1.0)

    with pytest.raises(ValueError):
        DirectObservationPipelineConfig(contrast_factor=0.0)


def test_direct_observation_pipeline_can_retain_displacement_snapshots():
    config = DirectObservationPipelineConfig(
        nx=2,
        ny=2,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=3,
        material_kind="healthy",
        store_displacement_snapshots=True,
    )

    result = run_direct_observation_pipeline(config)

    try:
        assert result.has_displacement_snapshots
        assert result.num_displacement_snapshots == 3

        report = result.as_dict()

        assert report["state"]["displacement_snapshots_stored"] is True
        assert report["state"]["num_displacement_snapshots"] == 3

        global_sizes = [vector.getSize() for vector in result.displacement_snapshots]
        local_sizes = [vector.getLocalSize() for vector in result.displacement_snapshots]
        norms = [vector.norm() for vector in result.displacement_snapshots]

        assert len(set(global_sizes)) == 1
        assert all(size >= 0 for size in local_sizes)
        assert all(np.isfinite(norm) for norm in norms)
        assert max(norms) > 0.0

    finally:
        result.destroy()
