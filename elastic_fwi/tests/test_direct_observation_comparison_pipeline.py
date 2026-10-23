import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.inversion.observation_functionals import (
    ObservationFunctionalWeights,
)
from elastic_fwi.pipelines.direct_observation import DirectObservationPipelineConfig
from elastic_fwi.pipelines.observation_comparison import (
    DirectObservationComparisonConfig,
    run_direct_observation_comparison,
)


def test_direct_observation_comparison_zero_for_identical_healthy_configs():
    config = DirectObservationComparisonConfig(
        simulated=DirectObservationPipelineConfig(
            nx=2,
            ny=2,
            degree=1,
            theta=0.25,
            dt=0.1,
            num_steps=5,
            material_kind="healthy",
        ),
        observed=DirectObservationPipelineConfig(
            nx=2,
            ny=2,
            degree=1,
            theta=0.25,
            dt=0.1,
            num_steps=5,
            material_kind="healthy",
        ),
    )

    result = run_direct_observation_comparison(config)
    report = result.as_dict()

    assert report["status"] == "ok"

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    values = report["functionals"]["simulated_vs_observed"]

    assert values["reflection"] == pytest.approx(0.0)
    assert values["transmission"] == pytest.approx(0.0)
    assert values["full"] == pytest.approx(0.0)
    assert values["weighted"] == pytest.approx(0.0)


def test_direct_observation_comparison_detects_homogeneous_contrast():
    config = DirectObservationComparisonConfig(
        simulated=DirectObservationPipelineConfig(
            nx=2,
            ny=2,
            degree=1,
            theta=0.25,
            dt=0.1,
            num_steps=5,
            material_kind="healthy",
        ),
        observed=DirectObservationPipelineConfig(
            nx=2,
            ny=2,
            degree=1,
            theta=0.25,
            dt=0.1,
            num_steps=5,
            material_kind="homogeneous_contrast",
            contrast_factor=0.8,
        ),
    )

    result = run_direct_observation_comparison(config)
    report = result.as_dict()

    assert report["status"] == "ok"

    values = report["functionals"]["simulated_vs_observed"]

    assert values["reflection"] > 0.0
    assert values["full"] > 0.0
    assert report["values"]["max_abs_observation_difference"] > 0.0


def test_direct_observation_comparison_detects_localized_defect_and_weights():
    config = DirectObservationComparisonConfig(
        simulated=DirectObservationPipelineConfig(
            nx=4,
            ny=4,
            degree=1,
            theta=0.25,
            dt=0.1,
            num_steps=5,
            material_kind="healthy",
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
        ),
        weights=ObservationFunctionalWeights(
            reflection=2.0,
            transmission=3.0,
        ),
    )

    result = run_direct_observation_comparison(config)
    report = result.as_dict()

    assert report["status"] == "ok"

    values = report["functionals"]["simulated_vs_observed"]

    assert report["values"]["observed_defect_cells"] > 0
    assert report["values"]["max_abs_observation_difference"] > 0.0

    assert values["reflection"] > 0.0
    assert values["full"] > 0.0
    assert values["weighted"] == pytest.approx(
        2.0 * values["reflection"] + 3.0 * values["transmission"]
    )

    assert np.isfinite(values["weighted"])


def test_direct_observation_comparison_report_is_json_serializable():
    result = run_direct_observation_comparison(
        DirectObservationComparisonConfig(
            simulated=DirectObservationPipelineConfig(
                nx=4,
                ny=4,
                degree=1,
                theta=0.25,
                dt=0.1,
                num_steps=5,
                material_kind="healthy",
            ),
            observed=DirectObservationPipelineConfig(
                nx=4,
                ny=4,
                degree=1,
                theta=0.25,
                dt=0.1,
                num_steps=5,
                material_kind="localized_disk_defect",
                contrast_factor=0.8,
            ),
        )
    )

    payload = json.dumps(result.as_dict(), sort_keys=True)

    assert '"direct_observation_comparison_result"' in payload
    assert '"simulated_vs_observed"' in payload
    assert '"localized_disk_defect"' in payload
