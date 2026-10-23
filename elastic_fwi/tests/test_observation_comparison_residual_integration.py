import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.inversion.observation_functionals import ObservationFunctionalWeights
from elastic_fwi.pipelines.direct_observation import DirectObservationPipelineConfig
from elastic_fwi.pipelines.observation_comparison import (
    DirectObservationComparisonConfig,
    run_direct_observation_comparison,
)


def test_direct_observation_comparison_contains_residual_data():
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

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    assert report["residuals"]["kind"] == "observation_residual_data"
    assert report["residuals"]["num_steps"] == 5
    assert report["residuals"]["reflection"]["tag"] == 1
    assert report["residuals"]["transmission"]["tag"] == 2

    simulated_values = result.simulated.data.stacked_values()
    observed_values = result.observed.data.stacked_values()
    residual_values = result.residuals.stacked_values()

    assert residual_values == pytest.approx(simulated_values - observed_values)

    values = report["functionals"]["simulated_vs_observed"]
    residual_functionals = report["residuals"]["functionals"]

    assert residual_functionals["reflection"] == pytest.approx(values["reflection"])
    assert residual_functionals["transmission"] == pytest.approx(values["transmission"])
    assert residual_functionals["full"] == pytest.approx(values["full"])
    assert residual_functionals["weighted"] == pytest.approx(values["weighted"])

    assert report["values"]["max_abs_residual"] == pytest.approx(
        np.max(np.abs(residual_values))
    )

    payload = json.dumps(report, sort_keys=True)

    assert '"observation_residual_data"' in payload
    assert '"residuals"' in payload
