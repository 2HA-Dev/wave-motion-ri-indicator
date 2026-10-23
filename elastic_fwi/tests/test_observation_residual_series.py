import json

import numpy as np
import pytest

from elastic_fwi.data.synthetic_observations import (
    BoundaryObservationSeries,
    SyntheticBoundaryObservationData,
)
from elastic_fwi.inversion.observation_functionals import ObservationFunctionalWeights
from elastic_fwi.inversion.observation_residual_series import (
    ObservationResidualData,
    ObservationResidualSeries,
    compute_observation_residual_data,
    compute_observation_residual_series,
)


def _data(
    *,
    reflection_values=(1.0, 2.0, 3.0),
    transmission_values=(0.1, 0.2, 0.3),
):
    times = [0.1, 0.2, 0.3]

    reflection = BoundaryObservationSeries(
        name="gamma1_reflection",
        tag=1,
        component="vector",
        times=times,
        values=reflection_values,
    )

    transmission = BoundaryObservationSeries(
        name="gamma2_transmission",
        tag=2,
        component="vector",
        times=times,
        values=transmission_values,
    )

    return SyntheticBoundaryObservationData(
        reflection=reflection,
        transmission=transmission,
    )


def test_observation_residual_series_properties_and_serialization():
    residual = ObservationResidualSeries(
        name="gamma1_reflection",
        tag=1,
        component="vector",
        times=[0.1, 0.2, 0.3],
        values=[1.0, -2.0, 3.0],
    )

    assert residual.num_steps == 3
    assert residual.final_time == pytest.approx(0.3)
    assert residual.max_abs_value == pytest.approx(3.0)
    assert residual.l2_discrete_norm_squared == pytest.approx(14.0)
    assert residual.half_l2_discrete_norm_squared == pytest.approx(7.0)

    payload = json.dumps(residual.as_dict(), sort_keys=True)

    assert '"gamma1_reflection"' in payload
    assert '"max_abs_value"' in payload


def test_compute_observation_residual_series():
    simulated = BoundaryObservationSeries(
        name="gamma1_reflection",
        tag=1,
        component="vector",
        times=[0.1, 0.2, 0.3],
        values=[2.0, 3.0, 5.0],
    )

    observed = BoundaryObservationSeries(
        name="gamma1_reflection",
        tag=1,
        component="vector",
        times=[0.1, 0.2, 0.3],
        values=[1.0, 2.0, 4.0],
    )

    residual = compute_observation_residual_series(simulated, observed)

    assert residual.values == pytest.approx([1.0, 1.0, 1.0])
    assert residual.half_l2_discrete_norm_squared == pytest.approx(1.5)


def test_compute_observation_residual_data_and_functionals():
    simulated = _data(
        reflection_values=(2.0, 3.0, 4.0),
        transmission_values=(0.2, 0.3, 0.4),
    )

    observed = _data()

    residual_data = compute_observation_residual_data(
        simulated,
        observed,
        weights=ObservationFunctionalWeights(reflection=2.0, transmission=3.0),
    )

    assert residual_data.num_steps == 3
    assert residual_data.final_time == pytest.approx(0.3)
    assert residual_data.max_abs_residual == pytest.approx(1.0)

    assert residual_data.reflection.values == pytest.approx([1.0, 1.0, 1.0])
    assert residual_data.transmission.values == pytest.approx([0.1, 0.1, 0.1])

    assert residual_data.functionals.reflection == pytest.approx(1.5)
    assert residual_data.functionals.transmission == pytest.approx(0.015)
    assert residual_data.functionals.full == pytest.approx(1.515)
    assert residual_data.functionals.weighted == pytest.approx(2.0 * 1.5 + 3.0 * 0.015)

    stacked = residual_data.stacked_values()

    assert stacked.shape == (2, 3)
    assert stacked[0] == pytest.approx([1.0, 1.0, 1.0])
    assert stacked[1] == pytest.approx([0.1, 0.1, 0.1])

    payload = json.dumps(residual_data.as_dict(), sort_keys=True)

    assert '"observation_residual_data"' in payload
    assert '"functionals"' in payload


def test_observation_residual_data_rejects_mismatched_times():
    reflection = ObservationResidualSeries(
        name="gamma1_reflection",
        tag=1,
        component="vector",
        times=[0.1, 0.2],
        values=[1.0, 2.0],
    )

    transmission = ObservationResidualSeries(
        name="gamma2_transmission",
        tag=2,
        component="vector",
        times=[0.1, 0.3],
        values=[1.0, 2.0],
    )

    with pytest.raises(ValueError):
        ObservationResidualData(
            reflection=reflection,
            transmission=transmission,
            functionals=compute_observation_residual_data(_data(), _data()).functionals,
        )


def test_compute_observation_residual_series_rejects_mismatched_observations():
    simulated = BoundaryObservationSeries(
        name="gamma1_reflection",
        tag=1,
        component="vector",
        times=[0.1, 0.2],
        values=[1.0, 2.0],
    )

    observed = BoundaryObservationSeries(
        name="gamma1_reflection",
        tag=2,
        component="vector",
        times=[0.1, 0.2],
        values=[1.0, 2.0],
    )

    with pytest.raises(ValueError):
        compute_observation_residual_series(simulated, observed)


def test_observation_residual_series_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        ObservationResidualSeries(
            name="",
            tag=1,
            component="vector",
            times=[0.1],
            values=[1.0],
        )

    with pytest.raises(ValueError):
        ObservationResidualSeries(
            name="gamma1",
            tag=0,
            component="vector",
            times=[0.1],
            values=[1.0],
        )

    with pytest.raises(ValueError):
        ObservationResidualSeries(
            name="gamma1",
            tag=1,
            component="vector",
            times=[0.1, 0.2],
            values=[1.0],
        )

    with pytest.raises(ValueError):
        ObservationResidualSeries(
            name="gamma1",
            tag=1,
            component="vector",
            times=[0.2, 0.1],
            values=[1.0, 2.0],
        )

    with pytest.raises(ValueError):
        ObservationResidualSeries(
            name="gamma1",
            tag=1,
            component="vector",
            times=[0.1],
            values=[np.nan],
        )
