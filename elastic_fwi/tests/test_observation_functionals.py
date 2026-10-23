import numpy as np
import pytest

from elastic_fwi.data.synthetic_observations import (
    BoundaryObservationSeries,
    SyntheticBoundaryObservationData,
)
from elastic_fwi.inversion.observation_functionals import (
    ObservationFunctionalWeights,
    compute_observation_functionals,
    full_aperture_misfit,
    observation_series_misfit,
    reflection_misfit,
    transmission_misfit,
    weighted_aperture_misfit,
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


def test_observation_functional_weights_validation():
    weights = ObservationFunctionalWeights(reflection=2.0, transmission=3.0)

    assert weights.reflection == pytest.approx(2.0)
    assert weights.transmission == pytest.approx(3.0)

    with pytest.raises(ValueError):
        ObservationFunctionalWeights(reflection=-1.0)

    with pytest.raises(ValueError):
        ObservationFunctionalWeights(transmission=np.nan)


def test_zero_misfit_when_simulated_equals_observed():
    observed = _data()
    simulated = _data()

    values = compute_observation_functionals(simulated, observed)

    assert values.reflection == pytest.approx(0.0)
    assert values.transmission == pytest.approx(0.0)
    assert values.full == pytest.approx(0.0)
    assert values.weighted == pytest.approx(0.0)

    data = values.as_dict()

    assert data["reflection"] == pytest.approx(0.0)
    assert data["transmission"] == pytest.approx(0.0)
    assert data["full"] == pytest.approx(0.0)
    assert data["weighted"] == pytest.approx(0.0)


def test_reflection_misfit_only():
    observed = _data()
    simulated = _data(reflection_values=(2.0, 3.0, 4.0))

    j_ref = reflection_misfit(simulated, observed)
    j_trans = transmission_misfit(simulated, observed)
    j_full = full_aperture_misfit(simulated, observed)

    assert j_ref == pytest.approx(1.5)
    assert j_trans == pytest.approx(0.0)
    assert j_full == pytest.approx(1.5)


def test_transmission_misfit_only():
    observed = _data()
    simulated = _data(transmission_values=(0.2, 0.3, 0.4))

    j_ref = reflection_misfit(simulated, observed)
    j_trans = transmission_misfit(simulated, observed)
    j_full = full_aperture_misfit(simulated, observed)

    assert j_ref == pytest.approx(0.0)
    assert j_trans == pytest.approx(0.015)
    assert j_full == pytest.approx(0.015)


def test_weighted_aperture_misfit():
    observed = _data()
    simulated = _data(
        reflection_values=(2.0, 3.0, 4.0),
        transmission_values=(0.2, 0.3, 0.4),
    )

    weights = ObservationFunctionalWeights(reflection=2.0, transmission=10.0)

    j_ref = reflection_misfit(simulated, observed)
    j_trans = transmission_misfit(simulated, observed)
    j_weighted = weighted_aperture_misfit(
        simulated,
        observed,
        weights=weights,
    )

    assert j_ref == pytest.approx(1.5)
    assert j_trans == pytest.approx(0.015)
    assert j_weighted == pytest.approx(2.0 * 1.5 + 10.0 * 0.015)

    values = compute_observation_functionals(
        simulated,
        observed,
        weights=weights,
    )

    assert values.reflection == pytest.approx(j_ref)
    assert values.transmission == pytest.approx(j_trans)
    assert values.full == pytest.approx(j_ref + j_trans)
    assert values.weighted == pytest.approx(j_weighted)
    assert values.reflection_weight == pytest.approx(2.0)
    assert values.transmission_weight == pytest.approx(10.0)


def test_time_weights_are_used():
    observed = _data()
    simulated = _data(reflection_values=(2.0, 3.0, 4.0))

    j_ref = reflection_misfit(
        simulated,
        observed,
        time_weights=[1.0, 2.0, 3.0],
    )

    assert j_ref == pytest.approx(0.5 * (1.0 + 2.0 + 3.0))


def test_observation_series_misfit_rejects_mismatched_series():
    simulated = BoundaryObservationSeries(
        name="gamma1_reflection",
        tag=1,
        component="vector",
        times=[0.1, 0.2],
        values=[1.0, 2.0],
    )

    observed_wrong_tag = BoundaryObservationSeries(
        name="gamma1_reflection",
        tag=2,
        component="vector",
        times=[0.1, 0.2],
        values=[1.0, 2.0],
    )

    with pytest.raises(ValueError):
        observation_series_misfit(simulated, observed_wrong_tag)

    observed_wrong_time = BoundaryObservationSeries(
        name="gamma1_reflection",
        tag=1,
        component="vector",
        times=[0.1, 0.3],
        values=[1.0, 2.0],
    )

    with pytest.raises(ValueError):
        observation_series_misfit(simulated, observed_wrong_time)


def test_observation_series_misfit_rejects_invalid_time_weights():
    observed = _data()
    simulated = _data(reflection_values=(2.0, 3.0, 4.0))

    with pytest.raises(ValueError):
        reflection_misfit(simulated, observed, time_weights=[1.0, 2.0])

    with pytest.raises(ValueError):
        reflection_misfit(simulated, observed, time_weights=[1.0, -1.0, 1.0])

    with pytest.raises(ValueError):
        reflection_misfit(simulated, observed, time_weights=[1.0, np.nan, 1.0])
