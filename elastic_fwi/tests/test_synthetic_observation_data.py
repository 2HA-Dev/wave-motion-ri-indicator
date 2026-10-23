import numpy as np
import pytest

from elastic_fwi.data.synthetic_observations import (
    BoundaryObservationSeries,
    SyntheticBoundaryObservationData,
    add_gaussian_noise,
    boundary_observation_series_from_records,
    synthetic_data_from_boundary_records,
)


def _reflection_records():
    return [
        {
            "step": 1,
            "time": 0.1,
            "observation_name": "gamma1_reflection",
            "tag": 1,
            "component": "vector",
            "norm_squared": 1.0e-3,
        },
        {
            "step": 2,
            "time": 0.2,
            "observation_name": "gamma1_reflection",
            "tag": 1,
            "component": "vector",
            "norm_squared": 2.0e-3,
        },
        {
            "step": 3,
            "time": 0.3,
            "observation_name": "gamma1_reflection",
            "tag": 1,
            "component": "vector",
            "norm_squared": 3.0e-3,
        },
    ]


def _transmission_records():
    return [
        {
            "step": 1,
            "time": 0.1,
            "observation_name": "gamma2_transmission",
            "tag": 2,
            "component": "vector",
            "norm_squared": 1.0e-5,
        },
        {
            "step": 2,
            "time": 0.2,
            "observation_name": "gamma2_transmission",
            "tag": 2,
            "component": "vector",
            "norm_squared": 2.0e-5,
        },
        {
            "step": 3,
            "time": 0.3,
            "observation_name": "gamma2_transmission",
            "tag": 2,
            "component": "vector",
            "norm_squared": 3.0e-5,
        },
    ]


def test_boundary_observation_series_validates_and_serializes():
    series = BoundaryObservationSeries(
        name="gamma1_reflection",
        tag=1,
        component="vector",
        times=[0.1, 0.2, 0.3],
        values=[1.0, 2.0, 3.0],
    )

    assert series.num_steps == 3
    assert series.final_time == pytest.approx(0.3)
    assert series.max_value == pytest.approx(3.0)
    assert series.final_value == pytest.approx(3.0)

    data = series.as_dict()

    assert data["name"] == "gamma1_reflection"
    assert data["tag"] == 1
    assert data["component"] == "vector"
    assert data["value_kind"] == "norm_squared"
    assert data["times"] == pytest.approx([0.1, 0.2, 0.3])
    assert data["values"] == pytest.approx([1.0, 2.0, 3.0])


def test_boundary_observation_series_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        BoundaryObservationSeries(
            name="",
            tag=1,
            component="vector",
            times=[0.1],
            values=[1.0],
        )

    with pytest.raises(ValueError):
        BoundaryObservationSeries(
            name="gamma1",
            tag=0,
            component="vector",
            times=[0.1],
            values=[1.0],
        )

    with pytest.raises(ValueError):
        BoundaryObservationSeries(
            name="gamma1",
            tag=1,
            component="vector",
            times=[0.1, 0.2],
            values=[1.0],
        )

    with pytest.raises(ValueError):
        BoundaryObservationSeries(
            name="gamma1",
            tag=1,
            component="vector",
            times=[0.2, 0.1],
            values=[1.0, 2.0],
        )

    with pytest.raises(ValueError):
        BoundaryObservationSeries(
            name="gamma1",
            tag=1,
            component="vector",
            times=[0.1],
            values=[np.nan],
        )


def test_series_from_records():
    series = boundary_observation_series_from_records(_reflection_records())

    assert series.name == "gamma1_reflection"
    assert series.tag == 1
    assert series.component == "vector"
    assert series.times == pytest.approx([0.1, 0.2, 0.3])
    assert series.values == pytest.approx([1.0e-3, 2.0e-3, 3.0e-3])


def test_synthetic_data_from_boundary_records():
    data = synthetic_data_from_boundary_records(
        reflection_records=_reflection_records(),
        transmission_records=_transmission_records(),
        metadata={"source": "unit-test"},
    )

    assert data.num_steps == 3
    assert data.final_time == pytest.approx(0.3)
    assert data.reflection.tag == 1
    assert data.transmission.tag == 2
    assert data.metadata["source"] == "unit-test"

    stacked = data.stacked_values()

    assert stacked.shape == (2, 3)
    assert stacked[0] == pytest.approx([1.0e-3, 2.0e-3, 3.0e-3])
    assert stacked[1] == pytest.approx([1.0e-5, 2.0e-5, 3.0e-5])

    serialized = data.as_dict()

    assert serialized["kind"] == "synthetic_boundary_observation_data"
    assert serialized["num_steps"] == 3
    assert serialized["reflection"]["name"] == "gamma1_reflection"
    assert serialized["transmission"]["name"] == "gamma2_transmission"


def test_synthetic_data_rejects_mismatched_times():
    reflection = BoundaryObservationSeries(
        name="gamma1_reflection",
        tag=1,
        component="vector",
        times=[0.1, 0.2],
        values=[1.0, 2.0],
    )

    transmission = BoundaryObservationSeries(
        name="gamma2_transmission",
        tag=2,
        component="vector",
        times=[0.1, 0.3],
        values=[1.0, 2.0],
    )

    with pytest.raises(ValueError):
        SyntheticBoundaryObservationData(
            reflection=reflection,
            transmission=transmission,
        )


def test_gaussian_noise_is_reproducible_and_does_not_mutate_clean_data():
    clean = synthetic_data_from_boundary_records(
        reflection_records=_reflection_records(),
        transmission_records=_transmission_records(),
    )

    noisy_a = add_gaussian_noise(clean, noise_level=0.05, seed=123)
    noisy_b = add_gaussian_noise(clean, noise_level=0.05, seed=123)

    assert noisy_a.reflection.values == pytest.approx(noisy_b.reflection.values)
    assert noisy_a.transmission.values == pytest.approx(noisy_b.transmission.values)

    assert clean.reflection.values == pytest.approx([1.0e-3, 2.0e-3, 3.0e-3])
    assert clean.transmission.values == pytest.approx([1.0e-5, 2.0e-5, 3.0e-5])

    assert np.all(noisy_a.reflection.values >= 0.0)
    assert np.all(noisy_a.transmission.values >= 0.0)

    assert noisy_a.metadata["noise_model"] == "gaussian"
    assert noisy_a.metadata["noise_level"] == pytest.approx(0.05)
    assert noisy_a.metadata["noise_seed"] == 123


def test_gaussian_noise_zero_level_returns_same_values():
    clean = synthetic_data_from_boundary_records(
        reflection_records=_reflection_records(),
        transmission_records=_transmission_records(),
    )

    noisy = add_gaussian_noise(clean, noise_level=0.0, seed=123)

    assert noisy.reflection.values == pytest.approx(clean.reflection.values)
    assert noisy.transmission.values == pytest.approx(clean.transmission.values)


def test_gaussian_noise_rejects_invalid_noise_level():
    clean = synthetic_data_from_boundary_records(
        reflection_records=_reflection_records(),
        transmission_records=_transmission_records(),
    )

    with pytest.raises(ValueError):
        add_gaussian_noise(clean, noise_level=-1.0)

    with pytest.raises(ValueError):
        add_gaussian_noise(clean, noise_level=np.nan)
