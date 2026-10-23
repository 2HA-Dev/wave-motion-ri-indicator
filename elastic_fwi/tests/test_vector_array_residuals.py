from __future__ import annotations

import numpy as np
import pytest

from elastic_fwi.inversion import (
    VectorArrayResidualSeries,
    build_vector_array_residual_pair,
    compute_vector_array_residual_series,
    reverse_vector_array_residual_series,
)
from elastic_fwi.observations import (
    VectorArrayObservationTimeSeries,
    vector_array_observation,
)
from elastic_fwi.transducers import build_line_transducer_array


def _make_series(values, *, array_name="gamma2", role="transmission"):
    array = build_line_transducer_array(
        name=array_name,
        role=role,
        start_x=3.0,
        start_y=9.0 if array_name == "gamma2" else 1.0,
        end_x=7.0,
        end_y=9.0 if array_name == "gamma2" else 1.0,
        num_elements=np.asarray(values).shape[1],
        sigma=0.25,
        tag=2 if array_name == "gamma2" else 1,
    )
    spec = vector_array_observation(array=array)

    values = np.asarray(values, dtype=float)
    return VectorArrayObservationTimeSeries(
        spec=spec,
        steps=np.arange(values.shape[0], dtype=np.int64),
        times=0.1 * np.arange(values.shape[0], dtype=float),
        values=values,
    )


def test_compute_vector_array_residual_series_reference_minus_observed():
    reference = _make_series(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    observed = _make_series(
        [
            [[0.5, 1.5], [2.0, 3.0]],
            [[4.0, 2.0], [10.0, 1.0]],
        ]
    )

    residual = compute_vector_array_residual_series(
        reference=reference,
        observed=observed,
    )

    expected = np.array(
        [
            [[0.5, 0.5], [1.0, 1.0]],
            [[1.0, 4.0], [-3.0, 7.0]],
        ]
    )

    assert residual.name == "gamma2_vector_array_residual"
    assert residual.array_name == "gamma2"
    assert residual.role == "transmission"
    np.testing.assert_allclose(residual.values, expected)
    assert residual.max_abs_residual == pytest.approx(7.0)
    assert residual.l2_norm_squared == pytest.approx(float(np.sum(expected * expected)))


def test_compute_vector_array_residual_series_scaling():
    reference = _make_series([[[2.0, 4.0]]])
    observed = _make_series([[[1.0, 1.0]]])

    residual = compute_vector_array_residual_series(
        reference=reference,
        observed=observed,
        scale=10.0,
    )

    np.testing.assert_allclose(residual.values, [[[10.0, 30.0]]])
    assert residual.scale == pytest.approx(10.0)


def test_reverse_vector_array_residual_series_reverses_time_axis():
    reference = _make_series(
        [
            [[1.0, 10.0]],
            [[2.0, 20.0]],
            [[3.0, 30.0]],
        ]
    )
    observed = _make_series(
        [
            [[0.0, 0.0]],
            [[0.0, 0.0]],
            [[0.0, 0.0]],
        ]
    )

    residual = compute_vector_array_residual_series(
        reference=reference,
        observed=observed,
        name="gamma2_residual",
    )
    reversed_residual = reverse_vector_array_residual_series(residual)

    np.testing.assert_allclose(
        reversed_residual.values[:, 0, :],
        [[3.0, 30.0], [2.0, 20.0], [1.0, 10.0]],
    )
    np.testing.assert_allclose(reversed_residual.times, [0.2, 0.1, 0.0])
    np.testing.assert_array_equal(reversed_residual.steps, [2, 1, 0])
    assert reversed_residual.name == "gamma2_residual_reversed_time"


def test_vector_array_residual_pair_reports_maximum():
    reflection = compute_vector_array_residual_series(
        reference=_make_series([[[1.0, 0.0]]], array_name="gamma1", role="source_reflection"),
        observed=_make_series([[[0.0, 0.0]]], array_name="gamma1", role="source_reflection"),
    )
    transmission = compute_vector_array_residual_series(
        reference=_make_series([[[0.0, 5.0]]], array_name="gamma2", role="transmission"),
        observed=_make_series([[[0.0, 0.0]]], array_name="gamma2", role="transmission"),
    )

    pair = build_vector_array_residual_pair(
        reflection=reflection,
        transmission=transmission,
    )

    assert pair.max_abs_residual == pytest.approx(5.0)
    reversed_pair = pair.reversed_time()
    assert reversed_pair.reflection is not None
    assert reversed_pair.transmission is not None


def test_residual_series_rejects_mismatched_shape():
    reference = _make_series([[[1.0, 2.0]]])
    observed = _make_series([[[1.0, 2.0], [3.0, 4.0]]])

    with pytest.raises(ValueError):
        compute_vector_array_residual_series(reference=reference, observed=observed)


def test_residual_series_rejects_mismatched_time():
    reference = _make_series([[[1.0, 2.0]]])
    observed = _make_series([[[0.0, 0.0]]])
    observed = VectorArrayObservationTimeSeries(
        spec=observed.spec,
        steps=observed.steps,
        times=np.array([99.0]),
        values=observed.values,
    )

    with pytest.raises(ValueError):
        compute_vector_array_residual_series(reference=reference, observed=observed)


def test_residual_series_rejects_mismatched_array_name():
    reference = _make_series([[[1.0, 2.0]]], array_name="gamma1", role="source_reflection")
    observed = _make_series([[[0.0, 0.0]]], array_name="gamma2", role="transmission")

    with pytest.raises(ValueError):
        compute_vector_array_residual_series(reference=reference, observed=observed)


def test_vector_array_residual_series_as_dict_is_json_friendly():
    reference = _make_series([[[1.0, 2.0]]])
    observed = _make_series([[[0.5, 1.0]]])

    residual = compute_vector_array_residual_series(
        reference=reference,
        observed=observed,
        name="test_residual",
    )

    data = residual.as_dict()

    assert data["kind"] == "vector_array_residual_series"
    assert data["name"] == "test_residual"
    assert data["array_name"] == "gamma2"
    assert data["role"] == "transmission"
    assert data["shape"] == [1, 1, 2]
    assert data["max_abs_residual"] == pytest.approx(1.0)


def test_vector_array_residual_pair_requires_one_series():
    with pytest.raises(ValueError):
        build_vector_array_residual_pair()
