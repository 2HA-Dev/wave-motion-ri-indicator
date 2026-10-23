from __future__ import annotations

import numpy as np
import pytest

from elastic_fwi.transducers import (
    TransducerArraySpec,
    auto_place_array_inside_absorbing_layer,
    build_line_transducer_array,
    build_pitch_catch_transducer_arrays,
    recommended_sigma_from_mesh_size,
)


def test_line_transducer_array_centers_are_uniform():
    spec = build_line_transducer_array(
        name="gamma1",
        role="source_reflection",
        start_x=3.0,
        start_y=1.0,
        end_x=7.0,
        end_y=1.0,
        num_elements=5,
        sigma=0.25,
        tag=1,
    )

    expected = np.array(
        [
            [3.0, 1.0],
            [4.0, 1.0],
            [5.0, 1.0],
            [6.0, 1.0],
            [7.0, 1.0],
        ]
    )

    np.testing.assert_allclose(spec.centers, expected)
    assert spec.spacing == pytest.approx(1.0)
    assert spec.sigma == pytest.approx(0.25)
    assert spec.role == "source_reflection"
    assert spec.tag == 1


def test_single_transducer_is_placed_at_segment_midpoint():
    spec = build_line_transducer_array(
        name="single",
        role="transmission",
        start_x=3.0,
        start_y=9.0,
        end_x=7.0,
        end_y=9.0,
        num_elements=1,
        sigma=0.25,
    )

    np.testing.assert_allclose(spec.centers, np.array([[5.0, 9.0]]))
    assert spec.spacing == pytest.approx(0.0)


def test_auto_place_array_inside_absorbing_layer_bottom_and_top():
    gamma1 = auto_place_array_inside_absorbing_layer(
        name="gamma1",
        role="source_reflection",
        side="bottom",
        domain_length_y=10.0,
        absorbing_layer_width=1.0,
        start_x=3.0,
        end_x=7.0,
        num_elements=5,
        sigma=0.25,
        tag=1,
    )

    gamma2 = auto_place_array_inside_absorbing_layer(
        name="gamma2",
        role="transmission",
        side="top",
        domain_length_y=10.0,
        absorbing_layer_width=1.0,
        start_x=3.0,
        end_x=7.0,
        num_elements=5,
        sigma=0.25,
        tag=2,
    )

    assert np.allclose(gamma1.centers[:, 1], 1.0)
    assert np.allclose(gamma2.centers[:, 1], 9.0)
    np.testing.assert_allclose(gamma1.centers[:, 0], gamma2.centers[:, 0])


def test_build_pitch_catch_transducer_arrays_l10_abs1():
    gamma1, gamma2 = build_pitch_catch_transducer_arrays(
        domain_length_y=10.0,
        absorbing_layer_width=1.0,
        start_x=3.0,
        end_x=7.0,
        num_elements=5,
        sigma=0.25,
    )

    assert gamma1.name == "gamma1"
    assert gamma1.role == "source_reflection"
    assert gamma1.tag == 1

    assert gamma2.name == "gamma2"
    assert gamma2.role == "transmission"
    assert gamma2.tag == 2

    np.testing.assert_allclose(gamma1.centers[:, 1], np.full(5, 1.0))
    np.testing.assert_allclose(gamma2.centers[:, 1], np.full(5, 9.0))
    np.testing.assert_allclose(gamma1.centers[:, 0], np.linspace(3.0, 7.0, 5))
    np.testing.assert_allclose(gamma2.centers[:, 0], np.linspace(3.0, 7.0, 5))


def test_recommended_sigma_from_mesh_size():
    assert recommended_sigma_from_mesh_size(0.125) == pytest.approx(0.25)
    assert recommended_sigma_from_mesh_size(0.125, factor=3.0) == pytest.approx(0.375)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(name="", role="source", start=(0.0, 0.0), end=(1.0, 0.0), num_elements=2),
        dict(name="bad", role="wrong", start=(0.0, 0.0), end=(1.0, 0.0), num_elements=2),
        dict(name="bad", role="source", start=(0.0, 0.0), end=(1.0, 0.0), num_elements=0),
        dict(name="bad", role="source", start=(0.0, 0.0), end=(1.0, 0.0), num_elements=2, sigma=0.0),
        dict(name="bad", role="source", start=(0.0, 0.0), end=(1.0, 0.0), num_elements=2, tag=0),
    ],
)
def test_invalid_transducer_array_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TransducerArraySpec(**kwargs)


def test_manifest_dict_is_json_friendly():
    spec = build_line_transducer_array(
        name="gamma2",
        role="transmission",
        start_x=3.0,
        start_y=9.0,
        end_x=7.0,
        end_y=9.0,
        num_elements=3,
        sigma=0.25,
        tag=2,
    )

    manifest = spec.as_manifest_dict()

    assert manifest["name"] == "gamma2"
    assert manifest["role"] == "transmission"
    assert manifest["num_elements"] == 3
    assert manifest["sigma"] == pytest.approx(0.25)
    assert manifest["tag"] == 2
    assert manifest["centers"] == [[3.0, 9.0], [5.0, 9.0], [7.0, 9.0]]


def test_polyline_transducer_array_builds_gamma2_left_top_right_without_duplicate_corners():
    from elastic_fwi.transducers import build_polyline_transducer_array

    gamma2 = build_polyline_transducer_array(
        name="gamma2",
        role="transmission",
        vertices=[
            (5.0, 3.0),
            (5.0, 17.0),
            (15.0, 17.0),
            (15.0, 3.0),
        ],
        num_elements_per_segment=[3, 3, 3],
        sigma=0.25,
        tag=2,
    )

    expected = np.array(
        [
            [5.0, 3.0],
            [5.0, 10.0],
            [5.0, 17.0],
            [10.0, 17.0],
            [15.0, 17.0],
            [15.0, 10.0],
            [15.0, 3.0],
        ]
    )

    np.testing.assert_allclose(gamma2.centers, expected)
    assert gamma2.num_elements == 7
    assert gamma2.role == "transmission"
    assert gamma2.tag == 2
    assert gamma2.as_manifest_dict()["geometry"] == "explicit_centers"
