import numpy as np
import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI
from dolfinx import mesh

from elastic_fwi.geometry.boundary_tags import (
    BoundarySegment2D,
    boundary_measures_by_tag,
    boundary_tag_counts_global,
    build_boundary_meshtags,
    legacy_pitch_catch_boundary_specs,
    unit_square_pitch_catch_boundary_specs,
    validate_boundary_specs,
)


def _create_unit_square_mesh():
    return mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)


def test_boundary_segment_marker_on_unit_square_points():
    segment = BoundarySegment2D(
        name="bottom",
        tag=1,
        fixed_axis=1,
        fixed_value=0.0,
        interval_axis=0,
        interval_min=0.0,
        interval_max=1.0,
    )

    x = np.array(
        [
            [0.0, 0.5, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    mask = segment.marker(x)

    np.testing.assert_array_equal(mask, np.array([True, True, True, False]))


def test_unit_square_boundary_tags_have_positive_counts_and_unit_measures():
    msh = _create_unit_square_mesh()

    specs = unit_square_pitch_catch_boundary_specs()
    facet_tags = build_boundary_meshtags(msh, specs)

    required_tags = (1, 2, 3, 4)

    counts = boundary_tag_counts_global(facet_tags, required_tags, msh.comm)
    measures = boundary_measures_by_tag(msh, facet_tags, required_tags)

    for tag in required_tags:
        assert counts[tag] > 0
        assert measures[tag] == pytest.approx(1.0)


def test_legacy_boundary_specs_are_coordinate_consistent():
    specs = legacy_pitch_catch_boundary_specs()

    by_name = {spec.name: spec for spec in specs}

    assert by_name["gamma1_reflection"].fixed_axis == 1
    assert by_name["gamma1_reflection"].fixed_value == pytest.approx(75.0)
    assert by_name["gamma1_reflection"].interval_min == pytest.approx(75.0)
    assert by_name["gamma1_reflection"].interval_max == pytest.approx(105.0)

    assert by_name["gamma2_transmission"].fixed_axis == 1
    assert by_name["gamma2_transmission"].fixed_value == pytest.approx(105.0)

    assert by_name["left_lateral"].fixed_axis == 0
    assert by_name["left_lateral"].fixed_value == pytest.approx(75.0)

    assert by_name["right_lateral"].fixed_axis == 0
    assert by_name["right_lateral"].fixed_value == pytest.approx(105.0)


def test_validate_boundary_specs_rejects_duplicate_names_but_allows_duplicate_tags():
    spec_a = BoundarySegment2D(
        name="a",
        tag=1,
        fixed_axis=1,
        fixed_value=0.0,
        interval_axis=0,
        interval_min=0.0,
        interval_max=1.0,
    )
    spec_b_same_name = BoundarySegment2D(
        name="a",
        tag=2,
        fixed_axis=1,
        fixed_value=1.0,
        interval_axis=0,
        interval_min=0.0,
        interval_max=1.0,
    )
    spec_b_same_tag = BoundarySegment2D(
        name="b",
        tag=1,
        fixed_axis=1,
        fixed_value=1.0,
        interval_axis=0,
        interval_min=0.0,
        interval_max=1.0,
    )

    with pytest.raises(ValueError):
        validate_boundary_specs((spec_a, spec_b_same_name))

    validate_boundary_specs((spec_a, spec_b_same_tag))


def test_invalid_boundary_segment_specs_are_rejected():
    with pytest.raises(ValueError):
        BoundarySegment2D(
            name="",
            tag=1,
            fixed_axis=1,
            fixed_value=0.0,
            interval_axis=0,
            interval_min=0.0,
            interval_max=1.0,
        )

    with pytest.raises(ValueError):
        BoundarySegment2D(
            name="bad_tag",
            tag=0,
            fixed_axis=1,
            fixed_value=0.0,
            interval_axis=0,
            interval_min=0.0,
            interval_max=1.0,
        )

    with pytest.raises(ValueError):
        BoundarySegment2D(
            name="bad_axes",
            tag=1,
            fixed_axis=1,
            fixed_value=0.0,
            interval_axis=1,
            interval_min=0.0,
            interval_max=1.0,
        )

    with pytest.raises(ValueError):
        BoundarySegment2D(
            name="bad_interval",
            tag=1,
            fixed_axis=1,
            fixed_value=0.0,
            interval_axis=0,
            interval_min=1.0,
            interval_max=0.0,
        )
