from __future__ import annotations

import numpy as np
import pytest

from elastic_fwi.inversion.material_image_cells import (
    aggregate_nodal_image_to_structured_cells,
)


def test_aggregate_nodal_image_to_structured_cells_mean_on_unit_square():
    node_coordinates = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    node_values = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float)

    result = aggregate_nodal_image_to_structured_cells(
        node_coordinates=node_coordinates,
        node_values=node_values,
        nx=1,
        ny=1,
        aggregation="mean",
        min_nodes_per_cell=4,
    )

    assert result.num_nodes == 4
    assert result.num_cells == 1
    assert result.cell_centers.shape == (1, 2)
    assert result.cell_values.shape == (1,)
    assert result.cell_centers[0, 0] == pytest.approx(0.5)
    assert result.cell_centers[0, 1] == pytest.approx(0.5)
    assert result.cell_values[0] == pytest.approx(4.0)
    assert result.local_min == pytest.approx(4.0)
    assert result.local_max == pytest.approx(4.0)
    assert result.local_l2_norm == pytest.approx(4.0)


def test_aggregate_nodal_image_to_structured_cells_sum_and_max():
    node_coordinates = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    node_values = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float)

    result_sum = aggregate_nodal_image_to_structured_cells(
        node_coordinates=node_coordinates,
        node_values=node_values,
        nx=1,
        ny=1,
        aggregation="sum",
        min_nodes_per_cell=4,
    )

    result_max = aggregate_nodal_image_to_structured_cells(
        node_coordinates=node_coordinates,
        node_values=node_values,
        nx=1,
        ny=1,
        aggregation="max",
        min_nodes_per_cell=4,
    )

    assert result_sum.cell_values[0] == pytest.approx(16.0)
    assert result_max.cell_values[0] == pytest.approx(7.0)


def test_aggregate_nodal_image_to_structured_cells_two_by_two_grid():
    node_coordinates = np.asarray(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    node_values = np.arange(9, dtype=float)

    result = aggregate_nodal_image_to_structured_cells(
        node_coordinates=node_coordinates,
        node_values=node_values,
        nx=2,
        ny=2,
        aggregation="mean",
        min_nodes_per_cell=4,
    )

    assert result.num_cells == 4
    assert result.cell_centers.shape == (4, 2)
    assert result.cell_values.shape == (4,)

    mapping = {
        tuple(center): value
        for center, value in zip(result.cell_centers, result.cell_values, strict=True)
    }

    assert mapping[(0.25, 0.25)] == pytest.approx(np.mean([0.0, 1.0, 3.0, 4.0]))
    assert mapping[(0.75, 0.25)] == pytest.approx(np.mean([1.0, 2.0, 4.0, 5.0]))
    assert mapping[(0.25, 0.75)] == pytest.approx(np.mean([3.0, 4.0, 6.0, 7.0]))
    assert mapping[(0.75, 0.75)] == pytest.approx(np.mean([4.0, 5.0, 7.0, 8.0]))


def test_aggregate_nodal_image_to_structured_cells_accepts_partial_mpi_like_nodes():
    node_coordinates = np.asarray(
        [
            [0.0, 0.0],
            [0.5, 0.0],
        ],
        dtype=float,
    )
    node_values = np.asarray([1.0, 3.0], dtype=float)

    result = aggregate_nodal_image_to_structured_cells(
        node_coordinates=node_coordinates,
        node_values=node_values,
        nx=2,
        ny=2,
        aggregation="mean",
        min_nodes_per_cell=1,
    )

    assert result.num_nodes == 2
    assert result.num_cells >= 1
    assert np.all(np.isfinite(result.cell_centers))
    assert np.all(np.isfinite(result.cell_values))


def test_aggregate_nodal_image_to_structured_cells_empty_input():
    result = aggregate_nodal_image_to_structured_cells(
        node_coordinates=np.empty((0, 2), dtype=float),
        node_values=np.empty((0,), dtype=float),
        nx=2,
        ny=2,
        aggregation="mean",
    )

    assert result.num_nodes == 0
    assert result.num_cells == 0
    assert result.cell_centers.shape == (0, 2)
    assert result.cell_values.shape == (0,)
    assert result.local_l2_norm == 0.0


def test_aggregate_nodal_image_to_structured_cells_as_dict_with_values():
    node_coordinates = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    node_values = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float)

    result = aggregate_nodal_image_to_structured_cells(
        node_coordinates=node_coordinates,
        node_values=node_values,
        nx=1,
        ny=1,
        aggregation="mean",
        min_nodes_per_cell=4,
    )

    data = result.as_dict(include_values=True)

    assert data["kind"] == "structured_cell_material_image"
    assert data["num_cells"] == 1
    assert "cell_centers" in data
    assert "cell_values" in data


@pytest.mark.parametrize(
    "bad_coordinates,bad_values,match",
    [
        (np.asarray([0.0, 1.0]), np.asarray([1.0]), "two-dimensional"),
        (np.asarray([[0.0, 0.0]]), np.asarray([1.0, 2.0]), "size mismatch"),
        (np.asarray([[0.0, np.nan]]), np.asarray([1.0]), "non-finite"),
        (np.asarray([[0.0, 0.0]]), np.asarray([np.nan]), "non-finite"),
    ],
)
def test_aggregate_nodal_image_to_structured_cells_rejects_invalid_inputs(
    bad_coordinates,
    bad_values,
    match,
):
    with pytest.raises(ValueError, match=match):
        aggregate_nodal_image_to_structured_cells(
            node_coordinates=bad_coordinates,
            node_values=bad_values,
            nx=1,
            ny=1,
        )


def test_aggregate_nodal_image_to_structured_cells_rejects_invalid_parameters():
    node_coordinates = np.asarray([[0.0, 0.0]], dtype=float)
    node_values = np.asarray([1.0], dtype=float)

    with pytest.raises(ValueError, match="Unsupported aggregation"):
        aggregate_nodal_image_to_structured_cells(
            node_coordinates=node_coordinates,
            node_values=node_values,
            nx=1,
            ny=1,
            aggregation="bad",
        )

    with pytest.raises(ValueError, match="strictly positive"):
        aggregate_nodal_image_to_structured_cells(
            node_coordinates=node_coordinates,
            node_values=node_values,
            nx=0,
            ny=1,
        )

    with pytest.raises(ValueError, match="min_nodes_per_cell"):
        aggregate_nodal_image_to_structured_cells(
            node_coordinates=node_coordinates,
            node_values=node_values,
            nx=1,
            ny=1,
            min_nodes_per_cell=0,
        )
