from __future__ import annotations

import numpy as np
import pytest

from elastic_fwi.inversion.material_image_nodes import (
    aggregate_dof_image_to_nodes,
)


def test_aggregate_dof_image_to_nodes_mean_groups_duplicate_coordinates():
    coordinates = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    values = np.asarray([1.0, 3.0, 2.0, 6.0, 5.0], dtype=float)

    result = aggregate_dof_image_to_nodes(
        local_coordinates=coordinates,
        local_values=values,
        aggregation="mean",
    )

    assert result.kind if hasattr(result, "kind") else True
    assert result.aggregation == "mean"
    assert result.num_dofs == 5
    assert result.num_nodes == 3
    assert result.node_coordinates.shape == (3, 2)
    assert result.node_values.shape == (3,)
    assert np.all(np.isfinite(result.node_coordinates))
    assert np.all(np.isfinite(result.node_values))

    mapping = {
        tuple(coord): value
        for coord, value in zip(result.node_coordinates, result.node_values, strict=True)
    }

    assert mapping[(0.0, 0.0)] == pytest.approx(2.0)
    assert mapping[(1.0, 0.0)] == pytest.approx(4.0)
    assert mapping[(0.0, 1.0)] == pytest.approx(5.0)


def test_aggregate_dof_image_to_nodes_sum():
    coordinates = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    values = np.asarray([1.0, 3.0, 2.0], dtype=float)

    result = aggregate_dof_image_to_nodes(
        local_coordinates=coordinates,
        local_values=values,
        aggregation="sum",
    )

    mapping = {
        tuple(coord): value
        for coord, value in zip(result.node_coordinates, result.node_values, strict=True)
    }

    assert mapping[(0.0, 0.0)] == pytest.approx(4.0)
    assert mapping[(1.0, 0.0)] == pytest.approx(2.0)


def test_aggregate_dof_image_to_nodes_max():
    coordinates = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    values = np.asarray([1.0, 3.0, 2.0], dtype=float)

    result = aggregate_dof_image_to_nodes(
        local_coordinates=coordinates,
        local_values=values,
        aggregation="max",
    )

    mapping = {
        tuple(coord): value
        for coord, value in zip(result.node_coordinates, result.node_values, strict=True)
    }

    assert mapping[(0.0, 0.0)] == pytest.approx(3.0)
    assert mapping[(1.0, 0.0)] == pytest.approx(2.0)


def test_aggregate_dof_image_to_nodes_empty_input():
    result = aggregate_dof_image_to_nodes(
        local_coordinates=np.empty((0, 2), dtype=float),
        local_values=np.empty((0,), dtype=float),
        aggregation="mean",
    )

    assert result.num_dofs == 0
    assert result.num_nodes == 0
    assert result.node_coordinates.shape == (0, 2)
    assert result.node_values.shape == (0,)
    assert result.local_l2_norm == 0.0


def test_aggregate_dof_image_to_nodes_rejects_invalid_aggregation():
    with pytest.raises(ValueError, match="Unsupported aggregation"):
        aggregate_dof_image_to_nodes(
            local_coordinates=np.asarray([[0.0, 0.0]], dtype=float),
            local_values=np.asarray([1.0], dtype=float),
            aggregation="bad",
        )


def test_aggregate_dof_image_to_nodes_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="size mismatch"):
        aggregate_dof_image_to_nodes(
            local_coordinates=np.asarray([[0.0, 0.0]], dtype=float),
            local_values=np.asarray([1.0, 2.0], dtype=float),
            aggregation="mean",
        )


def test_aggregate_dof_image_to_nodes_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="non-finite"):
        aggregate_dof_image_to_nodes(
            local_coordinates=np.asarray([[0.0, 0.0]], dtype=float),
            local_values=np.asarray([np.nan], dtype=float),
            aggregation="mean",
        )


def test_aggregate_dof_image_to_nodes_as_dict_with_values():
    coordinates = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    values = np.asarray([1.0, 3.0], dtype=float)

    result = aggregate_dof_image_to_nodes(
        local_coordinates=coordinates,
        local_values=values,
        aggregation="mean",
    )

    data = result.as_dict(include_values=True)

    assert data["kind"] == "nodal_material_image"
    assert data["aggregation"] == "mean"
    assert data["num_dofs"] == 2
    assert data["num_nodes"] == 1
    assert "node_coordinates" in data
    assert "node_values" in data
