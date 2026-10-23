from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from elastic_fwi.postprocessing.scattered_wavefield_reconstruction import (
    MeshCompatibilityError,
    _canonical_triangle_connectivity,
    _p1_triangle_cells_for_export,
    _require_p1_triangle_connectivity,
)


def _block(cell_type: str, data) -> SimpleNamespace:
    return SimpleNamespace(
        type=cell_type,
        data=np.asarray(data, dtype=np.int64),
    )


def _mesh(*blocks) -> SimpleNamespace:
    return SimpleNamespace(cells=list(blocks))


@pytest.mark.parametrize(
    "cell_type",
    [
        "triangle",
        "VTK_TRIANGLE",
        "VTK_LAGRANGE_TRIANGLE",
    ],
)
def test_three_node_triangle_labels_are_accepted(
    cell_type: str,
) -> None:
    connectivity = np.asarray(
        [[2, 0, 1], [4, 3, 2]],
        dtype=np.int64,
    )

    result = _require_p1_triangle_connectivity(
        cell_type=cell_type,
        connectivity=connectivity,
        context="unit test",
    )

    assert result is connectivity
    assert result.shape == (2, 3)


def test_lagrange_triangle_is_canonicalized_as_p1() -> None:
    mesh = _mesh(
        _block("vertex", [[0], [1]]),
        _block(
            "VTK_LAGRANGE_TRIANGLE",
            [[5, 4, 3], [2, 0, 1]],
        ),
    )

    result = _canonical_triangle_connectivity(
        mesh,
        piece=Path("lagrange_p1.vtu"),
    )

    np.testing.assert_array_equal(
        result,
        np.asarray(
            [[0, 1, 2], [3, 4, 5]],
            dtype=np.int64,
        ),
    )


@pytest.mark.parametrize("nodes_per_cell", [4, 6, 10])
def test_higher_order_lagrange_triangle_is_rejected(
    nodes_per_cell: int,
) -> None:
    connectivity = np.arange(
        nodes_per_cell,
        dtype=np.int64,
    ).reshape(1, nodes_per_cell)

    with pytest.raises(
        MeshCompatibilityError,
        match="exactly three connectivity nodes",
    ):
        _require_p1_triangle_connectivity(
            cell_type="VTK_LAGRANGE_TRIANGLE",
            connectivity=connectivity,
            context="higher-order triangle",
        )


def test_lagrange_triangle_is_normalized_for_meshio_export() -> None:
    mesh = _mesh(
        _block(
            "VTK_LAGRANGE_TRIANGLE",
            [[0, 1, 2], [2, 3, 0]],
        ),
    )

    cells = _p1_triangle_cells_for_export(
        mesh,
        context="export test",
    )

    assert len(cells) == 1
    assert cells[0][0] == "triangle"
    assert cells[0][1].shape == (2, 3)


def test_non_triangle_cell_remains_rejected() -> None:
    with pytest.raises(
        MeshCompatibilityError,
        match="Unsupported cell type",
    ):
        _require_p1_triangle_connectivity(
            cell_type="quad",
            connectivity=np.asarray(
                [[0, 1, 2, 3]],
                dtype=np.int64,
            ),
            context="quadrilateral test",
        )


@pytest.mark.parametrize(
    "cell_type",
    ["vertex", "line", "line3"],
)
def test_auxiliary_blocks_are_ignored(
    cell_type: str,
) -> None:
    result = _require_p1_triangle_connectivity(
        cell_type=cell_type,
        connectivity=np.zeros((1, 1), dtype=np.int64),
        context="auxiliary block",
    )

    assert result is None
