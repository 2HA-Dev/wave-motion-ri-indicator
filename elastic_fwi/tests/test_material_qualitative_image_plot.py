from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from elastic_fwi.diagnostics.plot_material_qualitative_image import (
    plot_material_qualitative_image_npz,
)


def _write_valid_export(
    path: Path,
    *,
    include_defect: bool = True,
    include_node_image: bool = True,
) -> None:
    payload = {
        "kind": np.asarray("material_qualitative_image_export", dtype=np.str_),
        "status": np.asarray("ok", dtype=np.str_),
        "mpi_rank": np.asarray(0, dtype=np.int64),
        "mpi_size": np.asarray(1, dtype=np.int64),
        "image_local_values": np.asarray([0.0, 0.25, 1.0, 0.5], dtype=np.float64),
        "image_local_coordinates": np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "image_coordinate_dim": np.asarray(2, dtype=np.int64),
        "image_local_size": np.asarray(4, dtype=np.int64),
        "image_global_size": np.asarray(4, dtype=np.int64),
        "image_global_l2_norm": np.asarray(1.14564392373896, dtype=np.float64),
        "image_global_max": np.asarray(1.0, dtype=np.float64),
        "normalized": np.asarray(True, dtype=np.bool_),
    }

    if include_node_image:
        payload.update(
            {
                "image_node_values": np.asarray([0.125, 0.75], dtype=np.float64),
                "image_node_coordinates": np.asarray(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                    ],
                    dtype=np.float64,
                ),
                "image_node_count": np.asarray(2, dtype=np.int64),
                "image_node_aggregation": np.asarray("mean", dtype=np.str_),
                "image_node_tolerance": np.asarray(1.0e-12, dtype=np.float64),
                "image_cell_values": np.asarray([0.4375], dtype=np.float64),
                "image_cell_centers": np.asarray(
                    [
                        [0.5, 0.5],
                    ],
                    dtype=np.float64,
                ),
                "image_cell_count": np.asarray(1, dtype=np.int64),
                "image_cell_aggregation": np.asarray("mean", dtype=np.str_),
                "image_cell_tolerance": np.asarray(1.0e-12, dtype=np.float64),
                "image_cell_min_nodes_per_cell": np.asarray(1, dtype=np.int64),
            }
        )

    if include_defect:
        payload.update(
            {
                "defect_center_x": np.asarray(0.5, dtype=np.float64),
                "defect_center_y": np.asarray(0.5, dtype=np.float64),
                "defect_radius": np.asarray(0.25, dtype=np.float64),
            }
        )

    np.savez(path, **payload)


def test_plot_material_qualitative_image_npz_creates_png_and_pdf_dof_mode(
    tmp_path: Path,
):
    input_path = tmp_path / "image.npz"
    output_png = tmp_path / "image.png"
    output_pdf = tmp_path / "image.pdf"

    _write_valid_export(input_path, include_defect=True, include_node_image=True)

    result = plot_material_qualitative_image_npz(
        input_path=input_path,
        output_png=output_png,
        output_pdf=output_pdf,
    )

    assert result["kind"] == "material_qualitative_image_plot"
    assert result["status"] == "ok"
    assert result["input"] == str(input_path)
    assert result["output_png"] == str(output_png)
    assert result["output_pdf"] == str(output_pdf)
    assert result["num_values"] == 4
    assert result["coordinate_dimension"] == 2
    assert result["image_mode"] == "dof"
    assert result["use_nodal_image"] is False
    assert result["node_aggregation"] == ""
    assert result["normalized"] is True
    assert result["defect_overlay_available"] is True
    assert result["defect_overlay_added"] is True

    assert output_png.exists()
    assert output_pdf.exists()
    assert output_png.stat().st_size > 0
    assert output_pdf.stat().st_size > 0


def test_plot_material_qualitative_image_npz_creates_png_nodal_mode(
    tmp_path: Path,
):
    input_path = tmp_path / "image_nodes.npz"
    output_png = tmp_path / "image_nodes.png"

    _write_valid_export(input_path, include_defect=True, include_node_image=True)

    result = plot_material_qualitative_image_npz(
        input_path=input_path,
        output_png=output_png,
        use_nodal_image=True,
    )

    assert result["status"] == "ok"
    assert result["output_pdf"] == ""
    assert result["num_values"] == 2
    assert result["coordinate_dimension"] == 2
    assert result["image_mode"] == "node"
    assert result["use_nodal_image"] is True
    assert result["node_aggregation"] == "mean"
    assert result["defect_overlay_available"] is True
    assert result["defect_overlay_added"] is True

    assert output_png.exists()
    assert output_png.stat().st_size > 0


def test_plot_material_qualitative_image_npz_creates_png_cell_mode(
    tmp_path: Path,
):
    input_path = tmp_path / "image_cells.npz"
    output_png = tmp_path / "image_cells.png"

    _write_valid_export(input_path, include_defect=True, include_node_image=True)

    result = plot_material_qualitative_image_npz(
        input_path=input_path,
        output_png=output_png,
        use_cell_image=True,
    )

    assert result["status"] == "ok"
    assert result["output_pdf"] == ""
    assert result["num_values"] == 1
    assert result["coordinate_dimension"] == 2
    assert result["image_mode"] == "cell"
    assert result["use_nodal_image"] is False
    assert result["use_cell_image"] is True
    assert result["cell_aggregation"] == "mean"
    assert result["node_aggregation"] == ""
    assert result["defect_overlay_available"] is True
    assert result["defect_overlay_added"] is True

    assert output_png.exists()
    assert output_png.stat().st_size > 0


def test_plot_material_qualitative_image_npz_creates_png_only_without_overlay(
    tmp_path: Path,
):
    input_path = tmp_path / "image.npz"
    output_png = tmp_path / "image.png"

    _write_valid_export(input_path, include_defect=True, include_node_image=True)

    result = plot_material_qualitative_image_npz(
        input_path=input_path,
        output_png=output_png,
        output_pdf=None,
        title="test image",
        show_defect_overlay=False,
    )

    assert result["status"] == "ok"
    assert result["output_pdf"] == ""
    assert result["coordinate_dimension"] == 2
    assert result["image_mode"] == "dof"
    assert result["defect_overlay_available"] is True
    assert result["defect_overlay_added"] is False
    assert output_png.exists()
    assert output_png.stat().st_size > 0


def test_plot_material_qualitative_image_npz_accepts_missing_defect_metadata(
    tmp_path: Path,
):
    input_path = tmp_path / "image_without_defect.npz"
    output_png = tmp_path / "image_without_defect.png"

    _write_valid_export(input_path, include_defect=False, include_node_image=True)

    result = plot_material_qualitative_image_npz(
        input_path=input_path,
        output_png=output_png,
    )

    assert result["status"] == "ok"
    assert result["defect_overlay_available"] is False
    assert result["defect_overlay_added"] is False
    assert output_png.exists()
    assert output_png.stat().st_size > 0


def test_plot_material_qualitative_image_rejects_missing_nodal_image_in_nodal_mode(
    tmp_path: Path,
):
    input_path = tmp_path / "missing_nodes.npz"
    output_png = tmp_path / "missing_nodes.png"

    _write_valid_export(input_path, include_defect=True, include_node_image=False)

    with pytest.raises(ValueError, match="Missing required NPZ keys"):
        plot_material_qualitative_image_npz(
            input_path=input_path,
            output_png=output_png,
            use_nodal_image=True,
        )


def test_plot_material_qualitative_image_rejects_invalid_kind(tmp_path: Path):
    input_path = tmp_path / "bad.npz"
    output_png = tmp_path / "bad.png"

    np.savez(
        input_path,
        kind=np.asarray("bad_kind", dtype=np.str_),
        status=np.asarray("ok", dtype=np.str_),
        mpi_rank=np.asarray(0, dtype=np.int64),
        mpi_size=np.asarray(1, dtype=np.int64),
        image_local_values=np.asarray([1.0], dtype=np.float64),
        image_local_coordinates=np.asarray([[0.0, 0.0]], dtype=np.float64),
        image_coordinate_dim=np.asarray(2, dtype=np.int64),
        image_local_size=np.asarray(1, dtype=np.int64),
        image_global_size=np.asarray(1, dtype=np.int64),
        image_global_l2_norm=np.asarray(1.0, dtype=np.float64),
        image_global_max=np.asarray(1.0, dtype=np.float64),
        normalized=np.asarray(True, dtype=np.bool_),
    )

    with pytest.raises(ValueError, match="Invalid NPZ kind"):
        plot_material_qualitative_image_npz(
            input_path=input_path,
            output_png=output_png,
        )


def test_plot_material_qualitative_image_rejects_nonfinite_dof_values(
    tmp_path: Path,
):
    input_path = tmp_path / "bad_values.npz"
    output_png = tmp_path / "bad_values.png"

    _write_valid_export(input_path, include_defect=True, include_node_image=True)

    with np.load(input_path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}

    payload["image_local_values"] = np.asarray([0.0, np.nan], dtype=np.float64)
    payload["image_local_coordinates"] = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    payload["image_local_size"] = np.asarray(2, dtype=np.int64)

    np.savez(input_path, **payload)

    with pytest.raises(ValueError, match="non-finite"):
        plot_material_qualitative_image_npz(
            input_path=input_path,
            output_png=output_png,
        )


def test_plot_material_qualitative_image_rejects_nonfinite_node_values(
    tmp_path: Path,
):
    input_path = tmp_path / "bad_node_values.npz"
    output_png = tmp_path / "bad_node_values.png"

    _write_valid_export(input_path, include_defect=True, include_node_image=True)

    with np.load(input_path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}

    payload["image_node_values"] = np.asarray([0.0, np.nan], dtype=np.float64)

    np.savez(input_path, **payload)

    with pytest.raises(ValueError, match="non-finite"):
        plot_material_qualitative_image_npz(
            input_path=input_path,
            output_png=output_png,
            use_nodal_image=True,
        )


def test_plot_material_qualitative_image_rejects_nonfinite_cell_values(
    tmp_path: Path,
):
    input_path = tmp_path / "bad_cell_values.npz"
    output_png = tmp_path / "bad_cell_values.png"

    _write_valid_export(input_path, include_defect=True, include_node_image=True)

    with np.load(input_path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}

    payload["image_cell_values"] = np.asarray([np.nan], dtype=np.float64)

    np.savez(input_path, **payload)

    with pytest.raises(ValueError, match="non-finite"):
        plot_material_qualitative_image_npz(
            input_path=input_path,
            output_png=output_png,
            use_cell_image=True,
        )


def test_plot_material_qualitative_image_rejects_coordinate_value_mismatch(
    tmp_path: Path,
):
    input_path = tmp_path / "bad_coordinates.npz"
    output_png = tmp_path / "bad_coordinates.png"

    _write_valid_export(input_path, include_defect=True, include_node_image=True)

    with np.load(input_path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}

    payload["image_local_values"] = np.asarray([0.0, 1.0], dtype=np.float64)
    payload["image_local_coordinates"] = np.asarray([[0.0, 0.0]], dtype=np.float64)
    payload["image_local_size"] = np.asarray(2, dtype=np.int64)

    np.savez(input_path, **payload)

    with pytest.raises(ValueError, match="row count"):
        plot_material_qualitative_image_npz(
            input_path=input_path,
            output_png=output_png,
        )


def test_plot_material_qualitative_image_rejects_invalid_defect_radius(
    tmp_path: Path,
):
    input_path = tmp_path / "bad_defect.npz"
    output_png = tmp_path / "bad_defect.png"

    _write_valid_export(input_path, include_defect=True, include_node_image=True)

    with np.load(input_path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}

    payload["defect_radius"] = np.asarray(0.0, dtype=np.float64)
    np.savez(input_path, **payload)

    with pytest.raises(ValueError, match="defect_radius"):
        plot_material_qualitative_image_npz(
            input_path=input_path,
            output_png=output_png,
        )


def test_plot_material_qualitative_image_accepts_empty_local_mpi_rank(
    tmp_path: Path,
):
    input_path = tmp_path / "empty_rank.npz"
    output_dof_png = tmp_path / "empty_rank_dof.png"
    output_node_png = tmp_path / "empty_rank_node.png"
    output_cell_png = tmp_path / "empty_rank_cell.png"

    np.savez(
        input_path,
        kind=np.asarray("material_qualitative_image_export", dtype=np.str_),
        status=np.asarray("ok", dtype=np.str_),
        mpi_rank=np.asarray(1, dtype=np.int64),
        mpi_size=np.asarray(4, dtype=np.int64),
        image_global_l2_norm=np.asarray(1.0, dtype=np.float64),
        image_global_max=np.asarray(1.0, dtype=np.float64),
        normalized=np.asarray(True, dtype=np.bool_),
        image_local_values=np.empty((0,), dtype=np.float64),
        image_local_coordinates=np.empty((0, 2), dtype=np.float64),
        image_coordinate_dim=np.asarray(2, dtype=np.int64),
        image_local_size=np.asarray(0, dtype=np.int64),
        image_global_size=np.asarray(10, dtype=np.int64),
        image_node_values=np.empty((0,), dtype=np.float64),
        image_node_coordinates=np.empty((0, 2), dtype=np.float64),
        image_node_count=np.asarray(0, dtype=np.int64),
        image_node_aggregation=np.asarray("mean", dtype=np.str_),
        image_node_tolerance=np.asarray(1.0e-12, dtype=np.float64),
        image_cell_values=np.empty((0,), dtype=np.float64),
        image_cell_centers=np.empty((0, 2), dtype=np.float64),
        image_cell_count=np.asarray(0, dtype=np.int64),
        image_cell_aggregation=np.asarray("mean", dtype=np.str_),
        image_cell_tolerance=np.asarray(1.0e-12, dtype=np.float64),
        image_cell_min_nodes_per_cell=np.asarray(1, dtype=np.int64),
        defect_center_x=np.asarray(0.5, dtype=np.float64),
        defect_center_y=np.asarray(0.5, dtype=np.float64),
        defect_radius=np.asarray(0.25, dtype=np.float64),
    )

    dof_result = plot_material_qualitative_image_npz(
        input_path=input_path,
        output_png=output_dof_png,
        use_nodal_image=False,
    )

    node_result = plot_material_qualitative_image_npz(
        input_path=input_path,
        output_png=output_node_png,
        use_nodal_image=True,
    )

    assert dof_result["status"] == "ok"
    assert dof_result["image_mode"] == "dof"
    assert dof_result["num_values"] == 0
    assert output_dof_png.exists()
    assert output_dof_png.stat().st_size > 0

    cell_result = plot_material_qualitative_image_npz(
        input_path=input_path,
        output_png=output_cell_png,
        use_cell_image=True,
    )

    assert node_result["status"] == "ok"
    assert node_result["image_mode"] == "node"
    assert node_result["num_values"] == 0
    assert output_node_png.exists()
    assert output_node_png.stat().st_size > 0

    assert cell_result["status"] == "ok"
    assert cell_result["image_mode"] == "cell"
    assert cell_result["num_values"] == 0
    assert output_cell_png.exists()
    assert output_cell_png.stat().st_size > 0



def test_plot_material_qualitative_image_rejects_nodal_and_cell_modes_together(
    tmp_path: Path,
):
    input_path = tmp_path / "mutual_exclusion.npz"
    output_png = tmp_path / "mutual_exclusion.png"

    _write_valid_export(input_path, include_defect=True, include_node_image=True)

    with pytest.raises(ValueError, match="mutually exclusive"):
        plot_material_qualitative_image_npz(
            input_path=input_path,
            output_png=output_png,
            use_nodal_image=True,
            use_cell_image=True,
        )


def test_plot_material_qualitative_image_cell_grid_style_creates_grid_plot(
    tmp_path: Path,
):
    input_path = tmp_path / "cell_grid.npz"
    output_png = tmp_path / "cell_grid.png"

    _write_valid_export(input_path, include_defect=True, include_node_image=True)

    with np.load(input_path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}

    payload["image_cell_values"] = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    payload["image_cell_centers"] = np.asarray(
        [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
        ],
        dtype=np.float64,
    )
    payload["image_cell_count"] = np.asarray(4, dtype=np.int64)

    np.savez(input_path, **payload)

    result = plot_material_qualitative_image_npz(
        input_path=input_path,
        output_png=output_png,
        use_cell_image=True,
        cell_plot_style="grid",
    )

    assert result["status"] == "ok"
    assert result["image_mode"] == "cell"
    assert result["cell_plot_style"] == "grid"
    assert result["num_values"] == 4
    assert output_png.exists()
    assert output_png.stat().st_size > 0


def test_plot_material_qualitative_image_cell_grid_style_falls_back_to_scatter(
    tmp_path: Path,
):
    input_path = tmp_path / "cell_grid_fallback.npz"
    output_png = tmp_path / "cell_grid_fallback.png"

    _write_valid_export(input_path, include_defect=True, include_node_image=True)

    with np.load(input_path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}

    payload["image_cell_values"] = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    payload["image_cell_centers"] = np.asarray(
        [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.50, 0.75],
        ],
        dtype=np.float64,
    )
    payload["image_cell_count"] = np.asarray(3, dtype=np.int64)

    np.savez(input_path, **payload)

    result = plot_material_qualitative_image_npz(
        input_path=input_path,
        output_png=output_png,
        use_cell_image=True,
        cell_plot_style="grid",
    )

    assert result["status"] == "ok"
    assert result["image_mode"] == "cell"
    assert result["cell_plot_style"] == "scatter-fallback"
    assert result["num_values"] == 3
    assert output_png.exists()
    assert output_png.stat().st_size > 0


def test_plot_material_qualitative_image_rejects_invalid_cell_plot_style(
    tmp_path: Path,
):
    input_path = tmp_path / "bad_cell_style.npz"
    output_png = tmp_path / "bad_cell_style.png"

    _write_valid_export(input_path, include_defect=True, include_node_image=True)

    with pytest.raises(ValueError, match="cell_plot_style"):
        plot_material_qualitative_image_npz(
            input_path=input_path,
            output_png=output_png,
            use_cell_image=True,
            cell_plot_style="bad",
        )
