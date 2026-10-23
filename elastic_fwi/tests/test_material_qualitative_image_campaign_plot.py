from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from elastic_fwi.diagnostics.plot_material_qualitative_image_campaign import (
    plot_material_qualitative_image_campaign_summary,
)


def _write_valid_summary_csv(path: Path) -> None:
    rows = [
        {
            "nx": 2,
            "ny": 2,
            "degree": 1,
            "num_steps": 3,
            "mpi_rank": 0,
            "mpi_size": 1,
            "status": "ok",
            "npz": "N0002/material_image.npz",
            "dof_png": "N0002/material_image_dof.png",
            "dof_pdf": "N0002/material_image_dof.pdf",
            "node_png": "N0002/material_image_node.png",
            "node_pdf": "N0002/material_image_node.pdf",
            "image_global_l2_norm": 1.2,
            "image_global_max": 1.0,
            "image_node_count": 9,
            "image_node_aggregation": "mean",
            "image_cell_count": 4,
            "image_cell_aggregation": "mean",
            "defect_metric_status": "ok",
            "defect_metric_max_distance_to_center": 0.25,
            "defect_metric_inside_mass_fraction": 0.10,
            "defect_metric_inside_outside_mean_ratio": 0.50,
        },
        {
            "nx": 4,
            "ny": 4,
            "degree": 1,
            "num_steps": 3,
            "mpi_rank": 0,
            "mpi_size": 1,
            "status": "ok",
            "npz": "N0004/material_image.npz",
            "dof_png": "N0004/material_image_dof.png",
            "dof_pdf": "N0004/material_image_dof.pdf",
            "node_png": "N0004/material_image_node.png",
            "node_pdf": "N0004/material_image_node.pdf",
            "image_global_l2_norm": 1.7,
            "image_global_max": 1.0,
            "image_node_count": 25,
            "image_node_aggregation": "mean",
            "image_cell_count": 16,
            "image_cell_aggregation": "mean",
            "defect_metric_status": "ok",
            "defect_metric_max_distance_to_center": 0.20,
            "defect_metric_inside_mass_fraction": 0.20,
            "defect_metric_inside_outside_mean_ratio": 0.75,
        },
    ]

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_plot_material_qualitative_image_campaign_summary_creates_figures(
    tmp_path: Path,
):
    summary_csv = tmp_path / "summary.csv"
    output_dir = tmp_path / "plots"

    _write_valid_summary_csv(summary_csv)

    result = plot_material_qualitative_image_campaign_summary(
        summary_csv=summary_csv,
        output_dir=output_dir,
    )

    assert result["kind"] == "material_qualitative_image_campaign_plot"
    assert result["status"] == "ok"
    assert result["num_rows"] == 2
    assert result["n_values"] == [2, 4]

    files = result["files"]

    for key in (
        "l2_norm_png",
        "l2_norm_pdf",
        "global_max_png",
        "global_max_pdf",
        "node_count_png",
        "node_count_pdf",
        "cell_count_png",
        "cell_count_pdf",
        "defect_inside_mass_fraction_png",
        "defect_inside_mass_fraction_pdf",
        "defect_max_distance_png",
        "defect_max_distance_pdf",
        "defect_inside_outside_ratio_png",
        "defect_inside_outside_ratio_pdf",
        "manifest_json",
    ):
        path = Path(files[key])
        assert path.exists()
        assert path.stat().st_size > 0

    with Path(files["manifest_json"]).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    assert manifest["kind"] == "material_qualitative_image_campaign_plot"
    assert manifest["status"] == "ok"
    assert manifest["num_rows"] == 2


def test_plot_material_qualitative_image_campaign_summary_rejects_failed_case(
    tmp_path: Path,
):
    summary_csv = tmp_path / "summary_failed.csv"
    output_dir = tmp_path / "plots"

    _write_valid_summary_csv(summary_csv)

    rows = []
    with summary_csv.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)

    rows[0]["status"] = "failed"

    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="failed cases"):
        plot_material_qualitative_image_campaign_summary(
            summary_csv=summary_csv,
            output_dir=output_dir,
        )


def test_plot_material_qualitative_image_campaign_summary_rejects_empty_csv(
    tmp_path: Path,
):
    summary_csv = tmp_path / "empty.csv"
    output_dir = tmp_path / "plots"

    summary_csv.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        plot_material_qualitative_image_campaign_summary(
            summary_csv=summary_csv,
            output_dir=output_dir,
        )
