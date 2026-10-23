from __future__ import annotations

import csv
import json
from pathlib import Path

from mpi4py import MPI

from elastic_fwi.diagnostics.run_material_qualitative_image_campaign import (
    run_material_qualitative_image_campaign,
)


def test_material_qualitative_image_campaign_generates_summary_and_manifests(
    tmp_path: Path,
):
    comm = MPI.COMM_WORLD

    # In serial, a genuine mini-campaign on two meshes is checked.
    # Under MPI, a short check is kept: each rank generates its artefacts,
    # and some ranks may legitimately own no local entities.
    if comm.size == 1:
        n_values = (2, 4)
    else:
        n_values = (2,)

    result = run_material_qualitative_image_campaign(
        output_dir=tmp_path,
        n_values=n_values,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=2,
        node_aggregation="mean",
        comm=comm,
    )

    assert result["kind"] == "material_qualitative_image_campaign"
    assert result["status"] == "ok"
    assert result["num_cases"] == len(n_values)
    assert result["mpi_rank"] == comm.rank
    assert result["mpi_size"] == comm.size
    assert result["summary_plots"] is not None

    diagnostics = result["diagnostics"]
    assert diagnostics["kind"] == "material_qualitative_image_campaign_diagnostics"
    assert diagnostics["status"] == "ok"
    assert diagnostics["mpi_rank"] == comm.rank
    assert diagnostics["mpi_size"] == comm.size
    assert diagnostics["n_values"] == list(n_values)
    assert diagnostics["l2_positive"] is True
    assert diagnostics["l2_finite"] is True
    assert diagnostics["max_positive"] is True
    assert diagnostics["max_finite"] is True
    assert diagnostics["normalize"] is True
    assert diagnostics["max_close_to_one"] is True
    assert diagnostics["max_abs_deviation_from_one"] <= 1.0e-12
    assert diagnostics["global_node_count_positive"] is True
    assert diagnostics["global_node_count_monotone_non_decreasing"] is True
    assert len(diagnostics["global_node_counts"]) == len(n_values)
    assert len(diagnostics["local_node_counts"]) == len(n_values)
    assert len(diagnostics["l2_growth_ratios"]) == max(len(n_values) - 1, 0)
    assert len(diagnostics["global_node_count_growth_ratios"]) == max(len(n_values) - 1, 0)

    summary_csv = Path(result["summary_csv"])
    manifest_json = Path(result["manifest_json"])
    summary_plot_manifest = Path(result["summary_plot_manifest"])

    assert summary_csv.exists()
    assert manifest_json.exists()
    assert summary_plot_manifest.exists()
    assert summary_csv.stat().st_size > 0
    assert manifest_json.stat().st_size > 0
    assert summary_plot_manifest.stat().st_size > 0

    with summary_csv.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == len(n_values)
    assert [int(row["nx"]) for row in rows] == list(n_values)
    assert [int(row["ny"]) for row in rows] == list(n_values)

    local_node_count_sum = 0

    for row in rows:
        local_node_count = int(row["image_node_count"])
        local_node_count_sum += local_node_count

        assert row["status"] == "ok"
        assert int(row["degree"]) == 1
        assert int(row["num_steps"]) == 2
        assert int(row["mpi_rank"]) == comm.rank
        assert int(row["mpi_size"]) == comm.size

        assert float(row["image_global_l2_norm"]) > 0.0
        assert float(row["image_global_max"]) > 0.0

        # Under MPI, a rank may own no local entities for N=2.
        assert local_node_count >= 0

        if comm.size == 1:
            assert local_node_count > 0

        assert row["image_node_aggregation"] == "mean"

        for key in ("npz", "dof_png", "dof_pdf", "node_png", "node_pdf"):
            path = Path(row[key])
            assert path.exists()
            assert path.stat().st_size > 0

    global_node_count_sum = comm.allreduce(local_node_count_sum, op=MPI.SUM)
    assert global_node_count_sum > 0

    with manifest_json.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    assert manifest["kind"] == "material_qualitative_image_campaign"
    assert manifest["status"] == "ok"
    assert manifest["mpi_rank"] == comm.rank
    assert manifest["mpi_size"] == comm.size
    assert manifest["config"]["n_values"] == list(n_values)
    assert manifest["config"]["num_steps"] == 2
    assert manifest["config"]["node_aggregation"] == "mean"
    assert manifest["config"]["generate_summary_plots"] is True
    assert len(manifest["cases"]) == len(n_values)

    assert manifest["summary_plots"] is not None
    assert manifest["summary_plots"]["status"] == "ok"
    assert manifest["summary_plots"]["num_rows"] == len(n_values)

    assert manifest["diagnostics"]["status"] == "ok"
    assert manifest["diagnostics"]["global_node_count_positive"] is True
    assert manifest["diagnostics"]["global_node_count_monotone_non_decreasing"] is True
    assert manifest["diagnostics"]["max_close_to_one"] is True

    for key, value in manifest["summary_plots"]["files"].items():
        path = Path(value)
        assert path.exists()
        assert path.stat().st_size > 0

    for case in manifest["cases"]:
        assert Path(case["artifact_manifest"]).exists()
        assert case["summary"]["status"] == "ok"

        with Path(case["artifact_manifest"]).open("r", encoding="utf-8") as stream:
            artifact_manifest = json.load(stream)

        assert artifact_manifest["config"]["cell_plot_style"] == "grid"
        assert artifact_manifest["plots"]["cell"]["image_mode"] == "cell"
        assert artifact_manifest["plots"]["cell"]["cell_plot_style"] in {
            "grid",
            "scatter-fallback",
        }

        case_node_count = int(case["summary"]["image_node_count"])
        assert case_node_count >= 0

        if comm.size == 1:
            assert case_node_count > 0


def test_material_qualitative_image_campaign_can_disable_summary_plots(
    tmp_path: Path,
):
    comm = MPI.COMM_WORLD

    result = run_material_qualitative_image_campaign(
        output_dir=tmp_path,
        n_values=(2,),
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=2,
        node_aggregation="mean",
        generate_summary_plots=False,
        comm=comm,
    )

    assert result["status"] == "ok"
    assert result["summary_plot_manifest"] == ""
    assert result["summary_plots"] is None
    assert result["diagnostics"]["status"] == "ok"
    assert result["diagnostics"]["global_node_count_positive"] is True

    with Path(result["manifest_json"]).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    assert manifest["config"]["generate_summary_plots"] is False
    assert manifest["summary_plots"] is None
    assert manifest["diagnostics"]["status"] == "ok"
    assert manifest["diagnostics"]["global_node_count_positive"] is True
    assert "summary_plot_manifest" not in manifest["files"]


def test_material_qualitative_image_campaign_writes_final_assessment(tmp_path: Path):
    comm = MPI.COMM_WORLD

    result = run_material_qualitative_image_campaign(
        output_dir=tmp_path,
        n_values=(2, 4, 8),
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        node_aggregation="mean",
        cell_aggregation="mean",
        cell_min_nodes_per_cell=1,
        cell_plot_style="grid",
        comm=comm,
    )

    assert result["status"] == "ok"
    assert result["final_assessment"]["kind"] == (
        "material_qualitative_image_campaign_final_assessment"
    )
    assert result["final_assessment"]["status"] == "measured"
    assert result["final_assessment"]["num_cases"] == 3
    assert result["final_assessment"]["interpretation"] == (
        "diagnostic_only_no_success_assertion"
    )

    manifest_path = Path(result["manifest"])

    assert manifest_path.exists()

    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    assessment = manifest["final_assessment"]

    assert assessment["kind"] == "material_qualitative_image_campaign_final_assessment"
    assert assessment["status"] == "measured"
    assert assessment["num_cases"] == 3
    assert assessment["interpretation"] == "diagnostic_only_no_success_assertion"
    assert assessment["best_n_by_inside_mass_fraction"] in {2, 4, 8}
    assert assessment["best_n_by_max_distance"] in {2, 4, 8}
    assert isinstance(assessment["inside_mass_fraction_trend_non_decreasing"], bool)
    assert isinstance(assessment["max_distance_trend_non_increasing"], bool)


def test_material_qualitative_image_campaign_writes_final_assessment(tmp_path: Path):
    comm = MPI.COMM_WORLD

    result = run_material_qualitative_image_campaign(
        output_dir=tmp_path,
        n_values=(2, 4, 8),
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        node_aggregation="mean",
        cell_aggregation="mean",
        cell_min_nodes_per_cell=1,
        cell_plot_style="grid",
        comm=comm,
    )

    assert result["status"] == "ok"
    assert result["final_assessment"]["kind"] == (
        "material_qualitative_image_campaign_final_assessment"
    )
    assert result["final_assessment"]["status"] == "measured"
    assert result["final_assessment"]["num_cases"] == 3
    assert result["final_assessment"]["interpretation"] == (
        "diagnostic_only_no_success_assertion"
    )

    manifest_path = Path(result["manifest"])

    assert manifest_path.exists()

    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    assessment = manifest["final_assessment"]

    assert assessment["kind"] == "material_qualitative_image_campaign_final_assessment"
    assert assessment["status"] == "measured"
    assert assessment["num_cases"] == 3
    assert assessment["interpretation"] == "diagnostic_only_no_success_assertion"
    assert assessment["best_n_by_inside_mass_fraction"] in {2, 4, 8}
    assert assessment["best_n_by_max_distance"] in {2, 4, 8}
    assert isinstance(assessment["inside_mass_fraction_trend_non_decreasing"], bool)
    assert isinstance(assessment["max_distance_trend_non_increasing"], bool)
