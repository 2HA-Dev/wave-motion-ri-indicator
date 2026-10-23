from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from elastic_fwi.postprocessing.article_figures import (
    ArticleFigureConfig,
    build_article_figures,
    discover_campaigns,
    parse_roi,
    receiver_points,
    source_points,
)


def _small_field() -> SimpleNamespace:
    yy, xx = np.mgrid[0.0:1.01:0.1, 0.0:1.01:0.1]
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    values = np.sin(np.pi * centers[:, 0]) * np.cos(np.pi * centers[:, 1])
    return SimpleNamespace(
        centers=centers,
        values=values,
        cell_measures=np.full(values.size, 0.01),
        selected_field_name="gradient_C11",
        num_ghost_cells_removed=0,
        cell_types=("triangle",),
    )


def _write_campaign(root: Path, *, geometry: str = "line") -> Path:
    root.mkdir(parents=True)
    pvd = root / "map.pvd"
    pvd.write_text("placeholder", encoding="utf-8")
    config = {
        "receiver_array_geometry": "box_except_gamma1" if geometry == "box" else "line",
        "domain_length_x": 10.0,
        "domain_length_y": 10.0,
        "source_array_start_x": 3.0,
        "source_array_start_y": 3.0,
        "source_array_end_x": 7.0,
        "source_array_end_y": 3.0,
        "source_array_num_elements": 5,
        "receiver_array_start_x": 3.0,
        "receiver_array_start_y": 7.0,
        "receiver_array_end_x": 7.0,
        "receiver_array_end_y": 7.0,
        "receiver_array_num_elements": 5,
        "defect_center_x": 5.0,
        "defect_center_y": 5.0,
        "defect_radius": 0.5,
        "dt": 0.025,
        "num_steps": 800,
    }
    if geometry == "box":
        config.update(
            {
                "receiver_box_xmin": 3.0,
                "receiver_box_xmax": 7.0,
                "receiver_box_ymin": 3.0,
                "receiver_box_ymax": 7.0,
                "receiver_box_left_num_elements": 5,
                "receiver_box_top_num_elements": 5,
                "receiver_box_right_num_elements": 5,
            }
        )
    found = {
        "classical_full": {"gradient_C11": str(pvd)},
        "residual_interaction": {"residual_interaction_C11": str(pvd)},
    }
    manifest = root / "gradient_combination_campaign_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "gradient_combination_campaign",
                "campaign_id": f"synthetic_{geometry}",
                "config": config,
                "validation": {"found": found},
                "raw_signed_validation": {"found": found},
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _write_receiver_summary(root: Path) -> None:
    path = root / "diagnostics" / "receiver_phase_spectrum" / "receiver_phase_spectrum_summary.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "residual_family,receiver_id,component,observed_dominant_frequency,cross_correlation_lag_time,cross_correlation_max,observed_temporal_rms\n"
        "reflection,0,x,0.4,0.1,0.8,1e-6\n"
        "reflection,1,x,0.45,0.0,0.9,2e-6\n"
        "transmission,0,y,0.2,0.0,0.99,3e-6\n"
        "transmission,1,y,0.2,0.0,0.995,3.1e-6\n",
        encoding="utf-8",
    )


def _write_metric_summary(root: Path) -> None:
    path = root / "diagnostics" / "image_metric_robustness" / "image_metric_robustness_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "status": "warning",
                "groups": [
                    {
                        "parameter_name": "C11",
                        "threshold_robustness_status": "ok",
                        "zone_robustness_status": "ok",
                        "joint_robustness_status": "warning",
                        "tie_stability_status": "ok",
                    },
                    {
                        "parameter_name": "rho",
                        "threshold_robustness_status": "failed",
                        "zone_robustness_status": "warning",
                        "joint_robustness_status": "failed",
                        "tie_stability_status": "ok",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_existing_figures(root: Path) -> None:
    for path in (
        root / "diagnostics" / "scattered" / "scattered_wavefield_snapshots",
        root.parent / "postprocessing" / "C66_multipolar" / "C66_multipolar_synthetic",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path.with_suffix(".png"), dpi=80)
        plt.close(fig)


def _complete_input(tmp_path: Path) -> Path:
    root = tmp_path / "outputs" / "revision_wave_motion"
    _write_campaign(root / "final_line_T20")
    _write_receiver_summary(root)
    _write_metric_summary(root)
    _write_existing_figures(root)
    return root


def test_campaign_geometry_and_auto_roi(tmp_path: Path) -> None:
    root = tmp_path / "outputs" / "revision_wave_motion"
    _write_campaign(root / "line", geometry="line")
    _write_campaign(root / "box", geometry="box")
    config = ArticleFigureConfig(input_root=root, output_dir=tmp_path / "out")
    campaigns = discover_campaigns(config)
    assert {item.geometry_kind for item in campaigns} == {"line", "box"}
    assert source_points(campaigns[0]).shape[1] == 2
    box = next(item for item in campaigns if item.geometry_kind == "box")
    assert receiver_points(box).shape[0] > 5
    roi = parse_roi("auto", campaigns)
    assert roi[0] < 3.0 < roi[1]
    assert roi[2] < 3.0 < roi[3]
    assert roi[0] < 7.0 < roi[1]


def test_build_complete_article_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _complete_input(tmp_path)
    monkeypatch.setattr("elastic_fwi.postprocessing.article_figures.load_vtk_cell_field", lambda *args, **kwargs: _small_field())
    output = tmp_path / "article"
    result = build_article_figures(
        ArticleFigureConfig(
            input_root=root,
            output_dir=output,
            formats=("pdf", "png"),
            parameters=("C11",),
            modalities=("classical_full", "residual_interaction"),
            signed_modalities=("classical_full", "residual_interaction"),
            overwrite=True,
        ),
        command=("python", "test"),
    )
    assert result.exit_code == 0
    assert result.status == "ok"
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["format_version"] == "wave_motion_article_figures_v1"
    assert payload["summary"]["num_figures_skipped"] == 0
    assert payload["scientific_scope"]["no_simulation_or_gradient_recomputation"] is True
    assert result.summary_path.is_file()
    for record in result.records:
        assert record.status == "ok"
        assert all(Path(path).is_file() for path in record.outputs)


def test_missing_groups_are_explicitly_skipped(tmp_path: Path) -> None:
    root = tmp_path / "outputs" / "revision_wave_motion"
    root.mkdir(parents=True)
    result = build_article_figures(
        ArticleFigureConfig(
            input_root=root,
            output_dir=tmp_path / "out",
            figure_groups=("receiver_summary", "image_metric_summary"),
            overwrite=True,
        )
    )
    assert result.status == "failed"
    assert result.exit_code == 2
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert all(item["status"] == "skipped" for item in payload["figures"])
    assert all(item["reason"] for item in payload["figures"])


def test_partial_non_strict_run_returns_success(tmp_path: Path) -> None:
    root = tmp_path / "outputs" / "revision_wave_motion"
    _write_receiver_summary(root)
    result = build_article_figures(
        ArticleFigureConfig(
            input_root=root,
            output_dir=tmp_path / "out",
            figure_groups=("receiver_summary", "image_metric_summary"),
            overwrite=True,
        )
    )
    assert result.status == "partial"
    assert result.exit_code == 0


def test_strict_run_fails_when_requested_group_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "outputs" / "revision_wave_motion"
    _write_receiver_summary(root)
    result = build_article_figures(
        ArticleFigureConfig(
            input_root=root,
            output_dir=tmp_path / "out",
            figure_groups=("receiver_summary", "image_metric_summary"),
            overwrite=True,
            strict=True,
        )
    )
    assert result.status == "failed"
    assert result.exit_code == 2


def test_output_directory_requires_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write_receiver_summary(root)
    output = tmp_path / "out"
    output.mkdir()
    (output / "existing.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_article_figures(
            ArticleFigureConfig(
                input_root=root,
                output_dir=output,
                figure_groups=("receiver_summary",),
            )
        )


def test_manifest_distinguishes_signed_and_normalized_maps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "outputs" / "revision_wave_motion"
    _write_campaign(root / "campaign")
    monkeypatch.setattr("elastic_fwi.postprocessing.article_figures.load_vtk_cell_field", lambda *args, **kwargs: _small_field())
    result = build_article_figures(
        ArticleFigureConfig(
            input_root=root,
            output_dir=tmp_path / "out",
            figure_groups=("normalized_maps", "signed_maps"),
            parameters=("C11",),
            modalities=("classical_full",),
            signed_modalities=("classical_full",),
            overwrite=True,
        )
    )
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    kinds = {item["conventions"].get("field_kind") for item in payload["figures"]}
    assert kinds == {"signed", "absolute_normalized"}


def test_cli_help() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "revision_wave_motion" / "build_wave_motion_article_figures.py"
    completed = subprocess.run([sys.executable, str(script), "--help"], text=True, capture_output=True, check=False)
    assert completed.returncode == 0
    assert "--input-root" in completed.stdout
    assert "--strict" in completed.stdout
