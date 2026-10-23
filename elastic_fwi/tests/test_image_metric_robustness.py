from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

from elastic_fwi.postprocessing.image_metric_robustness import (
    DefectCircle,
    ImageMap,
    RobustnessConfig,
    TargetZone,
    analyze_image_map,
    analyze_ranking_robustness,
    binary_dilation,
    binary_erosion,
    compare_rankings,
    compute_ranking,
    compute_threshold_metrics,
    discover_campaign_maps,
    load_explicit_mask,
    load_standalone_maps,
    make_circle_zones,
    make_explicit_mask_zones,
    metric_dependencies,
    normalize_qualitative_image,
)


def _grid_map(name: str, values: np.ndarray) -> ImageMap:
    array = np.asarray(values, dtype=float)
    yy, xx = np.indices(array.shape)
    normalized, factor = normalize_qualitative_image(array)
    return ImageMap(
        map_name=name,
        parameter_name="rho",
        modality=name,
        centers=np.column_stack((xx.reshape(-1), yy.reshape(-1))),
        values=normalized,
        weights=np.ones(normalized.size),
        source_path=f"{name}.npy",
        source_format="synthetic",
        grid_shape=array.shape,
        metadata={"normalization_factor": factor},
    )


def _zone(mask: np.ndarray) -> TargetZone:
    yy, xx = np.indices(mask.shape)
    points = np.column_stack((xx[mask], yy[mask]))
    center = points.mean(axis=0)
    return TargetZone(
        zone_id="exact",
        kind="explicit_mask",
        mask=mask.reshape(-1),
        scale=0.0,
        center_x=float(center[0]),
        center_y=float(center[1]),
        characteristic_radius=float("nan"),
    )


def test_threshold_metrics_detect_target_peak_and_weighted_fractions():
    values = np.asarray(
        [
            [0.0, 0.1, 0.0],
            [0.2, 1.0, 0.4],
            [0.0, 0.3, 0.8],
        ]
    )
    image = _grid_map("targeted", values)
    mask = np.zeros_like(values, dtype=bool)
    mask[1, 1] = True
    row = compute_threshold_metrics(image, _zone(mask), threshold=0.5)

    assert row["target_active_fraction"] == pytest.approx(1.0)
    assert row["outside_active_fraction"] == pytest.approx(1.0 / 8.0)
    assert row["peak_inside_target"] is True
    assert row["false_maximum_present"] is False
    assert row["target_peak"] == pytest.approx(1.0)
    assert row["outside_peak"] == pytest.approx(0.8)
    assert row["peak_to_artifact_ratio"] == pytest.approx(1.25)
    assert row["target_energy"] == pytest.approx(1.0)
    assert row["outside_energy"] == pytest.approx(
        0.1**2 + 0.2**2 + 0.4**2 + 0.3**2 + 0.8**2
    )


def test_threshold_metrics_flags_global_peak_outside_target():
    values = np.asarray([[1.0, 0.0], [0.0, 0.5]])
    image = _grid_map("artifact", values)
    mask = np.zeros_like(values, dtype=bool)
    mask[1, 1] = True
    row = compute_threshold_metrics(image, _zone(mask), threshold=0.4)
    assert row["peak_inside_target"] is False
    assert row["false_maximum_present"] is True
    assert row["peak_to_artifact_ratio"] == pytest.approx(0.5)
    assert row["false_maxima_count"] >= 1


def test_normalization_and_invalid_images():
    normalized, factor = normalize_qualitative_image(np.asarray([-2.0, 1.0]))
    assert factor == pytest.approx(2.0)
    assert normalized.tolist() == pytest.approx([1.0, 0.5])
    with pytest.raises(ValueError, match="NaN or Inf"):
        normalize_qualitative_image(np.asarray([1.0, np.nan]))
    with pytest.raises(ValueError, match="identically zero"):
        normalize_qualitative_image(np.zeros(3))


def test_binary_mask_dilation_and_erosion():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    dilated = binary_dilation(mask, 1)
    eroded = binary_erosion(mask, 1)
    assert np.count_nonzero(dilated) == 25
    assert np.count_nonzero(eroded) == 1
    assert eroded[2, 2]


def test_explicit_zone_variants_are_constructed():
    image = _grid_map("map", np.ones((7, 7)))
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    zones = make_explicit_mask_zones(
        image,
        mask.reshape(-1),
        modes=("exact", "dilated", "eroded"),
        dilation_pixels=(1,),
        erosion_pixels=(1,),
    )
    assert {zone.zone_id for zone in zones} == {"exact", "dilated_1px", "eroded_1px"}


def test_circle_zone_variants_use_physical_radius():
    yy, xx = np.indices((9, 9))
    centers = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    zones = make_circle_zones(
        centers,
        (DefectCircle(center_x=4.0, center_y=4.0, radius=2.0),),
        radius_factors=(0.5, 1.0, 1.5),
    )
    counts = [np.count_nonzero(zone.mask) for zone in zones]
    assert counts[0] < counts[1] < counts[2]
    assert "exact" in {zone.zone_id for zone in zones}


def _ranking_rows(scores_by_condition, *, metric="target_energy_ratio"):
    rows = []
    for threshold, zone, scores in scores_by_condition:
        for name, score in scores.items():
            rows.append(
                {
                    "ranking_group": "all",
                    "campaign_id": "synthetic",
                    "geometry_kind": "line",
                    "parameter_name": "rho",
                    "modality": name,
                    "map_name": name,
                    "threshold": threshold,
                    "target_zone_id": zone,
                    metric: score,
                }
            )
    return rows


def test_stable_ranking_returns_ok():
    rows = _ranking_rows(
        [
            (0.3, "exact", {"A": 3.0, "B": 2.0, "C": 1.0}),
            (0.2, "exact", {"A": 4.0, "B": 2.5, "C": 0.8}),
            (0.4, "dilated", {"A": 2.0, "B": 1.5, "C": 0.5}),
        ]
    )
    result = analyze_ranking_robustness(
        rows,
        RobustnessConfig(
            thresholds=(0.2, 0.3, 0.4),
            ranking_metrics=("target_energy_ratio",),
            reference_threshold=0.3,
            score_sensitivity_warning=10.0,
        ),
    )
    assert result.summary["status"] == "ok"
    assert result.summary["num_inversions"] == 0


def test_secondary_inversion_with_same_top_returns_warning():
    rows = _ranking_rows(
        [
            (0.3, "exact", {"A": 3.0, "B": 2.0, "C": 1.0}),
            (0.4, "exact", {"A": 3.0, "B": 1.0, "C": 2.0}),
        ],
        metric="active_target_energy_ratio",
    )
    result = analyze_ranking_robustness(
        rows,
        RobustnessConfig(
            thresholds=(0.3, 0.4),
            ranking_metrics=("active_target_energy_ratio",),
            reference_threshold=0.3,
            fail_top_change_fraction=1.0,
            score_sensitivity_warning=10.0,
        ),
    )
    assert result.summary["status"] == "warning"
    assert result.summary["num_inversions"] == 1


def test_top_ranking_inversion_returns_fail():
    rows = _ranking_rows(
        [
            (0.3, "exact", {"A": 2.0, "B": 1.0}),
            (0.4, "exact", {"A": 0.5, "B": 3.0}),
        ],
        metric="active_target_energy_ratio",
    )
    result = analyze_ranking_robustness(
        rows,
        RobustnessConfig(
            thresholds=(0.3, 0.4),
            ranking_metrics=("active_target_energy_ratio",),
            reference_threshold=0.3,
            fail_top_change_fraction=0.25,
            score_sensitivity_warning=10.0,
        ),
    )
    assert result.summary["status"] == "failed"
    assert result.group_summaries[0]["metrics"][0]["top_change_fraction"] == pytest.approx(1.0)


def test_compare_rankings_reports_complete_reversal():
    reference = [
        {"map_name": "A", "score": 3.0, "rank": 1},
        {"map_name": "B", "score": 2.0, "rank": 2},
        {"map_name": "C", "score": 1.0, "rank": 3},
    ]
    alternative = [
        {"map_name": "C", "score": 3.0, "rank": 1},
        {"map_name": "B", "score": 2.0, "rank": 2},
        {"map_name": "A", "score": 1.0, "rank": 3},
    ]
    comparison = compare_rankings(reference, alternative)
    assert comparison["complete_reversal"] is True
    assert comparison["num_inversions"] == 3


def test_threshold_independent_metric_is_not_repeated_over_thresholds():
    rows = _ranking_rows(
        [
            (0.2, "exact", {"A": 3.0, "B": 2.0, "C": 1.0}),
            (0.3, "exact", {"A": 3.0, "B": 2.0, "C": 1.0}),
            (0.4, "exact", {"A": 3.0, "B": 2.0, "C": 1.0}),
            (0.2, "dilated", {"A": 3.0, "B": 1.0, "C": 2.0}),
            (0.3, "dilated", {"A": 3.0, "B": 1.0, "C": 2.0}),
            (0.4, "dilated", {"A": 3.0, "B": 1.0, "C": 2.0}),
        ]
    )
    result = analyze_ranking_robustness(
        rows,
        RobustnessConfig(
            thresholds=(0.2, 0.3, 0.4),
            ranking_metrics=("target_energy_ratio",),
            reference_threshold=0.3,
            fail_top_change_fraction=1.0,
            score_sensitivity_warning=10.0,
        ),
    )
    metric = result.group_summaries[0]["metrics"][0]
    assert metric["threshold_robustness_status"] == "not_applicable"
    assert metric["zone_robustness_status"] == "warning"
    assert result.summary["num_raw_inversion_occurrences"] == 1
    assert result.summary["num_unique_inversion_patterns"] == 1
    assert len(result.ranking_rows) == 6  # 3 maps × reference/exact+dilated


def test_metric_dependency_registry_distinguishes_thresholded_metrics():
    assert metric_dependencies("target_energy_ratio") == {
        "threshold": False,
        "zone": True,
    }
    assert metric_dependencies("thresholded_target_energy_ratio") == {
        "threshold": True,
        "zone": True,
    }


def test_tie_break_is_reported_without_false_strict_inversion():
    rows = _ranking_rows(
        [
            (0.3, "exact", {"A": 2.0, "B": 2.0, "C": 1.0}),
            (0.4, "exact", {"A": 3.0, "B": 2.0, "C": 1.0}),
        ],
        metric="active_target_energy_ratio",
    )
    result = analyze_ranking_robustness(
        rows,
        RobustnessConfig(
            thresholds=(0.3, 0.4),
            ranking_metrics=("active_target_energy_ratio",),
            reference_threshold=0.3,
            fail_top_change_fraction=1.0,
            score_sensitivity_warning=10.0,
        ),
    )
    metric = result.group_summaries[0]["metrics"][0]
    assert result.summary["num_inversions"] == 0
    assert result.summary["num_tie_change_patterns"] == 1
    assert metric["tie_stability_status"] == "warning"
    assert result.summary["status"] == "warning"


def test_group_summary_exposes_campaign_geometry_and_parameter():
    rows = _ranking_rows(
        [(0.3, "exact", {"A": 2.0, "B": 1.0})]
    )
    result = analyze_ranking_robustness(
        rows,
        RobustnessConfig(
            thresholds=(0.3,),
            ranking_metrics=("target_energy_ratio",),
            reference_threshold=0.3,
            score_sensitivity_warning=10.0,
        ),
    )
    group = result.group_summaries[0]
    assert group["campaign_id"] == "synthetic"
    assert group["geometry_kind"] == "line"
    assert group["parameter_name"] == "rho"


def test_standalone_npz_and_mask_loaders(tmp_path):
    maps_path = tmp_path / "maps.npz"
    mask_path = tmp_path / "mask.npy"
    np.savez(
        maps_path,
        maps=np.stack((np.eye(5), np.fliplr(np.eye(5)))),
        map_names=np.asarray(["A", "B"]),
    )
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    np.save(mask_path, mask)
    maps = load_standalone_maps(maps_path)
    loaded_mask = load_explicit_mask(mask_path, expected_size=25)
    assert len(maps) == 2
    assert maps[0].grid_shape == (5, 5)
    assert np.array_equal(loaded_mask, mask.reshape(-1))


def test_campaign_discovery_reads_real_manifest_schema(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    fake_map = campaign / "map.pvd"
    fake_map.write_text("placeholder", encoding="utf-8")
    manifest = {
        "output_dir": str(campaign),
        "config": {
            "dt": 0.025,
            "num_steps": 800,
            "receiver_array_geometry": "line",
            "defect_center_x": 1.0,
            "defect_center_y": 2.0,
            "defect_radius": 0.25,
        },
        "validation": {
            "found": {
                "classical_full": {"gradient_C66": str(fake_map)},
                "residual_interaction": {"residual_interaction_C66": str(fake_map)},
            }
        },
    }
    (campaign / "gradient_combination_campaign_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    sources = discover_campaign_maps(
        campaign,
        parameters=("C66",),
        require_final_time=20.0,
    )
    assert len(sources) == 2
    assert {source.parameter_name for source in sources} == {"C66"}
    assert all(source.geometry_kind == "line" for source in sources)


def test_cli_generates_csv_json_and_figures(tmp_path):
    maps_path = tmp_path / "maps.npz"
    mask_path = tmp_path / "mask.npy"
    output_dir = tmp_path / "output"
    shape = (9, 9)
    yy, xx = np.indices(shape)
    radius2 = (xx - 4) ** 2 + (yy - 4) ** 2
    map_a = np.exp(-radius2 / 2.0)
    map_b = 0.5 * np.exp(-radius2 / 4.0) + 0.05
    np.savez(
        maps_path,
        maps=np.stack((map_a, map_b)),
        map_names=np.asarray(["A", "B"]),
    )
    mask = radius2 <= 4
    np.save(mask_path, mask)
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "revision_wave_motion"
        / "check_image_metric_robustness.py"
    )
    command = [
        sys.executable,
        str(script),
        "--maps",
        str(maps_path),
        "--mask",
        str(mask_path),
        "--output-dir",
        str(output_dir),
        "--thresholds",
        "0.2,0.4",
        "--target-zone-mode",
        "exact",
        "--metric",
        "target_energy_ratio,peak_to_artifact",
        "--score-sensitivity-warning",
        "10",
        "--overwrite",
        "--quiet",
    ]
    repo_root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root)
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "image_metric_robustness.csv").exists()
    assert (output_dir / "image_metric_robustness_summary.json").exists()
    assert (output_dir / "image_metric_robustness_tie_changes.csv").exists()
    assert (output_dir / "image_metric_robustness_sensitivity.png").exists()
    assert (output_dir / "image_metric_robustness_ranking_stability.png").exists()
    payload = json.loads(
        (output_dir / "image_metric_robustness_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "ok"
    assert payload["analysis_mode"] == "standalone"
    assert payload["num_maps"] == 2


def test_external_reference_ranking_cannot_be_silently_ignored():
    rows = _ranking_rows(
        [
            (0.3, "exact", {"A": 2.0, "B": 1.0}),
            (0.4, "exact", {"A": 0.5, "B": 2.5}),
        ]
    )
    result = analyze_ranking_robustness(
        rows,
        RobustnessConfig(
            thresholds=(0.3, 0.4),
            ranking_metrics=("target_energy_ratio",),
            reference_threshold=0.3,
            fail_top_change_fraction=1.0,
            score_sensitivity_warning=10.0,
        ),
        external_reference_rankings={"all": ("B", "A")},
    )
    assert result.summary["status"] == "warning"
    metric = result.group_summaries[0]["metrics"][0]
    assert metric["reference_condition_inversion"] is True


def test_cli_geometry_scope_contract() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "revision_wave_motion"
        / "check_image_metric_robustness.py"
    )
    spec = importlib.util.spec_from_file_location("image_metric_robustness_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inventory = [
        {"geometry_kind": "line"},
        {"geometry_kind": "line"},
    ]
    mode, geometries = module._resolve_analysis_mode(
        "line", inventory, campaign_mode=True
    )
    assert mode == "line-only"
    assert geometries == ["line"]
    with pytest.raises(ValueError, match="both line and box"):
        module._resolve_analysis_mode(
            "line-box", inventory, campaign_mode=True
        )
