from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from elastic_fwi.postprocessing.receiver_aperture_comparison import (
    ApertureComparisonConfig,
    ApertureComparisonInputs,
    ReceiverApertureError,
    ReceiverGeometryError,
    ResidualCSVError,
    TimeConventionError,
    build_receiver_geometry,
    compute_aperture_summaries,
    compute_receiver_rms_by_component,
    compute_receiver_rms_vector,
    load_and_validate_campaign,
    load_receiver_specs,
    run_receiver_aperture_comparison,
    trapezoidal_point_weights,
    validate_common_apertures,
    write_receiver_aperture_outputs,
)


def _line_centers(y: float) -> list[list[float]]:
    return [[30.0 + 0.65 * index, y] for index in range(21)]


def _box_centers() -> list[list[float]]:
    left = [[30.0, 30.0 + 0.65 * index] for index in range(21)]
    top = [[30.0 + 0.65 * index, 43.0] for index in range(1, 21)]
    right = [[43.0, 43.0 - 0.65 * index] for index in range(1, 21)]
    return left + top + right


def _array_spec(
    *,
    centers: list[list[float]],
    name: str,
    role: str,
    geometry: str,
) -> dict:
    return {
        "centers": centers,
        "name": name,
        "role": role,
        "geometry": geometry,
        "num_elements": len(centers),
        "spacing": 0.65,
        "sigma": 0.325,
    }


def _write_manifest(path: Path, *, geometry: str) -> Path:
    reflection = _array_spec(
        centers=_line_centers(30.0),
        name="gamma1",
        role="source_reflection",
        geometry="line",
    )
    transmission = _array_spec(
        centers=_line_centers(43.0) if geometry == "line" else _box_centers(),
        name="gamma2",
        role="transmission",
        geometry="line" if geometry == "line" else "explicit_centers",
    )
    payload = {
        "extra": {
            "comparison": {
                "observed_reflection": {"spec": {"array": reflection}},
                "simulated_reflection": {"spec": {"array": reflection}},
                "observed_transmission": {"spec": {"array": transmission}},
                "simulated_transmission": {"spec": {"array": transmission}},
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _residual_value(
    *,
    geometry: str,
    family: str,
    receiver_id: int,
    component: str,
    step: int,
) -> float:
    component_factor = 1.0 if component == "x" else 2.0
    family_factor = 1.0 if family == "reflection" else 3.0
    geometry_factor = 1.0 if geometry == "line" else 2.0
    # Preserve identical common apertures between line and box.
    if family == "reflection":
        geometry_factor = 1.0
    if family == "transmission" and geometry == "box" and 20 <= receiver_id <= 40:
        geometry_factor = 1.0
        receiver_id = receiver_id - 20
    return geometry_factor * family_factor * component_factor * (receiver_id + 1) * (step + 1)


def _write_csv(
    path: Path,
    *,
    geometry: str,
    num_steps: int = 4,
    physical_time: bool = True,
    global_sign: float = 1.0,
    shuffle: bool = True,
) -> Path:
    rows: list[dict] = []
    counts = {"reflection": 21, "transmission": 21 if geometry == "line" else 61}
    for family, count in counts.items():
        array = "gamma1" if family == "reflection" else "gamma2"
        role = "source_reflection" if family == "reflection" else "transmission"
        for step in range(num_steps):
            for receiver_id in range(count):
                for component in ("x", "y"):
                    residual = global_sign * _residual_value(
                        geometry=geometry,
                        family=family,
                        receiver_id=receiver_id,
                        component=component,
                        step=step,
                    )
                    observed = -0.25 * residual
                    reference = observed + residual
                    rows.append(
                        {
                            "time": 0.05 * step if physical_time else float(step),
                            "step": step,
                            "array": array,
                            "role": role,
                            "receiver_id": receiver_id,
                            "component": component,
                            "residual": residual,
                            "reference": reference,
                            "observed": observed,
                        }
                    )
    frame = pd.DataFrame(rows)
    if shuffle:
        frame = frame.sample(frac=1.0, random_state=123).reset_index(drop=True)
    frame.to_csv(path, index=False)
    return path


def _config(
    tmp_path: Path,
    *,
    physical_time: bool = True,
    time_policy: str = "strict-physical",
    sign: float = 1.0,
) -> ApertureComparisonConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    line_manifest = _write_manifest(tmp_path / "line_manifest.json", geometry="line")
    box_manifest = _write_manifest(tmp_path / "box_manifest.json", geometry="box")
    line_csv = _write_csv(
        tmp_path / "line.csv",
        geometry="line",
        physical_time=physical_time,
        global_sign=sign,
    )
    box_csv = _write_csv(
        tmp_path / "box.csv",
        geometry="box",
        physical_time=physical_time,
        global_sign=sign,
    )
    return ApertureComparisonConfig(
        inputs=ApertureComparisonInputs(
            line_residuals=line_csv,
            box_residuals=box_csv,
            line_manifest=line_manifest,
            box_manifest=box_manifest,
        ),
        output_dir=tmp_path / "out",
        time_column_policy=time_policy,
    )


def _line_only_config(
    tmp_path: Path,
    *,
    physical_time: bool = True,
    time_policy: str = "strict-physical",
) -> ApertureComparisonConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    line_manifest = _write_manifest(tmp_path / "line_manifest.json", geometry="line")
    line_csv = _write_csv(
        tmp_path / "line.csv",
        geometry="line",
        physical_time=physical_time,
    )
    return ApertureComparisonConfig(
        inputs=ApertureComparisonInputs(
            line_residuals=line_csv,
            line_manifest=line_manifest,
        ),
        output_dir=tmp_path / "out",
        geometry="line",
        time_column_policy=time_policy,
    )


def test_step_index_to_physical_time(tmp_path: Path) -> None:
    config = _config(tmp_path)
    specs = load_receiver_specs(config.inputs.line_manifest, geometry="line")
    frame, inspection = load_and_validate_campaign(
        config.inputs.line_residuals,
        geometry="line",
        specs=specs,
        config=config,
    )
    assert inspection.time_status == "physical_time"
    assert np.allclose(frame["time_physical"], 0.05 * frame["step_index"])
    assert frame["time_physical"].max() == pytest.approx(0.15)


def test_legacy_time_alias_requires_explicit_policy(tmp_path: Path) -> None:
    config = _config(tmp_path, physical_time=False)
    specs = load_receiver_specs(config.inputs.line_manifest, geometry="line")
    with pytest.raises(TimeConventionError, match="legacy-step-index"):
        load_and_validate_campaign(
            config.inputs.line_residuals,
            geometry="line",
            specs=specs,
            config=config,
        )


def test_legacy_time_alias_is_converted_when_acknowledged(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        physical_time=False,
        time_policy="legacy-step-index",
    )
    result = run_receiver_aperture_comparison(config)
    assert set(result.time_validation["time_column_status"]) == {
        "legacy_step_index_alias"
    }
    assert result.time_validation["time_physical_max"].iloc[0] == pytest.approx(0.15)
    assert result.time_validation["time_physical_max"].iloc[1] == pytest.approx(0.15)


def test_incoherent_time_column_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    frame = pd.read_csv(config.inputs.line_residuals)
    frame.loc[0, "time"] = 99.0
    frame.to_csv(config.inputs.line_residuals, index=False)
    specs = load_receiver_specs(config.inputs.line_manifest, geometry="line")
    with pytest.raises(TimeConventionError, match="incompatible CSV time"):
        load_and_validate_campaign(
            config.inputs.line_residuals,
            geometry="line",
            specs=specs,
            config=config,
        )


def test_duplicate_logical_record_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    frame = pd.read_csv(config.inputs.line_residuals)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_csv(config.inputs.line_residuals, index=False)
    specs = load_receiver_specs(config.inputs.line_manifest, geometry="line")
    with pytest.raises(ResidualCSVError, match="duplicate logical residual"):
        load_and_validate_campaign(
            config.inputs.line_residuals,
            geometry="line",
            specs=specs,
            config=config,
        )


def test_missing_value_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    frame = pd.read_csv(config.inputs.line_residuals)
    frame.loc[0, "residual"] = np.nan
    frame.to_csv(config.inputs.line_residuals, index=False)
    specs = load_receiver_specs(config.inputs.line_manifest, geometry="line")
    with pytest.raises(ResidualCSVError, match="missing values"):
        load_and_validate_campaign(
            config.inputs.line_residuals,
            geometry="line",
            specs=specs,
            config=config,
        )


def test_missing_component_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    frame = pd.read_csv(config.inputs.line_residuals)
    frame = frame.loc[frame["component"] != "y"]
    frame.to_csv(config.inputs.line_residuals, index=False)
    specs = load_receiver_specs(config.inputs.line_manifest, geometry="line")
    with pytest.raises(ResidualCSVError, match="required components"):
        load_and_validate_campaign(
            config.inputs.line_residuals,
            geometry="line",
            specs=specs,
            config=config,
        )


def test_scalar_and_vector_rms(tmp_path: Path) -> None:
    result = run_receiver_aperture_comparison(_config(tmp_path))
    component = result.rms_by_component.loc[
        (result.rms_by_component["geometry"] == "line")
        & (result.rms_by_component["residual_family"] == "reflection")
        & (result.rms_by_component["receiver_id"] == 0)
        & (result.rms_by_component["component"] == "x")
    ].iloc[0]
    expected_x = np.sqrt(np.mean(np.asarray([1.0, 2.0, 3.0, 4.0]) ** 2))
    assert component["rms"] == pytest.approx(expected_x)
    vector = result.rms_vector.loc[
        (result.rms_vector["geometry"] == "line")
        & (result.rms_vector["residual_family"] == "reflection")
        & (result.rms_vector["receiver_id"] == 0)
    ].iloc[0]
    assert vector["rms_vector_xy"] == pytest.approx(np.sqrt(5.0) * expected_x)


def test_global_normalization_by_receiver_count(tmp_path: Path) -> None:
    result = run_receiver_aperture_comparison(_config(tmp_path))
    row = result.rms_by_side.loc[
        (result.rms_by_side["geometry"] == "line")
        & (result.rms_by_side["residual_family"] == "reflection")
        & (result.rms_by_side["scope"] == "all")
    ].iloc[0]
    assert row["rms_global_normalized"] == pytest.approx(
        row["rms_global_sum"] / np.sqrt(row["num_receivers"])
    )


def test_trapezoidal_spatial_weights() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    weights = trapezoidal_point_weights(points)
    assert np.allclose(weights, [0.5, 1.5, 1.0])
    assert weights.sum() == pytest.approx(3.0)


def test_box_side_decomposition_and_shared_corners(tmp_path: Path) -> None:
    line_specs = load_receiver_specs(
        _write_manifest(tmp_path / "line.json", geometry="line"), geometry="line"
    )
    box_specs = load_receiver_specs(
        _write_manifest(tmp_path / "box.json", geometry="box"), geometry="box"
    )
    table = build_receiver_geometry(
        {"line": line_specs, "box": box_specs},
        geometry_atol=1.0e-12,
        corner_policy="shared",
    )
    box = table.loc[
        (table["geometry"] == "box")
        & (table["residual_family"] == "transmission")
    ]
    assert len(box) == 61
    assert box["is_corner"].sum() == 2
    assert box["weight_all"].sum() == pytest.approx(39.0)
    assert box["weight_left"].sum() == pytest.approx(13.0)
    assert box["weight_top"].sum() == pytest.approx(13.0)
    assert box["weight_right"].sum() == pytest.approx(13.0)
    assert np.allclose(
        box["weight_all"],
        box["weight_left"] + box["weight_top"] + box["weight_right"],
    )


def test_reflection_and_transmission_are_separate(tmp_path: Path) -> None:
    result = run_receiver_aperture_comparison(_config(tmp_path))
    assert set(result.rms_by_side["residual_family"]) == {
        "reflection",
        "transmission",
    }
    assert len(result.line_box_summary) == 2


def test_rms_is_invariant_under_global_sign_change(tmp_path: Path) -> None:
    positive = run_receiver_aperture_comparison(_config(tmp_path / "positive", sign=1.0))
    negative = run_receiver_aperture_comparison(_config(tmp_path / "negative", sign=-1.0))
    cols = ["geometry", "residual_family", "receiver_id", "rms_vector_xy"]
    pd.testing.assert_frame_equal(
        positive.rms_vector[cols].reset_index(drop=True),
        negative.rms_vector[cols].reset_index(drop=True),
    )


def test_line_and_box_receiver_counts(tmp_path: Path) -> None:
    result = run_receiver_aperture_comparison(_config(tmp_path))
    counts = {
        (row.geometry, row.residual_family): row.num_receivers
        for row in result.rms_by_side.itertuples()
        if row.scope == "all"
    }
    assert counts[("line", "reflection")] == 21
    assert counts[("line", "transmission")] == 21
    assert counts[("box", "reflection")] == 21
    assert counts[("box", "transmission")] == 61


def test_side_energy_contributions_sum_to_one(tmp_path: Path) -> None:
    result = run_receiver_aperture_comparison(_config(tmp_path))
    sides = result.rms_by_side.loc[
        (result.rms_by_side["geometry"] == "box")
        & (result.rms_by_side["residual_family"] == "transmission")
        & (result.rms_by_side["scope"].isin(["left", "top", "right"]))
    ]
    assert sides["energy_contribution_to_box_all"].sum() == pytest.approx(1.0)


def test_common_apertures_are_identical(tmp_path: Path) -> None:
    result = run_receiver_aperture_comparison(_config(tmp_path))
    assert (result.common_aperture_consistency["status"] == "ok").all()
    assert result.common_aperture_consistency["max_abs_difference"].max() == 0.0


def test_common_aperture_mismatch_is_reported(tmp_path: Path) -> None:
    config = _config(tmp_path)
    box = pd.read_csv(config.inputs.box_residuals)
    mask = (
        (box["role"] == "transmission")
        & (box["receiver_id"] == 20)
        & (box["component"] == "x")
        & (box["step"] == 0)
    )
    box.loc[mask, "residual"] += 1.0
    box.loc[mask, "reference"] += 1.0
    box.to_csv(config.inputs.box_residuals, index=False)
    with pytest.raises(ReceiverApertureError, match="Common-aperture consistency failed"):
        run_receiver_aperture_comparison(config)


def test_residual_convention_is_reference_minus_observed(tmp_path: Path) -> None:
    result = run_receiver_aperture_comparison(_config(tmp_path))
    assert {item.residual_convention for item in result.inspections} == {
        "reference_minus_observed"
    }


def test_row_order_does_not_change_metrics(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = run_receiver_aperture_comparison(config)
    for path in (config.inputs.line_residuals, config.inputs.box_residuals):
        frame = pd.read_csv(path)
        frame.sample(frac=1.0, random_state=999).to_csv(path, index=False)
    second = run_receiver_aperture_comparison(config)
    pd.testing.assert_frame_equal(first.rms_vector, second.rms_vector)
    pd.testing.assert_frame_equal(first.rms_by_side, second.rms_by_side)


def test_csv_json_latex_and_figures_are_generated(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_receiver_aperture_comparison(config)
    outputs = write_receiver_aperture_outputs(
        result,
        config,
        command=["python", "compare_line_box_receiver_apertures.py"],
        repository_root=tmp_path,
    )
    required = {
        "receiver_rms_by_component",
        "receiver_rms_vector",
        "receiver_rms_by_side",
        "line_box_summary",
        "time_validation_summary",
        "receiver_geometry_summary",
        "common_aperture_consistency",
        "representative_signals",
        "latex_table",
        "summary_json",
        "manifest_json",
        "figure_rms_reflection",
        "figure_rms_transmission",
        "figure_box_sides",
        "figure_signals_reflection",
        "figure_signals_transmission",
        "figure_summary",
    }
    assert required <= set(outputs)
    assert all(outputs[name].exists() for name in required)
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["parameters"]["dt"] == pytest.approx(0.05)


def test_invalid_corner_policy_is_rejected(tmp_path: Path) -> None:
    line_specs = load_receiver_specs(
        _write_manifest(tmp_path / "line.json", geometry="line"), geometry="line"
    )
    box_specs = load_receiver_specs(
        _write_manifest(tmp_path / "box.json", geometry="box"), geometry="box"
    )
    with pytest.raises(ReceiverGeometryError, match="shared"):
        build_receiver_geometry(
            {"line": line_specs, "box": box_specs},
            geometry_atol=1.0e-12,
            corner_policy="single-owner",
        )


def test_line_only_run_uses_only_line_campaign(tmp_path: Path) -> None:
    result = run_receiver_aperture_comparison(_line_only_config(tmp_path))
    assert len(result.inspections) == 1
    assert result.inspections[0].geometry == "line"
    assert set(result.canonical_residuals["geometry"]) == {"line"}
    assert set(result.receiver_geometry["geometry"]) == {"line"}
    assert result.line_box_summary.empty
    assert result.common_aperture_consistency.empty
    assert result.summary["analysis_mode"] == "line-only"
    assert result.summary["geometries"] == ["line"]
    assert set(result.summary["line_aperture_metrics"]) == {
        "reflection",
        "transmission",
    }


def test_line_only_outputs_exclude_box_products(tmp_path: Path) -> None:
    config = _line_only_config(tmp_path)
    result = run_receiver_aperture_comparison(config)
    outputs = write_receiver_aperture_outputs(
        result,
        config,
        command=["python", "compare_line_box_receiver_apertures.py", "--geometry", "line"],
        repository_root=tmp_path,
    )
    required = {
        "receiver_rms_by_component",
        "receiver_rms_vector",
        "receiver_rms_by_side",
        "line_aperture_summary",
        "time_validation_summary",
        "receiver_geometry_summary",
        "representative_signals",
        "latex_table",
        "summary_json",
        "manifest_json",
        "figure_rms_reflection",
        "figure_rms_transmission",
        "figure_signals_reflection",
        "figure_signals_transmission",
        "figure_summary",
    }
    forbidden = {
        "line_box_summary",
        "common_aperture_consistency",
        "figure_box_sides",
    }
    assert required <= set(outputs)
    assert forbidden.isdisjoint(outputs)
    assert all(outputs[name].exists() for name in required)
    assert outputs["figure_summary"].name == "line_weighted_rms_summary.png"
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["analysis_mode"] == "line-only"
    assert manifest["parameters"]["geometry"] == "line"
    assert set(manifest["inputs"]) == {"line_residuals", "line_manifest"}
    assert manifest["validation"]["common_aperture_status"] == "not_applicable"


def test_line_box_mode_requires_box_inputs(tmp_path: Path) -> None:
    line_manifest = _write_manifest(tmp_path / "line_manifest.json", geometry="line")
    line_csv = _write_csv(tmp_path / "line.csv", geometry="line")
    with pytest.raises(ValueError, match="box_residuals and box_manifest"):
        ApertureComparisonConfig(
            inputs=ApertureComparisonInputs(
                line_residuals=line_csv,
                line_manifest=line_manifest,
            ),
            output_dir=tmp_path / "out",
        )


def test_cli_accepts_line_only_mode(tmp_path: Path) -> None:
    config = _line_only_config(tmp_path / "inputs")
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "revision_wave_motion"
        / "compare_line_box_receiver_apertures.py"
    )
    output_dir = tmp_path / "cli_out"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--geometry",
            "line",
            "--line-residuals",
            str(config.inputs.line_residuals),
            "--line-manifest",
            str(config.inputs.line_manifest),
            "--dt",
            "0.05",
            "--time-column-policy",
            "strict-physical",
            "--output-dir",
            str(output_dir),
            "--quiet",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "csv" / "line_aperture_summary.csv").exists()
    assert not (output_dir / "csv" / "line_box_summary.csv").exists()



def test_line_only_one_based_steps_preserve_physical_time_bounds(tmp_path: Path) -> None:
    config = _line_only_config(tmp_path)
    frame = pd.read_csv(config.inputs.line_residuals)
    frame["step"] = frame["step"] + 1
    frame["time"] = 0.05 * frame["step"]
    frame.to_csv(config.inputs.line_residuals, index=False)

    result = run_receiver_aperture_comparison(config)
    all_scopes = result.rms_by_side.loc[result.rms_by_side["scope"] == "all"]
    assert set(all_scopes["step_index_min"]) == {1}
    assert set(all_scopes["step_index_max"]) == {4}
    assert np.allclose(all_scopes["time_initial"], 0.05)
    assert np.allclose(all_scopes["time_final"], 0.20)
