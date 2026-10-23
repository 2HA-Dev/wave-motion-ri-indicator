from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import elastic_fwi.postprocessing.vector_array_waveforms as waveform_module

from elastic_fwi.postprocessing.receiver_aperture_comparison import (
    ApertureComparisonConfig,
    ApertureComparisonInputs,
    ResidualCSVError,
    TimeConventionError,
    load_and_validate_campaign,
    load_receiver_specs,
)
from elastic_fwi.postprocessing.vector_array_waveforms import (
    SamplingIntervalError,
    WaveformAlignmentError,
    WaveformAnalysisConfig,
    WaveformAnalysisInputs,
    WaveformOutputPaths,
    analytic_envelope,
    apply_analysis_residual_convention,
    compute_comparison_metrics,
    normalized_cross_correlation_lag,
    run_vector_array_waveform_analysis,
    signal_characteristics,
    threshold_crossing_index,
    validate_protocol_time_interval,
    write_waveform_analysis_outputs,
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


def _signals(
    step: int,
    receiver_id: int,
    component: str,
    family: str,
    *,
    dt: float = 0.05,
) -> tuple[float, float]:
    t = dt * step
    component_factor = 1.0 if component == "x" else 0.5
    family_shift = 0.0 if family == "reflection" else 0.1
    healthy = component_factor * np.sin(2.0 * np.pi * (t - family_shift))
    defect = 1.1 * component_factor * np.sin(
        2.0 * np.pi * (t - family_shift - 0.05)
    )
    scale = 1.0 + 0.001 * receiver_id
    return float(scale * healthy), float(scale * defect)


def _write_csv(
    path: Path,
    *,
    geometry: str,
    num_steps: int = 4,
    legacy_time: bool = True,
    step_start: int = 0,
    dt: float = 0.05,
) -> Path:
    rows: list[dict] = []
    counts = {"reflection": 21, "transmission": 21 if geometry == "line" else 61}
    for family, count in counts.items():
        array = "gamma1" if family == "reflection" else "gamma2"
        role = "source_reflection" if family == "reflection" else "transmission"
        for step in range(step_start, step_start + num_steps):
            for receiver_id in range(count):
                for component in ("x", "y"):
                    healthy, defect = _signals(
                        step,
                        receiver_id,
                        component,
                        family,
                        dt=dt,
                    )
                    residual = healthy - defect
                    rows.append(
                        {
                            "time": float(step) if legacy_time else dt * step,
                            "step": step,
                            "array": array,
                            "role": role,
                            "receiver_id": receiver_id,
                            "component": component,
                            "residual": residual,
                            "reference": healthy,
                            "observed": defect,
                        }
                    )
    frame = pd.DataFrame(rows).sample(frac=1.0, random_state=13).reset_index(drop=True)
    frame.to_csv(path, index=False)
    return path


def _config(tmp_path: Path, *, receiver_ids: tuple[int, ...] | None = (0,)) -> WaveformAnalysisConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    line_manifest = _write_manifest(tmp_path / "line_manifest.json", geometry="line")
    box_manifest = _write_manifest(tmp_path / "box_manifest.json", geometry="box")
    line_csv = _write_csv(tmp_path / "line.csv", geometry="line")
    box_csv = _write_csv(tmp_path / "box.csv", geometry="box")
    return WaveformAnalysisConfig(
        inputs=WaveformAnalysisInputs(
            line_residuals=line_csv,
            box_residuals=box_csv,
            line_manifest=line_manifest,
            box_manifest=box_manifest,
        ),
        output_dir=tmp_path / "out",
        dt=0.05,
        final_time=0.2,
        time_column_policy="legacy-step-index",
        receiver_ids=receiver_ids,
    )


def _line_only_config(
    tmp_path: Path,
    *,
    receiver_ids: tuple[int, ...] | None = (0,),
    num_steps: int = 4,
    step_start: int = 1,
    dt: float = 0.05,
) -> WaveformAnalysisConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    line_manifest = _write_manifest(tmp_path / "line_manifest.json", geometry="line")
    line_csv = _write_csv(
        tmp_path / "line.csv",
        geometry="line",
        num_steps=num_steps,
        legacy_time=False,
        step_start=step_start,
        dt=dt,
    )
    return WaveformAnalysisConfig(
        inputs=WaveformAnalysisInputs(
            line_residuals=line_csv,
            line_manifest=line_manifest,
        ),
        output_dir=tmp_path / "out",
        dt=dt,
        final_time=float((step_start + num_steps - 1) * dt),
        time_column_policy="strict-physical",
        receiver_ids=receiver_ids,
        geometry="line",
    )


def test_analytic_envelope_of_periodic_cosine() -> None:
    n = 64
    indices = np.arange(n)
    signal = np.cos(2.0 * np.pi * 5.0 * indices / n)
    envelope = analytic_envelope(signal)
    assert np.allclose(envelope, 1.0, atol=1.0e-12, rtol=1.0e-12)


def test_threshold_crossing() -> None:
    envelope = np.asarray([0.0, 0.05, 0.2, 1.0])
    index, threshold, status = threshold_crossing_index(envelope, fraction=0.1)
    assert index == 2
    assert threshold == pytest.approx(0.1)
    assert status == "ok"


def test_threshold_crossing_zero_signal() -> None:
    index, threshold, status = threshold_crossing_index(np.zeros(4), fraction=0.1)
    assert index is None
    assert threshold == 0.0
    assert status == "zero_signal"


def test_signal_characteristics_maxima_and_energy() -> None:
    values = np.asarray([0.0, -2.0, 1.0, 0.5])
    steps = np.arange(4)
    times = 0.05 * steps
    metrics = signal_characteristics(
        values,
        steps=steps,
        times=times,
        dt=0.05,
        threshold_fraction=0.1,
    )
    assert metrics["signed_max"] == 1.0
    assert metrics["signed_min"] == -2.0
    assert metrics["absolute_max"] == 2.0
    assert metrics["absolute_peak_step"] == 1
    assert metrics["absolute_peak_time"] == pytest.approx(0.05)
    assert metrics["discrete_energy"] == pytest.approx(0.05 * 5.25)


def test_cross_correlation_positive_lag_means_defect_delayed() -> None:
    healthy = np.zeros(12)
    defect = np.zeros(12)
    healthy[2] = 1.0
    defect[5] = 1.0
    result = normalized_cross_correlation_lag(
        healthy, defect, dt=0.05, epsilon=1.0e-30
    )
    assert result["cross_correlation_lag_steps"] == 3
    assert result["cross_correlation_lag_time"] == pytest.approx(0.15)
    assert result["cross_correlation_status"] == "ok"


def test_residual_convention_can_be_flipped_explicitly(tmp_path: Path) -> None:
    result = run_vector_array_waveform_analysis(_config(tmp_path))
    source = result.canonical
    flipped, name = apply_analysis_residual_convention(
        source,
        convention="defect-minus-healthy",
        atol=1.0e-14,
        rtol=1.0e-12,
    )
    assert name == "observed_minus_reference"
    assert np.allclose(flipped["analysis_residual"], -source["residual"])


def test_time_reconstruction_and_right_open_protocol(tmp_path: Path) -> None:
    result = run_vector_array_waveform_analysis(_config(tmp_path))
    assert set(result.time_validation["sampling_status"]) == {
        "right_open_terminal_excluded"
    }
    assert np.allclose(result.time_validation["time_physical_max"], 0.15)
    assert np.allclose(
        result.canonical["time_physical"],
        0.05 * result.canonical["step_index"],
    )


def test_incompatible_final_time_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    bad = WaveformAnalysisConfig(
        **{**config.__dict__, "final_time": 0.5}
    )
    with pytest.raises(SamplingIntervalError, match="last physical sample"):
        run_vector_array_waveform_analysis(bad)




def test_legacy_time_alias_requires_explicit_policy(tmp_path: Path) -> None:
    config = _config(tmp_path)
    strict = WaveformAnalysisConfig(
        **{**config.__dict__, "time_column_policy": "strict-physical"}
    )
    with pytest.raises(TimeConventionError, match="legacy-step-index"):
        run_vector_array_waveform_analysis(strict)


def test_missing_time_sample_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    frame = pd.read_csv(config.inputs.line_residuals)
    mask = (
        (frame["array"] == "gamma1")
        & (frame["receiver_id"] == 0)
        & (frame["component"] == "x")
        & (frame["step"] == 3)
    )
    frame.loc[~mask].to_csv(config.inputs.line_residuals, index=False)
    with pytest.raises(ResidualCSVError, match="incomplete receiver/component"):
        run_vector_array_waveform_analysis(config)

def test_signal_metrics_keep_components_and_cases_separate(tmp_path: Path) -> None:
    result = run_vector_array_waveform_analysis(_config(tmp_path))
    assert set(result.signal_metrics["component"]) == {"x", "y"}
    assert set(result.signal_metrics["case"]) == {"healthy", "defect", "residual"}
    key = ["geometry", "residual_family", "receiver_id", "component", "case"]
    assert not result.signal_metrics.duplicated(key).any()


def test_comparison_metrics_include_vector_diagnostic(tmp_path: Path) -> None:
    result = run_vector_array_waveform_analysis(_config(tmp_path))
    assert set(result.comparison_metrics["component"]) == {"x", "y", "vector"}
    assert np.isfinite(result.comparison_metrics["residual_energy"]).all()


def test_incompatible_component_steps_are_rejected(tmp_path: Path) -> None:
    result = run_vector_array_waveform_analysis(_config(tmp_path))
    canonical = result.canonical.copy()
    mask = (
        (canonical["geometry"] == "line")
        & (canonical["residual_family"] == "reflection")
        & (canonical["receiver_id"] == 0)
        & (canonical["component"] == "y")
        & (canonical["step_index"] == 3)
    )
    canonical = canonical.loc[~mask]
    with pytest.raises(WaveformAlignmentError, match="incompatible step indices"):
        compute_comparison_metrics(
            canonical,
            dt=0.05,
            threshold_fraction=0.1,
            ratio_epsilon=1.0e-30,
        )


def test_healthy_and_defect_are_read_from_reference_observed(tmp_path: Path) -> None:
    result = run_vector_array_waveform_analysis(_config(tmp_path))
    assert np.allclose(result.canonical["healthy"], result.canonical["reference"])
    assert np.allclose(result.canonical["defect"], result.canonical["observed"])
    assert np.allclose(
        result.canonical["analysis_residual"],
        result.canonical["reference"] - result.canonical["observed"],
    )


def test_receiver_counts_are_validated_by_reused_contract(tmp_path: Path) -> None:
    result = run_vector_array_waveform_analysis(_config(tmp_path))
    counts = result.summary["receiver_counts"]
    assert counts["line"] == {"reflection": 21, "transmission": 21}
    assert counts["box"] == {"reflection": 21, "transmission": 61}


def test_outputs_generate_csv_json_png_and_pdf(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_vector_array_waveform_analysis(config)
    paths = WaveformOutputPaths.from_root(config.output_dir)
    paths.create(overwrite=True)
    outputs = write_waveform_analysis_outputs(
        result,
        config,
        command=["python", "analyze_vector_array_waveforms.py"],
        repository_root=tmp_path,
    )
    assert Path(outputs["signal_metrics_csv"]).exists()
    assert Path(outputs["comparison_metrics_csv"]).exists()
    assert Path(outputs["summary_json"]).exists()
    assert Path(outputs["manifest_json"]).exists()
    assert Path(outputs["figure_peak_amplitude_ratio_png"]).exists()
    assert Path(outputs["figure_peak_amplitude_ratio_pdf"]).exists()
    receiver_pngs = list((paths.figures_dir).glob("**/receiver_*_waveforms.png"))
    receiver_pdfs = list((paths.figures_dir).glob("**/receiver_*_waveforms.pdf"))
    assert receiver_pngs
    assert len(receiver_pngs) == len(receiver_pdfs)


def test_csv_metrics_are_reproducible(tmp_path: Path) -> None:
    config_a = _config(tmp_path / "a")
    result_a = run_vector_array_waveform_analysis(config_a)
    output_a = WaveformOutputPaths.from_root(config_a.output_dir)
    output_a.create(overwrite=True)
    files_a = write_waveform_analysis_outputs(
        result_a,
        config_a,
        command=["python", "analysis"],
        repository_root=tmp_path,
    )

    config_b = _config(tmp_path / "b")
    result_b = run_vector_array_waveform_analysis(config_b)
    output_b = WaveformOutputPaths.from_root(config_b.output_dir)
    output_b.create(overwrite=True)
    files_b = write_waveform_analysis_outputs(
        result_b,
        config_b,
        command=["python", "analysis"],
        repository_root=tmp_path,
    )

    frame_a = pd.read_csv(files_a["signal_metrics_csv"])
    frame_b = pd.read_csv(files_b["signal_metrics_csv"])
    # Source paths differ between temporary roots; scientific columns must match.
    scientific = [column for column in frame_a.columns if column != "source_csv"]
    pd.testing.assert_frame_equal(frame_a[scientific], frame_b[scientific])


def test_line_only_run_uses_only_line_campaign(tmp_path: Path) -> None:
    result = run_vector_array_waveform_analysis(_line_only_config(tmp_path))
    assert len(result.inspections) == 1
    assert result.inspections[0].geometry == "line"
    assert set(result.canonical["geometry"]) == {"line"}
    assert set(result.receiver_geometry["geometry"]) == {"line"}
    assert result.summary["analysis_mode"] == "line-only"
    assert result.summary["geometries"] == ["line"]
    assert result.summary["num_signal_metric_rows"] == 252
    assert result.summary["num_comparison_metric_rows"] == 126
    assert result.summary["receiver_counts"] == {
        "line": {"reflection": 21, "transmission": 21}
    }


def test_line_only_t20_one_based_sampling_contract(tmp_path: Path) -> None:
    config = _line_only_config(
        tmp_path,
        receiver_ids=(0,),
        num_steps=800,
        step_start=1,
        dt=0.025,
    )
    base_config = ApertureComparisonConfig(
        inputs=ApertureComparisonInputs(
            line_residuals=config.inputs.line_residuals,
            line_manifest=config.inputs.line_manifest,
        ),
        output_dir=config.output_dir,
        dt=config.dt,
        time_column_policy=config.time_column_policy,
        geometry="line",
    )
    specs = load_receiver_specs(config.inputs.line_manifest, geometry="line")
    canonical, inspection = load_and_validate_campaign(
        config.inputs.line_residuals,
        geometry="line",
        specs=specs,
        config=base_config,
    )
    validation = validate_protocol_time_interval(
        (inspection,),
        dt=config.dt,
        final_time=config.final_time,
        atol=config.time_atol,
        rtol=config.time_rtol,
    ).iloc[0]
    assert validation["step_index_min"] == 1
    assert validation["step_index_max"] == 800
    assert validation["time_physical_min"] == pytest.approx(0.025)
    assert validation["time_physical_max"] == pytest.approx(20.0)
    assert validation["initial_sampling_status"] == (
        "initial_state_excluded_first_sample_at_dt"
    )
    assert validation["sampling_status"] == "terminal_included"
    assert np.allclose(
        canonical["time_physical"],
        0.025 * canonical["step_index"],
    )
    assert canonical.shape[0] == 67_200


def test_line_box_mode_requires_box_inputs(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    line_manifest = _write_manifest(tmp_path / "line_manifest.json", geometry="line")
    line_csv = _write_csv(
        tmp_path / "line.csv",
        geometry="line",
        legacy_time=False,
        step_start=1,
    )
    with pytest.raises(ValueError, match="box_residuals and box_manifest"):
        WaveformAnalysisConfig(
            inputs=WaveformAnalysisInputs(
                line_residuals=line_csv,
                line_manifest=line_manifest,
            ),
            output_dir=tmp_path / "out",
        )


def test_line_only_outputs_exclude_box_inputs_and_record_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _touch_plot(path_png: Path, path_pdf: Path, *args, **kwargs):
        path_png.parent.mkdir(parents=True, exist_ok=True)
        path_png.touch()
        path_pdf.touch()
        return path_png, path_pdf

    monkeypatch.setattr(waveform_module, "_plot_selected_receiver", _touch_plot)
    monkeypatch.setattr(waveform_module, "_plot_network_metric", _touch_plot)
    config = _line_only_config(tmp_path, receiver_ids=(0,))
    result = run_vector_array_waveform_analysis(config)
    paths = WaveformOutputPaths.from_root(config.output_dir)
    paths.create(overwrite=True)
    outputs = write_waveform_analysis_outputs(
        result,
        config,
        command=[
            "python",
            "analyze_vector_array_waveforms.py",
            "--geometry",
            "line",
        ],
        repository_root=tmp_path,
    )
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="utf-8"))
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert manifest["analysis_mode"] == "line-only"
    assert manifest["configuration"]["geometry"] == "line"
    assert set(manifest["inputs"]) == {"line_residuals", "line_manifest"}
    assert summary["analysis_mode"] == "line-only"
    assert not list(config.output_dir.rglob("*box*"))
    assert Path(outputs["figure_peak_amplitude_ratio_png"]).exists()
    assert Path(outputs["figure_peak_amplitude_ratio_pdf"]).exists()


def test_cli_accepts_line_only_mode(tmp_path: Path) -> None:
    config = _line_only_config(tmp_path / "inputs", receiver_ids=(0,))
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "revision_wave_motion"
        / "analyze_vector_array_waveforms.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_vector_array_waveforms", script)
    assert spec is not None and spec.loader is not None
    cli_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli_module)
    output_dir = tmp_path / "parser_out"
    args = cli_module.build_parser().parse_args(
        [
            "--geometry",
            "line",
            "--line-residuals",
            str(config.inputs.line_residuals),
            "--line-manifest",
            str(config.inputs.line_manifest),
            "--dt",
            "0.05",
            "--final-time",
            "0.2",
            "--time-column-policy",
            "strict-physical",
            "--residual-convention",
            "source",
            "--arrival-threshold",
            "0.1",
            "--receiver-ids",
            "0",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert args.geometry == "line"
    assert args.line_residuals == config.inputs.line_residuals
    assert args.line_manifest == config.inputs.line_manifest
    assert args.box_residuals is None
    assert args.box_manifest is None
    assert args.time_column_policy == "strict-physical"
