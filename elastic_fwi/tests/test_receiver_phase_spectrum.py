from __future__ import annotations

import json
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest

from elastic_fwi.postprocessing.receiver_phase_spectrum import (
    ReceiverTracePair,
    SpectrumConfig,
    analyze_trace_pair,
    compare_line_box_cases,
    classify_frequency_against_source_band,
    compute_amplitude_metrics,
    compute_cross_correlation,
    compute_effective_source_spectrum,
    compute_fft_spectrum,
    compute_phase_difference,
    compute_polarization,
    load_campaign_trace_pairs,
    load_observation_npz_pair,
    load_residual_csv,
    load_residual_npz,
    validate_time_axis,
    write_analysis_outputs,
)


def _sinusoid(*, frequency: float, phase: float = 0.0, duration: float = 2.0, dt: float = 0.01):
    times = dt * np.arange(int(round(duration / dt)), dtype=float)
    return times, np.sin(2.0 * np.pi * frequency * times + phase)


def _trace_pair(*, geometry: str = "line", family: str = "transmission") -> ReceiverTracePair:
    times, x = _sinusoid(frequency=5.0, duration=2.0, dt=0.01)
    y = 0.5 * np.sin(2.0 * np.pi * 5.0 * times + 0.2)
    reference = np.empty((times.size, 3, 2), dtype=float)
    observed = np.empty_like(reference)
    for receiver in range(3):
        reference[:, receiver, 0] = (1.0 + receiver) * x
        reference[:, receiver, 1] = (1.0 + receiver) * y
        observed[:, receiver, 0] = 0.8 * (1.0 + receiver) * np.sin(
            2.0 * np.pi * 5.0 * times + 0.3
        )
        observed[:, receiver, 1] = 0.8 * (1.0 + receiver) * np.sin(
            2.0 * np.pi * 5.0 * times + 0.5
        )
    return ReceiverTracePair(
        case_name=f"synthetic_{geometry}_{family}",
        geometry_kind=geometry,
        residual_family=family,
        times=times,
        reference_values=reference,
        observed_values=observed,
        receiver_ids=np.arange(3),
        components=("x", "y"),
        centers=np.column_stack([np.arange(3, dtype=float), np.ones(3)]),
        source_paths=("synthetic",),
        metadata={"source_format": "synthetic"},
    )


def _write_residual_npz(path: Path, pair: ReceiverTracePair) -> None:
    np.savez(
        path,
        kind=np.asarray("vector_array_residual_series", dtype=np.str_),
        label=np.asarray(pair.residual_family, dtype=np.str_),
        name=np.asarray("synthetic", dtype=np.str_),
        array_name=np.asarray("gamma2", dtype=np.str_),
        role=np.asarray("transmission", dtype=np.str_),
        times=pair.times,
        values=pair.reference_values - pair.observed_values,
        reference_values=pair.reference_values,
        observed_values=pair.observed_values,
        components=np.asarray(pair.components, dtype=np.str_),
    )


def test_fft_recovers_sinus_frequency_and_amplitude() -> None:
    times, signal = _sinusoid(frequency=5.0)
    result = compute_fft_spectrum(signal, dt=times[1] - times[0], window="hann")
    assert result["dominant_frequency"] == pytest.approx(5.0)
    assert result["dominant_amplitude"] == pytest.approx(1.0, rel=2.0e-3)
    assert result["spectral_energy"] > 0.0


def test_known_spectral_phase_difference() -> None:
    times, reference = _sinusoid(frequency=5.0, phase=0.0)
    _, observed = _sinusoid(frequency=5.0, phase=0.4)
    ref = compute_fft_spectrum(reference, dt=times[1] - times[0])
    obs = compute_fft_spectrum(observed, dt=times[1] - times[0])
    result = compute_phase_difference(
        ref,
        obs,
        phase_amplitude_threshold=0.05,
    )
    assert result["phase_status"] == "ok"
    assert result["phase_difference_at_comparison"] == pytest.approx(0.4, abs=2.0e-3)


def test_phase_is_masked_when_observed_spectrum_is_zero() -> None:
    times, reference = _sinusoid(frequency=5.0)
    observed = np.zeros_like(reference)
    ref = compute_fft_spectrum(reference, dt=times[1] - times[0])
    obs = compute_fft_spectrum(observed, dt=times[1] - times[0])
    result = compute_phase_difference(
        ref,
        obs,
        phase_amplitude_threshold=0.05,
    )
    assert result["phase_status"] == "low_spectral_amplitude"
    assert result["phase_difference_at_comparison"] is None


def test_cross_correlation_recovers_positive_delay() -> None:
    _, reference = _sinusoid(frequency=3.0, duration=3.0)
    observed = np.zeros_like(reference)
    observed[4:] = reference[:-4]
    result = compute_cross_correlation(reference, observed, dt=0.01)
    assert result["cross_correlation_lag_steps"] == 4
    assert result["cross_correlation_lag_time"] == pytest.approx(0.04)
    assert result["cross_correlation_max"] > 0.98


def test_amplitude_rms_and_energy() -> None:
    times, signal = _sinusoid(frequency=5.0)
    metrics = compute_amplitude_metrics(signal, dt=times[1] - times[0])
    assert metrics["temporal_max_abs"] == pytest.approx(1.0)
    assert metrics["temporal_rms"] == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-12)
    assert metrics["temporal_energy"] == pytest.approx(1.0, rel=1e-12)


def test_linear_polarization_angle_and_ratio() -> None:
    _, x = _sinusoid(frequency=5.0)
    values = np.column_stack([x, 2.0 * x])
    result = compute_polarization(values, components=("x", "y"))
    assert result["polarization_available"] is True
    assert result["polarization_angle_degrees"] == pytest.approx(
        np.degrees(np.arctan2(2.0, 1.0)), abs=1e-10
    )
    assert result["polarization_rectilinearity"] == pytest.approx(1.0)
    assert result["polarization_rms_y_over_x"] == pytest.approx(2.0)


def test_single_component_reports_missing_polarization() -> None:
    _, x = _sinusoid(frequency=5.0)
    result = compute_polarization(x[:, None], components=("x",))
    assert result["polarization_available"] is False
    assert result["polarization_status"] == "missing_vector_component"


def test_residual_npz_loader_uses_repository_schema(tmp_path: Path) -> None:
    pair = _trace_pair()
    path = tmp_path / "vector_array_transmission_residuals.npz"
    _write_residual_npz(path, pair)
    loaded = load_residual_npz(path, case_name="loaded", geometry_kind="line")
    assert loaded.case_name == "loaded"
    assert loaded.residual_family == "transmission"
    assert loaded.reference_values.shape == pair.reference_values.shape
    np.testing.assert_allclose(loaded.observed_values, pair.observed_values)


def test_paired_observation_npz_loader(tmp_path: Path) -> None:
    pair = _trace_pair()
    reference_path = tmp_path / "reference.npz"
    observed_path = tmp_path / "observed.npz"
    common = {
        "kind": np.asarray("vector_array_observation_time_series", dtype=np.str_),
        "times": pair.times,
        "centers": pair.centers,
        "components": np.asarray(pair.components, dtype=np.str_),
        "array_name": np.asarray("gamma2", dtype=np.str_),
        "role": np.asarray("transmission", dtype=np.str_),
    }
    np.savez(reference_path, **common, values=pair.reference_values, label=np.asarray("reference"))
    np.savez(observed_path, **common, values=pair.observed_values, label=np.asarray("observed"))
    loaded = load_observation_npz_pair(
        reference_path,
        observed_path,
        case_name="paired",
        geometry_kind="line",
        residual_family="transmission",
    )
    assert loaded.num_receivers == 3
    np.testing.assert_allclose(loaded.times, pair.times)



def test_residual_long_csv_loader_selects_one_family(tmp_path: Path) -> None:
    pair = _trace_pair()
    path = tmp_path / "vector_array_residuals.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("time,step,array,role,receiver_id,component,residual,reference,observed\n")
        for k, time in enumerate(pair.times):
            for receiver in range(pair.num_receivers):
                for component_index, component in enumerate(pair.components):
                    reference = pair.reference_values[k, receiver, component_index]
                    observed = pair.observed_values[k, receiver, component_index]
                    handle.write(
                        f"{time},{k},gamma2,transmission,{receiver},{component},"
                        f"{reference-observed},{reference},{observed}\n"
                    )
    loaded = load_residual_csv(
        path,
        case_name="csv_case",
        geometry_kind="line",
        residual_family="transmission",
    )
    assert loaded.case_name == "csv_case"
    assert loaded.components == ("x", "y")
    np.testing.assert_allclose(loaded.reference_values, pair.reference_values)


def test_campaign_loader_validates_t20_temporal_contract(tmp_path: Path) -> None:
    root = tmp_path / "final_line_T20"
    vector_dir = root / "inverse" / "vector_array"
    vector_dir.mkdir(parents=True)
    times = 0.025 * np.arange(1, 801, dtype=float)
    values = np.zeros((800, 2, 2), dtype=float)
    values[:, :, 0] = np.sin(2.0 * np.pi * times)[:, None]
    values[:, :, 1] = 0.5 * values[:, :, 0]
    observed = 0.9 * values
    for family in ("reflection", "transmission"):
        np.savez(
            vector_dir / f"vector_array_{family}_residuals.npz",
            kind=np.asarray("vector_array_residual_series", dtype=np.str_),
            label=np.asarray(family, dtype=np.str_),
            array_name=np.asarray("gamma1" if family == "reflection" else "gamma2"),
            role=np.asarray(family),
            times=times,
            values=values-observed,
            reference_values=values,
            observed_values=observed,
            components=np.asarray(["x", "y"]),
        )
    (root / "gradient_combination_campaign_manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "dt": 0.025,
                    "num_steps": 800,
                    "source_frequency": 1.0,
                    "receiver_array_geometry": "line",
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "inverse" / "inverse_time_series_manifest.json").write_text(
        json.dumps(
            {
                "temporal_sampling": {
                    "dt": 0.025,
                    "num_steps": 800,
                    "T_final": 20.0,
                }
            }
        ),
        encoding="utf-8",
    )
    pairs, context = load_campaign_trace_pairs(root, require_final_time=20.0)
    assert len(pairs) == 2
    assert context["geometry_kind"] == "line"
    assert all(pair.num_steps == 800 for pair in pairs)
    assert all(pair.final_time == pytest.approx(20.0) for pair in pairs)
    assert all(pair.metadata["campaign_source_num_periods"] == pytest.approx(1.0) for pair in pairs)
    assert all(pair.metadata["campaign_source_t0"] == pytest.approx(0.0) for pair in pairs)

def test_case_analysis_and_line_box_comparison() -> None:
    config = SpectrumConfig(expected_source_frequency=5.0, source_frequency_tolerance=0.1)
    line = analyze_trace_pair(_trace_pair(geometry="line"), config)
    box = analyze_trace_pair(_trace_pair(geometry="box"), config)
    assert line.summary["status"] == "ok"
    assert line.summary["network_reference_dominant_frequency"] == pytest.approx(5.0)
    assert len(line.metrics_rows) == 6
    comparison = compare_line_box_cases([line, box])
    assert len(comparison) == 1
    assert comparison[0]["status"] == "ok"
    assert comparison[0]["time_axis_match"] is True


def test_write_outputs_produces_csv_json_npz_and_figures(tmp_path: Path) -> None:
    analysis = analyze_trace_pair(
        _trace_pair(),
        SpectrumConfig(expected_source_frequency=5.0, max_plot_frequency=10.0),
    )
    outputs = write_analysis_outputs(
        [analysis],
        output_dir=tmp_path / "out",
        analysis_mode="line-only",
        save_pdf=True,
        save_npz=True,
        command=["python", "test"],
    )
    assert Path(outputs["summary_csv"]).is_file()
    assert Path(outputs["manifest_json"]).is_file()
    case = outputs["cases"][analysis.pair.case_name]
    assert Path(case["spectra_npz"]).is_file()
    with np.load(case["spectra_npz"], allow_pickle=False) as spectra:
        assert "source_signal" in spectra.files
        assert "source_amplitude" in spectra.files
        assert "source_psd" in spectra.files
        assert spectra["source_signal"].size == analysis.pair.num_steps
    assert all(Path(path).is_file() for path in case["profile_figures"])
    assert len(case["component_spectrum_figures"]["x"]) == 2
    manifest = json.loads(Path(outputs["manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["kind"] == "receiver_phase_spectrum_manifest"
    assert manifest["status"] == "ok"
    assert manifest["analysis_mode"] == "line-only"
    assert manifest["geometry_kinds"] == ["line"]
    assert manifest["line_box_comparison_status"] == "not_applicable"
    assert "line_box_comparison_csv" not in outputs


def test_cli_runs_on_synthetic_repository_npz(tmp_path: Path) -> None:
    pair = _trace_pair()
    source = tmp_path / "vector_array_transmission_residuals.npz"
    _write_residual_npz(source, pair)
    output = tmp_path / "cli_output"
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "revision_wave_motion"
        / "analyze_receiver_phase_spectrum.py"
    )
    spec = importlib.util.spec_from_file_location("receiver_phase_spectrum_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return_code = module.main(
        [
            "--geometry",
            "line",
            "--input",
            str(source),
            "--campaign-name",
            "synthetic_cli",
            "--geometry-kind",
            "line",
            "--standalone-family",
            "transmission",
            "--expected-source-frequency",
            "5.0",
            "--source-frequency-tolerance",
            "0.1",
            "--max-plot-frequency",
            "10.0",
            "--output-dir",
            str(output),
            "--quiet",
            "--no-pdf",
            "--no-npz",
        ]
    )
    assert return_code == 0
    assert (output / "receiver_phase_spectrum_manifest.json").is_file()
    assert (output / "receiver_phase_spectrum_summary.csv").is_file()
    assert not (
        output / "receiver_phase_spectrum_line_box_comparison.csv"
    ).exists()
    manifest = json.loads(
        (output / "receiver_phase_spectrum_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["analysis_mode"] == "line-only"


def test_time_axis_validation_detects_expected_conventions() -> None:
    times = 0.025 * np.arange(1, 801, dtype=float)
    report = validate_time_axis(
        times,
        expected_dt=0.025,
        expected_final_time=20.0,
    )
    assert report["status"] == "ok"
    assert report["num_steps"] == 800
    assert report["first_time"] == pytest.approx(0.025)
    assert report["last_time"] == pytest.approx(20.0)

def test_effective_hann_source_band_uses_sampled_finite_pulse() -> None:
    times = 0.025 * np.arange(1, 801, dtype=float)
    result = compute_effective_source_spectrum(
        times,
        frequency=1.0,
        num_periods=1.0,
        t0=0.0,
        amplitude_threshold=0.05,
    )
    assert result["dominant_frequency"] == pytest.approx(1.10)
    assert result["significant_band_min"] == pytest.approx(0.05)
    assert result["significant_band_max"] == pytest.approx(2.75)
    status = classify_frequency_against_source_band(
        0.05,
        source_band_min=result["significant_band_min"],
        source_band_max=result["significant_band_max"],
        frequency_resolution=result["frequency_resolution"],
    )
    assert status == "source_spectrum_compatible"


def test_source_spectrum_compatibility_is_distinct_from_nominal_carrier() -> None:
    times = 0.025 * np.arange(1, 801, dtype=float)
    reference = np.zeros((times.size, 2, 2), dtype=float)
    observed = np.zeros_like(reference)
    for receiver in range(2):
        reference[:, receiver, 0] = np.sin(2.0 * np.pi * 0.20 * times)
        reference[:, receiver, 1] = 0.5 * np.sin(2.0 * np.pi * 0.20 * times)
        observed[:, receiver, :] = reference[:, receiver, :]
    pair = ReceiverTracePair(
        case_name="source_band_case",
        geometry_kind="line",
        residual_family="transmission",
        times=times,
        reference_values=reference,
        observed_values=observed,
        receiver_ids=np.arange(2),
        components=("x", "y"),
        centers=np.zeros((2, 2), dtype=float),
        source_paths=("synthetic",),
        metadata={"source_format": "synthetic"},
    )
    analysis = analyze_trace_pair(
        pair,
        SpectrumConfig(
            expected_source_frequency=1.0,
            source_frequency_tolerance=0.10,
            source_band_amplitude_threshold=0.05,
        ),
    )
    assert analysis.summary["nominal_source_frequency_compatible"] is False
    assert analysis.summary["source_frequency_compatible"] is True
    assert analysis.summary["source_spectrum_status"] == "source_spectrum_compatible"
    assert analysis.summary["status"] == "ok"
    assert analysis.summary["validation_notes"]


def test_component_amplitude_ratio_is_suppressed_for_quasi_zero_reference() -> None:
    times = 0.01 * np.arange(400, dtype=float)
    carrier = np.sin(2.0 * np.pi * 2.0 * times)
    reference = np.zeros((times.size, 1, 2), dtype=float)
    observed = np.zeros_like(reference)
    reference[:, 0, 0] = 1.0e-12 * carrier
    reference[:, 0, 1] = carrier
    observed[:, 0, 0] = 1.0e-4 * carrier
    observed[:, 0, 1] = 0.9 * carrier
    pair = ReceiverTracePair(
        case_name="quasi_zero_reference",
        geometry_kind="line",
        residual_family="reflection",
        times=times,
        reference_values=reference,
        observed_values=observed,
        receiver_ids=np.asarray([0]),
        components=("x", "y"),
        centers=np.zeros((1, 2), dtype=float),
        source_paths=("synthetic",),
        metadata={"source_format": "synthetic"},
    )
    analysis = analyze_trace_pair(
        pair,
        SpectrumConfig(
            expected_source_frequency=2.0,
            amplitude_ratio_reference_floor_relative=1.0e-6,
        ),
    )
    rows = {row["component"]: row for row in analysis.metrics_rows}
    assert rows["x"]["amplitude_rms_ratio"] is None
    assert rows["x"]["amplitude_rms_relative_change"] is None
    assert rows["x"]["amplitude_rms_ratio_status"] == "reference_amplitude_too_small"
    assert rows["x"]["temporal_energy_ratio"] is None
    assert rows["x"]["temporal_energy_ratio_status"] == "reference_amplitude_too_small"
    assert rows["y"]["amplitude_rms_ratio_status"] == "ok"
    assert rows["y"]["amplitude_rms_ratio"] == pytest.approx(0.9)
    assert analysis.summary["num_amplitude_ratios_unreliable"] == 1
    assert analysis.summary["amplitude_ratio_reliable_fraction"] == pytest.approx(0.5)

