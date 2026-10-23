from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from elastic_fwi.postprocessing.c66_multipolar_pattern import (
    C66Field,
    CaseInput,
    CaseMetadata,
    DefectGeometry,
    MultipolarConfig,
    _summarize_robustness_records,
    analyze_c66_case,
    compute_180_symmetry,
    compute_angular_fourier,
    compute_global_180_symmetry,
    load_c66_field,
)


def _synthetic_case(
    *,
    mode: int,
    orientation_degrees: float,
    pattern_center: tuple[float, float] = (0.0, 0.0),
    noise_level: float = 0.0,
    isotropic: bool = False,
    seed: int = 1234,
    points_per_axis: int = 181,
) -> CaseInput:
    axis = np.linspace(-3.0, 3.0, points_per_axis)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    dx = xx - pattern_center[0]
    dy = yy - pattern_center[1]
    radius = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    envelope = np.exp(-((radius - 1.5) / 0.28) ** 2)
    if isotropic:
        values = envelope
    else:
        orientation = math.radians(orientation_degrees)
        values = envelope * np.cos(mode * (theta - orientation))
    if noise_level > 0.0:
        rng = np.random.default_rng(seed)
        values = values + noise_level * rng.standard_normal(values.shape) * envelope
    flat = values.reshape(-1)
    flat /= np.max(np.abs(flat))
    spacing = float(axis[1] - axis[0])
    field = C66Field(
        centers=np.column_stack((xx.reshape(-1), yy.reshape(-1))),
        values=flat,
        cell_measures=np.full(flat.size, spacing * spacing, dtype=np.float64),
        source_path=Path("synthetic.npz"),
        source_format="synthetic_regular_grid",
        field_name="gradient_C66",
        normalization_factor=1.0,
        values_were_normalized=True,
    )
    return CaseInput(
        field=field,
        defect=DefectGeometry(center_x=0.0, center_y=0.0, radius=1.0),
        metadata=CaseMetadata(
            case_name="synthetic",
            observation_mode="classical_full",
            modality="classical",
            aperture="line",
        ),
    )


def _config() -> MultipolarConfig:
    return MultipolarConfig(
        num_radial_bins=60,
        num_angular_bins=180,
        radial_bin_values=(50, 60),
        annulus_width_factor=0.15,
        annulus_width_values=(0.10, 0.15),
        annulus_radius_factors=(1.0, 1.5, 2.0),
        max_radius_factor=3.0,
        primary_radius_min_factor=0.8,
        primary_radius_max_factor=2.4,
        center_search_steps=5,
        center_search_extent_factor=0.5,
        orientation_stability_degrees=5.0,
        symmetry_stability_tolerance=0.12,
        peak_radius_stability_factor=0.20,
    )


def _periodic_difference(left: float, right: float, period: float) -> float:
    return abs((left - right + 0.5 * period) % period - 0.5 * period)


def _robustness_record(
    *,
    num_radial_bins: int,
    width: float,
    mode: int,
    orientation: float,
    symmetry: float,
    coverage: float,
    status: str = "OK",
    reason: str = "OK",
) -> dict[str, object]:
    return {
        "num_radial_bins": num_radial_bins,
        "annulus_width_over_R": width,
        "peak_radius_over_R": 1.5 + 0.001 * (num_radial_bins - 80),
        "dominant_mode": mode if status == "OK" else 0,
        "dominant_mode_raw": mode,
        "dominant_mode_power_fraction": 0.35,
        "orientation_degrees": orientation if status == "OK" else math.nan,
        "orientation_period_degrees": 360.0 / mode,
        "signed_symmetry_180": symmetry,
        "angular_coverage": coverage,
        "angular_coverage_sufficient": status == "OK",
        "mode_power_sufficient": True,
        "interpretation_reason": reason,
        "status": status,
    }


def test_coverage_limited_annuli_do_not_reduce_valid_mode_consensus() -> None:
    records: list[dict[str, object]] = []
    for bins in (60, 80, 100):
        records.append(
            _robustness_record(
                num_radial_bins=bins,
                width=0.10,
                mode=4,
                orientation=45.0,
                symmetry=0.60,
                coverage=0.58,
                status="WEAK_OR_INCOMPLETE_ANGULAR_CONTENT",
                reason="INSUFFICIENT_ANGULAR_COVERAGE",
            )
        )
        for width in (0.15, 0.20):
            records.append(
                _robustness_record(
                    num_radial_bins=bins,
                    width=width,
                    mode=4,
                    orientation=45.0 + 0.05 * (bins - 80) / 20.0,
                    symmetry=0.60 + 0.01 * (width - 0.15) / 0.05,
                    coverage=0.80,
                )
            )

    summary = _summarize_robustness_records(records, _config())

    assert summary["num_configurations"] == 9
    assert summary["num_valid_configurations"] == 6
    assert summary["num_insufficient_coverage_configurations"] == 3
    assert summary["consensus_dominant_mode"] == 4
    assert summary["dominant_mode_agreement_fraction"] == pytest.approx(1.0)
    assert summary["dominant_mode_agreement_fraction_all_configurations"] == pytest.approx(
        2.0 / 3.0
    )
    assert summary["robustness_accepted"] is True
    assert summary["robustness_status"] == "ROBUST_WITH_COVERAGE_LIMITATION"


def test_stable_mode_with_variable_symmetry_is_reported_separately() -> None:
    records: list[dict[str, object]] = []
    valid_specifications = [
        (60, 0.15, -0.39),
        (60, 0.20, -0.45),
        (80, 0.20, -0.61),
        (100, 0.15, -0.53),
        (100, 0.20, -0.36),
    ]
    for bins, width, symmetry in valid_specifications:
        records.append(
            _robustness_record(
                num_radial_bins=bins,
                width=width,
                mode=3,
                orientation=20.0,
                symmetry=symmetry,
                coverage=0.80,
            )
        )
    for bins, width in ((60, 0.10), (80, 0.10), (80, 0.15), (100, 0.10)):
        records.append(
            _robustness_record(
                num_radial_bins=bins,
                width=width,
                mode=3,
                orientation=20.0,
                symmetry=-0.45,
                coverage=0.60,
                status="WEAK_OR_INCOMPLETE_ANGULAR_CONTENT",
                reason="INSUFFICIENT_ANGULAR_COVERAGE",
            )
        )

    summary = _summarize_robustness_records(records, _config())

    assert summary["num_valid_configurations"] == 5
    assert summary["consensus_dominant_mode"] == 3
    assert summary["dominant_mode_stable"] is True
    assert summary["symmetry_stable"] is False
    assert summary["robustness_accepted"] is False
    assert summary["robustness_status"] == "MODE_STABLE_WITH_SECONDARY_VARIATION"


def test_quadrupolar_field_recovers_mode_symmetry_and_orientation() -> None:
    analysis = analyze_c66_case(
        _synthetic_case(mode=2, orientation_degrees=25.0),
        config=_config(),
    )

    assert analysis.summary["dominant_mode"] == 2
    assert analysis.summary["signed_symmetry_180"] > 0.98
    assert _periodic_difference(
        analysis.summary["dominant_orientation_degrees"],
        25.0,
        180.0,
    ) < 1.0
    assert analysis.summary["symmetry_center_distance_over_R"] == pytest.approx(0.0)
    assert analysis.summary["robustness_status"] == "OK"


def test_dipolar_field_is_antisymmetric_and_mode_one() -> None:
    analysis = analyze_c66_case(
        _synthetic_case(mode=1, orientation_degrees=40.0),
        config=_config(),
    )

    assert analysis.summary["dominant_mode"] == 1
    assert analysis.summary["signed_symmetry_180"] < -0.98
    assert _periodic_difference(
        analysis.summary["dominant_orientation_degrees"],
        40.0,
        360.0,
    ) < 1.0


def test_isotropic_field_has_no_interpretable_nonzero_mode() -> None:
    analysis = analyze_c66_case(
        _synthetic_case(
            mode=2,
            orientation_degrees=0.0,
            isotropic=True,
        ),
        config=_config(),
    )

    assert analysis.summary["dominant_mode"] == 0
    assert math.isnan(analysis.summary["dominant_orientation_degrees"])
    assert analysis.summary["signed_symmetry_180"] > 0.98
    assert analysis.summary["status"] == "VALIDATION_WARNING"


def test_noisy_quadrupole_remains_stable() -> None:
    analysis = analyze_c66_case(
        _synthetic_case(
            mode=2,
            orientation_degrees=30.0,
            noise_level=0.08,
        ),
        config=_config(),
    )

    assert analysis.summary["dominant_mode"] == 2
    assert analysis.summary["dominant_mode_agreement_fraction"] >= 0.75
    assert analysis.summary["orientation_dispersion_degrees"] < 5.0
    assert analysis.summary["signed_symmetry_180"] > 0.9


def test_local_center_search_tracks_decentered_pattern() -> None:
    true_center = (0.25, -0.25)
    analysis = analyze_c66_case(
        _synthetic_case(
            mode=2,
            orientation_degrees=15.0,
            pattern_center=true_center,
        ),
        config=_config(),
    )

    estimated = np.asarray(
        [analysis.summary["symmetry_center_x"], analysis.summary["symmetry_center_y"]]
    )
    assert np.linalg.norm(estimated - np.asarray(true_center)) <= 0.26
    assert analysis.summary["symmetry_center_distance_over_R"] > 0.0


def test_direct_symmetry_score_distinguishes_pair_and_antipair() -> None:
    angles = np.linspace(-math.pi, math.pi, 180, endpoint=False)
    pair = np.cos(2.0 * angles)
    antipair = np.cos(angles)

    assert compute_180_symmetry(pair) == pytest.approx(1.0, abs=1.0e-12)
    assert compute_180_symmetry(antipair) == pytest.approx(-1.0, abs=1.0e-12)


def test_global_symmetry_matches_radii_before_pi_rotation() -> None:
    radial = np.repeat(np.asarray([1.0, 1.5]), 180)
    angles = np.tile(np.linspace(-math.pi, math.pi, 180, endpoint=False), 2)
    weights = np.ones_like(angles)
    pair = np.cos(2.0 * angles) * np.repeat(np.asarray([1.0, 0.5]), 180)
    antipair = np.cos(angles) * np.repeat(np.asarray([1.0, 0.5]), 180)

    pair_result = compute_global_180_symmetry(
        radial,
        angles,
        pair,
        weights,
        defect_radius=1.0,
        max_radius_factor=2.0,
        num_radial_bins=4,
        num_angular_bins=180,
    )
    antipair_result = compute_global_180_symmetry(
        radial,
        angles,
        antipair,
        weights,
        defect_radius=1.0,
        max_radius_factor=2.0,
        num_radial_bins=4,
        num_angular_bins=180,
    )

    assert pair_result["signed_symmetry_180"] == pytest.approx(1.0, abs=1.0e-12)
    assert antipair_result["signed_symmetry_180"] == pytest.approx(-1.0, abs=1.0e-12)


def test_fourier_mode_power_fraction_is_one_for_pure_mode() -> None:
    angles = np.linspace(-math.pi, math.pi, 720, endpoint=False)
    values = np.cos(4.0 * (angles - 0.2))
    weights = np.ones_like(values)

    result = compute_angular_fourier(
        angles,
        values,
        weights,
        max_mode=8,
        num_angular_bins=180,
    )

    assert result["dominant_mode"] == 4
    assert result["dominant_mode_power_fraction"] == pytest.approx(1.0, abs=1.0e-12)


def test_regular_npz_loader_preserves_sign_and_infers_area(tmp_path: Path) -> None:
    x = np.linspace(-1.0, 1.0, 11)
    y = np.linspace(-2.0, 2.0, 21)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    values = xx - 0.5 * yy
    path = tmp_path / "field.npz"
    np.savez(path, x=x, y=y, gradient_C66=values)

    field = load_c66_field(path, field_name="gradient_C66")

    assert field.num_values == values.size
    assert np.max(np.abs(field.values)) == pytest.approx(1.0)
    assert np.min(field.values) < 0.0 < np.max(field.values)
    np.testing.assert_allclose(field.cell_measures, 0.04)


def test_cli_creates_composite_csv_and_json(tmp_path: Path) -> None:
    case = _synthetic_case(
        mode=2,
        orientation_degrees=20.0,
        points_per_axis=101,
    )
    x = np.unique(case.field.centers[:, 0])
    y = np.unique(case.field.centers[:, 1])
    values = case.field.values.reshape(y.size, x.size)
    input_path = tmp_path / "quad.npz"
    np.savez(input_path, x=x, y=y, gradient_C66=values)
    output_dir = tmp_path / "outputs"
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "postprocessing"
        / "analyze_C66_multipolar_pattern.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--field-name",
            "gradient_C66",
            "--output-dir",
            str(output_dir),
            "--geometry",
            "line",
            "--defect-center-x",
            "0",
            "--defect-center-y",
            "0",
            "--defect-radius",
            "1",
            "--case-name",
            "synthetic_quad",
            "--modality",
            "classical",
            "--aperture",
            "line",
            "--num-radial-bins",
            "40",
            "--radial-bin-values",
            "32,40",
            "--num-angular-bins",
            "120",
            "--annulus-width-values",
            "0.10,0.15",
            "--max-radius-factor",
            "3",
            "--primary-radius-max-factor",
            "2.4",
            "--center-search-steps",
            "3",
            "--no-save-pdf",
            "--overwrite",
            "--quiet",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "C66_multipolar_summary.csv").is_file()
    assert (output_dir / "C66_multipolar_metrics.json").is_file()
    assert (output_dir / "C66_multipolar_synthetic_quad.png").is_file()
    payload = json.loads(
        (output_dir / "C66_multipolar_metrics.json").read_text(encoding="utf-8")
    )
    assert payload["analysis_mode"] == "line-only"
    assert payload["scope"]["analysis_mode"] == "line-only"
    assert payload["cases"][0]["summary"]["analysis_mode"] == "line-only"
    assert payload["cases"][0]["summary"]["dominant_mode"] == 2


def test_campaign_scope_line_only_contract() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "postprocessing"
        / "analyze_C66_multipolar_pattern.py"
    )
    spec = importlib.util.spec_from_file_location("c66_multipolar_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = _synthetic_case(mode=2, orientation_degrees=0.0, points_per_axis=31)
    args = SimpleNamespace(geometry="line", expected_apertures=None)
    mode, expected = module._resolve_campaign_scope(args, (case,))
    assert mode == "line-only"
    assert expected == ("line",)
    box_case = CaseInput(
        field=case.field,
        defect=case.defect,
        metadata=CaseMetadata(
            case_name="box_case",
            observation_mode="classical_full",
            modality="classical",
            aperture="box",
        ),
    )
    with pytest.raises(ValueError, match="only line campaigns"):
        module._resolve_campaign_scope(args, (case, box_case))
