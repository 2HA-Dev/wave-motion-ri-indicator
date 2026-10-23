from __future__ import annotations

import csv
import json
from pathlib import Path

from elastic_fwi.metrics.campaign_assessment import write_assessment_outputs


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_campaign_assessment_ranks_methods_and_writes_outputs(tmp_path):
    metrics_csv = tmp_path / "metrics.csv"

    rows = [
        {
            "method": "reflection_only",
            "imaging_family": "direct_adjoint_material_gradients",
            "field_name": "gradient_rho",
            "status": "ok",
            "peak_to_artifact_ratio": "1.0",
            "energy_outside_defect": "10.0",
            "localization_error": "2.0",
            "false_maxima_count": "4",
            "energy_in_defect": "1.0",
            "max_value_in_defect": "1.0",
            "max_value_outside_defect": "1.0",
        },
        {
            "method": "transmission_only",
            "imaging_family": "direct_adjoint_material_gradients",
            "field_name": "gradient_rho",
            "status": "ok",
            "peak_to_artifact_ratio": "3.0",
            "energy_outside_defect": "2.0",
            "localization_error": "0.5",
            "false_maxima_count": "1",
            "energy_in_defect": "2.0",
            "max_value_in_defect": "3.0",
            "max_value_outside_defect": "1.0",
        },
    ]

    _write_csv(metrics_csv, rows)

    payload = write_assessment_outputs(
        metrics_csv=metrics_csv,
        output_dir=tmp_path / "assessment",
        preferred_field="gradient_rho",
    )

    assert payload["claim_candidates"]["validity"] == "candidate_quantitative_assessment"
    assert payload["claim_candidates"]["best_peak_to_artifact_ratio"][0]["method"] == "transmission_only"
    assert payload["claim_candidates"]["lowest_energy_outside_defect"][0]["method"] == "transmission_only"
    assert payload["claim_candidates"]["num_eligible_preferred_groups"] == 2
    assert Path(payload["assessment_json"]).exists()
    assert Path(payload["assessment_markdown"]).exists()

    data = json.loads(Path(payload["assessment_json"]).read_text(encoding="utf-8"))
    assert data["summary"]["num_input_rows"] == 2
    assert data["summary"]["num_scientifically_eligible_groups"] == 2


def test_campaign_assessment_excludes_no_inside_samples_from_claims(tmp_path):
    metrics_csv = tmp_path / "metrics_no_inside.csv"

    rows = [
        {
            "method": "reflection_only",
            "imaging_family": "direct_adjoint_material_gradients",
            "field_name": "gradient_rho",
            "status": "no_inside_samples",
            "peak_to_artifact_ratio": "999.0",
            "energy_outside_defect": "0.0",
            "localization_error": "0.0",
            "false_maxima_count": "0",
            "energy_in_defect": "0.0",
            "max_value_in_defect": "0.0",
            "max_value_outside_defect": "0.0",
        },
        {
            "method": "transmission_only",
            "imaging_family": "direct_adjoint_material_gradients",
            "field_name": "gradient_C11",
            "status": "ok",
            "peak_to_artifact_ratio": "2.0",
            "energy_outside_defect": "3.0",
            "localization_error": "1.0",
            "false_maxima_count": "1",
            "energy_in_defect": "2.0",
            "max_value_in_defect": "2.0",
            "max_value_outside_defect": "1.0",
        },
    ]

    _write_csv(metrics_csv, rows)

    payload = write_assessment_outputs(
        metrics_csv=metrics_csv,
        output_dir=tmp_path / "assessment_no_inside",
        preferred_field="gradient_rho",
    )

    assert (
        payload["claim_candidates"]["validity"]
        == "not_scientifically_conclusive_no_valid_preferred_groups"
    )
    assert payload["claim_candidates"]["num_eligible_preferred_groups"] == 0
    assert payload["claim_candidates"]["best_peak_to_artifact_ratio"] == []
    assert payload["summary"]["num_nonzero_groups"] == 2
    assert payload["summary"]["num_scientifically_eligible_groups"] == 1

    markdown = Path(payload["assessment_markdown"]).read_text(encoding="utf-8")
    assert "no_inside_samples" in markdown
    assert "No scientifically eligible finite data" in markdown
