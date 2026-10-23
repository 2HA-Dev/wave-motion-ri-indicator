from __future__ import annotations

from elastic_fwi.campaigns.run_gradient_combination_campaign import (
    _validate_gradient_spatial_representation,
    build_parser,
)
from elastic_fwi.diagnostics.export_inverse_time_series_xdmf import (
    _gradient_export_semantics,
    _gradient_export_specs,
)


def test_gradient_export_specs_article_safe() -> None:
    specs = _gradient_export_specs("article_safe")

    assert len(specs) == 1
    assert specs[0]["manifest_key"] == "gradient_combinations"
    assert specs[0]["subdir"] == "gradient_combinations"
    assert specs[0]["absolute_value"] is True
    assert specs[0]["normalize"] is True


def test_gradient_export_specs_raw_signed() -> None:
    specs = _gradient_export_specs("raw_signed")

    assert len(specs) == 1
    assert specs[0]["manifest_key"] == "gradient_combinations"
    assert specs[0]["subdir"] == "gradient_combinations_raw_signed"
    assert specs[0]["absolute_value"] is False
    assert specs[0]["normalize"] is False


def test_gradient_export_specs_both() -> None:
    specs = _gradient_export_specs("both")

    assert [s["manifest_key"] for s in specs] == [
        "gradient_combinations",
        "gradient_combinations_raw_signed",
    ]
    assert [s["subdir"] for s in specs] == [
        "gradient_combinations",
        "gradient_combinations_raw_signed",
    ]
    assert specs[0]["absolute_value"] is True
    assert specs[0]["normalize"] is True
    assert specs[1]["absolute_value"] is False
    assert specs[1]["normalize"] is False


def test_campaign_parser_accepts_gradient_export_mode_both() -> None:
    args = build_parser().parse_args([
        "--output-dir",
        "outputs/tmp",
        "--gradient-export-mode",
        "both",
    ])

    assert args.gradient_export_mode == "both"


def test_gradient_export_semantics_disambiguate_article_safe_from_raw_signed() -> None:
    article = _gradient_export_semantics("article_safe")
    raw = _gradient_export_semantics("raw_signed")

    assert article["scientific_role"] == "normalized_qualitative_sensitivity_image"
    assert article["manuscript_label"] == "qualitative image"
    assert article["absolute_value"] is True
    assert article["absolute_value_stage"] == "after_time_integration"
    assert article["normalize"] is True
    assert article["signed"] is False
    assert article["manuscript_do_not_call"] == "raw gradient"

    assert raw["scientific_role"] == "signed_non_normalized_material_gradient_density"
    assert raw["manuscript_label"] == "signed non-normalized gradient density"
    assert raw["absolute_value"] is False
    assert raw["absolute_value_stage"] == "not_applied"
    assert raw["normalize"] is False
    assert raw["signed"] is True


def test_gradient_export_specs_carry_semantic_metadata() -> None:
    specs = _gradient_export_specs("both")

    assert specs[0]["semantics"]["scientific_role"] == "normalized_qualitative_sensitivity_image"
    assert specs[1]["semantics"]["scientific_role"] == "signed_non_normalized_material_gradient_density"


def test_campaign_spatial_contract_requires_l2_density_for_all_modalities() -> None:
    modalities = (
        "classical_full",
        "standard_reflection_only",
        "standard_transmission_only",
        "residual_interaction",
    )
    payload = {
        "exports": {
            key: {
                mode: {
                    "metadata": {
                        "spatial_representation": "DG0_L2_cell_average_density",
                        "cell_measure_normalization": True,
                    }
                }
                for mode in modalities
            }
            for key in (
                "gradient_combinations",
                "gradient_combinations_raw_signed",
            )
        }
    }

    report = _validate_gradient_spatial_representation(payload)

    assert report["status"] == "ok"
    assert report["num_rows"] == 8


def test_campaign_spatial_contract_rejects_cell_integral_export() -> None:
    modalities = (
        "classical_full",
        "standard_reflection_only",
        "standard_transmission_only",
        "residual_interaction",
    )
    payload = {
        "exports": {
            "gradient_combinations": {
                mode: {
                    "metadata": {
                        "spatial_representation": "DG0_cell_integral",
                        "cell_measure_normalization": False,
                    }
                }
                for mode in modalities
            }
        }
    }

    report = _validate_gradient_spatial_representation(payload)

    assert report["status"] == "error"
