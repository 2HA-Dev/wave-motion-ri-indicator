from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_manifest(manifest_path: Path) -> None:
    m = json.loads(manifest_path.read_text())
    exports = m["exports"]

    assert exports["gradient_export_mode"] == "both"

    specs = exports["gradient_export_specs"]

    assert specs[0]["label"] == "article_safe"
    assert specs[0]["absolute_value"] is True
    assert specs[0]["normalize"] is True
    assert (
        specs[0]["semantics"]["scientific_role"]
        == "normalized_qualitative_sensitivity_image"
    )

    assert specs[1]["label"] == "raw_signed"
    assert specs[1]["absolute_value"] is False
    assert specs[1]["normalize"] is False
    assert (
        specs[1]["semantics"]["scientific_role"]
        == "signed_non_normalized_material_gradient_density"
    )

    article = exports["gradient_combinations"]
    raw = exports["gradient_combinations_raw_signed"]

    for combo, payload in article.items():
        expected_role = (
            "residual_wave_interaction_indicator"
            if combo == "residual_interaction"
            else "normalized_qualitative_sensitivity_image"
        )

        assert payload["gradient_export_label"] == "article_safe", combo
        assert payload["scientific_role"] == expected_role, combo
        assert payload["metadata"]["absolute_value"] is True, combo
        assert (
            payload["metadata"]["absolute_value_stage"]
            == "after_time_integration"
        ), combo
        assert payload["metadata"]["normalize"] is True, combo

    for combo, payload in raw.items():
        expected_role = (
            "signed_non_normalized_residual_wave_interaction_density"
            if combo == "residual_interaction"
            else "signed_non_normalized_material_gradient_density"
        )

        assert payload["gradient_export_label"] == "raw_signed", combo
        assert payload["scientific_role"] == expected_role, combo
        assert payload["metadata"]["absolute_value"] is False, combo
        assert payload["metadata"]["absolute_value_stage"] == "not_applied", combo
        assert payload["metadata"]["normalize"] is False, combo


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check that article_safe qualitative images and raw_signed signed "
            "gradient-density exports are semantically separated in an inverse "
            "time-series manifest."
        )
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to inverse_time_series_manifest.json",
    )
    args = parser.parse_args()

    check_manifest(args.manifest)
    print(
        "OK: article_safe and raw_signed are semantically separated "
        "in the manifest."
    )


if __name__ == "__main__":
    main()
