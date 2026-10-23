from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

from elastic_fwi.campaigns.run_gradient_combination_campaign import (
    _config_for_args,
    build_parser,
)


def test_article_L73_T26_no_absorbing_preset_geometry() -> None:
    preset = "article_L73_N73_face_to_face_r025_T26_no_absorbing"

    args = build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/article_L73_test",
            "--preset",
            preset,
        ]
    )
    config = _config_for_args(args)

    assert config["domain_length_x"] == 73.0
    assert config["domain_length_y"] == 73.0
    assert config["nx"] == 73
    assert config["ny"] == 73

    assert config["absorbing_layer"] is False
    assert config["absorbing_layer_width"] is None

    assert (config["source_array_start_x"], config["source_array_start_y"]) == (30.0, 30.0)
    assert (config["source_array_end_x"], config["source_array_end_y"]) == (43.0, 30.0)

    assert (config["receiver_array_start_x"], config["receiver_array_start_y"]) == (30.0, 43.0)
    assert (config["receiver_array_end_x"], config["receiver_array_end_y"]) == (43.0, 43.0)

    domain_center = (
        0.5 * config["domain_length_x"],
        0.5 * config["domain_length_y"],
    )
    roi_center = (
        0.5 * (config["source_array_start_x"] + config["source_array_end_x"]),
        0.5 * (config["source_array_start_y"] + config["receiver_array_start_y"]),
    )
    defect_center = (
        config["defect_center_x"],
        config["defect_center_y"],
    )

    assert domain_center == roi_center == defect_center == (36.5, 36.5)

    assert config["observed_material_kind"] == "localized_disk_defect"
    assert config["contrast_factor"] == 0.75
    assert math.isclose(
        config["dt"] * config["num_steps"],
        26.0,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    )


def test_ranked_inverse_manifest_filename_convention() -> None:
    output_dir = Path("outputs/dummy")
    comm = SimpleNamespace(rank=0, size=2)

    inverse_manifest = str(
        output_dir
        / "inverse"
        / (
            f"inverse_time_series_manifest_rank{comm.rank:04d}of{comm.size:04d}.json"
            if comm.size > 1
            else "inverse_time_series_manifest.json"
        )
    )

    assert inverse_manifest.endswith(
        "outputs/dummy/inverse/inverse_time_series_manifest_rank0000of0002.json"
    )
