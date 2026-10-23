from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from elastic_fwi.postprocessing.article_figures import CampaignInfo


SCRIPT = (
    Path(__file__)
    .resolve()
    .parents[2]
    / "scripts"
    / "revision_wave_motion"
    / "export_normalized_sensitivity_map_gallery.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "export_normalized_sensitivity_map_gallery",
        SCRIPT,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    # dataclasses with postponed annotations require the module
    # to be visible in sys.modules during class construction.
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(
            module
        )
    except Exception:
        sys.modules.pop(
            spec.name,
            None,
        )
        raise

    return module


def test_canonical_geometry_aliases():
    module = load_module()

    assert (
        module.canonical_geometry("line")
        == "line"
    )

    assert (
        module.canonical_geometry("box")
        == "box"
    )

    assert (
        module.canonical_geometry(
            "box_except_gamma1"
        )
        == "box"
    )


def test_canonical_geometry_rejects_unknown():
    module = load_module()

    with pytest.raises(ValueError):
        module.canonical_geometry(
            "unknown"
        )


def test_normalize_values_uses_common_roi():
    module = load_module()

    centers = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    values = np.array(
        [
            -1.0,
            2.0,
            -4.0,
            8.0,
        ],
        dtype=float,
    )

    normalized, amplitude = (
        module.normalize_values(
            centers,
            values,
            (0.0, 1.0, 0.0, 0.5),
        )
    )

    assert amplitude == pytest.approx(
        2.0
    )

    assert normalized.tolist() == pytest.approx(
        [
            0.5,
            1.0,
            1.0,
            1.0,
        ]
    )


def test_parser_accepts_repeated_campaigns(
    tmp_path,
):
    module = load_module()

    parser = module.build_parser()

    args = parser.parse_args(
        [
            "--input-inventory",
            str(tmp_path / "maps.csv"),
            "--campaign-dir",
            str(tmp_path / "line"),
            "--campaign-dir",
            str(tmp_path / "box"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert len(args.campaign_dir) == 2
    assert args.dpi == 300
    assert args.map_grid_size == 500


def test_draw_raster_overlays_gamma1_and_gamma2_with_requested_styles(tmp_path):
    module = load_module()
    plt = module._configure_matplotlib()
    campaign = CampaignInfo(
        root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        campaign_id="line",
        geometry_kind="line",
        config={
            "source_array_start_x": 0.0,
            "source_array_start_y": 1.0,
            "source_array_end_x": 2.0,
            "source_array_end_y": 1.0,
            "source_array_num_elements": 3,
            "receiver_array_start_x": 0.0,
            "receiver_array_start_y": 3.0,
            "receiver_array_end_x": 2.0,
            "receiver_array_end_y": 3.0,
            "receiver_array_num_elements": 3,
        },
        payload={},
    )
    item = module.RasterMap(
        canonical_geometry="line",
        original_geometry_kind="line",
        campaign_id="line",
        modality="classical_full",
        parameter_name="rho",
        field_name="gradient_rho",
        source_path=tmp_path / "field.pvd",
        source_sha256="unused",
        source_minimum=0.0,
        source_maximum=1.0,
        normalization_amplitude_roi=1.0,
        display_minimum=0.0,
        display_maximum=1.0,
        raster=np.zeros((4, 4)),
        extent=(0.0, 4.0, 0.0, 4.0),
        rendering="test",
    )

    fig, ax = plt.subplots()
    module.draw_raster(
        ax,
        item,
        campaign,
        roi=(0.0, 4.0, 0.0, 4.0),
        cmap=module.colormaps["coolwarm"],
        title=module.PROPERTY_LABELS["rho"],
    )

    assert len(ax.lines) == 2
    assert ax.lines[0].get_color() == module.ARRAY_STYLES["gamma1"]["color"]
    assert ax.lines[1].get_color() == module.ARRAY_STYLES["gamma2"]["color"]
    assert ax.lines[0].get_linestyle() == "--"
    assert ax.lines[1].get_linestyle() == "--"
    assert ax.title.get_fontsize() == pytest.approx(13.0)
    plt.close(fig)
