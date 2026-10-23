#!/usr/bin/env python3
"""Export the complete normalized sensitivity-map gallery.

The script consumes existing article-safe campaign maps only. It does not
rerun a forward, adjoint, or gradient computation.

Outputs:
  * 40 individual PNG maps;
  * 8 geometry/modality boards containing five properties;
  * 2 geometry boards containing four modalities by five properties;
  * one source-to-image CSV manifest;
  * one JSON summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib import colormaps
from PIL import Image

from elastic_fwi.postprocessing.article_figures import (
    ArticleFigureConfig,
    CampaignInfo,
    _configure_matplotlib,
    _field_sources,
    _weighted_raster,
    discover_campaigns,
    parse_roi,
    receiver_points,
    sanitize_identifier,
    source_points,
)
from elastic_fwi.postprocessing.signed_sensitivity_fields import (
    load_vtk_cell_field,
)


GEOMETRIES = ("line", "box")

MODALITIES = (
    "classical_full",
    "standard_reflection_only",
    "standard_transmission_only",
    "residual_interaction",
)

PROPERTIES = (
    "rho",
    "C11",
    "C22",
    "C12",
    "C66",
)

GEOMETRY_ALIASES = {
    "line": "line",
    "box": "box",
    "box_except_gamma1": "box",
}

MODALITY_LABELS = {
    "classical_full": "Full classical kernel",
    "standard_reflection_only": "Reflection classical kernel",
    "standard_transmission_only": "Transmission classical kernel",
    "residual_interaction": "Residual interaction",
}

# Which acquisition arrays (Gamma_1 = reflection/source, Gamma_2 =
# transmission/receiver) are overlaid for each modality: the full-data
# kernel shows both, the two single-array kernels show only the array
# they actually use, and the residual-interaction indicator (which
# combines both arrays) always shows both.
MODALITY_ARRAY_VISIBILITY = {
    "classical_full": ("gamma1", "gamma2"),
    "standard_reflection_only": ("gamma1",),
    "standard_transmission_only": ("gamma2",),
    "residual_interaction": ("gamma1", "gamma2"),
}

PROPERTY_LABELS = {
    "rho": r"$\rho$",
    "C11": r"$C_{11}$",
    "C22": r"$C_{22}$",
    "C12": r"$C_{12}$",
    "C66": r"$C_{66}$",
}

ARRAY_STYLES = {
    "gamma1": {
        "color": "#d62728",
        "linestyle": "--",
        "linewidth": 3.2,
    },
    "gamma2": {
        "color": "#149c47",
        "linestyle": "--",
        "linewidth": 3.2,
    },
}

MODALITY_COLORS = {
    "classical_full": "#755638",
    "standard_reflection_only": "#d62728",
    "standard_transmission_only": "#149c47",
    "residual_interaction": "#1f77b4",
}


@dataclass(frozen=True)
class RasterMap:
    canonical_geometry: str
    original_geometry_kind: str
    campaign_id: str
    modality: str
    parameter_name: str
    field_name: str
    source_path: Path
    source_sha256: str
    source_minimum: float
    source_maximum: float
    normalization_amplitude_roi: float
    display_minimum: float
    display_maximum: float
    raster: np.ndarray
    extent: tuple[float, float, float, float]
    rendering: str


def now_utc() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                8 * 1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def canonical_geometry(value: str) -> str:
    key = str(value).strip()

    if key not in GEOMETRY_ALIASES:
        raise ValueError(
            f"Unsupported geometry label: {value!r}"
        )

    return GEOMETRY_ALIASES[key]


def normalize_values(
    centers: np.ndarray,
    values: np.ndarray,
    roi: tuple[float, float, float, float],
) -> tuple[np.ndarray, float]:
    coordinates = np.asarray(
        centers,
        dtype=float,
    )

    field = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    if coordinates.ndim != 2:
        raise ValueError(
            "Cell centres must form a two-dimensional array."
        )

    if coordinates.shape[0] != field.size:
        raise ValueError(
            "Cell centres and scalar values have incompatible sizes."
        )

    if not np.isfinite(coordinates).all():
        raise ValueError(
            "Cell centres contain non-finite values."
        )

    if not np.isfinite(field).all():
        raise ValueError(
            "Field values contain non-finite values."
        )

    xmin, xmax, ymin, ymax = roi

    roi_mask = (
        (coordinates[:, 0] >= xmin)
        & (coordinates[:, 0] <= xmax)
        & (coordinates[:, 1] >= ymin)
        & (coordinates[:, 1] <= ymax)
    )

    if not np.any(roi_mask):
        raise ValueError(
            "No field sample lies inside the common ROI."
        )

    absolute = np.abs(field)

    amplitude = float(
        np.max(
            absolute[roi_mask]
        )
    )

    if amplitude > 0.0:
        display = absolute / amplitude
    else:
        display = np.zeros_like(
            absolute,
            dtype=float,
        )

    display = np.clip(
        display,
        0.0,
        1.0,
    )

    return display, amplitude


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()

    with Image.open(path) as image:
        width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid PNG dimensions for {path}."
        )

    return int(width), int(height)


def save_png(
    fig: Any,
    path: Path,
    *,
    dpi: int,
) -> tuple[int, int]:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=dpi,
        facecolor="white",
        bbox_inches=None,
    )

    if not path.is_file():
        raise RuntimeError(
            f"PNG was not created: {path}"
        )

    if path.stat().st_size <= 0:
        raise RuntimeError(
            f"PNG is empty: {path}"
        )

    return image_size(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export 40 normalized qualitative sensitivity maps, "
            "8 modality boards, and 2 geometry boards from existing "
            "LINE and BOX campaigns."
        )
    )

    parser.add_argument(
        "--input-inventory",
        type=Path,
        required=True,
        help=(
            "Canonical 40-map CSV inventory generated during "
            "the Priority-10 preflight."
        ),
    )

    parser.add_argument(
        "--campaign-dir",
        type=Path,
        action="append",
        required=True,
        help=(
            "Campaign root containing "
            "gradient_combination_campaign_manifest.json. "
            "Repeat for LINE and BOX."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--roi",
        default="auto",
        help=(
            "Common physical ROI: auto, full, or "
            "xmin,xmax,ymin,ymax."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--map-grid-size",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--max-vector-points",
        type=int,
        default=50000,
        help=(
            "Recorded for compatibility with the article-figure "
            "configuration. This gallery uses the area-weighted raster path "
            "for identical individual-image dimensions."
        ),
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate the campaign/inventory correspondence "
            "without loading or plotting the 40 VTK fields."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    return parser


def prepare_output_dir(
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    if output_dir.exists():
        has_content = any(
            output_dir.iterdir()
        )

        if has_content and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use --overwrite."
            )

        if overwrite:
            shutil.rmtree(
                output_dir
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_inventory(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    required = {
        "campaign_id",
        "geometry_kind",
        "canonical_geometry",
        "map_name",
        "modality",
        "parameter_name",
        "source_format",
        "source_path",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            "Input inventory is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    if len(frame) != 40:
        raise ValueError(
            f"Expected 40 inventory rows, found {len(frame)}."
        )

    frame = frame.copy()

    frame[
        "canonical_geometry"
    ] = frame[
        "canonical_geometry"
    ].astype(str).map(
        canonical_geometry
    )

    frame[
        "resolved_source_path"
    ] = frame[
        "source_path"
    ].map(
        lambda value: str(
            Path(str(value))
            .expanduser()
            .resolve()
        )
    )

    keys = [
        "canonical_geometry",
        "modality",
        "parameter_name",
    ]

    if frame[keys].drop_duplicates().shape[0] != 40:
        raise ValueError(
            "The inventory does not contain a unique "
            "2 x 4 x 5 Cartesian product."
        )

    if set(
        frame["canonical_geometry"]
    ) != set(GEOMETRIES):
        raise ValueError(
            "The inventory must contain LINE and BOX."
        )

    if set(
        frame["modality"]
    ) != set(MODALITIES):
        raise ValueError(
            "The inventory does not contain the four modalities."
        )

    if set(
        frame["parameter_name"]
    ) != set(PROPERTIES):
        raise ValueError(
            "The inventory does not contain the five properties."
        )

    if frame[
        "resolved_source_path"
    ].nunique() != 40:
        raise ValueError(
            "The 40 source paths are not unique."
        )

    for raw_path in frame[
        "resolved_source_path"
    ]:
        source = Path(raw_path)

        if not source.is_file():
            raise FileNotFoundError(
                source
            )

        if source.stat().st_size <= 0:
            raise ValueError(
                f"Empty source descriptor: {source}"
            )

    return frame


def discover_source_records(
    campaigns: Sequence[CampaignInfo],
    inventory: pd.DataFrame,
) -> tuple[
    dict[str, CampaignInfo],
    dict[
        tuple[str, str, str],
        tuple[str, Path],
    ],
]:
    campaigns_by_geometry: dict[
        str,
        CampaignInfo,
    ] = {}

    source_records: dict[
        tuple[str, str, str],
        tuple[str, Path],
    ] = {}

    inventory_by_path = {
        str(row.resolved_source_path): row
        for row in inventory.itertuples(
            index=False
        )
    }

    for campaign in campaigns:
        geometry = canonical_geometry(
            campaign.geometry_kind
        )

        if geometry in campaigns_by_geometry:
            raise ValueError(
                f"Multiple campaigns resolve to geometry {geometry!r}."
            )

        campaigns_by_geometry[
            geometry
        ] = campaign

        discovered = _field_sources(
            campaign,
            modalities=MODALITIES,
            parameters=PROPERTIES,
            raw_signed=False,
        )

        if len(discovered) != 20:
            raise ValueError(
                f"Expected 20 article-safe maps for "
                f"{campaign.campaign_id}, found {len(discovered)}."
            )

        for (
            modality,
            parameter,
            field_name,
            raw_path,
        ) in discovered:
            path = (
                Path(raw_path)
                .expanduser()
                .resolve()
            )

            key = (
                geometry,
                str(modality),
                str(parameter),
            )

            if key in source_records:
                raise ValueError(
                    f"Duplicate campaign map key: {key}"
                )

            inventory_row = inventory_by_path.get(
                str(path)
            )

            if inventory_row is None:
                raise ValueError(
                    f"Discovered source is absent from the "
                    f"validated gallery inventory: {path}"
                )

            if (
                str(
                    inventory_row.canonical_geometry
                )
                != geometry
                or str(
                    inventory_row.modality
                )
                != str(modality)
                or str(
                    inventory_row.parameter_name
                )
                != str(parameter)
            ):
                raise ValueError(
                    "Campaign discovery and inventory metadata "
                    f"disagree for {path}."
                )

            source_records[key] = (
                str(field_name),
                path,
            )

    expected_keys = {
        (
            geometry,
            modality,
            parameter,
        )
        for geometry in GEOMETRIES
        for modality in MODALITIES
        for parameter in PROPERTIES
    }

    if set(
        source_records
    ) != expected_keys:
        missing = expected_keys - set(
            source_records
        )

        unexpected = set(
            source_records
        ) - expected_keys

        raise ValueError(
            "Incomplete campaign source product. "
            f"Missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}."
        )

    if set(
        campaigns_by_geometry
    ) != set(GEOMETRIES):
        raise ValueError(
            "Exactly one LINE campaign and one BOX campaign "
            "are required."
        )

    return (
        campaigns_by_geometry,
        source_records,
    )


def color_settings() -> tuple[Any, str]:
    # Matplotlib coolwarm is visually close to ParaView's
    # Cool to Warm preset: blue low values, neutral middle,
    # and red high values.
    cmap = colormaps["coolwarm"].copy()

    # Empty histogram bins are rendering artefacts inside a
    # continuous FE domain. Display them with the lowest colour
    # rather than as transparent checkerboard cells.
    cmap.set_bad(
        cmap(0.0)
    )

    label = (
        "Normalized absolute sensitivity "
        "(qualitative)"
    )

    return cmap, label


def draw_raster(
    ax: Any,
    item: RasterMap,
    campaign: CampaignInfo,
    *,
    roi: tuple[float, float, float, float],
    cmap: str,
    title: str | None = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    visible_arrays: tuple[str, ...] = ("gamma1", "gamma2"),
) -> Any:
    ax.set_facecolor(
        cmap(0.0)
    )

    artist = ax.imshow(
        item.raster,
        origin="lower",
        extent=item.extent,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
        aspect="equal",
        rasterized=True,
    )

    # Overlay the acquisition apertures without markers or repeated text:
    # Gamma_1 is the source/reflection array and Gamma_2 the transmission
    # aperture. Coordinates come directly from each campaign manifest.
    # Only the arrays listed in visible_arrays are drawn, so that each
    # row shows exactly the array(s) its imaging modality actually uses.
    for array_name, points, style in (
        ("gamma1", source_points(campaign), ARRAY_STYLES["gamma1"]),
        ("gamma2", receiver_points(campaign), ARRAY_STYLES["gamma2"]),
    ):
        if array_name not in visible_arrays:
            continue

        if points is not None and points.size:
            ax.plot(
                points[:, 0],
                points[:, 1],
                zorder=5,
                solid_capstyle="round",
                dash_capstyle="round",
                **style,
            )

    ax.grid(False)

    ax.set_xlim(
        roi[0],
        roi[1],
    )

    ax.set_ylim(
        roi[2],
        roi[3],
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.tick_params(
        axis="both",
        which="both",
        labelsize=7,
        length=2.5,
        width=0.6,
    )

    if title is not None:
        ax.set_title(
            title,
            fontsize=13,
            fontweight="semibold",
            pad=7,
        )

    ax.set_xlabel(
        "x" if show_xlabel else ""
    )

    ax.set_ylabel(
        "y" if show_ylabel else ""
    )

    if not show_xlabel:
        ax.tick_params(
            axis="x",
            labelbottom=False,
        )

    if not show_ylabel:
        ax.tick_params(
            axis="y",
            labelleft=False,
        )

    return artist


def build_gallery(
    *,
    output_dir: Path,
    campaigns_by_geometry: Mapping[
        str,
        CampaignInfo,
    ],
    source_records: Mapping[
        tuple[str, str, str],
        tuple[str, Path],
    ],
    roi: tuple[float, float, float, float],
    config: ArticleFigureConfig,
    quiet: bool,
) -> tuple[
    list[dict[str, Any]],
    list[Path],
]:
    plt = _configure_matplotlib()

    cmap, colorbar_label = (
        color_settings()
    )

    rasters: dict[
        tuple[str, str, str],
        RasterMap,
    ] = {}

    for geometry in GEOMETRIES:
        campaign = campaigns_by_geometry[
            geometry
        ]

        for modality in MODALITIES:
            for parameter in PROPERTIES:
                key = (
                    geometry,
                    modality,
                    parameter,
                )

                field_name, source_path = (
                    source_records[key]
                )

                if not quiet:
                    print(
                        "[GALLERY] loading "
                        f"{geometry} / {modality} / {parameter}"
                    )

                loaded = load_vtk_cell_field(
                    source_path,
                    field_name,
                )

                centers = np.asarray(
                    loaded.centers,
                    dtype=float,
                )

                values = np.asarray(
                    loaded.values,
                    dtype=float,
                ).reshape(-1)

                weights = np.asarray(
                    loaded.cell_measures,
                    dtype=float,
                ).reshape(-1)

                if (
                    centers.shape[0]
                    != values.size
                    or values.size
                    != weights.size
                ):
                    raise ValueError(
                        f"Incompatible DG0 arrays for {source_path}."
                    )

                if (
                    not np.isfinite(weights).all()
                    or np.any(weights <= 0.0)
                ):
                    raise ValueError(
                        f"Invalid DG0 cell measures for {source_path}."
                    )

                display, amplitude = (
                    normalize_values(
                        centers,
                        values,
                        roi,
                    )
                )

                raster, extent = (
                    _weighted_raster(
                        centers,
                        display,
                        weights,
                        roi,
                        config.map_grid_size,
                    )
                )

                finite_display = display[
                    np.isfinite(display)
                ]

                if finite_display.size == 0:
                    raise ValueError(
                        f"No finite display values for {source_path}."
                    )

                rasters[key] = RasterMap(
                    canonical_geometry=geometry,
                    original_geometry_kind=(
                        campaign.geometry_kind
                    ),
                    campaign_id=(
                        campaign.campaign_id
                    ),
                    modality=modality,
                    parameter_name=parameter,
                    field_name=field_name,
                    source_path=source_path,
                    source_sha256=sha256_file(
                        source_path
                    ),
                    source_minimum=float(
                        np.min(values)
                    ),
                    source_maximum=float(
                        np.max(values)
                    ),
                    normalization_amplitude_roi=(
                        amplitude
                    ),
                    display_minimum=float(
                        np.min(finite_display)
                    ),
                    display_maximum=float(
                        np.max(finite_display)
                    ),
                    raster=raster,
                    extent=tuple(
                        float(value)
                        for value in extent
                    ),
                    rendering=(
                        "area_weighted_raster_"
                        f"{config.map_grid_size}x"
                        f"{config.map_grid_size}"
                    ),
                )

    if len(rasters) != 40:
        raise AssertionError(
            f"Expected 40 loaded rasters, found {len(rasters)}."
        )

    individual_dir = (
        output_dir
        / "individual"
    )

    modality_board_dir = (
        output_dir
        / "modality_boards"
    )

    geometry_board_dir = (
        output_dir
        / "geometry_boards"
    )

    manifest_rows: list[
        dict[str, Any]
    ] = []

    generated_pngs: list[
        Path
    ] = []

    modality_board_paths: dict[
        tuple[str, str],
        Path,
    ] = {}

    geometry_board_paths: dict[
        str,
        Path,
    ] = {}

    # --------------------------------------------------------------
    # Forty fixed-size individual PNG files
    # --------------------------------------------------------------

    for geometry in GEOMETRIES:
        campaign = campaigns_by_geometry[
            geometry
        ]

        for modality in MODALITIES:
            for parameter in PROPERTIES:
                key = (
                    geometry,
                    modality,
                    parameter,
                )

                item = rasters[key]

                fig, ax = plt.subplots(
                    figsize=(6.4, 5.6),
                    constrained_layout=True,
                )

                artist = draw_raster(
                    ax,
                    item,
                    campaign,
                    roi=roi,
                    cmap=cmap,
                    title=(
                        f"{geometry.upper()} — "
                        f"{MODALITY_LABELS[modality]} — "
                        f"{PROPERTY_LABELS[parameter]}"
                    ),
                    visible_arrays=MODALITY_ARRAY_VISIBILITY[
                        modality
                    ],
                )

                colorbar = fig.colorbar(
                    artist,
                    ax=ax,
                    shrink=0.86,
                    pad=0.03,
                )

                colorbar.set_label(
                    colorbar_label
                )

                relative_path = (
                    Path("individual")
                    / sanitize_identifier(
                        geometry
                    )
                    / sanitize_identifier(
                        modality
                    )
                    / (
                        f"{sanitize_identifier(geometry)}"
                        f"__{sanitize_identifier(modality)}"
                        f"__{sanitize_identifier(parameter)}.png"
                    )
                )

                output_path = (
                    output_dir
                    / relative_path
                )

                width, height = save_png(
                    fig,
                    output_path,
                    dpi=config.dpi,
                )

                plt.close(fig)

                generated_pngs.append(
                    output_path
                )

                manifest_rows.append(
                    {
                        "canonical_geometry": (
                            geometry
                        ),
                        "original_geometry_kind": (
                            item.original_geometry_kind
                        ),
                        "campaign_id": (
                            item.campaign_id
                        ),
                        "modality": modality,
                        "parameter_name": (
                            parameter
                        ),
                        "field_name": (
                            item.field_name
                        ),
                        "source_path": str(
                            item.source_path
                        ),
                        "source_sha256": (
                            item.source_sha256
                        ),
                        "source_minimum": (
                            item.source_minimum
                        ),
                        "source_maximum": (
                            item.source_maximum
                        ),
                        "normalization_amplitude_roi": (
                            item.normalization_amplitude_roi
                        ),
                        "display_minimum": (
                            item.display_minimum
                        ),
                        "display_maximum": (
                            item.display_maximum
                        ),
                        "rendering": (
                            item.rendering
                        ),
                        "roi_xmin": roi[0],
                        "roi_xmax": roi[1],
                        "roi_ymin": roi[2],
                        "roi_ymax": roi[3],
                        "individual_png": str(
                            relative_path
                        ),
                        "individual_png_sha256": (
                            sha256_file(
                                output_path
                            )
                        ),
                        "individual_width_px": (
                            width
                        ),
                        "individual_height_px": (
                            height
                        ),
                    }
                )

    # --------------------------------------------------------------
    # Eight boards: one geometry/modality, five properties
    # --------------------------------------------------------------

    for geometry in GEOMETRIES:
        campaign = campaigns_by_geometry[
            geometry
        ]

        for modality in MODALITIES:
            fig, axes = plt.subplots(
                1,
                len(PROPERTIES),
                figsize=(16.0, 3.7),
                squeeze=False,
                constrained_layout=True,
            )

            artist = None

            for column, parameter in enumerate(
                PROPERTIES
            ):
                item = rasters[
                    (
                        geometry,
                        modality,
                        parameter,
                    )
                ]

                artist = draw_raster(
                    axes[0, column],
                    item,
                    campaign,
                    roi=roi,
                    cmap=cmap,
                    title=(
                        PROPERTY_LABELS[
                            parameter
                        ]
                    ),
                    show_xlabel=True,
                    show_ylabel=(
                        column == 0
                    ),
                    visible_arrays=MODALITY_ARRAY_VISIBILITY[
                        modality
                    ],
                )

            if artist is None:
                raise AssertionError(
                    "No artist was created for a modality board."
                )

            fig.suptitle(
                f"{geometry.upper()} — "
                f"{MODALITY_LABELS[modality]} — "
                "five normalized sensitivity maps"
            )

            colorbar = fig.colorbar(
                artist,
                ax=list(
                    axes.flat
                ),
                shrink=0.82,
                pad=0.02,
            )

            colorbar.set_label(
                colorbar_label
            )

            relative_path = (
                Path("modality_boards")
                / (
                    f"{sanitize_identifier(geometry)}"
                    f"__{sanitize_identifier(modality)}"
                    "__five_properties.png"
                )
            )

            output_path = (
                output_dir
                / relative_path
            )

            save_png(
                fig,
                output_path,
                dpi=config.dpi,
            )

            plt.close(fig)

            generated_pngs.append(
                output_path
            )

            modality_board_paths[
                (
                    geometry,
                    modality,
                )
            ] = relative_path

    # --------------------------------------------------------------
    # Two boards: four modalities by five properties
    # --------------------------------------------------------------

    for geometry in GEOMETRIES:
        campaign = campaigns_by_geometry[
            geometry
        ]

        fig, axes = plt.subplots(
            len(MODALITIES),
            len(PROPERTIES),
            figsize=(16.0, 11.8),
            squeeze=False,
            constrained_layout=True,
        )

        artist = None

        for row, modality in enumerate(
            MODALITIES
        ):
            for column, parameter in enumerate(
                PROPERTIES
            ):
                item = rasters[
                    (
                        geometry,
                        modality,
                        parameter,
                    )
                ]

                artist = draw_raster(
                    axes[row, column],
                    item,
                    campaign,
                    roi=roi,
                    cmap=cmap,
                    title=(
                        PROPERTY_LABELS[
                            parameter
                        ]
                        if row == 0
                        else None
                    ),
                    show_xlabel=(
                        row
                        == len(MODALITIES) - 1
                    ),
                    show_ylabel=(
                        column == 0
                    ),
                    visible_arrays=MODALITY_ARRAY_VISIBILITY[
                        modality
                    ],
                )

                # Redundant per-column "x" axis title, stripped from every
                # row (including the bottom one) while the coordinate tick
                # numbers remain visible.
                axes[row, column].set_xlabel("")

                if column == 0:
                    axes[
                        row,
                        column,
                    ].set_ylabel(
                        MODALITY_LABELS[
                            modality
                        ],
                        fontsize=12,
                        fontweight="semibold",
                        labelpad=7,
                        color=MODALITY_COLORS[
                            modality
                        ],
                    )

        if artist is None:
            raise AssertionError(
                "No artist was created for a geometry board."
            )

        colorbar = fig.colorbar(
            artist,
            ax=list(
                axes.flat
            ),
            shrink=0.82,
            pad=0.02,
        )

        colorbar.set_label(
            colorbar_label
        )

        relative_path = (
            Path("geometry_boards")
            / (
                f"{sanitize_identifier(geometry)}"
                "__four_modalities"
                "__five_properties.png"
            )
        )

        output_path = (
            output_dir
            / relative_path
        )

        save_png(
            fig,
            output_path,
            dpi=config.dpi,
        )

        # Native PDF export alongside the PNG: avoids the lossy JPEG
        # re-encoding introduced by an external PNG-to-PDF conversion
        # (visible as an 8x8 block grid artefact over coloured regions).
        pdf_output_path = output_path.with_suffix(".pdf")

        fig.savefig(
            pdf_output_path,
            dpi=config.dpi,
            facecolor="white",
            bbox_inches=None,
        )

        plt.close(fig)

        generated_pngs.append(
            output_path
        )

        geometry_board_paths[
            geometry
        ] = relative_path

    if len(generated_pngs) != 50:
        raise AssertionError(
            f"Expected 50 PNG outputs, found {len(generated_pngs)}."
        )

    for row in manifest_rows:
        geometry = str(
            row["canonical_geometry"]
        )

        modality = str(
            row["modality"]
        )

        row[
            "modality_board_png"
        ] = str(
            modality_board_paths[
                (
                    geometry,
                    modality,
                )
            ]
        )

        row[
            "geometry_board_png"
        ] = str(
            geometry_board_paths[
                geometry
            ]
        )

    return (
        manifest_rows,
        generated_pngs,
    )


def main(
    argv: list[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.dpi <= 0:
            raise ValueError(
                "--dpi must be positive."
            )

        if args.map_grid_size < 32:
            raise ValueError(
                "--map-grid-size must be at least 32."
            )

        output_dir = (
            args.output_dir
            .expanduser()
            .resolve()
        )

        prepare_output_dir(
            output_dir,
            overwrite=bool(
                args.overwrite
            ),
        )

        inventory = load_inventory(
            args.input_inventory
            .expanduser()
            .resolve()
        )

        config = ArticleFigureConfig(
            input_root=Path.cwd(),
            output_dir=output_dir,
            campaign_dirs=tuple(
                path.expanduser().resolve()
                for path in args.campaign_dir
            ),
            figure_groups=(),
            roi_spec=str(args.roi),
            formats=("png",),
            dpi=int(args.dpi),
            overwrite=False,
            quiet=bool(args.quiet),
            strict=True,
            modalities=MODALITIES,
            parameters=PROPERTIES,
            signed_modalities=(),
            field_rendering="rasterized",
            max_vector_points=int(
                args.max_vector_points
            ),
            map_grid_size=int(
                args.map_grid_size
            ),
        )

        campaigns = discover_campaigns(
            config
        )

        (
            campaigns_by_geometry,
            source_records,
        ) = discover_source_records(
            campaigns,
            inventory,
        )

        roi = parse_roi(
            str(args.roi),
            campaigns,
        )

        if args.validate_only:
            payload = {
                "status": "ok",
                "analysis_date_utc": (
                    now_utc()
                ),
                "kind": (
                    "normalized_sensitivity_"
                    "map_gallery_preflight"
                ),
                "num_campaigns": len(
                    campaigns
                ),
                "num_maps": len(
                    source_records
                ),
                "geometries": list(
                    GEOMETRIES
                ),
                "modalities": list(
                    MODALITIES
                ),
                "properties": list(
                    PROPERTIES
                ),
                "common_roi": list(
                    roi
                ),
                "validation_status": (
                    "GALLERY_PREFLIGHT_OK"
                ),
            }

            preflight_path = (
                output_dir
                / "P10_GALLERY_PREFLIGHT.json"
            )

            preflight_path.write_text(
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            print(
                "[GALLERY] "
                "validate-only status=ok "
                f"campaigns={len(campaigns)} "
                f"maps={len(source_records)} "
                f"roi={roi}"
            )

            return 0

        (
            manifest_rows,
            generated_pngs,
        ) = build_gallery(
            output_dir=output_dir,
            campaigns_by_geometry=(
                campaigns_by_geometry
            ),
            source_records=source_records,
            roi=roi,
            config=config,
            quiet=bool(args.quiet),
        )

        manifest = pd.DataFrame(
            manifest_rows
        ).sort_values(
            [
                "canonical_geometry",
                "modality",
                "parameter_name",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        )

        if len(manifest) != 40:
            raise AssertionError(
                "The gallery manifest must contain 40 rows."
            )

        individual_dimensions = {
            (
                int(row.individual_width_px),
                int(row.individual_height_px),
            )
            for row in manifest.itertuples(
                index=False
            )
        }

        if len(
            individual_dimensions
        ) != 1:
            raise AssertionError(
                "Individual PNG dimensions are not identical: "
                f"{sorted(individual_dimensions)}"
            )

        manifest_path = (
            output_dir
            / "P10_GALLERY_MANIFEST.csv"
        )

        manifest.to_csv(
            manifest_path,
            index=False,
        )

        all_pngs = sorted(
            output_dir.rglob(
                "*.png"
            )
        )

        if len(all_pngs) != 50:
            raise AssertionError(
                f"Expected 50 PNG files, found {len(all_pngs)}."
            )

        individual_pngs = sorted(
            (
                output_dir
                / "individual"
            ).rglob(
                "*.png"
            )
        )

        modality_boards = sorted(
            (
                output_dir
                / "modality_boards"
            ).glob(
                "*.png"
            )
        )

        geometry_boards = sorted(
            (
                output_dir
                / "geometry_boards"
            ).glob(
                "*.png"
            )
        )

        if len(
            individual_pngs
        ) != 40:
            raise AssertionError(
                "Expected exactly 40 individual PNG maps."
            )

        if len(
            modality_boards
        ) != 8:
            raise AssertionError(
                "Expected exactly 8 modality-board PNG files."
            )

        if len(
            geometry_boards
        ) != 2:
            raise AssertionError(
                "Expected exactly 2 geometry-board PNG files."
            )

        for path in all_pngs:
            image_size(path)

        width, height = next(
            iter(
                individual_dimensions
            )
        )

        summary = {
            "status": "ok",
            "analysis_date_utc": (
                now_utc()
            ),
            "kind": (
                "normalized_sensitivity_"
                "map_gallery"
            ),
            "format_version": (
                "normalized_sensitivity_"
                "map_gallery_v1"
            ),
            "input_representation": (
                "article_safe_normalized_"
                "qualitative_sensitivity"
            ),
            "normalization": (
                "absolute field divided by "
                "the maximum absolute value "
                "inside the common displayed ROI"
            ),
            "common_color_range": [
                0.0,
                1.0,
            ],
            "common_roi": list(
                roi
            ),
            "map_grid_size": int(
                args.map_grid_size
            ),
            "dpi": int(
                args.dpi
            ),
            "geometries": list(
                GEOMETRIES
            ),
            "modalities": list(
                MODALITIES
            ),
            "properties": list(
                PROPERTIES
            ),
            "counts": {
                "source_maps": 40,
                "individual_png": 40,
                "modality_board_png": 8,
                "geometry_board_png": 2,
                "total_png": 50,
                "manifest_rows": 40,
            },
            "individual_image_dimensions_px": {
                "width": width,
                "height": height,
            },
            "outputs": {
                "manifest_csv": str(
                    manifest_path
                ),
                "individual_root": str(
                    output_dir
                    / "individual"
                ),
                "modality_board_root": str(
                    output_dir
                    / "modality_boards"
                ),
                "geometry_board_root": str(
                    output_dir
                    / "geometry_boards"
                ),
            },
            "validation": {
                "complete_cartesian_product": True,
                "unique_source_paths": (
                    manifest[
                        "source_path"
                    ].nunique()
                    == 40
                ),
                "unique_individual_png": (
                    manifest[
                        "individual_png"
                    ].nunique()
                    == 40
                ),
                "identical_individual_dimensions": True,
                "all_png_readable": True,
                "no_raw_signed_source": (
                    not manifest[
                        "source_path"
                    ]
                    .astype(str)
                    .str.contains(
                        "raw_signed",
                        case=False,
                        regex=False,
                    )
                    .any()
                ),
            },
            "final_status": (
                "NORMALIZED_MAP_"
                "GALLERY_COMPLETED"
            ),
        }

        summary_path = (
            output_dir
            / "P10_GALLERY_SUMMARY.json"
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "[GALLERY] status=ok "
            "sources=40 "
            "individual_png=40 "
            "modality_boards=8 "
            "geometry_boards=2 "
            "total_png=50"
        )

        print(
            f"[GALLERY] manifest={manifest_path}"
        )

        print(
            f"[GALLERY] summary={summary_path}"
        )

        return 0

    except Exception as exc:
        print(
            "[GALLERY] FAIL: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
