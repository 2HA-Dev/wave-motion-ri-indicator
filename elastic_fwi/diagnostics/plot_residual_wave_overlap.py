from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from elastic_fwi.postprocessing.residual_wave_overlap import (
    DefectGeometry,
    analyze_overlap_snapshot,
    canonical_interaction_mode,
    compute_overlap_maps,
    extract_physical_xy_components,
)
from elastic_fwi.postprocessing.residual_wave_overlap_selection import (
    audit_activity_thresholds,
    select_temporal_overlap_states,
)
from elastic_fwi.postprocessing.vtk_residual_wave_io import (
    assemble_residual_wave_pair,
    build_residual_wave_assembly_plan,
    load_residual_wave_pair_at_time,
    read_pvd_series,
    read_raw_residual_wave_snapshot,
    validate_pvd_pair_times,
)


REFLECTION_FIELD_NAME = (
    "reflection_residual_wavefield"
)

TRANSMISSION_FIELD_NAME = (
    "transmission_residual_wavefield"
)

OUTPUT_FILENAMES = {
    "global_triptych_png": (
        "residual_wave_overlap_global_triptych.png"
    ),
    "global_triptych_pdf": (
        "residual_wave_overlap_global_triptych.pdf"
    ),
    "global_comparison_png": (
        "residual_wave_overlap_global_interactions.png"
    ),
    "global_comparison_pdf": (
        "residual_wave_overlap_global_interactions.pdf"
    ),
    "localization_triptych_png": (
        "residual_wave_overlap_localization_triptych.png"
    ),
    "localization_triptych_pdf": (
        "residual_wave_overlap_localization_triptych.pdf"
    ),
    "localization_comparison_png": (
        "residual_wave_overlap_localization_interactions.png"
    ),
    "localization_comparison_pdf": (
        "residual_wave_overlap_localization_interactions.pdf"
    ),
    "temporal_diagnostics_png": (
        "residual_wave_overlap_temporal_diagnostics.png"
    ),
    "temporal_diagnostics_pdf": (
        "residual_wave_overlap_temporal_diagnostics.pdf"
    ),
    "metrics_csv": (
        "residual_wave_overlap_metrics.csv"
    ),
    "metrics_json": (
        "residual_wave_overlap_metrics.json"
    ),
    "manifest_json": (
        "residual_wave_overlap_manifest.json"
    ),

    # Legacy outputs are included only so that --overwrite
    # removes obsolete single-selection figures.
    "legacy_triptych_png": (
        "residual_wave_overlap_triptych.png"
    ),
    "legacy_triptych_pdf": (
        "residual_wave_overlap_triptych.pdf"
    ),
    "legacy_localization_png": (
        "residual_wave_overlap_localization.png"
    ),
    "legacy_localization_pdf": (
        "residual_wave_overlap_localization.pdf"
    ),
}


CSV_COLUMNS = [
    "case_id",
    "analysis_mode",
    "geometry_kind",
    "time_index",
    "time_value",
    "interaction_definition",
    "metric_mode",
    "normalization_mode",
    "defect_mask_source",
    "defect_center_x",
    "defect_center_y",
    "defect_radius",
    "ring_width",

    "selected_for_figure",
    "selected_for_overlap_figure",
    "selected_for_localization_figure",

    "global_max",
    "global_max_x",
    "global_max_y",
    "global_max_inside_defect",
    "global_max_inside_ring",
    "distance_peak_to_defect_center",
    "weighted_centroid_x",
    "weighted_centroid_y",
    "distance_centroid_to_defect_center",

    "area_total",
    "area_defect",
    "area_ring",
    "area_outside",
    "area_defect_and_ring",
    "area_fraction_defect",
    "area_fraction_defect_and_ring",

    "energy_total",
    "energy_defect",
    "energy_ring",
    "energy_outside",
    "energy_defect_and_ring",
    "energy_fraction_defect",
    "energy_fraction_defect_and_ring",
    "relative_total_energy",

    "mean_interaction_density_global",
    "mean_interaction_density_defect",
    "mean_interaction_density_defect_and_ring",
    "mean_interaction_density_outside",

    "defect_energy_enrichment",
    "defect_ring_energy_enrichment",
    "defect_to_outside_density_ratio",
    "defect_ring_to_outside_density_ratio",

    "energy_inside_outside_ratio",
    "peak_defect",
    "peak_ring",
    "peak_outside",
    "peak_to_artifact_ratio",
    "peak_outside_region",
    "peak_defined",
    "ratio_epsilon",
    "energy_denominator_floor",
    "peak_denominator_floor",

    "reflection_duplicate_max_discrepancy",
    "transmission_duplicate_max_discrepancy",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None

    value = completed.stdout.strip()

    return value or None


def _check_sequential_execution() -> None:
    try:
        from mpi4py import MPI
    except ImportError:
        return

    size = int(MPI.COMM_WORLD.size)

    if size != 1:
        raise RuntimeError(
            "This PVD/PVTU post-processing command must be "
            "run sequentially. The VTK parallel reader already "
            "assembles the distributed pieces."
        )


def _prepare_output_directory(
    output_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, Path]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        key: output_dir / filename
        for key, filename
        in OUTPUT_FILENAMES.items()
    }

    existing = [
        path
        for path in paths.values()
        if path.exists()
    ]

    if existing and not overwrite:
        formatted = "\n".join(
            f"  - {path}"
            for path in existing
        )

        raise FileExistsError(
            "Output files already exist. Use --overwrite "
            "to replace them:\n"
            f"{formatted}"
        )

    if overwrite:
        for path in existing:
            path.unlink()

    return paths


def _selected_snapshot_indices(
    times: np.ndarray,
    *,
    start_time: float | None,
    end_time: float | None,
    single_time: float | None,
) -> list[int]:
    if single_time is not None:
        matches = np.flatnonzero(
            np.isclose(
                times,
                float(single_time),
                rtol=0.0,
                atol=1.0e-12,
            )
        )

        if matches.size != 1:
            raise ValueError(
                f"Expected one snapshot at t={single_time}, "
                f"found {matches.size}."
            )

        return [int(matches[0])]

    mask = np.ones(
        times.size,
        dtype=bool,
    )

    if start_time is not None:
        mask &= (
            times
            >= float(start_time) - 1.0e-12
        )

    if end_time is not None:
        mask &= (
            times
            <= float(end_time) + 1.0e-12
        )

    indices = np.flatnonzero(mask)

    if indices.size == 0:
        raise ValueError(
            "No PVD snapshot lies in the requested "
            "time interval."
        )

    return [
        int(index)
        for index in indices
    ]


def _metric_row(
    *,
    case_id: str,
    time_index: int,
    time_value: float,
    geometry: DefectGeometry,
    metrics: dict[str, Any],
    reflection_discrepancy: float,
    transmission_discrepancy: float,
) -> dict[str, Any]:
    row = {
        "case_id": str(case_id),
        "time_index": int(time_index),
        "time_value": float(time_value),
        "interaction_definition": (
            metrics["interaction_definition"]
        ),
        "metric_mode": metrics["metric_mode"],
        "normalization_mode": (
            metrics["normalization_mode"]
        ),
        "defect_mask_source": (
            "analytical_disk_from_campaign_metadata"
        ),
        "defect_center_x": float(
            geometry.center_x
        ),
        "defect_center_y": float(
            geometry.center_y
        ),
        "defect_radius": float(
            geometry.radius
        ),
        "ring_width": float(
            geometry.ring_width
        ),
        "selected_for_figure": False,
        "reflection_duplicate_max_discrepancy": (
            float(reflection_discrepancy)
        ),
        "transmission_duplicate_max_discrepancy": (
            float(transmission_discrepancy)
        ),
    }

    row.update(metrics)

    return row


def _scan_campaign(
    *,
    reflection_pvd: Path,
    transmission_pvd: Path,
    case_id: str,
    geometry: DefectGeometry,
    start_time: float | None,
    end_time: float | None,
    single_time: float | None,
    select_by: str,
    quiet: bool,
) -> dict[str, Any]:
    reflection_series = read_pvd_series(
        reflection_pvd
    )

    transmission_series = read_pvd_series(
        transmission_pvd
    )

    validate_pvd_pair_times(
        reflection_series,
        transmission_series,
    )

    times = reflection_series.times

    indices = _selected_snapshot_indices(
        times,
        start_time=start_time,
        end_time=end_time,
        single_time=single_time,
    )

    first_index = indices[0]

    first_reflection = (
        read_raw_residual_wave_snapshot(
            reflection_series.snapshots[
                first_index
            ].path,
            field_name=REFLECTION_FIELD_NAME,
        )
    )

    first_transmission = (
        read_raw_residual_wave_snapshot(
            transmission_series.snapshots[
                first_index
            ].path,
            field_name=TRANSMISSION_FIELD_NAME,
        )
    )

    plan = build_residual_wave_assembly_plan(
        first_reflection,
        first_transmission,
    )

    rows: list[dict[str, Any]] = []

    selected_index: int | None = None
    selected_time: float | None = None
    selected_score = -np.inf

    maximum_reflection_discrepancy = 0.0
    maximum_transmission_discrepancy = 0.0

    for position, index in enumerate(
        indices,
        start=1,
    ):
        reflection_snapshot = (
            reflection_series.snapshots[index]
        )

        transmission_snapshot = (
            transmission_series.snapshots[
                index
            ]
        )

        if index == first_index:
            reflection_raw = first_reflection
            transmission_raw = first_transmission
        else:
            reflection_raw = (
                read_raw_residual_wave_snapshot(
                    reflection_snapshot.path,
                    field_name=(
                        REFLECTION_FIELD_NAME
                    ),
                )
            )

            transmission_raw = (
                read_raw_residual_wave_snapshot(
                    transmission_snapshot.path,
                    field_name=(
                        TRANSMISSION_FIELD_NAME
                    ),
                )
            )

        pair = assemble_residual_wave_pair(
            plan,
            reflection_raw,
            transmission_raw,
        )

        maximum_reflection_discrepancy = max(
            maximum_reflection_discrepancy,
            pair.reflection_duplicate_max_discrepancy,
        )

        maximum_transmission_discrepancy = max(
            maximum_transmission_discrepancy,
            pair.transmission_duplicate_max_discrepancy,
        )

        absolute_result = (
            analyze_overlap_snapshot(
                points=pair.points,
                triangles=pair.triangles,
                reflection_field=(
                    pair.reflection_field
                ),
                transmission_field=(
                    pair.transmission_field
                ),
                geometry=geometry,
                metric_mode="absolute_dot",
            )
        )

        absolute_metrics = (
            absolute_result.metrics.as_dict()
        )

        rows.append(
            _metric_row(
                case_id=case_id,
                time_index=index,
                time_value=(
                    reflection_snapshot.time
                ),
                geometry=geometry,
                metrics=absolute_metrics,
                reflection_discrepancy=(
                    pair.reflection_duplicate_max_discrepancy
                ),
                transmission_discrepancy=(
                    pair.transmission_duplicate_max_discrepancy
                ),
            )
        )

        if select_by == "absolute_dot":
            candidate_score = float(
                absolute_metrics["energy_total"]
            )
        else:
            candidate_score = -np.inf

        del absolute_result
        del absolute_metrics
        gc.collect()

        energy_result = (
            analyze_overlap_snapshot(
                points=pair.points,
                triangles=pair.triangles,
                reflection_field=(
                    pair.reflection_field
                ),
                transmission_field=(
                    pair.transmission_field
                ),
                geometry=geometry,
                metric_mode="norm_product",
            )
        )

        energy_metrics = (
            energy_result.metrics.as_dict()
        )

        rows.append(
            _metric_row(
                case_id=case_id,
                time_index=index,
                time_value=(
                    reflection_snapshot.time
                ),
                geometry=geometry,
                metrics=energy_metrics,
                reflection_discrepancy=(
                    pair.reflection_duplicate_max_discrepancy
                ),
                transmission_discrepancy=(
                    pair.transmission_duplicate_max_discrepancy
                ),
            )
        )

        if select_by == "norm_product":
            candidate_score = float(
                energy_metrics["energy_total"]
            )

        if candidate_score > selected_score:
            selected_score = candidate_score
            selected_index = int(index)
            selected_time = float(
                reflection_snapshot.time
            )

        if not quiet:
            print(
                "[RESIDUAL-WAVE OVERLAP SCAN]",
                f"{position}/{len(indices)}",
                f"index={index}",
                f"time={reflection_snapshot.time:g}",
                "absolute_energy="
                f"{rows[-2]['energy_total']:.16e}",
                "norm_product_energy="
                f"{rows[-1]['energy_total']:.16e}",
            )

        del energy_result
        del energy_metrics
        del pair

        if index != first_index:
            del reflection_raw
            del transmission_raw

        gc.collect()

    if (
        selected_index is None
        or selected_time is None
        or not np.isfinite(selected_score)
    ):
        raise RuntimeError(
            "Unable to select a representative snapshot."
        )

    for row in rows:
        row["selected_for_figure"] = bool(
            int(row["time_index"])
            == selected_index
        )

    return {
        "rows": rows,
        "selected_index": selected_index,
        "selected_time": selected_time,
        "selected_score": float(
            selected_score
        ),
        "select_by": str(select_by),
        "processed_indices": indices,
        "processed_times": [
            float(times[index])
            for index in indices
        ],
        "series": {
            "reflection": (
                reflection_series.as_dict()
            ),
            "transmission": (
                transmission_series.as_dict()
            ),
        },
        "assembly_plan": plan.as_dict(),
        "maximum_reflection_duplicate_discrepancy": (
            maximum_reflection_discrepancy
        ),
        "maximum_transmission_duplicate_discrepancy": (
            maximum_transmission_discrepancy
        ),
    }


def _fill_missing_pixels(
    grid: np.ndarray,
    *,
    maximum_iterations: int | None = None,
) -> np.ndarray:
    """Fill missing raster pixels without periodic edge wrapping.

    Missing pixels are propagated from their finite four-neighbour
    values. The update is simultaneous at each iteration and does
    not connect opposite image boundaries.
    """
    result = np.asarray(
        grid,
        dtype=np.float64,
    ).copy()

    if result.ndim != 2:
        raise ValueError(
            "Raster grid must be two-dimensional."
        )

    finite = np.isfinite(result)

    if not np.any(finite):
        raise ValueError(
            "Raster grid contains no finite seed value."
        )

    if np.all(finite):
        return result

    if maximum_iterations is None:
        # The largest Manhattan distance between two pixels is
        # strictly smaller than n_rows + n_columns.
        maximum_iterations = (
            result.shape[0]
            + result.shape[1]
        )

    if maximum_iterations <= 0:
        raise ValueError(
            "maximum_iterations must be strictly positive."
        )

    for _ in range(maximum_iterations):
        missing = ~np.isfinite(result)

        if not np.any(missing):
            return result

        total = np.zeros_like(result)

        count = np.zeros(
            result.shape,
            dtype=np.int16,
        )

        # Neighbour above contributes to the pixel below.
        source = result[:-1, :]
        valid = np.isfinite(source)

        target_total = total[1:, :]
        target_count = count[1:, :]

        target_total[valid] += source[valid]
        target_count[valid] += 1

        # Neighbour below contributes to the pixel above.
        source = result[1:, :]
        valid = np.isfinite(source)

        target_total = total[:-1, :]
        target_count = count[:-1, :]

        target_total[valid] += source[valid]
        target_count[valid] += 1

        # Neighbour on the left contributes to the pixel on the right.
        source = result[:, :-1]
        valid = np.isfinite(source)

        target_total = total[:, 1:]
        target_count = count[:, 1:]

        target_total[valid] += source[valid]
        target_count[valid] += 1

        # Neighbour on the right contributes to the pixel on the left.
        source = result[:, 1:]
        valid = np.isfinite(source)

        target_total = total[:, :-1]
        target_count = count[:, :-1]

        target_total[valid] += source[valid]
        target_count[valid] += 1

        fill = (
            missing
            & (count > 0)
        )

        if not np.any(fill):
            break

        result[fill] = (
            total[fill]
            / count[fill]
        )

    if np.any(~np.isfinite(result)):
        remaining = int(
            np.count_nonzero(
                ~np.isfinite(result)
            )
        )

        raise RuntimeError(
            "Raster filling did not converge: "
            f"{remaining} pixels remain non-finite."
        )

    return result


def rasterize_nodal_values(

    points: np.ndarray,
    values: np.ndarray,
    *,
    resolution: int,
    signed: bool,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    coordinates = np.asarray(
        points,
        dtype=np.float64,
    )

    data = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if coordinates.ndim != 2:
        raise ValueError(
            "points must be two-dimensional."
        )

    if coordinates.shape[0] != data.size:
        raise ValueError(
            "points and values have incompatible sizes."
        )

    if resolution < 100:
        raise ValueError(
            "raster resolution must be at least 100."
        )

    if not np.all(np.isfinite(coordinates[:, :2])):
        raise ValueError(
            "Non-finite visualization coordinates."
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "Non-finite visualization values."
        )

    xmin = float(
        np.min(coordinates[:, 0])
    )

    xmax = float(
        np.max(coordinates[:, 0])
    )

    ymin = float(
        np.min(coordinates[:, 1])
    )

    ymax = float(
        np.max(coordinates[:, 1])
    )

    if xmax <= xmin or ymax <= ymin:
        raise ValueError(
            "Invalid visualization extent."
        )

    x_index = np.floor(
        (
            coordinates[:, 0] - xmin
        )
        / (xmax - xmin)
        * (resolution - 1)
    ).astype(np.int64)

    y_index = np.floor(
        (
            coordinates[:, 1] - ymin
        )
        / (ymax - ymin)
        * (resolution - 1)
    ).astype(np.int64)

    np.clip(
        x_index,
        0,
        resolution - 1,
        out=x_index,
    )

    np.clip(
        y_index,
        0,
        resolution - 1,
        out=y_index,
    )

    flat_index = (
        y_index * resolution
        + x_index
    )

    total_pixels = resolution * resolution

    if signed:
        sums = np.bincount(
            flat_index,
            weights=data,
            minlength=total_pixels,
        )

        counts = np.bincount(
            flat_index,
            minlength=total_pixels,
        )

        flat = np.full(
            total_pixels,
            np.nan,
            dtype=np.float64,
        )

        occupied = counts > 0

        flat[occupied] = (
            sums[occupied]
            / counts[occupied]
        )
    else:
        flat = np.full(
            total_pixels,
            -np.inf,
            dtype=np.float64,
        )

        np.maximum.at(
            flat,
            flat_index,
            data,
        )

        flat[~np.isfinite(flat)] = np.nan

    grid = flat.reshape(
        resolution,
        resolution,
    )

    grid = _fill_missing_pixels(grid)

    return (
        grid,
        (xmin, xmax, ymin, ymax),
    )


def _normalized_nonnegative(
    values: np.ndarray,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    maximum = float(
        np.max(array)
    )

    if maximum <= 0.0:
        return np.zeros_like(array)

    return array / maximum


def _add_geometry_overlay(
    ax,
    geometry: DefectGeometry,
) -> None:
    defect = Circle(
        (
            geometry.center_x,
            geometry.center_y,
        ),
        geometry.radius,
        fill=False,
        linewidth=1.2,
        label="defect",
    )

    ring = Circle(
        (
            geometry.center_x,
            geometry.center_y,
        ),
        geometry.radius
        + geometry.ring_width,
        fill=False,
        linewidth=1.0,
        linestyle="--",
        label="defect + ring",
    )

    ax.add_patch(defect)
    ax.add_patch(ring)


def _plot_grid(
    ax,
    grid: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
    title: str,
    signed: bool,
):
    finite = np.isfinite(grid)

    if not np.any(finite):
        raise RuntimeError(
            f"Visualization grid for {title!r} is empty."
        )

    if signed:
        scale = float(
            np.max(np.abs(grid[finite]))
        )

        if scale <= 0.0:
            scale = 1.0

        image = ax.imshow(
            grid,
            origin="lower",
            extent=extent,
            aspect="equal",
            interpolation="bilinear",
            vmin=-scale,
            vmax=scale,
            cmap="coolwarm",
        )
    else:
        image = ax.imshow(
            grid,
            origin="lower",
            extent=extent,
            aspect="equal",
            interpolation="bilinear",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return image


def _produce_figures(
    *,
    reflection_pvd: Path,
    transmission_pvd: Path,
    selected_time: float,
    geometry: DefectGeometry,
    interaction_mode: str,
    raster_resolution: int,
    figure_dpi: int,
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    loaded = load_residual_wave_pair_at_time(
        reflection_pvd=reflection_pvd,
        transmission_pvd=transmission_pvd,
        requested_time=selected_time,
    )

    reflection_xy, transmission_xy = (
        extract_physical_xy_components(
            loaded.pair.reflection_field,
            loaded.pair.transmission_field,
        )
    )

    reflection_norm = np.sqrt(
        np.sum(
            reflection_xy * reflection_xy,
            axis=1,
        )
    )

    transmission_norm = np.sqrt(
        np.sum(
            transmission_xy * transmission_xy,
            axis=1,
        )
    )

    maps = compute_overlap_maps(
        loaded.pair.reflection_field,
        loaded.pair.transmission_field,
    )

    canonical_mode = (
        canonical_interaction_mode(
            interaction_mode
        )
    )

    selected_metric_mode = (
        canonical_mode
        if canonical_mode != "signed_dot"
        else "absolute_dot"
    )

    selected_analysis = (
        analyze_overlap_snapshot(
            points=loaded.pair.points,
            triangles=loaded.pair.triangles,
            reflection_field=(
                loaded.pair.reflection_field
            ),
            transmission_field=(
                loaded.pair.transmission_field
            ),
            geometry=geometry,
            metric_mode=selected_metric_mode,
        )
    )

    normalized_reflection = (
        _normalized_nonnegative(
            reflection_norm
        )
    )

    normalized_transmission = (
        _normalized_nonnegative(
            transmission_norm
        )
    )

    selected_normalized = (
        maps.get_normalized(
            canonical_mode
        )
    )

    reflection_grid, extent = (
        rasterize_nodal_values(
            loaded.pair.points,
            normalized_reflection,
            resolution=raster_resolution,
            signed=False,
        )
    )

    transmission_grid, extent_2 = (
        rasterize_nodal_values(
            loaded.pair.points,
            normalized_transmission,
            resolution=raster_resolution,
            signed=False,
        )
    )

    if extent_2 != extent:
        raise RuntimeError(
            "Inconsistent visualization extents."
        )

    interaction_grid, extent_3 = (
        rasterize_nodal_values(
            loaded.pair.points,
            selected_normalized,
            resolution=raster_resolution,
            signed=(
                canonical_mode
                == "signed_dot"
            ),
        )
    )

    if extent_3 != extent:
        raise RuntimeError(
            "Inconsistent interaction extent."
        )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.8),
        constrained_layout=True,
    )

    images = [
        _plot_grid(
            axes[0],
            reflection_grid,
            extent=extent,
            title=(
                r"$\|\mathbf{W}_1\|/"
                r"\max\|\mathbf{W}_1\|$"
            ),
            signed=False,
        ),
        _plot_grid(
            axes[1],
            transmission_grid,
            extent=extent,
            title=(
                r"$\|\mathbf{\Lambda}_2\|/"
                r"\max\|\mathbf{\Lambda}_2\|$"
            ),
            signed=False,
        ),
        _plot_grid(
            axes[2],
            interaction_grid,
            extent=extent,
            title=(
                f"{canonical_mode} normalized"
            ),
            signed=(
                canonical_mode
                == "signed_dot"
            ),
        ),
    ]

    for axis, image in zip(
        axes,
        images,
        strict=True,
    ):
        _add_geometry_overlay(
            axis,
            geometry,
        )

        fig.colorbar(
            image,
            ax=axis,
            shrink=0.82,
        )

    axes[2].scatter(
        [
            selected_analysis.metrics.global_max_x
        ],
        [
            selected_analysis.metrics.global_max_y
        ],
        marker="x",
        label="global maximum",
    )

    axes[2].legend(
        loc="upper right",
        fontsize="small",
    )

    fig.suptitle(
        "Residual-wave spatial overlap "
        f"at t={selected_time:g}"
    )

    fig.savefig(
        output_paths["triptych_png"],
        dpi=figure_dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        output_paths["triptych_pdf"],
        dpi=figure_dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    signed_grid, signed_extent = (
        rasterize_nodal_values(
            loaded.pair.points,
            maps.normalized_signed_dot,
            resolution=raster_resolution,
            signed=True,
        )
    )

    absolute_grid, absolute_extent = (
        rasterize_nodal_values(
            loaded.pair.points,
            maps.normalized_absolute_dot,
            resolution=raster_resolution,
            signed=False,
        )
    )

    energy_grid, energy_extent = (
        rasterize_nodal_values(
            loaded.pair.points,
            maps.normalized_norm_product,
            resolution=raster_resolution,
            signed=False,
        )
    )

    if not (
        signed_extent
        == absolute_extent
        == energy_extent
        == extent
    ):
        raise RuntimeError(
            "Localization visualization extents differ."
        )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.8),
        constrained_layout=True,
    )

    localization_images = [
        _plot_grid(
            axes[0],
            signed_grid,
            extent=extent,
            title=(
                r"$\mathbf{W}_1\cdot"
                r"\mathbf{\Lambda}_2$"
                " normalized"
            ),
            signed=True,
        ),
        _plot_grid(
            axes[1],
            absolute_grid,
            extent=extent,
            title=(
                r"$|\mathbf{W}_1\cdot"
                r"\mathbf{\Lambda}_2|$"
                " normalized"
            ),
            signed=False,
        ),
        _plot_grid(
            axes[2],
            energy_grid,
            extent=extent,
            title=(
                r"$\|\mathbf{W}_1\|"
                r"\|\mathbf{\Lambda}_2\|$"
                " normalized"
            ),
            signed=False,
        ),
    ]

    for axis, image in zip(
        axes,
        localization_images,
        strict=True,
    ):
        _add_geometry_overlay(
            axis,
            geometry,
        )

        fig.colorbar(
            image,
            ax=axis,
            shrink=0.82,
        )

    marker_axis = (
        axes[1]
        if selected_metric_mode
        == "absolute_dot"
        else axes[2]
    )

    marker_axis.scatter(
        [
            selected_analysis.metrics.global_max_x
        ],
        [
            selected_analysis.metrics.global_max_y
        ],
        marker="x",
        label="global maximum",
    )

    marker_axis.scatter(
        [
            selected_analysis.metrics.weighted_centroid_x
        ],
        [
            selected_analysis.metrics.weighted_centroid_y
        ],
        marker="+",
        label="weighted centroid",
    )

    marker_axis.legend(
        loc="upper right",
        fontsize="small",
    )

    fig.suptitle(
        "Signed, absolute and energetic "
        f"residual-wave overlap at t={selected_time:g}"
    )

    fig.savefig(
        output_paths["localization_png"],
        dpi=figure_dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        output_paths["localization_pdf"],
        dpi=figure_dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "time_index": int(
            loaded.time_index
        ),
        "time_value": float(
            loaded.time_value
        ),
        "interaction_mode": canonical_mode,
        "metric_mode_for_markers": (
            selected_metric_mode
        ),
        "raster_resolution": int(
            raster_resolution
        ),
        "visualization_rasterization": {
            "nonnegative_fields": (
                "maximum_nodal_value_per_pixel"
            ),
            "signed_fields": (
                "mean_nodal_value_per_pixel"
            ),
            "missing_pixel_fill": (
                "iterative_four-neighbour_average"
            ),
            "physical_metrics_use_raster": False,
        },
        "selected_metrics": (
            selected_analysis.metrics.as_dict()
        ),
        "loaded_pair": loaded.as_dict(),
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column: row.get(column)
                    for column in CSV_COLUMNS
                }
            )


def _validate_metric_rows(
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError(
            "No metric rows were produced."
        )

    for row_index, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, float):
                if not np.isfinite(value):
                    raise RuntimeError(
                        "Non-finite metric value: "
                        f"row={row_index}, "
                        f"key={key}, value={value}."
                    )

        for key in (
            "energy_fraction_defect",
            "energy_fraction_defect_and_ring",
        ):
            value = float(row[key])

            if not 0.0 <= value <= 1.0:
                raise RuntimeError(
                    f"{key} lies outside [0, 1]: "
                    f"{value}."
                )


def _compute_temporal_selections(
    rows: list[dict[str, Any]],
    *,
    primary_mode: str,
    activity_threshold: float,
) -> dict[str, Any]:
    """Compute field-only and ground-truth-assisted selections."""
    if primary_mode not in (
        "absolute_dot",
        "norm_product",
    ):
        raise ValueError(
            "primary_mode must be 'absolute_dot' "
            "or 'norm_product'."
        )

    if not np.isfinite(activity_threshold):
        raise ValueError(
            "activity_threshold must be finite."
        )

    if not 0.0 < activity_threshold <= 1.0:
        raise ValueError(
            "activity_threshold must lie in (0, 1]."
        )

    selections: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    enriched_rows: list[dict[str, Any]] = []

    threshold_values = tuple(
        sorted(
            {
                0.01,
                0.05,
                0.10,
                float(activity_threshold),
            }
        )
    )

    for mode in (
        "absolute_dot",
        "norm_product",
    ):
        selection, mode_rows = (
            select_temporal_overlap_states(
                rows,
                metric_mode=mode,
                activity_threshold=(
                    activity_threshold
                ),
            )
        )

        selections[mode] = (
            selection.as_dict()
        )

        audits[mode] = (
            audit_activity_thresholds(
                rows,
                metric_mode=mode,
                thresholds=threshold_values,
            )
        )

        enriched_rows.extend(mode_rows)

    mode_order = {
        "absolute_dot": 0,
        "norm_product": 1,
    }

    enriched_rows.sort(
        key=lambda row: (
            int(row["time_index"]),
            mode_order[
                str(row["metric_mode"])
            ],
        )
    )

    primary = selections[primary_mode]

    overlap_index = int(
        primary["overlap_selection"][
            "time_index"
        ]
    )

    localization_index = int(
        primary["localization_selection"][
            "time_index"
        ]
    )

    for row in enriched_rows:
        index = int(row["time_index"])

        row[
            "selected_for_overlap_figure"
        ] = bool(
            index == overlap_index
        )

        row[
            "selected_for_localization_figure"
        ] = bool(
            index == localization_index
        )

        # Compatibility with the former single-selection output.
        row["selected_for_figure"] = bool(
            index == overlap_index
        )

    return {
        "primary_mode": str(primary_mode),
        "activity_threshold": float(
            activity_threshold
        ),
        "rows": enriched_rows,
        "selections": selections,
        "threshold_audits": audits,
        "primary_selection": primary,
    }


def _produce_temporal_diagnostics(
    *,
    rows: list[dict[str, Any]],
    primary_mode: str,
    primary_selection: dict[str, Any],
    output_png: Path,
    output_pdf: Path,
    figure_dpi: int,
) -> dict[str, Any]:
    mode_rows = sorted(
        (
            row
            for row in rows
            if str(row["metric_mode"])
            == str(primary_mode)
        ),
        key=lambda row: float(
            row["time_value"]
        ),
    )

    if not mode_rows:
        raise RuntimeError(
            "No rows are available for the temporal figure."
        )

    times = np.asarray(
        [
            float(row["time_value"])
            for row in mode_rows
        ],
        dtype=np.float64,
    )

    relative_energy = np.asarray(
        [
            float(
                row["relative_total_energy"]
            )
            for row in mode_rows
        ],
        dtype=np.float64,
    )

    enrichment = np.asarray(
        [
            float(
                row[
                    "defect_ring_energy_enrichment"
                ]
            )
            for row in mode_rows
        ],
        dtype=np.float64,
    )

    peak_distance = np.asarray(
        [
            (
                float(
                    row[
                        "distance_peak_to_defect_center"
                    ]
                )
                if bool(row["peak_defined"])
                else np.nan
            )
            for row in mode_rows
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(relative_energy)
    ):
        raise RuntimeError(
            "The relative-energy curve is not finite."
        )

    if not np.all(
        np.isfinite(enrichment)
    ):
        raise RuntimeError(
            "The enrichment curve is not finite."
        )

    finite_peak = np.isfinite(
        peak_distance
    )

    if not np.any(finite_peak):
        raise RuntimeError(
            "No finite peak-distance value is available."
        )

    overlap = primary_selection[
        "overlap_selection"
    ]

    localization = primary_selection[
        "localization_selection"
    ]

    overlap_time = float(
        overlap["time_value"]
    )

    localization_time = float(
        localization["time_value"]
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(9.2, 10.0),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        times,
        relative_energy,
        marker="o",
        markersize=3,
        label="relative total overlap energy",
    )

    axes[0].set_ylabel(
        "relative energy"
    )

    axes[0].set_ylim(
        bottom=0.0
    )

    axes[1].plot(
        times,
        enrichment,
        marker="o",
        markersize=3,
        label="defect-plus-ring enrichment",
    )

    axes[1].set_ylabel(
        "energy enrichment"
    )

    axes[1].set_ylim(
        bottom=0.0
    )

    axes[2].plot(
        times[finite_peak],
        peak_distance[finite_peak],
        marker="o",
        markersize=3,
        label="peak-to-defect-centre distance",
    )

    axes[2].set_xlabel("time")
    axes[2].set_ylabel("distance")
    axes[2].set_ylim(
        bottom=0.0
    )

    for axis_index, axis in enumerate(
        axes
    ):
        axis.grid(True)

        axis.axvline(
            overlap_time,
            linestyle="--",
            label=(
                "field-only overlap time"
                if axis_index == 0
                else None
            ),
        )

        axis.axvline(
            localization_time,
            linestyle=":",
            label=(
                "ground-truth localization time"
                if axis_index == 0
                else None
            ),
        )

        axis.legend(
            loc="best",
            fontsize="small",
        )

    figure.suptitle(
        "Residual-wave temporal diagnostics "
        f"({primary_mode})"
    )

    figure.savefig(
        output_png,
        dpi=figure_dpi,
        bbox_inches="tight",
    )

    figure.savefig(
        output_pdf,
        dpi=figure_dpi,
        bbox_inches="tight",
    )

    plt.close(figure)

    return {
        "metric_mode": str(primary_mode),
        "num_snapshots": len(mode_rows),
        "overlap_time": overlap_time,
        "localization_time": localization_time,
        "activity_threshold": float(
            primary_selection[
                "activity_threshold"
            ]
        ),
        "overlap_selection_semantics": (
            "field-only maximum total weighted "
            "interaction energy"
        ),
        "localization_selection_semantics": (
            "retrospective ground-truth-assisted "
            "maximum defect-plus-ring energy enrichment "
            "among active snapshots"
        ),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify and visualize the instantaneous "
            "spatial overlap between reflection and "
            "transmission residual wavefields. The product "
            "is an interaction indicator, not a FWI gradient."
        )
    )

    parser.add_argument(
        "--geometry",
        choices=("line", "box", "unknown"),
        default="unknown",
        help=(
            "Geometry metadata for this single-campaign overlap analysis. "
            "Use line for a genuine line-only run; no box field is required."
        ),
    )

    parser.add_argument(
        "--reflection-field",
        required=True,
        type=Path,
        help=(
            "PVD file containing "
            "reflection_residual_wavefield."
        ),
    )

    parser.add_argument(
        "--transmission-field",
        required=True,
        type=Path,
        help=(
            "PVD file containing "
            "transmission_residual_wavefield."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--case-id",
        required=True,
    )

    parser.add_argument(
        "--defect-center-x",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--defect-center-y",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--defect-radius",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--ring-width",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--interaction-mode",
        choices=(
            "signed_dot",
            "absolute_dot",
            "norm_product",
        ),
        default="norm_product",
        help=(
            "Interaction displayed in the triptych. "
            "All physical metrics are still exported for "
            "absolute_dot and norm_product."
        ),
    )

    parser.add_argument(
        "--select-by",
        choices=(
            "absolute_dot",
            "norm_product",
        ),
        default="norm_product",
        help=(
            "Non-negative interaction whose total weighted "
            "energy selects the representative snapshot."
        ),
    )

    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help=(
            "Analyze exactly one exported time instead of "
            "scanning an interval."
        ),
    )

    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--raster-resolution",
        type=int,
        default=700,
    )

    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--localization-activity-threshold",
        type=float,
        default=0.10,
        help=(
            "Minimum fraction of the maximum total "
            "interaction energy retained for the "
            "retrospective ground-truth localization "
            "selection."
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


def main(
    argv: list[str] | None = None,
) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    _check_sequential_execution()

    repository_root = Path.cwd().resolve()

    reflection_pvd = (
        args.reflection_field
        .expanduser()
        .resolve()
    )

    transmission_pvd = (
        args.transmission_field
        .expanduser()
        .resolve()
    )

    if not reflection_pvd.is_file():
        raise FileNotFoundError(
            reflection_pvd
        )

    if not transmission_pvd.is_file():
        raise FileNotFoundError(
            transmission_pvd
        )

    geometry = DefectGeometry(
        center_x=float(
            args.defect_center_x
        ),
        center_y=float(
            args.defect_center_y
        ),
        radius=float(
            args.defect_radius
        ),
        ring_width=float(
            args.ring_width
        ),
    )

    geometry.validate()

    if (
        args.time is not None
        and (
            args.start_time is not None
            or args.end_time is not None
        )
    ):
        raise ValueError(
            "--time cannot be combined with "
            "--start-time or --end-time."
        )

    if (
        args.start_time is not None
        and args.end_time is not None
        and args.end_time < args.start_time
    ):
        raise ValueError(
            "--end-time must be greater than or "
            "equal to --start-time."
        )

    if args.figure_dpi <= 0:
        raise ValueError(
            "--figure-dpi must be positive."
        )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    output_paths = (
        _prepare_output_directory(
            output_dir,
            overwrite=bool(args.overwrite),
        )
    )

    scan = _scan_campaign(
        reflection_pvd=reflection_pvd,
        transmission_pvd=transmission_pvd,
        case_id=str(args.case_id),
        geometry=geometry,
        start_time=args.start_time,
        end_time=args.end_time,
        single_time=args.time,
        select_by=str(args.select_by),
        quiet=bool(args.quiet),
    )

    rows = scan["rows"]
    analysis_mode = {
        "line": "line-only",
        "box": "box-only",
        "unknown": "single-campaign",
    }[str(args.geometry)]
    for row in rows:
        row["analysis_mode"] = analysis_mode
        row["geometry_kind"] = str(args.geometry)

    _validate_metric_rows(rows)

    temporal = _compute_temporal_selections(
        rows,
        primary_mode=str(args.select_by),
        activity_threshold=float(
            args.localization_activity_threshold
        ),
    )

    rows = temporal["rows"]

    _validate_metric_rows(rows)

    primary_selection = temporal[
        "primary_selection"
    ]

    overlap_selection = primary_selection[
        "overlap_selection"
    ]

    localization_selection = primary_selection[
        "localization_selection"
    ]

    if (
        int(scan["selected_index"])
        != int(overlap_selection["time_index"])
    ):
        raise RuntimeError(
            "The legacy scan selection and the reusable "
            "temporal-selection module disagree."
        )

    _write_csv(
        output_paths["metrics_csv"],
        rows,
    )

    metrics_payload = {
        "status": "OK",
        "case_id": str(args.case_id),
        "analysis_mode": analysis_mode,
        "geometry_kind": str(args.geometry),
        "scientific_role": (
            "instantaneous_residual_wave_spatial_overlap"
        ),
        "is_fwi_gradient": False,
        "is_descent_direction": False,
        "is_time_integrated_material_kernel": False,
        "physical_metrics_use_normalized_data": False,
        "defect_geometry": {
            "center_x": geometry.center_x,
            "center_y": geometry.center_y,
            "radius": geometry.radius,
            "ring_width": geometry.ring_width,
        },
        "interaction_definitions": {
            "signed_dot": (
                "W1_xy dot Lambda2_xy"
            ),
            "absolute_dot": (
                "abs(W1_xy dot Lambda2_xy)"
            ),
            "norm_product": (
                "norm(W1_xy) * norm(Lambda2_xy)"
            ),
        },
        "spatial_weighting": (
            "P1 nodal lumped area: "
            "sum(cell_area / 3)"
        ),
        "selection_semantics": {
            "overlap": (
                "field-only maximum total weighted "
                "interaction energy"
            ),
            "localization": (
                "retrospective ground-truth-assisted "
                "maximum defect-plus-ring energy enrichment "
                "among active snapshots"
            ),
            "localization_is_blind_inverse_criterion": False,
        },
        "primary_selection_mode": (
            temporal["primary_mode"]
        ),
        "localization_activity_threshold": (
            temporal[
                "activity_threshold"
            ]
        ),
        "temporal_selections": (
            temporal["selections"]
        ),
        "activity_threshold_audits": (
            temporal["threshold_audits"]
        ),
        "selection": {
            "select_by": temporal[
                "primary_mode"
            ],
            "time_index": int(
                overlap_selection[
                    "time_index"
                ]
            ),
            "time_value": float(
                overlap_selection[
                    "time_value"
                ]
            ),
            "weighted_energy": float(
                overlap_selection[
                    "energy_total"
                ]
            ),
            "overlap": overlap_selection,
            "localization": (
                localization_selection
            ),
        },
        "processed_indices": (
            scan["processed_indices"]
        ),
        "processed_times": (
            scan["processed_times"]
        ),
        "rows": rows,
    }

    output_paths["metrics_json"].write_text(
        json.dumps(
            metrics_payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    global_figure_metadata = _produce_figures(
        reflection_pvd=reflection_pvd,
        transmission_pvd=transmission_pvd,
        selected_time=float(
            overlap_selection[
                "time_value"
            ]
        ),
        geometry=geometry,
        interaction_mode=str(
            args.interaction_mode
        ),
        raster_resolution=int(
            args.raster_resolution
        ),
        figure_dpi=int(
            args.figure_dpi
        ),
        output_paths={
            "triptych_png": output_paths[
                "global_triptych_png"
            ],
            "triptych_pdf": output_paths[
                "global_triptych_pdf"
            ],
            "localization_png": output_paths[
                "global_comparison_png"
            ],
            "localization_pdf": output_paths[
                "global_comparison_pdf"
            ],
        },
    )

    localization_figure_metadata = (
        _produce_figures(
            reflection_pvd=reflection_pvd,
            transmission_pvd=transmission_pvd,
            selected_time=float(
                localization_selection[
                    "time_value"
                ]
            ),
            geometry=geometry,
            interaction_mode=str(
                args.interaction_mode
            ),
            raster_resolution=int(
                args.raster_resolution
            ),
            figure_dpi=int(
                args.figure_dpi
            ),
            output_paths={
                "triptych_png": output_paths[
                    "localization_triptych_png"
                ],
                "triptych_pdf": output_paths[
                    "localization_triptych_pdf"
                ],
                "localization_png": output_paths[
                    "localization_comparison_png"
                ],
                "localization_pdf": output_paths[
                    "localization_comparison_pdf"
                ],
            },
        )
    )

    temporal_figure_metadata = (
        _produce_temporal_diagnostics(
            rows=rows,
            primary_mode=temporal[
                "primary_mode"
            ],
            primary_selection=(
                primary_selection
            ),
            output_png=output_paths[
                "temporal_diagnostics_png"
            ],
            output_pdf=output_paths[
                "temporal_diagnostics_pdf"
            ],
            figure_dpi=int(
                args.figure_dpi
            ),
        )
    )

    figure_metadata = {
        "global_overlap": {
            "selection_semantics": (
                "field-only"
            ),
            **global_figure_metadata,
        },
        "retrospective_localization": {
            "selection_semantics": (
                "ground-truth-assisted"
            ),
            **localization_figure_metadata,
        },
        "temporal_diagnostics": (
            temporal_figure_metadata
        ),
    }

    required_output_keys = (
        "global_triptych_png",
        "global_triptych_pdf",
        "global_comparison_png",
        "global_comparison_pdf",
        "localization_triptych_png",
        "localization_triptych_pdf",
        "localization_comparison_png",
        "localization_comparison_pdf",
        "temporal_diagnostics_png",
        "temporal_diagnostics_pdf",
        "metrics_csv",
        "metrics_json",
    )

    for key in required_output_keys:
        path = output_paths[key]

        if not path.is_file():
            raise RuntimeError(
                f"Expected output is missing: {path}"
            )

        if path.stat().st_size <= 0:
            raise RuntimeError(
                f"Expected output is empty: {path}"
            )

    try:
        from vtkmodules.vtkCommonCore import (
            vtkVersion,
        )

        vtk_version = (
            vtkVersion().GetVTKVersion()
        )
    except ImportError:
        vtk_version = None

    manifest = {
        "status": "OK",
        "diagnostic": (
            "plot_residual_wave_overlap"
        ),
        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        "repository_root": str(
            repository_root
        ),
        "git_commit": _git_commit(
            repository_root
        ),
        "case_id": str(args.case_id),
        "analysis_mode": analysis_mode,
        "geometry_kind": str(args.geometry),
        "inputs": {
            "reflection_pvd": str(
                reflection_pvd
            ),
            "reflection_pvd_sha256": (
                _sha256(reflection_pvd)
            ),
            "transmission_pvd": str(
                transmission_pvd
            ),
            "transmission_pvd_sha256": (
                _sha256(transmission_pvd)
            ),
            "reflection_field_name": (
                REFLECTION_FIELD_NAME
            ),
            "transmission_field_name": (
                TRANSMISSION_FIELD_NAME
            ),
        },
        "defect_geometry": (
            metrics_payload[
                "defect_geometry"
            ]
        ),
        "scan": {
            key: value
            for key, value in scan.items()
            if key != "rows"
        },
        "figure_metadata": figure_metadata,
        "scientific_semantics": {
            "role": (
                "residual-wave spatial-overlap "
                "indicator"
            ),
            "is_fwi_gradient": False,
            "is_descent_direction": False,
            "is_time_integrated_material_kernel": False,
            "signed_and_absolute_maps_kept_separate": True,
            "normalization_reserved_for_visualization": True,
            "metrics_computed_on_raw_nonnegative_interactions": True,
            "overlap_selection_is_field_only": True,
            "localization_selection_uses_ground_truth": True,
            "localization_selection_is_blind_inverse_criterion": False,
        },
        "software": {
            "python": sys.version,
            "python_executable": (
                sys.executable
            ),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "matplotlib": (
                matplotlib.__version__
            ),
            "vtk": vtk_version,
        },
        "finiteness_checks": {
            "all_metric_rows_finite": True,
            "all_real_fields_checked_by_reader": True,
        },
        "outputs": {
            key: {
                "path": str(
                    output_paths[key]
                ),
                "size_bytes": int(
                    output_paths[key]
                    .stat()
                    .st_size
                ),
                "sha256": _sha256(
                    output_paths[key]
                ),
            }
            for key in required_output_keys
        },
    }

    output_paths[
        "manifest_json"
    ].write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if not args.quiet:
        print()
        print(
            "[RESIDUAL-WAVE OVERLAP] OK"
        )
        print(
            "case_id =",
            args.case_id,
        )
        print(
            "primary_selection_mode =",
            temporal["primary_mode"],
        )
        print(
            "overlap_time =",
            overlap_selection[
                "time_value"
            ],
        )
        print(
            "overlap_index =",
            overlap_selection[
                "time_index"
            ],
        )
        print(
            "overlap_energy =",
            overlap_selection[
                "energy_total"
            ],
        )
        print(
            "localization_time =",
            localization_selection[
                "time_value"
            ],
        )
        print(
            "localization_index =",
            localization_selection[
                "time_index"
            ],
        )
        print(
            "localization_relative_energy =",
            localization_selection[
                "relative_total_energy"
            ],
        )
        print(
            "localization_defect_ring_enrichment =",
            localization_selection[
                "defect_ring_energy_enrichment"
            ],
        )
        print(
            "output_dir =",
            output_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
