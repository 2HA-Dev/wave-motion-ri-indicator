from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def _require_keys(data: np.lib.npyio.NpzFile, required: set[str]) -> None:
    missing = sorted(required.difference(data.files))

    if missing:
        raise ValueError(f"Missing required NPZ keys: {missing}")


def _validate_common_npz(data: np.lib.npyio.NpzFile) -> None:
    _require_keys(data, {"kind", "status"})

    if str(data["kind"]) != "material_qualitative_image_export":
        raise ValueError(
            "Invalid NPZ kind. Expected 'material_qualitative_image_export'."
        )

    if str(data["status"]) != "ok":
        raise ValueError("Cannot plot an export whose status is not 'ok'.")

    _require_keys(
        data,
        {
            "mpi_rank",
            "mpi_size",
            "image_global_l2_norm",
            "image_global_max",
            "normalized",
        },
    )


def _validate_dof_image(data: np.lib.npyio.NpzFile) -> None:
    _require_keys(
        data,
        {
            "image_local_values",
            "image_local_coordinates",
            "image_coordinate_dim",
            "image_local_size",
            "image_global_size",
        },
    )

    values = np.asarray(data["image_local_values"], dtype=float)
    coordinates = np.asarray(data["image_local_coordinates"], dtype=float)
    mpi_size = int(data["mpi_size"])

    if values.ndim != 1:
        raise ValueError("image_local_values must be one-dimensional.")

    if values.size != int(data["image_local_size"]):
        raise ValueError("image_local_values size does not match image_local_size.")

    if values.size == 0 and mpi_size == 1:
        raise ValueError("image_local_values is empty.")

    if not np.all(np.isfinite(values)):
        raise ValueError("image_local_values contains non-finite entries.")

    if coordinates.ndim != 2:
        raise ValueError("image_local_coordinates must be two-dimensional.")

    if coordinates.shape[1] != 2:
        raise ValueError("image_local_coordinates must have shape (n_local_dofs, 2).")

    if coordinates.shape[0] != values.size:
        raise ValueError(
            "image_local_coordinates row count does not match image_local_values size."
        )

    if not np.all(np.isfinite(coordinates)):
        raise ValueError("image_local_coordinates contains non-finite entries.")

    if int(data["image_coordinate_dim"]) != 2:
        raise ValueError("image_coordinate_dim must be equal to 2 for 2D plotting.")


def _validate_nodal_image(data: np.lib.npyio.NpzFile) -> None:
    _require_keys(
        data,
        {
            "image_node_values",
            "image_node_coordinates",
            "image_node_count",
            "image_node_aggregation",
        },
    )

    values = np.asarray(data["image_node_values"], dtype=float)
    coordinates = np.asarray(data["image_node_coordinates"], dtype=float)
    mpi_size = int(data["mpi_size"])

    if values.ndim != 1:
        raise ValueError("image_node_values must be one-dimensional.")

    if values.size != int(data["image_node_count"]):
        raise ValueError("image_node_values size does not match image_node_count.")

    if values.size == 0 and mpi_size == 1:
        raise ValueError("image_node_values is empty.")

    if not np.all(np.isfinite(values)):
        raise ValueError("image_node_values contains non-finite entries.")

    if coordinates.ndim != 2:
        raise ValueError("image_node_coordinates must be two-dimensional.")

    if coordinates.shape[1] != 2:
        raise ValueError("image_node_coordinates must have shape (n_nodes, 2).")

    if coordinates.shape[0] != values.size:
        raise ValueError(
            "image_node_coordinates row count does not match image_node_values size."
        )

    if not np.all(np.isfinite(coordinates)):
        raise ValueError("image_node_coordinates contains non-finite entries.")


def _validate_cell_image(data: np.lib.npyio.NpzFile) -> None:
    _require_keys(
        data,
        {
            "image_cell_values",
            "image_cell_centers",
            "image_cell_count",
            "image_cell_aggregation",
        },
    )

    values = np.asarray(data["image_cell_values"], dtype=float)
    centers = np.asarray(data["image_cell_centers"], dtype=float)
    mpi_size = int(data["mpi_size"])

    if values.ndim != 1:
        raise ValueError("image_cell_values must be one-dimensional.")

    if values.size != int(data["image_cell_count"]):
        raise ValueError("image_cell_values size does not match image_cell_count.")

    if values.size == 0 and mpi_size == 1:
        raise ValueError("image_cell_values is empty.")

    if not np.all(np.isfinite(values)):
        raise ValueError("image_cell_values contains non-finite entries.")

    if centers.ndim != 2:
        raise ValueError("image_cell_centers must be two-dimensional.")

    if centers.shape[1] != 2:
        raise ValueError("image_cell_centers must have shape (n_cells, 2).")

    if centers.shape[0] != values.size:
        raise ValueError(
            "image_cell_centers row count does not match image_cell_values size."
        )

    if not np.all(np.isfinite(centers)):
        raise ValueError("image_cell_centers contains non-finite entries.")


def _validate_npz(
    data: np.lib.npyio.NpzFile,
    *,
    use_nodal_image: bool = False,
    use_cell_image: bool = False,
) -> None:
    _validate_common_npz(data)

    if use_nodal_image and use_cell_image:
        raise ValueError("use_nodal_image and use_cell_image are mutually exclusive.")

    if use_cell_image:
        _validate_cell_image(data)
    elif use_nodal_image:
        _validate_nodal_image(data)
    else:
        _validate_dof_image(data)


def _read_optional_defect_metadata(
    data: np.lib.npyio.NpzFile,
) -> tuple[float, float, float] | None:
    required = {"defect_center_x", "defect_center_y", "defect_radius"}

    if not required.issubset(data.files):
        return None

    center_x = float(data["defect_center_x"])
    center_y = float(data["defect_center_y"])
    radius = float(data["defect_radius"])

    if not np.isfinite(center_x) or not np.isfinite(center_y) or not np.isfinite(radius):
        raise ValueError("Defect overlay metadata contains non-finite values.")

    if radius <= 0.0:
        raise ValueError("defect_radius must be strictly positive for overlay plotting.")

    return center_x, center_y, radius


def _read_plot_arrays(
    data: np.lib.npyio.NpzFile,
    *,
    use_nodal_image: bool,
    use_cell_image: bool,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    if use_cell_image:
        values = np.asarray(data["image_cell_values"], dtype=float)
        coordinates = np.asarray(data["image_cell_centers"], dtype=float)
        aggregation = str(data["image_cell_aggregation"])
        return coordinates, values, "cell", aggregation

    if use_nodal_image:
        values = np.asarray(data["image_node_values"], dtype=float)
        coordinates = np.asarray(data["image_node_coordinates"], dtype=float)
        aggregation = str(data["image_node_aggregation"])
        return coordinates, values, "node", aggregation

    values = np.asarray(data["image_local_values"], dtype=float)
    coordinates = np.asarray(data["image_local_coordinates"], dtype=float)
    return coordinates, values, "dof", ""


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("Cell centers must be one-dimensional.")

    if values.size == 0:
        return np.empty((0,), dtype=float)

    if values.size == 1:
        half_width = 0.5
        return np.asarray(
            [values[0] - half_width, values[0] + half_width],
            dtype=float,
        )

    midpoints = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])

    return np.concatenate(
        [
            np.asarray([first], dtype=float),
            midpoints,
            np.asarray([last], dtype=float),
        ]
    )


def _structured_cell_grid_from_centers(
    coordinates: np.ndarray,
    values: np.ndarray,
    *,
    tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if values.size == 0:
        return None

    x_unique = np.unique(np.round(coordinates[:, 0] / tolerance).astype(np.int64))
    y_unique = np.unique(np.round(coordinates[:, 1] / tolerance).astype(np.int64))

    if x_unique.size * y_unique.size != values.size:
        return None

    x_centers = x_unique.astype(float) * tolerance
    y_centers = y_unique.astype(float) * tolerance

    x_order = np.argsort(x_centers)
    y_order = np.argsort(y_centers)

    x_centers = x_centers[x_order]
    y_centers = y_centers[y_order]

    x_index = {
        int(round(x / tolerance)): index
        for index, x in enumerate(x_centers)
    }
    y_index = {
        int(round(y / tolerance)): index
        for index, y in enumerate(y_centers)
    }

    grid = np.full((y_centers.size, x_centers.size), np.nan, dtype=float)

    for (x, y), value in zip(coordinates, values, strict=True):
        key_x = int(round(float(x) / tolerance))
        key_y = int(round(float(y) / tolerance))

        if key_x not in x_index or key_y not in y_index:
            return None

        grid[y_index[key_y], x_index[key_x]] = float(value)

    if not np.all(np.isfinite(grid)):
        return None

    x_edges = _centers_to_edges(x_centers)
    y_edges = _centers_to_edges(y_centers)

    return x_edges, y_edges, grid


def plot_material_qualitative_image_npz(
    *,
    input_path: str | Path,
    output_png: str | Path,
    output_pdf: str | Path | None = None,
    title: str | None = None,
    show_defect_overlay: bool = True,
    use_nodal_image: bool = False,
    use_cell_image: bool = False,
    cell_plot_style: str = "grid",
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_png = Path(output_png)
    output_pdf = Path(output_pdf) if output_pdf is not None else None

    if cell_plot_style not in {"grid", "scatter"}:
        raise ValueError("cell_plot_style must be either 'grid' or 'scatter'.")

    with np.load(input_path, allow_pickle=False) as data:
        _validate_npz(
            data,
            use_nodal_image=use_nodal_image,
            use_cell_image=use_cell_image,
        )

        coordinates, values, image_mode, image_aggregation = _read_plot_arrays(
            data,
            use_nodal_image=use_nodal_image,
            use_cell_image=use_cell_image,
        )

        rank = int(data["mpi_rank"])
        size = int(data["mpi_size"])
        global_l2 = float(data["image_global_l2_norm"])
        global_max = float(data["image_global_max"])
        normalized = bool(data["normalized"])

        defect_metadata = _read_optional_defect_metadata(data)

    output_png.parent.mkdir(parents=True, exist_ok=True)

    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    effective_cell_plot_style = "not-cell"

    if use_cell_image and cell_plot_style == "grid":
        grid_data = _structured_cell_grid_from_centers(coordinates, values)

        if grid_data is not None:
            x_edges, y_edges, grid = grid_data
            image_handle = ax.pcolormesh(
                x_edges,
                y_edges,
                grid,
                shading="auto",
            )
            effective_cell_plot_style = "grid"
        else:
            image_handle = ax.scatter(
                x,
                y,
                c=values,
                s=64.0,
                marker="s",
            )
            effective_cell_plot_style = "scatter-fallback"
    else:
        if use_cell_image:
            marker_size = 64.0
            marker = "s"
            effective_cell_plot_style = "scatter"
        elif use_nodal_image:
            marker_size = 42.0
            marker = "o"
        else:
            marker_size = 28.0
            marker = "o"

        image_handle = ax.scatter(
            x,
            y,
            c=values,
            s=marker_size,
            marker=marker,
        )

    colorbar = fig.colorbar(image_handle, ax=ax)
    colorbar.set_label("Indicateur qualitatif")

    defect_overlay_added = False

    if show_defect_overlay and defect_metadata is not None:
        center_x, center_y, radius = defect_metadata

        circle = Circle(
            (center_x, center_y),
            radius,
            fill=False,
            linestyle="--",
            linewidth=1.5,
            label="Défaut exact",
        )
        ax.add_patch(circle)
        ax.legend(loc="best")
        defect_overlay_added = True

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    if title is None:
        if use_cell_image:
            mode_label = "cellule"
        elif use_nodal_image:
            mode_label = "nodale"
        else:
            mode_label = "ddl"

        aggregation_label = (
            f", agrégation={image_aggregation}"
            if use_nodal_image or use_cell_image
            else ""
        )
        title = (
            f"Image qualitative matériau 2D {mode_label} "
            f"(rang {rank}/{size}, "
            f"normée={normalized}, "
            f"L2 globale={global_l2:.3e}, max global={global_max:.3e}"
            f"{aggregation_label})"
        )

    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_png, dpi=200)

    if output_pdf is not None:
        fig.savefig(output_pdf)

    plt.close(fig)

    return {
        "kind": "material_qualitative_image_plot",
        "status": "ok",
        "input": str(input_path),
        "output_png": str(output_png),
        "output_pdf": str(output_pdf) if output_pdf is not None else "",
        "mpi_rank": rank,
        "mpi_size": size,
        "num_values": int(values.size),
        "coordinate_dimension": 2,
        "image_mode": image_mode,
        "use_nodal_image": bool(use_nodal_image),
        "use_cell_image": bool(use_cell_image),
        "node_aggregation": image_aggregation if use_nodal_image else "",
        "cell_aggregation": image_aggregation if use_cell_image else "",
        "cell_plot_style": effective_cell_plot_style,
        "global_l2_norm": global_l2,
        "global_max": global_max,
        "normalized": normalized,
        "defect_overlay_available": defect_metadata is not None,
        "defect_overlay_added": defect_overlay_added,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot a 2D qualitative material image exported as NPZ."
    )

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-png", type=str, required=True)
    parser.add_argument("--output-pdf", type=str, default="")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--no-defect-overlay", action="store_true")
    parser.add_argument("--use-nodal-image", action="store_true")
    parser.add_argument("--use-cell-image", action="store_true")
    parser.add_argument(
        "--cell-plot-style",
        choices=("grid", "scatter"),
        default="grid",
    )

    args = parser.parse_args()

    result = plot_material_qualitative_image_npz(
        input_path=args.input,
        output_png=args.output_png,
        output_pdf=args.output_pdf or None,
        title=args.title or None,
        show_defect_overlay=not args.no_defect_overlay,
        use_nodal_image=args.use_nodal_image,
        use_cell_image=args.use_cell_image,
        cell_plot_style=args.cell_plot_style,
    )

    print(
        "[MATERIAL QUALITATIVE IMAGE PLOT] OK "
        f"for input={result['input']}, "
        f"png={result['output_png']}, "
        f"pdf={result['output_pdf'] or 'none'}, "
        f"rank={result['mpi_rank']}/{result['mpi_size']}, "
        f"values={result['num_values']}, "
        f"image_mode={result['image_mode']}, "
        f"cell_plot_style={result['cell_plot_style']}, "
        f"coordinate_dimension={result['coordinate_dimension']}, "
        f"defect_overlay_added={result['defect_overlay_added']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
