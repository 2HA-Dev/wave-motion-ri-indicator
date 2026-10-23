from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


_REQUIRED_COLUMNS = {
    "nx",
    "ny",
    "degree",
    "num_steps",
    "mpi_rank",
    "mpi_size",
    "status",
    "image_global_l2_norm",
    "image_global_max",
    "image_node_count",
    "image_node_aggregation",
    "image_cell_count",
    "image_cell_aggregation",
    "defect_metric_status",
    "defect_metric_max_distance_to_center",
    "defect_metric_inside_mass_fraction",
    "defect_metric_inside_outside_mean_ratio",
    "image_cell_count",
    "image_cell_aggregation",
}


def _read_campaign_summary_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Campaign summary CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)

    if not rows:
        raise ValueError("Campaign summary CSV is empty.")

    missing = sorted(_REQUIRED_COLUMNS.difference(rows[0].keys()))

    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    parsed_rows: list[dict[str, Any]] = []

    for row in rows:
        if row["status"] != "ok":
            raise ValueError("Cannot plot a campaign summary containing failed cases.")

        nx = int(row["nx"])
        ny = int(row["ny"])
        degree = int(row["degree"])
        num_steps = int(row["num_steps"])
        mpi_rank = int(row["mpi_rank"])
        mpi_size = int(row["mpi_size"])
        image_global_l2_norm = float(row["image_global_l2_norm"])
        image_global_max = float(row["image_global_max"])
        image_node_count = int(row["image_node_count"])
        image_node_aggregation = row["image_node_aggregation"]
        image_cell_count = int(row["image_cell_count"])
        image_cell_aggregation = row["image_cell_aggregation"]
        defect_metric_status = row["defect_metric_status"]
        defect_metric_max_distance_to_center = float(
            row["defect_metric_max_distance_to_center"]
        )
        defect_metric_inside_mass_fraction = float(
            row["defect_metric_inside_mass_fraction"]
        )
        defect_metric_inside_outside_mean_ratio = float(
            row["defect_metric_inside_outside_mean_ratio"]
        )
        image_cell_count = int(row["image_cell_count"])
        image_cell_aggregation = row["image_cell_aggregation"]

        if nx <= 0 or ny <= 0:
            raise ValueError("Mesh sizes must be strictly positive.")

        if degree <= 0:
            raise ValueError("degree must be strictly positive.")

        if num_steps <= 0:
            raise ValueError("num_steps must be strictly positive.")

        if mpi_rank < 0 or mpi_size <= 0:
            raise ValueError("Invalid MPI metadata in campaign summary.")

        if not np.isfinite(image_global_l2_norm) or image_global_l2_norm <= 0.0:
            raise ValueError("image_global_l2_norm must be finite and positive.")

        if not np.isfinite(image_global_max) or image_global_max <= 0.0:
            raise ValueError("image_global_max must be finite and positive.")

        if image_node_count < 0:
            raise ValueError("image_node_count must be non-negative.")

        if image_cell_count < 0:
            raise ValueError("image_cell_count must be non-negative.")

        if defect_metric_status not in {"ok", "empty"}:
            raise ValueError("Invalid defect_metric_status.")

        if defect_metric_status == "ok":
            if not np.isfinite(defect_metric_max_distance_to_center):
                raise ValueError(
                    "defect_metric_max_distance_to_center must be finite "
                    "when defect_metric_status is ok."
                )

            if not np.isfinite(defect_metric_inside_outside_mean_ratio):
                raise ValueError(
                    "defect_metric_inside_outside_mean_ratio must be finite "
                    "when defect_metric_status is ok."
                )

        if not np.isfinite(defect_metric_inside_mass_fraction):
            raise ValueError("defect_metric_inside_mass_fraction must be finite.")

        if defect_metric_inside_mass_fraction < 0.0:
            raise ValueError("defect_metric_inside_mass_fraction must be non-negative.")

        if image_cell_count < 0:
            raise ValueError("image_cell_count must be non-negative.")

        parsed_rows.append(
            {
                "nx": nx,
                "ny": ny,
                "degree": degree,
                "num_steps": num_steps,
                "mpi_rank": mpi_rank,
                "mpi_size": mpi_size,
                "image_global_l2_norm": image_global_l2_norm,
                "image_global_max": image_global_max,
                "image_node_count": image_node_count,
                "image_node_aggregation": image_node_aggregation,
                "image_cell_count": image_cell_count,
                "image_cell_aggregation": image_cell_aggregation,
                "defect_metric_status": defect_metric_status,
                "defect_metric_max_distance_to_center": (
                    defect_metric_max_distance_to_center
                ),
                "defect_metric_inside_mass_fraction": (
                    defect_metric_inside_mass_fraction
                ),
                "defect_metric_inside_outside_mean_ratio": (
                    defect_metric_inside_outside_mean_ratio
                ),
                "image_cell_count": image_cell_count,
                "image_cell_aggregation": image_cell_aggregation,
            }
        )

    parsed_rows.sort(key=lambda item: (item["nx"], item["ny"], item["mpi_rank"]))

    return parsed_rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def _plot_series(
    *,
    x: np.ndarray,
    y: np.ndarray,
    ylabel: str,
    title: str,
    output_png: Path,
    output_pdf: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    ax.plot(x, y, marker="o", linewidth=1.5)
    ax.set_xlabel("N")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    fig.savefig(output_pdf)

    plt.close(fig)


def plot_material_qualitative_image_campaign_summary(
    *,
    summary_csv: str | Path,
    output_dir: str | Path,
    prefix: str = "material_image_campaign",
) -> dict[str, Any]:
    summary_csv = Path(summary_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_campaign_summary_csv(summary_csv)

    n_values = np.asarray([row["nx"] for row in rows], dtype=float)
    l2_values = np.asarray([row["image_global_l2_norm"] for row in rows], dtype=float)
    max_values = np.asarray([row["image_global_max"] for row in rows], dtype=float)
    node_counts = np.asarray([row["image_node_count"] for row in rows], dtype=float)
    cell_counts = np.asarray([row["image_cell_count"] for row in rows], dtype=float)
    inside_mass_fractions = np.asarray(
        [row["defect_metric_inside_mass_fraction"] for row in rows],
        dtype=float,
    )
    max_distances_to_center = np.asarray(
        [
            row["defect_metric_max_distance_to_center"]
            for row in rows
            if row["defect_metric_status"] == "ok"
        ],
        dtype=float,
    )
    n_values_for_max_distances = np.asarray(
        [
            row["nx"]
            for row in rows
            if row["defect_metric_status"] == "ok"
        ],
        dtype=float,
    )
    inside_outside_mean_ratios = np.asarray(
        [
            row["defect_metric_inside_outside_mean_ratio"]
            for row in rows
            if row["defect_metric_status"] == "ok"
        ],
        dtype=float,
    )
    n_values_for_ratios = np.asarray(
        [
            row["nx"]
            for row in rows
            if row["defect_metric_status"] == "ok"
        ],
        dtype=float,
    )

    l2_png = output_dir / f"{prefix}_l2_norm.png"
    l2_pdf = output_dir / f"{prefix}_l2_norm.pdf"
    max_png = output_dir / f"{prefix}_global_max.png"
    max_pdf = output_dir / f"{prefix}_global_max.pdf"
    node_count_png = output_dir / f"{prefix}_node_count.png"
    node_count_pdf = output_dir / f"{prefix}_node_count.pdf"
    cell_count_png = output_dir / f"{prefix}_cell_count.png"
    cell_count_pdf = output_dir / f"{prefix}_cell_count.pdf"
    defect_inside_mass_fraction_png = (
        output_dir / f"{prefix}_defect_inside_mass_fraction.png"
    )
    defect_inside_mass_fraction_pdf = (
        output_dir / f"{prefix}_defect_inside_mass_fraction.pdf"
    )
    defect_max_distance_png = (
        output_dir / f"{prefix}_defect_max_distance_to_center.png"
    )
    defect_max_distance_pdf = (
        output_dir / f"{prefix}_defect_max_distance_to_center.pdf"
    )
    defect_inside_outside_ratio_png = (
        output_dir / f"{prefix}_defect_inside_outside_mean_ratio.png"
    )
    defect_inside_outside_ratio_pdf = (
        output_dir / f"{prefix}_defect_inside_outside_mean_ratio.pdf"
    )
    manifest_json = output_dir / f"{prefix}_plot_manifest.json"

    _plot_series(
        x=n_values,
        y=l2_values,
        ylabel="Norme L2 globale de l'image",
        title="Campagne image qualitative : norme L2 globale",
        output_png=l2_png,
        output_pdf=l2_pdf,
    )

    _plot_series(
        x=n_values,
        y=max_values,
        ylabel="Maximum global de l'image",
        title="Campagne image qualitative : maximum global",
        output_png=max_png,
        output_pdf=max_pdf,
    )

    _plot_series(
        x=n_values,
        y=node_counts,
        ylabel="Nombre de noeuds locaux exportés",
        title="Campagne image qualitative : nombre de noeuds",
        output_png=node_count_png,
        output_pdf=node_count_pdf,
    )

    _plot_series(
        x=n_values,
        y=cell_counts,
        ylabel="Nombre de cellules locales exportées",
        title="Campagne image qualitative : nombre de cellules",
        output_png=cell_count_png,
        output_pdf=cell_count_pdf,
    )

    _plot_series(
        x=n_values,
        y=inside_mass_fractions,
        ylabel="Masse relative dans le défaut",
        title="Campagne image qualitative : masse dans le défaut",
        output_png=defect_inside_mass_fraction_png,
        output_pdf=defect_inside_mass_fraction_pdf,
    )

    _plot_series(
        x=n_values_for_max_distances,
        y=max_distances_to_center,
        ylabel="Distance max image / centre défaut",
        title="Campagne image qualitative : localisation du maximum",
        output_png=defect_max_distance_png,
        output_pdf=defect_max_distance_pdf,
    )

    _plot_series(
        x=n_values_for_ratios,
        y=inside_outside_mean_ratios,
        ylabel="Moyenne intérieur / extérieur",
        title="Campagne image qualitative : contraste défaut",
        output_png=defect_inside_outside_ratio_png,
        output_pdf=defect_inside_outside_ratio_pdf,
    )

    manifest = {
        "kind": "material_qualitative_image_campaign_plot",
        "status": "ok",
        "summary_csv": str(summary_csv),
        "num_rows": len(rows),
        "n_values": [int(row["nx"]) for row in rows],
        "files": {
            "l2_norm_png": str(l2_png),
            "l2_norm_pdf": str(l2_pdf),
            "global_max_png": str(max_png),
            "global_max_pdf": str(max_pdf),
            "node_count_png": str(node_count_png),
            "node_count_pdf": str(node_count_pdf),
            "cell_count_png": str(cell_count_png),
            "cell_count_pdf": str(cell_count_pdf),
            "defect_inside_mass_fraction_png": str(defect_inside_mass_fraction_png),
            "defect_inside_mass_fraction_pdf": str(defect_inside_mass_fraction_pdf),
            "defect_max_distance_png": str(defect_max_distance_png),
            "defect_max_distance_pdf": str(defect_max_distance_pdf),
            "defect_inside_outside_ratio_png": str(defect_inside_outside_ratio_png),
            "defect_inside_outside_ratio_pdf": str(defect_inside_outside_ratio_pdf),
            "manifest_json": str(manifest_json),
        },
    }

    _write_json(manifest_json, manifest)

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot summary figures from a qualitative material image campaign CSV."
    )

    parser.add_argument("--summary-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="material_image_campaign")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    result = plot_material_qualitative_image_campaign_summary(
        summary_csv=args.summary_csv,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )

    if not args.quiet:
        print(
            "[MATERIAL QUALITATIVE IMAGE CAMPAIGN PLOT] OK "
            f"for summary={args.summary_csv}, "
            f"output_dir={args.output_dir}, "
            f"num_rows={result['num_rows']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
