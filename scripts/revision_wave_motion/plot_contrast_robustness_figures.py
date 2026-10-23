#!/usr/bin/env python
"""
Contrast-robustness figures of the accompanying article.

PURPOSE
-------
This script produces the two figures ``ContrastRobustnessLocalization.pdf``
and ``ContrastRobustnessEnergyRatio.pdf`` from the upstream image-quality
metric CSVs.

WHAT IT DRAWS
-------------
Localisation figure: localisation error, in defect radii, against the perturbation
amplitude ``c = 1 - contrast_factor``; one panel per material coefficient,
two constructions per panel.

Energy-ratio figure: ratio of the sensitivity energy deposited in the target zone,
residual interaction over classical, against ``c``; the five coefficients on
one pair of logarithmic axes.

VISUAL ENCODING. The published captions state "classical full-data
construction (blue circles)" and "residual-interaction construction (orange
diamonds)", and logarithmic axes for the energy-ratio figure. The encoding
below reproduces that specification exactly, so the figures stay consistent
with their captions. Blue/orange is also a colour-vision-deficiency-safe
two-series pair. Identity is additionally carried by marker shape, never by
colour alone.

UNITS.  ``localization_error`` is stored in the non-dimensional length units of
the model; the article reports defect radii, so the column is divided by
``R = 0.25``. Check against the published table: classical C66 at
delta = 0.0125 gives 5.5347 / 0.25 = 22.14, and residual interaction gives
3.3451 / 0.25 = 13.38, both exactly the published entries.

No PDE is solved. The input data are read only.

USAGE
-----
    PYTHONPATH=. python path/to/plot_contrast_robustness_figures.py \
        --output-dir path/to/output_directory --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

SWEEP_METRICS = "revision_analyses/point47_contrast_image_quality_metrics"
FREEZE_P1 = (
    "outputs/Priorité 1 — postprocess_single_defect_method_comparison.py/"
    "final_freeze_20260727/line_vs_box_method_comparison/single_defect_method_comparison.csv"
)

DEFECT_RADIUS = 0.25

# The reference point (contrast 0.75) comes from the reference campaign; its
# absence from the sweep is deliberate.
SWEEP_TAGS: tuple[tuple[str, float], ...] = (
    ("0p25", 0.25),
    ("0p50", 0.50),
    ("0p90", 0.90),
    ("0p95", 0.95),
    ("0p975", 0.975),
    ("0p9875", 0.9875),
)
FREEZE_CONTRAST = 0.75

COEFFICIENTS = ("rho", "C11", "C22", "C12", "C66")
LABELS = {
    "rho": r"$\rho$",
    "C11": r"$C_{11}$",
    "C22": r"$C_{22}$",
    "C12": r"$C_{12}$",
    "C66": r"$C_{66}$",
}

# Encoding fixed by the published captions (see module docstring).
SERIES = {
    "classical_full": {
        "label": "Classical, full data",
        "color": "#1f77b4",
        "marker": "o",
    },
    "residual_interaction": {
        "label": "Residual interaction",
        "color": "#ff7f0e",
        "marker": "D",
    },
}

FIELD_PREFIX = {"classical_full": "gradient", "residual_interaction": "residual_interaction"}


def _set_data_point_xticks(ax, deltas) -> None:
    """Tick the log x-axis at exactly the measured c values.

    The default LogLocator/LogFormatter pair crowds this axis, since the
    sweep spans under two decades and picks up unfiltered minor-tick labels
    (2, 3, 4, 6 x 10^-n). There are only seven distinct c values in the
    data; ticking exactly there is legible and shows the measured values
    rather than a generic log grid.
    """
    ax.xaxis.set_major_locator(mticker.FixedLocator(list(deltas)))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{d:g}" for d in deltas]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def read_sweep(repo: Path) -> dict[tuple[str, str, float], dict[str, float]]:
    """(modality, coefficient, delta) -> {localization_error, energy_fraction}."""
    out: dict[tuple[str, str, float], dict[str, float]] = {}
    for tag, contrast in SWEEP_TAGS:
        path = repo / SWEEP_METRICS / f"contrast_{tag}_line" / "gradient_image_quality_metrics.csv"
        if not path.exists():
            raise SystemExit(f"Missing metrics CSV: {path}")
        delta = 1.0 - contrast
        for row in csv.DictReader(path.open()):
            modality = row["combination"]
            if modality not in SERIES:
                continue
            field = row["expected_field"]
            expected = f"{FIELD_PREFIX[modality]}_"
            if not field.startswith(expected):
                continue
            coefficient = field[len(expected) :]
            if coefficient not in COEFFICIENTS:
                continue
            out[(modality, coefficient, delta)] = {
                "localization_error": float(row["localization_error"]),
                "energy_fraction_in_defect": float(row["energy_fraction_in_defect"]),
            }
    return out


def read_freeze(repo: Path) -> dict[tuple[str, str, float], dict[str, float]]:
    """The seventh point (contrast 0.75), line aperture, from the reference-campaign CSV."""
    path = repo / FREEZE_P1
    if not path.exists():
        raise SystemExit(f"Missing frozen P1 CSV: {path}")
    delta = 1.0 - FREEZE_CONTRAST
    out: dict[tuple[str, str, float], dict[str, float]] = {}
    # The reference-campaign CSV groups by (geometry, method, property) and
    # prefixes its metric columns with ``article_safe__``. This schema differs
    # from that of the sweep CSVs, hence the two separate readers.
    for row in csv.DictReader(path.open()):
        if (row.get("geometry") or "").strip() != "line":
            continue
        modality = (row.get("method") or "").strip()
        if modality not in SERIES:
            continue
        coefficient = (row.get("property") or "").strip()
        if coefficient not in COEFFICIENTS:
            continue
        out[(modality, coefficient, delta)] = {
            "localization_error": float(row["article_safe__localization_error"]),
            "energy_fraction_in_defect": float(row["article_safe__energy_fraction_in_defect"]),
        }
    return out


def figure_localisation(data, deltas, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(13.0, 2.9), sharex=True)
    for ax, coefficient in zip(axes, COEFFICIENTS, strict=True):
        for modality, style in SERIES.items():
            xs, ys = [], []
            for delta in deltas:
                entry = data.get((modality, coefficient, delta))
                if entry is None:
                    continue
                xs.append(delta)
                ys.append(entry["localization_error"] / DEFECT_RADIUS)
            ax.plot(
                xs,
                ys,
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linewidth=1.6,
                markersize=5.0,
                markerfacecolor="white",
                markeredgewidth=1.4,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(LABELS[coefficient], fontsize=11)
        ax.grid(True, which="major", linewidth=0.4, alpha=0.35)
        ax.tick_params(labelsize=8)
        _set_data_point_xticks(ax, deltas)
        ax.set_xlabel(r"$c = 1-\mathrm{contrast\ factor}$", fontsize=9)
    axes[0].set_ylabel("Localisation error\n(defect radii)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def figure_energy_ratio(data, deltas, out_path: Path) -> None:
    # The (6.4, 4.0) shape is kept: a wider aspect ratio looks stretched.
    # The printed footprint is reduced from the LaTeX side, at
    # \includegraphics width < 1.0\columnwidth; the y-axis title fontsize is
    # 12 (instead of 10) so it stays legible once the whole figure is
    # printed smaller.
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    # Sequential-by-identity is not appropriate here: the five coefficients are
    # categorical.  Fixed order, never cycled.
    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#000000")
    markers = ("o", "D", "s", "^", "v")
    for coefficient, color, marker in zip(COEFFICIENTS, palette, markers, strict=True):
        xs, ys = [], []
        for delta in deltas:
            classical = data.get(("classical_full", coefficient, delta))
            residual = data.get(("residual_interaction", coefficient, delta))
            if classical is None or residual is None:
                continue
            denominator = classical["energy_fraction_in_defect"]
            if denominator <= 0.0:
                continue
            xs.append(delta)
            ys.append(residual["energy_fraction_in_defect"] / denominator)
        ax.plot(
            xs,
            ys,
            marker=marker,
            color=color,
            label=LABELS[coefficient],
            linewidth=1.6,
            markersize=5.0,
            markerfacecolor="white",
            markeredgewidth=1.4,
        )
    ax.axhline(1.0, color="0.35", linewidth=1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$c = 1-\mathrm{contrast\ factor}$", fontsize=10)
    ax.set_ylabel("Energy fraction ratio\n(residual interaction / classical)", fontsize=12)
    ax.grid(True, which="major", linewidth=0.4, alpha=0.35)
    ax.tick_params(labelsize=9)
    _set_data_point_xticks(ax, deltas)
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = args.output_dir
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    data = read_sweep(args.repo_root)
    data.update(read_freeze(args.repo_root))
    deltas = sorted({key[2] for key in data})

    loc_path = out / "ContrastRobustnessLocalization.pdf"
    energy_path = out / "ContrastRobustnessEnergyRatio.pdf"
    figure_localisation(data, deltas, loc_path)
    figure_energy_ratio(data, deltas, energy_path)

    manifest = {
        "analysis_date_utc": datetime.now(timezone.utc).isoformat(),
        "format_version": "contrast_robustness_figures_v1",
        "purpose": "Reproduce Figures 4 and 5.",
        "defect_radius": DEFECT_RADIUS,
        "deltas": deltas,
        "encoding_source": "published captions (blue circles / orange diamonds, log axes)",
        "figures": [str(loc_path), str(energy_path)],
        "num_points": len(data),
    }
    (out / "contrast_robustness_figures_manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"deltas covered : {deltas}")
    print(f"data points    : {len(data)}")
    print(f"written        : {loc_path}")
    print(f"                 {energy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
