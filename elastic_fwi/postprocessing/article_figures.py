from __future__ import annotations

"""Assembly of the figures of the accompanying Wave Motion article.

The module only consumes existing campaign and post-processing outputs.  It
never launches a direct/adjoint solve and never modifies scientific data.
Dense DG0 maps are displayed either as vector point collections or as an
area-weighted raster layer inside a PDF with vector text/annotations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from elastic_fwi.postprocessing.image_metric_robustness import (
    CampaignMapSource,
    discover_campaign_maps,
    load_campaign_map,
)
from elastic_fwi.postprocessing.signed_sensitivity_fields import (
    DefectCircle,
    load_vtk_cell_field,
)

FORMAT_VERSION = "wave_motion_article_figures_v1"
DEFAULT_GROUPS = (
    "geometry",
    "wave_interaction",
    "receiver_summary",
    "normalized_maps",
    "signed_maps",
    "image_metric_summary",
    "c66_multipolar",
)
DEFAULT_MODALITIES = (
    "classical_full",
    "standard_reflection_only",
    "standard_transmission_only",
    "residual_interaction",
)
DEFAULT_PARAMETERS = ("rho", "C11", "C22", "C12", "C66")
DEFAULT_SIGNED_MODALITIES = ("classical_full", "residual_interaction")

COLOR_CONVENTIONS = {
    "signed_field": {
        "cmap": "RdBu_r",
        "normalization": "symmetric about zero using max(abs(field)) within the displayed ROI",
        "label": "Signed sensitivity density (qualitative first-iteration field)",
    },
    "absolute_normalized_image": {
        "cmap": "viridis",
        "normalization": "abs(S_q)/max_Omega(abs(S_q)); range [0,1]",
        "label": "Normalized absolute sensitivity (qualitative)",
    },
    "scalar_metric": {
        "cmap": "none",
        "normalization": "native metric units",
        "label": "Scalar diagnostic metric",
    },
}


@dataclass(frozen=True)
class CampaignInfo:
    root: Path
    manifest_path: Path
    campaign_id: str
    geometry_kind: str
    config: Mapping[str, Any]
    payload: Mapping[str, Any]


@dataclass
class FigureRecord:
    figure_id: str
    group: str
    title: str
    status: str
    reason: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    caption: str = ""
    conventions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "group": self.group,
            "title": self.title,
            "status": self.status,
            "reason": self.reason,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "caption": self.caption,
            "conventions": dict(self.conventions),
        }


@dataclass(frozen=True)
class ArticleFigureConfig:
    input_root: Path
    output_dir: Path
    campaign_dirs: tuple[Path, ...] = ()
    figure_groups: tuple[str, ...] = DEFAULT_GROUPS
    roi_spec: str = "auto"
    formats: tuple[str, ...] = ("pdf", "png")
    dpi: int = 300
    overwrite: bool = False
    quiet: bool = False
    strict: bool = False
    manifest_name: str = "article_figures_manifest.json"
    caption_summary_name: str = "article_figures_summary.md"
    modalities: tuple[str, ...] = DEFAULT_MODALITIES
    parameters: tuple[str, ...] = DEFAULT_PARAMETERS
    signed_modalities: tuple[str, ...] = DEFAULT_SIGNED_MODALITIES
    field_rendering: str = "auto"
    max_vector_points: int = 50_000
    map_grid_size: int = 500


@dataclass(frozen=True)
class ArticleFigureResult:
    status: str
    records: tuple[FigureRecord, ...]
    manifest_path: Path
    summary_path: Path
    exit_code: int


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_identifier(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_") or "item"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def resolve_manifest_path(raw: str | Path, *, manifest_path: Path, campaign_root: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (
        manifest_path.parent / path,
        campaign_root / path,
        Path.cwd() / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _campaign_id(payload: Mapping[str, Any], root: Path) -> str:
    for key in ("campaign_id", "case_name", "name"):
        value = payload.get(key)
        if value:
            return sanitize_identifier(str(value))
    return sanitize_identifier(root.name)


def _geometry_kind(config: Mapping[str, Any], root: Path) -> str:
    value = str(config.get("receiver_array_geometry", "")).lower()
    if "box" in value:
        return "box"
    if "line" in value:
        return "line"
    name = root.name.lower()
    if "box" in name:
        return "box"
    if "line" in name:
        return "line"
    return "unknown"


def discover_campaigns(config: ArticleFigureConfig) -> tuple[CampaignInfo, ...]:
    manifests: list[Path] = []
    if config.campaign_dirs:
        for root in config.campaign_dirs:
            path = Path(root).expanduser().resolve() / "gradient_combination_campaign_manifest.json"
            if path.is_file():
                manifests.append(path)
    else:
        manifests.extend(sorted(config.input_root.rglob("gradient_combination_campaign_manifest.json")))
    output: list[CampaignInfo] = []
    seen: set[Path] = set()
    for manifest in manifests:
        manifest = manifest.resolve()
        if manifest in seen:
            continue
        seen.add(manifest)
        payload = read_json(manifest)
        raw_config = payload.get("config", {})
        campaign_config = dict(raw_config) if isinstance(raw_config, Mapping) else {}
        output.append(
            CampaignInfo(
                root=manifest.parent,
                manifest_path=manifest,
                campaign_id=_campaign_id(payload, manifest.parent),
                geometry_kind=_geometry_kind(campaign_config, manifest.parent),
                config=campaign_config,
                payload=payload,
            )
        )
    return tuple(sorted(output, key=lambda item: (item.geometry_kind, item.campaign_id)))


def _defects(campaign: CampaignInfo) -> tuple[DefectCircle, ...]:
    cfg = campaign.config
    if all(key in cfg for key in ("defect_center_x", "defect_center_y", "defect_radius")):
        try:
            radius = float(cfg["defect_radius"])
            if radius > 0:
                return (
                    DefectCircle(
                        center_x=float(cfg["defect_center_x"]),
                        center_y=float(cfg["defect_center_y"]),
                        radius=radius,
                        name="defect",
                    ),
                )
        except (TypeError, ValueError):
            pass
    return ()


def _segment_points(config: Mapping[str, Any], prefix: str) -> np.ndarray | None:
    keys = (
        f"{prefix}_start_x",
        f"{prefix}_start_y",
        f"{prefix}_end_x",
        f"{prefix}_end_y",
    )
    if not all(key in config for key in keys):
        return None
    try:
        n = int(config.get(f"{prefix}_num_elements", 2))
        n = max(n, 2)
        return np.column_stack(
            (
                np.linspace(float(config[keys[0]]), float(config[keys[2]]), n),
                np.linspace(float(config[keys[1]]), float(config[keys[3]]), n),
            )
        )
    except (TypeError, ValueError):
        return None


def source_points(campaign: CampaignInfo) -> np.ndarray | None:
    return _segment_points(campaign.config, "source_array")


def receiver_points(campaign: CampaignInfo) -> np.ndarray | None:
    cfg = campaign.config
    geometry = campaign.geometry_kind
    if geometry == "box" and all(
        key in cfg for key in ("receiver_box_xmin", "receiver_box_xmax", "receiver_box_ymin", "receiver_box_ymax")
    ):
        try:
            xmin = float(cfg["receiver_box_xmin"])
            xmax = float(cfg["receiver_box_xmax"])
            ymin = float(cfg["receiver_box_ymin"])
            ymax = float(cfg["receiver_box_ymax"])
            nl = max(int(cfg.get("receiver_box_left_num_elements", 21)), 2)
            nt = max(int(cfg.get("receiver_box_top_num_elements", 21)), 2)
            nr = max(int(cfg.get("receiver_box_right_num_elements", 21)), 2)
            left = np.column_stack((np.full(nl, xmin), np.linspace(ymin, ymax, nl)))
            top = np.column_stack((np.linspace(xmin, xmax, nt), np.full(nt, ymax)))
            right = np.column_stack((np.full(nr, xmax), np.linspace(ymin, ymax, nr)))
            return np.vstack((left, top[1:], right[-2::-1]))
        except (TypeError, ValueError):
            return None
    return _segment_points(cfg, "receiver_array")


def _all_annotation_points(campaigns: Sequence[CampaignInfo]) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for campaign in campaigns:
        for points in (source_points(campaign), receiver_points(campaign)):
            if points is not None and points.size:
                blocks.append(points)
        for defect in _defects(campaign):
            blocks.append(
                np.asarray(
                    [
                        [defect.center_x - defect.radius, defect.center_y - defect.radius],
                        [defect.center_x + defect.radius, defect.center_y + defect.radius],
                    ],
                    dtype=float,
                )
            )
    return np.vstack(blocks) if blocks else np.empty((0, 2), dtype=float)


def parse_roi(spec: str, campaigns: Sequence[CampaignInfo]) -> tuple[float, float, float, float]:
    text = str(spec).strip().lower()
    if text == "full":
        xmax = max(float(c.config.get("domain_length_x", 1.0)) for c in campaigns) if campaigns else 1.0
        ymax = max(float(c.config.get("domain_length_y", 1.0)) for c in campaigns) if campaigns else 1.0
        return (0.0, xmax, 0.0, ymax)
    if text == "auto":
        points = _all_annotation_points(campaigns)
        if points.size == 0:
            return parse_roi("full", campaigns)
        xmin, ymin = np.min(points, axis=0)
        xmax, ymax = np.max(points, axis=0)
        span = max(xmax - xmin, ymax - ymin, 1.0)
        padding = 0.10 * span
        return (float(xmin - padding), float(xmax + padding), float(ymin - padding), float(ymax + padding))
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("ROI must be auto, full, or xmin,xmax,ymin,ymax.")
    xmin, xmax, ymin, ymax = (float(value) for value in parts)
    if not all(math.isfinite(value) for value in (xmin, xmax, ymin, ymax)) or xmin >= xmax or ymin >= ymax:
        raise ValueError("Explicit ROI coordinates are invalid.")
    return (xmin, xmax, ymin, ymax)


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _prepare_output_dir(config: ArticleFigureConfig) -> None:
    root = config.output_dir
    if root.exists() and any(root.iterdir()):
        if not config.overwrite:
            raise FileExistsError(f"Output directory is not empty: {root}. Use --overwrite.")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: Any, base: Path, formats: Sequence[str], dpi: int) -> list[Path]:
    outputs: list[Path] = []
    for suffix in formats:
        path = base.with_suffix(f".{suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
        outputs.append(path)
    return outputs


def _annotate_setup(ax: Any, campaign: CampaignInfo, *, label_arrays: bool = True) -> None:
    from matplotlib.patches import Circle

    src = source_points(campaign)
    rec = receiver_points(campaign)
    if src is not None:
        ax.plot(src[:, 0], src[:, 1], color="tab:red", linewidth=1.2, marker="o", markersize=2.2)
        if label_arrays:
            ax.text(float(np.mean(src[:, 0])), float(np.mean(src[:, 1])), r"  $\Gamma_1$ / source", color="tab:red", va="bottom")
    if rec is not None:
        ax.plot(rec[:, 0], rec[:, 1], color="tab:blue", linewidth=1.2, marker="s", markersize=2.0)
        if label_arrays:
            ax.text(float(np.mean(rec[:, 0])), float(np.mean(rec[:, 1])), r"  $\Gamma_2$", color="tab:blue", va="bottom")
    for defect in _defects(campaign):
        ax.add_patch(Circle((defect.center_x, defect.center_y), defect.radius, fill=False, edgecolor="black", linewidth=1.2, linestyle="--"))
        ax.plot(defect.center_x, defect.center_y, marker="+", color="black", markersize=6)
        ax.text(defect.center_x + defect.radius, defect.center_y + defect.radius, "defect", fontsize=7.5)


def make_geometry_figure(campaigns: Sequence[CampaignInfo], *, roi: tuple[float, float, float, float], config: ArticleFigureConfig) -> FigureRecord:
    record = FigureRecord("figure_01_geometry_setup", "geometry", "Geometry and acquisition setup", "skipped")
    if not campaigns:
        record.reason = "No campaign manifest was discovered."
        return record
    plt = _configure_matplotlib()
    fig, axes = plt.subplots(1, len(campaigns), figsize=(6.0 * len(campaigns), 5.4), squeeze=False, constrained_layout=True)
    from matplotlib.patches import Rectangle

    for ax, campaign in zip(axes.flat, campaigns, strict=True):
        lx = float(campaign.config.get("domain_length_x", roi[1]))
        ly = float(campaign.config.get("domain_length_y", roi[3]))
        ax.add_patch(Rectangle((0.0, 0.0), lx, ly, fill=False, edgecolor="0.25", linewidth=1.0))
        _annotate_setup(ax, campaign)
        ax.add_patch(Rectangle((roi[0], roi[2]), roi[1] - roi[0], roi[3] - roi[2], fill=False, edgecolor="tab:green", linewidth=1.0, linestyle=":"))
        ax.set_xlim(0.0, lx)
        ax.set_ylim(0.0, ly)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{campaign.geometry_kind}: {campaign.campaign_id}")
    fig.suptitle("Computational domain, defect, source array and receiver aperture")
    outputs = _save_figure(fig, config.output_dir / record.figure_id, config.formats, config.dpi)
    plt.close(fig)
    record.status = "ok"
    record.inputs = [str(c.manifest_path) for c in campaigns]
    record.outputs = [str(path) for path in outputs]
    record.caption = (
        "Computational and acquisition geometry. The dashed circle marks the defect; "
        r"$\Gamma_1$ denotes the source/reflection array and $\Gamma_2$ the transmission aperture. "
        "The dotted rectangle is the common region of interest used in all comparable map panels."
    )
    record.conventions = {"roi": list(roi), "axes": "same physical coordinates and equal aspect"}
    return record


def _weighted_raster(centers: np.ndarray, values: np.ndarray, weights: np.ndarray, roi: tuple[float, float, float, float], grid_size: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    xmin, xmax, ymin, ymax = roi
    mask = (
        (centers[:, 0] >= xmin)
        & (centers[:, 0] <= xmax)
        & (centers[:, 1] >= ymin)
        & (centers[:, 1] <= ymax)
    )
    if not np.any(mask):
        raise ValueError("No field samples fall inside the common ROI.")
    x = centers[mask, 0]
    y = centers[mask, 1]
    v = values[mask]
    w = weights[mask]
    numerator, xedges, yedges = np.histogram2d(x, y, bins=grid_size, range=((xmin, xmax), (ymin, ymax)), weights=w * v)
    denominator, _, _ = np.histogram2d(x, y, bins=(xedges, yedges), weights=w)
    raster = np.full(numerator.shape, np.nan, dtype=float)
    valid = denominator > 0.0
    raster[valid] = numerator[valid] / denominator[valid]
    return raster.T, (xmin, xmax, ymin, ymax)


def _draw_field(ax: Any, centers: np.ndarray, values: np.ndarray, weights: np.ndarray, *, roi: tuple[float, float, float, float], cmap: str, vmin: float, vmax: float, config: ArticleFigureConfig) -> tuple[Any, str]:
    mask = (
        (centers[:, 0] >= roi[0])
        & (centers[:, 0] <= roi[1])
        & (centers[:, 1] >= roi[2])
        & (centers[:, 1] <= roi[3])
    )
    n = int(np.count_nonzero(mask))
    mode = config.field_rendering
    if mode == "auto":
        mode = "vector" if n <= config.max_vector_points else "rasterized"
    if mode == "vector":
        size = max(0.2, min(9.0, 50_000.0 / max(n, 1)))
        artist = ax.scatter(centers[mask, 0], centers[mask, 1], c=values[mask], s=size, marker="s", linewidths=0.0, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=False)
        return artist, "vector_scatter"
    raster, extent = _weighted_raster(centers, values, weights, roi, config.map_grid_size)
    artist = ax.imshow(raster, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal", rasterized=True)
    return artist, f"area_weighted_raster_{config.map_grid_size}x{config.map_grid_size}"


def _field_sources(campaign: CampaignInfo, *, modalities: Sequence[str], parameters: Sequence[str], raw_signed: bool) -> list[tuple[str, str, str, Path]]:
    section_name = "raw_signed_validation" if raw_signed else "validation"
    section = campaign.payload.get(section_name, {})
    found = section.get("found", {}) if isinstance(section, Mapping) else {}
    output: list[tuple[str, str, str, Path]] = []
    if not isinstance(found, Mapping):
        return output
    for modality in modalities:
        mapping = found.get(modality, {})
        if not isinstance(mapping, Mapping):
            continue
        for field_name, raw_path in mapping.items():
            parameter = str(field_name)
            for prefix in ("residual_interaction_", "gradient_", "kernel_"):
                if parameter.startswith(prefix):
                    parameter = parameter[len(prefix):]
            if parameter not in parameters:
                continue
            path = resolve_manifest_path(raw_path, manifest_path=campaign.manifest_path, campaign_root=campaign.root)
            if path.exists():
                output.append((modality, parameter, str(field_name), path))
    return output


def make_map_board(campaign: CampaignInfo, *, roi: tuple[float, float, float, float], config: ArticleFigureConfig, signed: bool) -> FigureRecord:
    group = "signed_maps" if signed else "normalized_maps"
    figure_id = f"figure_{'05' if signed else '04'}_{'signed_sensitivity' if signed else 'sensitivity_maps'}_{campaign.geometry_kind}"
    title = "Signed sensitivity fields" if signed else "Normalized absolute sensitivity maps"
    record = FigureRecord(figure_id, group, title, "skipped")
    modalities = config.signed_modalities if signed else config.modalities
    sources = _field_sources(campaign, modalities=modalities, parameters=config.parameters, raw_signed=signed)
    source_map = {(m, p): (f, path) for m, p, f, path in sources}
    present_modalities = [m for m in modalities if any((m, p) in source_map for p in config.parameters)]
    present_parameters = [p for p in config.parameters if any((m, p) in source_map for m in present_modalities)]
    if not present_modalities or not present_parameters:
        record.reason = f"No {'raw signed' if signed else 'article-safe normalized'} map was found in {campaign.manifest_path}."
        return record
    plt = _configure_matplotlib()
    fig, axes = plt.subplots(len(present_modalities), len(present_parameters), figsize=(3.0 * len(present_parameters), 2.75 * len(present_modalities)), squeeze=False, constrained_layout=True)
    renderings: set[str] = set()
    inputs: list[str] = []
    last_artist = None
    for row, modality in enumerate(present_modalities):
        for col, parameter in enumerate(present_parameters):
            ax = axes[row, col]
            item = source_map.get((modality, parameter))
            if item is None:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                continue
            field_name, path = item
            loaded = load_vtk_cell_field(path, field_name)
            values = np.asarray(loaded.values, dtype=float)
            if not np.all(np.isfinite(values)):
                ax.set_axis_off()
                ax.text(0.5, 0.5, "non-finite field", ha="center", va="center", transform=ax.transAxes)
                continue
            if signed:
                roi_mask = (
                    (loaded.centers[:, 0] >= roi[0]) & (loaded.centers[:, 0] <= roi[1])
                    & (loaded.centers[:, 1] >= roi[2]) & (loaded.centers[:, 1] <= roi[3])
                )
                amplitude = float(np.max(np.abs(values[roi_mask]))) if np.any(roi_mask) else float(np.max(np.abs(values)))
                amplitude = amplitude if amplitude > 0.0 else 1.0
                display_values = values
                cmap, vmin, vmax = "RdBu_r", -amplitude, amplitude
            else:
                magnitude = np.abs(values)
                scale = float(np.max(magnitude))
                if scale <= 0.0:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "zero field", ha="center", va="center", transform=ax.transAxes)
                    continue
                display_values = magnitude / scale
                cmap, vmin, vmax = "viridis", 0.0, 1.0
            artist, rendering = _draw_field(ax, loaded.centers, display_values, loaded.cell_measures, roi=roi, cmap=cmap, vmin=vmin, vmax=vmax, config=config)
            last_artist = artist
            renderings.add(rendering)
            inputs.append(str(path))
            _annotate_setup(ax, campaign, label_arrays=False)
            ax.set_xlim(roi[0], roi[1])
            ax.set_ylim(roi[2], roi[3])
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(parameter)
            if col == 0:
                ax.set_ylabel(modality)
            else:
                ax.set_yticklabels([])
            if row < len(present_modalities) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("x")
    if last_artist is not None:
        label = "signed value" if signed else r"$|S_q|/\max_\Omega |S_q|$"
        fig.colorbar(last_artist, ax=axes.ravel().tolist(), shrink=0.82, label=label)
    fig.suptitle(f"{title} — {campaign.geometry_kind} — common ROI")
    outputs = _save_figure(fig, config.output_dir / figure_id, config.formats, config.dpi)
    plt.close(fig)
    record.status = "ok"
    record.inputs = sorted(set(inputs))
    record.outputs = [str(path) for path in outputs]
    record.caption = (
        ("Signed first-iteration sensitivity fields. A zero-centred diverging scale is used independently for each parameter panel; signs are not removed." if signed else "Qualitative normalized sensitivity maps, defined as $I_q=|S_q|/\\max_\\Omega|S_q|$. All panels use the same physical ROI and the common range $[0,1]$; the maps are not quantitative parameter updates.")
        + r" The dashed circle marks the defect and the source/receiver apertures are overlaid."
    )
    record.conventions = {
        "roi": list(roi),
        "field_kind": "signed" if signed else "absolute_normalized",
        "color_convention": COLOR_CONVENTIONS["signed_field" if signed else "absolute_normalized_image"],
        "rendering": sorted(renderings),
    }
    return record


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan
    return value


def _find_first(search_roots: Sequence[Path], filename: str) -> Path | None:
    candidates: list[Path] = []
    for root in search_roots:
        if root.exists():
            candidates.extend(path for path in root.rglob(filename) if path.is_file())
    return sorted(set(path.resolve() for path in candidates))[0] if candidates else None


def _search_roots(config: ArticleFigureConfig) -> tuple[Path, ...]:
    roots = [config.input_root]
    parent = config.input_root.parent
    if parent not in roots:
        roots.append(parent)
    return tuple(roots)


def make_receiver_summary(config: ArticleFigureConfig) -> FigureRecord:
    record = FigureRecord("figure_03_reflection_transmission_receiver_summary", "receiver_summary", "Receiver phase, delay and amplitude summary", "skipped")
    path = _find_first(_search_roots(config), "receiver_phase_spectrum_summary.csv")
    if path is None:
        record.reason = "receiver_phase_spectrum_summary.csv was not found."
        return record
    rows = _read_csv(path)
    if not rows:
        record.reason = f"Receiver summary is empty: {path}"
        return record
    plt = _configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    styles: dict[tuple[str, str], dict[str, Any]] = {}
    families = sorted({row.get("residual_family", "unknown") for row in rows})
    components = sorted({row.get("component", "?") for row in rows})
    markers = ("o", "s", "^", "d")
    for i, family in enumerate(families):
        for j, component in enumerate(components):
            styles[(family, component)] = {"marker": markers[(i * len(components) + j) % len(markers)], "label": f"{family}, {component}"}
    for key, style in styles.items():
        family, component = key
        group = [row for row in rows if row.get("residual_family") == family and row.get("component") == component]
        group.sort(key=lambda row: int(float(row.get("receiver_id", 0))))
        x = [int(float(row.get("receiver_id", 0))) for row in group]
        series = (
            ("observed_dominant_frequency", axes[0, 0]),
            ("cross_correlation_lag_time", axes[0, 1]),
            ("cross_correlation_max", axes[1, 0]),
            ("observed_temporal_rms", axes[1, 1]),
        )
        for column, ax in series:
            y = [_float(row, column) for row in group]
            ax.plot(x, y, linewidth=1.0, markersize=3.0, **style)
    labels = (
        (axes[0, 0], "Observed dominant frequency", "frequency"),
        (axes[0, 1], "Cross-correlation delay", "time"),
        (axes[1, 0], "Maximum normalized correlation", "coefficient"),
        (axes[1, 1], "Observed temporal RMS amplitude", "RMS amplitude"),
    )
    for ax, title, ylabel in labels:
        ax.set_title(title)
        ax.set_xlabel("receiver id")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    handles, labels_text = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_text, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncols=max(1, len(handles)))
    fig.suptitle("Receiver diagnostics: reflection and transmission")
    outputs = _save_figure(fig, config.output_dir / record.figure_id, config.formats, config.dpi)
    plt.close(fig)
    record.status = "ok"
    record.inputs = [str(path)]
    record.outputs = [str(output) for output in outputs]
    record.caption = (
        "Receiver-domain comparison of reflection and transmission traces. Panels report the measured dominant frequency, correlation-based delay, maximum normalized correlation, and absolute observed RMS amplitude for each receiver and displacement component. Phase or amplitude ratios flagged as unreliable by the preceding diagnostic are not substituted by artificial values."
    )
    record.conventions = {"metric_units": "native CSV units", "x_axis": "receiver identifier"}
    return record


def make_metric_summary(config: ArticleFigureConfig) -> FigureRecord:
    record = FigureRecord("figure_06_image_metric_robustness", "image_metric_summary", "Image-metric ranking robustness", "skipped")
    path = _find_first(_search_roots(config), "image_metric_robustness_summary.json")
    if path is None:
        record.reason = "image_metric_robustness_summary.json was not found."
        return record
    payload = read_json(path)
    groups = payload.get("groups", [])
    if not isinstance(groups, list) or not groups:
        record.reason = f"No robustness group is available in {path}."
        return record
    axes_names = ("threshold", "zone", "joint", "ties")
    status_keys = {
        "threshold": "threshold_robustness_status",
        "zone": "zone_robustness_status",
        "joint": "joint_robustness_status",
        "ties": "tie_stability_status",
    }
    severity = {"ok": 0.0, "not_applicable": 0.25, "warning": 0.5, "failed": 1.0, "fail": 1.0}
    parameters = [str(group.get("parameter_name", group.get("ranking_group", f"group_{i}"))) for i, group in enumerate(groups)]
    matrix = np.empty((len(groups), len(axes_names)), dtype=float)
    labels: list[list[str]] = []
    for i, group in enumerate(groups):
        row_labels: list[str] = []
        for j, axis_name in enumerate(axes_names):
            status = str(group.get(status_keys[axis_name], "unknown")).lower()
            matrix[i, j] = severity.get(status, 0.75)
            row_labels.append(status.upper())
        labels.append(row_labels)
    plt = _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.65 * len(groups) + 2.0)), constrained_layout=True)
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(["#4daf4a", "#bdbdbd", "#ffbf00", "#d73027"])
    norm = BoundaryNorm([-0.01, 0.125, 0.375, 0.75, 1.01], cmap.N)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(axes_names)), labels=[name.capitalize() for name in axes_names])
    ax.set_yticks(range(len(parameters)), labels=parameters)
    for i in range(len(groups)):
        for j in range(len(axes_names)):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7.5, color="black")
    ax.set_title(f"Image-metric ranking robustness — global status: {payload.get('status', 'unknown')}")
    cbar = fig.colorbar(image, ax=ax, ticks=[0.05, 0.25, 0.5, 0.9], shrink=0.8)
    cbar.ax.set_yticklabels(["OK", "N/A", "WARNING", "FAIL"])
    outputs = _save_figure(fig, config.output_dir / record.figure_id, config.formats, config.dpi)
    plt.close(fig)
    record.status = "ok"
    record.inputs = [str(path)]
    record.outputs = [str(output) for output in outputs]
    record.caption = (
        "Robustness of modality rankings derived from qualitative normalized images. Each cell reports the status under threshold changes, target-zone changes, joint changes, and tie-set changes. A failed status indicates that a universal strict ranking is not supported for that parameter; it does not invalidate the underlying qualitative map."
    )
    record.conventions = {"status_colors": {"ok": "green", "not_applicable": "gray", "warning": "amber", "failed": "red"}}
    return record


def _copy_existing_figure(source: Path, destination_base: Path, formats: Sequence[str]) -> list[Path]:
    outputs: list[Path] = []
    for suffix in formats:
        candidate = source.with_suffix(f".{suffix}")
        if not candidate.is_file():
            continue
        destination = destination_base.with_suffix(f".{suffix}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, destination)
        outputs.append(destination)
    return outputs


def make_wave_interaction(config: ArticleFigureConfig) -> FigureRecord:
    record = FigureRecord("figure_02_wave_defect_interaction", "wave_interaction", "Wave-defect interaction", "skipped")
    roots = _search_roots(config)
    pdf = _find_first(roots, "scattered_wavefield_snapshots.pdf")
    png = _find_first(roots, "scattered_wavefield_snapshots.png")
    source = pdf or png
    if source is None:
        record.reason = "No scattered_wavefield_snapshots PDF/PNG produced by the reconstruction diagnostic was found."
        return record
    outputs = _copy_existing_figure(source, config.output_dir / record.figure_id, config.formats)
    if not outputs:
        record.reason = "The discovered wave-interaction figure does not match the requested formats."
        return record
    record.status = "ok"
    record.inputs = sorted({str(path) for path in (pdf, png) if path is not None})
    record.outputs = [str(path) for path in outputs]
    record.caption = (
        "Selected snapshots of the physical scattered displacement, defined as the defective-field displacement minus the healthy-reference displacement. Signed displacement components use zero-centred diverging scales, while the vector norm uses a non-negative sequential scale."
    )
    record.conventions = {"provenance": "copied without modification from scattered-wavefield reconstruction output"}
    return record


def make_c66_figures(config: ArticleFigureConfig) -> list[FigureRecord]:
    roots = _search_roots(config)
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.rglob("C66_multipolar_*.pdf"))
    candidates = [path.resolve() for path in sorted(set(candidates)) if "_summary" not in path.name]
    if not candidates:
        return [FigureRecord("figure_07_C66_multipolar", "c66_multipolar", "C66 multipolar analysis", "skipped", reason="No C66 multipolar PDF was found.")]
    records: list[FigureRecord] = []
    for index, source in enumerate(candidates, start=1):
        stem = sanitize_identifier(source.stem.replace("C66_multipolar_", ""))
        figure_id = f"figure_07_{index:02d}_C66_multipolar_{stem}"
        outputs = _copy_existing_figure(source, config.output_dir / figure_id, config.formats)
        record = FigureRecord(figure_id, "c66_multipolar", f"C66 multipolar pattern: {stem}", "ok" if outputs else "skipped")
        record.inputs = [str(source)]
        record.outputs = [str(path) for path in outputs]
        record.reason = "" if outputs else "Requested output formats were unavailable."
        record.caption = (
            "Signed normalized $C_{66}$ multipolar-pattern diagnostic, including radial profiles, annular energy, angular Fourier content, $180^\\circ$ symmetry and dominant-lobe orientation. The figure characterizes morphology and does not by itself establish quantitative parameter identifiability."
        )
        record.conventions = {"provenance": "copied without modification from priority-7 output"}
        records.append(record)
    return records


def _input_inventory(records: Sequence[FigureRecord]) -> list[dict[str, Any]]:
    paths = sorted({Path(raw) for record in records for raw in record.inputs if raw})
    output: list[dict[str, Any]] = []
    for path in paths:
        item: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            item.update({"size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
        output.append(item)
    return output


def _write_summary(path: Path, records: Sequence[FigureRecord], *, roi: tuple[float, float, float, float]) -> None:
    lines = [
        "# Wave Motion article figures",
        "",
        f"Common ROI: `x=[{roi[0]:g}, {roi[1]:g}]`, `y=[{roi[2]:g}, {roi[3]:g}]`.",
        "",
        "Dense DG0 field layers may be area-weighted and rasterized inside the PDF when the mesh is too large for a practical vector point collection; text, axes, contours and annotations remain vector objects. This rendering choice is recorded in the manifest and does not alter the numerical data.",
        "",
    ]
    for record in records:
        lines.extend((f"## {record.figure_id} — {record.title}", "", f"Status: **{record.status}**."))
        if record.reason:
            lines.extend(("", f"Reason: {record.reason}"))
        if record.caption:
            lines.extend(("", "Proposed autonomous caption:", "", record.caption))
        if record.outputs:
            lines.extend(("", "Outputs:", *[f"- `{path}`" for path in record.outputs]))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_article_figures(config: ArticleFigureConfig, *, command: Sequence[str] = ()) -> ArticleFigureResult:
    if config.dpi <= 0 or config.map_grid_size < 32 or config.max_vector_points < 1:
        raise ValueError("dpi, map_grid_size, and max_vector_points must be positive.")
    if config.field_rendering not in {"auto", "vector", "rasterized"}:
        raise ValueError("field_rendering must be auto, vector, or rasterized.")
    unsupported_formats = set(config.formats) - {"pdf", "png"}
    if unsupported_formats:
        raise ValueError(f"Unsupported figure formats: {sorted(unsupported_formats)}")
    unsupported_groups = set(config.figure_groups) - set(DEFAULT_GROUPS)
    if unsupported_groups:
        raise ValueError(f"Unsupported figure groups: {sorted(unsupported_groups)}")

    _prepare_output_dir(config)
    campaigns = discover_campaigns(config)
    roi = parse_roi(config.roi_spec, campaigns)
    records: list[FigureRecord] = []

    if "geometry" in config.figure_groups:
        records.append(make_geometry_figure(campaigns, roi=roi, config=config))
    if "wave_interaction" in config.figure_groups:
        records.append(make_wave_interaction(config))
    if "receiver_summary" in config.figure_groups:
        records.append(make_receiver_summary(config))
    if "normalized_maps" in config.figure_groups:
        if campaigns:
            for campaign in campaigns:
                records.append(make_map_board(campaign, roi=roi, config=config, signed=False))
        else:
            records.append(FigureRecord("figure_04_sensitivity_maps", "normalized_maps", "Normalized maps", "skipped", reason="No campaign was discovered."))
    if "signed_maps" in config.figure_groups:
        if campaigns:
            for campaign in campaigns:
                records.append(make_map_board(campaign, roi=roi, config=config, signed=True))
        else:
            records.append(FigureRecord("figure_05_signed_sensitivity", "signed_maps", "Signed maps", "skipped", reason="No campaign was discovered."))
    if "image_metric_summary" in config.figure_groups:
        records.append(make_metric_summary(config))
    if "c66_multipolar" in config.figure_groups:
        records.extend(make_c66_figures(config))

    num_ok = sum(record.status == "ok" for record in records)
    num_skipped = sum(record.status == "skipped" for record in records)
    strict_failure = config.strict and num_skipped > 0
    status = "failed" if strict_failure or num_ok == 0 else ("partial" if num_skipped else "ok")
    exit_code = 2 if strict_failure or num_ok == 0 else 0

    manifest_path = config.output_dir / config.manifest_name
    summary_path = config.output_dir / config.caption_summary_name
    manifest = {
        "kind": "wave_motion_article_figures_manifest",
        "format_version": FORMAT_VERSION,
        "analysis_date_utc": now_utc(),
        "status": status,
        "command": list(command),
        "configuration": {
            "input_root": str(config.input_root),
            "output_dir": str(config.output_dir),
            "campaign_dirs": [str(path) for path in config.campaign_dirs],
            "figure_groups": list(config.figure_groups),
            "roi_spec": config.roi_spec,
            "common_roi": list(roi),
            "formats": list(config.formats),
            "dpi": config.dpi,
            "strict": config.strict,
            "modalities": list(config.modalities),
            "parameters": list(config.parameters),
            "signed_modalities": list(config.signed_modalities),
            "field_rendering": config.field_rendering,
            "max_vector_points": config.max_vector_points,
            "map_grid_size": config.map_grid_size,
        },
        "campaigns": [
            {
                "campaign_id": campaign.campaign_id,
                "geometry_kind": campaign.geometry_kind,
                "root": str(campaign.root),
                "manifest_path": str(campaign.manifest_path),
            }
            for campaign in campaigns
        ],
        "color_conventions": COLOR_CONVENTIONS,
        "scientific_scope": {
            "signed_fields": "sign preserved; zero-centred color scale",
            "absolute_maps": "absolute value explicitly identified",
            "normalized_maps": "qualitative abs(S_q)/max(abs(S_q)) in [0,1]; not a quantitative update",
            "no_simulation_or_gradient_recomputation": True,
        },
        "summary": {"num_figures_ok": num_ok, "num_figures_skipped": num_skipped, "strict_failure": strict_failure},
        "figures": [record.to_dict() for record in records],
        "input_inventory": _input_inventory(records),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary(summary_path, records, roi=roi)

    if not config.quiet:
        print(f"[ARTICLE FIGURES] status={status} generated={num_ok} skipped={num_skipped} output={config.output_dir}")
        for record in records:
            if record.status == "skipped":
                print(f"[ARTICLE FIGURES] skipped {record.figure_id}: {record.reason}")
        print(f"[ARTICLE FIGURES] manifest={manifest_path}")
        print(f"[ARTICLE FIGURES] captions={summary_path}")
    return ArticleFigureResult(status, tuple(records), manifest_path, summary_path, exit_code)
