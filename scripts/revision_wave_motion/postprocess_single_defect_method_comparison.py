#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

# Permit direct execution from a source checkout without requiring an editable install.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from elastic_fwi.postprocessing import single_defect_method_comparison as _comparison
from elastic_fwi.postprocessing.single_defect_method_comparison import (
    CardContractError,
    CardInput,
    CardKey,
    ComparisonConfig,
    ComparisonInputs,
    ComparisonResult,
    DEFAULT_RANKING_WEIGHTS,
    EXPECTED_METHODS,
    EXPECTED_PROJECT_PROPERTIES,
    METRIC_DEFINITIONS,
    REPRESENTATIONS,
    SourceMetricError,
    compute_raw_signed_metrics_streaming,
    read_source_metrics_csv,
    run_single_defect_method_comparison,
    write_comparison_outputs,
)
from elastic_fwi.postprocessing.vtk_cell_fields import resolve_vtk_dataset


LINE_ONLY_GEOMETRY = "line"
LINE_BOX_GEOMETRY = "line-box"


@dataclass(frozen=True)
class LineOnlyConfig:
    line_metrics_csv: Path
    line_manifest: Path
    output_dir: Path
    rtol: float = 1.0e-12
    atol: float = 1.0e-14
    gain_epsilon: float = 1.0e-15
    overwrite: bool = False
    quiet: bool = False
    ranking_weights: Mapping[str, float] | None = None
    false_max_threshold_fraction: float = 0.5
    false_max_min_distance_factor: float = 1.0
    max_false_max_candidates: int = 5000
    remove_ghost_cells: bool = True

    def __post_init__(self) -> None:
        if self.rtol < 0.0 or self.atol < 0.0:
            raise ValueError("rtol and atol must be non-negative.")
        if self.gain_epsilon <= 0.0:
            raise ValueError("gain_epsilon must be strictly positive.")
        if not (0.0 < self.false_max_threshold_fraction <= 1.0):
            raise ValueError("false_max_threshold_fraction must lie in (0, 1].")
        if self.false_max_min_distance_factor <= 0.0:
            raise ValueError("false_max_min_distance_factor must be positive.")
        if self.max_false_max_candidates <= 0:
            raise ValueError("max_false_max_candidates must be positive.")

        weights = dict(self.ranking_weights or DEFAULT_RANKING_WEIGHTS)
        unknown = sorted(set(weights) - set(METRIC_DEFINITIONS))
        if unknown:
            raise ValueError(f"Unknown ranking metric(s): {unknown}")
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in weights.values()
        ):
            raise ValueError("Ranking weights must be finite and non-negative.")
        if sum(float(value) for value in weights.values()) <= 0.0:
            raise ValueError("At least one ranking weight must be positive.")



def _ranking_weights(value: str | None) -> dict[str, float]:
    if value is None:
        return dict(DEFAULT_RANKING_WEIGHTS)
    candidate = Path(value).expanduser()
    try:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        else:
            payload = json.loads(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(
            "--ranking-weights must be a JSON object or a path to a JSON file."
        ) from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("Ranking weights JSON must be an object.")
    try:
        return {str(key): float(weight) for key, weight in payload.items()}
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Ranking weights must be numeric.") from exc



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process validated single-defect imaging maps. Use '--geometry line' "
            "for a genuine line-only T20 comparison, or '--geometry line-box' for the "
            "historical line-versus-box workflow. Raw-signed VTK fields are streamed "
            "one piece at a time."
        )
    )
    parser.add_argument(
        "--geometry",
        choices=(LINE_ONLY_GEOMETRY, LINE_BOX_GEOMETRY),
        default=LINE_BOX_GEOMETRY,
        help=(
            "Processing scope. 'line' requires only line inputs and produces 20 "
            "consolidated rows. 'line-box' preserves the historical 80-card/40-row "
            "comparison. Default: line-box."
        ),
    )
    parser.add_argument("--line-metrics-csv", type=Path, required=True)
    parser.add_argument("--box-metrics-csv", type=Path, default=None)
    parser.add_argument("--line-manifest", type=Path, required=True)
    parser.add_argument("--box-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    parser.add_argument("--atol", type=float, default=1.0e-14)
    parser.add_argument("--gain-epsilon", type=float, default=1.0e-15)
    parser.add_argument(
        "--ranking-weights",
        type=str,
        default=None,
        help="JSON object or JSON file path; defaults to the documented reproducible weights.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser



def _configure_logger(output_dir: Path, *, quiet: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "single_defect_method_comparison.log"
    logger = logging.getLogger("single_defect_method_comparison")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if not quiet:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger



def _validate_line_only_card_contract(cards: Sequence[CardInput]) -> tuple[CardInput, ...]:
    expected = {
        CardKey(LINE_ONLY_GEOMETRY, representation, method, property_name)
        for representation in REPRESENTATIONS
        for method in EXPECTED_METHODS
        for property_name in EXPECTED_PROJECT_PROPERTIES
    }
    by_key: dict[CardKey, list[CardInput]] = {}
    unexpected: list[CardInput] = []
    forbidden: list[CardInput] = []

    for card in cards:
        if (
            _comparison.FORBIDDEN_TOKEN in card.key.method
            or _comparison.FORBIDDEN_TOKEN in card.expected_field_name
        ):
            forbidden.append(card)
            continue
        if card.key not in expected:
            unexpected.append(card)
        by_key.setdefault(card.key, []).append(card)

    missing = sorted(expected - set(by_key))
    duplicates = {key: values for key, values in by_key.items() if len(values) > 1}
    expected_count = len(expected)
    if missing or duplicates or unexpected or forbidden or len(cards) != expected_count:
        details = {
            "mode": "line-only",
            "expected_count": expected_count,
            "observed_count": len(cards),
            "missing_keys": [key.as_dict() for key in missing],
            "duplicated_keys": [
                {
                    "key": key.as_dict(),
                    "files": [str(card.pvd_path) for card in values],
                }
                for key, values in sorted(duplicates.items())
            ],
            "unexpected_keys": [
                {"key": card.key.as_dict(), "file": str(card.pvd_path)}
                for card in unexpected
            ],
            "forbidden_cards": [
                {"key": card.key.as_dict(), "file": str(card.pvd_path)}
                for card in forbidden
            ],
        }
        raise CardContractError(
            "The 40-card line-only contract is not satisfied:\n"
            + json.dumps(details, indent=2, sort_keys=True)
        )
    return tuple(sorted(cards, key=lambda card: card.key))



def _build_line_only_rows(
    *,
    cards: Sequence[CardInput],
    source_metrics: Mapping[CardKey, Any],
    raw_metrics: Mapping[CardKey, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    card_by_key = {card.key: card for card in cards}
    rows: list[dict[str, Any]] = []

    for method in EXPECTED_METHODS:
        for property_name in EXPECTED_PROJECT_PROPERTIES:
            article_key = CardKey(
                LINE_ONLY_GEOMETRY, "article_safe", method, property_name
            )
            raw_key = CardKey(
                LINE_ONLY_GEOMETRY, "raw_signed", method, property_name
            )
            source = source_metrics[article_key]
            raw = raw_metrics[raw_key]
            article_card = card_by_key[article_key]
            raw_card = card_by_key[raw_key]

            row: dict[str, Any] = {
                "geometry": LINE_ONLY_GEOMETRY,
                "method": method,
                "property": property_name,
                "validation__status": "ok",
                "validation__representations_validated": 2,
                "article_safe__metric_provenance": (
                    "read_exactly_from_validated_source_csv"
                ),
                "article_safe__source_csv": str(source.source_csv),
                "article_safe__source_row_number": source.source_row_number,
                "article_safe__pvd_path": str(article_card.pvd_path),
                "article_safe__expected_field": article_card.expected_field_name,
                "raw_signed__metric_provenance": "computed_streaming_from_vtk",
                "raw_signed__pvd_path": str(raw_card.pvd_path),
                "raw_signed__expected_field": raw_card.expected_field_name,
            }
            for metric, exact in source.exact_metrics.items():
                row[f"article_safe__{metric}"] = exact
            for metric, value in raw.items():
                row[f"raw_signed__{metric}"] = value
            rows.append(row)

    if len(rows) != 20:
        raise _comparison.ComparisonError(
            f"Expected 20 line-only consolidated rows, produced {len(rows)}."
        )
    keys = [(row["geometry"], row["method"], row["property"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise _comparison.ComparisonError(
            "Duplicate geometry/method/property rows after line-only consolidation."
        )
    return rows



def _apply_line_only_ranking(
    rows: list[dict[str, Any]],
    *,
    weights: Mapping[str, float],
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    active_weights = {
        str(key): float(value)
        for key, value in weights.items()
        if float(value) > 0.0
    }
    total_weight = sum(active_weights.values())
    normalized_weights = {
        key: value / total_weight for key, value in active_weights.items()
    }
    ranking_records: list[dict[str, Any]] = []

    for property_name in EXPECTED_PROJECT_PROPERTIES:
        candidates = [row for row in rows if row["property"] == property_name]
        candidates.sort(key=lambda row: str(row["method"]))
        if len(candidates) != len(EXPECTED_METHODS):
            raise _comparison.ComparisonError(
                f"Expected {len(EXPECTED_METHODS)} line-only ranking candidates for "
                f"{property_name}, got {len(candidates)}."
            )

        per_metric_scores: dict[str, list[float]] = {}
        for metric, weight in normalized_weights.items():
            del weight
            direction = str(METRIC_DEFINITIONS[metric]["direction"])
            if direction not in {"higher_is_better", "lower_is_better"}:
                raise _comparison.ComparisonError(
                    f"Ranking metric {metric!r} has non-rankable direction {direction!r}."
                )
            values: list[float] = []
            missing: list[str] = []
            for row in candidates:
                value = _comparison._numeric_or_none(
                    row.get(f"article_safe__{metric}")
                )
                if value is None:
                    missing.append(str(row["method"]))
                else:
                    values.append(value)
            if missing:
                raise _comparison.ComparisonError(
                    f"Ranking metric {metric!r} is missing for property "
                    f"{property_name}: {missing}."
                )
            tolerance = atol + rtol * max(max(abs(value) for value in values), 1.0)
            per_metric_scores[metric] = _comparison._normalize_scores(
                values,
                direction=direction,
                tolerance=tolerance,
            )

        scores: list[float] = []
        records_by_method: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(candidates):
            score = 0.0
            intermediate: dict[str, float] = {}
            for metric, weight in normalized_weights.items():
                normalized = per_metric_scores[metric][index]
                score += weight * normalized
                row[f"ranking__normalized__{metric}"] = normalized
                intermediate[metric] = normalized
            row["ranking__composite_score"] = score
            scores.append(score)
            record = {
                "geometry": LINE_ONLY_GEOMETRY,
                "method": row["method"],
                "property": property_name,
                "normalized_metrics": intermediate,
                "weights": normalized_weights,
                "composite_score": score,
            }
            ranking_records.append(record)
            records_by_method[str(row["method"])] = record

        unique_scores = sorted(scores, reverse=True)
        tie_tolerance = atol + rtol
        for row, score in zip(candidates, scores):
            rank = 1 + sum(other > score + tie_tolerance for other in unique_scores)
            tie_size = sum(
                abs(other - score) <= tie_tolerance for other in unique_scores
            )
            row["ranking__rank"] = rank
            row["ranking__tie_group_size"] = tie_size
            record = records_by_method[str(row["method"])]
            record["rank"] = rank
            record["tie_group_size"] = tie_size

    return sorted(
        ranking_records,
        key=lambda record: (
            str(record["property"]),
            int(record["rank"]),
            str(record["method"]),
        ),
    )



def _run_line_only(
    config: LineOnlyConfig,
    *,
    logger: logging.Logger,
) -> ComparisonResult:
    started = time.perf_counter()
    discovery = _comparison._discover_manifest(
        LINE_ONLY_GEOMETRY,
        config.line_manifest,
    )
    cards = _validate_line_only_card_contract(discovery.cards)
    methods = tuple(
        method for method in EXPECTED_METHODS if method in discovery.methods
    )
    properties = tuple(
        property_name
        for property_name in EXPECTED_PROJECT_PROPERTIES
        if property_name in discovery.properties
    )

    logger.info("Mode: line-only")
    logger.info("Discovered and logically validated %d cards.", len(cards))
    logger.info("Methods: %s", ", ".join(methods))
    logger.info("Properties: %s", ", ".join(properties))
    if discovery.exclusions:
        logger.info(
            "Explicit exclusions: %s",
            json.dumps(discovery.exclusions, sort_keys=True),
        )

    _comparison._validate_card_dataset_headers(cards)
    logger.info("Resolved all line PVD/PVTU/VTU datasets and serial pieces.")

    source_metrics = read_source_metrics_csv(
        config.line_metrics_csv,
        geometry=LINE_ONLY_GEOMETRY,
    )
    if len(source_metrics) != 20:
        raise SourceMetricError(
            "Expected 20 validated line article_safe source rows, "
            f"got {len(source_metrics)}."
        )
    _comparison._validate_source_metric_card_alignment(cards, source_metrics)
    logger.info(
        "Loaded and aligned 20 line article_safe metric rows without redefining "
        "source metrics."
    )

    raw_metrics: dict[CardKey, Mapping[str, Any]] = {}
    raw_cards = [card for card in cards if card.key.representation == "raw_signed"]
    for index, card in enumerate(raw_cards, start=1):
        article_key = CardKey(
            LINE_ONLY_GEOMETRY,
            "article_safe",
            card.key.method,
            card.key.property,
        )
        expected_num_cells = int(
            source_metrics[article_key].numeric_metrics["num_cells"]
        )
        logger.info(
            "[%d/%d] Streaming raw_signed card %s from %s",
            index,
            len(raw_cards),
            card.key.label(),
            card.pvd_path,
        )
        raw_metrics[card.key] = compute_raw_signed_metrics_streaming(
            card,
            defect_center=discovery.defect_center,
            defect_radius=discovery.defect_radius,
            false_max_threshold_fraction=config.false_max_threshold_fraction,
            false_max_min_distance_factor=config.false_max_min_distance_factor,
            max_false_max_candidates=config.max_false_max_candidates,
            remove_ghost_cells=config.remove_ghost_cells,
            expected_num_cells=expected_num_cells,
        )
        gc.collect()

    if len(raw_metrics) != 20:
        raise _comparison.ComparisonError(
            f"Expected 20 line raw_signed metric records, got {len(raw_metrics)}."
        )

    rows = _build_line_only_rows(
        cards=cards,
        source_metrics=source_metrics,
        raw_metrics=raw_metrics,
    )
    ranking_weights = dict(config.ranking_weights or DEFAULT_RANKING_WEIGHTS)
    rankings = _apply_line_only_ranking(
        rows,
        weights=ranking_weights,
        atol=config.atol,
        rtol=config.rtol,
    )
    reflection_control = {
        "method": "standard_reflection_only",
        "status": "not_applicable",
        "reason": "line_only_mode_has_no_box_geometry",
        "num_checks": 0,
        "num_failures": 0,
        "checks": [],
        "failures": [],
    }

    elapsed = time.perf_counter() - started
    logger.info("Line-only comparison completed in %.6f s.", elapsed)
    return ComparisonResult(
        cards=tuple(cards),
        properties=properties,
        methods=methods,
        source_metrics=source_metrics,
        raw_signed_metrics=raw_metrics,
        consolidated_rows=tuple(rows),
        gains=tuple(),
        rankings=tuple(rankings),
        reflection_control=reflection_control,
        exclusions=tuple(discovery.exclusions),
        warnings=(
            "Box-versus-line gains and reflection-only cross-geometry checks are "
            "not applicable in line-only mode.",
        ),
        defect_center=discovery.defect_center,
        defect_radius=discovery.defect_radius,
        elapsed_seconds=elapsed,
    )



def _line_only_summary(result: ComparisonResult) -> dict[str, Any]:
    best_by_property: dict[str, Any] = {}
    for property_name in EXPECTED_PROJECT_PROPERTIES:
        candidates = [
            record
            for record in result.rankings
            if record["property"] == property_name
        ]
        best_rank = min(int(record["rank"]) for record in candidates)
        best_by_property[property_name] = [
            record
            for record in candidates
            if int(record["rank"]) == best_rank
        ]

    return {
        "schema_version": _comparison.SCHEMA_VERSION,
        "kind": "single_defect_method_comparison",
        "mode": "line-only",
        "status": "ok",
        "num_entries_discovered": len(result.cards),
        "num_cards_validated": len(result.cards),
        "num_consolidated_rows": len(result.consolidated_rows),
        "geometries": [LINE_ONLY_GEOMETRY],
        "representations": list(REPRESENTATIONS),
        "methods": list(result.methods),
        "properties": list(result.properties),
        "defect_center": list(result.defect_center),
        "defect_radius": result.defect_radius,
        "best_rankings_by_property": best_by_property,
        "relative_gains": [],
        "reflection_only_control": result.reflection_control,
        "exclusions": list(result.exclusions),
        "warnings": list(result.warnings),
    }



def _make_line_only_figures(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        import numpy as np
    except ImportError as exc:
        raise _comparison.ComparisonError(
            "matplotlib and numpy are required to generate line-only figures."
        ) from exc

    methods = list(EXPECTED_METHODS)
    properties = list(EXPECTED_PROJECT_PROPERTIES)
    row_map = {
        (str(row["method"]), str(row["property"])): row for row in rows
    }

    score_matrix = np.asarray(
        [
            [
                float(row_map[(method, property_name)]["ranking__composite_score"])
                for method in methods
            ]
            for property_name in properties
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    image = ax.imshow(score_matrix, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticks(range(len(properties)))
    ax.set_yticklabels(properties)
    ax.set_title("Line-only per-property composite image-quality score")
    for row_index in range(score_matrix.shape[0]):
        for column_index in range(score_matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{score_matrix[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="per-property normalized score")
    fig.tight_layout()
    score_path = output_dir / "single_defect_method_comparison_scores.png"
    fig.savefig(score_path, dpi=180)
    plt.close(fig)

    balance_matrix = np.full((len(properties), len(methods)), np.nan, dtype=float)
    for row_index, property_name in enumerate(properties):
        for column_index, method in enumerate(methods):
            ratio = _comparison._numeric_or_none(
                row_map[(method, property_name)].get(
                    "raw_signed__positive_to_negative_sum_ratio"
                )
            )
            if ratio is not None and ratio > 0.0:
                balance_matrix[row_index, column_index] = math.log10(ratio)

    finite = balance_matrix[np.isfinite(balance_matrix)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
    max_abs = max(max_abs, 1.0e-12)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    image = ax.imshow(
        balance_matrix,
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
    )
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticks(range(len(properties)))
    ax.set_yticklabels(properties)
    ax.set_title(
        "Line-only raw-signed balance: log10(positive sum / negative absolute sum)"
    )
    for row_index in range(balance_matrix.shape[0]):
        for column_index in range(balance_matrix.shape[1]):
            value = balance_matrix[row_index, column_index]
            label = "n/a" if not math.isfinite(float(value)) else f"{value:.3f}"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(
        image,
        ax=ax,
        label="log10 ratio; >0 positive dominance, <0 negative dominance",
    )
    fig.tight_layout()
    balance_path = output_dir / "single_defect_method_comparison_signed_balance.png"
    fig.savefig(balance_path, dpi=180)
    plt.close(fig)

    return [score_path, balance_path]



def _write_line_only_outputs(
    result: ComparisonResult,
    config: LineOnlyConfig,
    *,
    command: Sequence[str] | None,
    logger: logging.Logger,
) -> dict[str, Path]:
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "csv": output_dir / "single_defect_method_comparison.csv",
        "latex": output_dir / "single_defect_method_comparison_table.tex",
        "summary": output_dir / "single_defect_method_comparison_summary.json",
        "manifest": output_dir / "single_defect_method_comparison_manifest.json",
        "log": output_dir / "single_defect_method_comparison.log",
    }
    figure_names = (
        "single_defect_method_comparison_scores.png",
        "single_defect_method_comparison_signed_balance.png",
    )
    protected = [paths[name] for name in ("csv", "latex", "summary", "manifest")]
    protected.extend(output_dir / name for name in figure_names)
    existing = [str(path) for path in protected if path.exists()]
    if existing and not config.overwrite:
        raise FileExistsError(
            "Output files already exist and --overwrite was not requested: "
            + ", ".join(existing)
        )

    rows = [dict(row) for row in result.consolidated_rows]
    _comparison._write_csv(paths["csv"], rows)
    _comparison._write_latex_table(paths["latex"], rows)
    _comparison._write_json(paths["summary"], _line_only_summary(result))
    figure_paths = _make_line_only_figures(output_dir, rows)

    input_paths = {
        "line_metrics_csv": config.line_metrics_csv,
        "line_manifest": config.line_manifest,
    }
    card_metadata = []
    for card in result.cards:
        card_metadata.append(
            {
                "key": card.key.as_dict(),
                "field_name": card.expected_field_name,
                "pvd": _comparison._file_metadata(card.pvd_path, hash_file=True),
                "dataset": str(resolve_vtk_dataset(card.pvd_path).dataset_path),
            }
        )

    generated_files = [
        paths["csv"],
        paths["latex"],
        paths["summary"],
        paths["manifest"],
        paths["log"],
        *figure_paths,
    ]
    ranking_weights = dict(config.ranking_weights or DEFAULT_RANKING_WEIGHTS)
    total_weight = sum(float(value) for value in ranking_weights.values())
    manifest = {
        "schema_version": _comparison.SCHEMA_VERSION,
        "kind": "single_defect_method_comparison",
        "mode": "line-only",
        "status": "ok",
        "execution_date_utc": _comparison._execution_timestamp(
            ComparisonConfig(
                inputs=ComparisonInputs(
                    line_metrics_csv=config.line_metrics_csv,
                    box_metrics_csv=config.line_metrics_csv,
                    line_manifest=config.line_manifest,
                    box_manifest=config.line_manifest,
                ),
                output_dir=output_dir,
                rtol=config.rtol,
                atol=config.atol,
                gain_epsilon=config.gain_epsilon,
                overwrite=config.overwrite,
                quiet=config.quiet,
                ranking_weights=ranking_weights,
            )
        ),
        "command": list(command or ()),
        "inputs": {
            name: _comparison._file_metadata(Path(path), hash_file=True)
            for name, path in input_paths.items()
        },
        "card_inputs": card_metadata,
        "tolerances": {
            "rtol": config.rtol,
            "atol": config.atol,
        },
        "defect": {
            "center": list(result.defect_center),
            "radius": result.defect_radius,
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "signed_statistic_definitions": {
            "positive_sum": "Unweighted sum of strictly positive raw cell values.",
            "negative_sum_abs": (
                "Absolute value of the unweighted sum of strictly negative raw cell values."
            ),
            "positive_cell_fraction": (
                "Number of strictly positive cells divided by the finite cell count."
            ),
            "negative_cell_fraction": (
                "Number of strictly negative cells divided by the finite cell count."
            ),
            "positive_quadratic_energy": (
                "Unweighted sum of squared strictly positive raw cell values."
            ),
            "negative_quadratic_energy": (
                "Unweighted sum of squared strictly negative raw cell values."
            ),
            "positive_to_negative_sum_ratio": (
                "positive_sum / negative_sum_abs; null when the denominator is zero "
                "and the numerator is non-zero."
            ),
            "inside_outside_extrema": (
                "Signed extrema evaluated using the manifest defect disk."
            ),
        },
        "ranking": {
            "scope": "four methods within line geometry, separately for each property",
            "source_representation": "article_safe",
            "normalization": (
                "per-property min-max across four line methods; 0.5 when all "
                "candidates are equal within tolerance"
            ),
            "aggregation": "weighted arithmetic sum of normalized criteria",
            "weights_requested": ranking_weights,
            "weights_normalized": {
                key: float(value) / total_weight
                for key, value in ranking_weights.items()
            },
            "tie_rule": "same rank when composite scores differ by at most atol+rtol",
        },
        "cross_geometry_operations": {
            "relative_gains": "not_applicable",
            "reflection_only_consistency": result.reflection_control,
            "line_box_gain_figure": "not_generated",
        },
        "memory_strategy": {
            "vtk": "one serial VTU piece and one cell block at a time",
            "raw_signed": "one card at a time",
            "half_maximum": (
                "temporary on-disk float64 spool; deleted after each card"
            ),
            "article_safe": (
                "validated metrics read exactly from source CSV; no full-field "
                "recomputation"
            ),
        },
        "validation": {
            "num_cards": len(result.cards),
            "num_consolidated_rows": len(result.consolidated_rows),
            "geometries": [LINE_ONLY_GEOMETRY],
            "representations": list(REPRESENTATIONS),
            "exclusions": list(result.exclusions),
        },
        "runtime": {
            "python": sys.version,
            "platform": _comparison.platform.platform(),
            "dependencies": _comparison._dependency_versions(),
            "git_commit": _comparison._git_commit(Path.cwd()),
            "elapsed_seconds": result.elapsed_seconds,
        },
        "files_produced": [str(path) for path in generated_files],
    }
    _comparison._write_json(paths["manifest"], manifest)

    if not paths["log"].exists():
        paths["log"].write_text(
            "single_defect_method_comparison line-only completed successfully\n"
            f"cards={len(result.cards)}\n"
            f"rows={len(result.consolidated_rows)}\n",
            encoding="utf-8",
        )

    logger.info("Wrote line-only comparison CSV: %s", paths["csv"])
    logger.info("Wrote line-only LaTeX table: %s", paths["latex"])
    logger.info("Wrote line-only summary JSON: %s", paths["summary"])
    logger.info("Wrote line-only reproducibility manifest: %s", paths["manifest"])
    for path in figure_paths:
        logger.info("Wrote line-only figure: %s", path)
    return {
        **paths,
        **{f"figure_{index + 1}": path for index, path in enumerate(figure_paths)},
    }



def _validate_cli_inputs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.geometry == LINE_BOX_GEOMETRY:
        missing = []
        if args.box_metrics_csv is None:
            missing.append("--box-metrics-csv")
        if args.box_manifest is None:
            missing.append("--box-manifest")
        if missing:
            parser.error(
                "--geometry line-box requires " + ", ".join(missing) + "."
            )
    elif args.geometry == LINE_ONLY_GEOMETRY:
        if args.box_metrics_csv is not None or args.box_manifest is not None:
            parser.error(
                "--geometry line accepts only line inputs; remove --box-metrics-csv "
                "and --box-manifest."
            )



def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_cli_inputs(args, parser)

    output_dir = args.output_dir.expanduser().resolve()
    logger = _configure_logger(output_dir, quiet=args.quiet)
    command = (
        [sys.executable, *sys.argv]
        if argv is None
        else [sys.executable, sys.argv[0], *argv]
    )

    try:
        ranking_weights = _ranking_weights(args.ranking_weights)
        if args.geometry == LINE_ONLY_GEOMETRY:
            config = LineOnlyConfig(
                line_metrics_csv=args.line_metrics_csv,
                line_manifest=args.line_manifest,
                output_dir=output_dir,
                rtol=args.rtol,
                atol=args.atol,
                gain_epsilon=args.gain_epsilon,
                overwrite=args.overwrite,
                quiet=args.quiet,
                ranking_weights=ranking_weights,
            )
            logger.info("Starting single-defect method comparison in line-only mode.")
            logger.info("line_metrics_csv=%s", args.line_metrics_csv)
            logger.info("line_manifest=%s", args.line_manifest)
            logger.info("output_dir=%s", output_dir)
            result = _run_line_only(config, logger=logger)
            paths = _write_line_only_outputs(
                result,
                config,
                command=command,
                logger=logger,
            )
        else:
            assert args.box_metrics_csv is not None
            assert args.box_manifest is not None
            config = ComparisonConfig(
                inputs=ComparisonInputs(
                    line_metrics_csv=args.line_metrics_csv,
                    box_metrics_csv=args.box_metrics_csv,
                    line_manifest=args.line_manifest,
                    box_manifest=args.box_manifest,
                ),
                output_dir=output_dir,
                rtol=args.rtol,
                atol=args.atol,
                gain_epsilon=args.gain_epsilon,
                overwrite=args.overwrite,
                quiet=args.quiet,
                ranking_weights=ranking_weights,
            )
            logger.info("Starting single-defect method comparison in line-box mode.")
            logger.info("line_metrics_csv=%s", args.line_metrics_csv)
            logger.info("box_metrics_csv=%s", args.box_metrics_csv)
            logger.info("line_manifest=%s", args.line_manifest)
            logger.info("box_manifest=%s", args.box_manifest)
            logger.info("output_dir=%s", output_dir)
            result = run_single_defect_method_comparison(config, logger=logger)
            paths = write_comparison_outputs(
                result,
                config,
                command=command,
                logger=logger,
            )

        logger.info("Status: OK")
        if not args.quiet:
            print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return 0
    except Exception as exc:
        logger.exception("Post-processing failed: %s", exc)
        if args.quiet:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
