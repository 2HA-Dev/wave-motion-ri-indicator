from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


VALID_STATUS_FOR_CLAIMS = {"ok"}

POSITIVE_METRICS = {
    "peak_to_artifact_ratio": "higher_is_better",
}

NEGATIVE_METRICS = {
    "energy_outside_defect": "lower_is_better",
    "localization_error": "lower_is_better",
    "false_maxima_count": "lower_is_better",
    "half_max_area": "lower_is_better_when_artifact_dominated",
}


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def read_metrics_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("method", "")),
        str(row.get("imaging_family", "")),
        str(row.get("field_name", "")),
    )


def _has_valid_status(status_values: list[str]) -> bool:
    return any(status in VALID_STATUS_FOR_CLAIMS for status in status_values)


def summarize_metric_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)

    summaries: list[dict[str, Any]] = []

    for (method, imaging_family, field_name), items in sorted(grouped.items()):
        status_values = sorted({str(item.get("status", "")) for item in items})
        item = items[0]

        summary = {
            "method": method,
            "imaging_family": imaging_family,
            "field_name": field_name,
            "status_values": status_values,
            "num_rows": len(items),
            "peak_to_artifact_ratio": _as_float(item.get("peak_to_artifact_ratio")),
            "energy_outside_defect": _as_float(item.get("energy_outside_defect")),
            "localization_error": _as_float(item.get("localization_error")),
            "false_maxima_count": _as_float(item.get("false_maxima_count")),
            "energy_in_defect": _as_float(item.get("energy_in_defect")),
            "max_value_in_defect": _as_float(item.get("max_value_in_defect")),
            "max_value_outside_defect": _as_float(item.get("max_value_outside_defect")),
            "reflection_weight": _as_float(item.get("reflection_weight")),
            "transmission_weight": _as_float(item.get("transmission_weight")),
            "eta": _as_float(item.get("eta")),
        }

        finite_signal = any(
            value is not None and abs(value) > 0.0
            for value in [
                summary["peak_to_artifact_ratio"],
                summary["energy_outside_defect"],
                summary["energy_in_defect"],
                summary["max_value_in_defect"],
                summary["max_value_outside_defect"],
            ]
        )

        valid_status = _has_valid_status(status_values)

        summary["has_finite_nonzero_signal"] = finite_signal
        summary["has_valid_status_for_claims"] = valid_status
        summary["is_scientifically_eligible"] = bool(finite_signal and valid_status)

        summaries.append(summary)

    return {
        "num_input_rows": len(rows),
        "num_grouped_rows": len(summaries),
        "num_nonzero_groups": sum(1 for item in summaries if item["has_finite_nonzero_signal"]),
        "num_valid_status_groups": sum(1 for item in summaries if item["has_valid_status_for_claims"]),
        "num_scientifically_eligible_groups": sum(
            1 for item in summaries if item["is_scientifically_eligible"]
        ),
        "groups": summaries,
    }


def _rank_groups(
    groups: list[Mapping[str, Any]],
    *,
    field_name: str | None,
    imaging_family: str | None,
    metric: str,
    reverse: bool,
    require_scientific_eligibility: bool = True,
) -> list[dict[str, Any]]:
    selected = []

    for group in groups:
        if require_scientific_eligibility and not group.get("is_scientifically_eligible", False):
            continue
        if field_name is not None and group.get("field_name") != field_name:
            continue
        if imaging_family is not None and group.get("imaging_family") != imaging_family:
            continue

        value = group.get(metric)
        if value is None:
            continue

        selected.append(dict(group))

    selected.sort(key=lambda item: float(item[metric]), reverse=reverse)
    return selected


def _eligible_groups_for_preference(
    groups: list[Mapping[str, Any]],
    *,
    preferred_field: str,
    preferred_family: str | None,
) -> list[Mapping[str, Any]]:
    selected = []

    for group in groups:
        if group.get("field_name") != preferred_field:
            continue
        if preferred_family is not None and group.get("imaging_family") != preferred_family:
            continue
        if not group.get("is_scientifically_eligible", False):
            continue
        selected.append(group)

    return selected


def build_article_claim_candidates(
    summary: Mapping[str, Any],
    *,
    preferred_field: str = "gradient_rho",
    preferred_family: str | None = None,
) -> dict[str, Any]:
    groups = list(summary.get("groups", []))

    eligible_preferred = _eligible_groups_for_preference(
        groups,
        preferred_field=preferred_field,
        preferred_family=preferred_family,
    )

    best_peak_to_artifact = _rank_groups(
        groups,
        field_name=preferred_field,
        imaging_family=preferred_family,
        metric="peak_to_artifact_ratio",
        reverse=True,
    )
    best_energy_outside = _rank_groups(
        groups,
        field_name=preferred_field,
        imaging_family=preferred_family,
        metric="energy_outside_defect",
        reverse=False,
    )
    best_localization = _rank_groups(
        groups,
        field_name=preferred_field,
        imaging_family=preferred_family,
        metric="localization_error",
        reverse=False,
    )
    best_false_maxima = _rank_groups(
        groups,
        field_name=preferred_field,
        imaging_family=preferred_family,
        metric="false_maxima_count",
        reverse=False,
    )

    if eligible_preferred:
        validity = "candidate_quantitative_assessment"
    else:
        validity = "not_scientifically_conclusive_no_valid_preferred_groups"

    return {
        "validity": validity,
        "preferred_field": preferred_field,
        "preferred_family": preferred_family,
        "num_eligible_preferred_groups": len(eligible_preferred),
        "best_peak_to_artifact_ratio": best_peak_to_artifact[:5],
        "lowest_energy_outside_defect": best_energy_outside[:5],
        "lowest_localization_error": best_localization[:5],
        "lowest_false_maxima_count": best_false_maxima[:5],
    }


def write_assessment_outputs(
    *,
    metrics_csv: str | Path,
    output_dir: str | Path,
    preferred_field: str = "gradient_rho",
    preferred_family: str | None = None,
) -> dict[str, Any]:
    metrics_csv = Path(metrics_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_metrics_csv(metrics_csv)
    summary = summarize_metric_rows(rows)
    claims = build_article_claim_candidates(
        summary,
        preferred_field=preferred_field,
        preferred_family=preferred_family,
    )

    payload = {
        "kind": "campaign_metrics_assessment",
        "status": "ok",
        "metrics_csv": str(metrics_csv),
        "preferred_field": preferred_field,
        "preferred_family": preferred_family,
        "summary": summary,
        "claim_candidates": claims,
    }

    json_path = output_dir / "assessment_summary.json"
    md_path = output_dir / "assessment_summary.md"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")

    payload["assessment_json"] = str(json_path)
    payload["assessment_markdown"] = str(md_path)
    return payload


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _render_rank_table(title: str, rows: list[Mapping[str, Any]], metric: str) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        lines += ["No scientifically eligible finite data.", ""]
        return lines

    lines.append("| rank | method | family | field | metric | status |")
    lines.append("|---:|---|---|---|---:|---|")
    for i, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    str(row.get("method", "")),
                    str(row.get("imaging_family", "")),
                    str(row.get("field_name", "")),
                    _format_value(row.get(metric)),
                    ",".join(row.get("status_values", [])),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _render_markdown(payload: Mapping[str, Any]) -> str:
    claims = payload["claim_candidates"]
    summary = payload["summary"]

    lines = [
        "# Campaign metrics assessment",
        "",
        f"- metrics CSV: `{payload['metrics_csv']}`",
        f"- preferred field: `{payload['preferred_field']}`",
        f"- preferred family: `{payload['preferred_family']}`",
        f"- validity: `{claims['validity']}`",
        f"- input rows: `{summary['num_input_rows']}`",
        f"- grouped rows: `{summary['num_grouped_rows']}`",
        f"- nonzero groups: `{summary['num_nonzero_groups']}`",
        f"- valid-status groups: `{summary['num_valid_status_groups']}`",
        f"- scientifically eligible groups: `{summary['num_scientifically_eligible_groups']}`",
        f"- eligible preferred groups: `{claims['num_eligible_preferred_groups']}`",
        "",
    ]

    if claims["validity"] != "candidate_quantitative_assessment":
        lines += [
            "The current metrics are not sufficient for article-level claims on the preferred field/family.",
            "Rows with statuses such as `no_inside_samples`, `no_outside_samples`, `zero_image`, or `empty` are excluded from rankings.",
            "Use a less degenerate case, typically `--preset medium`, before drawing scientific conclusions.",
            "",
        ]

    lines += _render_rank_table(
        "Best peak-to-artifact ratio",
        claims["best_peak_to_artifact_ratio"],
        "peak_to_artifact_ratio",
    )
    lines += _render_rank_table(
        "Lowest out-of-defect energy",
        claims["lowest_energy_outside_defect"],
        "energy_outside_defect",
    )
    lines += _render_rank_table(
        "Lowest localization error",
        claims["lowest_localization_error"],
        "localization_error",
    )
    lines += _render_rank_table(
        "Lowest false maxima count",
        claims["lowest_false_maxima_count"],
        "false_maxima_count",
    )

    return "\n".join(lines)
