from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_COMBINATIONS = (
    "classical_full",
    "standard_reflection_only",
    "standard_transmission_only",
    "residual_interaction",
)

REQUIRED_GRADIENT_FIELDS = (
    "gradient_rho",
    "gradient_C11",
    "gradient_C22",
    "gradient_C12",
    "gradient_C66",
)

REQUIRED_KERNEL_FIELDS = (
    "residual_interaction_rho",
    "residual_interaction_C11",
    "residual_interaction_C22",
    "residual_interaction_C12",
    "residual_interaction_C66",
)


def _required_fields_for_combination(combination: str) -> tuple[str, ...]:
    if str(combination) == "residual_interaction":
        return REQUIRED_KERNEL_FIELDS
    return REQUIRED_GRADIENT_FIELDS


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_campaign_manifest(input_dir: Path) -> Path:
    candidates = [
        input_dir / "gradient_combination_campaign_manifest.json",
        input_dir / "campaign_manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(input_dir.glob("*campaign*manifest*.json"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No campaign manifest found in {input_dir}")


def _extract_found_paths(campaign_manifest: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    validation = campaign_manifest.get("validation", {})
    if isinstance(validation, Mapping):
        found = validation.get("found", {})
        if isinstance(found, Mapping):
            out: dict[str, dict[str, str]] = {}
            for combination, fields in found.items():
                if isinstance(fields, Mapping):
                    out[str(combination)] = {
                        str(field): str(path)
                        for field, path in fields.items()
                        if path is not None
                    }
            return out
    return {}


def _row_for_image(
    *,
    campaign_name: str,
    preset: str,
    combination: str,
    field: str,
    path: str | None,
) -> dict[str, Any]:
    exists = bool(path) and Path(str(path)).exists()
    size_bytes = Path(str(path)).stat().st_size if exists else 0

    return {
        "campaign_name": campaign_name,
        "preset": preset,
        "combination": combination,
        "field": field,
        "expected_field_name": field,
        "pvd_path": "" if path is None else str(path),
        "file_exists": bool(exists),
        "file_size_bytes": int(size_bytes),
        "metric_status": "ok" if exists else "missing_file",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "campaign_name",
        "preset",
        "combination",
        "field",
        "expected_field_name",
        "pvd_path",
        "file_exists",
        "file_size_bytes",
        "metric_status",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def postprocess(
    *,
    input_dir: Path,
    output_dir: Path,
    campaign_name: str,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    campaign_manifest_path = _find_campaign_manifest(input_dir)
    campaign_manifest = _read_json(campaign_manifest_path)

    preset = str(campaign_manifest.get("preset", "unknown"))
    found = _extract_found_paths(campaign_manifest)

    rows: list[dict[str, Any]] = []
    missing_pairs: list[dict[str, str]] = []
    missing_files: list[dict[str, str]] = []

    for combination in REQUIRED_COMBINATIONS:
        for field in _required_fields_for_combination(combination):
            path = found.get(combination, {}).get(field)
            row = _row_for_image(
                campaign_name=campaign_name,
                preset=preset,
                combination=combination,
                field=field,
                path=path,
            )
            rows.append(row)

            if not path:
                missing_pairs.append({"combination": combination, "field": field})
            elif not Path(path).exists():
                missing_files.append(
                    {"combination": combination, "field": field, "path": path}
                )

            print(f"[METRICS] {combination} / {field} -> {path}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "gradient_combination_metrics.csv"
    json_path = output_dir / "gradient_combination_metrics.json"
    manifest_path = output_dir / "postprocess_gradient_combination_metrics_manifest.json"

    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "kind": "postprocess_gradient_combination_metrics",
        "status": "ok" if not missing_pairs and not missing_files else "error",
        "campaign_name": campaign_name,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "campaign_manifest": str(campaign_manifest_path),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "num_expected_primary_images": sum(
            len(_required_fields_for_combination(c)) for c in REQUIRED_COMBINATIONS
        ),
        "num_metric_rows": len(rows),
        "num_missing_pairs": len(missing_pairs),
        "num_missing_files": len(missing_files),
        "missing_pairs": missing_pairs,
        "missing_files": missing_files,
        "required_combinations": list(REQUIRED_COMBINATIONS),
        "required_fields_by_combination": {
            c: list(_required_fields_for_combination(c)) for c in REQUIRED_COMBINATIONS
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Postprocess gradient-combination PVD maps and residual-wave kernel maps."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--campaign-name", type=str, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    postprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        campaign_name=args.campaign_name,
    )


if __name__ == "__main__":
    main()
