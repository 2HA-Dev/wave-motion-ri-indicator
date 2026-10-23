from __future__ import annotations

"""Single-defect comparison of first-iteration imaging methods.

The module keeps validated ``article_safe`` metrics byte-for-byte as read from
source CSV files and computes new ``raw_signed`` statistics by streaming one
VTU piece at a time.  It contains no CLI parsing and never writes into input
campaign directories.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import csv
import gc
import hashlib
import heapq
import json
import logging
import math
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

from .vtk_cell_fields import (
    VTKCellFieldError,
    iter_vtk_cell_field_chunks,
    resolve_vtk_dataset,
)


GEOMETRIES = ("line", "box")
REPRESENTATIONS = ("article_safe", "raw_signed")
EXPECTED_METHODS = (
    "classical_full",
    "standard_reflection_only",
    "standard_transmission_only",
    "residual_interaction",
)
EXPECTED_PROJECT_PROPERTIES = ("rho", "C11", "C22", "C12", "C66")
FORBIDDEN_TOKEN = "material_gradient_proxy"
SCHEMA_VERSION = "1.0.0"

SOURCE_ID_COLUMNS = {
    "campaign_name",
    "combination",
    "expected_field",
    "actual_field",
    "pvd_path",
    "data_path",
}

SOURCE_METRIC_COLUMNS = (
    "num_cells",
    "global_peak_value",
    "global_peak_x",
    "global_peak_y",
    "localization_error",
    "peak_in_defect",
    "peak_outside_defect",
    "peak_to_artifact_ratio",
    "mean_in_defect",
    "mean_outside_defect",
    "contrast_defect_to_background",
    "energy_in_defect",
    "energy_outside_defect",
    "energy_fraction_in_defect",
    "half_max_threshold",
    "half_max_area",
    "false_max_threshold",
    "false_maxima_count",
)

SIGNED_STATISTIC_COLUMNS = (
    "signed_minimum",
    "signed_maximum",
    "positive_sum",
    "negative_sum_abs",
    "positive_cell_fraction",
    "negative_cell_fraction",
    "zero_cell_fraction",
    "positive_quadratic_energy",
    "negative_quadratic_energy",
    "positive_to_negative_sum_ratio",
    "positive_to_negative_ratio_status",
    "max_positive_in_defect",
    "min_negative_in_defect",
    "max_positive_outside_defect",
    "min_negative_outside_defect",
)

RAW_QUALITY_COLUMNS = SOURCE_METRIC_COLUMNS + (
    "metric_status",
    "num_inside_defect",
    "num_outside_defect",
    "num_ghost_cells_removed",
    "selected_vtk_field",
)

METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "localization_error": {
        "category": "localization",
        "direction": "lower_is_better",
        "unit": "model_length",
        "description": "Euclidean distance between the global absolute peak and the defect centre.",
    },
    "global_peak_x": {
        "category": "localization",
        "direction": "descriptive",
        "unit": "model_length",
        "description": "x coordinate of the global absolute peak.",
    },
    "global_peak_y": {
        "category": "localization",
        "direction": "descriptive",
        "unit": "model_length",
        "description": "y coordinate of the global absolute peak.",
    },
    "peak_in_defect": {
        "category": "concentration",
        "direction": "higher_is_better",
        "unit": "map_value",
        "description": "Maximum absolute map value inside the defect disk.",
    },
    "mean_in_defect": {
        "category": "concentration",
        "direction": "higher_is_better",
        "unit": "map_value",
        "description": "Mean absolute map value inside the defect disk.",
    },
    "energy_in_defect": {
        "category": "concentration",
        "direction": "higher_is_better",
        "unit": "map_value_squared_times_model_area",
        "description": "Cell-area-weighted integral of squared absolute map values inside the defect disk.",
    },
    "energy_fraction_in_defect": {
        "category": "concentration",
        "direction": "higher_is_better",
        "unit": "dimensionless",
        "description": "Fraction of cell-area-weighted quadratic map energy inside the defect disk.",
    },
    "half_max_area": {
        "category": "concentration",
        "direction": "lower_is_better",
        "unit": "model_area",
        "description": "Exact sum of DG0 cell areas whose absolute map value is at least half the global peak.",
    },
    "peak_to_artifact_ratio": {
        "category": "contrast",
        "direction": "higher_is_better",
        "unit": "dimensionless",
        "description": "Maximum absolute value inside the defect divided by the maximum outside.",
    },
    "contrast_defect_to_background": {
        "category": "contrast",
        "direction": "higher_is_better",
        "unit": "dimensionless",
        "description": "Mean absolute value inside the defect divided by the mean outside.",
    },
    "peak_outside_defect": {
        "category": "artifacts",
        "direction": "lower_is_better",
        "unit": "map_value",
        "description": "Maximum absolute map value outside the defect disk.",
    },
    "mean_outside_defect": {
        "category": "artifacts",
        "direction": "lower_is_better",
        "unit": "map_value",
        "description": "Mean absolute map value outside the defect disk.",
    },
    "energy_outside_defect": {
        "category": "artifacts",
        "direction": "lower_is_better",
        "unit": "map_value_squared_sum",
        "description": "Unweighted sum of squared absolute map values outside the defect disk.",
    },
    "false_maxima_count": {
        "category": "artifacts",
        "direction": "lower_is_better",
        "unit": "count",
        "description": "Greedy count of spatially separated outside-defect candidates above the validated threshold.",
    },
}

DEFAULT_RANKING_WEIGHTS: dict[str, float] = {
    "localization_error": 0.25,
    "energy_fraction_in_defect": 0.25,
    "peak_to_artifact_ratio": 0.20,
    "contrast_defect_to_background": 0.15,
    "false_maxima_count": 0.15,
}


class ComparisonError(RuntimeError):
    """Base error for deterministic comparison failures."""


class CardContractError(ComparisonError):
    """The 80-card logical contract is incomplete, duplicated or ambiguous."""


class SourceMetricError(ComparisonError):
    """Validated source metrics are missing, duplicated or malformed."""


class ReflectionConsistencyError(ComparisonError):
    """Reflection-only line/box consistency failed."""


@dataclass(frozen=True, order=True)
class CardKey:
    geometry: str
    representation: str
    method: str
    property: str

    def as_dict(self) -> dict[str, str]:
        return {
            "geometry": self.geometry,
            "representation": self.representation,
            "method": self.method,
            "property": self.property,
        }

    def label(self) -> str:
        return "/".join((self.geometry, self.representation, self.method, self.property))


@dataclass(frozen=True)
class CardInput:
    key: CardKey
    manifest_path: Path
    declared_path: str
    pvd_path: Path
    expected_field_name: str


@dataclass(frozen=True)
class ComparisonInputs:
    line_metrics_csv: Path
    box_metrics_csv: Path
    line_manifest: Path
    box_manifest: Path


@dataclass(frozen=True)
class ComparisonConfig:
    inputs: ComparisonInputs
    output_dir: Path
    rtol: float = 1.0e-12
    atol: float = 1.0e-14
    gain_epsilon: float = 1.0e-15
    overwrite: bool = False
    quiet: bool = False
    ranking_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_RANKING_WEIGHTS)
    )
    false_max_threshold_fraction: float = 0.5
    false_max_min_distance_factor: float = 1.0
    max_false_max_candidates: int = 5000
    remove_ghost_cells: bool = True
    strict_reflection_check: bool = True
    execution_timestamp: str | None = None

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
        weights = dict(self.ranking_weights)
        unknown = sorted(set(weights) - set(METRIC_DEFINITIONS))
        if unknown:
            raise ValueError(f"Unknown ranking metric(s): {unknown}")
        if any((not math.isfinite(float(v)) or float(v) < 0.0) for v in weights.values()):
            raise ValueError("Ranking weights must be finite and non-negative.")
        if sum(float(v) for v in weights.values()) <= 0.0:
            raise ValueError("At least one ranking weight must be positive.")


@dataclass(frozen=True)
class SourceMetricRecord:
    key: CardKey
    source_csv: Path
    source_row_number: int
    pvd_path_declared: str
    actual_field: str
    exact_metrics: Mapping[str, str]
    numeric_metrics: Mapping[str, float]


@dataclass(frozen=True)
class ComparisonResult:
    cards: tuple[CardInput, ...]
    properties: tuple[str, ...]
    methods: tuple[str, ...]
    source_metrics: Mapping[CardKey, SourceMetricRecord]
    raw_signed_metrics: Mapping[CardKey, Mapping[str, Any]]
    consolidated_rows: tuple[Mapping[str, Any], ...]
    gains: tuple[Mapping[str, Any], ...]
    rankings: tuple[Mapping[str, Any], ...]
    reflection_control: Mapping[str, Any]
    exclusions: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...]
    defect_center: tuple[float, float]
    defect_radius: float
    elapsed_seconds: float


@dataclass
class _ManifestDiscovery:
    geometry: str
    manifest_path: Path
    cards: list[CardInput]
    methods: tuple[str, ...]
    properties: tuple[str, ...]
    exclusions: list[dict[str, Any]]
    defect_center: tuple[float, float]
    defect_radius: float


def _read_json(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ComparisonError(f"Malformed JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonError(f"Expected a JSON object in {path}.")
    return value


def _normalize_property(field_name: str) -> str:
    name = str(field_name)
    if FORBIDDEN_TOKEN in name:
        raise CardContractError(f"Forbidden field encountered: {name}")
    for prefix in ("gradient_", "residual_interaction_"):
        if name.startswith(prefix):
            prop = name[len(prefix) :]
            if not prop:
                break
            return prop
    raise CardContractError(
        f"Cannot identify a project property from field name {name!r}; expected "
        "an explicit gradient_ or residual_interaction_ prefix."
    )


def _expected_field_name(method: str, property_name: str) -> str:
    if method == "residual_interaction":
        return f"residual_interaction_{property_name}"
    return f"gradient_{property_name}"


def _resolve_declared_path(declared: str, *, manifest_path: Path) -> Path:
    raw = Path(str(declared)).expanduser()
    attempted: list[Path] = []

    if raw.is_absolute():
        attempted.append(raw)
        if raw.exists():
            return raw.resolve()
        parts = raw.parts
        if "outputs" in parts:
            relative_outputs = Path(*parts[parts.index("outputs") :])
            for base in manifest_path.resolve().parents:
                candidate = base / relative_outputs
                attempted.append(candidate)
                if candidate.exists():
                    return candidate.resolve()
    else:
        for base in (Path.cwd(), *manifest_path.resolve().parents):
            candidate = base / raw
            attempted.append(candidate)
            if candidate.exists():
                return candidate.resolve()

    attempted_text = "\n  - ".join(str(p) for p in attempted)
    raise FileNotFoundError(
        f"Could not resolve manifest path {declared!r} declared in {manifest_path}. "
        f"Attempted:\n  - {attempted_text}"
    )


def _ordered_properties_from_validation(
    required_fields_by_combination: Mapping[str, Any],
    methods: Sequence[str],
) -> tuple[str, ...]:
    property_orders: list[tuple[str, ...]] = []
    for method in methods:
        fields = required_fields_by_combination.get(method)
        if not isinstance(fields, list):
            raise CardContractError(
                f"Missing list required_fields_by_combination[{method!r}]."
            )
        props = tuple(_normalize_property(str(field)) for field in fields)
        if len(props) != len(set(props)):
            raise CardContractError(
                f"Duplicate properties after field normalization for method {method}: {props}"
            )
        property_orders.append(props)

    first = property_orders[0]
    first_set = set(first)
    for method, props in zip(methods[1:], property_orders[1:]):
        if set(props) != first_set:
            raise CardContractError(
                f"Property mismatch between methods: {methods[0]} has {first}, "
                f"whereas {method} has {props}."
            )
    return first


def _discover_manifest(geometry: str, manifest_path: Path) -> _ManifestDiscovery:
    if geometry not in GEOMETRIES:
        raise ValueError(f"Unknown geometry {geometry!r}; expected one of {GEOMETRIES}.")
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_path)

    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise CardContractError(f"Missing config object in {manifest_path}.")
    try:
        defect_center = (
            float(config["defect_center_x"]),
            float(config["defect_center_y"]),
        )
        defect_radius = float(config["defect_radius"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CardContractError(
            f"Manifest {manifest_path} does not contain a valid defect centre/radius."
        ) from exc
    if defect_radius <= 0.0:
        raise CardContractError(f"Invalid non-positive defect radius in {manifest_path}.")

    cards: list[CardInput] = []
    exclusions: list[dict[str, Any]] = []
    discovered_methods: tuple[str, ...] | None = None
    discovered_properties: tuple[str, ...] | None = None

    for representation, section_name in (
        ("article_safe", "validation"),
        ("raw_signed", "raw_signed_validation"),
    ):
        section = manifest.get(section_name)
        if not isinstance(section, Mapping):
            raise CardContractError(
                f"Missing {section_name!r} object for {representation} in {manifest_path}."
            )
        found = section.get("found")
        if not isinstance(found, Mapping):
            raise CardContractError(
                f"Missing mapping {section_name}.found in {manifest_path}."
            )
        required_methods_raw = section.get("required_combinations")
        if not isinstance(required_methods_raw, list):
            raise CardContractError(
                f"Missing list {section_name}.required_combinations in {manifest_path}."
            )
        required_methods = tuple(str(v) for v in required_methods_raw)

        forbidden_methods = [m for m in found if FORBIDDEN_TOKEN in str(m)]
        for method in forbidden_methods:
            exclusions.append(
                {
                    "geometry": geometry,
                    "representation": representation,
                    "kind": "method",
                    "name": str(method),
                    "reason": "explicitly_forbidden_material_gradient_proxy",
                }
            )

        effective_found_methods = tuple(
            str(m) for m in found if FORBIDDEN_TOKEN not in str(m)
        )
        missing_methods = sorted(set(required_methods) - set(effective_found_methods))
        unexpected_methods = sorted(set(effective_found_methods) - set(required_methods))
        duplicated_required = sorted(
            method for method in set(required_methods) if required_methods.count(method) > 1
        )
        if missing_methods or unexpected_methods or duplicated_required:
            raise CardContractError(
                f"Method contract failure in {manifest_path} ({representation}): "
                f"missing={missing_methods}, unexpected={unexpected_methods}, "
                f"duplicates={duplicated_required}."
            )
        if set(required_methods) != set(EXPECTED_METHODS):
            raise CardContractError(
                f"The real manifest methods in {manifest_path} are {required_methods}, "
                f"but the project expects exactly {EXPECTED_METHODS}."
            )

        required_fields_by = section.get("required_fields_by_combination")
        if not isinstance(required_fields_by, Mapping):
            raise CardContractError(
                f"Missing {section_name}.required_fields_by_combination in {manifest_path}."
            )
        properties = _ordered_properties_from_validation(required_fields_by, required_methods)
        if set(properties) != set(EXPECTED_PROJECT_PROPERTIES) or len(properties) != 5:
            raise CardContractError(
                f"Properties identified from {manifest_path} are {properties}; expected the "
                f"five project properties {EXPECTED_PROJECT_PROPERTIES}."
            )

        if discovered_methods is None:
            discovered_methods = required_methods
            discovered_properties = properties
        else:
            if set(discovered_methods) != set(required_methods):
                raise CardContractError(
                    f"Method mismatch between representations in {manifest_path}."
                )
            if set(discovered_properties or ()) != set(properties):
                raise CardContractError(
                    f"Property mismatch between representations in {manifest_path}."
                )

        for method in required_methods:
            fields = found.get(method)
            if not isinstance(fields, Mapping):
                raise CardContractError(
                    f"Missing found fields for method {method!r} in {manifest_path}."
                )
            seen_properties: dict[str, str] = {}
            for field_name, declared_path in fields.items():
                field_name = str(field_name)
                if FORBIDDEN_TOKEN in field_name:
                    exclusions.append(
                        {
                            "geometry": geometry,
                            "representation": representation,
                            "kind": "field",
                            "method": method,
                            "name": field_name,
                            "path": str(declared_path),
                            "reason": "explicitly_forbidden_material_gradient_proxy",
                        }
                    )
                    continue
                prop = _normalize_property(field_name)
                if prop in seen_properties:
                    raise CardContractError(
                        f"Duplicate logical card in {manifest_path}: method={method}, "
                        f"representation={representation}, property={prop}, fields="
                        f"{seen_properties[prop]!r} and {field_name!r}."
                    )
                seen_properties[prop] = field_name
                expected_field = _expected_field_name(method, prop)
                if field_name != expected_field:
                    raise CardContractError(
                        f"Ambiguous field name for {method}/{prop} in {manifest_path}: "
                        f"observed {field_name!r}, expected exact name {expected_field!r}."
                    )
                if not isinstance(declared_path, str) or not declared_path:
                    raise CardContractError(
                        f"Missing path for {representation}/{method}/{field_name} in "
                        f"{manifest_path}."
                    )
                pvd_path = _resolve_declared_path(declared_path, manifest_path=manifest_path)
                if pvd_path.suffix.lower() not in {".pvd", ".pvtu", ".vtu"}:
                    raise CardContractError(
                        f"Unsupported card file {pvd_path} for {representation}/{method}/{prop}."
                    )
                cards.append(
                    CardInput(
                        key=CardKey(geometry, representation, method, prop),
                        manifest_path=manifest_path,
                        declared_path=declared_path,
                        pvd_path=pvd_path,
                        expected_field_name=field_name,
                    )
                )

            missing_properties = sorted(set(properties) - set(seen_properties))
            unexpected_properties = sorted(set(seen_properties) - set(properties))
            if missing_properties or unexpected_properties:
                raise CardContractError(
                    f"Field contract failure in {manifest_path} for {representation}/{method}: "
                    f"missing properties={missing_properties}, "
                    f"unexpected properties={unexpected_properties}."
                )

    assert discovered_methods is not None and discovered_properties is not None
    return _ManifestDiscovery(
        geometry=geometry,
        manifest_path=manifest_path,
        cards=cards,
        methods=discovered_methods,
        properties=discovered_properties,
        exclusions=exclusions,
        defect_center=defect_center,
        defect_radius=defect_radius,
    )


def validate_card_contract(cards: Sequence[CardInput]) -> tuple[CardInput, ...]:
    expected = {
        CardKey(geometry, representation, method, prop)
        for geometry in GEOMETRIES
        for representation in REPRESENTATIONS
        for method in EXPECTED_METHODS
        for prop in EXPECTED_PROJECT_PROPERTIES
    }
    by_key: dict[CardKey, list[CardInput]] = {}
    unexpected: list[CardInput] = []
    forbidden: list[CardInput] = []

    for card in cards:
        if FORBIDDEN_TOKEN in card.key.method or FORBIDDEN_TOKEN in card.expected_field_name:
            forbidden.append(card)
            continue
        if card.key not in expected:
            unexpected.append(card)
        by_key.setdefault(card.key, []).append(card)

    missing = sorted(expected - set(by_key))
    duplicates = {key: values for key, values in by_key.items() if len(values) > 1}
    if missing or duplicates or unexpected or forbidden or len(cards) != 80:
        details = {
            "expected_count": 80,
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
            "The 80-card contract is not satisfied:\n"
            + json.dumps(details, indent=2, sort_keys=True)
        )
    return tuple(sorted(cards, key=lambda c: c.key))


def discover_comparison_inputs(
    line_manifest: Path,
    box_manifest: Path,
) -> tuple[
    tuple[CardInput, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[Mapping[str, Any], ...],
    tuple[float, float],
    float,
]:
    line = _discover_manifest("line", line_manifest)
    box = _discover_manifest("box", box_manifest)

    if set(line.methods) != set(box.methods):
        raise CardContractError(
            f"Line/box method mismatch: line={line.methods}, box={box.methods}."
        )
    if set(line.properties) != set(box.properties):
        raise CardContractError(
            f"Line/box property mismatch: line={line.properties}, box={box.properties}."
        )
    if line.defect_center != box.defect_center or line.defect_radius != box.defect_radius:
        raise CardContractError(
            "Line and box manifests do not describe the same defect: "
            f"line center/radius={line.defect_center}/{line.defect_radius}, "
            f"box center/radius={box.defect_center}/{box.defect_radius}."
        )

    cards = validate_card_contract([*line.cards, *box.cards])
    methods = tuple(method for method in EXPECTED_METHODS if method in line.methods)
    properties = tuple(prop for prop in EXPECTED_PROJECT_PROPERTIES if prop in line.properties)
    exclusions = tuple([*line.exclusions, *box.exclusions])
    return cards, methods, properties, exclusions, line.defect_center, line.defect_radius


def _finite_float(value: Any, *, context: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise SourceMetricError(f"Non-numeric value {value!r} for {context}.") from exc
    if not math.isfinite(out):
        raise SourceMetricError(f"Non-finite value {value!r} for {context}.")
    return out


def read_source_metrics_csv(
    path: Path,
    *,
    geometry: str,
) -> dict[CardKey, SourceMetricRecord]:
    path = Path(path).expanduser().resolve()
    if geometry not in GEOMETRIES:
        raise ValueError(f"Unknown geometry {geometry!r}.")
    try:
        handle = path.open(newline="", encoding="utf-8")
    except FileNotFoundError:
        raise

    records: dict[CardKey, SourceMetricRecord] = {}
    with handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        missing_columns = sorted(
            {
                "combination",
                "expected_field",
                "actual_field",
                "pvd_path",
                *SOURCE_METRIC_COLUMNS,
            }
            - set(header)
        )
        if missing_columns:
            raise SourceMetricError(
                f"Source metrics CSV {path} is missing columns: {missing_columns}"
            )
        if FORBIDDEN_TOKEN in " ".join(header):
            raise SourceMetricError(
                f"Forbidden token {FORBIDDEN_TOKEN!r} appears in the header of {path}."
            )

        for row_number, row in enumerate(reader, start=2):
            method = str(row.get("combination", ""))
            expected_field = str(row.get("expected_field", ""))
            actual_field = str(row.get("actual_field", ""))
            if FORBIDDEN_TOKEN in method or FORBIDDEN_TOKEN in expected_field or FORBIDDEN_TOKEN in actual_field:
                raise SourceMetricError(
                    f"Forbidden material_gradient_proxy row in {path}:{row_number}."
                )
            if method not in EXPECTED_METHODS:
                raise SourceMetricError(
                    f"Unexpected method {method!r} in {path}:{row_number}."
                )
            prop = _normalize_property(expected_field)
            if prop not in EXPECTED_PROJECT_PROPERTIES:
                raise SourceMetricError(
                    f"Unexpected property {prop!r} in {path}:{row_number}."
                )
            if expected_field != _expected_field_name(method, prop):
                raise SourceMetricError(
                    f"Unexpected expected_field {expected_field!r} for {method}/{prop} "
                    f"in {path}:{row_number}."
                )
            if actual_field != expected_field:
                raise SourceMetricError(
                    f"Ambiguous actual_field in {path}:{row_number}: expected "
                    f"{expected_field!r}, observed {actual_field!r}."
                )

            key = CardKey(geometry, "article_safe", method, prop)
            if key in records:
                previous = records[key]
                raise SourceMetricError(
                    f"Duplicate source metric key {key.label()} in {path}: rows "
                    f"{previous.source_row_number} and {row_number}."
                )

            exact_metrics: dict[str, str] = {}
            numeric_metrics: dict[str, float] = {}
            for metric in SOURCE_METRIC_COLUMNS:
                exact = str(row.get(metric, ""))
                if exact == "":
                    raise SourceMetricError(
                        f"Missing metric {metric!r} for {key.label()} in {path}:{row_number}."
                    )
                exact_metrics[metric] = exact
                numeric_metrics[metric] = _finite_float(
                    exact,
                    context=f"{path}:{row_number}:{metric}",
                )

            records[key] = SourceMetricRecord(
                key=key,
                source_csv=path,
                source_row_number=row_number,
                pvd_path_declared=str(row.get("pvd_path", "")),
                actual_field=actual_field,
                exact_metrics=exact_metrics,
                numeric_metrics=numeric_metrics,
            )

    expected = {
        CardKey(geometry, "article_safe", method, prop)
        for method in EXPECTED_METHODS
        for prop in EXPECTED_PROJECT_PROPERTIES
    }
    missing = sorted(expected - set(records))
    unexpected = sorted(set(records) - expected)
    if missing or unexpected or len(records) != 20:
        raise SourceMetricError(
            f"Source metrics contract failure for {path}: expected 20 rows, observed "
            f"{len(records)}, missing={[k.label() for k in missing]}, "
            f"unexpected={[k.label() for k in unexpected]}."
        )
    return records




def _validate_source_metric_card_alignment(
    cards: Sequence[CardInput],
    source_metrics: Mapping[CardKey, SourceMetricRecord],
) -> None:
    card_by_key = {card.key: card for card in cards}
    for key, source in source_metrics.items():
        card = card_by_key.get(key)
        if card is None:
            raise SourceMetricError(
                f"Source metric key {key.label()} has no matching article_safe manifest card."
            )
        if source.actual_field != card.expected_field_name:
            raise SourceMetricError(
                f"Source/manifest field mismatch for {key.label()}: CSV actual_field="
                f"{source.actual_field!r}, manifest field={card.expected_field_name!r}."
            )
        source_pvd = _resolve_declared_path(
            source.pvd_path_declared,
            manifest_path=card.manifest_path,
        )
        if source_pvd != card.pvd_path:
            raise SourceMetricError(
                f"Source/manifest PVD mismatch for {key.label()}: CSV resolves to "
                f"{source_pvd}, manifest resolves to {card.pvd_path}."
            )


def _safe_ratio(numerator: float, denominator: float) -> tuple[float | None, str]:
    if denominator == 0.0:
        if numerator == 0.0:
            return 0.0, "both_zero"
        return None, "undefined_zero_denominator"
    return float(numerator / denominator), "ok"


def _update_extreme(
    current: float | None,
    candidate_values: np.ndarray,
    *,
    mode: str,
) -> float | None:
    if candidate_values.size == 0:
        return current
    candidate = float(np.max(candidate_values) if mode == "max" else np.min(candidate_values))
    if current is None:
        return candidate
    if mode == "max":
        return max(current, candidate)
    return min(current, candidate)


def _count_spatially_separated_candidates(
    candidates: Sequence[tuple[float, int, float, float]],
    *,
    threshold: float,
    min_distance: float,
) -> int:
    if threshold <= 0.0 and not any(value > 0.0 for value, _, _, _ in candidates):
        return 0
    ordered = sorted(
        (item for item in candidates if item[0] >= threshold),
        key=lambda item: (-item[0], item[1], item[2], item[3]),
    )
    if not ordered:
        return 0

    selected: list[tuple[float, float]] = []
    buckets: dict[tuple[int, int], list[int]] = {}
    cell_size = float(min_distance)
    min_distance_sq = cell_size * cell_size

    for _, _, x, y in ordered:
        gx = math.floor(x / cell_size)
        gy = math.floor(y / cell_size)
        accept = True
        for ix in range(gx - 1, gx + 2):
            for iy in range(gy - 1, gy + 2):
                for selected_index in buckets.get((ix, iy), ()):
                    sx, sy = selected[selected_index]
                    if (sx - x) ** 2 + (sy - y) ** 2 < min_distance_sq:
                        accept = False
                        break
                if not accept:
                    break
            if not accept:
                break
        if accept:
            index = len(selected)
            selected.append((x, y))
            buckets.setdefault((gx, gy), []).append(index)
    return len(selected)


def compute_raw_signed_metrics_streaming(
    card: CardInput,
    *,
    defect_center: tuple[float, float],
    defect_radius: float,
    false_max_threshold_fraction: float = 0.5,
    false_max_min_distance_factor: float = 1.0,
    max_false_max_candidates: int = 5000,
    remove_ghost_cells: bool = True,
    expected_num_cells: int | None = None,
) -> dict[str, Any]:
    """Compute raw-signed and absolute-image metrics in one VTK streaming pass.

    Only the absolute values needed for the exact half-maximum count are spooled
    to a temporary binary file.  No complete coordinate cloud or multi-field
    array is retained in RAM.
    """

    if card.key.representation != "raw_signed":
        raise ValueError(
            f"Signed statistics require a raw_signed card, got {card.key.label()}."
        )
    if defect_radius <= 0.0:
        raise ValueError("defect_radius must be positive.")

    centre = np.asarray(defect_center, dtype=np.float64)
    num_cells = 0
    num_inside = 0
    num_outside = 0
    num_positive = 0
    num_negative = 0
    num_zero = 0
    num_ghost = 0

    signed_min: float | None = None
    signed_max: float | None = None
    positive_sum = 0.0
    negative_sum_abs = 0.0
    positive_energy = 0.0
    negative_energy = 0.0

    max_pos_in: float | None = None
    min_neg_in: float | None = None
    max_pos_out: float | None = None
    min_neg_out: float | None = None

    abs_sum_in = 0.0
    abs_sum_out = 0.0
    area_in = 0.0
    area_out = 0.0
    energy_in = 0.0
    energy_out = 0.0
    peak_in = 0.0
    peak_out = 0.0
    global_peak = -1.0
    global_peak_x = math.nan
    global_peak_y = math.nan

    xmin = math.inf
    xmax = -math.inf
    ymin = math.inf
    ymax = -math.inf
    ordinal = 0
    candidates_heap: list[tuple[float, int, float, float]] = []
    selected_field: str | None = None

    temp_path: Path | None = None
    with tempfile.NamedTemporaryFile(prefix="elastic_fwi_abs_", suffix=".bin", delete=False) as spool:
        temp_path = Path(spool.name)
        try:
            for chunk in iter_vtk_cell_field_chunks(
                card.pvd_path,
                (card.expected_field_name,),
                time_index=0,
                remove_ghost_cells=remove_ghost_cells,
                require_finite=True,
            ):
                selected_field = chunk.field_name if selected_field is None else selected_field
                centers = chunk.centers
                values = chunk.values
                measures = np.asarray(chunk.cell_measures, dtype=np.float64)
                n = int(values.shape[0])
                num_ghost += int(chunk.num_ghost_cells_removed)
                if n == 0:
                    del chunk, centers, values, measures
                    continue

                distances = np.linalg.norm(centers - centre[None, :], axis=1)
                inside = distances <= defect_radius
                outside = ~inside
                positive = values > 0.0
                negative = values < 0.0
                zeros = ~(positive | negative)
                abs_values = np.abs(values)

                spool.write(
                    np.column_stack((abs_values, measures))
                    .astype("<f8", copy=False)
                    .tobytes(order="C")
                )

                num_cells += n
                nin = int(np.count_nonzero(inside))
                nout = n - nin
                num_inside += nin
                num_outside += nout
                num_positive += int(np.count_nonzero(positive))
                num_negative += int(np.count_nonzero(negative))
                num_zero += int(np.count_nonzero(zeros))

                local_min = float(np.min(values))
                local_max = float(np.max(values))
                signed_min = local_min if signed_min is None else min(signed_min, local_min)
                signed_max = local_max if signed_max is None else max(signed_max, local_max)

                positive_values = values[positive]
                negative_values = values[negative]
                positive_measures = measures[positive]
                negative_measures = measures[negative]
                positive_sum += float(
                    np.sum(positive_values * positive_measures, dtype=np.float64)
                )
                negative_sum_abs += float(
                    -np.sum(negative_values * negative_measures, dtype=np.float64)
                )
                positive_energy += float(
                    np.sum(positive_values * positive_values * positive_measures, dtype=np.float64)
                )
                negative_energy += float(
                    np.sum(negative_values * negative_values * negative_measures, dtype=np.float64)
                )

                max_pos_in = _update_extreme(max_pos_in, values[inside & positive], mode="max")
                min_neg_in = _update_extreme(min_neg_in, values[inside & negative], mode="min")
                max_pos_out = _update_extreme(max_pos_out, values[outside & positive], mode="max")
                min_neg_out = _update_extreme(min_neg_out, values[outside & negative], mode="min")

                if nin:
                    vals = abs_values[inside]
                    weights = measures[inside]
                    peak_in = max(peak_in, float(np.max(vals)))
                    area_in += float(np.sum(weights, dtype=np.float64))
                    abs_sum_in += float(np.sum(vals * weights, dtype=np.float64))
                    energy_in += float(np.sum(vals * vals * weights, dtype=np.float64))
                if nout:
                    vals = abs_values[outside]
                    weights = measures[outside]
                    peak_out = max(peak_out, float(np.max(vals)))
                    area_out += float(np.sum(weights, dtype=np.float64))
                    abs_sum_out += float(np.sum(vals * weights, dtype=np.float64))
                    energy_out += float(np.sum(vals * vals * weights, dtype=np.float64))

                local_peak_index = int(np.argmax(abs_values))
                local_peak = float(abs_values[local_peak_index])
                if local_peak > global_peak:
                    global_peak = local_peak
                    global_peak_x = float(centers[local_peak_index, 0])
                    global_peak_y = float(centers[local_peak_index, 1])

                xmin = min(xmin, float(np.min(centers[:, 0])))
                xmax = max(xmax, float(np.max(centers[:, 0])))
                ymin = min(ymin, float(np.min(centers[:, 1])))
                ymax = max(ymax, float(np.max(centers[:, 1])))

                outside_indices = np.flatnonzero(outside)
                for local_index in outside_indices:
                    value = float(abs_values[local_index])
                    x = float(centers[local_index, 0])
                    y = float(centers[local_index, 1])
                    item = (value, -ordinal, x, y)
                    if len(candidates_heap) < max_false_max_candidates:
                        heapq.heappush(candidates_heap, item)
                    elif item > candidates_heap[0]:
                        heapq.heapreplace(candidates_heap, item)
                    ordinal += 1

                del distances, inside, outside, positive, negative, zeros, abs_values
                del chunk, centers, values, measures
        finally:
            spool.flush()
            os.fsync(spool.fileno())

    try:
        if num_cells == 0 or selected_field is None:
            raise VTKCellFieldError(f"No finite cells read for {card.key.label()} from {card.pvd_path}.")
        if expected_num_cells is not None and num_cells != int(expected_num_cells):
            raise VTKCellFieldError(
                f"Cell-count mismatch for {card.key.label()}: raw_signed stream contains "
                f"{num_cells} cells, validated article_safe metrics report "
                f"{int(expected_num_cells)} cells."
            )

        assert temp_path is not None
        spooled = np.memmap(temp_path, dtype="<f8", mode="r", shape=(num_cells, 2))
        abs_memmap = spooled[:, 0]
        measure_memmap = spooled[:, 1]
        if global_peak <= 0.0:
            half_max_threshold = 0.0
            half_max_count = 0
            false_max_threshold = 0.0
            false_maxima_count = 0
            metric_status = "zero_image"
        else:
            half_max_threshold = 0.5 * global_peak
            half_max_mask = abs_memmap >= half_max_threshold
            half_max_count = int(np.count_nonzero(half_max_mask))
            false_max_threshold = false_max_threshold_fraction * global_peak
            candidates = [
                (value, -negative_ordinal, x, y)
                for value, negative_ordinal, x, y in candidates_heap
            ]
            false_maxima_count = _count_spatially_separated_candidates(
                candidates,
                threshold=false_max_threshold,
                min_distance=false_max_min_distance_factor * defect_radius,
            )
            metric_status = "ok"
        half_max_area = (
            float(np.sum(measure_memmap[half_max_mask], dtype=np.float64))
            if global_peak > 0.0 else 0.0
        )
        del abs_memmap, measure_memmap, spooled

        localization_error = float(
            math.hypot(global_peak_x - defect_center[0], global_peak_y - defect_center[1])
        )
        mean_in = abs_sum_in / area_in if area_in > 0.0 else 0.0
        mean_out = abs_sum_out / area_out if area_out > 0.0 else 0.0
        total_energy = energy_in + energy_out
        energy_fraction = energy_in / total_energy if total_energy > 0.0 else 0.0

        peak_ratio, peak_ratio_status = _safe_ratio(peak_in, peak_out)
        contrast, contrast_status = _safe_ratio(mean_in, mean_out)
        sign_ratio, sign_ratio_status = _safe_ratio(positive_sum, negative_sum_abs)

        return {
            "num_cells": num_cells,
            "global_peak_value": float(max(global_peak, 0.0)),
            "global_peak_x": global_peak_x,
            "global_peak_y": global_peak_y,
            "localization_error": localization_error,
            "peak_in_defect": peak_in,
            "peak_outside_defect": peak_out,
            "peak_to_artifact_ratio": peak_ratio,
            "peak_to_artifact_ratio_status": peak_ratio_status,
            "mean_in_defect": mean_in,
            "mean_outside_defect": mean_out,
            "contrast_defect_to_background": contrast,
            "contrast_defect_to_background_status": contrast_status,
            "energy_in_defect": energy_in,
            "energy_outside_defect": energy_out,
            "energy_fraction_in_defect": energy_fraction,
            "half_max_threshold": half_max_threshold,
            "half_max_area": half_max_area,
            "false_max_threshold": false_max_threshold,
            "false_maxima_count": false_maxima_count,
            "metric_status": metric_status,
            "num_inside_defect": num_inside,
            "num_outside_defect": num_outside,
            "num_ghost_cells_removed": num_ghost,
            "selected_vtk_field": selected_field,
            "area_in_defect": area_in,
            "area_outside_defect": area_out,
            "domain_area": area_in + area_out,
            "spatial_measure": "exact_cell_area_weighted",
            "signed_minimum": signed_min,
            "signed_maximum": signed_max,
            "positive_sum": positive_sum,
            "negative_sum_abs": negative_sum_abs,
            "positive_cell_fraction": num_positive / num_cells,
            "negative_cell_fraction": num_negative / num_cells,
            "zero_cell_fraction": num_zero / num_cells,
            "positive_quadratic_energy": positive_energy,
            "negative_quadratic_energy": negative_energy,
            "positive_to_negative_sum_ratio": sign_ratio,
            "positive_to_negative_ratio_status": sign_ratio_status,
            "max_positive_in_defect": max_pos_in,
            "min_negative_in_defect": min_neg_in,
            "max_positive_outside_defect": max_pos_out,
            "min_negative_outside_defect": min_neg_out,
        }
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
        gc.collect()


def relative_gain(
    line_value: float,
    box_value: float,
    *,
    direction: str,
    epsilon: float,
) -> float:
    if not math.isfinite(float(line_value)) or not math.isfinite(float(box_value)):
        raise ValueError("Relative gain inputs must be finite.")
    denominator = max(abs(float(line_value)), float(epsilon))
    if direction == "higher_is_better":
        return (float(box_value) - float(line_value)) / denominator
    if direction == "lower_is_better":
        return (float(line_value) - float(box_value)) / denominator
    raise ValueError(
        f"Metric direction must be higher_is_better or lower_is_better, got {direction!r}."
    )


def _numeric_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _compute_gains(rows: list[dict[str, Any]], *, epsilon: float) -> list[dict[str, Any]]:
    by_base = {
        (str(row["geometry"]), str(row["method"]), str(row["property"])): row
        for row in rows
    }
    gains: list[dict[str, Any]] = []

    for method in EXPECTED_METHODS:
        for prop in EXPECTED_PROJECT_PROPERTIES:
            line = by_base[("line", method, prop)]
            box = by_base[("box", method, prop)]
            for representation in REPRESENTATIONS:
                for metric, definition in METRIC_DEFINITIONS.items():
                    direction = str(definition["direction"])
                    if direction not in {"higher_is_better", "lower_is_better"}:
                        continue
                    column = f"{representation}__{metric}"
                    line_value = _numeric_or_none(line.get(column))
                    box_value = _numeric_or_none(box.get(column))
                    gain_column = f"box_vs_line_gain__{representation}__{metric}"
                    if line_value is None or box_value is None:
                        gain = None
                        status = "missing_metric"
                    else:
                        gain = relative_gain(
                            line_value,
                            box_value,
                            direction=direction,
                            epsilon=epsilon,
                        )
                        status = "ok"
                    line[gain_column] = gain
                    box[gain_column] = gain
                    gains.append(
                        {
                            "method": method,
                            "property": prop,
                            "representation": representation,
                            "metric": metric,
                            "direction": direction,
                            "line_value": line_value,
                            "box_value": box_value,
                            "gain_box_vs_line": gain,
                            "epsilon": epsilon,
                            "status": status,
                        }
                    )
    return gains


def _normalize_scores(values: Sequence[float], *, direction: str, tolerance: float) -> list[float]:
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    if span <= tolerance:
        return [0.5 for _ in values]
    if direction == "higher_is_better":
        return [(value - minimum) / span for value in values]
    if direction == "lower_is_better":
        return [(maximum - value) / span for value in values]
    raise ValueError(f"Unsupported ranking direction {direction!r}.")


def apply_multi_criteria_ranking(
    rows: list[dict[str, Any]],
    *,
    weights: Mapping[str, float],
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    weights = {str(k): float(v) for k, v in weights.items() if float(v) > 0.0}
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    ranking_records: list[dict[str, Any]] = []

    for prop in EXPECTED_PROJECT_PROPERTIES:
        candidates = [row for row in rows if row["property"] == prop]
        candidates.sort(key=lambda r: (str(r["geometry"]), str(r["method"])))
        if len(candidates) != len(GEOMETRIES) * len(EXPECTED_METHODS):
            raise ComparisonError(
                f"Ranking for property {prop} expected 8 candidates, got {len(candidates)}."
            )

        per_metric_scores: dict[str, list[float]] = {}
        for metric in weights:
            definition = METRIC_DEFINITIONS.get(metric)
            if definition is None:
                raise ComparisonError(f"No definition for ranking metric {metric!r}.")
            direction = str(definition["direction"])
            values: list[float] = []
            missing: list[str] = []
            column = f"article_safe__{metric}"
            for row in candidates:
                value = _numeric_or_none(row.get(column))
                if value is None:
                    missing.append(f"{row['geometry']}/{row['method']}")
                else:
                    values.append(value)
            if missing:
                raise ComparisonError(
                    f"Ranking metric {metric!r} is missing for property {prop}: {missing}."
                )
            tolerance = atol + rtol * max(max(abs(v) for v in values), 1.0)
            per_metric_scores[metric] = _normalize_scores(
                values,
                direction=direction,
                tolerance=tolerance,
            )

        scores: list[float] = []
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
            ranking_records.append(
                {
                    "geometry": row["geometry"],
                    "method": row["method"],
                    "property": prop,
                    "normalized_metrics": intermediate,
                    "weights": normalized_weights,
                    "composite_score": score,
                }
            )

        unique_scores = sorted(scores, reverse=True)
        tie_tolerance = atol + rtol
        for row, score in zip(candidates, scores):
            rank = 1 + sum(other > score + tie_tolerance for other in unique_scores)
            tie_size = sum(abs(other - score) <= tie_tolerance for other in unique_scores)
            row["ranking__rank"] = rank
            row["ranking__tie_group_size"] = tie_size
            for record in ranking_records:
                if (
                    record["geometry"] == row["geometry"]
                    and record["method"] == row["method"]
                    and record["property"] == prop
                ):
                    record["rank"] = rank
                    record["tie_group_size"] = tie_size
                    break

    return sorted(
        ranking_records,
        key=lambda r: (str(r["property"]), int(r["rank"]), str(r["geometry"]), str(r["method"])),
    )


def check_reflection_only_consistency(
    rows: Sequence[Mapping[str, Any]],
    *,
    rtol: float,
    atol: float,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    by_key = {
        (str(row["geometry"]), str(row["method"]), str(row["property"])): row
        for row in rows
    }
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    comparable_metrics = {
        "article_safe": SOURCE_METRIC_COLUMNS,
        "raw_signed": tuple(
            dict.fromkeys(
                [
                    *SOURCE_METRIC_COLUMNS,
                    *SIGNED_STATISTIC_COLUMNS,
                    "num_inside_defect",
                    "num_outside_defect",
                    "num_ghost_cells_removed",
                ]
            )
        ),
    }

    for prop in EXPECTED_PROJECT_PROPERTIES:
        line = by_key[("line", "standard_reflection_only", prop)]
        box = by_key[("box", "standard_reflection_only", prop)]
        for representation, metrics in comparable_metrics.items():
            for metric in metrics:
                column = f"{representation}__{metric}"
                a = _numeric_or_none(line.get(column))
                b = _numeric_or_none(box.get(column))
                if a is None and b is None:
                    passed = True
                    abs_diff = 0.0
                    rel_diff = 0.0
                    tolerance = atol
                    status = "both_missing_or_undefined"
                elif a is None or b is None:
                    passed = False
                    abs_diff = None
                    rel_diff = None
                    tolerance = None
                    status = "one_value_missing_or_undefined"
                else:
                    abs_diff = abs(a - b)
                    tolerance = atol + rtol * abs(b)
                    passed = abs_diff <= tolerance
                    rel_diff = abs_diff / abs(b) if b != 0.0 else (0.0 if abs_diff == 0.0 else None)
                    status = "ok" if passed else "mismatch"
                check = {
                    "method": "standard_reflection_only",
                    "property": prop,
                    "representation": representation,
                    "metric": metric,
                    "line_value": a,
                    "box_value": b,
                    "absolute_difference": abs_diff,
                    "relative_difference": rel_diff,
                    "rtol": rtol,
                    "atol": atol,
                    "tolerance": tolerance,
                    "passed": passed,
                    "status": status,
                }
                checks.append(check)
                if not passed:
                    failures.append(check)

    result = {
        "method": "standard_reflection_only",
        "formula": "abs(line-box) <= atol + rtol*abs(box)",
        "rtol": rtol,
        "atol": atol,
        "num_checks": len(checks),
        "num_failures": len(failures),
        "status": "ok" if not failures else "error",
        "checks": checks,
        "failures": failures,
    }
    if failures and raise_on_failure:
        first = failures[0]
        raise ReflectionConsistencyError(
            "reflection_only consistency failed: "
            f"method={first['method']}, property={first['property']}, "
            f"representation={first['representation']}, metric={first['metric']}, "
            f"line={first['line_value']!r}, box={first['box_value']!r}, "
            f"abs_diff={first['absolute_difference']!r}, "
            f"rel_diff={first['relative_difference']!r}, "
            f"tolerance={first['tolerance']!r}, rtol={rtol}, atol={atol}. "
            f"Total failures={len(failures)}."
        )
    return result


def _validate_card_dataset_headers(cards: Sequence[CardInput]) -> None:
    """Resolve every PVD/PVTU and all referenced pieces without loading fields."""

    for pvd_path in sorted({card.pvd_path for card in cards}, key=str):
        resolve_vtk_dataset(pvd_path, time_index=0)


def _build_consolidated_rows(
    *,
    cards: Sequence[CardInput],
    source_metrics: Mapping[CardKey, SourceMetricRecord],
    raw_metrics: Mapping[CardKey, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    card_by_key = {card.key: card for card in cards}
    rows: list[dict[str, Any]] = []

    for geometry in GEOMETRIES:
        for method in EXPECTED_METHODS:
            for prop in EXPECTED_PROJECT_PROPERTIES:
                article_key = CardKey(geometry, "article_safe", method, prop)
                raw_key = CardKey(geometry, "raw_signed", method, prop)
                source = source_metrics[article_key]
                raw = raw_metrics[raw_key]
                article_card = card_by_key[article_key]
                raw_card = card_by_key[raw_key]

                row: dict[str, Any] = {
                    "geometry": geometry,
                    "method": method,
                    "property": prop,
                    "validation__status": "ok",
                    "validation__representations_validated": 2,
                    "article_safe__metric_provenance": "read_exactly_from_validated_source_csv",
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

    if len(rows) != 40:
        raise ComparisonError(f"Expected 40 consolidated rows, produced {len(rows)}.")
    keys = [(row["geometry"], row["method"], row["property"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ComparisonError("Duplicate geometry/method/property rows after consolidation.")
    return rows


def run_single_defect_method_comparison(
    config: ComparisonConfig,
    *,
    logger: logging.Logger | None = None,
) -> ComparisonResult:
    logger = logger or logging.getLogger(__name__)
    started = time.perf_counter()

    cards, methods, properties, exclusions, defect_center, defect_radius = (
        discover_comparison_inputs(
            config.inputs.line_manifest,
            config.inputs.box_manifest,
        )
    )
    logger.info("Discovered and logically validated %d cards.", len(cards))
    logger.info("Methods: %s", ", ".join(methods))
    logger.info("Properties: %s", ", ".join(properties))
    if exclusions:
        logger.info("Explicit exclusions: %s", json.dumps(exclusions, sort_keys=True))

    _validate_card_dataset_headers(cards)
    logger.info("Resolved all PVD/PVTU/VTU datasets and referenced serial pieces.")

    source_metrics: dict[CardKey, SourceMetricRecord] = {}
    source_metrics.update(
        read_source_metrics_csv(config.inputs.line_metrics_csv, geometry="line")
    )
    source_metrics.update(
        read_source_metrics_csv(config.inputs.box_metrics_csv, geometry="box")
    )
    if len(source_metrics) != 40:
        raise SourceMetricError(
            f"Expected 40 validated article_safe source rows, got {len(source_metrics)}."
        )
    _validate_source_metric_card_alignment(cards, source_metrics)
    logger.info("Loaded and aligned 40 article_safe metric rows without redefining source metrics.")

    raw_metrics: dict[CardKey, Mapping[str, Any]] = {}
    raw_cards = [card for card in cards if card.key.representation == "raw_signed"]
    for index, card in enumerate(raw_cards, start=1):
        article_key = CardKey(card.key.geometry, "article_safe", card.key.method, card.key.property)
        expected_num_cells = int(source_metrics[article_key].numeric_metrics["num_cells"])
        logger.info(
            "[%d/%d] Streaming raw_signed card %s from %s",
            index,
            len(raw_cards),
            card.key.label(),
            card.pvd_path,
        )
        raw_metrics[card.key] = compute_raw_signed_metrics_streaming(
            card,
            defect_center=defect_center,
            defect_radius=defect_radius,
            false_max_threshold_fraction=config.false_max_threshold_fraction,
            false_max_min_distance_factor=config.false_max_min_distance_factor,
            max_false_max_candidates=config.max_false_max_candidates,
            remove_ghost_cells=config.remove_ghost_cells,
            expected_num_cells=expected_num_cells,
        )
        gc.collect()

    if len(raw_metrics) != 40:
        raise ComparisonError(f"Expected 40 raw_signed metric records, got {len(raw_metrics)}.")

    rows = _build_consolidated_rows(
        cards=cards,
        source_metrics=source_metrics,
        raw_metrics=raw_metrics,
    )
    gains = _compute_gains(rows, epsilon=config.gain_epsilon)
    rankings = apply_multi_criteria_ranking(
        rows,
        weights=config.ranking_weights,
        atol=config.atol,
        rtol=config.rtol,
    )
    reflection = check_reflection_only_consistency(
        rows,
        rtol=config.rtol,
        atol=config.atol,
        raise_on_failure=config.strict_reflection_check,
    )

    elapsed = time.perf_counter() - started
    logger.info("Comparison completed in %.6f s.", elapsed)
    return ComparisonResult(
        cards=tuple(cards),
        properties=tuple(properties),
        methods=tuple(methods),
        source_metrics=source_metrics,
        raw_signed_metrics=raw_metrics,
        consolidated_rows=tuple(rows),
        gains=tuple(gains),
        rankings=tuple(rankings),
        reflection_control=reflection,
        exclusions=tuple(exclusions),
        warnings=tuple(),
        defect_center=defect_center,
        defect_radius=defect_radius,
        elapsed_seconds=elapsed,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _csv_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    identity = ["geometry", "method", "property"]
    all_names = set().union(*(row.keys() for row in rows))
    return identity + sorted(all_names - set(identity))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = _csv_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _fmt_latex(value: Any, *, digits: int = 3) -> str:
    number = _numeric_or_none(value)
    if number is None:
        return "--"
    if number == 0.0:
        return "0"
    if abs(number) >= 1.0e4 or abs(number) < 1.0e-3:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}"


def _write_latex_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "% Auto-generated by postprocess_single_defect_method_comparison.py",
        "% Requires: \\usepackage{booktabs,longtable}",
        "% Arrows: $\\downarrow$ lower is better; $\\uparrow$ higher is better.",
        "% Composite score: min-max normalized weighted score from validated article_safe metrics.",
        r"\begin{longtable}{lllrrrrrr}",
        r"\caption{Single-defect image-quality comparison.}\label{tab:single_defect_method_comparison}\\",
        r"\toprule",
        r"Geometry & Method & Property & Loc. $\downarrow$ & $E_D/E$ $\uparrow$ & PAR $\uparrow$ & Contrast $\uparrow$ & False peaks $\downarrow$ & Score $\uparrow$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Geometry & Method & Property & Loc. $\downarrow$ & $E_D/E$ $\uparrow$ & PAR $\uparrow$ & Contrast $\uparrow$ & False peaks $\downarrow$ & Score $\uparrow$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        values = [
            _latex_escape(str(row["geometry"])),
            _latex_escape(str(row["method"])),
            _latex_escape(str(row["property"])),
            _fmt_latex(row.get("article_safe__localization_error")),
            _fmt_latex(row.get("article_safe__energy_fraction_in_defect")),
            _fmt_latex(row.get("article_safe__peak_to_artifact_ratio")),
            _fmt_latex(row.get("article_safe__contrast_defect_to_background")),
            _fmt_latex(row.get("article_safe__false_maxima_count"), digits=0),
            _fmt_latex(row.get("ranking__composite_score")),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            "",
            "% Abbreviations: Loc. = localization error; E_D/E = quadratic energy fraction",
            "% inside the defect; PAR = peak-to-artifact ratio.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_figures(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ComparisonError("matplotlib is required to generate comparison figures.") from exc

    methods = list(EXPECTED_METHODS)
    properties = list(EXPECTED_PROJECT_PROPERTIES)
    configurations = [(g, m) for g in GEOMETRIES for m in methods]
    row_map = {
        (str(r["geometry"]), str(r["method"]), str(r["property"])): r for r in rows
    }

    score_matrix = np.asarray(
        [
            [
                float(row_map[(geometry, method, prop)]["ranking__composite_score"])
                for geometry, method in configurations
            ]
            for prop in properties
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(12, 4.8))
    image = ax.imshow(score_matrix, aspect="auto")
    ax.set_xticks(range(len(configurations)))
    ax.set_xticklabels([f"{g}\n{m}" for g, m in configurations], rotation=35, ha="right")
    ax.set_yticks(range(len(properties)))
    ax.set_yticklabels(properties)
    ax.set_title("Composite article-safe image-quality score")
    fig.colorbar(image, ax=ax, label="score")
    fig.tight_layout()
    score_path = output_dir / "single_defect_method_comparison_scores.png"
    fig.savefig(score_path, dpi=180)
    plt.close(fig)

    gain_matrix = np.asarray(
        [
            [
                float(
                    row_map[("box", method, prop)][
                        "box_vs_line_gain__article_safe__energy_fraction_in_defect"
                    ]
                )
                for method in methods
            ]
            for prop in properties
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    image = ax.imshow(gain_matrix, aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticks(range(len(properties)))
    ax.set_yticklabels(properties)
    ax.set_title("Box-versus-line relative gain: defect energy fraction")
    fig.colorbar(image, ax=ax, label="relative gain")
    fig.tight_layout()
    gain_path = output_dir / "single_defect_method_comparison_line_box_gains.png"
    fig.savefig(gain_path, dpi=180)
    plt.close(fig)

    balance_matrix = np.empty((len(properties), len(configurations)), dtype=float)
    balance_matrix.fill(np.nan)
    for i, prop in enumerate(properties):
        for j, (geometry, method) in enumerate(configurations):
            ratio = _numeric_or_none(
                row_map[(geometry, method, prop)].get(
                    "raw_signed__positive_to_negative_sum_ratio"
                )
            )
            if ratio is not None and ratio > 0.0:
                balance_matrix[i, j] = math.log10(ratio)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    image = ax.imshow(balance_matrix, aspect="auto")
    ax.set_xticks(range(len(configurations)))
    ax.set_xticklabels([f"{g}\n{m}" for g, m in configurations], rotation=35, ha="right")
    ax.set_yticks(range(len(properties)))
    ax.set_yticklabels(properties)
    ax.set_title("Raw-signed balance: log10(positive sum / negative absolute sum)")
    fig.colorbar(image, ax=ax, label="log10 ratio")
    fig.tight_layout()
    balance_path = output_dir / "single_defect_method_comparison_signed_balance.png"
    fig.savefig(balance_path, dpi=180)
    plt.close(fig)

    return [score_path, gain_path, balance_path]


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    path = Path(path).resolve()
    stat = path.stat()
    result = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_file:
        result["sha256"] = _sha256(path)
    return result


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "meshio", "matplotlib"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = None
    return versions


def _git_commit(start: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=start,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _execution_timestamp(config: ComparisonConfig) -> str:
    if config.execution_timestamp is not None:
        return config.execution_timestamp
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch is not None:
        try:
            dt = datetime.fromtimestamp(int(source_date_epoch), tz=timezone.utc)
        except ValueError as exc:
            raise ComparisonError("SOURCE_DATE_EPOCH must be an integer Unix timestamp.") from exc
        return dt.isoformat()
    return datetime.now(timezone.utc).isoformat()


def _summary(result: ComparisonResult) -> dict[str, Any]:
    best_by_property: dict[str, Any] = {}
    for prop in EXPECTED_PROJECT_PROPERTIES:
        candidates = [r for r in result.rankings if r["property"] == prop]
        best_rank = min(int(r["rank"]) for r in candidates)
        best_by_property[prop] = [r for r in candidates if int(r["rank"]) == best_rank]

    gain_records = [g for g in result.gains if g["status"] == "ok"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "num_entries_discovered": len(result.cards),
        "num_cards_validated": len(result.cards),
        "num_consolidated_rows": len(result.consolidated_rows),
        "geometries": list(GEOMETRIES),
        "representations": list(REPRESENTATIONS),
        "methods": list(result.methods),
        "properties": list(result.properties),
        "defect_center": list(result.defect_center),
        "defect_radius": result.defect_radius,
        "best_rankings_by_property": best_by_property,
        "relative_gains": gain_records,
        "reflection_only_control": result.reflection_control,
        "exclusions": list(result.exclusions),
        "warnings": list(result.warnings),
    }


def write_comparison_outputs(
    result: ComparisonResult,
    config: ComparisonConfig,
    *,
    command: Sequence[str] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    logger = logger or logging.getLogger(__name__)
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
        "single_defect_method_comparison_line_box_gains.png",
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
    _write_csv(paths["csv"], rows)
    _write_latex_table(paths["latex"], rows)
    _write_json(paths["summary"], _summary(result))
    figure_paths = _make_figures(output_dir, rows)

    input_paths = {
        "line_metrics_csv": config.inputs.line_metrics_csv,
        "box_metrics_csv": config.inputs.box_metrics_csv,
        "line_manifest": config.inputs.line_manifest,
        "box_manifest": config.inputs.box_manifest,
    }
    pvd_metadata = []
    for card in result.cards:
        pvd_metadata.append(
            {
                "key": card.key.as_dict(),
                "field_name": card.expected_field_name,
                "pvd": _file_metadata(card.pvd_path, hash_file=True),
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
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "single_defect_method_comparison",
        "status": "ok",
        "execution_date_utc": _execution_timestamp(config),
        "command": list(command or ()),
        "inputs": {
            name: _file_metadata(Path(path), hash_file=True)
            for name, path in input_paths.items()
        },
        "card_inputs": pvd_metadata,
        "tolerances": {
            "rtol": config.rtol,
            "atol": config.atol,
            "gain_epsilon": config.gain_epsilon,
        },
        "defect": {
            "center": list(result.defect_center),
            "radius": result.defect_radius,
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "signed_statistic_definitions": {
            "positive_sum": "Cell-area-weighted integral of strictly positive raw density values.",
            "negative_sum_abs": "Absolute value of the unweighted sum of strictly negative raw cell values.",
            "positive_cell_fraction": "Number of strictly positive cells divided by the finite cell count.",
            "negative_cell_fraction": "Number of strictly negative cells divided by the finite cell count.",
            "positive_quadratic_energy": "Unweighted sum of squared strictly positive raw cell values.",
            "negative_quadratic_energy": "Unweighted sum of squared strictly negative raw cell values.",
            "positive_to_negative_sum_ratio": "positive_sum / negative_sum_abs; null when the denominator is zero and the numerator is non-zero.",
            "inside_outside_extrema": "Signed extrema evaluated using the manifest defect disk.",
        },
        "ranking": {
            "source_representation": "article_safe",
            "normalization": "per-property min-max; 0.5 when all candidates are equal within tolerance",
            "aggregation": "weighted arithmetic sum of normalized criteria",
            "weights_requested": dict(config.ranking_weights),
            "weights_normalized": {
                key: float(value) / sum(float(v) for v in config.ranking_weights.values())
                for key, value in config.ranking_weights.items()
            },
            "tie_rule": "same rank when composite scores differ by at most atol+rtol",
        },
        "relative_gain": {
            "higher_is_better": "(box-line)/max(abs(line), epsilon)",
            "lower_is_better": "(line-box)/max(abs(line), epsilon)",
            "epsilon": config.gain_epsilon,
        },
        "memory_strategy": {
            "vtk": "one serial VTU piece and one cell block at a time",
            "raw_signed": "one card at a time",
            "half_maximum": "temporary on-disk float64 spool; deleted after each card",
            "article_safe": "validated metrics read exactly from source CSV; no full-field recomputation",
        },
        "validation": {
            "num_cards": len(result.cards),
            "num_consolidated_rows": len(result.consolidated_rows),
            "reflection_only": result.reflection_control,
            "exclusions": list(result.exclusions),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": _dependency_versions(),
            "git_commit": _git_commit(Path.cwd()),
            "elapsed_seconds": result.elapsed_seconds,
        },
        "files_produced": [str(path) for path in generated_files],
    }
    _write_json(paths["manifest"], manifest)

    if not paths["log"].exists():
        paths["log"].write_text(
            "single_defect_method_comparison completed successfully\n"
            f"cards={len(result.cards)}\n"
            f"rows={len(result.consolidated_rows)}\n"
            f"reflection_status={result.reflection_control['status']}\n",
            encoding="utf-8",
        )

    logger.info("Wrote comparison CSV: %s", paths["csv"])
    logger.info("Wrote LaTeX table: %s", paths["latex"])
    logger.info("Wrote summary JSON: %s", paths["summary"])
    logger.info("Wrote reproducibility manifest: %s", paths["manifest"])
    for path in figure_paths:
        logger.info("Wrote figure: %s", path)
    return {**paths, **{f"figure_{i+1}": p for i, p in enumerate(figure_paths)}}
