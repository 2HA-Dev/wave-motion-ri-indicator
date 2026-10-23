from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_file(path: str | Path) -> bool:
    return Path(path).exists()


def _finite_float(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def check_vector_array_export_manifest(
    manifest_path: Path,
    *,
    require_vector_array: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = _load_json(manifest_path)

    observation_kind = manifest.get("observation_kind")
    vector_comparison = manifest.get("vector_array_comparison")
    adjoint_rhs = manifest.get("adjoint_rhs", {})
    exports = manifest.get("exports", {})
    vector_export = exports.get("vector_array")

    errors: list[str] = []

    if require_vector_array and observation_kind != "vector_array":
        errors.append(f"Expected observation_kind='vector_array', got {observation_kind!r}.")

    if require_vector_array and vector_comparison is None:
        errors.append("Missing vector_array_comparison block.")

    if require_vector_array and vector_export is None:
        errors.append("Missing exports.vector_array block.")

    resolved_sigma = manifest.get("resolved_receiver_sigma")
    if require_vector_array and not _finite_float(resolved_sigma):
        errors.append("resolved_receiver_sigma is missing or non-finite.")

    adjoint_kind = adjoint_rhs.get("kind")
    if require_vector_array and adjoint_kind != "vector_array_full_adjoint_rhs_result":
        errors.append(
            "Expected adjoint_rhs.kind='vector_array_full_adjoint_rhs_result', "
            f"got {adjoint_kind!r}."
        )

    max_residual = None
    if vector_comparison is not None:
        max_residual = vector_comparison.get("max_abs_residual")
        if not _finite_float(max_residual):
            errors.append("vector_array_comparison.max_abs_residual is missing or non-finite.")

    max_rhs_norm = adjoint_rhs.get("max_rhs_norm")
    if require_vector_array and not _finite_float(max_rhs_norm):
        errors.append("adjoint_rhs.max_rhs_norm is missing or non-finite.")

    file_checks: dict[str, bool] = {}
    if vector_export is not None:
        if "manifest" in vector_export:
            file_checks["vector_manifest"] = _check_file(vector_export["manifest"])
        if "residual_csv" in vector_export:
            file_checks["residual_csv"] = _check_file(vector_export["residual_csv"])

        for key, value in vector_export.get("residual_npz", {}).items():
            file_checks[f"residual_npz_{key}"] = _check_file(value)

        for key, value in vector_export.get("signals", {}).items():
            file_checks[f"signal_{key}"] = _check_file(value)

        for key, value in vector_export.get("residual_figures", {}).items():
            file_checks[f"residual_figure_{key}"] = _check_file(value)

    missing_files = [name for name, ok in file_checks.items() if not ok]
    for name in missing_files:
        errors.append(f"Missing vector_array export file: {name}")

    gamma1_receivers = None
    gamma2_receivers = None

    if vector_comparison is not None:
        residuals = vector_comparison.get("residuals", {})
        reflection = residuals.get("reflection")
        transmission = residuals.get("transmission")

        if reflection is not None:
            gamma1_receivers = reflection.get("num_receivers")
        if transmission is not None:
            gamma2_receivers = transmission.get("num_receivers")

    return {
        "status": "ok" if not errors else "fail",
        "manifest": str(manifest_path),
        "observation_kind": observation_kind,
        "resolved_receiver_sigma": resolved_sigma,
        "adjoint_rhs_kind": adjoint_kind,
        "max_abs_residual": max_residual,
        "max_rhs_norm": max_rhs_norm,
        "gamma1_num_receivers": gamma1_receivers,
        "gamma2_num_receivers": gamma2_receivers,
        "file_checks": file_checks,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check a vector_array inverse export manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--allow-non-vector-array",
        action="store_true",
        help="Do not require observation_kind=vector_array.",
    )
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)

    result = check_vector_array_export_manifest(
        args.manifest,
        require_vector_array=not args.allow_non_vector_array,
    )

    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))

    if result["status"] != "ok":
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
