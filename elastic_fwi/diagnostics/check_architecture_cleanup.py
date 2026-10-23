from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_DIRS = (
    "elastic_fwi/config",
    "elastic_fwi/metrics",
    "elastic_fwi/campaigns",
    "elastic_fwi/inversion",
    "elastic_fwi/inverse",
    "elastic_fwi/tracking",
)

EXPECTED_INIT_DIRS = (
    "elastic_fwi",
    "elastic_fwi/config",
    "elastic_fwi/metrics",
    "elastic_fwi/campaigns",
    "elastic_fwi/inversion",
    "elastic_fwi/inverse",
    "elastic_fwi/tracking",
)

STANDARD_ROOT_FILES = {
    "pyproject.toml",
    "README.md",
    "README.rst",
    "LICENSE",
    "LICENSE.txt",
    ".gitignore",
}


def _py_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.py"))


def _grep_imports(root: Path, pattern: str) -> list[str]:
    found: list[str] = []
    rx = re.compile(pattern)

    for path in _py_files(root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                found.append(f"{path}:{lineno}:{line.strip()}")

    return found


def _root_suspicious_files(root: Path) -> list[str]:
    suspicious: list[str] = []
    allowed_suffixes = {
        ".py",
        ".md",
        ".rst",
        ".toml",
        ".txt",
        ".yml",
        ".yaml",
        ".json",
        ".gitignore",
    }

    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.name in STANDARD_ROOT_FILES:
            continue
        if path.suffix in allowed_suffixes:
            continue
        suspicious.append(path.name)

    return suspicious


def _empty_dirs(root: Path) -> list[str]:
    empty: list[str] = []

    for rel in EXPECTED_DIRS:
        path = root / rel
        if path.exists() and path.is_dir():
            has_py = any(path.glob("*.py"))
            has_md = any(path.glob("*.md"))
            if not has_py and not has_md:
                empty.append(rel)

    return empty


def _missing_dirs(root: Path) -> list[str]:
    return [rel for rel in EXPECTED_DIRS if not (root / rel).is_dir()]


def _missing_init(root: Path) -> list[str]:
    return [
        rel
        for rel in EXPECTED_INIT_DIRS
        if not (root / rel / "__init__.py").is_file()
    ]


def _direct_observation_config_duplicates(root: Path) -> dict[str, Any]:
    target_files: list[Path] = []

    for path in _py_files(root / "elastic_fwi"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "class DirectObservationPipelineConfig" in text:
            target_files.append(path)

    records: list[dict[str, Any]] = []

    for path in target_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(text)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DirectObservationPipelineConfig":
                names: list[str] = []
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        names.append(item.target.id)

                duplicates = sorted({name for name in names if names.count(name) > 1})
                records.append(
                    {
                        "file": str(path),
                        "num_fields": len(names),
                        "duplicates": duplicates,
                    }
                )

    return {
        "num_classes": len(records),
        "records": records,
        "has_duplicates": any(record["duplicates"] for record in records),
    }


def build_report(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root).resolve()

    missing_dirs = _missing_dirs(root)
    missing_init = _missing_init(root)
    empty_dirs = _empty_dirs(root)
    suspicious_root_files = _root_suspicious_files(root)

    inverse_imports = _grep_imports(
        root / "elastic_fwi",
        r"elastic_fwi\.inverse|from\s+elastic_fwi\.inverse|import\s+elastic_fwi\.inverse",
    )
    inversion_imports = _grep_imports(
        root / "elastic_fwi",
        r"elastic_fwi\.inversion|from\s+elastic_fwi\.inversion",
    )

    config_duplicates = _direct_observation_config_duplicates(root)

    status = "ok"
    warnings: list[str] = []
    failures: list[str] = []

    if missing_dirs:
        failures.append("missing_expected_directories")
    if missing_init:
        failures.append("missing_init_files")
    if suspicious_root_files:
        warnings.append("suspicious_root_files")
    if empty_dirs:
        warnings.append("empty_or_placeholder_dirs")
    if config_duplicates["has_duplicates"]:
        failures.append("duplicated_direct_observation_config_fields")

    if failures:
        status = "failed"
    elif warnings:
        status = "warning"

    return {
        "kind": "architecture_cleanup_check",
        "status": status,
        "root": str(root),
        "missing_dirs": missing_dirs,
        "missing_init": missing_init,
        "empty_dirs": empty_dirs,
        "suspicious_root_files": suspicious_root_files,
        "inverse_imports": inverse_imports,
        "num_inverse_imports": len(inverse_imports),
        "num_inversion_imports": len(inversion_imports),
        "direct_observation_config": config_duplicates,
        "warnings": warnings,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check architectural cleanup status.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("cleanup_reports/architecture_cleanup_check.json"),
    )
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args.root)

    print(f"[ARCHITECTURE CLEANUP CHECK] status={report['status']}")
    print("missing_dirs =", report["missing_dirs"])
    print("missing_init =", report["missing_init"])
    print("empty_dirs =", report["empty_dirs"])
    print("suspicious_root_files =", report["suspicious_root_files"])
    print("num_inverse_imports =", report["num_inverse_imports"])
    print("num_inversion_imports =", report["num_inversion_imports"])
    print("direct_config_duplicates =", report["direct_observation_config"]["records"])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("output_json =", args.output_json)

    if args.strict and report["status"] != "ok":
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
