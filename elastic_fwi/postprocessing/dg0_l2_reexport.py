from __future__ import annotations

"""Re-export legacy DG0 cell-integral maps as true DG0/L2 densities.

This module is VTK-only.  It never calls a forward, adjoint, or
residual-wave solver.  It converts 20 raw-signed maps stored as per-cell
integrals in place (via temporary trees and an atomic directory swap) and
reconstructs the 20 ``article_safe`` maps from the resulting densities.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import math
import shutil

import numpy as np

from elastic_fwi.postprocessing.gradient_validation import validate_gradient_decomposition
from elastic_fwi.postprocessing.vtk_cell_fields import (
    _cell_geometry_2d,
    _load_meshio_piece,
    _scalar_block,
    iter_vtk_cell_field_chunks,
    resolve_vtk_dataset,
)


MODALITIES = (
    "classical_full",
    "standard_reflection_only",
    "standard_transmission_only",
    "residual_interaction",
)
PARAMETERS = ("rho", "C11", "C22", "C12", "C66")
SPATIAL_REPRESENTATION = "DG0_L2_cell_average_density"


@dataclass(frozen=True)
class MapPair:
    modality: str
    parameter: str
    field_name: str
    raw_pvd: Path
    safe_pvd: Path


def _filename(modality: str, parameter: str) -> str:
    if modality == "residual_interaction":
        return f"residual_interaction_{parameter}.pvd"
    return f"material_parameter_gradients_gradient_{parameter}.pvd"


def _field_name(modality: str, parameter: str) -> str:
    if modality == "residual_interaction":
        return f"residual_interaction_{parameter}"
    return f"gradient_{parameter}"


def discover_20_map_pairs(campaign_dir: Path) -> tuple[MapPair, ...]:
    inverse = Path(campaign_dir).expanduser().resolve() / "inverse"
    pairs: list[MapPair] = []
    for modality in MODALITIES:
        for parameter in PARAMETERS:
            filename = _filename(modality, parameter)
            raw = inverse / "gradient_combinations_raw_signed" / modality / "vtk" / filename
            safe = inverse / "gradient_combinations" / modality / "vtk" / filename
            if not raw.is_file():
                raise FileNotFoundError(f"Missing legacy raw-signed map: {raw}")
            if not safe.is_file():
                raise FileNotFoundError(f"Missing legacy article-safe map: {safe}")
            pairs.append(
                MapPair(
                    modality=modality,
                    parameter=parameter,
                    field_name=_field_name(modality, parameter),
                    raw_pvd=raw,
                    safe_pvd=safe,
                )
            )
    if len(pairs) != 20:
        raise RuntimeError(f"Expected 20 map pairs, got {len(pairs)}.")
    return tuple(pairs)


def _cell_density_blocks(mesh: Any, field_name: str, *, piece: Path) -> list[np.ndarray]:
    points = np.asarray(mesh.points, dtype=np.float64)
    blocks = list(mesh.cells)
    field_blocks = mesh.cell_data.get(field_name)
    if field_blocks is None:
        raise RuntimeError(
            f"Cell field {field_name!r} is missing from {piece}; "
            f"available={sorted(mesh.cell_data)}"
        )
    if len(field_blocks) != len(blocks):
        raise RuntimeError(
            f"Field/block count mismatch for {field_name!r} in {piece}: "
            f"{len(field_blocks)} data blocks vs {len(blocks)} cell blocks."
        )

    densities: list[np.ndarray] = []
    for index, (block, values) in enumerate(zip(blocks, field_blocks)):
        connectivity = np.asarray(block.data, dtype=np.int64)
        _, measures = _cell_geometry_2d(
            points=points,
            connectivity=connectivity,
            cell_type=str(block.type),
        )
        scalar = _scalar_block(np.asarray(values), field_name=field_name, piece=piece)
        if scalar.shape != measures.shape:
            raise RuntimeError(
                f"Cell/value/measure mismatch for {field_name!r} in {piece}, block {index}."
            )
        densities.append(np.asarray(scalar / measures, dtype=np.float64))
    return densities


def _replace_scalar_blocks(mesh: Any, field_name: str, blocks: list[np.ndarray]) -> None:
    current = mesh.cell_data.get(field_name)
    if current is None or len(current) != len(blocks):
        raise RuntimeError(f"Cannot replace missing/incompatible field {field_name!r}.")
    replacement: list[np.ndarray] = []
    for old, new in zip(current, blocks):
        arr = np.asarray(old)
        values = np.asarray(new, dtype=np.float64)
        if arr.ndim == 1:
            replacement.append(values)
        elif arr.ndim == 2 and arr.shape[1] == 1:
            replacement.append(values[:, None])
        else:
            raise RuntimeError(
                f"Expected scalar cell field {field_name!r}; observed shape={arr.shape}."
            )
    mesh.cell_data[field_name] = replacement


def _verify_piece_geometry(raw_mesh: Any, safe_mesh: Any, raw_piece: Path, safe_piece: Path) -> None:
    if not np.array_equal(np.asarray(raw_mesh.points), np.asarray(safe_mesh.points)):
        if not np.allclose(raw_mesh.points, safe_mesh.points, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"Raw/safe point geometry differs: {raw_piece} vs {safe_piece}")
    if len(raw_mesh.cells) != len(safe_mesh.cells):
        raise RuntimeError(f"Raw/safe cell-block counts differ: {raw_piece} vs {safe_piece}")
    for a, b in zip(raw_mesh.cells, safe_mesh.cells):
        if str(a.type) != str(b.type) or not np.array_equal(a.data, b.data):
            raise RuntimeError(f"Raw/safe connectivity differs: {raw_piece} vs {safe_piece}")


def _write_mesh(path: Path, mesh: Any) -> None:
    import meshio

    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(path, mesh, file_format="vtu", binary=True)


def _relative_piece(piece: Path, root: Path) -> Path:
    try:
        return piece.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"VTU piece {piece} is outside expected tree {root}.") from exc


def _corrected_global_max(pair: MapPair) -> float:
    dataset = resolve_vtk_dataset(pair.raw_pvd)
    maximum = 0.0
    for piece in dataset.piece_paths:
        mesh = _load_meshio_piece(piece)
        for density in _cell_density_blocks(mesh, pair.field_name, piece=piece):
            if density.size:
                maximum = max(maximum, float(np.max(np.abs(density))))
    return maximum


def _rewrite_pair(
    pair: MapPair,
    *,
    raw_root: Path,
    safe_root: Path,
    raw_temp_root: Path,
    safe_temp_root: Path,
    global_max: float,
) -> dict[str, Any]:
    raw_dataset = resolve_vtk_dataset(pair.raw_pvd)
    safe_dataset = resolve_vtk_dataset(pair.safe_pvd)
    if len(raw_dataset.piece_paths) != len(safe_dataset.piece_paths):
        raise RuntimeError(
            f"Raw/safe partition count mismatch for {pair.modality}/{pair.parameter}."
        )

    max_abs_error = 0.0
    num_cells = 0
    for raw_piece, safe_piece in zip(raw_dataset.piece_paths, safe_dataset.piece_paths):
        raw_mesh = _load_meshio_piece(raw_piece)
        safe_mesh = _load_meshio_piece(safe_piece)
        _verify_piece_geometry(raw_mesh, safe_mesh, raw_piece, safe_piece)

        density_blocks = _cell_density_blocks(raw_mesh, pair.field_name, piece=raw_piece)
        if global_max > 0.0:
            safe_blocks = [np.abs(values) / global_max for values in density_blocks]
        else:
            safe_blocks = [np.zeros_like(values) for values in density_blocks]

        _replace_scalar_blocks(raw_mesh, pair.field_name, density_blocks)
        _replace_scalar_blocks(safe_mesh, pair.field_name, safe_blocks)

        raw_out = raw_temp_root / _relative_piece(raw_piece, raw_root)
        safe_out = safe_temp_root / _relative_piece(safe_piece, safe_root)
        _write_mesh(raw_out, raw_mesh)
        _write_mesh(safe_out, safe_mesh)
        num_cells += sum(int(values.size) for values in density_blocks)

        for density, safe_values in zip(density_blocks, safe_blocks):
            expected = np.abs(density) / global_max if global_max > 0.0 else np.zeros_like(density)
            if expected.size:
                max_abs_error = max(
                    max_abs_error,
                    float(np.max(np.abs(expected - safe_values))),
                )

    return {
        "modality": pair.modality,
        "parameter": pair.parameter,
        "field_name": pair.field_name,
        "global_density_max_abs": float(global_max),
        "num_piece_cells_including_ghosts": int(num_cells),
        "article_safe_reconstruction_max_abs_error_prewrite": float(max_abs_error),
        "status": "ok",
    }


def _mapped_pvd(original: Path, original_root: Path, temp_root: Path) -> Path:
    return temp_root / original.resolve().relative_to(original_root.resolve())


def _validate_rewritten_pair(
    raw_pvd: Path,
    safe_pvd: Path,
    field_name: str,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    raw_iter = iter(iter_vtk_cell_field_chunks(raw_pvd, field_name, remove_ghost_cells=True))
    safe_iter = iter(iter_vtk_cell_field_chunks(safe_pvd, field_name, remove_ghost_cells=True))
    raw_chunks: list[tuple[np.ndarray, np.ndarray]] = []
    maximum = 0.0
    while True:
        try:
            chunk = next(raw_iter)
        except StopIteration:
            break
        values = np.asarray(chunk.values, dtype=np.float64)
        maximum = max(maximum, float(np.max(np.abs(values))) if values.size else 0.0)
        raw_chunks.append((values.copy(), np.asarray(chunk.cell_measures, dtype=np.float64).copy()))

    max_error = 0.0
    l2_error_sq = 0.0
    l2_ref_sq = 0.0
    count = 0
    for raw_values, measures in raw_chunks:
        try:
            safe_chunk = next(safe_iter)
        except StopIteration as exc:
            raise RuntimeError(f"Safe map ended before raw map: {safe_pvd}") from exc
        safe_values = np.asarray(safe_chunk.values, dtype=np.float64)
        expected = np.abs(raw_values) / maximum if maximum > 0.0 else np.zeros_like(raw_values)
        if safe_values.shape != expected.shape:
            raise RuntimeError(f"Raw/safe shape mismatch in {safe_pvd}.")
        error = safe_values - expected
        if error.size:
            max_error = max(max_error, float(np.max(np.abs(error))))
        l2_error_sq += float(np.sum(error * error * measures, dtype=np.float64))
        l2_ref_sq += float(np.sum(expected * expected * measures, dtype=np.float64))
        count += int(error.size)
    try:
        next(safe_iter)
    except StopIteration:
        pass
    else:
        raise RuntimeError(f"Safe map contains extra chunks: {safe_pvd}")

    l2_error = math.sqrt(l2_error_sq)
    relative_l2 = l2_error / math.sqrt(l2_ref_sq) if l2_ref_sq > 0.0 else l2_error
    ok = max_error <= atol + rtol and relative_l2 <= atol + rtol
    return {
        "status": "ok" if ok else "failed",
        "num_owned_cells": count,
        "article_safe_max_abs_error": max_error,
        "article_safe_relative_area_weighted_l2_error": relative_l2,
        "raw_density_global_max_abs": maximum,
    }


def regenerate_legacy_campaign_dg0_l2(
    campaign_dir: Path,
    *,
    backup_suffix: str = "_legacy_cell_integral",
    overwrite_backup: bool = False,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> dict[str, Any]:
    campaign = Path(campaign_dir).expanduser().resolve()
    inverse = campaign / "inverse"
    raw_root = inverse / "gradient_combinations_raw_signed"
    safe_root = inverse / "gradient_combinations"
    if not raw_root.is_dir() or not safe_root.is_dir():
        raise FileNotFoundError("Expected both legacy gradient export directories under inverse/.")

    pairs = discover_20_map_pairs(campaign)
    raw_temp = inverse / ".gradient_combinations_raw_signed_dg0_l2_tmp"
    safe_temp = inverse / ".gradient_combinations_dg0_l2_tmp"
    for tmp in (raw_temp, safe_temp):
        if tmp.exists():
            shutil.rmtree(tmp)
    shutil.copytree(raw_root, raw_temp)
    shutil.copytree(safe_root, safe_temp)

    rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    try:
        for index, pair in enumerate(pairs, start=1):
            maximum = _corrected_global_max(pair)
            row = _rewrite_pair(
                pair,
                raw_root=raw_root,
                safe_root=safe_root,
                raw_temp_root=raw_temp,
                safe_temp_root=safe_temp,
                global_max=maximum,
            )
            row["index"] = index
            rows.append(row)

        for pair in pairs:
            raw_pvd = _mapped_pvd(pair.raw_pvd, raw_root, raw_temp)
            safe_pvd = _mapped_pvd(pair.safe_pvd, safe_root, safe_temp)
            check = _validate_rewritten_pair(
                raw_pvd,
                safe_pvd,
                pair.field_name,
                rtol=rtol,
                atol=atol,
            )
            validation_rows.append({
                "modality": pair.modality,
                "parameter": pair.parameter,
                **check,
            })

        for parameter in PARAMETERS:
            def raw_path(modality: str) -> Path:
                pair = next(
                    item for item in pairs
                    if item.modality == modality and item.parameter == parameter
                )
                return _mapped_pvd(pair.raw_pvd, raw_root, raw_temp)

            decomposition = validate_gradient_decomposition(
                classical_path=raw_path("classical_full"),
                reflection_path=raw_path("standard_reflection_only"),
                transmission_path=raw_path("standard_transmission_only"),
                field_name=f"gradient_{parameter}",
                input_semantics="l2-density",
                acceptance_relative_l2_tolerance=1.0e-6,
                acceptance_relative_linf_tolerance=1.0e-6,
            )
            decomposition_rows.append({"parameter": parameter, **decomposition})

        if not all(row["status"] == "ok" for row in validation_rows):
            raise RuntimeError("Corrected article-safe reconstruction validation failed.")
        if not all(row.get("status") == "ok" for row in decomposition_rows):
            raise RuntimeError("Corrected full=reflection+transmission decomposition failed.")

        raw_backup = inverse / f"gradient_combinations_raw_signed{backup_suffix}"
        safe_backup = inverse / f"gradient_combinations{backup_suffix}"
        for backup in (raw_backup, safe_backup):
            if backup.exists():
                if not overwrite_backup:
                    raise FileExistsError(
                        f"Backup already exists: {backup}. Use --overwrite-backup only deliberately."
                    )
                shutil.rmtree(backup)

        raw_root.rename(raw_backup)
        try:
            safe_root.rename(safe_backup)
            raw_temp.rename(raw_root)
            safe_temp.rename(safe_root)
        except Exception:
            if not raw_root.exists() and raw_backup.exists():
                raw_backup.rename(raw_root)
            if not safe_root.exists() and safe_backup.exists():
                safe_backup.rename(safe_root)
            raise

        report = {
            "kind": "legacy_dg0_l2_reexport",
            "format_version": "1.0",
            "status": "ok",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "campaign_dir": str(campaign),
            "wave_solves_rerun": False,
            "input_semantics": "DG0_cell_integral",
            "output_spatial_representation": SPATIAL_REPRESENTATION,
            "cell_measure_normalization": True,
            "article_safe_definition": "abs(raw_signed_density)/global_max_abs(raw_signed_density)",
            "num_raw_signed_maps": 20,
            "num_article_safe_maps": 20,
            "raw_backup": str(raw_backup),
            "article_safe_backup": str(safe_backup),
            "rewrite_rows": rows,
            "article_safe_validation": validation_rows,
            "gradient_decomposition": decomposition_rows,
            "tolerances": {"rtol": rtol, "atol": atol},
        }
        manifest = inverse / "dg0_l2_reexport_manifest.json"
        manifest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["manifest_path"] = str(manifest)
        return report
    except Exception:
        for tmp in (raw_temp, safe_temp):
            if tmp.exists():
                shutil.rmtree(tmp)
        raise
