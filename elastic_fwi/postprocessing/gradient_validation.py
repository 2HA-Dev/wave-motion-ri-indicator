from __future__ import annotations

"""Validation utilities for signed material-gradient exports.

The routines in this module are deliberately independent from plotting.  They
stream PVD/PVTU/VTU pieces, preserve DG0 cell-area weights, and provide two
checks:

* integrity of the distributed VTK partition;
* pointwise identity ``classical = reflection + transmission``.
"""

from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from elastic_fwi.postprocessing.vtk_cell_fields import (
    CellFieldChunk,
    VTKCellFieldError,
    _cell_geometry_2d,
    _corner_count_for_cell_type,
    _ghost_mask_for_block,
    _load_meshio_piece,
    _normalize_cell_type,
    iter_vtk_cell_field_chunks,
    resolve_vtk_dataset,
)


VALIDATION_FORMAT_VERSION = "1.2"
MATERIAL_PARAMETERS = ("rho", "C11", "C22", "C12", "C66")


@dataclass(frozen=True)
class PartitionGeometryChunk:
    source_piece: Path
    cell_type: str
    geometry_hash_primary: np.ndarray
    geometry_hash_secondary: np.ndarray
    cell_measures: np.ndarray
    original_cell_ids: np.ndarray | None
    num_cells_before_ghost_filter: int
    num_ghost_cells_removed: int
    ghost_array_present: bool

    @property
    def num_cells(self) -> int:
        return int(self.cell_measures.size)


def _fixed_type_seed(cell_type: str, corner_count: int) -> tuple[np.uint64, np.uint64]:
    payload = f"{_normalize_cell_type(cell_type)}:{corner_count}".encode("utf-8")
    digest = blake2b(payload, digest_size=16).digest()
    return (
        np.uint64(int.from_bytes(digest[:8], "little", signed=False)),
        np.uint64(int.from_bytes(digest[8:], "little", signed=False)),
    )


def canonical_cell_geometry_hashes(
    *,
    points: np.ndarray,
    connectivity: np.ndarray,
    cell_type: str,
    coordinate_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute two order-invariant 64-bit hashes per 2-D cell.

    Coordinates are quantized using ``coordinate_tolerance`` and cell vertices
    are lexicographically sorted.  Two independent hashes make an accidental
    collision negligible for the intended partition-integrity diagnostic.
    """

    tolerance = float(coordinate_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("coordinate_tolerance must be finite and strictly positive.")

    points = np.asarray(points, dtype=np.float64)
    connectivity = np.asarray(connectivity, dtype=np.int64)
    if connectivity.ndim != 2:
        raise ValueError("connectivity must be two-dimensional.")
    if connectivity.shape[0] == 0:
        empty = np.empty(0, dtype=np.uint64)
        return empty, empty.copy()

    corner_count = _corner_count_for_cell_type(cell_type, connectivity.shape[1])
    vertices = points[connectivity[:, :corner_count], :2]
    quantized = np.rint(vertices / tolerance).astype(np.int64)
    order = np.lexsort((quantized[:, :, 1], quantized[:, :, 0]), axis=1)
    canonical = np.take_along_axis(quantized, order[:, :, None], axis=1)
    words = np.ascontiguousarray(canonical.reshape(canonical.shape[0], -1)).view(
        np.uint64
    )

    seed1, seed2 = _fixed_type_seed(cell_type, corner_count)
    h1 = np.full(canonical.shape[0], seed1 ^ np.uint64(1469598103934665603), dtype=np.uint64)
    h2 = np.full(canonical.shape[0], seed2 ^ np.uint64(1099511628211), dtype=np.uint64)
    prime1 = np.uint64(1099511628211)
    prime2 = np.uint64(14029467366897019727)

    for column in range(words.shape[1]):
        word = words[:, column]
        h1 = (h1 ^ word) * prime1
        h2 = (h2 + word + np.uint64(column + 1)) * prime2
        h2 ^= h2 >> np.uint64(29)

    return h1, h2


def _original_cell_ids_for_block(mesh: Any, block_index: int, num_cells: int) -> np.ndarray | None:
    blocks = getattr(mesh, "cell_data", {}).get("vtkOriginalCellIds")
    if blocks is None or block_index >= len(blocks):
        return None
    values = np.asarray(blocks[block_index]).reshape(-1)
    if values.size != num_cells:
        raise VTKCellFieldError(
            "vtkOriginalCellIds size does not match the associated cell block."
        )
    return np.asarray(values, dtype=np.int64)


def iter_vtk_partition_geometry_chunks(
    path: str | Path,
    *,
    time_index: int = 0,
    coordinate_tolerance: float = 1.0e-11,
    remove_ghost_cells: bool = True,
) -> Iterator[PartitionGeometryChunk]:
    """Yield compact geometry metadata for every distributed VTK cell block."""

    dataset = resolve_vtk_dataset(path, time_index=time_index)
    for piece in dataset.piece_paths:
        mesh = _load_meshio_piece(piece)
        points = np.asarray(getattr(mesh, "points", np.empty((0, 3))), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 2:
            raise VTKCellFieldError(f"Invalid point array in {piece}: {points.shape}.")

        for block_index, block in enumerate(getattr(mesh, "cells", [])):
            connectivity = np.asarray(getattr(block, "data", block), dtype=np.int64)
            cell_type = str(getattr(block, "type", "cell"))
            num_before = int(connectivity.shape[0])
            _, measures = _cell_geometry_2d(
                points=points,
                connectivity=connectivity,
                cell_type=cell_type,
            )
            h1, h2 = canonical_cell_geometry_hashes(
                points=points,
                connectivity=connectivity,
                cell_type=cell_type,
                coordinate_tolerance=coordinate_tolerance,
            )
            original_ids = _original_cell_ids_for_block(mesh, block_index, num_before)
            keep = _ghost_mask_for_block(mesh, block_index, num_before)
            ghost_present = keep is not None
            removed = 0
            if remove_ghost_cells and keep is not None:
                removed = int(num_before - np.count_nonzero(keep))
                measures = measures[keep]
                h1 = h1[keep]
                h2 = h2[keep]
                if original_ids is not None:
                    original_ids = original_ids[keep]

            yield PartitionGeometryChunk(
                source_piece=piece,
                cell_type=cell_type,
                geometry_hash_primary=np.asarray(h1, dtype=np.uint64),
                geometry_hash_secondary=np.asarray(h2, dtype=np.uint64),
                cell_measures=np.asarray(measures, dtype=np.float64),
                original_cell_ids=original_ids,
                num_cells_before_ghost_filter=num_before,
                num_ghost_cells_removed=removed,
                ghost_array_present=ghost_present,
            )
        del mesh, points


def inspect_vtk_partition_integrity(
    path: str | Path,
    *,
    time_index: int = 0,
    coordinate_tolerance: float = 1.0e-11,
    expected_num_cells: int | None = None,
    expected_area: float | None = None,
    area_rtol: float = 1.0e-9,
    area_atol: float = 1.0e-8,
) -> dict[str, Any]:
    """Validate uniqueness and global area of a PVD/PVTU partition."""

    dataset = resolve_vtk_dataset(path, time_index=time_index)
    primary: list[np.ndarray] = []
    secondary: list[np.ndarray] = []
    measures: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    ids_complete = True
    num_before = 0
    num_removed = 0
    num_chunks = 0
    chunks_with_ghost_array = 0
    cell_types: set[str] = set()

    for chunk in iter_vtk_partition_geometry_chunks(
        path,
        time_index=time_index,
        coordinate_tolerance=coordinate_tolerance,
        remove_ghost_cells=True,
    ):
        primary.append(chunk.geometry_hash_primary)
        secondary.append(chunk.geometry_hash_secondary)
        measures.append(chunk.cell_measures)
        num_before += chunk.num_cells_before_ghost_filter
        num_removed += chunk.num_ghost_cells_removed
        num_chunks += 1
        chunks_with_ghost_array += int(chunk.ghost_array_present)
        cell_types.add(chunk.cell_type)
        if chunk.original_cell_ids is None:
            ids_complete = False
        else:
            ids.append(chunk.original_cell_ids)

    h1 = np.concatenate(primary) if primary else np.empty(0, dtype=np.uint64)
    h2 = np.concatenate(secondary) if secondary else np.empty(0, dtype=np.uint64)
    area_values = np.concatenate(measures) if measures else np.empty(0, dtype=np.float64)
    if not np.all(np.isfinite(area_values)) or np.any(area_values <= 0.0):
        raise VTKCellFieldError("Partition contains non-finite or non-positive cell areas.")

    keys = np.empty(h1.size, dtype=[("h1", "<u8"), ("h2", "<u8")])
    keys["h1"] = h1
    keys["h2"] = h2
    _, unique_indices, multiplicities = np.unique(
        keys,
        return_index=True,
        return_counts=True,
    )
    num_unique = int(unique_indices.size)
    duplicate_cells = int(h1.size - num_unique)
    max_geometry_multiplicity = int(multiplicities.max()) if multiplicities.size else 0
    raw_area = float(np.sum(area_values, dtype=np.float64))
    unique_area = float(np.sum(area_values[unique_indices], dtype=np.float64))

    original_ids_available = bool(ids_complete and len(ids) == num_chunks)
    original_id_duplicate_cells: int | None = None
    original_id_max_multiplicity: int | None = None
    original_id_scope = "absent_or_partial"
    if original_ids_available:
        all_ids = np.concatenate(ids) if ids else np.empty(0, dtype=np.int64)
        _, id_counts = np.unique(all_ids, return_counts=True)
        original_id_duplicate_cells = int(all_ids.size - id_counts.size)
        original_id_max_multiplicity = int(id_counts.max()) if id_counts.size else 0
        original_id_scope = (
            "globally_unique"
            if original_id_duplicate_cells == 0
            else "piece_local_or_nonunique"
        )

    count_matches = (
        None if expected_num_cells is None else bool(num_unique == int(expected_num_cells))
    )
    area_matches = (
        None
        if expected_area is None
        else bool(
            np.isclose(
                unique_area,
                float(expected_area),
                rtol=float(area_rtol),
                atol=float(area_atol),
            )
        )
    )
    checks = {
        "nonempty_partition": bool(h1.size > 0),
        "geometry_unique_after_ghost_filter": bool(duplicate_cells == 0),
        "cell_areas_finite_positive": True,
        "expected_num_cells_matches": count_matches,
        "expected_area_matches": area_matches,
    }
    required = [value for value in checks.values() if value is not None]
    status = "ok" if all(required) else "failed"

    warnings: list[str] = []
    if chunks_with_ghost_array < num_chunks:
        warnings.append(
            "At least one VTK block has no vtkGhostType/vtkGhostLevels array; "
            "geometry uniqueness and manifest cell counts are therefore the decisive locks."
        )
    if original_ids_available and original_id_scope != "globally_unique":
        warnings.append(
            "vtkOriginalCellIds are not globally unique and appear piece-local; "
            "they are not used as the primary deduplication key."
        )

    return {
        "kind": "vtk_partition_integrity",
        "format_version": VALIDATION_FORMAT_VERSION,
        "status": status,
        "input_path": str(Path(path).expanduser().resolve()),
        "dataset_path": str(dataset.dataset_path),
        "num_pieces": len(dataset.piece_paths),
        "num_chunks": num_chunks,
        "cell_types": sorted(cell_types),
        "num_cells_before_ghost_filter": int(num_before),
        "num_ghost_cells_removed": int(num_removed),
        "num_cells_after_ghost_filter": int(h1.size),
        "num_unique_geometry_cells": num_unique,
        "duplicate_geometry_cells": duplicate_cells,
        "max_geometry_multiplicity": max_geometry_multiplicity,
        "chunks_with_ghost_array": chunks_with_ghost_array,
        "original_cell_ids_available_for_all_chunks": original_ids_available,
        "original_cell_id_scope": original_id_scope,
        "original_cell_id_duplicate_cells": original_id_duplicate_cells,
        "original_cell_id_max_multiplicity": original_id_max_multiplicity,
        "raw_area_sum": raw_area,
        "unique_area_sum": unique_area,
        "expected_num_cells": expected_num_cells,
        "expected_area": expected_area,
        "area_absolute_error": (
            None if expected_area is None else float(abs(unique_area - float(expected_area)))
        ),
        "area_relative_error": (
            None
            if expected_area is None or float(expected_area) == 0.0
            else float(abs(unique_area - float(expected_area)) / abs(float(expected_area)))
        ),
        "coordinate_tolerance": float(coordinate_tolerance),
        "area_rtol": float(area_rtol),
        "area_atol": float(area_atol),
        "checks": checks,
        "warnings": warnings,
    }


def _next_or_none(iterator: Iterator[CellFieldChunk]) -> CellFieldChunk | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def validate_gradient_decomposition(
    *,
    classical_path: str | Path,
    reflection_path: str | Path,
    transmission_path: str | Path,
    field_name: str,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-18,
    geometry_atol: float = 1.0e-12,
    acceptance_relative_l2_tolerance: float = 1.0e-6,
    acceptance_relative_linf_tolerance: float = 1.0e-6,
    input_semantics: str = "l2-density",
) -> dict[str, Any]:
    """Check ``classical = reflection + transmission`` cell by cell.

    ``rtol`` and ``atol`` define a deliberately strict pointwise ``allclose``
    diagnostic.  The principal scientific verdict is solver-aware and uses
    relative area-weighted L2 and L-infinity residuals.  This distinction is
    necessary because the three gradients are produced by separate iterative
    adjoint solves and therefore need not agree at machine precision.
    """

    if input_semantics not in {"l2-density", "cell-integral"}:
        raise ValueError(
            "input_semantics must be either 'l2-density' or 'cell-integral'."
        )

    iterators = [
        iter(iter_vtk_cell_field_chunks(classical_path, field_name)),
        iter(iter_vtk_cell_field_chunks(reflection_path, field_name)),
        iter(iter_vtk_cell_field_chunks(transmission_path, field_name)),
    ]
    acceptance_relative_l2_tolerance = float(acceptance_relative_l2_tolerance)
    acceptance_relative_linf_tolerance = float(acceptance_relative_linf_tolerance)
    if (
        not np.isfinite(acceptance_relative_l2_tolerance)
        or acceptance_relative_l2_tolerance < 0.0
    ):
        raise ValueError(
            "acceptance_relative_l2_tolerance must be finite and non-negative."
        )
    if (
        not np.isfinite(acceptance_relative_linf_tolerance)
        or acceptance_relative_linf_tolerance < 0.0
    ):
        raise ValueError(
            "acceptance_relative_linf_tolerance must be finite and non-negative."
        )

    num_chunks = 0
    num_values = 0
    max_abs_residual = 0.0
    classical_linf = 0.0
    component_sum_linf = 0.0
    residual_l2_squared = 0.0
    classical_l2_squared = 0.0
    component_sum_l2_squared = 0.0
    residual_integral = 0.0
    classical_integral = 0.0
    component_sum_integral = 0.0
    pointwise_allclose = True
    geometry_compatible = True
    values_finite = True
    message = ""

    while True:
        chunks = [_next_or_none(iterator) for iterator in iterators]
        if all(chunk is None for chunk in chunks):
            break
        if any(chunk is None for chunk in chunks):
            geometry_compatible = False
            message = "The three VTK datasets contain different numbers of chunks."
            break
        classical, reflection, transmission = chunks  # type: ignore[misc]
        num_chunks += 1

        compatible = bool(
            classical.cell_type == reflection.cell_type == transmission.cell_type
            and classical.values.shape == reflection.values.shape == transmission.values.shape
            and np.allclose(classical.centers, reflection.centers, rtol=0.0, atol=geometry_atol)
            and np.allclose(classical.centers, transmission.centers, rtol=0.0, atol=geometry_atol)
            and np.allclose(classical.cell_measures, reflection.cell_measures, rtol=rtol, atol=atol)
            and np.allclose(classical.cell_measures, transmission.cell_measures, rtol=rtol, atol=atol)
        )
        if not compatible:
            geometry_compatible = False
            message = f"Incompatible VTK geometry at chunk {num_chunks}."
            break

        measures = classical.cell_measures
        if input_semantics == "cell-integral":
            classical_values = classical.values / measures
            reflection_values = reflection.values / measures
            transmission_values = transmission.values / measures
        else:
            classical_values = classical.values
            reflection_values = reflection.values
            transmission_values = transmission.values

        component_sum = reflection_values + transmission_values
        residual = classical_values - component_sum
        values_finite = bool(
            values_finite
            and np.all(np.isfinite(classical_values))
            and np.all(np.isfinite(reflection_values))
            and np.all(np.isfinite(transmission_values))
            and np.all(np.isfinite(residual))
        )
        if not values_finite:
            message = f"Non-finite values detected at chunk {num_chunks}."
            break

        num_values += int(residual.size)
        if residual.size:
            max_abs_residual = max(max_abs_residual, float(np.max(np.abs(residual))))
            classical_linf = max(
                classical_linf,
                float(np.max(np.abs(classical_values))),
            )
            component_sum_linf = max(
                component_sum_linf,
                float(np.max(np.abs(component_sum))),
            )
        residual_l2_squared += float(np.dot(measures, residual * residual))
        classical_l2_squared += float(np.dot(measures, classical_values * classical_values))
        component_sum_l2_squared += float(np.dot(measures, component_sum * component_sum))
        residual_integral += float(np.dot(measures, residual))
        classical_integral += float(np.dot(measures, classical_values))
        component_sum_integral += float(np.dot(measures, component_sum))
        pointwise_allclose = bool(
            pointwise_allclose
            and np.allclose(classical_values, component_sum, rtol=rtol, atol=atol)
        )

    residual_l2 = float(np.sqrt(max(residual_l2_squared, 0.0)))
    classical_l2 = float(np.sqrt(max(classical_l2_squared, 0.0)))
    component_sum_l2 = float(np.sqrt(max(component_sum_l2_squared, 0.0)))
    relative_l2 = (
        0.0 if classical_l2 == 0.0 and residual_l2 == 0.0
        else float(residual_l2 / classical_l2) if classical_l2 > 0.0
        else float("inf")
    )
    linf_reference = max(classical_linf, component_sum_linf)
    relative_linf = (
        0.0 if linf_reference == 0.0 and max_abs_residual == 0.0
        else float(max_abs_residual / linf_reference) if linf_reference > 0.0
        else float("inf")
    )
    strict_pointwise_allclose = bool(pointwise_allclose)
    solver_aware_validation = bool(
        num_values > 0
        and geometry_compatible
        and values_finite
        and relative_l2 <= acceptance_relative_l2_tolerance
        and relative_linf <= acceptance_relative_linf_tolerance
    )
    status = "ok" if solver_aware_validation else "failed"
    if status == "ok" and strict_pointwise_allclose:
        acceptance_status = "OK_STRICT"
    elif status == "ok":
        acceptance_status = "OK_SOLVER_TOLERANCE"
        if not message:
            message = (
                "Strict pointwise allclose was not met, but the relative L2 and "
                "L-infinity residuals satisfy the solver-aware acceptance criteria."
            )
    else:
        acceptance_status = "FAILED"
        if not message and num_values > 0 and geometry_compatible and values_finite:
            message = (
                "Relative decomposition residual exceeds at least one solver-aware "
                "acceptance tolerance."
            )

    return {
        "kind": "gradient_decomposition_validation",
        "format_version": VALIDATION_FORMAT_VERSION,
        "status": status,
        "field_name": str(field_name),
        "classical_path": str(Path(classical_path).expanduser().resolve()),
        "reflection_path": str(Path(reflection_path).expanduser().resolve()),
        "transmission_path": str(Path(transmission_path).expanduser().resolve()),
        "num_chunks": num_chunks,
        "num_values": num_values,
        "geometry_compatible": geometry_compatible,
        "values_finite": values_finite,
        "pointwise_allclose": strict_pointwise_allclose,
        "strict_pointwise_allclose": strict_pointwise_allclose,
        "solver_aware_validation": solver_aware_validation,
        "acceptance_status": acceptance_status,
        "max_abs_residual": max_abs_residual,
        "relative_linf_error": relative_linf,
        "classical_linf_norm": classical_linf,
        "component_sum_linf_norm": component_sum_linf,
        "residual_l2_error": residual_l2,
        "relative_l2_error": relative_l2,
        "classical_l2_norm": classical_l2,
        "component_sum_l2_norm": component_sum_l2,
        "residual_signed_integral": residual_integral,
        "classical_signed_integral": classical_integral,
        "component_sum_signed_integral": component_sum_integral,
        "integral_absolute_error": float(abs(classical_integral - component_sum_integral)),
        "rtol": float(rtol),
        "atol": float(atol),
        "geometry_atol": float(geometry_atol),
        "acceptance_relative_l2_tolerance": acceptance_relative_l2_tolerance,
        "acceptance_relative_linf_tolerance": acceptance_relative_linf_tolerance,
        "validation_basis": "solver_aware_relative_l2_and_linf",
        "input_semantics": str(input_semantics),
        "spatial_representation": "DG0_L2_cell_average_density",
        "norm_measure": "cell_area_weighted",
        "decomposition_identity": (
            "gradient_density_classical_full = "
            "gradient_density_reflection_only + gradient_density_transmission_only"
        ),
        "message": message,
    }


def taylor_order_estimates(records: Sequence[Mapping[str, Any]]) -> list[float]:
    """Estimate local convergence orders of the second-order Taylor remainder."""

    ordered = sorted(records, key=lambda item: float(item["eps"]), reverse=True)
    orders: list[float] = []
    for left, right in zip(ordered, ordered[1:]):
        eps_left = float(left["eps"])
        eps_right = float(right["eps"])
        rem_left = float(left["second_order_remainder"])
        rem_right = float(right["second_order_remainder"])
        if min(eps_left, eps_right, rem_left, rem_right) <= 0.0:
            orders.append(float("nan"))
        else:
            orders.append(
                float(np.log(rem_left / rem_right) / np.log(eps_left / eps_right))
            )
    return orders
