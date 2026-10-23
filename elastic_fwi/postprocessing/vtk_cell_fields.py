from __future__ import annotations

"""Streaming access to scalar VTK cell fields.

The public API deliberately reads one serial VTU piece and one cell block at a
 time.  A PVD or PVTU file is therefore never merged into a global in-memory
mesh.  Serial VTU parsing is delegated to the compatibility reader already
used by :mod:`elastic_fwi.metrics.artifact_metrics`, so PETSc/FEniCSx XML VTK
2.2 files keep the same handling throughout the project.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from elastic_fwi.metrics.artifact_metrics import _meshio_read_serial_vtu_compat


class VTKCellFieldError(RuntimeError):
    """Base error for VTK cell-field discovery and validation failures."""


@dataclass(frozen=True)
class VTKDataset:
    """Resolved VTK dataset and its serial VTU pieces."""

    requested_path: Path
    dataset_path: Path
    piece_paths: tuple[Path, ...]
    time_index: int


@dataclass(frozen=True)
class CellFieldChunk:
    """One scalar cell-data block read from one serial VTU piece."""

    source_piece: Path
    cell_type: str
    field_name: str
    centers: np.ndarray
    values: np.ndarray
    cell_measures: np.ndarray
    num_cells_before_ghost_filter: int
    num_ghost_cells_removed: int

    @property
    def num_cells(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True)
class CellFieldInspection:
    """Metadata obtained by streaming through a scalar cell field."""

    dataset: VTKDataset
    selected_field_name: str
    num_pieces: int
    num_chunks: int
    num_cells: int
    num_ghost_cells_removed: int


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _require_existing_file(path: Path, *, role: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing {role}: {path}")
    if not path.is_file():
        raise VTKCellFieldError(f"Expected a file for {role}, got: {path}")
    return path


def _dataset_from_pvd(path: Path, *, time_index: int) -> Path:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise VTKCellFieldError(f"Malformed PVD XML file {path}: {exc}") from exc

    datasets = [node for node in root.iter() if _local_name(node.tag) == "DataSet"]
    if not datasets:
        raise VTKCellFieldError(f"No DataSet entry found in PVD file: {path}")

    try:
        dataset_node = datasets[time_index]
    except IndexError as exc:
        raise VTKCellFieldError(
            f"PVD time_index={time_index} is out of range for {path}; "
            f"the file contains {len(datasets)} DataSet entries."
        ) from exc

    file_attr = dataset_node.attrib.get("file")
    if not file_attr:
        raise VTKCellFieldError(f"PVD DataSet without a file attribute in {path}")

    dataset = Path(file_attr)
    if not dataset.is_absolute():
        dataset = path.parent / dataset
    return dataset.resolve()


def _pieces_from_pvtu(path: Path) -> tuple[Path, ...]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise VTKCellFieldError(f"Malformed PVTU XML file {path}: {exc}") from exc

    pieces: list[Path] = []
    for node in root.iter():
        if _local_name(node.tag) != "Piece":
            continue
        source = node.attrib.get("Source") or node.attrib.get("source")
        if not source:
            raise VTKCellFieldError(f"PVTU Piece without Source in {path}")
        piece = Path(source)
        if not piece.is_absolute():
            piece = path.parent / piece
        pieces.append(piece.resolve())

    if not pieces:
        raise VTKCellFieldError(f"No serial VTU Piece entry found in PVTU file: {path}")
    if len(set(pieces)) != len(pieces):
        duplicates = sorted(str(p) for p in pieces if pieces.count(p) > 1)
        raise VTKCellFieldError(
            f"Duplicate VTU Piece references in {path}: {duplicates}"
        )
    return tuple(pieces)


def resolve_vtk_dataset(path: str | Path, *, time_index: int = 0) -> VTKDataset:
    """Resolve a PVD/PVTU/VTU input to ordered serial VTU pieces.

    Parameters
    ----------
    path:
        PVD, PVTU or VTU path.
    time_index:
        PVD DataSet index.  Gradient maps are normally single-snapshot files,
        therefore the deterministic default is the first entry.
    """

    requested = _require_existing_file(Path(path), role="VTK input")
    suffix = requested.suffix.lower()

    if suffix == ".pvd":
        dataset = _dataset_from_pvd(requested, time_index=time_index)
        dataset = _require_existing_file(dataset, role=f"dataset referenced by {requested}")
    elif suffix in {".pvtu", ".vtu"}:
        dataset = requested
    else:
        raise VTKCellFieldError(
            f"Unsupported VTK input extension {requested.suffix!r} for {requested}; "
            "expected .pvd, .pvtu or .vtu."
        )

    if dataset.suffix.lower() == ".pvtu":
        pieces = _pieces_from_pvtu(dataset)
    elif dataset.suffix.lower() == ".vtu":
        pieces = (dataset,)
    else:
        raise VTKCellFieldError(
            f"PVD dataset {dataset} has unsupported extension {dataset.suffix!r}; "
            "expected .pvtu or .vtu."
        )

    missing = [str(piece) for piece in pieces if not piece.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} VTU piece(s) referenced by {dataset} are missing: "
            + ", ".join(missing)
        )

    return VTKDataset(
        requested_path=requested,
        dataset_path=dataset,
        piece_paths=pieces,
        time_index=int(time_index),
    )


def _load_meshio_piece(path: Path):
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise VTKCellFieldError(
            "meshio>=5.3 is required to read VTU pieces. Install the project "
            "dependencies before running this post-processing."
        ) from exc

    try:
        return _meshio_read_serial_vtu_compat(meshio, path)
    except Exception as exc:
        if isinstance(exc, VTKCellFieldError):
            raise
        raise VTKCellFieldError(f"Could not read VTU piece {path}: {exc}") from exc


def _scalar_block(values: np.ndarray, *, field_name: str, piece: Path) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        out = arr
    elif arr.ndim == 2 and arr.shape[1] == 1:
        out = arr[:, 0]
    else:
        raise VTKCellFieldError(
            f"Cell field {field_name!r} in {piece} is not scalar; "
            f"observed shape={arr.shape}."
        )
    try:
        out = np.asarray(out, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise VTKCellFieldError(
            f"Cell field {field_name!r} in {piece} cannot be converted to float64."
        ) from exc
    return out


def _select_field_name(
    available: Sequence[str],
    requested_names: Sequence[str],
    *,
    piece: Path,
) -> str:
    requested = tuple(dict.fromkeys(str(name) for name in requested_names if str(name)))
    if not requested:
        raise ValueError("At least one explicit cell field name must be requested.")
    matches = [name for name in requested if name in available]
    if not matches:
        raise VTKCellFieldError(
            f"None of the requested cell fields {list(requested)} is present in {piece}. "
            f"Available cell fields: {sorted(available)}"
        )
    if len(matches) > 1:
        raise VTKCellFieldError(
            f"Ambiguous cell field selection in {piece}: requested names "
            f"{list(requested)} match multiple arrays {matches}."
        )
    return matches[0]


def _normalize_cell_type(cell_type: str) -> str:
    """Normalize a VTK/meshio cell-type name for geometric dispatch."""

    normalized = str(cell_type).strip().lower().replace("-", "_")

    if normalized.startswith("vtk_"):
        normalized = normalized[4:]

    return normalized


def _corner_count_for_cell_type(cell_type: str, width: int) -> int:
    """Return the number of geometric corner nodes for a 2-D VTK cell."""

    normalized = _normalize_cell_type(cell_type)

    triangle_types = {
        "triangle",
        "lagrange_triangle",
        "quadratic_triangle",
    }
    quadrilateral_types = {
        "quad",
        "quadrilateral",
        "lagrange_quadrilateral",
        "lagrange_quad",
        "quadratic_quad",
        "pixel",
    }

    if (
        normalized in triangle_types
        or normalized.startswith("triangle")
        or normalized.endswith("_triangle")
    ):
        if width < 3:
            raise VTKCellFieldError(
                f"Cell type {cell_type!r} requires at least three connectivity "
                f"indices, but width={width}."
            )
        return 3

    if (
        normalized in quadrilateral_types
        or normalized.startswith("quad")
        or normalized.startswith("pixel")
        or normalized.endswith("_quadrilateral")
        or normalized.endswith("_quad")
        or normalized.endswith("_pixel")
    ):
        if width < 4:
            raise VTKCellFieldError(
                f"Cell type {cell_type!r} requires at least four connectivity "
                f"indices, but width={width}."
            )
        return 4

    if normalized.startswith("polygon") or normalized.endswith("_polygon"):
        if width < 3:
            raise VTKCellFieldError(
                f"Polygon cell {cell_type!r} requires at least three vertices, "
                f"but width={width}."
            )
        return int(width)

    raise VTKCellFieldError(
        f"Unsupported cell type {cell_type!r} for 2-D area-weighted analysis. "
        "Expected triangle, quadrilateral, pixel, or polygon cells, including "
        "their VTK Lagrange variants."
    )

def _cell_geometry_2d(
    *,
    points: np.ndarray,
    connectivity: np.ndarray,
    cell_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact polygon centroids and positive areas in the x-y plane."""

    if connectivity.shape[0] == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    corner_count = _corner_count_for_cell_type(cell_type, connectivity.shape[1])
    if connectivity.shape[1] < corner_count:
        raise VTKCellFieldError(
            f"Cell type {cell_type!r} requires at least {corner_count} nodes, "
            f"but connectivity has width {connectivity.shape[1]}."
        )

    corner_connectivity = connectivity[:, :corner_count]
    if _normalize_cell_type(cell_type).startswith("pixel"):
        corner_connectivity = corner_connectivity[:, [0, 1, 3, 2]]
    vertices = np.asarray(points[corner_connectivity, :2], dtype=np.float64)
    x = vertices[:, :, 0]
    y = vertices[:, :, 1]
    x_next = np.roll(x, shift=-1, axis=1)
    y_next = np.roll(y, shift=-1, axis=1)
    cross = x * y_next - y * x_next
    signed_double_area = np.sum(cross, axis=1, dtype=np.float64)
    measures = 0.5 * np.abs(signed_double_area)

    if not np.all(np.isfinite(measures)):
        raise VTKCellFieldError(
            f"Non-finite cell areas computed for cell type {cell_type!r}."
        )
    if np.any(measures <= 0.0):
        count = int(np.count_nonzero(measures <= 0.0))
        raise VTKCellFieldError(
            f"Computed {count} non-positive cell area(s) for cell type "
            f"{cell_type!r}; the mesh is degenerate or incompatible."
        )

    centers_x = np.sum((x + x_next) * cross, axis=1, dtype=np.float64)
    centers_y = np.sum((y + y_next) * cross, axis=1, dtype=np.float64)
    denominator = 3.0 * signed_double_area
    centers = np.column_stack((centers_x / denominator, centers_y / denominator))
    if not np.all(np.isfinite(centers)):
        raise VTKCellFieldError(
            f"Non-finite cell centroids computed for cell type {cell_type!r}."
        )
    return (
        np.ascontiguousarray(centers, dtype=np.float64),
        np.ascontiguousarray(measures, dtype=np.float64),
    )


def _ghost_mask_for_block(mesh, block_index: int, num_cells: int) -> np.ndarray | None:
    cell_data = getattr(mesh, "cell_data", {})
    for ghost_name in ("vtkGhostType", "vtkGhostLevels"):
        blocks = cell_data.get(ghost_name)
        if blocks is None or block_index >= len(blocks):
            continue
        values = np.asarray(blocks[block_index]).reshape(-1)
        if values.shape[0] != num_cells:
            raise VTKCellFieldError(
                f"Ghost array {ghost_name!r} has {values.shape[0]} values for "
                f"a cell block containing {num_cells} cells."
            )
        return values == 0
    return None


def iter_vtk_cell_field_chunks(
    path: str | Path,
    field_names: Sequence[str] | str,
    *,
    time_index: int = 0,
    remove_ghost_cells: bool = True,
    require_finite: bool = True,
) -> Iterator[CellFieldChunk]:
    """Yield scalar cell-field chunks without merging PVTU pieces.

    Only explicitly requested field names are accepted.  The same unique field
    name must be selected in every piece.  Each yielded chunk owns its arrays;
    callers should release the chunk before requesting the next one when memory
    pressure matters.
    """

    if isinstance(field_names, str):
        requested_names = (field_names,)
    else:
        requested_names = tuple(field_names)

    dataset = resolve_vtk_dataset(path, time_index=time_index)
    selected_globally: str | None = None

    for piece in dataset.piece_paths:
        mesh = _load_meshio_piece(piece)
        points = np.asarray(getattr(mesh, "points", np.empty((0, 3))), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 2:
            raise VTKCellFieldError(
                f"Points in {piece} must have shape (n_points, dim>=2); "
                f"observed shape={points.shape}."
            )
        if not np.all(np.isfinite(points[:, :2])):
            raise VTKCellFieldError(f"Non-finite point coordinates found in {piece}.")

        cell_data = getattr(mesh, "cell_data", {})
        selected = _select_field_name(tuple(cell_data.keys()), requested_names, piece=piece)
        if selected_globally is None:
            selected_globally = selected
        elif selected != selected_globally:
            raise VTKCellFieldError(
                f"Inconsistent field selection across pieces for {dataset.dataset_path}: "
                f"{selected_globally!r} then {selected!r} in {piece}."
            )

        blocks = list(getattr(mesh, "cells", []))
        values_by_block = cell_data.get(selected, [])
        if len(values_by_block) != len(blocks):
            raise VTKCellFieldError(
                f"Cell field {selected!r} in {piece} has {len(values_by_block)} block(s), "
                f"but the mesh has {len(blocks)} cell block(s)."
            )

        for block_index, block in enumerate(blocks):
            connectivity = np.asarray(getattr(block, "data", block), dtype=np.int64)
            cell_type = str(getattr(block, "type", "cell"))
            if connectivity.ndim != 2:
                raise VTKCellFieldError(
                    f"Connectivity for cell block {block_index} ({cell_type}) in {piece} "
                    f"must be two-dimensional; observed shape={connectivity.shape}."
                )
            if connectivity.size and (
                np.min(connectivity) < 0 or np.max(connectivity) >= points.shape[0]
            ):
                raise VTKCellFieldError(
                    f"Connectivity in {piece}, block {block_index}, references points "
                    "outside the valid point index range."
                )

            values = _scalar_block(
                np.asarray(values_by_block[block_index]),
                field_name=selected,
                piece=piece,
            )
            num_before = int(connectivity.shape[0])
            if values.shape[0] != num_before:
                raise VTKCellFieldError(
                    f"Cell/value size mismatch for {selected!r} in {piece}, block "
                    f"{block_index}: {num_before} cells versus {values.shape[0]} values."
                )

            if connectivity.size:
                centers, cell_measures = _cell_geometry_2d(
                    points=points,
                    connectivity=connectivity,
                    cell_type=cell_type,
                )
            else:
                centers = np.empty((0, 2), dtype=np.float64)
                cell_measures = np.empty((0,), dtype=np.float64)

            removed = 0
            if remove_ghost_cells:
                keep = _ghost_mask_for_block(mesh, block_index, num_before)
                if keep is not None:
                    removed = int(num_before - np.count_nonzero(keep))
                    centers = centers[keep]
                    values = values[keep]
                    cell_measures = cell_measures[keep]

            if require_finite:
                if not np.all(np.isfinite(centers)):
                    raise VTKCellFieldError(
                        f"Non-finite cell centers found in {piece}, block {block_index}."
                    )
                if not np.all(np.isfinite(values)):
                    raise VTKCellFieldError(
                        f"Non-finite values found for {selected!r} in {piece}, "
                        f"block {block_index}."
                    )
                if not np.all(np.isfinite(cell_measures)) or np.any(cell_measures <= 0.0):
                    raise VTKCellFieldError(
                        f"Invalid cell measures found in {piece}, block {block_index}."
                    )

            yield CellFieldChunk(
                source_piece=piece,
                cell_type=cell_type,
                field_name=selected,
                centers=np.asarray(centers, dtype=np.float64),
                values=np.asarray(values, dtype=np.float64),
                cell_measures=np.asarray(cell_measures, dtype=np.float64),
                num_cells_before_ghost_filter=num_before,
                num_ghost_cells_removed=removed,
            )

        del mesh, points


def inspect_vtk_cell_field(
    path: str | Path,
    field_names: Sequence[str] | str,
    *,
    time_index: int = 0,
    remove_ghost_cells: bool = True,
    expected_num_cells: int | None = None,
) -> CellFieldInspection:
    """Stream a field and return validated counts without retaining chunks."""

    dataset = resolve_vtk_dataset(path, time_index=time_index)
    selected: str | None = None
    num_chunks = 0
    num_cells = 0
    num_ghost = 0

    for chunk in iter_vtk_cell_field_chunks(
        path,
        field_names,
        time_index=time_index,
        remove_ghost_cells=remove_ghost_cells,
    ):
        selected = chunk.field_name if selected is None else selected
        num_chunks += 1
        num_cells += chunk.num_cells
        num_ghost += chunk.num_ghost_cells_removed
        del chunk

    if selected is None:
        raise VTKCellFieldError(f"No cell block was read from {dataset.dataset_path}.")
    if expected_num_cells is not None and num_cells != int(expected_num_cells):
        raise VTKCellFieldError(
            f"Unexpected cell count for {dataset.requested_path}: observed {num_cells}, "
            f"expected {int(expected_num_cells)}."
        )

    return CellFieldInspection(
        dataset=dataset,
        selected_field_name=selected,
        num_pieces=len(dataset.piece_paths),
        num_chunks=num_chunks,
        num_cells=num_cells,
        num_ghost_cells_removed=num_ghost,
    )
