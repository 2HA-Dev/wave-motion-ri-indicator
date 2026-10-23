from __future__ import annotations

"""Streaming reconstruction of the physical scattered displacement field.

The module pairs healthy and defective PVD time series, validates every PVTU/VTU
partition before subtraction, computes mesh-weighted P1 triangle metrics one
snapshot at a time, and optionally exports selected or all scattered snapshots
for ParaView.  It deliberately contains no adjoint, gradient, or inverse-image
logic.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from elastic_fwi.metrics.artifact_metrics import _meshio_read_serial_vtu_compat
from elastic_fwi.wave_physics.scattered_field import compute_scattered_field


class ScatteredWavefieldError(RuntimeError):
    """Base error raised by the scattered-wavefield post-processing."""


class PVDSeriesError(ScatteredWavefieldError):
    """Malformed, incomplete, or incompatible PVD time series."""


class MeshCompatibilityError(ScatteredWavefieldError):
    """Healthy and defective VTK meshes are not strictly compatible."""


class FieldSelectionError(ScatteredWavefieldError):
    """A requested displacement field is absent or ambiguous."""


@dataclass(frozen=True, order=True)
class PVDEntry:
    index: int
    time: float
    dataset_path: Path


@dataclass(frozen=True)
class PVDSeries:
    path: Path
    entries: tuple[PVDEntry, ...]


@dataclass(frozen=True)
class SnapshotPair:
    index: int
    time: float
    healthy_dataset: Path
    defect_dataset: Path


@dataclass(frozen=True)
class AnalysisBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def validate(self) -> None:
        values = (self.xmin, self.xmax, self.ymin, self.ymax)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Analysis-box coordinates must be finite.")
        if not self.xmin < self.xmax or not self.ymin < self.ymax:
            raise ValueError(
                "Analysis box must satisfy xmin < xmax and ymin < ymax."
            )


@dataclass(frozen=True)
class ScatteredWavefieldInputs:
    healthy_pvd: Path
    defect_pvd: Path
    campaign_manifest: Path | None = None


@dataclass(frozen=True)
class ScatteredWavefieldConfig:
    inputs: ScatteredWavefieldInputs
    output_dir: Path
    geometry_kind: str = "unknown"
    healthy_field: str = "direct_simulated_displacement"
    defect_field: str = "direct_observed_displacement"
    time_atol: float = 1.0e-12
    time_rtol: float = 1.0e-12
    geometry_atol: float = 1.0e-12
    geometry_rtol: float = 0.0
    pre_atol: float = 1.0e-14
    pre_rtol: float = 1.0e-10
    detection_fraction: float = 1.0e-6
    zone_detection_fraction: float = 0.1
    ratio_epsilon: float = 1.0e-30
    defect_center: tuple[float, float] | None = None
    defect_radius: float | None = None
    analysis_box: AnalysisBox | None = None
    zone_width: float = 0.65
    arrival_time: float | None = None
    snapshot_times: tuple[float, ...] | None = None
    vtk_export_mode: str = "selected"
    raster_resolution: int = 500
    overwrite: bool = False
    quiet: bool = False

    def validate(self) -> None:
        if self.geometry_kind not in {"line", "box", "unknown"}:
            raise ValueError("geometry_kind must be one of: line, box, unknown.")
        if not self.healthy_field or not self.defect_field:
            raise ValueError("Healthy and defect field names must be non-empty.")
        for name, value in (
            ("time_atol", self.time_atol),
            ("time_rtol", self.time_rtol),
            ("geometry_atol", self.geometry_atol),
            ("geometry_rtol", self.geometry_rtol),
            ("pre_atol", self.pre_atol),
            ("pre_rtol", self.pre_rtol),
            ("ratio_epsilon", self.ratio_epsilon),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name, value in (
            ("detection_fraction", self.detection_fraction),
            ("zone_detection_fraction", self.zone_detection_fraction),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1].")
        if not math.isfinite(self.zone_width) or self.zone_width <= 0.0:
            raise ValueError("zone_width must be finite and positive.")
        if self.defect_radius is not None and (
            not math.isfinite(self.defect_radius) or self.defect_radius <= 0.0
        ):
            raise ValueError("defect_radius must be finite and positive.")
        if self.arrival_time is not None and not math.isfinite(self.arrival_time):
            raise ValueError("arrival_time must be finite when provided.")
        if self.vtk_export_mode not in {"none", "selected", "all"}:
            raise ValueError("vtk_export_mode must be one of: none, selected, all.")
        if self.raster_resolution < 64:
            raise ValueError("raster_resolution must be at least 64.")
        if self.analysis_box is not None:
            self.analysis_box.validate()


@dataclass(frozen=True)
class ScatteredOutputPaths:
    root: Path
    csv_dir: Path
    figures_dir: Path
    manifests_dir: Path
    logs_dir: Path
    vtk_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "ScatteredOutputPaths":
        root = Path(root).expanduser().resolve()
        return cls(
            root=root,
            csv_dir=root / "csv",
            figures_dir=root / "figures",
            manifests_dir=root / "manifests",
            logs_dir=root / "logs",
            vtk_dir=root / "vtk",
        )

    def create(self) -> None:
        for path in (
            self.root,
            self.csv_dir,
            self.figures_dir,
            self.manifests_dir,
            self.logs_dir,
            self.vtk_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ScatteredWavefieldResult:
    metrics: tuple[dict[str, Any], ...]
    selected_times: tuple[dict[str, Any], ...]
    characteristic_times: Mapping[str, float | None]
    validation: Mapping[str, Any]
    metadata: Mapping[str, Any]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _require_file(path: Path, *, role: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {role}: {resolved}")
    return resolved


def parse_pvd_series(path: str | Path) -> PVDSeries:
    pvd = _require_file(Path(path), role="PVD time series")
    if pvd.suffix.lower() != ".pvd":
        raise PVDSeriesError(f"Expected a .pvd file, got: {pvd}")
    try:
        root = ET.parse(pvd).getroot()
    except ET.ParseError as exc:
        raise PVDSeriesError(f"Malformed PVD XML {pvd}: {exc}") from exc

    entries: list[PVDEntry] = []
    for node in root.iter():
        if _local_name(node.tag) != "DataSet":
            continue
        file_attr = node.attrib.get("file")
        time_attr = node.attrib.get("timestep")
        if not file_attr or time_attr is None:
            raise PVDSeriesError(
                f"PVD DataSet in {pvd} must define both file and timestep."
            )
        try:
            time_value = float(time_attr)
        except ValueError as exc:
            raise PVDSeriesError(
                f"Invalid PVD timestep {time_attr!r} in {pvd}."
            ) from exc
        if not math.isfinite(time_value):
            raise PVDSeriesError(f"Non-finite timestep in {pvd}: {time_value}")
        dataset = Path(file_attr)
        if not dataset.is_absolute():
            dataset = pvd.parent / dataset
        entries.append(
            PVDEntry(
                index=len(entries),
                time=time_value,
                dataset_path=dataset.resolve(),
            )
        )

    if not entries:
        raise PVDSeriesError(f"No DataSet entry found in PVD file: {pvd}")
    times = np.asarray([entry.time for entry in entries], dtype=np.float64)
    if np.any(np.diff(times) <= 0.0):
        raise PVDSeriesError(
            f"PVD timesteps must be strictly increasing in file order: {pvd}"
        )
    return PVDSeries(path=pvd, entries=tuple(entries))


def pair_pvd_series(
    healthy: PVDSeries,
    defect: PVDSeries,
    *,
    atol: float,
    rtol: float,
) -> tuple[SnapshotPair, ...]:
    if len(healthy.entries) != len(defect.entries):
        raise PVDSeriesError(
            "Healthy and defect PVD series contain different snapshot counts: "
            f"{len(healthy.entries)} != {len(defect.entries)}."
        )
    pairs: list[SnapshotPair] = []
    for index, (healthy_entry, defect_entry) in enumerate(
        zip(healthy.entries, defect.entries, strict=True)
    ):
        if not np.isclose(
            healthy_entry.time,
            defect_entry.time,
            atol=atol,
            rtol=rtol,
        ):
            raise PVDSeriesError(
                f"Time mismatch at snapshot {index}: healthy={healthy_entry.time:.17g}, "
                f"defect={defect_entry.time:.17g}, atol={atol}, rtol={rtol}."
            )
        pairs.append(
            SnapshotPair(
                index=index,
                time=float(healthy_entry.time),
                healthy_dataset=healthy_entry.dataset_path,
                defect_dataset=defect_entry.dataset_path,
            )
        )
    return tuple(pairs)


def _parse_pvtu_pieces(path: Path) -> tuple[Path, ...]:
    path = _require_file(path, role="PVTU dataset")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise PVDSeriesError(f"Malformed PVTU XML {path}: {exc}") from exc
    pieces: list[Path] = []
    for node in root.iter():
        if _local_name(node.tag) != "Piece":
            continue
        source = node.attrib.get("Source") or node.attrib.get("source")
        if not source:
            raise PVDSeriesError(f"PVTU Piece without Source in {path}")
        piece = Path(source)
        if not piece.is_absolute():
            piece = path.parent / piece
        pieces.append(piece.resolve())
    if not pieces:
        raise PVDSeriesError(f"No Piece entry found in PVTU file: {path}")
    if len(set(pieces)) != len(pieces):
        raise PVDSeriesError(f"Duplicate Piece references in PVTU file: {path}")
    missing = [str(piece) for piece in pieces if not piece.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} VTU piece(s) referenced by {path} are missing: "
            + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else "")
        )
    return tuple(pieces)


def dataset_piece_paths(path: Path) -> tuple[Path, ...]:
    suffix = path.suffix.lower()
    if suffix == ".pvtu":
        return _parse_pvtu_pieces(path)
    if suffix == ".vtu":
        return (_require_file(path, role="VTU dataset"),)
    raise PVDSeriesError(
        f"Unsupported PVD dataset extension {path.suffix!r}: {path}. "
        "Expected .pvtu or .vtu."
    )


def _load_mesh(path: Path):
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ScatteredWavefieldError(
            "meshio>=5.3 is required for scattered-wavefield reconstruction."
        ) from exc
    try:
        return _meshio_read_serial_vtu_compat(meshio, path)
    except Exception as exc:
        raise ScatteredWavefieldError(f"Could not read VTU piece {path}: {exc}") from exc


def _extract_vector_point_field(mesh: Any, name: str, *, piece: Path) -> np.ndarray:
    point_data = getattr(mesh, "point_data", {})
    if name not in point_data:
        raise FieldSelectionError(
            f"Point field {name!r} is absent from {piece}. "
            f"Available point fields: {sorted(str(key) for key in point_data)}"
        )
    raw = np.asarray(point_data[name])
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise FieldSelectionError(
            f"Point field {name!r} in {piece} must have shape (n_points, >=2); "
            f"observed {raw.shape}."
        )
    values = np.asarray(raw[:, :2], dtype=np.float64)
    points = np.asarray(mesh.points)
    if values.shape[0] != points.shape[0]:
        raise FieldSelectionError(
            f"Point field {name!r} in {piece} has {values.shape[0]} rows, "
            f"but the mesh has {points.shape[0]} points."
        )
    if not np.all(np.isfinite(values)):
        raise FieldSelectionError(
            f"Point field {name!r} in {piece} contains NaN or infinity."
        )
    return values


def _cell_type_and_data(block: Any) -> tuple[str, np.ndarray]:
    cell_type = str(getattr(block, "type", ""))
    data = np.asarray(getattr(block, "data", block), dtype=np.int64)
    return cell_type, data


_AUXILIARY_CELL_TYPES = frozenset({"vertex", "line", "line3"})

# PETSc/FEniCSx VTK 2.2 may label geometrically linear three-node
# triangles as VTK_LAGRANGE_TRIANGLE.  Acceptance depends on both
# the cell label and the actual connectivity width.
_P1_TRIANGLE_CELL_TYPES = frozenset(
    {
        "triangle",
        "VTK_TRIANGLE",
        "VTK_LAGRANGE_TRIANGLE",
    }
)


def _require_p1_triangle_connectivity(
    *,
    cell_type: str,
    connectivity: np.ndarray,
    context: str,
) -> np.ndarray | None:
    """Validate and return a three-node triangle connectivity block."""

    if cell_type in _AUXILIARY_CELL_TYPES:
        return None

    if cell_type not in _P1_TRIANGLE_CELL_TYPES:
        raise MeshCompatibilityError(
            f"Unsupported cell type {cell_type!r} in {context}; expected P1 "
            "triangles plus optional vertex/line blocks."
        )

    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise MeshCompatibilityError(
            f"Cell type {cell_type!r} in {context} must contain exactly "
            "three connectivity nodes per P1 triangle; observed shape "
            f"{connectivity.shape}."
        )

    return connectivity


def _p1_triangle_cells_for_export(
    mesh: Any,
    *,
    context: str,
) -> list[tuple[str, np.ndarray]]:
    """Return meshio-compatible linear triangle blocks for VTK export."""

    cells: list[tuple[str, np.ndarray]] = []

    for block in mesh.cells:
        cell_type, connectivity = _cell_type_and_data(block)

        triangles = _require_p1_triangle_connectivity(
            cell_type=cell_type,
            connectivity=connectivity,
            context=context,
        )

        if triangles is None or triangles.size == 0:
            continue

        cells.append(
            (
                "triangle",
                np.asarray(triangles, dtype=np.int64),
            )
        )

    if not cells:
        raise MeshCompatibilityError(
            f"No P1 triangle cells available for export in {context}."
        )

    return cells


def _canonical_triangle_connectivity(mesh: Any, *, piece: Path) -> np.ndarray:
    """Return a block-order- and orientation-independent triangle topology.

    FEniCSx/VTK writers may split or order auxiliary and triangle cell blocks
    differently for two fields defined on the same point mesh.  The physical
    subtraction and the P1 domain integrals only require identical point order
    and identical triangle topology.  Vertex/line blocks are therefore ignored
    for this compatibility check, while every non-auxiliary non-triangle block
    remains an explicit error.
    """

    triangle_blocks: list[np.ndarray] = []

    for block in mesh.cells:
        cell_type, data = _cell_type_and_data(block)

        triangles = _require_p1_triangle_connectivity(
            cell_type=cell_type,
            connectivity=data,
            context=str(piece),
        )

        if triangles is None:
            continue

        if triangles.size:
            # Triangle orientation and block/file order do not affect the mesh
            # topology used by the present P1 scalar integrations.
            triangle_blocks.append(np.sort(triangles, axis=1))

    if not triangle_blocks:
        raise MeshCompatibilityError(f"No triangle cells found in {piece}.")

    triangles = np.concatenate(triangle_blocks, axis=0)
    order = np.lexsort((triangles[:, 2], triangles[:, 1], triangles[:, 0]))
    return triangles[order]


def validate_mesh_pair(
    healthy_mesh: Any,
    defect_mesh: Any,
    *,
    healthy_piece: Path,
    defect_piece: Path,
    atol: float,
    rtol: float,
) -> None:
    healthy_points = np.asarray(healthy_mesh.points, dtype=np.float64)
    defect_points = np.asarray(defect_mesh.points, dtype=np.float64)
    if healthy_points.shape != defect_points.shape:
        raise MeshCompatibilityError(
            "Mesh point-count/coordinate-shape mismatch between pieces "
            f"{healthy_piece} ({healthy_points.shape}) and "
            f"{defect_piece} ({defect_points.shape})."
        )
    if not np.all(np.isfinite(healthy_points)) or not np.all(np.isfinite(defect_points)):
        raise MeshCompatibilityError(
            f"Non-finite mesh coordinates in {healthy_piece} or {defect_piece}."
        )
    if not np.allclose(healthy_points, defect_points, atol=atol, rtol=rtol):
        difference = float(np.max(np.abs(healthy_points - defect_points)))
        raise MeshCompatibilityError(
            f"Mesh coordinates differ between {healthy_piece} and {defect_piece}; "
            f"max_abs_difference={difference:.17g}, atol={atol}, rtol={rtol}."
        )

    healthy_triangles = _canonical_triangle_connectivity(
        healthy_mesh, piece=healthy_piece
    )
    defect_triangles = _canonical_triangle_connectivity(
        defect_mesh, piece=defect_piece
    )
    if healthy_triangles.shape != defect_triangles.shape:
        raise MeshCompatibilityError(
            "Triangle-count mismatch between pieces "
            f"{healthy_piece} ({healthy_triangles.shape[0]}) and "
            f"{defect_piece} ({defect_triangles.shape[0]})."
        )
    if not np.array_equal(healthy_triangles, defect_triangles):
        raise MeshCompatibilityError(
            "Triangle connectivity mismatch between pieces "
            f"{healthy_piece} and {defect_piece}, after canonicalization of "
            "triangle orientation, block grouping, and block order."
        )


def _cell_data_block(mesh: Any, name: str, block_index: int) -> np.ndarray | None:
    blocks = getattr(mesh, "cell_data", {}).get(name)
    if blocks is None or block_index >= len(blocks):
        return None
    return np.asarray(blocks[block_index]).reshape(-1)


def _owned_cell_mask(mesh: Any, block_index: int, num_cells: int) -> np.ndarray:
    for name in ("vtkGhostType", "vtkGhostLevels"):
        values = _cell_data_block(mesh, name, block_index)
        if values is None:
            continue
        if values.shape[0] != num_cells:
            raise MeshCompatibilityError(
                f"Ghost-cell array {name!r} has {values.shape[0]} entries for "
                f"a block with {num_cells} cells."
            )
        return values == 0
    return np.ones(num_cells, dtype=bool)


def _triangle_area(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    xy = points[triangles, :2]
    cross = (
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 1, 1] - xy[:, 0, 1]) * (xy[:, 2, 0] - xy[:, 0, 0])
    )
    area = 0.5 * np.abs(cross)
    if np.any(~np.isfinite(area)) or np.any(area <= 0.0):
        raise MeshCompatibilityError("Triangle mesh contains non-positive cell areas.")
    return area


def _p1_triangle_square_integral(values: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Exact integral of a scalar P1 field squared on each triangle."""
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Triangle vertex values must have shape (n_cells, 3).")
    squares = np.sum(values * values, axis=1)
    pairs = (
        values[:, 0] * values[:, 1]
        + values[:, 0] * values[:, 2]
        + values[:, 1] * values[:, 2]
    )
    return area * (squares + pairs) / 6.0


def _safe_ratio(numerator: float, denominator: float, epsilon: float) -> tuple[float | None, str]:
    if abs(denominator) <= epsilon:
        if abs(numerator) <= epsilon:
            return 0.0, "both_zero"
        return None, "undefined_zero_denominator"
    return float(numerator / denominator), "ok"


def _zone_masks(
    centers: np.ndarray,
    *,
    analysis_box: AnalysisBox | None,
    zone_width: float,
    defect_center: tuple[float, float] | None,
    defect_radius: float | None,
) -> dict[str, np.ndarray]:
    count = centers.shape[0]
    masks: dict[str, np.ndarray] = {"all": np.ones(count, dtype=bool)}
    if analysis_box is not None:
        half = 0.5 * zone_width
        x = centers[:, 0]
        y = centers[:, 1]
        within_y = (y >= analysis_box.ymin - half) & (y <= analysis_box.ymax + half)
        within_x = (x >= analysis_box.xmin - half) & (x <= analysis_box.xmax + half)
        masks["left"] = (np.abs(x - analysis_box.xmin) <= half) & within_y
        masks["right"] = (np.abs(x - analysis_box.xmax) <= half) & within_y
        masks["top"] = (np.abs(y - analysis_box.ymax) <= half) & within_x
        masks["bottom"] = (np.abs(y - analysis_box.ymin) <= half) & within_x
    if defect_center is not None and defect_radius is not None:
        dx = centers[:, 0] - defect_center[0]
        dy = centers[:, 1] - defect_center[1]
        masks["defect"] = dx * dx + dy * dy <= defect_radius * defect_radius
    return masks


def _snapshot_metrics(
    pair: SnapshotPair,
    *,
    healthy_field: str,
    defect_field: str,
    geometry_atol: float,
    geometry_rtol: float,
    analysis_box: AnalysisBox | None,
    zone_width: float,
    defect_center: tuple[float, float] | None,
    defect_radius: float | None,
    ratio_epsilon: float,
) -> dict[str, Any]:
    healthy_pieces = dataset_piece_paths(pair.healthy_dataset)
    defect_pieces = dataset_piece_paths(pair.defect_dataset)
    if len(healthy_pieces) != len(defect_pieces):
        raise MeshCompatibilityError(
            f"Snapshot t={pair.time:g} has different PVTU partition counts: "
            f"healthy={len(healthy_pieces)}, defect={len(defect_pieces)}."
        )

    energy_scattered = 0.0
    energy_healthy = 0.0
    area_total = 0.0
    zone_energy: dict[str, float] = {}
    max_abs_ux = 0.0
    max_abs_uy = 0.0
    max_norm = 0.0
    max_norm_healthy = 0.0
    max_x = math.nan
    max_y = math.nan
    num_cells = 0
    ghost_cells_removed = 0
    point_count_sum = 0

    for piece_index, (healthy_piece, defect_piece) in enumerate(
        zip(healthy_pieces, defect_pieces, strict=True)
    ):
        healthy_mesh = _load_mesh(healthy_piece)
        defect_mesh = _load_mesh(defect_piece)
        validate_mesh_pair(
            healthy_mesh,
            defect_mesh,
            healthy_piece=healthy_piece,
            defect_piece=defect_piece,
            atol=geometry_atol,
            rtol=geometry_rtol,
        )
        healthy_values = _extract_vector_point_field(
            healthy_mesh, healthy_field, piece=healthy_piece
        )
        defect_values = _extract_vector_point_field(
            defect_mesh, defect_field, piece=defect_piece
        )
        scattered = compute_scattered_field(
            defect=defect_values,
            reference=healthy_values,
        )
        if not np.all(np.isfinite(scattered)):
            raise ScatteredWavefieldError(
                f"Non-finite scattered field at t={pair.time:g}, piece={piece_index}."
            )

        points = np.asarray(healthy_mesh.points, dtype=np.float64)
        point_count_sum += int(points.shape[0])
        norm_values = np.linalg.norm(scattered, axis=1)
        healthy_norm_values = np.linalg.norm(healthy_values, axis=1)
        local_max_index = int(np.argmax(norm_values)) if norm_values.size else 0
        local_max = float(norm_values[local_max_index]) if norm_values.size else 0.0
        if local_max > max_norm:
            max_norm = local_max
            max_x = float(points[local_max_index, 0])
            max_y = float(points[local_max_index, 1])
        if scattered.size:
            max_abs_ux = max(max_abs_ux, float(np.max(np.abs(scattered[:, 0]))))
            max_abs_uy = max(max_abs_uy, float(np.max(np.abs(scattered[:, 1]))))
        if healthy_norm_values.size:
            max_norm_healthy = max(
                max_norm_healthy, float(np.max(healthy_norm_values))
            )

        found_triangle = False

        for block_index, block in enumerate(healthy_mesh.cells):
            cell_type, connectivity = _cell_type_and_data(block)

            connectivity = _require_p1_triangle_connectivity(
                cell_type=cell_type,
                connectivity=connectivity,
                context=str(healthy_piece),
            )

            if connectivity is None:
                continue

            found_triangle = True
            owned = _owned_cell_mask(
                healthy_mesh,
                block_index,
                connectivity.shape[0],
            )
            ghost_cells_removed += int(np.count_nonzero(~owned))
            triangles = connectivity[owned]
            if triangles.size == 0:
                continue
            area = _triangle_area(points, triangles)
            centers = np.mean(points[triangles, :2], axis=1)
            sx = scattered[triangles, 0]
            sy = scattered[triangles, 1]
            hx = healthy_values[triangles, 0]
            hy = healthy_values[triangles, 1]
            cell_scattered = (
                _p1_triangle_square_integral(sx, area)
                + _p1_triangle_square_integral(sy, area)
            )
            cell_healthy = (
                _p1_triangle_square_integral(hx, area)
                + _p1_triangle_square_integral(hy, area)
            )
            energy_scattered += float(np.sum(cell_scattered))
            energy_healthy += float(np.sum(cell_healthy))
            area_total += float(np.sum(area))
            num_cells += int(triangles.shape[0])
            for zone, mask in _zone_masks(
                centers,
                analysis_box=analysis_box,
                zone_width=zone_width,
                defect_center=defect_center,
                defect_radius=defect_radius,
            ).items():
                zone_energy[zone] = zone_energy.get(zone, 0.0) + float(
                    np.sum(cell_scattered[mask])
                )
        if not found_triangle:
            raise MeshCompatibilityError(
                f"No P1 triangle block found in VTU piece {healthy_piece}."
            )

        del healthy_mesh, defect_mesh, healthy_values, defect_values, scattered

    l2_scattered = math.sqrt(max(energy_scattered, 0.0))
    l2_healthy = math.sqrt(max(energy_healthy, 0.0))
    relative_l2, relative_status = _safe_ratio(
        l2_scattered, l2_healthy, ratio_epsilon
    )
    row: dict[str, Any] = {
        "snapshot_index": int(pair.index),
        "time": float(pair.time),
        "healthy_dataset": str(pair.healthy_dataset),
        "defect_dataset": str(pair.defect_dataset),
        "num_pieces": int(len(healthy_pieces)),
        "num_cells_owned": int(num_cells),
        "num_ghost_cells_removed": int(ghost_cells_removed),
        "point_count_sum_with_partition_duplicates": int(point_count_sum),
        "domain_area_integrated": float(area_total),
        "energy_scattered": float(energy_scattered),
        "l2_scattered": float(l2_scattered),
        "energy_healthy": float(energy_healthy),
        "l2_healthy": float(l2_healthy),
        "relative_l2": relative_l2,
        "relative_l2_status": relative_status,
        "max_abs_ux_scattered": float(max_abs_ux),
        "max_abs_uy_scattered": float(max_abs_uy),
        "max_norm_scattered": float(max_norm),
        "max_norm_healthy": float(max_norm_healthy),
        "max_x": float(max_x),
        "max_y": float(max_y),
    }
    for zone, value in sorted(zone_energy.items()):
        row[f"energy_zone_{zone}"] = float(value)
    return row


def _first_threshold_time(
    metrics: Sequence[Mapping[str, Any]],
    column: str,
    *,
    fraction: float,
) -> tuple[float | None, float, str]:
    values = np.asarray([float(row.get(column, 0.0)) for row in metrics])
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None, 0.0, "invalid"
    maximum = float(np.max(values))
    if maximum <= 0.0:
        return None, 0.0, "zero_signal"
    threshold = fraction * maximum
    indices = np.flatnonzero(values >= threshold)
    if indices.size == 0:
        return None, threshold, "not_reached"
    return float(metrics[int(indices[0])]["time"]), threshold, "ok"


def detect_characteristic_times(
    metrics: Sequence[Mapping[str, Any]],
    *,
    detection_fraction: float,
    zone_detection_fraction: float,
    imposed_arrival_time: float | None,
) -> dict[str, float | None]:
    if not metrics:
        raise ValueError("Cannot detect characteristic times from an empty series.")
    energy = np.asarray([float(row["energy_scattered"]) for row in metrics])
    max_index = int(np.argmax(energy))
    detected_birth, _, _ = _first_threshold_time(
        metrics,
        "energy_scattered",
        fraction=detection_fraction,
    )
    defect_arrival: float | None = None
    if any("energy_zone_defect" in row for row in metrics):
        defect_arrival, _, _ = _first_threshold_time(
            metrics,
            "energy_zone_defect",
            fraction=detection_fraction,
        )
    arrival = (
        imposed_arrival_time
        if imposed_arrival_time is not None
        else defect_arrival if defect_arrival is not None else detected_birth
    )
    out: dict[str, Any] = {
        # Preferred terminology: these are threshold-detection times,
        # not exact physical first-arrival times.
        "global_scattered_detection_time": detected_birth,
        "defect_activity_detection_time": defect_arrival,
        "selected_detection_time": arrival,
        "time_semantics": "threshold_detection_not_physical_first_arrival",

        # Backward-compatible aliases.
        "detected_scattered_birth_time": detected_birth,
        "defect_arrival_time": defect_arrival,
        "interaction_arrival_time": arrival,

        "max_energy_time": float(metrics[max_index]["time"]),
    }
    for zone in ("left", "right", "top", "bottom"):
        key = f"energy_zone_{zone}"
        if any(key in row for row in metrics):
            zone_time, _, _ = _first_threshold_time(
                metrics,
                key,
                fraction=zone_detection_fraction,
            )
            out[f"{zone}_arrival_time"] = zone_time
    return out


def validate_pre_interaction(
    metrics: Sequence[dict[str, Any]],
    *,
    arrival_time: float | None,
    atol: float,
    rtol: float,
    transition_guard_snapshots: int = 1,
) -> dict[str, Any]:
    """Validate the strictly pre-interaction scattered-field background.

    The last exported snapshot strictly preceding the detected arrival is
    treated as a transition sample because the arrival is quantized on the
    export-time grid.

    The integrated P1 L2 criterion is the primary conformity criterion.
    The pointwise maximum remains a secondary sensitivity diagnostic.
    """

    if (
        not isinstance(transition_guard_snapshots, int)
        or isinstance(transition_guard_snapshots, bool)
        or transition_guard_snapshots < 0
    ):
        raise ValueError(
            "transition_guard_snapshots must be a non-negative integer."
        )

    for row in metrics:
        row["pre_interaction"] = False
        row["pre_interaction_transition"] = False
        row["pre_interaction_pass"] = None
        row["pre_interaction_l2_pass"] = None
        row["pre_interaction_pointwise_pass"] = None
        row.pop("pre_interaction_max_tolerance", None)
        row.pop("pre_interaction_l2_tolerance", None)

    criterion = (
        "PRIMARY: l2_scattered <= "
        "atol*sqrt(domain_area) + rtol*l2_healthy"
    )
    secondary = (
        "max_norm_scattered <= "
        "atol + rtol*max_norm_healthy"
    )

    if arrival_time is None:
        return {
            "status": "WARNING",
            "reason": "arrival_time_unavailable",
            "arrival_time": None,
            "strict_preinteraction_cutoff_time": None,
            "num_pre_arrival_candidates": 0,
            "num_pre_interaction_snapshots": 0,
            "num_transition_snapshots": 0,
            "transition_times": [],
            "num_pass": 0,
            "num_fail": 0,
            "num_l2_pass": 0,
            "num_l2_fail": 0,
            "num_pointwise_pass": 0,
            "num_pointwise_fail": 0,
            "max_pre_interaction_scattered_norm": None,
            "max_pre_interaction_relative_l2": None,
            "transition_guard_snapshots_requested": int(
                transition_guard_snapshots
            ),
            "transition_guard_snapshots_applied": 0,
            "criterion": criterion,
            "secondary_diagnostic": secondary,
            "atol": float(atol),
            "rtol": float(rtol),
        }

    candidate_indices = [
        index
        for index, row in enumerate(metrics)
        if float(row["time"]) < float(arrival_time)
    ]

    # Preserve at least one strict pre-interaction sample whenever
    # pre-arrival data exist.  The transition guard must never consume
    # the complete validation core.
    max_guard_count = max(
        0,
        len(candidate_indices) - 1,
    )

    guard_count = min(
        transition_guard_snapshots,
        max_guard_count,
    )

    if guard_count:
        strict_indices = candidate_indices[:-guard_count]
        transition_indices = candidate_indices[-guard_count:]
    else:
        strict_indices = candidate_indices
        transition_indices = []

    strict_index_set = set(strict_indices)
    transition_index_set = set(transition_indices)

    l2_pass_count = 0
    l2_fail_count = 0
    pointwise_pass_count = 0
    pointwise_fail_count = 0
    max_scattered = 0.0
    max_relative = 0.0

    for index, row in enumerate(metrics):
        if index in transition_index_set:
            row["pre_interaction_transition"] = True
            continue

        if index not in strict_index_set:
            continue

        row["pre_interaction"] = True

        max_tolerance = (
            float(atol)
            + float(rtol) * float(row["max_norm_healthy"])
        )

        l2_tolerance = (
            float(atol)
            * math.sqrt(
                max(
                    float(row["domain_area_integrated"]),
                    0.0,
                )
            )
            + float(rtol) * float(row["l2_healthy"])
        )

        l2_passed = (
            float(row["l2_scattered"])
            <= l2_tolerance
        )

        pointwise_passed = (
            float(row["max_norm_scattered"])
            <= max_tolerance
        )

        row["pre_interaction_max_tolerance"] = float(
            max_tolerance
        )
        row["pre_interaction_l2_tolerance"] = float(
            l2_tolerance
        )
        row["pre_interaction_l2_pass"] = bool(l2_passed)
        row["pre_interaction_pointwise_pass"] = bool(
            pointwise_passed
        )

        # Backward-compatible field governed by the primary criterion.
        row["pre_interaction_pass"] = bool(l2_passed)

        l2_pass_count += int(l2_passed)
        l2_fail_count += int(not l2_passed)
        pointwise_pass_count += int(pointwise_passed)
        pointwise_fail_count += int(not pointwise_passed)

        max_scattered = max(
            max_scattered,
            float(row["max_norm_scattered"]),
        )

        relative = row.get("relative_l2")

        if (
            relative is not None
            and math.isfinite(float(relative))
        ):
            max_relative = max(
                max_relative,
                float(relative),
            )

    if transition_indices:
        strict_cutoff_time = float(
            metrics[transition_indices[0]]["time"]
        )
    else:
        strict_cutoff_time = float(arrival_time)

    if not strict_indices:
        status = "WARNING"
        reason = (
            "no_exported_snapshot_in_strict_"
            "preinteraction_core"
        )
    elif l2_fail_count:
        status = "FAIL"
        reason = (
            "strict_preinteraction_energy_"
            "tolerance_exceeded"
        )
    elif pointwise_fail_count:
        status = "WARNING"
        reason = (
            "strict_preinteraction_energy_pass_"
            "pointwise_tolerance_exceeded"
        )
    else:
        status = "PASS"
        reason = (
            "strict_preinteraction_energy_and_"
            "pointwise_checks_pass"
        )

    transition_times = [
        float(metrics[index]["time"])
        for index in transition_indices
    ]

    return {
        "status": status,
        "reason": reason,
        "arrival_time": float(arrival_time),
        "strict_preinteraction_cutoff_time": float(
            strict_cutoff_time
        ),
        "num_pre_arrival_candidates": int(
            len(candidate_indices)
        ),
        "num_pre_interaction_snapshots": int(
            len(strict_indices)
        ),
        "num_transition_snapshots": int(
            len(transition_indices)
        ),
        "transition_times": transition_times,
        "num_pass": int(l2_pass_count),
        "num_fail": int(l2_fail_count),
        "num_l2_pass": int(l2_pass_count),
        "num_l2_fail": int(l2_fail_count),
        "num_pointwise_pass": int(
            pointwise_pass_count
        ),
        "num_pointwise_fail": int(
            pointwise_fail_count
        ),
        "max_pre_interaction_scattered_norm": (
            float(max_scattered)
            if strict_indices
            else None
        ),
        "max_pre_interaction_relative_l2": (
            float(max_relative)
            if strict_indices
            else None
        ),
        "transition_guard_snapshots_requested": int(
            transition_guard_snapshots
        ),
        "transition_guard_snapshots_applied": int(
            guard_count
        ),
        "criterion": criterion,
        "secondary_diagnostic": secondary,
        "atol": float(atol),
        "rtol": float(rtol),
    }





def validate_pre_detection_relative(
    metrics: Sequence[dict[str, Any]],
    *,
    detection_time: float | None,
    detection_fraction: float,
    transition_guard_snapshots: int = 1,
) -> dict[str, Any]:
    """Validate that the scattered field is negligible before threshold detection.

    ``detection_time`` is an algorithmic threshold-detection time, not a
    geometrical or physical first-arrival time.

    The primary criterion is global and relative:

        max_pre(E_scattered) / max_all(E_scattered)
            <= detection_fraction

    together with the equivalent L2-amplitude criterion

        max_pre(||u_scattered||_L2) / max_all(||u_scattered||_L2)
            <= sqrt(detection_fraction).

    The pointwise Linf ratio is reported as a secondary diagnostic only.
    """

    fraction = float(detection_fraction)

    if (
        not math.isfinite(fraction)
        or fraction <= 0.0
        or fraction > 1.0
    ):
        raise ValueError(
            "detection_fraction must lie in (0, 1]."
        )

    if (
        not isinstance(transition_guard_snapshots, int)
        or isinstance(transition_guard_snapshots, bool)
        or transition_guard_snapshots < 0
    ):
        raise ValueError(
            "transition_guard_snapshots must be a non-negative integer."
        )

    for row in metrics:
        # Preferred threshold-detection terminology.
        row["pre_detection"] = False
        row["pre_detection_transition"] = False
        row["pre_detection_pass"] = None

        # Backward-compatible aliases.
        row["pre_interaction"] = False
        row["pre_interaction_transition"] = False
        row["pre_interaction_pass"] = None

        row["pre_detection_energy_pass"] = None
        row["pre_detection_l2_pass"] = None
        row["pre_detection_pointwise_pass"] = None

    criterion = (
        "PRIMARY: max_pre(energy_scattered)/max_all(energy_scattered) "
        "<= detection_fraction AND "
        "max_pre(l2_scattered)/max_all(l2_scattered) "
        "<= sqrt(detection_fraction)"
    )

    secondary = (
        "SECONDARY: "
        "max_pre(max_norm_scattered)/max_all(max_norm_scattered) "
        "<= sqrt(detection_fraction)"
    )

    if detection_time is None:
        return {
            "status": "WARNING",
            "reason": "threshold_detection_time_unavailable",
            "validation_mode": "relative_pre_detection",
            "time_semantics": (
                "threshold_detection_not_physical_first_arrival"
            ),
            "detection_time": None,
            "arrival_time": None,
            "detection_fraction": fraction,
            "amplitude_fraction": math.sqrt(fraction),
            "criterion": criterion,
            "secondary_diagnostic": secondary,
            "num_pre_arrival_candidates": 0,
            "num_pre_interaction_snapshots": 0,
            "num_transition_snapshots": 0,
            "transition_times": [],
        }

    candidate_indices = [
        index
        for index, row in enumerate(metrics)
        if float(row["time"]) < float(detection_time)
    ]

    max_guard_count = max(
        0,
        len(candidate_indices) - 1,
    )

    guard_count = min(
        transition_guard_snapshots,
        max_guard_count,
    )

    if guard_count:
        strict_indices = candidate_indices[:-guard_count]
        transition_indices = candidate_indices[-guard_count:]
    else:
        strict_indices = candidate_indices
        transition_indices = []

    for index in transition_indices:
        metrics[index]["pre_detection_transition"] = True
        metrics[index]["pre_interaction_transition"] = True

    for index in strict_indices:
        metrics[index]["pre_detection"] = True
        metrics[index]["pre_interaction"] = True

    amplitude_fraction = math.sqrt(fraction)

    global_energy = max(
        (float(row["energy_scattered"]) for row in metrics),
        default=0.0,
    )
    global_l2 = max(
        (float(row["l2_scattered"]) for row in metrics),
        default=0.0,
    )
    global_linf = max(
        (float(row["max_norm_scattered"]) for row in metrics),
        default=0.0,
    )

    def safe_ratio(numerator: float, denominator: float) -> float:
        if denominator > 0.0:
            return float(numerator / denominator)

        if numerator == 0.0:
            return 0.0

        return math.inf

    if strict_indices:
        pre_energy = max(
            float(metrics[index]["energy_scattered"])
            for index in strict_indices
        )
        pre_l2 = max(
            float(metrics[index]["l2_scattered"])
            for index in strict_indices
        )
        pre_linf = max(
            float(metrics[index]["max_norm_scattered"])
            for index in strict_indices
        )
    else:
        pre_energy = 0.0
        pre_l2 = 0.0
        pre_linf = 0.0

    energy_ratio = safe_ratio(
        pre_energy,
        global_energy,
    )
    l2_ratio = safe_ratio(
        pre_l2,
        global_l2,
    )
    pointwise_ratio = safe_ratio(
        pre_linf,
        global_linf,
    )

    energy_pass = bool(
        energy_ratio <= fraction
    )
    l2_pass = bool(
        l2_ratio <= amplitude_fraction
    )
    pointwise_pass = bool(
        pointwise_ratio <= amplitude_fraction
    )

    # Per-snapshot fields are useful in the exported metrics CSV.
    for index in strict_indices:
        row = metrics[index]

        row_energy_ratio = safe_ratio(
            float(row["energy_scattered"]),
            global_energy,
        )
        row_l2_ratio = safe_ratio(
            float(row["l2_scattered"]),
            global_l2,
        )
        row_pointwise_ratio = safe_ratio(
            float(row["max_norm_scattered"]),
            global_linf,
        )

        row["pre_detection_energy_ratio"] = row_energy_ratio
        row["pre_detection_l2_ratio"] = row_l2_ratio
        row["pre_detection_pointwise_ratio"] = row_pointwise_ratio

        row["pre_detection_energy_pass"] = bool(
            row_energy_ratio <= fraction
        )
        row["pre_detection_l2_pass"] = bool(
            row_l2_ratio <= amplitude_fraction
        )
        row["pre_detection_pointwise_pass"] = bool(
            row_pointwise_ratio <= amplitude_fraction
        )

        # Backward-compatible generic flag that follows the primary
        # relative pre-detection criterion.
        primary_pass = bool(
            row["pre_detection_energy_pass"]
            and row["pre_detection_l2_pass"]
        )

        row["pre_detection_pass"] = primary_pass
        row["pre_interaction_pass"] = primary_pass

    if transition_indices:
        cutoff_time = float(
            metrics[transition_indices[0]]["time"]
        )
    else:
        cutoff_time = float(detection_time)

    if not strict_indices:
        status = "WARNING"
        reason = "no_snapshot_in_strict_pre_detection_core"
    elif not energy_pass or not l2_pass:
        status = "FAIL"
        reason = "relative_pre_detection_primary_criterion_exceeded"
    else:
        # Pointwise Linf is intentionally secondary and does not invalidate
        # the primary energy/L2 conformity criterion.
        status = "PASS"
        reason = "relative_pre_detection_energy_and_l2_pass"

    return {
        "status": status,
        "reason": reason,
        "validation_mode": "relative_pre_detection",
        "time_semantics": (
            "threshold_detection_not_physical_first_arrival"
        ),
        # Keep arrival_time as a compatibility alias only.
        "arrival_time": float(detection_time),
        "detection_time": float(detection_time),
        "strict_preinteraction_cutoff_time": float(cutoff_time),
        "strict_pre_detection_cutoff_time": float(cutoff_time),
        "detection_fraction": fraction,
        "amplitude_fraction": amplitude_fraction,
        "num_pre_arrival_candidates": int(
            len(candidate_indices)
        ),
        "num_pre_interaction_snapshots": int(
            len(strict_indices)
        ),
        "num_pre_detection_snapshots": int(
            len(strict_indices)
        ),
        "num_transition_snapshots": int(
            len(transition_indices)
        ),
        "transition_times": [
            float(metrics[index]["time"])
            for index in transition_indices
        ],
        "global_max_energy_scattered": float(global_energy),
        "global_max_l2_scattered": float(global_l2),
        "global_max_norm_scattered": float(global_linf),
        "max_pre_detection_energy_scattered": float(pre_energy),
        "max_pre_detection_l2_scattered": float(pre_l2),
        "max_pre_detection_norm_scattered": float(pre_linf),
        "pre_to_global_energy_ratio": float(energy_ratio),
        "pre_to_global_l2_ratio": float(l2_ratio),
        "pre_to_global_pointwise_ratio": float(pointwise_ratio),
        "relative_energy_pass": bool(energy_pass),
        "relative_l2_pass": bool(l2_pass),
        "relative_pointwise_pass": bool(pointwise_pass),
        "secondary_pointwise_status": (
            "PASS" if pointwise_pass else "WARNING"
        ),
        "criterion": criterion,
        "secondary_diagnostic": secondary,
    }



def select_scientific_times(
    pairs: Sequence[SnapshotPair],
    characteristic_times: Mapping[str, float | None],
    *,
    user_times: Sequence[float] | None,
) -> tuple[dict[str, Any], ...]:
    available = np.asarray([pair.time for pair in pairs], dtype=np.float64)
    labels: list[tuple[str, float]] = []
    arrival = characteristic_times.get("selected_detection_time")
    if arrival is None:
        arrival = characteristic_times.get("interaction_arrival_time")

    if arrival is not None:
        earlier = available[available < float(arrival)]
        if earlier.size:
            labels.append(
                ("before_defect_activity_detection", float(earlier[-1]))
            )
        labels.append(
            ("defect_activity_detection", float(arrival))
        )
    for key in (
        "detected_scattered_birth_time",
        "max_energy_time",
        "left_arrival_time",
        "right_arrival_time",
        "top_arrival_time",
        "bottom_arrival_time",
    ):
        value = characteristic_times.get(key)
        if value is not None:
            labels.append((key.removesuffix("_time"), float(value)))
    if user_times:
        labels.extend((f"user_{index}", float(value)) for index, value in enumerate(user_times))
    if not labels:
        labels = [("first", float(available[0])), ("last", float(available[-1]))]

    selected: dict[int, dict[str, Any]] = {}
    for label, target in labels:
        index = int(np.argmin(np.abs(available - target)))
        item = selected.setdefault(
            index,
            {
                "snapshot_index": index,
                "time": float(available[index]),
                "labels": [],
                "requested_times": [],
            },
        )
        item["labels"].append(label)
        item["requested_times"].append(float(target))
    return tuple(selected[index] for index in sorted(selected))


def _infer_metadata_from_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    manifest = _require_file(path, role="campaign manifest")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScatteredWavefieldError(f"Malformed JSON manifest {manifest}: {exc}") from exc
    out: dict[str, Any] = {"campaign_manifest": str(manifest)}
    for key in (
        "defect_center_x",
        "defect_center_y",
        "defect_radius",
        "domain_length_x",
        "domain_length_y",
        "dt",
        "every_k",
        "num_steps",
        "final_time",
        "simulated_material_kind",
        "observed_material_kind",
    ):
        if key in payload:
            out[key] = payload[key]
    box = payload.get("receiver_box")
    if isinstance(box, Mapping):
        coordinates = [box.get(key) for key in ("xmin", "xmax", "ymin", "ymax")]
        if all(value is not None for value in coordinates):
            out["analysis_box"] = {
                "xmin": float(coordinates[0]),
                "xmax": float(coordinates[1]),
                "ymin": float(coordinates[2]),
                "ymax": float(coordinates[3]),
            }
    return out


def resolve_analysis_metadata(config: ScatteredWavefieldConfig) -> dict[str, Any]:
    inferred = _infer_metadata_from_manifest(config.inputs.campaign_manifest)
    defect_center = config.defect_center
    if defect_center is None and {
        "defect_center_x",
        "defect_center_y",
    }.issubset(inferred):
        defect_center = (
            float(inferred["defect_center_x"]),
            float(inferred["defect_center_y"]),
        )
    defect_radius = config.defect_radius
    if defect_radius is None and "defect_radius" in inferred:
        defect_radius = float(inferred["defect_radius"])
    analysis_box = config.analysis_box
    if analysis_box is None and "analysis_box" in inferred:
        box = inferred["analysis_box"]
        analysis_box = AnalysisBox(
            xmin=float(box["xmin"]),
            xmax=float(box["xmax"]),
            ymin=float(box["ymin"]),
            ymax=float(box["ymax"]),
        )
    return {
        "inferred": inferred,
        "defect_center": defect_center,
        "defect_radius": defect_radius,
        "analysis_box": analysis_box,
    }


def run_scattered_wavefield_analysis(
    config: ScatteredWavefieldConfig,
    *,
    logger: logging.Logger | None = None,
) -> ScatteredWavefieldResult:
    config.validate()
    healthy_series = parse_pvd_series(config.inputs.healthy_pvd)
    defect_series = parse_pvd_series(config.inputs.defect_pvd)
    pairs = pair_pvd_series(
        healthy_series,
        defect_series,
        atol=config.time_atol,
        rtol=config.time_rtol,
    )
    metadata = resolve_analysis_metadata(config)
    analysis_box = metadata["analysis_box"]
    defect_center = metadata["defect_center"]
    defect_radius = metadata["defect_radius"]

    metrics: list[dict[str, Any]] = []
    start = time.perf_counter()
    for index, pair in enumerate(pairs, start=1):
        if logger is not None:
            logger.info(
                "[%d/%d] Processing scattered snapshot t=%g",
                index,
                len(pairs),
                pair.time,
            )
        metrics.append(
            _snapshot_metrics(
                pair,
                healthy_field=config.healthy_field,
                defect_field=config.defect_field,
                geometry_atol=config.geometry_atol,
                geometry_rtol=config.geometry_rtol,
                analysis_box=analysis_box,
                zone_width=config.zone_width,
                defect_center=defect_center,
                defect_radius=defect_radius,
                ratio_epsilon=config.ratio_epsilon,
            )
        )
    elapsed = time.perf_counter() - start

    characteristic_times = detect_characteristic_times(
        metrics,
        detection_fraction=config.detection_fraction,
        zone_detection_fraction=config.zone_detection_fraction,
        imposed_arrival_time=config.arrival_time,
    )
    validation = validate_pre_detection_relative(
        metrics,
        detection_time=characteristic_times.get("interaction_arrival_time"),
        detection_fraction=config.detection_fraction,
    )
    selected = select_scientific_times(
        pairs,
        characteristic_times,
        user_times=config.snapshot_times,
    )
    metadata_out = {
        **metadata,
        "healthy_series": healthy_series,
        "defect_series": defect_series,
        "pairs": pairs,
        "processing_seconds": float(elapsed),
        "subtraction_convention": "u_scattered = u_defect - u_healthy",
        "energy_definition": (
            "Exact P1 triangle integral of ux_scattered^2 + uy_scattered^2, "
            "summed over owned cells."
        ),
        "zone_energy_definition": (
            "Same P1 integral restricted to cells whose centroid lies in the "
            "configured receiver-band zone."
        ),
    }
    return ScatteredWavefieldResult(
        metrics=tuple(metrics),
        selected_times=selected,
        characteristic_times=characteristic_times,
        validation=validation,
        metadata=metadata_out,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _file_metadata(path: Path) -> dict[str, Any]:
    resolved = _require_file(path, role="input file")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _git_commit(repository_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, AnalysisBox):
        return asdict(value)
    if isinstance(value, (PVDSeries, PVDEntry, SnapshotPair)):
        return asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_snapshot_scattered_pieces(
    pair: SnapshotPair,
    *,
    healthy_field: str,
    defect_field: str,
    geometry_atol: float,
    geometry_rtol: float,
) -> Iterator[tuple[int, Path, Any, np.ndarray]]:
    healthy_pieces = dataset_piece_paths(pair.healthy_dataset)
    defect_pieces = dataset_piece_paths(pair.defect_dataset)
    if len(healthy_pieces) != len(defect_pieces):
        raise MeshCompatibilityError(
            f"Partition count mismatch at t={pair.time:g}."
        )
    for piece_index, (healthy_piece, defect_piece) in enumerate(
        zip(healthy_pieces, defect_pieces, strict=True)
    ):
        healthy_mesh = _load_mesh(healthy_piece)
        defect_mesh = _load_mesh(defect_piece)
        validate_mesh_pair(
            healthy_mesh,
            defect_mesh,
            healthy_piece=healthy_piece,
            defect_piece=defect_piece,
            atol=geometry_atol,
            rtol=geometry_rtol,
        )
        healthy_values = _extract_vector_point_field(
            healthy_mesh, healthy_field, piece=healthy_piece
        )
        defect_values = _extract_vector_point_field(
            defect_mesh, defect_field, piece=defect_piece
        )
        scattered = compute_scattered_field(defect=defect_values, reference=healthy_values)
        yield piece_index, healthy_piece, healthy_mesh, scattered


def _write_pvtu(path: Path, piece_names: Sequence[str]) -> None:
    root = ET.Element(
        "VTKFile",
        {
            "type": "PUnstructuredGrid",
            "version": "1.0",
            "byte_order": "LittleEndian",
        },
    )
    grid = ET.SubElement(root, "PUnstructuredGrid", {"GhostLevel": "0"})
    point_data = ET.SubElement(grid, "PPointData", {"Vectors": "u_scattered"})
    ET.SubElement(
        point_data,
        "PDataArray",
        {"type": "Float64", "Name": "u_scattered", "NumberOfComponents": "3"},
    )
    for name in ("u_scattered_x", "u_scattered_y", "u_scattered_norm"):
        ET.SubElement(
            point_data,
            "PDataArray",
            {"type": "Float64", "Name": name, "NumberOfComponents": "1"},
        )
    ET.SubElement(grid, "PCellData")
    points = ET.SubElement(grid, "PPoints")
    ET.SubElement(
        points,
        "PDataArray",
        {"type": "Float64", "NumberOfComponents": "3"},
    )
    for piece_name in piece_names:
        ET.SubElement(grid, "Piece", {"Source": piece_name})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_pvd(path: Path, datasets: Sequence[tuple[float, str]]) -> None:
    root = ET.Element("VTKFile", {"type": "Collection", "version": "1.0"})
    collection = ET.SubElement(root, "Collection")
    for time_value, file_name in datasets:
        ET.SubElement(
            collection,
            "DataSet",
            {"timestep": f"{time_value:.17g}", "part": "0", "file": file_name},
        )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def export_scattered_vtk(
    pairs: Sequence[SnapshotPair],
    selected_indices: Sequence[int],
    *,
    config: ScatteredWavefieldConfig,
    output_dir: Path,
    logger: logging.Logger | None = None,
) -> Path | None:
    if config.vtk_export_mode == "none":
        return None
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover
        raise ScatteredWavefieldError("meshio is required for VTK export.") from exc

    if config.vtk_export_mode == "all":
        indices = tuple(range(len(pairs)))
    else:
        indices = tuple(sorted(set(int(index) for index in selected_indices)))
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets: list[tuple[float, str]] = []
    for output_index, pair_index in enumerate(indices):
        pair = pairs[pair_index]
        if logger is not None:
            logger.info("Exporting scattered VTK snapshot t=%g", pair.time)
        piece_names: list[str] = []
        for piece_index, healthy_piece, mesh, scattered in _load_snapshot_scattered_pieces(
            pair,
            healthy_field=config.healthy_field,
            defect_field=config.defect_field,
            geometry_atol=config.geometry_atol,
            geometry_rtol=config.geometry_rtol,
        ):
            vector3 = np.zeros((scattered.shape[0], 3), dtype=np.float64)
            vector3[:, :2] = scattered
            point_data = {
                "u_scattered": vector3,
                "u_scattered_x": scattered[:, 0],
                "u_scattered_y": scattered[:, 1],
                "u_scattered_norm": np.linalg.norm(scattered, axis=1),
            }
            piece_name = (
                f"u_scattered_{output_index:06d}_piece{piece_index:04d}.vtu"
            )
            piece_path = output_dir / piece_name
            output_cells = _p1_triangle_cells_for_export(
                mesh,
                context=str(healthy_piece),
            )

            output_mesh = meshio.Mesh(
                points=np.asarray(mesh.points, dtype=np.float64),
                cells=output_cells,
                point_data=point_data,
            )
            meshio.write(piece_path, output_mesh, binary=True)
            piece_names.append(piece_name)
            del mesh, scattered, output_mesh
        pvtu_name = f"u_scattered_{output_index:06d}.pvtu"
        _write_pvtu(output_dir / pvtu_name, piece_names)
        datasets.append((pair.time, pvtu_name))
    pvd_path = output_dir / "u_scattered.pvd"
    _write_pvd(pvd_path, datasets)
    return pvd_path


def _rasterize_selected_snapshot(
    pair: SnapshotPair,
    *,
    config: ScatteredWavefieldConfig,
    extent: tuple[float, float, float, float],
) -> dict[str, np.ndarray]:
    xmin, xmax, ymin, ymax = extent
    resolution = config.raster_resolution
    sums = {
        "ux": np.zeros((resolution, resolution), dtype=np.float64),
        "uy": np.zeros((resolution, resolution), dtype=np.float64),
        "norm": np.zeros((resolution, resolution), dtype=np.float64),
        "weight": np.zeros((resolution, resolution), dtype=np.float64),
    }
    for _, _, mesh, scattered in _load_snapshot_scattered_pieces(
        pair,
        healthy_field=config.healthy_field,
        defect_field=config.defect_field,
        geometry_atol=config.geometry_atol,
        geometry_rtol=config.geometry_rtol,
    ):
        points = np.asarray(mesh.points, dtype=np.float64)
        for block_index, block in enumerate(mesh.cells):
            cell_type, connectivity = _cell_type_and_data(block)

            connectivity = _require_p1_triangle_connectivity(
                cell_type=cell_type,
                connectivity=connectivity,
                context=(
                    "scattered-wavefield rasterization at "
                    f"t={pair.time:g}"
                ),
            )

            if connectivity is None:
                continue

            owned = _owned_cell_mask(
                mesh,
                block_index,
                connectivity.shape[0],
            )
            triangles = connectivity[owned]
            if triangles.size == 0:
                continue
            centers = np.mean(points[triangles, :2], axis=1)
            area = _triangle_area(points, triangles)
            cell_ux = np.mean(scattered[triangles, 0], axis=1)
            cell_uy = np.mean(scattered[triangles, 1], axis=1)
            cell_norm = np.mean(np.linalg.norm(scattered[triangles], axis=2), axis=1)
            ix = np.floor((centers[:, 0] - xmin) / (xmax - xmin) * resolution).astype(int)
            iy = np.floor((centers[:, 1] - ymin) / (ymax - ymin) * resolution).astype(int)
            valid = (ix >= 0) & (ix < resolution) & (iy >= 0) & (iy < resolution)
            flat = iy[valid] * resolution + ix[valid]
            size = resolution * resolution
            sums["weight"].ravel()[:] += np.bincount(
                flat, weights=area[valid], minlength=size
            )
            sums["ux"].ravel()[:] += np.bincount(
                flat, weights=area[valid] * cell_ux[valid], minlength=size
            )
            sums["uy"].ravel()[:] += np.bincount(
                flat, weights=area[valid] * cell_uy[valid], minlength=size
            )
            sums["norm"].ravel()[:] += np.bincount(
                flat, weights=area[valid] * cell_norm[valid], minlength=size
            )
        del mesh, scattered
    weight = sums.pop("weight")
    occupied = weight > 0.0
    for key in tuple(sums):
        out = np.full_like(weight, np.nan)
        out[occupied] = sums[key][occupied] / weight[occupied]
        sums[key] = out
    return sums


def _plot_time_series(
    metrics: Sequence[Mapping[str, Any]],
    paths: ScatteredOutputPaths,
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    times = np.asarray([row["time"] for row in metrics], dtype=float)
    outputs: list[Path] = []

    figures = [
        (
            "scattered_energy_evolution",
            [("energy_scattered", r"$\int_\Omega |u^{sc}|^2\,dx$")],
            "Scattered displacement quadratic integral",
        ),
        (
            "scattered_component_maxima",
            [
                ("max_abs_ux_scattered", r"$\max |u_x^{sc}|$"),
                ("max_abs_uy_scattered", r"$\max |u_y^{sc}|$"),
                ("max_norm_scattered", r"$\max \|u^{sc}\|_2$"),
            ],
            "Scattered-field maxima",
        ),
        (
            "scattered_relative_l2",
            [("relative_l2", r"$\|u^{sc}\|_{L^2}/\|u^{healthy}\|_{L^2}$")],
            "Relative scattered-field norm",
        ),
    ]
    zone_columns = [
        (column, column.removeprefix("energy_zone_").replace("_", " "))
        for column in metrics[0]
        if column.startswith("energy_zone_") and column != "energy_zone_all"
    ]
    if zone_columns:
        figures.append(
            (
                "scattered_zone_energy",
                zone_columns,
                "Scattered displacement indicator by zone",
            )
        )

    for stem, columns, title in figures:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for column, label in columns:
            values = np.asarray(
                [np.nan if row.get(column) is None else row.get(column) for row in metrics],
                dtype=float,
            )
            ax.plot(times, values, label=label)
        ax.set_xlabel("Physical time")
        ax.set_ylabel("Amplitude" if "maxima" in stem else "Quadratic indicator")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if len(columns) > 1:
            ax.legend()
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = paths.figures_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=180 if suffix == "png" else None)
            outputs.append(path)
        plt.close(fig)
    return outputs


def _plot_snapshot_board(
    pairs: Sequence[SnapshotPair],
    selected: Sequence[Mapping[str, Any]],
    *,
    config: ScatteredWavefieldConfig,
    paths: ScatteredOutputPaths,
    extent: tuple[float, float, float, float],
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rasters: list[tuple[Mapping[str, Any], dict[str, np.ndarray]]] = []
    for item in selected:
        index = int(item["snapshot_index"])
        rasters.append(
            (
                item,
                _rasterize_selected_snapshot(
                    pairs[index],
                    config=config,
                    extent=extent,
                ),
            )
        )
    if not rasters:
        return []
    signed_max = max(
        float(np.nanmax(np.abs(values[key])))
        for _, values in rasters
        for key in ("ux", "uy")
    )
    norm_max = max(float(np.nanmax(values["norm"])) for _, values in rasters)
    signed_max = max(signed_max, np.finfo(float).tiny)
    norm_max = max(norm_max, np.finfo(float).tiny)
    rows = len(rasters)
    fig, axes = plt.subplots(rows, 3, figsize=(12.0, 3.4 * rows), squeeze=False)
    for row_index, (item, values) in enumerate(rasters):
        label = ", ".join(str(value) for value in item["labels"])
        for column_index, (key, title) in enumerate(
            (("ux", r"$u_x^{sc}$"), ("uy", r"$u_y^{sc}$"), ("norm", r"$\|u^{sc}\|_2$"))
        ):
            ax = axes[row_index, column_index]
            if key == "norm":
                image = ax.imshow(
                    values[key],
                    origin="lower",
                    extent=extent,
                    vmin=0.0,
                    vmax=norm_max,
                    cmap="viridis",
                    aspect="equal",
                )
            else:
                image = ax.imshow(
                    values[key],
                    origin="lower",
                    extent=extent,
                    vmin=-signed_max,
                    vmax=signed_max,
                    cmap="RdBu_r",
                    aspect="equal",
                )
            ax.set_title(f"{title}, t={float(item['time']):g}\n{label}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Physical scattered displacement: defect minus healthy", y=1.0)
    fig.tight_layout()
    outputs: list[Path] = []
    for suffix in ("png", "pdf"):
        path = paths.figures_dir / f"scattered_wavefield_snapshots.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_scattered_wavefield_outputs(
    result: ScatteredWavefieldResult,
    config: ScatteredWavefieldConfig,
    *,
    command: Sequence[str],
    repository_root: Path,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    paths = ScatteredOutputPaths.from_root(config.output_dir)
    paths.create()
    analysis_mode = {
        "line": "line-only",
        "box": "box-only",
        "unknown": "single-campaign",
    }[config.geometry_kind]
    metrics_rows = tuple(
        {
            "analysis_mode": analysis_mode,
            "geometry_kind": config.geometry_kind,
            **dict(row),
        }
        for row in result.metrics
    )
    selected_rows = tuple(
        {
            "analysis_mode": analysis_mode,
            "geometry_kind": config.geometry_kind,
            **dict(row),
        }
        for row in result.selected_times
    )
    metrics_path = paths.csv_dir / "scattered_wavefield_metrics.csv"
    selected_path = paths.csv_dir / "scattered_wavefield_selected_times.csv"
    _write_csv(metrics_path, metrics_rows)
    _write_csv(selected_path, selected_rows)

    pairs = tuple(result.metadata["pairs"])
    healthy_series = result.metadata["healthy_series"]
    defect_series = result.metadata["defect_series"]
    selected_indices = [int(item["snapshot_index"]) for item in result.selected_times]

    inferred = result.metadata.get("inferred", {})
    domain_x = float(inferred.get("domain_length_x", 0.0) or 0.0)
    domain_y = float(inferred.get("domain_length_y", 0.0) or 0.0)
    if domain_x > 0.0 and domain_y > 0.0:
        extent = (0.0, domain_x, 0.0, domain_y)
    else:
        first_piece = dataset_piece_paths(pairs[0].healthy_dataset)[0]
        mesh = _load_mesh(first_piece)
        points = np.asarray(mesh.points, dtype=float)
        extent = (
            float(np.min(points[:, 0])),
            float(np.max(points[:, 0])),
            float(np.min(points[:, 1])),
            float(np.max(points[:, 1])),
        )
        del mesh

    figure_paths = _plot_time_series(result.metrics, paths)
    figure_paths.extend(
        _plot_snapshot_board(
            pairs,
            result.selected_times,
            config=config,
            paths=paths,
            extent=extent,
        )
    )
    pvd_path = export_scattered_vtk(
        pairs,
        selected_indices,
        config=config,
        output_dir=paths.vtk_dir,
        logger=logger,
    )

    summary = {
        "schema_version": "1.0",
        "status": "ok" if result.validation.get("status") != "FAIL" else "warning",
        "analysis_mode": analysis_mode,
        "geometry_kind": config.geometry_kind,
        "num_snapshots": len(result.metrics),
        "time_min": float(result.metrics[0]["time"]),
        "time_max": float(result.metrics[-1]["time"]),
        "characteristic_times": result.characteristic_times,
        "pre_interaction_validation": result.validation,
        "selected_times": result.selected_times,
        "subtraction_convention": "u_scattered = u_defect - u_healthy",
        "healthy_field": config.healthy_field,
        "defect_field": config.defect_field,
        "vtk_export_mode": config.vtk_export_mode,
        "warnings": [],
    }
    if result.validation.get("status") == "WARNING":
        summary["warnings"].append(result.validation.get("reason"))
    summary_path = paths.manifests_dir / "scattered_wavefield_summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    input_metadata = {
        "healthy_pvd": _file_metadata(config.inputs.healthy_pvd),
        "defect_pvd": _file_metadata(config.inputs.defect_pvd),
    }
    if config.inputs.campaign_manifest is not None:
        input_metadata["campaign_manifest"] = _file_metadata(
            config.inputs.campaign_manifest
        )
    produced = [metrics_path, selected_path, summary_path, *figure_paths]
    if pvd_path is not None:
        produced.append(pvd_path)
    manifest = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_mode": analysis_mode,
        "geometry_kind": config.geometry_kind,
        "command": list(command),
        "inputs": input_metadata,
        "config": _json_safe(asdict(config)),
        "series": {
            "healthy": _json_safe(healthy_series),
            "defect": _json_safe(defect_series),
        },
        "definitions": {
            "subtraction": "u_scattered = u_defect - u_healthy",
            "energy_scattered": result.metadata["energy_definition"],
            "zone_energy": result.metadata["zone_energy_definition"],
            "pre_interaction": result.validation.get("criterion"),
        },
        "characteristic_times": result.characteristic_times,
        "validation": result.validation,
        "processing_seconds": result.metadata["processing_seconds"],
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {},
        "git_commit": _git_commit(repository_root),
        "produced_files": [str(path) for path in produced],
    }
    for module_name in ("numpy", "meshio", "matplotlib"):
        try:
            module = __import__(module_name)
            manifest["versions"][module_name] = getattr(module, "__version__", None)
        except ImportError:
            manifest["versions"][module_name] = None
    manifest_path = paths.manifests_dir / "scattered_wavefield_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    outputs = {
        "metrics_csv": metrics_path,
        "selected_times_csv": selected_path,
        "summary_json": summary_path,
        "manifest_json": manifest_path,
    }
    for index, path in enumerate(figure_paths, start=1):
        outputs[f"figure_{index}"] = path
    if pvd_path is not None:
        outputs["scattered_pvd"] = pvd_path
    return outputs
