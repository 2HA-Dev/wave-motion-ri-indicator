from __future__ import annotations

import csv
import json
import math
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from elastic_fwi.metrics.image_quality import (
    ImageQualityMetrics,
    compute_image_quality_metrics,
)


@dataclass(frozen=True)
class LoadedImageField:
    field_name: str
    centers: np.ndarray
    values: np.ndarray
    xdmf_path: str = ""
    h5_path: str = ""
    h5_dataset: str = ""
    association: str = "unknown"
    source_format: str = "xdmf"
    source_path: str | None = None
    source_dataset: str | None = None

    def as_dict(self) -> dict[str, Any]:
        source_path = self.source_path or self.xdmf_path
        source_dataset = self.source_dataset or self.h5_dataset
        return {
            "field_name": self.field_name,
            "source_format": self.source_format,
            "source_path": source_path,
            "source_dataset": source_dataset,
            "xdmf_path": self.xdmf_path,
            "h5_path": self.h5_path,
            "h5_dataset": self.h5_dataset,
            "association": self.association,
            "num_values": int(np.asarray(self.values).shape[0]),
        }


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _children_by_name(node: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(node) if _local_name(child.tag) == name]


def _first_descendant(node: ET.Element, name: str) -> ET.Element | None:
    for item in node.iter():
        if _local_name(item.tag) == name:
            return item
    return None


def _parse_dataitem_reference(text: str, *, base_dir: Path) -> tuple[Path, str]:
    raw = str(text).strip()
    if not raw:
        raise ValueError("Empty XDMF DataItem reference.")
    if ":" not in raw:
        raise ValueError(f"Unsupported XDMF DataItem reference without HDF5 path: {raw!r}.")
    file_part, dataset_part = raw.split(":", 1)
    h5_path = Path(file_part)
    if not h5_path.is_absolute():
        h5_path = base_dir / h5_path
    dataset_part = dataset_part.strip()
    if not dataset_part.startswith("/"):
        dataset_part = "/" + dataset_part
    return h5_path, dataset_part


def _find_grid_with_mesh(root: ET.Element) -> ET.Element:
    for grid in root.iter():
        if _local_name(grid.tag) != "Grid":
            continue
        if _first_descendant(grid, "Topology") is not None and _first_descendant(grid, "Geometry") is not None:
            return grid
    raise ValueError("Could not find an XDMF Grid containing Topology and Geometry.")


def _read_hdf5_dataset(h5_path: Path, dataset_path: str) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read XDMF/HDF5 image fields.") from exc

    with h5py.File(h5_path, "r") as handle:
        if dataset_path not in handle:
            available: list[str] = []
            handle.visititems(lambda name, obj: available.append("/" + name) if hasattr(obj, "shape") else None)
            raise KeyError(
                f"Dataset {dataset_path!r} not found in {h5_path}. "
                f"Available datasets include: {available[:20]}"
            )
        return np.asarray(handle[dataset_path])


def _cell_centers_from_topology_geometry(topology: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    topo = np.asarray(topology)
    geom = np.asarray(geometry, dtype=float)

    if geom.ndim != 2 or geom.shape[1] < 2:
        raise ValueError("Geometry dataset must have shape (n_points, dim>=2).")

    if topo.ndim == 1:
        topo = topo.reshape(-1, 1)
    if topo.ndim != 2:
        raise ValueError("Topology dataset must be one- or two-dimensional.")

    topo_int = np.asarray(topo, dtype=np.int64)
    if topo_int.size == 0:
        return np.empty((0, 2), dtype=float)

    valid = (topo_int >= 0) & (topo_int < geom.shape[0])
    if not np.all(valid):
        keep_columns = np.all(valid, axis=0)
        topo_int = topo_int[:, keep_columns]
        if topo_int.shape[1] == 0:
            raise ValueError("Could not infer valid topology node columns from XDMF data.")

    return np.mean(geom[topo_int, :2], axis=1)


def _load_mesh_centers(xdmf_path: Path, root: ET.Element) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = _find_grid_with_mesh(root)
    topology_node = _first_descendant(grid, "Topology")
    geometry_node = _first_descendant(grid, "Geometry")
    if topology_node is None or geometry_node is None:
        raise ValueError("Missing topology or geometry in XDMF grid.")

    topology_dataitem = _first_descendant(topology_node, "DataItem")
    geometry_dataitem = _first_descendant(geometry_node, "DataItem")
    if topology_dataitem is None or geometry_dataitem is None:
        raise ValueError("Missing topology or geometry DataItem in XDMF file.")

    topo_h5, topo_dataset = _parse_dataitem_reference(topology_dataitem.text or "", base_dir=xdmf_path.parent)
    geom_h5, geom_dataset = _parse_dataitem_reference(geometry_dataitem.text or "", base_dir=xdmf_path.parent)

    topology = _read_hdf5_dataset(topo_h5, topo_dataset)
    geometry = _read_hdf5_dataset(geom_h5, geom_dataset)
    cell_centers = _cell_centers_from_topology_geometry(topology, geometry)
    point_centers = np.asarray(geometry, dtype=float)[:, :2]
    return cell_centers, point_centers, geometry


def _attribute_dataitem(attribute: ET.Element) -> ET.Element | None:
    direct = _children_by_name(attribute, "DataItem")
    if direct:
        return direct[0]
    return _first_descendant(attribute, "DataItem")


def _as_scalar_values(values: np.ndarray, *, vector_mode: str) -> np.ndarray | None:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    if arr.ndim == 2 and arr.shape[1] > 1:
        if vector_mode != "norm":
            return None
        return np.linalg.norm(arr, axis=1)
    if arr.ndim == 1:
        return arr
    return None


def load_scalar_image_fields_from_xdmf(
    xdmf_path: str | Path,
    *,
    field_names: Sequence[str] | None = None,
    vector_mode: str = "norm",
) -> list[LoadedImageField]:
    xdmf_path = Path(xdmf_path)
    if not xdmf_path.exists():
        raise FileNotFoundError(str(xdmf_path))

    selected = None if field_names is None else {str(name) for name in field_names}
    root = ET.parse(xdmf_path).getroot()
    cell_centers, point_centers, _ = _load_mesh_centers(xdmf_path, root)

    fields: list[LoadedImageField] = []
    seen: set[tuple[str, str]] = set()

    for attribute in root.iter():
        if _local_name(attribute.tag) != "Attribute":
            continue
        name = str(attribute.attrib.get("Name", "")).strip()
        if not name:
            continue
        if selected is not None and name not in selected:
            continue

        dataitem = _attribute_dataitem(attribute)
        if dataitem is None or not (dataitem.text or "").strip():
            continue

        h5_path, dataset_path = _parse_dataitem_reference(dataitem.text or "", base_dir=xdmf_path.parent)
        key = (name, str(h5_path) + dataset_path)
        if key in seen:
            continue
        seen.add(key)

        values = _as_scalar_values(_read_hdf5_dataset(h5_path, dataset_path), vector_mode=vector_mode)
        if values is None:
            continue

        if values.shape[0] == cell_centers.shape[0]:
            centers = cell_centers
            association = "cell"
        elif values.shape[0] == point_centers.shape[0]:
            centers = point_centers
            association = "point"
        else:
            continue

        fields.append(
            LoadedImageField(
                field_name=name,
                centers=np.asarray(centers, dtype=float),
                values=np.asarray(values, dtype=float),
                xdmf_path=str(xdmf_path),
                h5_path=str(h5_path),
                h5_dataset=dataset_path,
                association=association,
                source_format="xdmf",
                source_path=str(xdmf_path),
                source_dataset=dataset_path,
            )
        )

    return fields


def _pvd_dataset_path(path: Path, *, time_index: int = -1) -> Path:
    if path.suffix.lower() != ".pvd":
        return path
    root = ET.parse(path).getroot()
    datasets = [node for node in root.iter() if _local_name(node.tag) == "DataSet"]
    if not datasets:
        raise ValueError(f"No DataSet entry found in PVD file: {path}")
    dataset = datasets[time_index]
    file_attr = dataset.attrib.get("file")
    if not file_attr:
        raise ValueError(f"PVD DataSet without file attribute in {path}")
    candidate = Path(file_attr)
    if not candidate.is_absolute():
        candidate = path.parent / candidate
    return candidate


def _pvtu_piece_paths(path: Path) -> list[Path]:
    """Return the serial VTU pieces referenced by a parallel PVTU file."""
    root = ET.parse(path).getroot()
    pieces: list[Path] = []
    for node in root.iter():
        if _local_name(node.tag) != "Piece":
            continue
        source = node.attrib.get("Source") or node.attrib.get("source")
        if not source:
            continue
        piece = Path(source)
        if not piece.is_absolute():
            piece = path.parent / piece
        pieces.append(piece)
    return pieces


def _merge_meshio_meshes(meshes: Sequence[Any]):
    """Merge several serial VTU pieces into one meshio-like mesh.

    This is intentionally conservative: it concatenates points, cells and data
    arrays that are present in every piece. Possible ghost/duplicate entities do
    not alter the qualitative image metrics, which only need coordinates and
    scalar values.
    """
    if not meshes:
        raise ValueError("Cannot merge an empty list of VTK piece meshes.")
    if len(meshes) == 1:
        return meshes[0]

    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - guarded by caller
        raise RuntimeError("meshio is required to merge PVTU piece files.") from exc

    offsets: list[int] = []
    point_parts: list[np.ndarray] = []
    offset = 0
    for mesh in meshes:
        pts = np.asarray(mesh.points, dtype=float)
        offsets.append(offset)
        point_parts.append(pts)
        offset += int(pts.shape[0])
    points = np.vstack(point_parts) if point_parts else np.empty((0, 3), dtype=float)

    point_names = set(getattr(meshes[0], "point_data", {}).keys())
    for mesh in meshes[1:]:
        point_names &= set(getattr(mesh, "point_data", {}).keys())
    point_data = {
        str(name): np.concatenate([np.asarray(mesh.point_data[name]) for mesh in meshes], axis=0)
        for name in sorted(point_names)
    }

    cell_names = set(getattr(meshes[0], "cell_data", {}).keys())
    for mesh in meshes[1:]:
        cell_names &= set(getattr(mesh, "cell_data", {}).keys())
    cell_data = {str(name): [] for name in sorted(cell_names)}

    cells = []
    for mesh, point_offset in zip(meshes, offsets):
        mesh_cells = list(getattr(mesh, "cells", []))
        for block_index, block in enumerate(mesh_cells):
            cell_type = str(getattr(block, "type", "cell"))
            data = np.asarray(getattr(block, "data", block), dtype=np.int64)
            cells.append((cell_type, data + point_offset))
            for name in cell_data:
                values_by_block = getattr(mesh, "cell_data", {}).get(name, [])
                if block_index < len(values_by_block):
                    cell_data[name].append(np.asarray(values_by_block[block_index]))
                else:
                    cell_data[name].append(np.empty((data.shape[0],), dtype=float))

    return meshio.Mesh(points=points, cells=cells, point_data=point_data, cell_data=cell_data)


def _meshio_read_serial_vtu_compat(meshio_module: Any, path: Path):
    """Read a serial VTU file, accepting FEniCSx/PETSc XML VTK 2.2 output.

    meshio 5.3 can terminate via SystemExit on PETSc/FEniCSx VTU files with
    XML header version="2.2".  To avoid that hard exit, detect this header
    before the first meshio.read call and read a temporary byte-length-preserving
    copy with version="1.0".  Appended/binary offsets are unchanged.
    """
    path = Path(path)
    if path.suffix.lower() != ".vtu":
        return meshio_module.read(path)

    raw = path.read_bytes()
    patched = raw.replace(b'version="2.2"', b'version="1.0"', 1)
    patched = patched.replace(b"version='2.2'", b"version='1.0'", 1)

    if patched != raw:
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".vtu", delete=False) as tmp:
                tmp.write(patched)
                tmp_path = Path(tmp.name)
            return meshio_module.read(tmp_path)
        except BaseException as patched_exc:
            if isinstance(patched_exc, KeyboardInterrupt):
                raise
            raise RuntimeError(
                f"Could not read VTU file {path} after XML version 2.2 -> 1.0 compatibility patch."
            ) from patched_exc
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass

    try:
        return meshio_module.read(path)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RuntimeError(f"Could not read VTU file {path} with meshio.") from exc


def _meshio_read_vtk(path: Path):
    try:
        import meshio
    except ImportError as exc:
        raise RuntimeError(
            "meshio is required to read PVD/VTK files for image metrics. "
            "Install it in the FEniCSx environment with: python -m pip install meshio"
        ) from exc

    path = Path(path)
    if path.suffix.lower() == ".pvtu":
        piece_paths = _pvtu_piece_paths(path)
        if not piece_paths:
            raise ValueError(f"No serial VTU Piece entries found in PVTU file: {path}")
        missing = [str(piece) for piece in piece_paths if not piece.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing VTU piece files referenced by " + str(path) + ": " + ", ".join(missing)
            )
        return _merge_meshio_meshes([_meshio_read_serial_vtu_compat(meshio, piece) for piece in piece_paths])

    if path.suffix.lower() == ".vtu":
        return _meshio_read_serial_vtu_compat(meshio, path)

    return meshio.read(path)


def _cell_block_centers(points: np.ndarray, cells: Any) -> list[tuple[str, np.ndarray]]:
    pts = np.asarray(points, dtype=float)
    out: list[tuple[str, np.ndarray]] = []
    for block in cells:
        cell_type = str(getattr(block, "type", "cell"))
        data = np.asarray(getattr(block, "data", block), dtype=np.int64)
        if data.size == 0:
            centers = np.empty((0, 2), dtype=float)
        else:
            centers = np.mean(pts[data, :2], axis=1)
        out.append((cell_type, centers))
    return out


def load_scalar_image_fields_from_pvd(
    pvd_path: str | Path,
    *,
    field_names: Sequence[str] | None = None,
    vector_mode: str = "norm",
    time_index: int = -1,
) -> list[LoadedImageField]:
    pvd_path = Path(pvd_path)
    if not pvd_path.exists():
        raise FileNotFoundError(str(pvd_path))

    vtk_path = _pvd_dataset_path(pvd_path, time_index=time_index)
    if not vtk_path.exists():
        raise FileNotFoundError(str(vtk_path))

    selected = None if field_names is None else {str(name) for name in field_names}
    mesh = _meshio_read_vtk(vtk_path)
    points = np.asarray(mesh.points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"VTK points in {vtk_path} must have shape (n_points, dim>=2).")

    fields: list[LoadedImageField] = []

    for name, raw_values in getattr(mesh, "point_data", {}).items():
        name = str(name)
        if selected is not None and name not in selected:
            continue
        values = _as_scalar_values(np.asarray(raw_values), vector_mode=vector_mode)
        if values is None or values.shape[0] != points.shape[0]:
            continue
        fields.append(
            LoadedImageField(
                field_name=name,
                centers=points[:, :2],
                values=np.asarray(values, dtype=float),
                association="point",
                source_format="pvd",
                source_path=str(pvd_path),
                source_dataset=str(vtk_path),
            )
        )

    block_centers = _cell_block_centers(points, mesh.cells)
    cell_data = getattr(mesh, "cell_data", {})
    for name, block_values in cell_data.items():
        name = str(name)
        if selected is not None and name not in selected:
            continue
        for block_index, raw_values in enumerate(block_values):
            if block_index >= len(block_centers):
                continue
            cell_type, centers = block_centers[block_index]
            values = _as_scalar_values(np.asarray(raw_values), vector_mode=vector_mode)
            if values is None or values.shape[0] != centers.shape[0]:
                continue
            fields.append(
                LoadedImageField(
                    field_name=name,
                    centers=np.asarray(centers, dtype=float),
                    values=np.asarray(values, dtype=float),
                    association=f"cell:{cell_type}",
                    source_format="pvd",
                    source_path=str(pvd_path),
                    source_dataset=str(vtk_path),
                )
            )

    return fields


def load_scalar_image_fields(
    image_path: str | Path,
    *,
    field_names: Sequence[str] | None = None,
    vector_mode: str = "norm",
) -> list[LoadedImageField]:
    path = Path(image_path)
    suffix = path.suffix.lower()
    if suffix == ".xdmf":
        return load_scalar_image_fields_from_xdmf(path, field_names=field_names, vector_mode=vector_mode)
    if suffix in {".pvd", ".vtu", ".pvtu"}:
        return load_scalar_image_fields_from_pvd(path, field_names=field_names, vector_mode=vector_mode)
    raise ValueError(f"Unsupported image artifact format for metrics: {path}")


def compute_metrics_for_loaded_fields(
    fields: Iterable[LoadedImageField],
    *,
    method: str,
    defect_center_x: float,
    defect_center_y: float,
    defect_radius: float,
    threshold_false_max: float = 0.5,
    half_max_fraction: float = 0.5,
    false_max_min_distance: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> list[ImageQualityMetrics]:
    rows: list[ImageQualityMetrics] = []
    for field in fields:
        field_extra = {
            "source_format": field.source_format,
            "source_path": field.source_path or field.xdmf_path,
            "source_dataset": field.source_dataset or field.h5_dataset,
            "source_xdmf": field.xdmf_path,
            "source_h5": field.h5_path,
            "source_h5_dataset": field.h5_dataset,
            "field_association": field.association,
        }
        if extra:
            field_extra.update(dict(extra))
        rows.append(
            compute_image_quality_metrics(
                cell_centers=field.centers,
                image_values=field.values,
                defect_center_x=defect_center_x,
                defect_center_y=defect_center_y,
                defect_radius=defect_radius,
                method=method,
                field_name=field.field_name,
                threshold_false_max=threshold_false_max,
                half_max_fraction=half_max_fraction,
                false_max_min_distance=false_max_min_distance,
                value_mode="abs",
                extra=field_extra,
            )
        )
    return rows


def compute_metrics_for_image_file(
    *,
    image_path: str | Path,
    method: str,
    defect_center_x: float,
    defect_center_y: float,
    defect_radius: float,
    field_names: Sequence[str] | None = None,
    threshold_false_max: float = 0.5,
    half_max_fraction: float = 0.5,
    false_max_min_distance: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> list[ImageQualityMetrics]:
    fields = load_scalar_image_fields(image_path, field_names=field_names)
    return compute_metrics_for_loaded_fields(
        fields,
        method=method,
        defect_center_x=defect_center_x,
        defect_center_y=defect_center_y,
        defect_radius=defect_radius,
        threshold_false_max=threshold_false_max,
        half_max_fraction=half_max_fraction,
        false_max_min_distance=false_max_min_distance,
        extra=extra,
    )


def compute_metrics_for_xdmf(
    *,
    xdmf_path: str | Path,
    method: str,
    defect_center_x: float,
    defect_center_y: float,
    defect_radius: float,
    field_names: Sequence[str] | None = None,
    threshold_false_max: float = 0.5,
    half_max_fraction: float = 0.5,
    false_max_min_distance: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> list[ImageQualityMetrics]:
    return compute_metrics_for_image_file(
        image_path=xdmf_path,
        method=method,
        defect_center_x=defect_center_x,
        defect_center_y=defect_center_y,
        defect_radius=defect_radius,
        field_names=field_names,
        threshold_false_max=threshold_false_max,
        half_max_fraction=half_max_fraction,
        false_max_min_distance=false_max_min_distance,
        extra=extra,
    )


def write_metrics_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return path


def metrics_to_dicts(rows: Sequence[ImageQualityMetrics]) -> list[dict[str, Any]]:
    return [row.as_dict() for row in rows]


def write_metrics_json(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, indent=2, sort_keys=True)
    return path


def summarize_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_rows": 0}

    def finite_values(name: str) -> list[float]:
        out: list[float] = []
        for row in rows:
            try:
                value = float(row[name])
            except Exception:
                continue
            if math.isfinite(value):
                out.append(value)
        return out

    summary: dict[str, Any] = {"num_rows": len(rows)}
    for name in (
        "peak_to_artifact_ratio",
        "energy_inside_defect",
        "energy_outside_defect",
        "localization_error",
        "false_maxima_count",
    ):
        values = finite_values(name)
        if values:
            summary[f"{name}_min"] = float(np.min(values))
            summary[f"{name}_max"] = float(np.max(values))
            summary[f"{name}_mean"] = float(np.mean(values))
    return summary
