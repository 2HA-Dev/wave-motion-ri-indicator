from __future__ import annotations

import gc
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import (
    vtkXMLPUnstructuredGridReader,
    vtkXMLUnstructuredGridReader,
)


VTK_TRIANGLE = 5
VTK_LAGRANGE_TRIANGLE = 69

SUPPORTED_TRIANGLE_TYPE_IDS = frozenset(
    {
        VTK_TRIANGLE,
        VTK_LAGRANGE_TRIANGLE,
    }
)


@dataclass(frozen=True)
class PVDSnapshot:
    index: int
    time: float
    path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "time": float(self.time),
            "path": str(self.path),
        }


@dataclass(frozen=True)
class PVDSeries:
    path: Path
    snapshots: tuple[PVDSnapshot, ...]

    @property
    def times(self) -> np.ndarray:
        return np.asarray(
            [
                snapshot.time
                for snapshot in self.snapshots
            ],
            dtype=np.float64,
        )

    def as_dict(self) -> dict[str, Any]:
        times = self.times

        return {
            "path": str(self.path),
            "num_snapshots": len(self.snapshots),
            "first_time": (
                float(times[0])
                if times.size
                else None
            ),
            "last_time": (
                float(times[-1])
                if times.size
                else None
            ),
            "times": times.tolist(),
        }


@dataclass(frozen=True)
class RawResidualWaveSnapshot:
    path: Path
    field_name: str
    points: np.ndarray
    triangles_owned: np.ndarray
    field: np.ndarray
    number_of_raw_points: int
    number_of_raw_cells: int
    number_of_owned_cells: int
    owned_cell_type_ids: tuple[int, ...]
    available_point_arrays: tuple[str, ...]


@dataclass(frozen=True)
class ResidualWaveAssemblyPlan:
    reference_points: np.ndarray
    reference_triangles_owned: np.ndarray
    used_raw_indices: np.ndarray
    used_to_global: np.ndarray
    duplicate_counts: np.ndarray
    global_points: np.ndarray
    global_triangles: np.ndarray
    maximum_abs_z_coordinate: float

    @property
    def num_global_points(self) -> int:
        return int(
            self.global_points.shape[0]
        )

    @property
    def num_global_triangles(self) -> int:
        return int(
            self.global_triangles.shape[0]
        )

    @property
    def duplicate_used_points(self) -> int:
        return int(
            self.used_raw_indices.size
            - self.global_points.shape[0]
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_points": int(
                self.reference_points.shape[0]
            ),
            "used_raw_points": int(
                self.used_raw_indices.size
            ),
            "global_unique_points": (
                self.num_global_points
            ),
            "duplicate_used_points": (
                self.duplicate_used_points
            ),
            "owned_triangles": (
                self.num_global_triangles
            ),
            "maximum_abs_z_coordinate": float(
                self.maximum_abs_z_coordinate
            ),
        }


@dataclass(frozen=True)
class AssembledResidualWavePair:
    points: np.ndarray
    triangles: np.ndarray
    reflection_field: np.ndarray
    transmission_field: np.ndarray
    reflection_duplicate_max_discrepancy: float
    transmission_duplicate_max_discrepancy: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_unique_points": int(
                self.points.shape[0]
            ),
            "owned_triangles": int(
                self.triangles.shape[0]
            ),
            "reflection_field_shape": list(
                self.reflection_field.shape
            ),
            "transmission_field_shape": list(
                self.transmission_field.shape
            ),
            "reflection_duplicate_max_discrepancy": float(
                self.reflection_duplicate_max_discrepancy
            ),
            "transmission_duplicate_max_discrepancy": float(
                self.transmission_duplicate_max_discrepancy
            ),
        }


@dataclass(frozen=True)
class LoadedResidualWavePair:
    time_index: int
    time_value: float
    reflection_pvd: Path
    transmission_pvd: Path
    reflection_dataset: Path
    transmission_dataset: Path
    plan: ResidualWaveAssemblyPlan
    pair: AssembledResidualWavePair

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_index": int(self.time_index),
            "time_value": float(self.time_value),
            "reflection_pvd": str(
                self.reflection_pvd
            ),
            "transmission_pvd": str(
                self.transmission_pvd
            ),
            "reflection_dataset": str(
                self.reflection_dataset
            ),
            "transmission_dataset": str(
                self.transmission_dataset
            ),
            "assembly_plan": self.plan.as_dict(),
            "assembled_pair": self.pair.as_dict(),
        }


def read_pvd_series(
    path: str | Path,
) -> PVDSeries:
    pvd_path = Path(path).expanduser().resolve()

    if not pvd_path.is_file():
        raise FileNotFoundError(pvd_path)

    try:
        root = ET.parse(pvd_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(
            f"Invalid PVD XML file: {pvd_path}"
        ) from exc

    snapshots: list[PVDSnapshot] = []

    for index, dataset in enumerate(
        root.findall(".//DataSet")
    ):
        raw_file = dataset.attrib.get("file")
        raw_time = dataset.attrib.get("timestep")

        if raw_file is None:
            raise ValueError(
                "A PVD DataSet entry has no file attribute: "
                f"{pvd_path}"
            )

        if raw_time is None:
            raise ValueError(
                "A PVD DataSet entry has no timestep: "
                f"{pvd_path}"
            )

        try:
            time = float(raw_time)
        except ValueError as exc:
            raise ValueError(
                f"Invalid PVD timestep {raw_time!r}."
            ) from exc

        if not np.isfinite(time):
            raise ValueError(
                "PVD contains a non-finite timestep."
            )

        dataset_path = (
            pvd_path.parent
            / raw_file
        ).resolve()

        if not dataset_path.is_file():
            raise FileNotFoundError(
                dataset_path
            )

        snapshots.append(
            PVDSnapshot(
                index=index,
                time=time,
                path=dataset_path,
            )
        )

    if not snapshots:
        raise ValueError(
            f"No snapshots found in {pvd_path}."
        )

    times = np.asarray(
        [
            snapshot.time
            for snapshot in snapshots
        ],
        dtype=np.float64,
    )

    if times.size > 1:
        increments = np.diff(times)

        if np.any(increments <= 0.0):
            raise ValueError(
                "PVD timesteps must be strictly increasing."
            )

    return PVDSeries(
        path=pvd_path,
        snapshots=tuple(snapshots),
    )


def select_pvd_snapshot(
    series: PVDSeries,
    *,
    requested_time: float,
    atol: float = 1.0e-12,
) -> PVDSnapshot:
    if not np.isfinite(requested_time):
        raise ValueError(
            "requested_time must be finite."
        )

    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError(
            "atol must be finite and non-negative."
        )

    matches = [
        snapshot
        for snapshot in series.snapshots
        if np.isclose(
            snapshot.time,
            requested_time,
            rtol=0.0,
            atol=atol,
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one snapshot at "
            f"t={requested_time}, found {len(matches)}."
        )

    return matches[0]


def validate_pvd_pair_times(
    reflection: PVDSeries,
    transmission: PVDSeries,
) -> None:
    reflection_times = reflection.times
    transmission_times = transmission.times

    if reflection_times.shape != transmission_times.shape:
        raise ValueError(
            "Reflection and transmission PVD series "
            "contain different numbers of snapshots."
        )

    if not np.array_equal(
        reflection_times,
        transmission_times,
    ):
        raise ValueError(
            "Reflection and transmission PVD timesteps differ."
        )


def _reader_for(path: Path):
    suffix = path.suffix.lower()

    if suffix == ".pvtu":
        return vtkXMLPUnstructuredGridReader()

    if suffix == ".vtu":
        return vtkXMLUnstructuredGridReader()

    raise ValueError(
        "Expected a .pvtu or .vtu dataset, "
        f"received {path}."
    )


def _vtk_array_copy(
    container,
    name: str,
) -> np.ndarray | None:
    values = container.GetArray(name)

    if values is None:
        return None

    return np.asarray(
        vtk_to_numpy(values)
    ).copy()


def _per_cell_type_array(
    grid,
) -> np.ndarray:
    number_of_cells = int(
        grid.GetNumberOfCells()
    )

    # VTK 9.6 does not expose an equivalent vectorized
    # per-cell method. GetCellTypes() only returns unique
    # types, whereas this diagnostic requires one identifier
    # per cell.
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            DeprecationWarning,
        )

        vtk_values = (
            grid.GetCellTypesArray()
        )

    if vtk_values is None:
        raise RuntimeError(
            "Unable to obtain the VTK per-cell type array."
        )

    values = np.asarray(
        vtk_to_numpy(vtk_values)
    ).reshape(-1)

    if values.size != number_of_cells:
        raise RuntimeError(
            "Invalid VTK cell-type array size: "
            f"{values.size} != {number_of_cells}."
        )

    return values.astype(
        np.int64,
        copy=False,
    )


def read_raw_residual_wave_snapshot(
    path: str | Path,
    *,
    field_name: str,
) -> RawResidualWaveSnapshot:
    dataset_path = Path(path).expanduser().resolve()

    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    if not field_name:
        raise ValueError(
            "field_name must not be empty."
        )

    reader = _reader_for(dataset_path)
    reader.SetFileName(str(dataset_path))
    reader.Update()

    grid = reader.GetOutput()

    if grid is None:
        raise RuntimeError(
            f"VTK returned no grid for {dataset_path}."
        )

    number_of_points = int(
        grid.GetNumberOfPoints()
    )

    number_of_cells = int(
        grid.GetNumberOfCells()
    )

    if number_of_points <= 0:
        raise RuntimeError(
            f"No points found in {dataset_path}."
        )

    if number_of_cells <= 0:
        raise RuntimeError(
            f"No cells found in {dataset_path}."
        )

    points_object = grid.GetPoints()

    if points_object is None:
        raise RuntimeError(
            f"No coordinates found in {dataset_path}."
        )

    points = np.asarray(
        vtk_to_numpy(
            points_object.GetData()
        ),
        dtype=np.float64,
    ).copy()

    if points.ndim != 2:
        raise RuntimeError(
            f"Unexpected coordinate shape: {points.shape}."
        )

    if points.shape[0] != number_of_points:
        raise RuntimeError(
            "VTK coordinate count is inconsistent."
        )

    if points.shape[1] < 2:
        raise RuntimeError(
            "VTK coordinates expose fewer than two components."
        )

    if not np.all(np.isfinite(points)):
        raise RuntimeError(
            "VTK coordinates contain non-finite values."
        )

    cells = grid.GetCells()

    offsets = np.asarray(
        vtk_to_numpy(
            cells.GetOffsetsArray()
        ),
        dtype=np.int64,
    ).reshape(-1)

    connectivity = np.asarray(
        vtk_to_numpy(
            cells.GetConnectivityArray()
        ),
        dtype=np.int64,
    ).reshape(-1)

    if offsets.size != number_of_cells + 1:
        raise RuntimeError(
            "Unexpected VTK cell-offset array size."
        )

    if offsets[0] != 0:
        raise RuntimeError(
            "VTK connectivity does not start at zero."
        )

    if offsets[-1] != connectivity.size:
        raise RuntimeError(
            "VTK final cell offset is inconsistent."
        )

    widths = np.diff(offsets)
    cell_types = _per_cell_type_array(grid)

    cell_data = grid.GetCellData()
    point_data = grid.GetPointData()

    ghost = _vtk_array_copy(
        cell_data,
        "vtkGhostType",
    )

    if ghost is None:
        owned_mask = np.ones(
            number_of_cells,
            dtype=bool,
        )
    else:
        ghost = np.asarray(
            ghost
        ).reshape(-1)

        if ghost.size != number_of_cells:
            raise RuntimeError(
                "Invalid vtkGhostType array size."
            )

        owned_mask = ghost == 0

    if not np.any(owned_mask):
        raise RuntimeError(
            f"No owned cells found in {dataset_path}."
        )

    owned_types = cell_types[owned_mask]
    owned_widths = widths[owned_mask]

    unsupported_types = sorted(
        set(
            int(value)
            for value in np.unique(
                owned_types
            )
        )
        - SUPPORTED_TRIANGLE_TYPE_IDS
    )

    if unsupported_types:
        raise RuntimeError(
            "Unsupported owned VTK cell types: "
            f"{unsupported_types}."
        )

    if not np.all(
        owned_widths == 3
    ):
        raise RuntimeError(
            "Only three-node P1 triangles are supported. "
            "Owned cell widths are "
            f"{np.unique(owned_widths).tolist()}."
        )

    owned_starts = offsets[:-1][
        owned_mask
    ]

    connectivity_indices = (
        owned_starts[:, None]
        + np.arange(
            3,
            dtype=np.int64,
        )[None, :]
    )

    triangles_owned = np.ascontiguousarray(
        connectivity[
            connectivity_indices
        ],
        dtype=np.int64,
    )

    if np.min(triangles_owned) < 0:
        raise RuntimeError(
            "VTK connectivity contains negative indices."
        )

    if np.max(triangles_owned) >= number_of_points:
        raise RuntimeError(
            "VTK connectivity references an unknown point."
        )

    field = _vtk_array_copy(
        point_data,
        field_name,
    )

    available_point_arrays = tuple(
        str(
            point_data.GetArrayName(index)
        )
        for index in range(
            point_data.GetNumberOfArrays()
        )
    )

    if field is None:
        raise RuntimeError(
            f"Field {field_name!r} is absent from "
            f"{dataset_path}. Available point arrays: "
            f"{available_point_arrays}."
        )

    field = np.asarray(
        field,
        dtype=np.float64,
    )

    if field.ndim != 2:
        raise RuntimeError(
            f"Unexpected field shape: {field.shape}."
        )

    if field.shape[0] != number_of_points:
        raise RuntimeError(
            "Field and coordinate counts are incompatible."
        )

    if field.shape[1] not in (2, 3):
        raise RuntimeError(
            "Residual wavefields must expose two physical "
            "components or three VTK components."
        )

    if not np.all(np.isfinite(field)):
        raise RuntimeError(
            f"Field {field_name!r} contains non-finite values."
        )

    result = RawResidualWaveSnapshot(
        path=dataset_path,
        field_name=str(field_name),
        points=np.ascontiguousarray(
            points,
            dtype=np.float64,
        ),
        triangles_owned=triangles_owned,
        field=np.ascontiguousarray(
            field,
            dtype=np.float64,
        ),
        number_of_raw_points=number_of_points,
        number_of_raw_cells=number_of_cells,
        number_of_owned_cells=int(
            triangles_owned.shape[0]
        ),
        owned_cell_type_ids=tuple(
            int(value)
            for value in np.unique(
                owned_types
            )
        ),
        available_point_arrays=(
            available_point_arrays
        ),
    )

    del grid
    del reader

    gc.collect()

    return result


def _validate_raw_pair_mesh(
    reflection: RawResidualWaveSnapshot,
    transmission: RawResidualWaveSnapshot,
) -> None:
    if not np.array_equal(
        reflection.points,
        transmission.points,
    ):
        raise ValueError(
            "Reflection and transmission coordinates differ."
        )

    if not np.array_equal(
        reflection.triangles_owned,
        transmission.triangles_owned,
    ):
        raise ValueError(
            "Reflection and transmission connectivities differ."
        )

    if reflection.field.shape != transmission.field.shape:
        raise ValueError(
            "Reflection and transmission field shapes differ."
        )


def build_residual_wave_assembly_plan(
    reflection: RawResidualWaveSnapshot,
    transmission: RawResidualWaveSnapshot,
    *,
    z_atol: float = 0.0,
    z_rtol: float = 1.0e-14,
) -> ResidualWaveAssemblyPlan:
    _validate_raw_pair_mesh(
        reflection,
        transmission,
    )

    if not np.isfinite(z_atol) or z_atol < 0.0:
        raise ValueError(
            "z_atol must be finite and non-negative."
        )

    if not np.isfinite(z_rtol) or z_rtol < 0.0:
        raise ValueError(
            "z_rtol must be finite and non-negative."
        )

    points = reflection.points
    triangles = reflection.triangles_owned

    used_raw_indices = np.unique(
        triangles.reshape(-1)
    )

    used_coordinates = np.ascontiguousarray(
        points[used_raw_indices, :2],
        dtype=np.float64,
    )

    if points.shape[1] >= 3:
        maximum_abs_z_coordinate = float(
            np.max(
                np.abs(
                    points[
                        used_raw_indices,
                        2,
                    ]
                )
            )
        )

        xy_scale = float(
            np.max(
                np.abs(
                    points[
                        used_raw_indices,
                        :2,
                    ]
                )
            )
        )

        z_tolerance = (
            float(z_atol)
            + float(z_rtol) * xy_scale
        )

        if maximum_abs_z_coordinate > z_tolerance:
            raise ValueError(
                "The nominally two-dimensional mesh has a "
                "significant z coordinate: "
                f"{maximum_abs_z_coordinate:.16e} > "
                f"{z_tolerance:.16e}."
            )
    else:
        maximum_abs_z_coordinate = 0.0

    global_points, used_to_global = np.unique(
        used_coordinates,
        axis=0,
        return_inverse=True,
    )

    raw_to_used = np.full(
        points.shape[0],
        -1,
        dtype=np.int64,
    )

    raw_to_used[
        used_raw_indices
    ] = np.arange(
        used_raw_indices.size,
        dtype=np.int64,
    )

    triangles_used = raw_to_used[
        triangles
    ]

    if np.any(triangles_used < 0):
        raise RuntimeError(
            "An owned triangle references an unused point."
        )

    global_triangles = used_to_global[
        triangles_used
    ]

    degenerate = (
        (
            global_triangles[:, 0]
            == global_triangles[:, 1]
        )
        | (
            global_triangles[:, 0]
            == global_triangles[:, 2]
        )
        | (
            global_triangles[:, 1]
            == global_triangles[:, 2]
        )
    )

    if np.any(degenerate):
        raise RuntimeError(
            "Coordinate assembly produced a degenerate triangle."
        )

    duplicate_counts = np.bincount(
        used_to_global,
        minlength=global_points.shape[0],
    ).astype(np.int64)

    if np.any(duplicate_counts <= 0):
        raise RuntimeError(
            "Invalid assembled point multiplicity."
        )

    return ResidualWaveAssemblyPlan(
        reference_points=np.ascontiguousarray(
            points,
            dtype=np.float64,
        ),
        reference_triangles_owned=np.ascontiguousarray(
            triangles,
            dtype=np.int64,
        ),
        used_raw_indices=np.ascontiguousarray(
            used_raw_indices,
            dtype=np.int64,
        ),
        used_to_global=np.ascontiguousarray(
            used_to_global,
            dtype=np.int64,
        ),
        duplicate_counts=np.ascontiguousarray(
            duplicate_counts,
            dtype=np.int64,
        ),
        global_points=np.ascontiguousarray(
            global_points,
            dtype=np.float64,
        ),
        global_triangles=np.ascontiguousarray(
            global_triangles,
            dtype=np.int64,
        ),
        maximum_abs_z_coordinate=(
            maximum_abs_z_coordinate
        ),
    )


def _assemble_field_with_plan(
    plan: ResidualWaveAssemblyPlan,
    snapshot: RawResidualWaveSnapshot,
    *,
    duplicate_rtol: float,
    duplicate_atol: float,
) -> tuple[np.ndarray, float]:
    if not np.isfinite(
        duplicate_rtol
    ) or duplicate_rtol < 0.0:
        raise ValueError(
            "duplicate_rtol must be finite and non-negative."
        )

    if not np.isfinite(
        duplicate_atol
    ) or duplicate_atol < 0.0:
        raise ValueError(
            "duplicate_atol must be finite and non-negative."
        )

    if not np.array_equal(
        snapshot.points,
        plan.reference_points,
    ):
        raise ValueError(
            "Snapshot coordinates differ from the "
            "assembly-plan coordinates."
        )

    if not np.array_equal(
        snapshot.triangles_owned,
        plan.reference_triangles_owned,
    ):
        raise ValueError(
            "Snapshot connectivity differs from the "
            "assembly-plan connectivity."
        )

    values = snapshot.field[
        plan.used_raw_indices
    ]

    assembled = np.zeros(
        (
            plan.global_points.shape[0],
            snapshot.field.shape[1],
        ),
        dtype=np.float64,
    )

    np.add.at(
        assembled,
        plan.used_to_global,
        values,
    )

    assembled /= (
        plan.duplicate_counts[:, None]
    )

    reconstructed = assembled[
        plan.used_to_global
    ]

    maximum_discrepancy = float(
        np.max(
            np.abs(
                values - reconstructed
            )
        )
    )

    scale = float(
        np.max(np.abs(values))
    )

    tolerance = (
        float(duplicate_atol)
        + float(duplicate_rtol) * scale
    )

    if maximum_discrepancy > tolerance:
        raise ValueError(
            f"Inconsistent duplicated nodal values for "
            f"{snapshot.field_name!r}: "
            f"maximum_discrepancy="
            f"{maximum_discrepancy:.16e}, "
            f"tolerance={tolerance:.16e}."
        )

    return (
        np.ascontiguousarray(
            assembled,
            dtype=np.float64,
        ),
        maximum_discrepancy,
    )


def assemble_residual_wave_pair(
    plan: ResidualWaveAssemblyPlan,
    reflection: RawResidualWaveSnapshot,
    transmission: RawResidualWaveSnapshot,
    *,
    duplicate_rtol: float = 1.0e-12,
    duplicate_atol: float = 0.0,
) -> AssembledResidualWavePair:
    _validate_raw_pair_mesh(
        reflection,
        transmission,
    )

    reflection_field, reflection_discrepancy = (
        _assemble_field_with_plan(
            plan,
            reflection,
            duplicate_rtol=duplicate_rtol,
            duplicate_atol=duplicate_atol,
        )
    )

    transmission_field, transmission_discrepancy = (
        _assemble_field_with_plan(
            plan,
            transmission,
            duplicate_rtol=duplicate_rtol,
            duplicate_atol=duplicate_atol,
        )
    )

    return AssembledResidualWavePair(
        points=plan.global_points,
        triangles=plan.global_triangles,
        reflection_field=reflection_field,
        transmission_field=transmission_field,
        reflection_duplicate_max_discrepancy=(
            reflection_discrepancy
        ),
        transmission_duplicate_max_discrepancy=(
            transmission_discrepancy
        ),
    )


def load_residual_wave_pair_at_time(
    *,
    reflection_pvd: str | Path,
    transmission_pvd: str | Path,
    requested_time: float,
    reflection_field_name: str = (
        "reflection_residual_wavefield"
    ),
    transmission_field_name: str = (
        "transmission_residual_wavefield"
    ),
    duplicate_rtol: float = 1.0e-12,
    duplicate_atol: float = 0.0,
    z_atol: float = 0.0,
    z_rtol: float = 1.0e-14,
) -> LoadedResidualWavePair:
    reflection_series = read_pvd_series(
        reflection_pvd
    )

    transmission_series = read_pvd_series(
        transmission_pvd
    )

    validate_pvd_pair_times(
        reflection_series,
        transmission_series,
    )

    reflection_snapshot = select_pvd_snapshot(
        reflection_series,
        requested_time=requested_time,
    )

    transmission_snapshot = select_pvd_snapshot(
        transmission_series,
        requested_time=requested_time,
    )

    if (
        reflection_snapshot.index
        != transmission_snapshot.index
    ):
        raise RuntimeError(
            "Reflection and transmission time indices differ."
        )

    reflection_raw = (
        read_raw_residual_wave_snapshot(
            reflection_snapshot.path,
            field_name=reflection_field_name,
        )
    )

    transmission_raw = (
        read_raw_residual_wave_snapshot(
            transmission_snapshot.path,
            field_name=transmission_field_name,
        )
    )

    plan = build_residual_wave_assembly_plan(
        reflection_raw,
        transmission_raw,
        z_atol=z_atol,
        z_rtol=z_rtol,
    )

    pair = assemble_residual_wave_pair(
        plan,
        reflection_raw,
        transmission_raw,
        duplicate_rtol=duplicate_rtol,
        duplicate_atol=duplicate_atol,
    )

    return LoadedResidualWavePair(
        time_index=reflection_snapshot.index,
        time_value=reflection_snapshot.time,
        reflection_pvd=reflection_series.path,
        transmission_pvd=transmission_series.path,
        reflection_dataset=(
            reflection_snapshot.path
        ),
        transmission_dataset=(
            transmission_snapshot.path
        ),
        plan=plan,
        pair=pair,
    )
